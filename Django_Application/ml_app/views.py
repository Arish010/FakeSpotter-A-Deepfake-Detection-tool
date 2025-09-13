# views.py — binary REAL/FAKE, non‑blocking face preview

from django.shortcuts import render, redirect
from .inference import predict_npy_files, _load_thresholds
from django.conf import settings
from django.http import HttpResponse
from django.views.decorators.cache import never_cache
import uuid
import os, time, shutil, json, logging
import cv2
import torch
import numpy as np
import torch.nn as nn
from torchvision import transforms
from torchvision.models import (
    resnext50_32x4d,
    ResNeXt50_32X4D_Weights,
    resnet50,
    ResNet50_Weights,
)
from PIL import Image
import face_recognition
from .forms import VideoUploadForm
from .models_lstm import LSTMClassifier
from .inference import (
    predict_folder,
    predict_npy_files,
)

try:
    from .inference import __version__ as predictor_version
except Exception:
    predictor_version = "unknown"

logger = logging.getLogger("ml_app")

from django.http import JsonResponse


def debug_thr(request):
    thr = None
    margin = getattr(settings, "UNCERTAIN_BAND", 0.0)
    thr_path = getattr(settings, "CALIBRATED_THRESHOLD_PATH", None)

    if thr_path and os.path.exists(thr_path):
        try:
            raw = open(thr_path, "r", encoding="utf-8").read().strip()
            try:
                data = json.loads(raw)
                if isinstance(data, dict):
                    thr = float(data.get("thr", data.get("threshold", thr or 0.5)))
                    margin = float(data.get("margin", margin))
                elif isinstance(data, (int, float, str)):
                    thr = float(data)
            except Exception:
                # support "thr=0.45"
                if "=" in raw:
                    thr = float(raw.split("=", 1)[1].strip())
        except Exception:
            pass

    payload = {
        "predictor_version": predictor_version,
        "threshold": thr,
        "margin": margin,
        "input_size": IM_SIZE,
        "backbone": FEATURE_BACKBONE,
    }
    return JsonResponse(payload)


# ------------------------------ Global config ------------------------------
ALLOWED_VIDEO_EXTENSIONS = {"mp4", "gif", "webm", "avi", "3gp", "wmv", "flv", "mkv"}

INDEX_TMPL = "index.html"
PREDICT_TMPL = "predict.html"
ABOUT_TMPL = "about.html"

MAX_SEQ_LEN = int(getattr(settings, "SEQ_LEN", 24))
FEATURE_BACKBONE = getattr(settings, "FEATURE_BACKBONE", "resnext50").lower()
IM_SIZE = int(getattr(settings, "INPUT_SIZE", 112))

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

HOG_UPSAMPLE = 0
# ------------------------------ Lazy singletons ------------------------------
_FEATURE_MODEL = None
_LSTM_MODEL = None
_IDX_TO_CLASS = None


def _device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _log(msg: str):
    try:
        safe = (
            msg.replace("→", "->")
            .replace("↔", "<->")
            .replace("✓", "[OK]")
            .replace("✗", "[X]")
        )
        logger.info(safe)
    except Exception:
        logger.info(str(msg))


# ------------------------------ Backbone loader -> 2048-D features ------------------------------
def get_feature_model():
    global _FEATURE_MODEL
    if _FEATURE_MODEL is not None:
        return _FEATURE_MODEL

    device = _device()
    if FEATURE_BACKBONE == "resnet50":
        m = resnet50(weights=ResNet50_Weights.DEFAULT)
    else:
        m = resnext50_32x4d(weights=ResNeXt50_32X4D_Weights.DEFAULT)
    m.fc = nn.Identity()
    m.eval().to(device)
    _FEATURE_MODEL = m
    _log(f"[info] Backbone={FEATURE_BACKBONE}  IM_SIZE={IM_SIZE}")
    return _FEATURE_MODEL


# ------------------------------ LSTM loader (binary) ------------------------------
def get_lstm_model():
    global _LSTM_MODEL, _IDX_TO_CLASS
    if _LSTM_MODEL is not None:
        return _LSTM_MODEL

    device = _device()
    chk = torch.load(settings.LSTM_WEIGHTS_PATH, map_location=device)
    state = chk.get("state_dict", chk) if isinstance(chk, dict) else chk

    model = LSTMClassifier(
        input_size=2048,
        hidden_size=512,
        num_layers=2,
        num_classes=2,
        dropout=0.5,
        bidirectional=True,
    ).to(device)

    missing, unexpected = model.load_state_dict(state, strict=False)
    if missing:
        _log(f"[warn] Missing keys: {missing}")
    if unexpected:
        _log(f"[warn] Unexpected keys: {unexpected}")

    model.eval()
    _LSTM_MODEL = model

    mapping = None
    if isinstance(chk, dict) and chk.get("idx_to_class"):
        mapping = {int(k): str(v).lower() for k, v in chk["idx_to_class"].items()}
    if mapping is None and getattr(settings, "FORCE_IDX_TO_CLASS", None):
        mapping = {
            int(k): str(v).lower() for k, v in settings.FORCE_IDX_TO_CLASS.items()
        }
    if mapping is None:
        mapping = {0: "fake", 1: "real"}

    _IDX_TO_CLASS = mapping
    _log(f"[info] LSTM loaded (binary). mapping={_IDX_TO_CLASS}")
    return _LSTM_MODEL


# ------------------------------ Video / frames utilities ------------------------------
def allowed_video_file(filename: str) -> bool:
    return ("." in filename) and (
        filename.rsplit(".", 1)[1].lower() in ALLOWED_VIDEO_EXTENSIONS
    )


def _read_frames_bgr(video_path):
    cap = cv2.VideoCapture(video_path)
    frames, ok = [], True
    while ok:
        ok, f = cap.read()
        if ok:
            frames.append(f)
    cap.release()
    return frames


def _evenly_spaced_indices(n_total, n_take):
    if n_total <= 0:
        return []
    if n_total <= n_take:
        return list(range(n_total))
    step = n_total / float(n_take)
    return [int(i * step) for i in range(n_take)]


def _tensor_sequence_from_frames(frames, T_max):
    tfm = transforms.Compose(
        [
            transforms.ToPILImage(),
            transforms.Resize((IM_SIZE, IM_SIZE)),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ]
    )

    idxs = _evenly_spaced_indices(len(frames), T_max)
    xs = [tfm(cv2.cvtColor(frames[i], cv2.COLOR_BGR2RGB)) for i in idxs]
    x = torch.stack(xs, dim=0)
    return x.unsqueeze(0).contiguous()


def _crop_primary_face(rgb_full, padding=30):
    # Downscale moderately for faster HOG
    target_w = 320
    if rgb_full.shape[1] > target_w:
        h = int(target_w * rgb_full.shape[0] / rgb_full.shape[1])
        rgb_small = cv2.resize(rgb_full, (target_w, h))
    else:
        rgb_small = rgb_full

    boxes = face_recognition.face_locations(
        rgb_small, model="hog", number_of_times_to_upsample=HOG_UPSAMPLE
    )
    if not boxes:
        return None

    # Assume first face
    top, right, bottom, left = boxes[0]
    h, w = rgb_full.shape[:2]
    top = max(0, top - padding)
    left = max(0, left - padding)
    bottom = min(h, bottom + padding)
    right = min(w, right + padding)
    crop = rgb_full[top:bottom, left:right]
    return crop if crop.size > 0 else None


def _tensor_sequence_from_frames_faces(frames, T_max):
    tfm = transforms.Compose(
        [
            transforms.ToPILImage(),
            transforms.Resize((IM_SIZE, IM_SIZE)),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ]
    )

    idxs = _evenly_spaced_indices(len(frames), T_max)
    xs = []
    for i in idxs:
        rgb = cv2.cvtColor(frames[i], cv2.COLOR_BGR2RGB)
        face = _crop_primary_face(rgb, padding=30)
        img = tfm(face if face is not None else rgb)
        xs.append(img)
    x = torch.stack(xs, dim=0)
    return x.unsqueeze(0).contiguous()


@torch.inference_mode()
def extract_features_batch(x_1_T_3_H_W, feature_model):
    device = next(feature_model.parameters()).device
    _, T, C, H, W = x_1_T_3_H_W.shape
    x = x_1_T_3_H_W.view(T, C, H, W).to(device)
    feats = feature_model(x)
    if feats.ndim == 4:
        feats = torch.flatten(nn.AdaptiveAvgPool2d(1)(feats), 1)
    return feats.unsqueeze(0)  # [1,T,2048]


# ------------------------------ Views ------------------------------
def index(request):
    if request.method == "GET":
        form = VideoUploadForm()
        for k in ("file_name", "preprocessed_images", "faces_cropped_images"):
            if k in request.session:
                del request.session[k]
        return render(request, INDEX_TMPL, {"form": form})

    # POST (upload)
    form = VideoUploadForm(request.POST, request.FILES)
    if not form.is_valid():
        return render(request, INDEX_TMPL, {"form": form})

    video_file = form.cleaned_data["upload_video_file"]
    seq_len = int(getattr(settings, "SEQ_LEN", 24))

    if video_file.content_type.split("/")[0] not in getattr(
        settings, "CONTENT_TYPES", {"video"}
    ):
        form.add_error("upload_video_file", "Invalid content type")
        return render(request, INDEX_TMPL, {"form": form})

    if video_file.size > int(getattr(settings, "MAX_UPLOAD_SIZE", 100 * 1024 * 1024)):
        form.add_error("upload_video_file", "Maximum file size 100 MB")
        return render(request, INDEX_TMPL, {"form": form})

    if not allowed_video_file(video_file.name):
        form.add_error("upload_video_file", "Only video files are allowed")
        return render(request, INDEX_TMPL, {"form": form})

    # Save upload
    saved_name = f"uploaded_file_{int(time.time())}.{video_file.name.split('.')[-1]}"
    dst = os.path.join(settings.PROJECT_DIR, "uploaded_videos", saved_name)
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    with open(dst, "wb") as vFile:
        shutil.copyfileobj(video_file, vFile)

    request.session["file_name"] = dst
    request.session["sequence_length"] = int(getattr(settings, "SEQ_LEN", 24))
    return redirect("ml_app:predict")


@never_cache
def predict_page(request):
    # ---------------- Guard / session ----------------
    if request.method != "GET":
        return redirect("ml_app:home")

    video_path = request.session.get("file_name")
    if not video_path:
        _log("[error] No file_name in session — redirecting to home.")
        return redirect("ml_app:home")

    print("[ml_app] >>> HIT /predict <<<", flush=True)

    for k in ["prediction", "preds_csv", "thresholds"]:
        request.session.pop(k, None)
    request.session.modified = True

    video_name = os.path.basename(video_path)
    video_root = os.path.splitext(video_name)[0]
    out_img_dir = os.path.join(settings.PROJECT_DIR, "uploaded_images")
    os.makedirs(out_img_dir, exist_ok=True)

    def _coerce_results_info(ret):
        results, info = [], {}
        try:
            if isinstance(ret, tuple):
                if len(ret) >= 2:
                    results, info = ret[0], ret[1]
                elif len(ret) == 1:
                    results = ret[0]
            elif isinstance(ret, dict):
                if "results" in ret or "preds" in ret:
                    results = ret.get("results") or ret.get("preds") or []
                    info = ret.get("info") or {
                        k: v for k, v in ret.items() if k not in ("results", "preds")
                    }
                else:
                    info = ret
            else:
                results = ret if ret is not None else []
        except Exception as e:
            _log(f"[warn] _coerce_results_info failed: {e}")
        return results or [], info or {}

    # ---------------- Read frames (for UI + inference) ----------------
    try:
        frames = _read_frames_bgr(video_path)
    except Exception as e:
        _log(f"[error] Failed to read frames: {e}")
        frames = []

    preprocessed_images, faces_cropped_images = [], []

    if frames:
        preview_idxs = _evenly_spaced_indices(len(frames), min(6, MAX_SEQ_LEN))
        for i, idx in enumerate(preview_idxs):
            rgb_full = cv2.cvtColor(frames[idx], cv2.COLOR_BGR2RGB)
            img_path = os.path.join(out_img_dir, f"{video_root}_pre_{i+1}.png")
            Image.fromarray(rgb_full).save(img_path)
            preprocessed_images.append(os.path.basename(img_path))

    try:
        _log(
            f"[debug] frames={len(frames)} "
            f"sampleW={frames[0].shape[1] if frames else 'n/a'} "
            f"sampleH={frames[0].shape[0] if frames else 'n/a'}"
        )

        if frames:
            max_previews = 6
            detect_idxs = _evenly_spaced_indices(
                len(frames), min(max_previews, len(frames))
            )
            target_w = 320
            padding = 30

            for i, idx in enumerate(detect_idxs):
                rgb_full = cv2.cvtColor(frames[idx], cv2.COLOR_BGR2RGB)

                h, w = rgb_full.shape[:2]
                if w > target_w:
                    scale = target_w / float(w)
                    rgb_small = cv2.resize(
                        rgb_full,
                        (target_w, int(h * scale)),
                        interpolation=cv2.INTER_LINEAR,
                    )
                else:
                    scale = 1.0
                    rgb_small = rgb_full

                boxes_small = face_recognition.face_locations(
                    rgb_small, model="hog", number_of_times_to_upsample=HOG_UPSAMPLE
                )
                if not boxes_small:
                    continue

                # Rescale boxes back
                boxes = []
                for top, right, bottom, left in boxes_small:
                    boxes.append(
                        (
                            int(top / scale),
                            int(right / scale),
                            int(bottom / scale),
                            int(left / scale),
                        )
                    )

                # Save first crop
                top, right, bottom, left = boxes[0]
                H, W = rgb_full.shape[:2]
                top = max(0, top - padding)
                left = max(0, left - padding)
                bottom = min(H, bottom + padding)
                right = min(W, right + padding)
                crop = rgb_full[top:bottom, left:right]
                if crop.size == 0:
                    continue

                cp = os.path.join(out_img_dir, f"{video_root}_face_{i+1}.png")
                Image.fromarray(crop).save(cp)
                faces_cropped_images.append(os.path.basename(cp))
    except Exception as e:
        _log(f"[warn] Face-crop preview failed: {e}")

    seq_len = int(getattr(settings, "SEQ_LEN", 24))

    if not frames:
        ctx = {
            "preprocessed_images": [],
            "faces_cropped_images": [],
            "heatmap_images": [],
            "original_video": video_name,
            "models_location": os.path.join(settings.PROJECT_DIR, "models"),
            "output": "N/A",
            "display_label": "N/A",
            "display_icon": "❔",
            "confidence": 0.0,
            "prob_real": 0.0,
            "prob_fake": 0.0,
            "seq_len": seq_len,
            "threshold": None,
            "margin": None,
            "debug_stamp": f"views.py @ {__file__}",
            "class_probs": [],
            "message": "Could not read any frames from the video.",
        }
        return render(request, PREDICT_TMPL, ctx)

    try:
        sel_idx = _evenly_spaced_indices(len(frames), min(seq_len, len(frames)))
        sampled_frames = [frames[i] for i in sel_idx]
    except Exception as e:
        _log(f"[error] Sampling frames failed: {e}")
        sampled_frames = frames[:seq_len]

    try:
        x = _tensor_sequence_from_frames(sampled_frames, seq_len)
    except Exception as e:
        _log(f"[error] Full-frame preprocess failed: {e}")
        try:
            x = _tensor_sequence_from_frames_faces(sampled_frames, seq_len)
        except Exception as ee:
            _log(f"[fatal] Could not preprocess video frames: {ee}")
            ctx = {
                "preprocessed_images": preprocessed_images,
                "faces_cropped_images": faces_cropped_images,
                "heatmap_images": [],
                "original_video": video_name,
                "models_location": os.path.join(settings.PROJECT_DIR, "models"),
                "output": "N/A",
                "display_label": "N/A",
                "display_icon": "❔",
                "confidence": 0.0,
                "prob_real": 0.0,
                "prob_fake": 0.0,
                "seq_len": seq_len,
                "threshold": None,
                "margin": None,
                "debug_stamp": f"views.py @ {__file__}",
                "class_probs": [],
                "message": "Could not preprocess video frames.",
            }
            return render(request, PREDICT_TMPL, ctx)

    features_dir = os.path.join(settings.PROJECT_DIR, "extracted_features")
    os.makedirs(features_dir, exist_ok=True)

    uniq = f"{int(time.time()*1000)}_{uuid.uuid4().hex[:8]}"
    npy_path = os.path.join(
        features_dir, f"{os.path.splitext(video_name)[0]}__{uniq}.npy"
    )
    feat_model = get_feature_model()

    try:
        if os.path.exists(npy_path):
            feats_np = np.load(npy_path, allow_pickle=False)
            _log(f"[features] loaded cache {feats_np.shape} <- {npy_path}")
        else:
            feats = extract_features_batch(x, feat_model)  # [1,T,2048]
            feats_np = feats.squeeze(0).cpu().numpy()  # [T,2048]
            np.save(npy_path, feats_np)
            _log(f"[features] wrote {feats_np.shape} -> {npy_path}")
    except Exception as e:
        _log(f"[error] Feature extraction / save failed: {e}")
        ctx = {
            "preprocessed_images": preprocessed_images,
            "faces_cropped_images": faces_cropped_images,
            "heatmap_images": [],
            "original_video": video_name,
            "models_location": os.path.join(settings.PROJECT_DIR, "models"),
            "output": "N/A",
            "display_label": "N/A",
            "display_icon": "❔",
            "confidence": 0.0,
            "prob_real": 0.0,
            "prob_fake": 0.0,
            "seq_len": seq_len,
            "threshold": None,
            "margin": None,
            "debug_stamp": f"views.py @ {__file__}",
            "class_probs": [],
            "message": "Feature extraction failed. Please retry this video.",
        }
        return render(request, PREDICT_TMPL, ctx)
    else:
        ret = predict_npy_files([npy_path], require_top1=True, delta=0.0)
        results, info = _coerce_results_info(ret)
        thr_str = ""
        try:
            if isinstance(info, dict):
                thr_str = str(info.get("thresholds", ""))
        except Exception:
            thr_str = ""
        try:
            os.remove(npy_path)
        except Exception:
            pass

    # ---------------- Unpack results and render ----------------
    r0 = results[0] if results else {}
    display_label = "Prediction ready"
    display_icon = "✅"
    confidence = 0.0
    prob_real = 0.0
    prob_fake = 0.0
    class_probs = []

    try:
        if isinstance(r0, dict):
            display_label = r0.get("label") or r0.get("decision") or display_label
            confidence = float(r0.get("confidence", confidence))
            prob_real = float(r0.get("prob_real", prob_real))
            prob_fake = float(r0.get("prob_fake", prob_fake))
            class_probs = r0.get("class_probs", class_probs)
        elif isinstance(r0, (list, tuple)) and len(r0) >= 2:
            probs = r0[1]
            if hasattr(probs, "tolist"):
                probs = probs.tolist()
            if isinstance(probs, (list, tuple)) and len(probs) >= 2:
                prob_fake = float(probs[0])  # index 0 = fake
                prob_real = float(probs[1])  # index 1 = real
                confidence = max(prob_real, prob_fake)
                display_label = "real" if prob_real >= prob_fake else "fake"
                class_probs = [
                    {"label": "fake", "prob": prob_fake},
                    {"label": "real", "prob": prob_real},
                ]
                print(
                    f"[debug] probs={probs} -> fake={prob_fake:.3f}, real={prob_real:.3f}"
                )

    except Exception as e:
        _log(f"[warn] Could not unpack results cleanly: {e}")

    if display_label.lower().startswith("real"):
        display_icon = "🟢"
    elif display_label.lower().startswith("fake"):
        display_icon = "🔴"
    else:
        display_icon = "ℹ️"

    ctx = {
        "preprocessed_images": preprocessed_images,
        "faces_cropped_images": faces_cropped_images,
        "heatmap_images": [],
        "original_video": video_name,
        "models_location": os.path.join(settings.PROJECT_DIR, "models"),
        "output": display_label,
        "display_label": display_label,
        "display_icon": display_icon,
        "confidence": (
            round(float(confidence), 4) if isinstance(confidence, (int, float)) else 0.0
        ),
        "prob_real": (
            round(float(prob_real), 4) if isinstance(prob_real, (int, float)) else 0.0
        ),
        "prob_fake": (
            round(float(prob_fake), 4) if isinstance(prob_fake, (int, float)) else 0.0
        ),
        "seq_len": seq_len,
        "threshold": thr_str,
        "margin": None,
        "debug_stamp": f"views.py @ {__file__}",
        "class_probs": class_probs,
        "message": None,
    }

    return render(request, PREDICT_TMPL, ctx)


def about(request):
    return render(request, ABOUT_TMPL)


def handler404(request, exception):
    return render(request, "404.html", status=404)


def cuda_full(request):
    return render(request, "cuda_full.html")
