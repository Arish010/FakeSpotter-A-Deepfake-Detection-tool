# debug_checkpoint.py
import os, json, torch, numpy as np
from Django_Application.ml_app.models_lstm import LSTMClassifier

WEIGHTS = r"C:\Users\Asus\Desktop\Project Msc\Deepfake_detection_using_deep_learning\lstm_binary_weights.pth"


def safe_load(path, device):
    try:
        return torch.load(path, map_location=device, weights_only=True)
    except TypeError:
        return torch.load(path, map_location=device)


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    chk = safe_load(WEIGHTS, device)
    state = chk.get("state_dict") or chk
    h = chk.get("hparams", {})
    input_size = chk.get("input_size", 2048)
    hidden_size = h.get("hidden_size", 512)
    num_layers = h.get("num_layers", 2)
    dropout = h.get("dropout", 0.5)
    bidir = h.get("bidirectional", True)
    num_classes = chk.get("num_classes", 2)

    print(
        "[hparams]",
        dict(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout,
            bidirectional=bidir,
            num_classes=num_classes,
        ),
    )

    m = LSTMClassifier(input_size, hidden_size, num_layers, num_classes, dropout, bidir)
    missing, unexpected = m.load_state_dict(state, strict=False)
    print("missing keys:", len(missing), "→", missing[:6])
    print("unexpected keys:", len(unexpected), "→", unexpected[:6])

    # parameter norms
    for n, p in m.named_parameters():
        print(f"norm[{n:30s}] = {p.detach().abs().mean().item():.6f}")

    m.eval()

    # probe with different inputs
    T, F = 60, input_size
    x_zero = torch.zeros(1, T, F)
    x_rand1 = torch.randn(1, T, F) * 0.5
    x_rand2 = torch.randn(1, T, F) * 1.5

    with torch.no_grad():
        for name, xb in [("zeros", x_zero), ("rand1", x_rand1), ("rand2", x_rand2)]:
            logits = m(xb)
            probs = torch.softmax(logits, dim=1).numpy()[0]
            print(f"{name:6s} → probs = {probs}")


if __name__ == "__main__":
    main()
