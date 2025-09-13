// static/js/script.js
// Robust client-side helpers for the upload page.

(function ($) {
    "use strict";

    // ---- Config (client-side UX only; server still enforces real limits) ----
    const MAX_UPLOAD_BYTES = 100 * 1024 * 1024; // 100 MB
    const SLIDER_VALUES = [10, 20, 40, 60, 80, 100]; // sequence_length options

    function showFormAlert($form, msg, type = "danger") {
        $form.find(".js-form-alert").remove();
        const $alert = $(`
      <div class="alert alert-${type} js-form-alert" role="alert">
        ${msg}
      </div>
    `);
        const $btn = $form.find('button[type="submit"]').first();
        if ($btn.length) {
            $btn.closest(".form-group, .mb-3, form").before($alert);
        } else {
            $form.prepend($alert);
        }
        setTimeout(() => $alert.fadeOut(200, () => $alert.remove()), 5000);
    }

    // Format bytes for a friendly message
    function prettyBytes(bytes) {
        if (bytes === 0) return "0 B";
        const k = 1024;
        const sizes = ["B", "KB", "MB", "GB", "TB"];
        const i = Math.floor(Math.log(bytes) / Math.log(k));
        return `${(bytes / Math.pow(k, i)).toFixed(i ? 1 : 0)} ${sizes[i]}`;
    }

    $(function () {

        const $form = $("#video-upload");
        if (!$form.length) return;

        const $fileInput = $("#id_upload_video_file");
        const $video = $("#videos");
        const $videoSource = $("#video_source");

        const $slider = $("#slider");
        const $seqHidden = $("#id_sequence_length"); // Django default id is usually "id_<fieldname>"
        const $seqLabel = $("#slider-value");

        if ($slider.length && $seqHidden.length && $seqLabel.length) {
            // Initial value index (default: second element = 20)
            const initialIndex = 1;
            $slider.slider({
                value: initialIndex,
                min: 0,
                max: SLIDER_VALUES.length - 1,
                slide: function (_event, ui) {
                    const v = SLIDER_VALUES[ui.value];
                    $seqHidden.val(v);
                    $seqLabel.text(v);
                }
            });
            const v0 = SLIDER_VALUES[$slider.slider("value")];
            $seqHidden.val(v0);
            $seqLabel.text(v0);
        }

        // --- File change: validate + preview video ---
        $fileInput.on("change", function () {
            const file = this.files && this.files[0];
            $form.find(".js-form-alert").remove();

            if (!file) {
                // Nothing chosen
                return;
            }
            if (!file.type || !file.type.startsWith("video/")) {
                showFormAlert($form, "Please choose a valid video file.");
                this.value = ""; // reset input
                return;
            }
            if (file.size > MAX_UPLOAD_BYTES) {
                showFormAlert(
                    $form,
                    `File is too large (${prettyBytes(file.size)}). Maximum allowed is ${prettyBytes(MAX_UPLOAD_BYTES)}.`
                );
                this.value = ""; // reset input
                return;
            }

            try {
                // Setup preview
                const url = URL.createObjectURL(file);
                $videoSource.attr("src", url);
                $video.get(0).load();
                $video.css("display", "block");


            } catch (e) {
                showFormAlert($form, "Could not preview this video. You can still upload it.");
            }
        });

        $form.on("submit", function (e) {
            const file = $fileInput.get(0).files && $fileInput.get(0).files[0];
            if (!file) {
                e.preventDefault();
                showFormAlert($form, "Please select a video file before uploading.");
                return;
            }

            const $btn = $("#videoUpload");
            $btn.prop("disabled", true)
                .attr("aria-busy", "true")
                .html('Uploading Video&nbsp;<span class="spinner-border spinner-border-sm" role="status" aria-hidden="true"></span><span class="visually-hidden">Loading...</span>');
        });
    });
})(jQuery);
