using System.Collections.Generic;
using UnityEngine;

namespace UnitEye
{
    /// <summary>
    /// Platform seam between the webcam→gaze COMPUTER VISION — which differs per platform (native
    /// MediaPipe + Inference Engine on desktop; browser JavaScript on WebGL) — and the
    /// platform-independent gaze CONSUMER layer (calibration, One-Euro filtering, AOI hit-testing,
    /// CSV logging), which is identical everywhere.
    ///
    /// A provider owns its own input source (the webcam) and produces a raw, pre-calibration gaze
    /// estimate plus the per-frame signals the rest of UnitEye needs. Swap the implementation per
    /// platform; everything downstream stays the same.
    /// </summary>
    public interface IGazeProvider
    {
        /// <summary>Advance one frame of inference. Returns true if a fresh gaze estimate is available.</summary>
        bool Tick();

        /// <summary>Raw (pre-calibration) gaze in pixels, (0,0) at the top-left.</summary>
        Vector2 RawGaze { get; }

        /// <summary>
        /// Feature vector for the calibration model. May be empty if only a raw gaze point is available.
        /// Implementations may return a reused per-frame buffer valid only until the next Tick(); callers
        /// that retain the values (e.g. calibration sample capture) must copy it.
        /// </summary>
        float[] GetFeatures();

        /// <summary>True while a face is currently being tracked.</summary>
        bool IsFacePresent { get; }

        /// <summary>Capture time (Time.unscaledTimeAsDouble) of the camera frame behind the current
        /// RawGaze — the closest observable proxy for when the user actually looked. Consumers use it to
        /// pair gaze samples with world/AOI state at that moment (a 100-300ms pipeline lag times a moving
        /// object's speed is a systematic AOI error no gaze-model improvement can fix). 0 until the first
        /// sample; on WebGL this is the sample's ARRIVAL time (browser capture latency is not observable).</summary>
        double CaptureTimestamp { get; }

        bool IsBlinking { get; }
        bool IsDrowsy { get; }

        /// <summary>Disagreement between the two eyes' normalized iris offsets (conjugate eyes should
        /// nearly agree) — a free per-frame quality proxy that spikes on half-blinks/occlusion/landmark
        /// failures. 0 when unavailable.</summary>
        float BinocularIrisDisagreement { get; }

        /// <summary>Estimated distance from the camera in mm (negative sentinel if unavailable).</summary>
        float DistanceMm { get; }

        /// <summary>Eye-aspect-ratio-based feature used for CSV logging / drowsiness (NaN if unavailable).</summary>
        float EyeFeature { get; }

        /// <summary>Head pose in radians as (pitch, yaw, roll); Vector3.zero if unavailable.</summary>
        Vector3 HeadPoseEuler { get; }

        /// <summary>Debug eye-crop textures; may be null.</summary>
        RenderTexture LeftEyeTexture { get; }
        RenderTexture RightEyeTexture { get; }

        /// <summary>Toggle the debug face-mesh landmark overlay (native MediaPipe only; WebGL no-ops).</summary>
        bool AnnotateFaceMesh { get; set; }

        /// <summary>Show/hide the provider's debug rendering (the native camera preview; the face-mesh
        /// overlay is gated separately by AnnotateFaceMesh). Calibration/evaluation hide it so the preview
        /// does not distract. WebGL no-ops — the browser owns rendering.</summary>
        void SetRendering(bool rendering);

        /// <summary>Swap the gaze model at runtime (native only; WebGL no-ops — the browser owns the CV).
        /// The calibration is per-backbone, so the pipeline falls back to raw gaze until recalibrated.</summary>
        void SetBackbone(GazeBackbone backbone);

        // Calibration of the auxiliary signals (blink / drowsy / distance). WebGL providers may no-op.
        bool IsCalibratingDrowsy { get; }
        int DrowsyCalibrationCount { get; }
        void CalibrateDistance();
        void CalibrateBlinking();
        void CalibrateDrowsy(bool calibrating);

        // Webcam selection (native). WebGL implementations may no-op or drive the JS side.
        string CurrentCameraName { get; }
        void NextCamera();
        void PreviousCamera();

        void Dispose();
    }
}
