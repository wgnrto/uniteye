// Excluded from WebGL player builds: the native backbones depend on the MediaPipe plugin + Inference
// Engine (no wasm). On WebGL the browser owns the CV, so this seam is native-only.
#if !UNITY_WEBGL || UNITY_EDITOR
using Mediapipe.Unity;
using UnityEngine;

namespace UnitEye
{
    /// <summary>
    /// The gaze-estimation model behind the native provider. Given the current webcam frame it produces a
    /// raw (pre-calibration) gaze estimate plus a feature vector for the calibration layer. This is the
    /// seam that lets the user swap EyeMU for a different model (e.g. a yakhyo/gaze-estimation ONNX) without
    /// touching the calibration / filtering / AOI / CSV stack, which consumes RawGaze + Features unchanged.
    ///
    /// Head pose and the auxiliary blink/drowsy/distance signals stay in NativeGazeProvider/HomulerEyeHelper
    /// (they come from the shared MediaPipe FaceMesh), so a backbone only has to output gaze + features.
    /// </summary>
    public interface IGazeBackbone : System.IDisposable
    {
        /// <summary>Run one frame of inference. Returns true if a fresh gaze estimate is available.</summary>
        bool PerformInference(WebCamSource webcam);

        /// <summary>Raw (pre-calibration) gaze in pixels, (0,0) at the top-left.</summary>
        Vector2 RawGaze { get; }

        /// <summary>
        /// The backbone's own per-frame uncertainty about the CURRENTLY PUBLISHED estimate, as an angular
        /// standard deviation per axis (yaw, pitch) in radians. <c>NaN</c> on either axis means "this
        /// backbone has no uncertainty to report" — which is a real answer, not a failure, and consumers
        /// must treat it as such rather than substituting a number.
        ///
        /// Only the bin-classification backbones can fill this in: their output IS a distribution, so its
        /// spread is free (see GazeEstimationRunner.DecodeAngleRadians). A backbone that regresses a point
        /// directly, like EyeMU, has nothing to measure and reports NaN.
        ///
        /// Absolute values are NOT comparable across backbones — a model with broad heads always reports a
        /// large sigma. Consumers compare a frame against the SESSION's own running baseline; see
        /// <see cref="GazeConfidence"/>.
        /// </summary>
        Vector2 AngularUncertainty { get; }

        /// <summary>
        /// Feature vector fed to the calibration model. Its layout is backbone-specific (EyeMU emits its
        /// 12-value embedding+gaze+head-geometry vector; a direction-based backbone emits pitch/yaw+head
        /// pose). The calibration models (RidgeRegression / SimpleMLP) are generic over the vector, so each
        /// backbone needs its own calibration files — recalibrate after switching backbones.
        /// </summary>
        float[] Features { get; }

        /// <summary>Debug eye-crop (or face-crop) textures shown by the Gaze UI; may be null.</summary>
        RenderTexture LeftEyeTexture { get; }
        RenderTexture RightEyeTexture { get; }

        /// <summary>Capture time (Time.unscaledTimeAsDouble) of the camera frame behind the CURRENTLY
        /// published RawGaze/Features — one frame older than "now" in async-readback mode. Consumers use it
        /// to pair gaze with world/AOI state as it was when the user actually looked.</summary>
        double CaptureTimestamp { get; }
    }
}
#endif
