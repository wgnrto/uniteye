#if UNITY_WEBGL
using UnityEngine;

namespace UnitEye
{
    /// <summary>
    /// WebGL gaze provider — receives browser-computed gaze via the JS bridge (verified end-to-end
    /// in real Unity 6.3/6.5 WebGL builds: JS -> UnitEyeWebGL.jslib -> SendMessage -> WebGLGazeReceiver -> here).
    ///
    /// On WebGL the native MediaPipe plugin cannot run, so the computer vision must happen in the browser:
    /// getUserMedia → MediaPipe FaceLandmarker (@mediapipe/tasks-vision) → EyeMU in onnxruntime-web / TF.js.
    /// That JS pipeline computes the raw gaze point and pushes it into Unity via a .jslib plugin that calls
    /// the static ReportGaze(...) below on each frame. Everything downstream — calibration, One-Euro filter,
    /// AOI hit-testing, CSV logging — is shared with the native path through IGazeProvider, unchanged.
    ///
    /// Build order (see docs/WEBGL.md): (1) author the JS pipeline; (2) a UnitEyeWebGL.jslib that receives
    /// the browser gaze and calls SendMessage / an emscripten callback into ReportGaze; (3) leave the rest.
    /// </summary>
    public class WebGLGazeProvider : IGazeProvider
    {
        // The JS side writes the latest browser-computed sample here via the .jslib bridge.
        private static Vector2 s_rawGaze;
        private static bool s_facePresent;
        private static bool s_blinking;
        private static bool s_newSample;
        private static float[] s_features = new float[0];

        private static GameObject s_receiverGO;

        public WebGLGazeProvider()
        {
            // Spawn the JS<->Unity bridge: it starts the browser pipeline (getUserMedia + MediaPipe +
            // EyeMU) and calls ReportGaze/ReportFeatures each frame. Idempotent: a scene reload creating
            // a new provider reuses the existing receiver instead of stacking pipelines (the JS side is
            // also idempotent, see uniteye-webgl-boot.js).
            if (s_receiverGO == null)
            {
                s_receiverGO = new GameObject("UnitEyeWebGLReceiver");
                Object.DontDestroyOnLoad(s_receiverGO);
                s_receiverGO.AddComponent<WebGLGazeReceiver>();
            }
        }

        /// <summary>Called from JavaScript (via UnitEyeWebGL.jslib) with the latest browser-computed gaze, in pixels.</summary>
        public static void ReportGaze(float pixelX, float pixelY, bool facePresent, bool blinking)
        {
            s_rawGaze = new Vector2(pixelX, pixelY);
            s_facePresent = facePresent;
            s_blinking = blinking;
            //ARRIVAL time: browser capture latency upstream of this call is not observable from here.
            s_captureTimestamp = Time.unscaledTimeAsDouble;
            s_newSample = true;
        }
        private static double s_captureTimestamp;

        /// <summary>Optional: the JS side can also supply the EyeMU feature vector so Unity-side calibration matches the native path.</summary>
        public static void ReportFeatures(float[] features) => s_features = features ?? new float[0];

        private static bool s_loggedFirstTick;

        public bool Tick()
        {
            if (!s_newSample) return false;
            s_newSample = false;
            if (!s_loggedFirstTick)
            {
                s_loggedFirstTick = true;
                // Proves HomulerGaze's update loop consumed a browser-provided sample end to end.
                Debug.Log($"UNITEYE_WEBGL_PROVIDER_OK gaze=({s_rawGaze.x:F1},{s_rawGaze.y:F1})");
            }
            return true;
        }

        public Vector2 RawGaze => s_rawGaze;
        public float[] GetFeatures() => s_features;
        public double CaptureTimestamp => s_captureTimestamp;
        public bool IsFacePresent => s_facePresent;
        public bool IsBlinking => s_blinking;
        public bool IsDrowsy => false;                 // TODO: compute browser-side if you need drowsiness
        public float BinocularIrisDisagreement => 0f;  // TODO: compute browser-side if needed
        public float DistanceMm => -1000f;             // TODO: browser-side distance if needed
        public float EyeFeature => float.NaN;
        public Vector3 HeadPoseEuler => Vector3.zero;  // TODO: head pose from FaceLandmarker if needed
        public RenderTexture LeftEyeTexture => null;
        public RenderTexture RightEyeTexture => null;
        public bool AnnotateFaceMesh { get => false; set { } }  // browser owns rendering on WebGL
        public void SetRendering(bool rendering) { }            // browser owns rendering on WebGL
        public void SetBackbone(GazeBackbone backbone) { }      // browser owns the CV on WebGL

        public bool IsCalibratingDrowsy => false;
        public int DrowsyCalibrationCount => 0;
        public void CalibrateDistance() { }
        public void CalibrateBlinking() { }
        public void CalibrateDrowsy(bool calibrating) { }

        public string CurrentCameraName => "browser";
        public void NextCamera() { }
        public void PreviousCamera() { }

        public void Dispose() { }
    }
}
#endif
