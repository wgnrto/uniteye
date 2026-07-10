// Excluded from WebGL player builds: depends on the native MediaPipe plugin (Mediapipe.Runtime
// has no wasm library, so IL2CPP linking fails). Kept for the Editor regardless of build target.
#if !UNITY_WEBGL || UNITY_EDITOR
using Mediapipe.Unity;
using Mediapipe.Unity.FaceMesh;
using UnityEngine;

namespace UnitEye
{
    /// <summary>
    /// Native (Windows/macOS/Linux) gaze provider: the homuler MediaPipe FaceMesh for landmarks +
    /// the EyeMU model on Unity's Inference Engine (via HomulerEyeMURunner), plus HomulerEyeHelper for
    /// blink/drowsy/distance. This is the desktop implementation of IGazeProvider; it cannot run on WebGL
    /// (native plugin), which is exactly why the seam exists.
    /// </summary>
    public class NativeGazeProvider : IGazeProvider
    {
        private readonly FaceMeshSolution _faceMesh;
        private readonly WebCamSource _webcam;
        private readonly HomulerEyeMURunner _runner;
        private readonly HomulerEyeHelper _eyeHelper;

        private Vector2 _rawGaze;
        private bool _isBlinking;
        private bool _isDrowsy;
        private float _distanceMm = -1000f;
        //EyeFeature() is 8 distance calcs + trig; compute it once per Tick and reuse for blink, drowsy
        //and the CSV EyeFeature accessor instead of recomputing it 3x per frame.
        private float _eyeFeature = float.NaN;

        public NativeGazeProvider(GameObject mediaPipeGO)
        {
            _webcam = mediaPipeGO.GetComponent<WebCamSource>();
            _faceMesh = mediaPipeGO.GetComponent<FaceMeshSolution>();
            _eyeHelper = new HomulerEyeHelper(_faceMesh, _webcam.name);
            _runner = new HomulerEyeMURunner(_faceMesh);
        }

        public bool Tick()
        {
            if (!_runner.PerformInference(_webcam))
                return false;

            _rawGaze = new Vector2(_runner.NetworkOutput[0], _runner.NetworkOutput[1]);
            //Compute EyeFeature once, then derive blink/drowsy from it (was recomputed inside each call).
            _eyeFeature = _eyeHelper.EyeFeature();
            _isDrowsy = _eyeHelper.IsDrowsyFromFeature(_eyeFeature);
            _isBlinking = _eyeHelper.IsBlinkingFromFeature(_eyeFeature);
            _distanceMm = _eyeHelper.CalculateCamDistanceFocal();
            return true;
        }

        public Vector2 RawGaze => _rawGaze;
        //Returns the runner's reused feature buffer (no per-frame copy). Valid only until the next
        //Tick; the calibration capture, which retains samples, clones it (see HomulerGazeCalibration).
        public float[] GetFeatures() => _runner.Features;
        public bool IsFacePresent => _faceMesh != null && _faceMesh.FaceLandmarks != null;
        public bool IsBlinking => _isBlinking;
        public bool IsDrowsy => _isDrowsy;
        public float DistanceMm => _distanceMm;
        public float EyeFeature => _eyeFeature;
        public Vector3 HeadPoseEuler => new Vector3(_runner.HeadPitch, _runner.HeadYaw, _runner.HeadRoll);
        public RenderTexture LeftEyeTexture => _runner.LeftEyeTexture;
        public RenderTexture RightEyeTexture => _runner.RightEyeTexture;

        public bool AnnotateFaceMesh
        {
            get => _faceMesh != null && _faceMesh.Annotate;
            set { if (_faceMesh != null) _faceMesh.Annotate = value; }
        }

        public bool IsCalibratingDrowsy => _eyeHelper.Calibrating;
        public int DrowsyCalibrationCount => _eyeHelper.CalibrationCount;
        public void CalibrateDistance() => _eyeHelper.CalibrateFocalLength();
        public void CalibrateBlinking() => _eyeHelper.CalibrateBlinking();
        public void CalibrateDrowsy(bool calibrating) => _eyeHelper.CalibrateDrowsyStats(calibrating);

        public string CurrentCameraName => _webcam != null ? _webcam.sourceName : "";

        public void NextCamera()
        {
            if (_webcam == null) return;
            _webcam.SelectSource(_webcam.GetCameraIndex() + 1);
            _eyeHelper.CameraChanged(_webcam.sourceName);
        }

        public void PreviousCamera()
        {
            if (_webcam == null) return;
            _webcam.SelectSource(_webcam.GetCameraIndex() - 1);
            _eyeHelper.CameraChanged(_webcam.sourceName);
        }

        public void Dispose() => _runner?.Dispose();
    }
}
#endif
