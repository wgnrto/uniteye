// Excluded from WebGL player builds: depends on the native MediaPipe plugin (Mediapipe.Runtime
// has no wasm library, so IL2CPP linking fails). Kept for the Editor regardless of build target.
#if !UNITY_WEBGL || UNITY_EDITOR
using System.Collections.Generic;
using Unity.Mathematics;
using UnityEngine;
using Mediapipe.Unity.FaceMesh;

namespace UnitEye
{
    public class HomulerEyeHelper
    {
        //FaceMesh reference to be able to access vertex data
        private FaceMeshSolution _faceMesh;

        //Apply barrel distortion (true) or pincushion distortion (false)
        public bool barrelDistortion = false;

        //Camera FOV based on a Logitech C920
        float _cameraFOV = 78f;
        public float CameraFOV
        {
            get => _cameraFOV;
            set
            {
                _cameraFOV = value;
                SaveCameraValues();
            }
        }

        //Average iris size across population in mm
        float _irisInMm = 11.8f;
        public float IrisInMm
        {
            get => _irisInMm;
            set
            {
                _irisInMm = value;
                SaveCameraValues();
            }
        }

        //Dummy focalLength value for mediapipe approach, based on a Logitech C920
        //Ideally should be calculated or calibrated
        float _focalLengthToFloat = 0.7471345f;
        public float FocalLengthToFloat
        {
            get => _focalLengthToFloat;
            set
            {
                _focalLengthToFloat = value;
                SaveCameraValues();
            }
        }

        //Distance values
        private float _distanceSmoothingWeight = 0.1f;
        private float _distanceInMMSmoothed = -1000f;
        private float _distanceInMMFocalSmoothed = -1000f;

        //Blinking Drowsiness values
        public float BlinkingThreshold { get; private set; } = 0.135f;
        public bool Calibrating { get; private set; }
        public int CalibrationCount { get; private set; } = 0;
        private readonly int _maxCalibrationCount = 1000;
        private List<float> _eyeFeatures;
        public float EFMean { get; private set; } = -1000f;
        public float EFStd { get; private set; } = -1000f;
        public float EFSmooth { get; private set; } = -1000f;
        public float Decay { get; private set; } = 0.01f;

        #region Constructors

        /// <summary>
        /// Instantiates EyeHelper and loads saved camera values (FOV, FocalLengthToFloat, IrisInMm) from PlayerPrefs
        /// </summary>
        /// <param name="holisticPipeline">Holistic Pipeline Reference</param>
        /// <param name="webCamName">Webcam name from the WebCamTexture device</param>
        public HomulerEyeHelper(FaceMeshSolution faceMesh, string webCamName)
        {
            _faceMesh = faceMesh;
            LoadSavedValues(webCamName);
        }

        /// <summary>
        /// Instantiates EyeHelper and overwrites camera FOV, other values are default
        /// </summary>
        /// <param name="holisticPipeline">Holistic Pipeline Reference</param>
        /// <param name="camFov">Camera FOV for calculations</param>
        public HomulerEyeHelper(FaceMeshSolution faceMesh, float camFov)
        {
            _faceMesh = faceMesh;
            _cameraFOV = camFov;
        }

        /// <summary>
        /// Instantiates EyeHelper and overwrites camera FOV and focalLengthToFloat, other values are default
        /// </summary>
        /// <param name="holisticPipeline">Holistic Pipeline Reference</param>
        /// <param name="camFov">Camera FOV for calculations</param>
        /// <param name="focalLengthToFloat">Focal Length to Float distance value</param>
        public HomulerEyeHelper(FaceMeshSolution faceMesh, float camFov, float focalLengthToFloat)
        {
            _faceMesh = faceMesh;
            _cameraFOV = camFov;
            _focalLengthToFloat = focalLengthToFloat;
        }

        /// <summary>
        /// Instantiates EyeHelper and overwrites camera FOV, focalLengthToFloat and irisInMm
        /// </summary>
        /// <param name="holisticPipeline">Holistic Pipeline Reference</param>
        /// <param name="camFov">Camera FOV for calculations</param>
        /// <param name="focalLengthToFloat">Focal Length to Float distance value</param>
        /// <param name="irisInMM">Real life iris size in mm</param>
        public HomulerEyeHelper(FaceMeshSolution faceMesh, float camFov, float focalLengthToFloat, float irisInMM)
        {
            _faceMesh = faceMesh;
            _cameraFOV = camFov;
            _focalLengthToFloat = focalLengthToFloat;
            _irisInMm = irisInMM;
        }

        #endregion

        #region HelperFunctions

        /// <summary>
        /// Attempts to load saved values from PlayerPrefs, sets them to default values if not saved.
        /// </summary>
        /// <param name="webCamName">Webcam name from the WebCamTexture device</param>
        void LoadSavedValues(string webCamName)
        {
            string savedCam = PlayerPrefs.GetString("webCamName", "");
            bool camChanged = !string.IsNullOrEmpty(webCamName) && savedCam != webCamName;
            if (camChanged && !string.IsNullOrEmpty(savedCam))
                Debug.Log("Webcam changed, camera values could be wrong. Please calibrate blinking, drowsiness and distance to get accurate values.");

            bool needsInitialSave = !PlayerPrefs.HasKey("camFov") || !PlayerPrefs.HasKey("BlinkThresh");

            //Load saved values in PlayerPrefs
            _cameraFOV = PlayerPrefs.GetFloat("camFov", 78f);
            _focalLengthToFloat = PlayerPrefs.GetFloat("fLtF", 0.7471345f);
            _irisInMm = PlayerPrefs.GetFloat("irisMm", 11.8f);
            EFMean = PlayerPrefs.GetFloat("EFMean", -1000f);
            EFStd = PlayerPrefs.GetFloat("EFStd", -1000f);
            BlinkingThreshold = PlayerPrefs.GetFloat("BlinkThresh", 0.135f);

            if (camChanged)
                PlayerPrefs.SetString("webCamName", webCamName);

            if (needsInitialSave)
            {
                SaveCameraValues();
                SaveDrowsyBlinkingValues();
            }
            else if (camChanged)
            {
                PlayerPrefs.Save();
            }
        }

        /// <summary>
        /// Saves all 3 camera values to PlayerPrefs
        /// </summary>
        void SaveCameraValues()
        {
            PlayerPrefs.SetFloat("camFov", _cameraFOV);
            PlayerPrefs.SetFloat("fLtF", _focalLengthToFloat);
            PlayerPrefs.SetFloat("irisMm", _irisInMm);
            PlayerPrefs.Save();
        }

        /// <summary>
        /// Saves all 3 drowsy and blinking values to PlayerPrefs
        /// </summary>
        void SaveDrowsyBlinkingValues()
        {
            PlayerPrefs.SetFloat("EFMean", EFMean);
            PlayerPrefs.SetFloat("EFStd", EFStd);
            PlayerPrefs.SetFloat("BlinkThresh", BlinkingThreshold);
            PlayerPrefs.Save();
        }

        /// <summary>
        /// Reloads saved values in case newWebCamName is different from current webCamName.
        /// </summary>
        /// <param name="newWebCamName">string new webcam name</param>
        public void CameraChanged(string newWebCamName)
        {
            LoadSavedValues(newWebCamName);
        }

        /// <summary>
        /// Calculates 2D euclidean distance between two Vector2 points
        /// </summary>
        /// <param name="pointA">First point</param>
        /// <param name="pointB">Second point</param>
        /// <returns>Distance in float</returns>
        public float DistanceFloat2(Vector2 pointA, Vector2 pointB)
        {
            Vector2 distance = pointB - pointA;
            return Mathf.Sqrt(distance.x * distance.x + distance.y * distance.y);
        }

        //Camera-frame pixel basis for landmark math. Landmarks are normalized in CAMERA-FRAME space;
        //converting them with the game window's Screen dims made EAR (blink/drowsy) and iris size
        //(distance-to-camera) depend on the window size and aspect — resizing the window or going
        //fullscreen after calibrating silently shifted the calibrated thresholds. Falls back to the
        //window size only while the camera has not delivered a frame yet (calibrate/use with the camera
        //running and the values are consistent).
        private float FrameWidth
        {
            get { float w = _faceMesh != null ? _faceMesh.FrameWidth : 0f; return w > 0f ? w : Screen.width; }
        }
        private float FrameHeight
        {
            get { float h = _faceMesh != null ? _faceMesh.FrameHeight : 0f; return h > 0f ? h : Screen.height; }
        }
        private float FrameMax => Mathf.Max(FrameWidth, FrameHeight);

        /// <summary>
        /// Calculates 2D euclidean distance between two normalized landmark points in CAMERA-FRAME pixels
        /// </summary>
        /// <param name="pointA">First point</param>
        /// <param name="pointB">Second point</param>
        /// <returns>Distance in pixels as float</returns>
        public float DistanceFloat2Pixels(Vector2 pointA, Vector2 pointB)
        {
            float w = FrameWidth, h = FrameHeight;
            Vector2 pointAPixels = new Vector2(pointA.x * w, pointA.y * h);
            Vector2 pointBPixels = new Vector2(pointB.x * w, pointB.y * h);

            Vector2 distancePixels = pointAPixels - pointBPixels;

            return Mathf.Sqrt(distancePixels.x * distancePixels.x + distancePixels.y * distancePixels.y);
        }

        /// <summary>
        /// Calculates 4D euclidean distance between two float4 points
        /// </summary>
        /// <param name="pointA">First point</param>
        /// <param name="pointB">Second point</param>
        /// <returns>Distance in float</returns>
        public float DistanceFloat4(float4 pointA, float4 pointB)
        {
            float4 distance = pointB - pointA;
            return Mathf.Sqrt(distance.x * distance.x + distance.y * distance.y + distance.z * distance.z + distance.w * distance.w);
        }

        #endregion

        #region DrowsyBlinking
        /// <summary>
        /// Removes aspect ratio influence on a point for distance calculation
        /// </summary>
        /// <param name="point">Point, not in pixels</param>
        /// <returns>Vector2 for accurate distance calculation</returns>
        public Vector2 RemoveAspectRatio(Vector2 point)
        {
            //Landmarks live in camera-frame space, so it is the FRAME aspect that distorts distances.
            float aspectRatio = FrameWidth / FrameHeight;

            Vector2 factor;

            if (aspectRatio > 1)
            {
                //horizontal
                factor = new Vector2(1f, 1f / aspectRatio);
            }
            else
            {
                //vertical
                factor = new Vector2(1f * aspectRatio, 1f);
            }

            return point * factor;
        }

        /// <summary>
        /// Get eye vertex from index
        /// </summary>
        /// <param name="index">Vertex index</param>
        /// <returns>float4 vertex</returns>
        public float4 GetEyeVertex(int index)
        {
            var landMark = _faceMesh.FaceLandmarks[index];
            return new float4(landMark.X, landMark.Y, landMark.Z, 0f);
        }

        /// <summary>
        /// Calculates eye aspect ratio for the left eye based on pixels
        /// </summary>
        /// <returns>Aspect ratio as float</returns>
        public float EyeAspectRatioLeft()
        {
            if (_faceMesh == null || _faceMesh.FaceLandmarks == null || _faceMesh.FaceLandmarks.Count <= 387)
                return float.NaN;
            float d1 = DistanceFloat2Pixels(GetEyeVertex(385).xy, GetEyeVertex(380).xy);
            float d2 = DistanceFloat2Pixels(GetEyeVertex(386).xy, GetEyeVertex(374).xy);
            float d3 = DistanceFloat2Pixels(GetEyeVertex(387).xy, GetEyeVertex(373).xy);
            float D = DistanceFloat2Pixels(GetEyeVertex(362).xy, GetEyeVertex(263).xy);
            return D > 1e-6f ? (d1 + d2 + d3) / (3f * D) : float.NaN;
        }

        /// <summary>
        /// Calculates eye aspect ratio for the right eye based on pixels
        /// </summary>
        /// <returns>Aspect ratio as float</returns>
        public float EyeAspectRatioRight()
        {
            if (_faceMesh == null || _faceMesh.FaceLandmarks == null || _faceMesh.FaceLandmarks.Count <= 160)
                return float.NaN;
            var d1 = DistanceFloat2Pixels(GetEyeVertex(160).xy, GetEyeVertex(144).xy);
            var d2 = DistanceFloat2Pixels(GetEyeVertex(159).xy, GetEyeVertex(145).xy);
            var d3 = DistanceFloat2Pixels(GetEyeVertex(158).xy, GetEyeVertex(153).xy);
            var D = DistanceFloat2Pixels(GetEyeVertex(33).xy, GetEyeVertex(133).xy);
            return D > 1e-6f ? (d1 + d2 + d3) / (3f * D) : float.NaN;
        }

        /// <summary>
        /// Calculates average of both eye aspect ratios and multiplies with cosine of head pitch and yaw
        /// </summary>
        /// <returns>Average aspect ratio multiplied by cosine of head pitch and yaw</returns>
        public float EyeFeature()
        {
            if (_faceMesh == null || _faceMesh.FaceLandmarks == null)
                return float.NaN;
            float yaw = _faceMesh.HeadYaw;
            float pitch = _faceMesh.HeadPitch;

            // Looking steeply up/down makes cos(pitch) approach zero. Keep the correction finite so an
            // extreme pose cannot poison blink/drowsiness calibration or the smoothing state.
            //0.05 caps the correction at 20x (within roughly 3° of ±90°), where face landmarks are
            //already too foreshortened for a reliable eye-aspect-ratio correction.
            float pitchCorrection = Mathf.Max(Mathf.Abs(Mathf.Cos(pitch)), 0.05f);
            return ((EyeAspectRatioLeft() + EyeAspectRatioRight()) / 2) * Mathf.Abs(Mathf.Cos(yaw)) / pitchCorrection;
        }

        /// <summary>
        /// Checks if user is drowsy by comparing EyeFeature() to threshold
        /// </summary>
        /// <param name="threshold">Custom threshold, 0.135f is standard, bigger than 0</param>
        /// <returns>true if blinking, false if not</returns>
        public bool IsBlinking(float threshold = -1.0f)
        {
            return IsBlinkingFromFeature(EyeFeature(), threshold);
        }

        /// <summary>
        /// IsBlinking using an already-computed EyeFeature() value, so callers that also need the
        /// feature / drowsiness this frame don't recompute EyeFeature() (8 distance calcs + trig) 3x.
        /// </summary>
        public bool IsBlinkingFromFeature(float eyeFeature, float threshold = -1.0f)
        {
            if (threshold < 0)
                threshold = BlinkingThreshold;

            return eyeFeature < threshold;
        }

        /// <summary>
        /// Checks if user is drowsy by comparing EyeFeature() to calibrated stats
        /// </summary>
        /// <param name="threshold">Custom threshold, -3f is standard</param>
        /// <returns>true if drowsy, false if not</returns>
        public bool IsDrowsy(float threshold = -3f)
        {
            return IsDrowsyFromFeature(EyeFeature(), threshold);
        }

        /// <summary>
        /// IsDrowsy using an already-computed EyeFeature() value (see IsBlinkingFromFeature). Mutates the
        /// smoothed EFSmooth state, so it should be called at most once per frame — as the provider does.
        /// </summary>
        public bool IsDrowsyFromFeature(float eyeFeature, float threshold = -3f)
        {
            //If not calibrated or EyeFeauture() isNaN return false
            if (EFMean < 0 || EFStd <= 1e-6f || float.IsNaN(eyeFeature) || float.IsInfinity(eyeFeature))
                return false;

            if (EFSmooth < -999f)
            {
                //Initialize value
                EFSmooth = eyeFeature;
            }
            else if (EFSmooth >= 0)
            {
                //Smoothed value with decay
                EFSmooth += (eyeFeature - EFSmooth) * Decay;
            }

            //Calculate deviation from calibrated stats
            return ((EFSmooth - EFMean) / EFStd) < threshold;
        }

        #endregion

        #region IrisSize

        /// <summary>
        /// Returns bigger iris size of the two eyes.
        /// Mediapipe tends to degrade the iris size when an eye is partially occluded by the face, hence the bigger one should be accurate
        /// </summary>
        /// <param name="camFOV">Camera fov in degrees, -1f to use saved value</param>
        /// <param name="undistort">true attempts undistortion, false doesnt</param>
        /// <returns>Biggest iris size in pixels float</returns>
        public float GetIrisBigger(float camFOV = -1.0f, bool undistort = false)
        {
            if (_faceMesh == null) return 0f;
            float lSize = GetIrisSize(_faceMesh.LeftIrisLandmarks, camFOV, undistort);
            float rSize = GetIrisSize(_faceMesh.RightIrisLandmarks, camFOV, undistort);

            return lSize >= rSize ? lSize : rSize;
        }

        /// <summary>
        /// Returns bigger iris size of the two eyes and writes according center point to given Vector2.
        /// Mediapipe tends to degrade the iris size when an eye is partially occluded by the face, hence the bigger one should be accurate
        /// </summary>
        /// <param name="center">Center point to be returned</param>
        /// <param name="camFOV">Camera fov in degrees, -1f to use saved value</param>
        /// <param name="undistort">true attempts undistortion, false doesnt</param>
        /// <returns>Biggest iris size in pixels float and Vector2 centerpoint of iris</returns>
        public float GetIrisBigger(out Vector2 center, float camFOV = -1.0f, bool undistort = false)
        {
            if (_faceMesh == null)
            {
                center = new Vector2(0.5f, 0.5f);
                return 0f;
            }
            Vector2 centerLeft;
            Vector2 centerRight;

            float lSize = GetIrisSize(_faceMesh.LeftIrisLandmarks, out centerLeft, camFOV, undistort);
            float rSize = GetIrisSize(_faceMesh.RightIrisLandmarks, out centerRight, camFOV, undistort);

            if (lSize >= rSize)
            {
                center = centerLeft;
                return lSize;
            }
            else
            {
                center = centerRight;
                return rSize;
            }
        }

        /// <summary>
        /// Calculates iris size based on ComputeBuffer
        /// </summary>
        /// <param name="irisLandmarks">Left or right iris Landmarks</param>
        /// <param name="camFOV">Camera fov in degrees, -1f to use saved value</param>
        /// <param name="undistort">true attempts undistortion, false doesnt</param>
        /// <returns>Bigger iris size in pixels float</returns>
        public float GetIrisSize(IList<Mediapipe.NormalizedLandmark> irisLandmarks, float camFOV = -1.0f, bool undistort = false)
        {
            if (irisLandmarks == null || irisLandmarks.Count < 5)
                return 0f;

            float hDistance = (undistort) ? DistanceFloat2Pixels(FOVModelUndistort(new Vector2(irisLandmarks[1].X, irisLandmarks[1].Y), camFOV), FOVModelUndistort(new Vector2(irisLandmarks[3].X, irisLandmarks[3].Y), camFOV))
                : DistanceFloat2Pixels(new Vector2(irisLandmarks[1].X, irisLandmarks[1].Y), new Vector2(irisLandmarks[3].X, irisLandmarks[3].Y));
            float vDistance = (undistort) ? DistanceFloat2Pixels(FOVModelUndistort(new Vector2(irisLandmarks[2].X, irisLandmarks[2].Y), camFOV), FOVModelUndistort(new Vector2(irisLandmarks[4].X, irisLandmarks[4].Y), camFOV))
                : DistanceFloat2Pixels(new Vector2(irisLandmarks[2].X, irisLandmarks[2].Y), new Vector2(irisLandmarks[4].X, irisLandmarks[4].Y));

            return hDistance >= vDistance ? hDistance : vDistance;
        }

        /// <summary>
        /// Calculates iris size based on ComputeBuffer and writes iris centerpoint to given Vector2
        /// </summary>
        /// <param name="irisLandmarks">Left or right iris Landmarks</param>
        /// <param name="center">Center point to be returned</param>
        /// <param name="camFOV">Camera fov in degrees, -1f to use saved value</param>
        /// <param name="undistort">true attempts undistortion, false doesnt</param>
        /// <returns>Bigger iris size in pixels float and Vector2 centerpoint of iris</returns>
        public float GetIrisSize(IList<Mediapipe.NormalizedLandmark> irisLandmarks, out Vector2 center, float camFOV = -1.0f, bool undistort = false)
        {
            if (irisLandmarks == null || irisLandmarks.Count < 5)
            {
                center = new Vector2(0.5f, 0.5f);
                return 0f;
            }

            float hDistance = (undistort) ? DistanceFloat2Pixels(FOVModelUndistort(new Vector2(irisLandmarks[1].X, irisLandmarks[1].Y), camFOV), FOVModelUndistort(new Vector2(irisLandmarks[3].X, irisLandmarks[3].Y), camFOV))
                : DistanceFloat2Pixels(new Vector2(irisLandmarks[1].X, irisLandmarks[1].Y), new Vector2(irisLandmarks[3].X, irisLandmarks[3].Y));
            float vDistance = (undistort) ? DistanceFloat2Pixels(FOVModelUndistort(new Vector2(irisLandmarks[2].X, irisLandmarks[2].Y), camFOV), FOVModelUndistort(new Vector2(irisLandmarks[4].X, irisLandmarks[4].Y), camFOV))
                : DistanceFloat2Pixels(new Vector2(irisLandmarks[2].X, irisLandmarks[2].Y), new Vector2(irisLandmarks[4].X, irisLandmarks[4].Y));

            center = new Vector2(irisLandmarks[0].X, irisLandmarks[0].Y);
            return hDistance >= vDistance ? hDistance : vDistance;
        }

        #endregion

        #region Calibration

        /// <summary>
        /// Reset saved iris size to 11.8f
        /// </summary>
        public void ResetIrisSize()
        {
            _irisInMm = 11.8f;
            SaveCameraValues();
        }

        /// <summary>
        /// Reset all camera values to default
        /// </summary>
        public void ResetCameraValues()
        {
            _cameraFOV = 78f;
            _focalLengthToFloat = 0.7471345f;
            ResetIrisSize();
        }

        /// <summary>
        /// Calibrate iris size based on given distance, standard is 500mm
        /// </summary>
        /// <param name="camFOV">Camera fov in degrees, -1f to use saved value</param>
        /// <param name="undistort">true attempts undistortion, false doesnt</param>
        /// <param name="distanceInMm">Calibration distance in mm</param>
        public void CalibrateIrisSize(float camFOV = -1.0f, bool undistort = false, float distanceInMm = 500f)
        {
            if (camFOV < 0)
                camFOV = _cameraFOV;

            float frameMax = FrameMax;
            float sizeOnScreen = frameMax > 0f ? GetIrisBigger(camFOV, undistort) / frameMax : 0f;
            if (sizeOnScreen <= 1e-6f) return;
            float angleCovered = sizeOnScreen * camFOV * Mathf.Deg2Rad;

            _irisInMm = 2 * distanceInMm * Mathf.Tan(angleCovered / 2);

            Debug.Log($"New Iris Size in mm: {_irisInMm}");
            SaveCameraValues();
        }

        /// <summary>
        /// Calibrate iris size using focal length approach based on given distance, standard is 500mm
        /// </summary>
        /// <param name="camFOV">Camera fov in degrees, -1f to use saved value</param>
        /// <param name="undistort">true attempts undistortion, false doesnt</param>
        /// <param name="distanceInMm">Calibration distance in mm</param>
        public void CalibrateIrisSizeFocal(float camFOV = -1.0f, bool undistort = false, float distanceInMm = 500f)
        {
            Vector2 center = Vector2.zero;
            float frameMax = FrameMax;
            float sizeOnScreen = frameMax > 0f ? GetIrisBigger(out center, camFOV, undistort) / frameMax : 0f;
            if (sizeOnScreen <= 1e-6f) return;

            var y = DistanceFloat2Pixels(center, new Vector2(0.5f, 0.5f)) / frameMax;
            var x = Mathf.Sqrt(_focalLengthToFloat * _focalLengthToFloat + y * y);
            if (x <= 1e-6f) return;

            _irisInMm = distanceInMm * sizeOnScreen / x;

            Debug.Log($"New Iris Size in mm: {_irisInMm}");
            SaveCameraValues();
        }

        /// <summary>
        /// Calibrates mediapipe focal length value based on given distance, standard is 500mm
        /// </summary>
        /// <param name="camFOV">Camera fov in degrees, -1f to use saved value</param>
        /// <param name="undistort">true attempts undistortion, false doesnt</param>
        /// <param name="distanceInMm">Calibration distance in mm</param>
        public void CalibrateFocalLength(float camFOV = -1.0f, bool undistort = false, float distanceInMm = 500f)
        {
            Vector2 center = Vector2.zero;
            float frameMax = FrameMax;
            float sizeOnScreen = frameMax > 0f ? GetIrisBigger(out center, camFOV, undistort) / frameMax : 0f;
            if (sizeOnScreen <= 1e-6f || _irisInMm <= 1e-6f) return;

            var x = distanceInMm / _irisInMm * sizeOnScreen;
            var y = DistanceFloat2Pixels(center, new Vector2(0.5f, 0.5f)) / frameMax;
            _focalLengthToFloat = Mathf.Sqrt(x * x + y * y);

            Debug.Log($"Focal Length Value: {_focalLengthToFloat}");
            SaveCameraValues();
        }

        /// <summary>
        /// Calibrates camera FOV based on given distance, standard is 500mm
        /// </summary>
        /// <param name="distanceInMm">Calibration distance in mm</param>
        public void CalibrateFOV(float distanceInMm = 500f)
        {
            float frameMax = FrameMax;
            float sizeOnScreen = frameMax > 0f ? GetIrisBigger() / frameMax : 0f;
            if (sizeOnScreen <= 1e-6f || distanceInMm <= 1e-6f) return;

            float angleCovered = 2f * Mathf.Atan(_irisInMm / (2f * distanceInMm));
            float camFOV = angleCovered / (sizeOnScreen * Mathf.Deg2Rad);

            _cameraFOV = camFOV;

            Debug.Log($"New FOV in degrees: {_cameraFOV}");
            SaveCameraValues();
        }

        /// <summary>
        /// Sets blinking threshold based on current EyeFeature()
        /// </summary>
        public void CalibrateBlinking()
        {
            //Add 10% margin only when the current frame produced a usable feature.
            float eyeFeature = EyeFeature();
            if (!float.IsNaN(eyeFeature) && !float.IsInfinity(eyeFeature))
            {
                BlinkingThreshold = eyeFeature * 1.10f;
                SaveDrowsyBlinkingValues();
            }
        }

        /// <summary>
        /// Calibrates drowsiness stats, calibrates until 60 calibrations and calibrating is false
        /// </summary>
        /// <param name="calibrating">true to start and keep calibrating, false to stop</param>
        public void CalibrateDrowsyStats(bool calibrating)
        {
            if (!Calibrating && !calibrating)
                return;

            if (calibrating && !Calibrating)
            {
                //Calibration start, initialize values
                Calibrating = true;
                CalibrationCount = 0;
                _eyeFeatures = new List<float>();
            }
            if (Calibrating && (CalibrationCount < 61 || calibrating) && CalibrationCount < _maxCalibrationCount)
            {
                //Calibrating until CalibrationCount > 60 and calibrating is false
                //Calibrationg can be ongoing after CalibrationCount > 60 if calibrating stays true
                //_maxCalibrationCount to not calibrate forever
                float eyeFeature = EyeFeature();
                if (!float.IsNaN(eyeFeature) && !float.IsInfinity(eyeFeature))
                {
                    _eyeFeatures.Add(eyeFeature);
                    CalibrationCount++;
                }
            }
            else if (Calibrating)
            {
                //Calibration end, calculate stats by calculating mean and standard deviation of calibrations
                Calibrating = false;
                if (_eyeFeatures != null && _eyeFeatures.Count > 0)
                {
                    EFMean = CalcListMean(_eyeFeatures);
                    EFStd = CalcListStd(_eyeFeatures, EFMean);
                    SaveDrowsyBlinkingValues();
                }
            }
        }
        #endregion

        #region DistanceToCamera

        /// <summary>
        /// Calculate iris distance to camera based on camera fov
        /// </summary>
        /// <param name="camFOV">Camera fov in degrees, -1f to use saved value</param>
        /// <param name="undistort">true attempts undistortion, false doesnt</param>
        /// <param name="irisInMm">Iris size in mm, -1f to use saved value</param>
        /// <returns>Distance in mm as float</returns>
        public float CalculateCamDistance(float camFOV = -1.0f, bool undistort = false, float irisInMm = -1.0f)
        {
            if (irisInMm < 0.0f)
                irisInMm = _irisInMm;
            if (camFOV < 0)
                camFOV = _cameraFOV;

            float frameMax = FrameMax;
            float sizeOnScreen = frameMax > 0f ? GetIrisBigger(camFOV, undistort) / frameMax : 0f;
            float angleCovered = sizeOnScreen * camFOV * Mathf.Deg2Rad;
            if (sizeOnScreen <= 1e-6f || angleCovered <= 1e-6f)
                return -1000f;

            float tanHalf = Mathf.Tan(angleCovered / 2f);
            if (Mathf.Abs(tanHalf) <= 1e-6f)
                return -1000f;

            float distanceInMm = (irisInMm / 2f) / tanHalf;
            if (float.IsNaN(distanceInMm) || float.IsInfinity(distanceInMm) || distanceInMm <= 0f)
                return -1000f;

            _distanceInMMSmoothed = _distanceInMMSmoothed < -999f || float.IsNaN(_distanceInMMSmoothed) || float.IsInfinity(_distanceInMMSmoothed)
                ? distanceInMm
                : _distanceInMMSmoothed + (distanceInMm - _distanceInMMSmoothed) * _distanceSmoothingWeight;

            return _distanceInMMSmoothed;
        }

        /// <summary>
        /// Mimics mediapipe approach to calculate distance based on focal length value
        /// </summary>
        /// <param name="focalLengthToFloat">Focal length to pixels, -1f to use saved value</param>
        /// <param name="camFOV">Camera fov in degrees, -1f to use saved value</param>
        /// <param name="undistort">true attempts undistortion, false doesnt</param>
        /// <param name="irisInMm">Iris size in mm, -1f to use saved value</param>
        /// <returns>Distance in mm as float</returns>
        public float CalculateCamDistanceFocal(float focalLengthToFloat = -1.0f, float camFOV = -1.0f, bool undistort = false, float irisInMm = -1.0f)
        {
            if (irisInMm < 0.0f)
                irisInMm = _irisInMm;
            if (focalLengthToFloat < 0.0f)
                focalLengthToFloat = _focalLengthToFloat;

            Vector2 center = Vector2.zero;
            float frameMax = FrameMax;
            float sizeOnScreen = frameMax > 0f ? GetIrisBigger(out center, camFOV, undistort) / frameMax : 0f;
            if (sizeOnScreen <= 1e-6f)
                return -1000f;

            var y = DistanceFloat2Pixels(center, new Vector2(0.5f, 0.5f)) / frameMax;
            var x = Mathf.Sqrt(focalLengthToFloat * focalLengthToFloat + y * y);

            float distanceInMm = (irisInMm * x) / sizeOnScreen;
            if (float.IsNaN(distanceInMm) || float.IsInfinity(distanceInMm) || distanceInMm <= 0f)
                return -1000f;

            _distanceInMMFocalSmoothed = _distanceInMMFocalSmoothed < -999f || float.IsNaN(_distanceInMMFocalSmoothed) || float.IsInfinity(_distanceInMMFocalSmoothed)
                ? distanceInMm
                : _distanceInMMFocalSmoothed + (distanceInMm - _distanceInMMFocalSmoothed) * _distanceSmoothingWeight;

            return _distanceInMMFocalSmoothed;
        }

        /// <summary>
        /// Calculates visual angle based on distance between two points
        /// </summary>
        /// <param name="pointA">First point in normalized screen space [0, 1]</param>
        /// <param name="pointB">Second point in normalized screen space [0, 1]</param>
        /// <param name="camFOV">Camera fov in degrees, -1f to use saved value</param>
        /// <param name="undistort">true attempts undistortion, false doesnt</param>
        /// <param name="irisInMm">Iris size in mm, -1f to use saved value</param>
        /// <returns>visual angle in degrees</returns>
        public float DistanceToVisualAngle(Vector2 pointA, Vector2 pointB, float camFOV = -1.0f, bool undistort = false, float irisInMm = -1.0f)
        {
            float camDistance = CalculateCamDistance(camFOV, undistort, irisInMm);
            return ComputeScreenSubtendedAngleDeg(pointA, pointB, camDistance);
        }

        /// <summary>
        /// Calculates visual angle based on distance between two points
        /// </summary>
        /// <param name="pointA">First point in normalized screen space [0, 1]</param>
        /// <param name="pointB">Second point in normalized screen space [0, 1]</param>
        /// <param name="focalLengthToPixels">Focal length to pixels, -1f to use saved value</param>
        /// <param name="camFOV">Camera fov in degrees, -1f to use saved value</param>
        /// <param name="undistort">true attempts undistortion, false doesnt</param>
        /// <param name="irisInMm">Iris size in mm, -1f to use saved value</param>
        /// <returns>visual angle in degrees</returns>
        public float DistanceToVisualAngleFocal(Vector2 pointA, Vector2 pointB, float focalLengthToPixels = -1.0f, float camFOV = -1.0f, bool undistort = false, float irisInMm = -1.0f)
        {
            float camDistance = CalculateCamDistanceFocal(focalLengthToPixels, camFOV, undistort, irisInMm);
            return ComputeScreenSubtendedAngleDeg(pointA, pointB, camDistance);
        }

        private static float ComputeScreenSubtendedAngleDeg(Vector2 pointA, Vector2 pointB, float camDistanceMm)
        {
            if (camDistanceMm <= 0f || float.IsNaN(camDistanceMm) || float.IsInfinity(camDistanceMm))
                return 0f;

            float dpi = Screen.dpi;
            float widthMm = Functions.PixelsToMm(Screen.width, dpi);
            float heightMm = Functions.PixelsToMm(Screen.height, dpi);
            Vector3 vA = new Vector3((pointA.x - 0.5f) * widthMm, (pointA.y - 0.5f) * heightMm, camDistanceMm);
            Vector3 vB = new Vector3((pointB.x - 0.5f) * widthMm, (pointB.y - 0.5f) * heightMm, camDistanceMm);
            return Vector3.Angle(vA, vB);
        }

        #endregion

        #region FOVModel

        /// <summary>
        /// Attempts to undistort the image based on given camera FOV according to the FOV Model in:
        /// https://doi.org/10.1049/iet-its:20080017
        /// barrelDistortion bool decides if barrel or pincushion distortion will be used
        /// </summary>
        /// <param name="point">Point to be undistorted</param>
        /// <param name="camFOV">Camera fov in degrees, -1f to use saved value</param>
        /// <returns>Undistorted Vector2 point</returns>
        public Vector2 FOVModelUndistort(Vector2 point, float camFOV = -1.0f)
        {
            return FOVModelUndistort(point, camFOV, barrelDistortion);
        }
        /// <summary>
        /// Attempts to undistort the image based on given camera FOV according to the FOV Model in:
        /// https://doi.org/10.1049/iet-its:20080017
        /// </summary>
        /// <param name="point">Point to be undistorted</param>
        /// <param name="camFOV">Camera fov in degrees, -1f to use saved value</param>
        /// <param name="barrelDistortion">true to apply barrel, false to apply pincushion distortion</param>
        /// <returns>Undistorted Vector2 point</returns>
        public Vector2 FOVModelUndistort(Vector2 point, float camFOV = -1.0f, bool barrelDistortion = false)
        {
            float factor = FOVModelUndistortFactor(point, camFOV, barrelDistortion);
            Vector2 center = new Vector2(0.5f, 0.5f);
            return center + (point - center) * factor;
        }

        /// <summary>
        /// Attempts to undistort the image based on given camera FOV according to the FOV Model in:
        /// https://doi.org/10.1049/iet-its:20080017
        /// barrelDistortion bool decides if barrel or pincushion distortion will be used
        /// </summary>
        /// <param name="point">Point at which the factor will be calculated</param>
        /// <param name="camFOV">Camera fov in degrees, -1f to use saved value</param>
        /// <returns>Undistortion factor in float</returns>
        public float FOVModelUndistortFactor(Vector2 point, float camFOV = -1.0f)
        {
            return FOVModelUndistortFactor(point, camFOV, barrelDistortion);
        }
        /// <summary>
        /// Attempts to undistort the image based on given camera FOV according to the FOV Model in:
        /// https://doi.org/10.1049/iet-its:20080017
        /// </summary>
        /// <param name="point">Point at which the factor will be calculated</param>
        /// <param name="camFOV">Camera fov in degrees, -1f to use saved value</param>
        /// <param name="barrelDistortion">True to apply barrel, false to apply pincushion distortion</param>
        /// <returns>Undistortion factor in float</returns>
        public float FOVModelUndistortFactor(Vector2 point, float camFOV = -1.0f, bool barrelDistortion = false)
        {
            Vector2 center = new Vector2(0.5f, 0.5f);
            if (camFOV < 0)
                camFOV = _cameraFOV;

            float frameMax = FrameMax;
            var fovRad = camFOV * Mathf.Deg2Rad;
            if (frameMax <= 0f || Mathf.Abs(fovRad) <= 1e-6f)
                return 1f;

            var dist = DistanceFloat2Pixels(point, center) / frameMax;
            if (dist <= 1e-6f)
                return 1f;

            float halfTan = Mathf.Tan(fovRad * 0.5f);
            if (Mathf.Abs(halfTan) <= 1e-6f)
                return 1f;

            if (barrelDistortion)
            {
                var barrelizedDist = (1f / fovRad) * Mathf.Atan(2f * dist * halfTan);
                return barrelizedDist / dist;
            }
            else
            {
                var pincushionedDist = Mathf.Tan(dist * fovRad) / (2f * halfTan);
                return pincushionedDist / dist;
            }
        }

        #endregion

        #region ListFunctions

        /// <summary>
        /// Calculate mean of List<float>
        /// </summary>
        /// <param name="list">List of float values</param>
        /// <returns>Mean of list</returns>
        public static float CalcListMean(List<float> list)
        {
            if (list == null || list.Count == 0)
                return -1000f;
            float sum = 0f;
            foreach (float entry in list)
            {
                sum += entry;
            }
            return sum / list.Count;
        }

        /// <summary>
        /// Calculate standard deviation of List<float>
        /// </summary>
        /// <param name="list">List of float values</param>
        /// <param name="mean">Mean of list</param>
        /// <returns>Standard deviation of list</returns>
        public static float CalcListStd(List<float> list, float mean)
        {
            if (list == null || list.Count == 0)
                return -1000f;
            float sum = 0f;
            foreach (float entry in list)
            {
                float d = entry - mean;
                sum += d * d;
            }
            return Mathf.Sqrt(sum / list.Count);
        }

        #endregion

        #region Debug

        /// <summary>
        /// Shows Iris point locations on the GUI. Must only be called in OnGUI().
        /// </summary>
        public void ShowIrisPointsGUI()
        {
            if (_faceMesh == null) return;
            var left = _faceMesh.LeftIrisLandmarks;
            if (left != null && left.Count >= 5)
            {
                for (int i = 0; i < 5; i++)
                    GUI.Label(new Rect((1 - left[i].X) * Screen.width, left[i].Y * Screen.height, 50, 50), i.ToString());
            }
            var right = _faceMesh.RightIrisLandmarks;
            if (right != null && right.Count >= 5)
            {
                for (int i = 0; i < 5; i++)
                    GUI.Label(new Rect((1 - right[i].X) * Screen.width, right[i].Y * Screen.height, 50, 50), i.ToString());
            }
        }

        #endregion
    }
}
#endif
