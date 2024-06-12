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
            if (PlayerPrefs.GetString("webCamName") != webCamName)
                Debug.Log("Webcam changed, camera values could be wrong. Please calibrate blinking, drowsiness and distance to get accurate values.");

            //Load saved values in PlayerPrefs
            _cameraFOV = PlayerPrefs.GetFloat("camFov", 78f);
            _focalLengthToFloat = PlayerPrefs.GetFloat("fLtF", 0.7471345f);
            _irisInMm = PlayerPrefs.GetFloat("irisMm", 11.8f);
            EFMean = PlayerPrefs.GetFloat("EFMean", -1000f);
            EFStd = PlayerPrefs.GetFloat("EFStd", -1000f);
            BlinkingThreshold = PlayerPrefs.GetFloat("BlinkThresh", 0.135f);

            PlayerPrefs.SetString("webCamName", webCamName);
            //Save in case default values were loaded from PlayerPrefs
            SaveCameraValues();
            SaveDrowsyBlinkingValues();
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

        /// <summary>
        /// Calculates 2D euclidean distance between two Vector2 points in pixels based on Screen size
        /// </summary>
        /// <param name="pointA">First point</param>
        /// <param name="pointB">Second point</param>
        /// <returns>Distance in pixels as float</returns>
        public float DistanceFloat2Pixels(Vector2 pointA, Vector2 pointB)
        {
            Vector2 pointAPixels = new Vector2(pointA.x * Screen.width, pointA.y * Screen.height);
            Vector2 pointBPixels = new Vector2(pointB.x * Screen.width, pointB.y * Screen.height);

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
            float aspectRatio = (float)Screen.width / (float)Screen.height;

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
            float d1 = DistanceFloat2Pixels(GetEyeVertex(385).xy, GetEyeVertex(380).xy);
            float d2 = DistanceFloat2Pixels(GetEyeVertex(386).xy, GetEyeVertex(374).xy);
            float d3 = DistanceFloat2Pixels(GetEyeVertex(387).xy, GetEyeVertex(373).xy);
            float D = DistanceFloat2Pixels(GetEyeVertex(362).xy, GetEyeVertex(263).xy);
            return (d1 + d2 + d3) / (3 * D);
        }

        /// <summary>
        /// Calculates eye aspect ratio for the right eye based on pixels
        /// </summary>
        /// <returns>Aspect ratio as float</returns>
        public float EyeAspectRatioRight()
        {
            var d1 = DistanceFloat2Pixels(GetEyeVertex(160).xy, GetEyeVertex(144).xy);
            var d2 = DistanceFloat2Pixels(GetEyeVertex(159).xy, GetEyeVertex(145).xy);
            var d3 = DistanceFloat2Pixels(GetEyeVertex(158).xy, GetEyeVertex(153).xy);
            var D = DistanceFloat2Pixels(GetEyeVertex(33).xy, GetEyeVertex(133).xy);
            return (d1 + d2 + d3) / (3 * D);
        }

        /// <summary>
        /// Calculates average of both eye aspect ratios and multiplies with cosine of head pitch and yaw
        /// </summary>
        /// <returns>Average aspect ratio multiplied by cosine of head pitch and yaw</returns>
        public float EyeFeature()
        {
            float yaw = _faceMesh.HeadYaw;
            float pitch = _faceMesh.HeadPitch;

            return ((EyeAspectRatioLeft() + EyeAspectRatioRight()) / 2) * Mathf.Abs(Mathf.Cos(yaw)) / Mathf.Abs(Mathf.Cos(pitch));
        }

        /// <summary>
        /// Checks if user is drowsy by comparing EyeFeature() to threshold
        /// </summary>
        /// <param name="threshold">Custom threshold, 0.135f is standard, bigger than 0</param>
        /// <returns>true if blinking, false if not</returns>
        public bool IsBlinking(float threshold = -1.0f)
        {
            if (threshold < 0)
                threshold = BlinkingThreshold;

            return EyeFeature() < threshold;
        }

        /// <summary>
        /// Checks if user is drowsy by comparing EyeFeature() to calibrated stats
        /// </summary>
        /// <param name="threshold">Custom threshold, -3f is standard</param>
        /// <returns>true if drowsy, false if not</returns>
        public bool IsDrowsy(float threshold = -3f)
        {
            var eyeFeature = EyeFeature();

            //If not calibrated or EyeFeauture() isNaN return false
            if (EFMean < 0 || EFStd < 0 || float.IsNaN(eyeFeature))
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
            float lSize = GetIrisSize(_faceMesh.LeftIrisLandmarks, camFOV, undistort);
            float rSize = GetIrisSize(_faceMesh.LeftIrisLandmarks, camFOV, undistort);

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

            float sizeOnScreen = GetIrisBigger(camFOV, undistort) / Mathf.Max(Screen.width, Screen.height);
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
            Vector2 center;

            float sizeOnScreen = GetIrisBigger(out center, camFOV, undistort) / Mathf.Max(Screen.width, Screen.height);

            var y = DistanceFloat2Pixels(center, new Vector2(0.5f, 0.5f)) / Mathf.Max(Screen.width, Screen.height);
            var x = Mathf.Sqrt(_focalLengthToFloat * _focalLengthToFloat + y * y);

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
            Vector2 center;

            float sizeOnScreen = GetIrisBigger(out center, camFOV, undistort) / Mathf.Max(Screen.width, Screen.height);

            var x = distanceInMm / _irisInMm * sizeOnScreen;
            var y = DistanceFloat2Pixels(center, new Vector2(0.5f, 0.5f)) / Mathf.Max(Screen.width, Screen.height);
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
            float sizeOnScreen = GetIrisBigger() / Mathf.Max(Screen.width, Screen.height);

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
            //Add 10% margin
            BlinkingThreshold = EyeFeature() * 1.10f;
            SaveDrowsyBlinkingValues();
        }

        /// <summary>
        /// Calibrates drowsiness stats, calibrates until 60 calibrations and calibrating is false
        /// </summary>
        /// <param name="calibrating">true to start and keep calibrating, false to stop</param>
        public void CalibrateDrowsyStats(bool calibrating)
        {
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
                _eyeFeatures.Add(EyeFeature());
                CalibrationCount++;
            }
            else
            {
                //Calibration end, calculate stats by calculating mean and standard deviation of calibrations
                Calibrating = false;
                EFMean = CalcListMean(_eyeFeatures);
                EFStd = CalcListStd(_eyeFeatures, EFMean);
            }
            SaveDrowsyBlinkingValues();
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

            float sizeOnScreen = GetIrisBigger(camFOV, undistort) / Mathf.Max(Screen.width, Screen.height);
            float angleCovered = sizeOnScreen * camFOV * Mathf.Deg2Rad;

            float distanceInMm = (irisInMm / 2) / Mathf.Tan(angleCovered / 2);

            _distanceInMMSmoothed = _distanceInMMSmoothed < -999f || float.IsPositiveInfinity(_distanceInMMSmoothed) ? distanceInMm : _distanceInMMSmoothed + (distanceInMm - _distanceInMMSmoothed) * _distanceSmoothingWeight;

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

            Vector2 center;

            float sizeOnScreen = GetIrisBigger(out center, camFOV, undistort) / Mathf.Max(Screen.width, Screen.height);

            var y = DistanceFloat2Pixels(center, new Vector2(0.5f, 0.5f)) / Mathf.Max(Screen.width, Screen.height);
            var x = Mathf.Sqrt(focalLengthToFloat * focalLengthToFloat + y * y);

            float distanceInMm = (irisInMm * x) / sizeOnScreen;

            _distanceInMMFocalSmoothed = _distanceInMMFocalSmoothed < -999f || float.IsPositiveInfinity(_distanceInMMFocalSmoothed) ? distanceInMm : _distanceInMMFocalSmoothed + (distanceInMm - _distanceInMMFocalSmoothed) * _distanceSmoothingWeight;

            return _distanceInMMFocalSmoothed;
        }

        /// <summary>
        /// Calculates visual angle based on distance between two points
        /// </summary>
        /// <param name="pointA">First point, not in pixels</param>
        /// <param name="pointB">Second point, not in pixels</param>
        /// <param name="camFOV">Camera fov in degrees, -1f to use saved value</param>
        /// <param name="undistort">true attempts undistortion, false doesnt</param>
        /// <param name="irisInMm">Iris size in mm, -1f to use saved value</param>
        /// <returns>visual angle in degrees</returns>
        public float DistanceToVisualAngle(Vector2 pointA, Vector2 pointB, float camFOV = -1.0f, bool undistort = false, float irisInMm = -1.0f)
        {
            Vector2 center = new Vector2(0.5f, 0.5f);
            float dpi = Screen.dpi;

            float distanceAInMm = Functions.PixelsToMm(DistanceFloat2Pixels(pointA, center), dpi);
            float distanceBInMm = Functions.PixelsToMm(DistanceFloat2Pixels(pointB, center), dpi);

            float camDistance = CalculateCamDistance(camFOV, undistort, irisInMm);

            float angleA = Mathf.Atan(distanceAInMm / camDistance);
            float angleB = Mathf.Atan(distanceBInMm / camDistance);

            return Mathf.Abs(angleA - angleB) * Mathf.Rad2Deg;
        }

        /// <summary>
        /// Calculates visual angle based on distance between two points
        /// </summary>
        /// <param name="pointA">First point, not in pixels</param>
        /// <param name="pointB">Second point, not in pixels</param>
        /// <param name="focalLengthToPixels">Focal length to pixels, -1f to use saved value</param>
        /// <param name="camFOV">Camera fov in degrees, -1f to use saved value</param>
        /// <param name="undistort">true attempts undistortion, false doesnt</param>
        /// <param name="irisInMm">Iris size in mm, -1f to use saved value</param>
        /// <returns>visual angle in degrees</returns>
        public float DistanceToVisualAngleFocal(Vector2 pointA, Vector2 pointB, float focalLengthToPixels = -1.0f, float camFOV = -1.0f, bool undistort = false, float irisInMm = -1.0f)
        {
            Vector2 center = new Vector2(0.5f, 0.5f);
            float dpi = Screen.dpi;

            float distanceAInMm = Functions.PixelsToMm(DistanceFloat2Pixels(pointA, center), dpi);
            float distanceBInMm = Functions.PixelsToMm(DistanceFloat2Pixels(pointB, center), dpi);

            float camDistance = CalculateCamDistanceFocal(focalLengthToPixels, camFOV, undistort, irisInMm);

            float angleA = Mathf.Atan(distanceAInMm / camDistance);
            float angleB = Mathf.Atan(distanceBInMm / camDistance);

            return Mathf.Abs(angleA - angleB) * Mathf.Rad2Deg;
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
            Vector2 center = new Vector2(0.5f, 0.5f);
            if (camFOV < 0)
                camFOV = _cameraFOV;

            var fovRad = camFOV * Mathf.Deg2Rad;

            var dist = DistanceFloat2Pixels(point, center) / Mathf.Max(Screen.width, Screen.height);
            if (barrelDistortion)
            {
                var barrelizedDist = (1 / fovRad) * Mathf.Atan(2 * dist * Mathf.Tan(fovRad * 0.5f));
                return center + (point - center) * (barrelizedDist / dist);
            }
            else
            {
                var pincushionedDist = (Mathf.Tan(dist * fovRad)) / (2 * Mathf.Tan(fovRad * 0.5f));
                return center + (point - center) * (pincushionedDist / dist);
            }
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

            var fovRad = camFOV * Mathf.Deg2Rad;

            var dist = DistanceFloat2Pixels(point, center) / Mathf.Max(Screen.width, Screen.height);
            if (barrelDistortion)
            {
                var barrelizedDist = (1 / fovRad) * Mathf.Atan(2 * dist * Mathf.Tan(fovRad * 0.5f));
                return barrelizedDist / dist;
            }
            else
            {
                var pincushionedDist = (Mathf.Tan(dist * fovRad)) / (2 * Mathf.Tan(fovRad * 0.5f));
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
            float sum = 0f;
            foreach (float entry in list)
            {
                sum += Mathf.Pow((entry - mean), 2);
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
            GUI.Label(new Rect((1 - _faceMesh.LeftIrisLandmarks[0].X) * Screen.width, _faceMesh.LeftIrisLandmarks[0].Y * Screen.height, 50, 50), $"0");
            GUI.Label(new Rect((1 - _faceMesh.LeftIrisLandmarks[1].X) * Screen.width, _faceMesh.LeftIrisLandmarks[1].Y * Screen.height, 50, 50), $"1");
            GUI.Label(new Rect((1 - _faceMesh.LeftIrisLandmarks[2].X) * Screen.width, _faceMesh.LeftIrisLandmarks[2].Y * Screen.height, 50, 50), $"2");
            GUI.Label(new Rect((1 - _faceMesh.LeftIrisLandmarks[3].X) * Screen.width, _faceMesh.LeftIrisLandmarks[3].Y * Screen.height, 50, 50), $"3");
            GUI.Label(new Rect((1 - _faceMesh.LeftIrisLandmarks[4].X) * Screen.width, _faceMesh.LeftIrisLandmarks[4].Y * Screen.height, 50, 50), $"4");

            GUI.Label(new Rect((1 - _faceMesh.RightIrisLandmarks[0].X) * Screen.width, _faceMesh.RightIrisLandmarks[0].Y * Screen.height, 50, 50), $"0");
            GUI.Label(new Rect((1 - _faceMesh.RightIrisLandmarks[1].X) * Screen.width, _faceMesh.RightIrisLandmarks[1].Y * Screen.height, 50, 50), $"1");
            GUI.Label(new Rect((1 - _faceMesh.RightIrisLandmarks[2].X) * Screen.width, _faceMesh.RightIrisLandmarks[2].Y * Screen.height, 50, 50), $"2");
            GUI.Label(new Rect((1 - _faceMesh.RightIrisLandmarks[3].X) * Screen.width, _faceMesh.RightIrisLandmarks[3].Y * Screen.height, 50, 50), $"3");
            GUI.Label(new Rect((1 - _faceMesh.RightIrisLandmarks[4].X) * Screen.width, _faceMesh.RightIrisLandmarks[4].Y * Screen.height, 50, 50), $"4");
        }

        #endregion
    }
}
