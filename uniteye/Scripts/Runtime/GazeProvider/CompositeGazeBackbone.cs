// Excluded from WebGL player builds: depends on the native MediaPipe plugin + Inference Engine.
#if !UNITY_WEBGL || UNITY_EDITOR
using Mediapipe.Unity;
using UnityEngine;

namespace UnitEye
{
    /// <summary>
    /// Ensemble gaze backbone: runs EyeMU (eye crops) AND a direction model (face crop) every frame and
    /// exposes their CONCATENATED calibration features. The two signals are complementary — EyeMU resolves
    /// fine eye detail while the L2CS-style direction model is robust to the whole-face pose — and the
    /// per-user calibration (ridge/MLP) learns how to weight them. Roughly doubles the per-frame inference
    /// cost; both sub-backbones must produce a result for a frame to count (a partial feature vector would
    /// be jagged and poison the calibration). RawGaze and the debug thumbnails come from EyeMU.
    /// </summary>
    public class CompositeGazeBackbone : IGazeBackbone
    {
        /// <summary>
        /// EyeMU's full vector plus the direction model's leading gaze-angle polynomial block. The direction
        /// model's own head-pose + iris tail is EXCLUDED: those values come from the shared FaceMesh and are
        /// already in the EyeMU block — duplicating them adds collinearity, not information.
        /// </summary>
        public const int FeatureCount = HomulerEyeMURunner.FeatureCount + GazeEstimationRunner.GazeAngleTermCount;

        private readonly HomulerEyeMURunner _eyeMU;
        private readonly GazeEstimationRunner _direction;
        private readonly float[] _features = new float[FeatureCount];
        private bool _eyeMUReady, _directionReady, _discardPartial;

        public CompositeGazeBackbone(HomulerEyeMURunner eyeMU, GazeEstimationRunner direction)
        {
            _eyeMU = eyeMU;
            _direction = direction;
        }

        public bool HasPendingReadback => _eyeMU.HasPendingReadback || _direction.HasPendingReadback;

        public bool PollAsyncReadback()
        {
            if (_eyeMU.PollAsyncReadback()) _eyeMUReady = true;
            if (_direction.PollAsyncReadback()) _directionReady = true;

            if (!HasPendingReadback && (_eyeMUReady || _directionReady))
            {
                bool both = _eyeMUReady && _directionReady && !_discardPartial;
                _eyeMUReady = false;
                _directionReady = false;
                _discardPartial = false;
                return both;
            }
            return false;
        }

        public bool PerformInference(WebCamSource webcam)
        {
            bool published = PollAsyncReadback();
            if (HasPendingReadback)
                return published;

            //Evaluate both sub-backbones without short-circuiting in async mode so both schedule on the
            //exact same camera frame and never drift a frame apart.
            bool eyeSync = _eyeMU.PerformInference(webcam);
            if (!eyeSync && !_eyeMU.HasPendingReadback)
                return published;

            bool dirSync = _direction.PerformInference(webcam);
            if (_eyeMU.HasPendingReadback || _direction.HasPendingReadback)
            {
                if (_eyeMU.HasPendingReadback != _direction.HasPendingReadback)
                    _discardPartial = true;
                return published;
            }

            return (eyeSync && dirSync) || published;
        }

        public Vector2 RawGaze => _eyeMU.RawGaze;

        /// <summary>
        /// From the direction sub-backbone — the only half with an output distribution. RawGaze comes from
        /// EyeMU, but the ensemble's calibrated prediction is fit on BOTH feature blocks, so a frame the
        /// direction model was unsure about is a frame the ensemble's fit is shakier on too.
        /// </summary>
        public Vector2 AngularUncertainty => _direction.AngularUncertainty;

        public float[] Features
        {
            get
            {
                ConcatFeatures(_eyeMU.Features, _direction.Features, _features);
                return _features;
            }
        }

        /// <summary>
        /// [EyeMU features (19), direction gaze-angle polynomial (first 7)]. The EyeMU block leads, so the
        /// head-pose slots stay at EyeMU's indices 11/12/13 (HomulerGazeCalibration.HeadPoseFeatureIndices
        /// relies on this). Static so the layout is smoke-testable without instantiating GPU workers.
        /// </summary>
        public static void ConcatFeatures(float[] eyeMUFeatures, float[] directionFeatures, float[] dest)
        {
            System.Array.Copy(eyeMUFeatures, 0, dest, 0, HomulerEyeMURunner.FeatureCount);
            System.Array.Copy(directionFeatures, 0, dest, HomulerEyeMURunner.FeatureCount,
                GazeEstimationRunner.GazeAngleTermCount);
        }

        public RenderTexture LeftEyeTexture => _eyeMU.LeftEyeTexture;
        public RenderTexture RightEyeTexture => _eyeMU.RightEyeTexture;

        //RawGaze comes from EyeMU, so its published frame's capture time is the honest timestamp.
        public double CaptureTimestamp => _eyeMU.CaptureTimestamp;

        public void Dispose()
        {
            _eyeMU?.Dispose();
            _direction?.Dispose();
        }
    }
}
#endif
