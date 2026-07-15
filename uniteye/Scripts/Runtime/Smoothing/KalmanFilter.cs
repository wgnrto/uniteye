using UnityEngine;
namespace UnitEye
{

    public class KalmanFilter : Smoothing
    {
        //Frame rate the Q/R sliders were tuned at (HomulerGaze sets Application.targetFrameRate = 30).
        private const float ReferenceRate = 30f;

        public float Q { get; set; }

        public float R { get; set; }

        private float _k;

        private Vector2 _x;

        private float _p;

        public KalmanFilter(float q = 1e-5f, float r = 1e-4f)
        {
            Q = q;
            R = r;
            Reset();
        }

        //Timestamp of the last Update; < 0 means "no update yet" (first call uses the frame delta).
        private float _lastUpdateTime;

        public void Reset()
        {
            _k = 0;
            _x = Vector2.zero;
            _p = 1.0f;
            _lastUpdateTime = -1f;
        }

        public override Vector2 Update(Vector2 measurement)
        {
            // prediction
            // no state transition, just grow the covariance by the process noise. Scale it by the REAL
            // elapsed time since the last filter update (normalized to the 30 fps reference, so this
            // equals the old Q at 30 fps) — not just this frame's delta: HomulerGaze skips Update entirely
            // while the provider tick fails (face lost, webcam stall), and after such a gap the covariance
            // must grow by the whole gap or the gain is smallest exactly when the input likely jumped.
            float now = Time.unscaledTime;
            float dt = _lastUpdateTime < 0f ? Time.unscaledDeltaTime : now - _lastUpdateTime;
            _lastUpdateTime = now;
            _p = _p + Q * (dt * ReferenceRate);

            // measurement update
            _k = _p / (_p + R);
            _x = _x + _k * (measurement - _x);
            _p = (1.0f - _k) * _p;

            return _x;
        }
    }
}
