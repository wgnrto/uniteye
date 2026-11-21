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
            if (float.IsNaN(measurement.x) || float.IsNaN(measurement.y) ||
                float.IsInfinity(measurement.x) || float.IsInfinity(measurement.y))
                return _x;

            float now = Time.unscaledTime;
            if (_lastUpdateTime < 0f)
            {
                _x = measurement;
                _p = Mathf.Max(R, 1e-6f);
                _lastUpdateTime = now;
                return _x;
            }

            float dt = Mathf.Max(0f, now - _lastUpdateTime);
            _lastUpdateTime = now;
            _p = _p + Mathf.Max(0f, Q) * (dt * ReferenceRate);

            // measurement update
            float denom = _p + Mathf.Max(0f, R);
            _k = denom > 1e-12f ? _p / denom : 1f;
            _x = _x + _k * (measurement - _x);
            _p = (1.0f - _k) * _p;

            return _x;
        }
    }
}
