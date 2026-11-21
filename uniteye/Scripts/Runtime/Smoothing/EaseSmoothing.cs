using UnityEngine;
namespace UnitEye
{

    public class EaseSmoothing : Smoothing
    {
        //Frame rate the Factor slider was tuned at (HomulerGaze sets Application.targetFrameRate = 30).
        private const float ReferenceRate = 30f;

        public float Factor { get; set; }

        private Vector2 _easeMeasurement = Vector2.zero;
        //Timestamp of the last Update; < 0 means "no update yet" (first call uses the frame delta).
        private float _lastUpdateTime = -1f;

        public EaseSmoothing(float factor)
        {
            Factor = factor;
        }

        public void Reset()
        {
            _easeMeasurement = Vector2.zero;
            _lastUpdateTime = -1f;
        }

        public override Vector2 Update(Vector2 measurement)
        {
            if (float.IsNaN(measurement.x) || float.IsNaN(measurement.y) ||
                float.IsInfinity(measurement.x) || float.IsInfinity(measurement.y))
                return _easeMeasurement;

            float now = Time.unscaledTime;
            if (_lastUpdateTime < 0f)
            {
                _easeMeasurement = measurement;
                _lastUpdateTime = now;
                return _easeMeasurement;
            }

            float dt = Mathf.Max(0f, now - _lastUpdateTime);
            _lastUpdateTime = now;
            float clamped = Mathf.Clamp01(Factor);
            float f = clamped >= 1f ? 1f : 1f - Mathf.Pow(1f - clamped, dt * ReferenceRate);

            _easeMeasurement.x += (measurement.x - _easeMeasurement.x) * f;
            _easeMeasurement.y += (measurement.y - _easeMeasurement.y) * f;

            return _easeMeasurement;
        }
    }
}
