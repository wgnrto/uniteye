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

        public override Vector2 Update(Vector2 measurement)
        {
            //Frame-rate-independent easing: the effective factor is Factor at the 30 fps reference
            //(so existing tuning is preserved) and stays consistent at other frame rates, instead of the
            //old fixed-per-frame factor that made a 60 fps run ~twice as responsive as a 30 fps run.
            //Uses the REAL elapsed time since the last filter update, not this frame's delta: HomulerGaze
            //skips Update while the provider tick fails, and after such a gap the ease must cover the
            //whole gap instead of crawling from the stale pre-gap position.
            float now = Time.unscaledTime;
            float dt = _lastUpdateTime < 0f ? Time.unscaledDeltaTime : now - _lastUpdateTime;
            _lastUpdateTime = now;
            float f = 1f - Mathf.Pow(1f - Mathf.Clamp01(Factor), dt * ReferenceRate);

            _easeMeasurement.x += (measurement.x - _easeMeasurement.x) * f;
            _easeMeasurement.y += (measurement.y - _easeMeasurement.y) * f;

            return _easeMeasurement;
        }
    }
}
