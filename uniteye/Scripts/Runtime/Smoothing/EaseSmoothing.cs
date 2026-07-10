using UnityEngine;

public class EaseSmoothing : Smoothing
{
    //Frame rate the Factor slider was tuned at (HomulerGaze sets Application.targetFrameRate = 30).
    private const float ReferenceRate = 30f;

    public float Factor { get; set; }

    private Vector2 _easeMeasurement = Vector2.zero;

    public EaseSmoothing(float factor)
    {
        Factor = factor;
    }

    public override Vector2 Update(Vector2 measurement)
    {
        //Frame-rate-independent easing: the effective factor is Factor at the 30 fps reference
        //(so existing tuning is preserved) and stays consistent at other frame rates, instead of the
        //old fixed-per-frame factor that made a 60 fps run ~twice as responsive as a 30 fps run.
        float dt = Time.unscaledDeltaTime;
        float f = 1f - Mathf.Pow(1f - Mathf.Clamp01(Factor), dt * ReferenceRate);

        _easeMeasurement.x += (measurement.x - _easeMeasurement.x) * f;
        _easeMeasurement.y += (measurement.y - _easeMeasurement.y) * f;

        return _easeMeasurement;
    }
}
