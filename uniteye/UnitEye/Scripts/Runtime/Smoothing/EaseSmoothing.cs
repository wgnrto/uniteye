using UnityEngine;

public class EaseSmoothing : Smoothing
{
    public float Factor { get; set; }

    private Vector2 _easeMeasurement = Vector2.zero;

    public EaseSmoothing(float factor)
    {
        Factor = factor;
    }

    public override Vector2 Update(Vector2 measurement)
    {
        _easeMeasurement.x += (measurement.x - _easeMeasurement.x) * Factor;
        _easeMeasurement.y += (measurement.y - _easeMeasurement.y) * Factor;

        return _easeMeasurement;
    }
}
