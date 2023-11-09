using UnityEngine;

public class KalmanFilter : Smoothing
{
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

    public void Reset()
    {
        _k = 0;
        _x = Vector2.zero;
        _p = 1.0f;
    }

    public override Vector2 Update(Vector2 measurement)
    {
        // prediction
        // no state transition, just update the covariance
        _p = _p + Q;

        // measurement update
        _k = _p / (_p + R);
        _x = _x + _k * (measurement - _x);
        _p = (1.0f - _k) * _p;

        return _x;
    }
}
