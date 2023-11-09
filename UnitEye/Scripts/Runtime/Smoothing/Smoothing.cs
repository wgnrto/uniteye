using UnityEngine;

public class Smoothing
{
    public virtual Vector2 Update(Vector2 measurment)
    {
        return measurment;
    }
}

//Filter options
public enum Filtering
{
    None, Kalman, Easing, KalmanEasing, EasingKalman, OneEuro
}
