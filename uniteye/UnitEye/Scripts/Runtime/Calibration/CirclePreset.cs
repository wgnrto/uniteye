using System.Collections;
using System.Collections.Generic;
using UnityEngine;

/// <summary>
/// Calibration preset which moves in a circle pattern
/// </summary>
public class CirclePreset : CalibrationPreset
{
    private float _radius;
    private int _steps;

    public CirclePreset(float padding, float radius, int steps, out bool stopAtWaypoint) : base(padding)
    {
        _radius = radius;
        stopAtWaypoint = false;
        _steps = steps;
    }

    public override List<Vector2> GetPoints()
    {
        List<Vector2> points = new List<Vector2>();
        float angle = 360 / _steps;

        for (int i = 0; i < _steps; i++)
        {
            points.Add(new Vector2(_radius * Mathf.Cos(Mathf.Deg2Rad * angle * i) + Screen.width / 2, _radius * Mathf.Sin(Mathf.Deg2Rad * angle * i) + Screen.height / 2));
        }

        // add start point at at the end to complete the circle
        points.Add(points[0]);

        return points;
    }
}
