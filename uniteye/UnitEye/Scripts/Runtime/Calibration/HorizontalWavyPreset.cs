using System.Collections;
using System.Collections.Generic;
using UnityEngine;

/// <summary>
/// Calibration preset which moves in a horziontal wavy pattern
/// </summary>
public class HorizontalWavyPreset : CalibrationPreset
{
    private float _radius = Screen.height / 8;
    private int _steps = 36;

    public HorizontalWavyPreset(float padding, out bool stopAtWaypoint) : base(padding)
    {
        stopAtWaypoint = false;
    }

    public override List<Vector2> GetPoints()
    {
        // "Padding" of 1/4 of Full HD height (270px)
        List<Vector2> points = new();
        points.Add(new(Screen.height / 4, Screen.height / 4));

        // angle for stepsize in radians
        float angle = 180 / _steps * Mathf.Deg2Rad;

        // Center point for first half circle with radius = 135 (i.e. 1/8 of Full HD height): (1525, 405) 
        for (int i = -_steps/2; i <= _steps/2; i++)
        {
            points.Add(new Vector2(_radius * Mathf.Cos(angle * i) + Screen.width - Screen.height * 3/8, _radius * Mathf.Sin(angle * i) + Screen.height * 3/8));
        }

        // Center point for second half circle: (405, 675)
        for (int i = -_steps; i <= 0; i++)
        {
            points.Add(new Vector2(_radius * Mathf.Sin(angle * i) + Screen.height * 3 / 8, _radius * Mathf.Cos(angle * i) + Screen.height * 5/8));
        }

        points.Add(new(Screen.width - Screen.height/4, Screen.height * 3/4));

        return points;
    }
}
