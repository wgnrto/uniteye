using System.Collections.Generic;
using UnityEngine;

/// <summary>
/// Calibration preset which moves in a horziontal wavy pattern
/// </summary>
public class VerticalWavyPreset : CalibrationPreset
{
    private int _radius = Screen.width / 12;
    private const int STEPS = 36;

    public VerticalWavyPreset(float padding, out bool stopAtWaypoint) : base(padding)
    {
        stopAtWaypoint = false;
    }

    public override List<Vector2> GetPoints()
    {
        // "Padding" of 1/4 of Full HD height (270px) and 1/6 of Full HD width (320px)
        List<Vector2> points = new()
        {
            new(Screen.width / 6, Screen.height / 4)
        };

        // angle for stepsize in radians
        float angle = 180 / STEPS * Mathf.Deg2Rad;

        for (int i = 0; i < 2; i++)
        {
            // Center point for first half circle with radius = 160 (i.e. 1/12 of Full HD width): (480, 675) 
            for (int n = -STEPS / 2; n <= STEPS / 2; n++)
            {
                points.Add(new Vector2(_radius * Mathf.Sin(angle * n) + Screen.width / 4 + Screen.width / 3 * i, _radius * Mathf.Cos(angle * n) + Screen.height *3/4 - Screen.width / 12));
            }

            // Center point for second half circle: (800, 405) 
            for (int n = -STEPS; n <= 0; n++)
            {
                points.Add(new Vector2(_radius * Mathf.Cos(angle * n) + Screen.width * 5/12 + Screen.width / 3 * i, _radius * Mathf.Sin(angle * n) + Screen.height/4 + Screen.width/12));
            }
        }

        points.Add(new(Screen.width * 5/6, Screen.height * 3/4));

        return points;
    }
}
