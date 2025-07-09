using System.Collections.Generic;
using UnityEngine;

/// <summary>
/// Calibration preset which moves in a ZigZag pattern
/// </summary>
public class ZigZagPreset : CalibrationPreset
{
    private bool _vertical;

    private int _numberOfZigs;

    public ZigZagPreset(float padding, bool vertical, int numberOfZigs) :
        base(padding)
    {
        _vertical = vertical;
        _numberOfZigs = numberOfZigs;
    }

    public override List<Vector2> GetPoints()
    {
        List<Vector2> points = new List<Vector2>();

        if (_vertical)
        {
            var screenWidthPadded = Screen.width - 2 * padding;
            var screenWidth4th = screenWidthPadded / _numberOfZigs;
            float currentSegmentX = 2 * padding;
            float currentSegmentY = Screen.height - padding;

            for (int i = 0; i < _numberOfZigs + 1; i++)
            {
                points
                    .Add(new Vector2(currentSegmentX - padding,
                        currentSegmentY));
                points
                    .Add(new Vector2(currentSegmentX - padding,
                        currentSegmentY - Screen.height + 2 * padding));
                currentSegmentX += screenWidth4th;
            }
        }
        else
        {
            var screenHeightPadded = Screen.height - 2 * padding;
            var screenHeight4th = screenHeightPadded / _numberOfZigs;
            float currentSegmentY = padding;
            float currentSegmentX = padding;

            for (int i = 0; i < _numberOfZigs + 1; i++)
            {
                points.Add(new Vector2(currentSegmentX, currentSegmentY));
                points
                    .Add(new Vector2(currentSegmentX +
                        Screen.width -
                        2 * padding,
                        currentSegmentY));
                currentSegmentY += screenHeight4th;
            }
        }

        return points;
    }
}
