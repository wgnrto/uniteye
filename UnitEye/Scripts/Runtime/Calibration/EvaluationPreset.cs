using System.Collections.Generic;
using UnityEngine;

/// <summary>
/// Calibration preset consisting of a grid of points. Used for evaluation only currently.
/// </summary>
public class EvaluationPreset : CalibrationPreset
{
    private int _rows, _columns;

    public EvaluationPreset(float padding, int rows, int columns) :
        base(padding)
    {
        _rows = rows;
        _columns = columns;
    }

    public override List<Vector2> GetPoints()
    {
        List<Vector2> points = new List<Vector2>();

        var screenWidthPadded = Screen.width - 2 * padding;
        var screenHeightPadded = Screen.height - 2 * padding;

        float currentSegmentX = 2 * padding;
        float currentSegmentY = padding;

        var rowSegment = screenHeightPadded / (_rows - 1);
        var colSegment = screenWidthPadded / (_columns - 1);

        for (int y = 0; y < _rows; y++)
        {
            for (int x = 0; x < _columns; x++)
            {
                points.Add(new Vector2(currentSegmentX - padding, currentSegmentY));
                currentSegmentX += colSegment;
            }
            currentSegmentX = 2 * padding;
            currentSegmentY += rowSegment;
        }

        return points;
    }
}
