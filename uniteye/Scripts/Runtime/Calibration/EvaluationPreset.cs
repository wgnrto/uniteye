using System.Collections.Generic;
using UnityEngine;
namespace UnitEye
{

    /// <summary>
    /// Calibration preset consisting of a grid of points. Used for evaluation only currently.
    /// </summary>
    public class EvaluationPreset : CalibrationPreset
    {
        private int _rows, _columns;
        private readonly float _normalizedSafeMargin;

        public EvaluationPreset(float padding, int rows, int columns, float normalizedSafeMargin = 0f) :
            base(padding)
        {
            _rows = Mathf.Max(2, rows);
            _columns = Mathf.Max(2, columns);
            _normalizedSafeMargin = Mathf.Clamp(normalizedSafeMargin, 0f, 0.45f);
        }

        public override List<Vector2> GetPoints()
        {
            List<Vector2> points = new List<Vector2>();

            var horizontalPadding = Mathf.Max(padding, Screen.width * _normalizedSafeMargin);
            var verticalPadding = Mathf.Max(padding, Screen.height * _normalizedSafeMargin);
            var screenWidthPadded = Screen.width - 2 * horizontalPadding;
            var screenHeightPadded = Screen.height - 2 * verticalPadding;

            float currentSegmentX = 2 * horizontalPadding;
            float currentSegmentY = verticalPadding;

            var rowSegment = screenHeightPadded / (_rows - 1);
            var colSegment = screenWidthPadded / (_columns - 1);

            for (int y = 0; y < _rows; y++)
            {
                for (int x = 0; x < _columns; x++)
                {
                    points.Add(new Vector2(currentSegmentX - horizontalPadding, currentSegmentY));
                    currentSegmentX += colSegment;
                }
                currentSegmentX = 2 * horizontalPadding;
                currentSegmentY += rowSegment;
            }

            return points;
        }
    }
}
