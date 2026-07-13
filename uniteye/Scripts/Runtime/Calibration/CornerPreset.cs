using System.Collections.Generic;
using UnityEngine;
namespace UnitEye
{

    /// <summary>
    /// /// Calibration preset which moves to all corners
    /// </summary>
    public class CornerPreset : CalibrationPreset
    {
        private readonly bool mirrored;
        private readonly int visits;
        private readonly float dwellSeconds;
        private readonly float normalizedSafeMargin;

        public CornerPreset(float padding, int visits = 2, float dwellSeconds = 3f,
            float normalizedSafeMargin = 0.08f, bool mirrored = false) :
            base(padding)
        {
            this.mirrored = mirrored;
            this.visits = Mathf.Max(1, visits);
            this.dwellSeconds = Mathf.Max(0f, dwellSeconds);
            this.normalizedSafeMargin = Mathf.Clamp(normalizedSafeMargin, 0f, 0.45f);
        }

        public override float DwellSeconds => dwellSeconds;

        public override List<Vector2> GetPoints()
        {
            var horizontalPadding = Mathf.Max(padding, Screen.width * normalizedSafeMargin);
            var verticalPadding = Mathf.Max(padding, Screen.height * normalizedSafeMargin);
            var pass = new List<Vector2>();

            if (mirrored)
            {
                pass
                    .AddRange(new Vector2[] {
                                new Vector2(Screen.width - horizontalPadding, verticalPadding), // TR
                                new Vector2(Screen.width / 2, verticalPadding), // TM
                                new Vector2(horizontalPadding, verticalPadding), // TL
                                new Vector2(horizontalPadding, Screen.height / 2), // ML
                                new Vector2(horizontalPadding, Screen.height - verticalPadding), // LB
                                new Vector2(Screen.width / 2, Screen.height - verticalPadding), // BM
                                new Vector2(Screen.width - horizontalPadding, Screen.height - verticalPadding), // BR
                                new Vector2(Screen.width - horizontalPadding, Screen.height / 2), // MR
                });
            }
            else
            {
                pass
                    .AddRange(new Vector2[] {
                                new Vector2(horizontalPadding, verticalPadding), // TL
                                new Vector2(Screen.width / 2, verticalPadding), // TM
                                new Vector2(Screen.width - horizontalPadding, verticalPadding), // TR
                                new Vector2(Screen.width - horizontalPadding, Screen.height / 2), // RM
                                new Vector2(Screen.width - horizontalPadding, Screen.height - verticalPadding), // BR
                                new Vector2(Screen.width / 2, Screen.height - verticalPadding), // BM
                                new Vector2(horizontalPadding, Screen.height - verticalPadding), // BL
                                new Vector2(horizontalPadding, Screen.height / 2), // LM
                });
            }

            //The additional slot accounts for the closing point that returns to the starting target.
            var points = new List<Vector2>(pass.Count * visits + 1);
            for (var visit = 0; visit < visits; visit++)
                points.AddRange(pass);
            points.Add(points[0]);

            return points;
        }
    }
}
