using System.Collections.Generic;
using UnityEngine;
namespace UnitEye
{

    /// <summary>
    /// Calibration preset that DWELLS at interior screen positions: the centre plus the four points at
    /// 25%/75% of each axis. Its purpose is the thin-plate-spline warp: the warp is fitted on dwell
    /// anchors, and with only CornerPreset dwelling (4 corners + 4 edge midpoints) every anchor sat on the
    /// screen BOUNDARY — the spline's interior behaviour was pure affine extrapolation, so it could fix a
    /// bad corner but never an interior residual bump, which is where a game's AOIs actually live. Five
    /// short dwells (~1.5s each, ~8s total) give the warp interior support; the existing keep-only-if-
    /// holdout-improves gate makes the extra anchors strictly-no-worse.
    /// </summary>
    public class InteriorPreset : CalibrationPreset
    {
        private readonly float dwellSeconds;

        public InteriorPreset(float padding, float dwellSeconds = 1.5f) : base(padding)
        {
            this.dwellSeconds = Mathf.Max(0f, dwellSeconds);
        }

        public override float DwellSeconds => dwellSeconds;

        public override List<Vector2> GetPoints()
        {
            float w = Screen.width, h = Screen.height;
            //A leading duplicate primes a dwell on the first target (the calibration only dwells at
            //points[1..n-2], same trick as HeadRotationPreset).
            return new List<Vector2>
            {
                new Vector2(w * 0.5f, h * 0.5f),
                new Vector2(w * 0.5f, h * 0.5f),   // C
                new Vector2(w * 0.25f, h * 0.25f), // upper-left quadrant
                new Vector2(w * 0.75f, h * 0.25f), // upper-right quadrant
                new Vector2(w * 0.75f, h * 0.75f), // lower-right quadrant
                new Vector2(w * 0.25f, h * 0.75f), // lower-left quadrant
                new Vector2(w * 0.5f, h * 0.5f),
            };
        }
    }
}
