using System.Collections.Generic;
using UnityEngine;
namespace UnitEye
{

    /// <summary>
    /// Calibration preset for HEAD-POSE coverage. The dot dwells at the screen centre and the four corners
    /// while the user is prompted to slowly rotate their head (left/right/up/down). Head yaw/pitch/roll are
    /// model features, but a normal "sit still and follow the dot" calibration captures them at essentially
    /// one pose, so they standardize to near-zero variance and the fit ignores them — then gaze drifts as
    /// soon as the head moves during real use. Capturing the SAME screen target across a range of head poses
    /// gives those features real leverage, so the calibration learns to compensate for head movement.
    /// </summary>
    public class HeadRotationPreset : CalibrationPreset
    {
        private readonly float dwellSeconds;
        private readonly float normalizedSafeMargin;

        public HeadRotationPreset(float padding, float dwellSeconds = 5f, float normalizedSafeMargin = 0.08f)
            : base(padding)
        {
            //A head roll through yaw + pitch needs a few seconds; keep a sensible floor.
            this.dwellSeconds = Mathf.Max(2f, dwellSeconds);
            this.normalizedSafeMargin = Mathf.Clamp(normalizedSafeMargin, 0f, 0.45f);
        }

        //Dwell at each fixation target; the user rotates their head during the dwell.
        public override bool StopAtWaypoints => true;
        public override float DwellSeconds => dwellSeconds;
        //Marks this as the head-movement stage (see CalibrationPreset.IsHeadMovement).
        public override bool IsHeadMovement => true;

        public override List<Vector2> GetPoints()
        {
            var horizontalPadding = Mathf.Max(padding, Screen.width * normalizedSafeMargin);
            var verticalPadding = Mathf.Max(padding, Screen.height * normalizedSafeMargin);
            var centre = new Vector2(Screen.width * 0.5f, Screen.height * 0.5f);

            //HomulerGazeCalibration dwells at points[1 .. n-2] (the first and last waypoints are the
            //approach and the round-closing return, neither of which dwells). Listing the centre twice at
            //the start therefore makes the stage OPEN with a centre dwell — the most natural place to rotate
            //the head — followed by the four corners, before returning to centre to close the round.
            return new List<Vector2>
            {
                centre,                                                              // approach (no dwell)
                centre,                                                              // centre dwell
                new Vector2(horizontalPadding, verticalPadding),                     // TL dwell
                new Vector2(Screen.width - horizontalPadding, verticalPadding),      // TR dwell
                new Vector2(Screen.width - horizontalPadding, Screen.height - verticalPadding), // BR dwell
                new Vector2(horizontalPadding, Screen.height - verticalPadding),     // BL dwell
                centre,                                                              // close round (no dwell)
            };
        }
    }
}
