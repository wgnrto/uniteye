using System.Collections.Generic;
using UnityEngine;
namespace UnitEye
{

    /// <summary>
    /// Base class for calibrations pattern used during calibration.
    /// </summary>
    //[System.Serializable]
    public abstract class CalibrationPreset
    {
        protected float padding;

        /// <summary>
        /// Base constructor for the CalibrationPreset
        /// </summary>
        /// <param name="padding">Distance to the border of the screen</param>
        public CalibrationPreset(float padding)
        {
            this.padding = padding;
        }

        /// <summary>
        /// Returns a list of points that will be moved between
        /// </summary>
        public abstract List<Vector2> GetPoints();

        /// <summary>
        /// Whether the calibration dot should DWELL (pause + keep capturing) at each waypoint of this
        /// preset. True for discrete presets whose waypoints are meaningful fixation targets (corners,
        /// grids) — sustained fixation there denoises the samples and, crucially, gives the fit real
        /// leverage at the screen extremes. False for the continuous "wavy" sweeps, whose ~150 waypoints
        /// are just a smooth path (dwelling at each would take minutes and pile samples mid-screen).
        /// </summary>
        public virtual bool StopAtWaypoints => true;

        /// <summary>
        /// Seconds to hold a fixation target. Continuous presets override StopAtWaypoints instead.
        /// </summary>
        public virtual float DwellSeconds => 2f;

        /// <summary>
        /// Whether this preset is the HEAD-MOVEMENT stage: the dot dwells while the user is prompted to
        /// slowly rotate their head, so the SAME screen target is captured across a range of head poses.
        /// Head yaw/pitch/roll are model features but are otherwise captured at a single still pose (near-
        /// zero variance, so the fit ignores them); this stage gives them leverage so the calibration can
        /// compensate for head movement during use. The calibration shows the rotate-your-head prompt for
        /// these presets and exempts their (intentionally high-variance) samples from the corner stability
        /// rejection. False for ordinary gaze presets.
        /// </summary>
        public virtual bool IsHeadMovement => false;
    }
}
