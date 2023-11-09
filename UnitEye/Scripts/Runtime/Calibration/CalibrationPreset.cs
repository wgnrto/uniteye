using System.Collections.Generic;
using UnityEngine;

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
}
