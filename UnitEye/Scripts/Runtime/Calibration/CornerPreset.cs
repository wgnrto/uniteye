using System.Collections.Generic;
using UnityEngine;

/// <summary>
/// /// Calibration preset which moves to all corners
/// </summary>
public class CornerPreset : CalibrationPreset
{
    private bool mirrored;

    public CornerPreset(float padding, bool mirrored = false) :
        base(padding)
    {
        this.mirrored = mirrored;
    }

    public override List<Vector2> GetPoints()
    {
        List<Vector2> points = new List<Vector2>();

        if (mirrored)
        {
            points
                .AddRange(new Vector2[] {
                            new Vector2(Screen.width - padding, padding), // TR
                            new Vector2(Screen.width / 2, padding), // TM
                            new Vector2(padding, padding), // TL
                            new Vector2(padding, Screen.height / 2), // ML
                            new Vector2(padding, Screen.height - padding), // LB 
                            new Vector2(Screen.width / 2, Screen.height - padding), // BM
                            new Vector2(Screen.width - padding, Screen.height - padding), // BR
                            new Vector2(Screen.width - padding, Screen.height / 2), // MR
            });
        }
        else
        {
            points
                .AddRange(new Vector2[] {
                            new Vector2(padding, padding), // TL
                            new Vector2(Screen.width / 2, padding), // TM
                            new Vector2(Screen.width - padding, padding), // TR
                            new Vector2(Screen.width - padding, Screen.height / 2), // RM
                            new Vector2(Screen.width - padding, Screen.height - padding), // BR
                            new Vector2(Screen.width / 2, Screen.height - padding), // BM
                            new Vector2(padding, Screen.height - padding), // BL
                            new Vector2(padding, Screen.height / 2), // LM
            });
        }

        points.Add(points[0]);

        return points;
    }
}
