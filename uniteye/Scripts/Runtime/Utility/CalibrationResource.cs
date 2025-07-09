using UnityEngine;

[CreateAssetMenu(fileName = "CalibrationDefaultFiles",
                 menuName = "ScriptableObjects/UnitEye/Calibration Default File Set")]
public class CalibrationResource : ScriptableObject
{
    public TextAsset mlpAsset;
    public TextAsset regXAsset;
    public TextAsset regYAsset;
}