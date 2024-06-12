using Unity.Barracuda;
using UnityEngine;

[CreateAssetMenu(fileName = "EyeMU",
                 menuName = "ScriptableObjects/UnitEye/EyeMU Resource Set")]
public class EyeMUResource : ScriptableObject
{
    public NNModel modelAsset;
    public ComputeShader preprocessCompute;
}
