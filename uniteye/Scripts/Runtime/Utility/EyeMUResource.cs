using Unity.InferenceEngine;
using UnityEngine;
namespace UnitEye
{

    [CreateAssetMenu(fileName = "EyeMU",
                     menuName = "ScriptableObjects/UnitEye/EyeMU Resource Set")]
    public class EyeMUResource : ScriptableObject
    {
        // Migrated from Barracuda NNModel to Unity Inference Engine ModelAsset.
        // The serialized reference in EyeMU.asset must be rebound after the .onnx re-imports
        // via the Inference Engine importer (the fileID changes); see EyeMUAssetRebinder.
        public ModelAsset modelAsset;
        public ComputeShader preprocessCompute;
    }
}
