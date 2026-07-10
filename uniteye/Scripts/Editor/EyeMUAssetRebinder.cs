using Unity.InferenceEngine;
using UnityEditor;
using UnityEngine;
using UnitEye;

/// <summary>
/// Rebinds the EyeMU ScriptableObject's model reference after the Barracuda->Inference Engine migration.
/// The EyeMU .onnx re-imports via the Inference Engine importer as a ModelAsset with a different fileID
/// than the old Barracuda NNModel, so the serialized reference in EyeMU.asset goes null and must be reassigned.
/// Run in batch mode with:
/// Unity.exe -batchmode -projectPath [host] -executeMethod EyeMUAssetRebinder.Rebind -logFile [log]
/// </summary>
public static class EyeMUAssetRebinder
{
    const string OnnxPath = "Packages/de.uniulm.uniteye/Resources/ONNX/EyeMUEmbedding.onnx";
    const string ResourcePath = "Packages/de.uniulm.uniteye/Resources/EyeMU.asset";

    [MenuItem("UnitEye/Rebind EyeMU Model Asset")]
    public static void Rebind()
    {
        var model = AssetDatabase.LoadAssetAtPath<ModelAsset>(OnnxPath);
        var resource = AssetDatabase.LoadAssetAtPath<EyeMUResource>(ResourcePath);

        if (model == null)
        {
            Debug.LogError($"EYEMU_REBIND_FAILED: no ModelAsset imported at {OnnxPath} (is the Inference Engine importing .onnx?)");
            if (Application.isBatchMode) EditorApplication.Exit(1);
            return;
        }
        if (resource == null)
        {
            Debug.LogError($"EYEMU_REBIND_FAILED: no EyeMUResource at {ResourcePath}");
            if (Application.isBatchMode) EditorApplication.Exit(1);
            return;
        }

        resource.modelAsset = model;
        EditorUtility.SetDirty(resource);
        AssetDatabase.SaveAssets();

        var reloaded = AssetDatabase.LoadAssetAtPath<EyeMUResource>(ResourcePath);
        if (reloaded != null && reloaded.modelAsset != null)
            Debug.Log("EYEMU_REBIND_OK");
        else
            Debug.LogError("EYEMU_REBIND_FAILED: reference did not persist");

        if (Application.isBatchMode) EditorApplication.Exit(reloaded != null && reloaded.modelAsset != null ? 0 : 1);
    }
}
