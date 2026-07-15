using UnityEditor;
using UnityEditor.SceneManagement;
using UnityEngine;

/// <summary>
/// One-time migration helper: strips MonoBehaviour components whose script is missing from the UnitEye
/// scenes/prefabs. After the MediaPipe 0.12 -> 0.16.3 (Task API) migration the old Solution-era components
/// (FaceMeshGraph / Screen / TextureFramePool / the old annotation controllers) were deleted, leaving
/// "missing script" placeholders on the Mediapipe GameObject and the annotation prefabs. Run once:
/// menu 'UnitEye ▸ Cleanup Missing Scripts', or -executeMethod MigrationCleanup.Run.
/// </summary>
public static class MigrationCleanup
{
    static readonly string[] Scenes =
    {
        "Packages/de.uniulm.uniteye/Scenes/HomulerGazeScene.unity",
        "Packages/de.uniulm.uniteye/Scenes/HomulerGazeCalibration.unity",
    };

    static readonly string[] Prefabs =
    {
        "Packages/de.uniulm.uniteye/Prefabs/UnitEyeUsingHomulerMediapipe.prefab",
        "Packages/de.uniulm.uniteye/Prefabs/MediapipeAnnotation.prefab",
    };

    [MenuItem("UnitEye/Cleanup Missing Scripts")]
    public static void Run()
    {
        int total = 0;

        foreach (var path in Prefabs)
        {
            var root = PrefabUtility.LoadPrefabContents(path);
            if (root == null) { Debug.LogWarning($"CLEANUP: prefab not found: {path}"); continue; }
            int removed = StripHierarchy(root);
            if (removed > 0) PrefabUtility.SaveAsPrefabAsset(root, path);
            PrefabUtility.UnloadPrefabContents(root);
            total += removed;
            Debug.Log($"CLEANUP: {path} -> removed {removed} missing-script component(s)");
        }

        foreach (var path in Scenes)
        {
            var scene = EditorSceneManager.OpenScene(path, OpenSceneMode.Single);
            int removed = 0;
            foreach (var go in scene.GetRootGameObjects())
                removed += StripHierarchy(go);
            if (removed > 0)
            {
                EditorSceneManager.MarkSceneDirty(scene);
                EditorSceneManager.SaveScene(scene);
            }
            total += removed;
            Debug.Log($"CLEANUP: {path} -> removed {removed} missing-script component(s)");
        }

        AssetDatabase.SaveAssets();
        Debug.Log($"CLEANUP_DONE: removed {total} missing-script component(s) total");
        if (Application.isBatchMode) EditorApplication.Exit(0);
    }

    private static int StripHierarchy(GameObject root)
    {
        int removed = 0;
        foreach (var t in root.GetComponentsInChildren<Transform>(true))
            removed += GameObjectUtility.RemoveMonoBehavioursWithMissingScript(t.gameObject);
        return removed;
    }
}
