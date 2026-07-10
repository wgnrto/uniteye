using System.IO;
using UnityEditor;
using UnityEngine;

/// <summary>
/// Copies the MediaPipe model files UnitEye's native FaceMesh path needs from the homuler package's
/// PackageResources into the consuming project's Assets/StreamingAssets folder.
///
/// This is REQUIRED for a native (Windows/Mac/Linux) build: homuler's StreamingAssetsResourceManager
/// loads its .bytes models from StreamingAssets at runtime and throws FileNotFoundException on desktop
/// if they are missing (the auto-download fallback only exists for Android/WebGL). Run this once per
/// project (menu: UnitEye ▸ Install MediaPipe StreamingAssets), or via
/// -executeMethod MediaPipeAssetInstaller.Install in batch mode.
/// </summary>
public static class MediaPipeAssetInstaller
{
    const string PackageResources = "Packages/com.github.homuler.mediapipe/PackageResources/MediaPipe";

    // The FaceMesh (+iris via the attention model) path requests exactly these; face_landmark.bytes is
    // the non-attention fallback in case refineLandmarks is turned off.
    static readonly string[] RequiredAssets =
    {
        "face_detection_short_range.bytes",
        "face_landmark_with_attention.bytes",
        "face_landmark.bytes",
    };

    [MenuItem("UnitEye/Install MediaPipe StreamingAssets")]
    public static void Install()
    {
        var srcDir = Path.GetFullPath(PackageResources);
        if (!Directory.Exists(srcDir))
        {
            Debug.LogError($"MEDIAPIPE_INSTALL_FAILED: package resources not found at {PackageResources}. Is com.github.homuler.mediapipe installed?");
            if (Application.isBatchMode) EditorApplication.Exit(1);
            return;
        }

        var dstDir = Path.Combine(Application.streamingAssetsPath);
        Directory.CreateDirectory(dstDir);

        int copied = 0, missing = 0;
        foreach (var asset in RequiredAssets)
        {
            var src = Path.Combine(srcDir, asset);
            if (!File.Exists(src)) { Debug.LogWarning($"MediaPipe asset not in package: {asset}"); missing++; continue; }
            File.Copy(src, Path.Combine(dstDir, asset), overwrite: true);
            copied++;
        }

        AssetDatabase.Refresh();
        Debug.Log($"MEDIAPIPE_INSTALL_OK: copied {copied} MediaPipe model file(s) to {dstDir}" + (missing > 0 ? $" ({missing} missing)" : ""));
        if (Application.isBatchMode) EditorApplication.Exit(missing > 0 ? 1 : 0);
    }
}
