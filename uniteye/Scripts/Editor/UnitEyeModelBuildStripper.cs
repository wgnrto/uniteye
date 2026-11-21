using System;
using System.IO;
using UnityEditor;
using UnityEditor.Build;
using UnityEditor.Build.Reporting;
using UnityEngine;

namespace UnitEye.EditorTools
{
    /// <summary>
    /// Excludes unused Resources/ONNX weights during WebGL builds (saving ~101 MB: WebGL runs onnxruntime-web
    /// in the browser and never loads Unity Inference Engine Resources/ONNX assets), and restores them cleanly
    /// in OnPostprocessBuild and on Editor domain reload if a build was aborted.
    /// </summary>
    [InitializeOnLoad]
    public class UnitEyeModelBuildStripper : IPreprocessBuildWithReport, IPostprocessBuildWithReport
    {
        private const string PackageRoot = "Packages/de.uniulm.uniteye";
        private const string StripDesktopPrefKey = "UnitEye.StripDirectionBackbonesOnDesktop";

        public int callbackOrder => 0;

        static UnitEyeModelBuildStripper()
        {
            EditorApplication.delayCall += RestoreIfNeeded;
        }

        [MenuItem("UnitEye/Build Options/Toggle Strip Direction ONNX Models on Desktop")]
        private static void ToggleStripDesktop()
        {
            bool next = !EditorPrefs.GetBool(StripDesktopPrefKey, false);
            EditorPrefs.SetBool(StripDesktopPrefKey, next);
            Debug.Log($"UnitEye: Strip Direction ONNX Models on Desktop builds = {next}");
        }

        public void OnPreprocessBuild(BuildReport report)
        {
            if (report == null) return;
            bool isWebGL = report.summary.platform == BuildTarget.WebGL;
            bool stripDirectionOnly = !isWebGL && EditorPrefs.GetBool(StripDesktopPrefKey, false);
            if (!isWebGL && !stripDirectionOnly)
                return;

            try
            {
                string pkgFullPath = Path.GetFullPath(PackageRoot);
                if (!Directory.Exists(pkgFullPath))
                    return;

                string stashRoot = Path.Combine(pkgFullPath, "ResourcesBuildStash~");
                if (isWebGL)
                {
                    StashDirectory(
                        Path.Combine(pkgFullPath, "Resources", "ONNX"),
                        Path.Combine(stashRoot, "ONNX"));
                }
                else if (stripDirectionOnly)
                {
                    StashDirectory(
                        Path.Combine(pkgFullPath, "Resources", "ONNX", "GazeEstimation"),
                        Path.Combine(stashRoot, "GazeEstimation"));
                }
                AssetDatabase.Refresh(ImportAssetOptions.ForceSynchronousImport);
            }
            catch (Exception e)
            {
                Debug.LogWarning($"UnitEyeModelBuildStripper: could not stash optional ONNX weights before build ({e.Message}).");
            }
        }

        public void OnPostprocessBuild(BuildReport report)
        {
            RestoreIfNeeded();
        }

        private static void StashDirectory(string srcDir, string dstDir)
        {
            if (!Directory.Exists(srcDir))
                return;
            var parent = Path.GetDirectoryName(dstDir);
            if (!string.IsNullOrEmpty(parent))
                Directory.CreateDirectory(parent);
            if (Directory.Exists(dstDir))
                Directory.Delete(dstDir, recursive: true);
            Directory.Move(srcDir, dstDir);

            string srcMeta = srcDir + ".meta";
            string dstMeta = dstDir + ".meta";
            if (File.Exists(srcMeta))
            {
                if (File.Exists(dstMeta)) File.Delete(dstMeta);
                File.Move(srcMeta, dstMeta);
            }
        }

        private static void RestoreDirectory(string stashDir, string targetDir)
        {
            if (!Directory.Exists(stashDir))
                return;
            var parent = Path.GetDirectoryName(targetDir);
            if (!string.IsNullOrEmpty(parent))
                Directory.CreateDirectory(parent);
            if (!Directory.Exists(targetDir))
                Directory.Move(stashDir, targetDir);

            string stashMeta = stashDir + ".meta";
            string targetMeta = targetDir + ".meta";
            if (File.Exists(stashMeta) && !File.Exists(targetMeta))
                File.Move(stashMeta, targetMeta);
        }

        public static void RestoreIfNeeded()
        {
            try
            {
                string pkgFullPath = Path.GetFullPath(PackageRoot);
                if (!Directory.Exists(pkgFullPath))
                    return;

                string stashRoot = Path.Combine(pkgFullPath, "ResourcesBuildStash~");
                if (!Directory.Exists(stashRoot))
                    return;

                RestoreDirectory(
                    Path.Combine(stashRoot, "ONNX"),
                    Path.Combine(pkgFullPath, "Resources", "ONNX"));
                RestoreDirectory(
                    Path.Combine(stashRoot, "GazeEstimation"),
                    Path.Combine(pkgFullPath, "Resources", "ONNX", "GazeEstimation"));

                if (Directory.GetFileSystemEntries(stashRoot).Length == 0)
                    Directory.Delete(stashRoot);

                AssetDatabase.Refresh(ImportAssetOptions.ForceSynchronousImport);
            }
            catch (Exception e)
            {
                Debug.LogWarning($"UnitEyeModelBuildStripper: could not restore stashed ONNX weights ({e.Message}).");
            }
        }
    }
}
