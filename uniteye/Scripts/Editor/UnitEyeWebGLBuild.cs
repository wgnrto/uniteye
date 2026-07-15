using UnityEditor;
using UnityEditor.Build.Reporting;
using UnityEngine;

/// <summary>
/// Batch WebGL build of the UnitEye gaze scene, for CI / headless verification:
/// Unity.exe -batchmode -projectPath [host] -executeMethod UnitEyeWebGLBuild.Build -logFile [log]
/// Output goes to [project]/BuildWebGL with compression disabled so any static file server can serve it.
/// Logs UNITEYE_WEBGL_BUILD_OK / UNITEYE_WEBGL_BUILD_FAILED and exits 0/1.
/// </summary>
public static class UnitEyeWebGLBuild
{
    [MenuItem("UnitEye/Build WebGL (gaze scene)")]
    public static void Build()
    {
        var scenes = new[] { "Packages/de.uniulm.uniteye/Scenes/HomulerGazeScene.unity" };

        // Uncompressed output so a plain static file server works without Content-Encoding headers.
        PlayerSettings.WebGL.compressionFormat = WebGLCompressionFormat.Disabled;
        PlayerSettings.WebGL.decompressionFallback = false;

        var options = new BuildPlayerOptions
        {
            scenes = scenes,
            locationPathName = "BuildWebGL",
            target = BuildTarget.WebGL,
            options = BuildOptions.Development, // faster iteration + readable errors
        };

        var report = BuildPipeline.BuildPlayer(options);
        var summary = report.summary;

        if (summary.result == BuildResult.Succeeded)
        {
            Debug.Log($"UNITEYE_WEBGL_BUILD_OK size={summary.totalSize} bytes, time={summary.totalTime}, warnings={summary.totalWarnings}");
            if (Application.isBatchMode) EditorApplication.Exit(0);
        }
        else
        {
            Debug.LogError($"UNITEYE_WEBGL_BUILD_FAILED result={summary.result}, errors={summary.totalErrors}");
            // Surface the first errors so the log is greppable
            int logged = 0;
            foreach (var step in report.steps)
            {
                foreach (var msg in step.messages)
                {
                    if (msg.type == LogType.Error || msg.type == LogType.Exception)
                    {
                        Debug.LogError($"UNITEYE_WEBGL_BUILD_ERR [{step.name}] {msg.content}");
                        if (++logged >= 25) break;
                    }
                }
                if (logged >= 25) break;
            }
            if (Application.isBatchMode) EditorApplication.Exit(1);
        }
    }
}
