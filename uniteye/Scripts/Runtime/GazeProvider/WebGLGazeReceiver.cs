#if UNITY_WEBGL
using System.Globalization;
using System.Runtime.InteropServices;
using UnityEngine;

namespace UnitEye
{
    /// <summary>
    /// Bridges the browser gaze pipeline (uniteye-cv.js, via UnitEyeWebGL.jslib) to WebGLGazeProvider.
    /// Spawned automatically by WebGLGazeProvider on WebGL. On Start it asks the JS side to begin, and
    /// receives one SendMessage per frame with "rawX,rawY,facePresent,blinking,f0,...,f11".
    /// </summary>
    public class WebGLGazeReceiver : MonoBehaviour
    {
#if !UNITY_EDITOR
        [DllImport("__Internal")] private static extern void UnitEyeWebGL_Start(string gazeObjectName);
#else
        private static void UnitEyeWebGL_Start(string gazeObjectName) { } // no-op in editor (native path runs there)
#endif

        void Start()
        {
            UnitEyeWebGL_Start(gameObject.name);
        }

        // Called from JavaScript when the browser pipeline fails to start (camera denied, model 404, CDN down).
        public void OnWebGazeError(string message)
        {
            Debug.LogError($"UNITEYE_WEBGL_PIPELINE_ERROR: {message}");
        }

        private static bool s_loggedFirstSample;

        // Called from JavaScript via SendMessage.
        public void OnWebGaze(string csv)
        {
            if (string.IsNullOrEmpty(csv)) return;
            var p = csv.Split(',');
            if (p.Length < 4) return;

            if (!s_loggedFirstSample)
            {
                s_loggedFirstSample = true;
                // Shows up in the browser console; proves the JS -> jslib -> SendMessage -> C# chain works.
                Debug.Log($"UNITEYE_WEBGL_BRIDGE_OK first sample: {csv.Substring(0, Mathf.Min(48, csv.Length))}");
            }

            float x = ParseF(p[0]), y = ParseF(p[1]);
            bool facePresent = p[2] == "1";
            bool blinking = p[3] == "1";

            if (p.Length > 4)
            {
                var feats = new float[p.Length - 4];
                for (int i = 0; i < feats.Length; i++) feats[i] = ParseF(p[4 + i]);
                WebGLGazeProvider.ReportFeatures(feats);
            }

            WebGLGazeProvider.ReportGaze(x, y, facePresent, blinking);
        }

        private static float ParseF(string s)
        {
            float.TryParse(s, NumberStyles.Float, CultureInfo.InvariantCulture, out var v);
            return v;
        }
    }
}
#endif
