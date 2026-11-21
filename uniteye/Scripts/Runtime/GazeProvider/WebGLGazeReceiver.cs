#if UNITY_WEBGL
using System;
using System.Globalization;
using System.Runtime.InteropServices;
using UnityEngine;

namespace UnitEye
{
    /// <summary>
    /// Bridges the browser gaze pipeline (uniteye-cv.js, via UnitEyeWebGL.jslib) to WebGLGazeProvider.
    /// Spawned automatically by WebGLGazeProvider on WebGL. Uses a pinned float[] shared buffer written
    /// directly via Emscripten HEAPF32 (zero string formatting or per-frame GC allocations), with an
    /// OnWebGaze SendMessage fallback for compatibility.
    /// </summary>
    public class WebGLGazeReceiver : MonoBehaviour
    {
        private const int SharedHeaderFloats = 6; // [seq, rawX, rawY, facePresent, blinking, featCount]
        private const int MaxSharedFeatures = 64;

        private readonly float[] _sharedBuffer = new float[SharedHeaderFloats + MaxSharedFeatures];
        private GCHandle _pinnedHandle;
        private int _lastSeq;
        private float[] _reusedFeatures = Array.Empty<float>();
        private bool _started;

#if !UNITY_EDITOR
        [DllImport("__Internal")] private static extern void UnitEyeWebGL_Start(string gazeObjectName, IntPtr sharedBuffer, int capacity);
        [DllImport("__Internal")] private static extern void UnitEyeWebGL_SetPaused(int paused);
#else
        private static void UnitEyeWebGL_Start(string gazeObjectName, IntPtr sharedBuffer, int capacity) { }
        private static void UnitEyeWebGL_SetPaused(int paused) { }
#endif

        void Start()
        {
            _pinnedHandle = GCHandle.Alloc(_sharedBuffer, GCHandleType.Pinned);
            UnitEyeWebGL_Start(gameObject.name, _pinnedHandle.AddrOfPinnedObject(), _sharedBuffer.Length);
            _started = true;
        }

        void OnEnable()
        {
            if (_started) UnitEyeWebGL_SetPaused(0);
        }

        void OnDisable()
        {
            if (_started) UnitEyeWebGL_SetPaused(1);
        }

        void OnDestroy()
        {
            if (_pinnedHandle.IsAllocated)
                _pinnedHandle.Free();
        }

        void Update()
        {
            int seq = (int)_sharedBuffer[0];
            if (seq == 0 || seq == _lastSeq) return;
            _lastSeq = seq;

            float x = _sharedBuffer[1];
            float y = _sharedBuffer[2];
            bool facePresent = _sharedBuffer[3] >= 0.5f;
            bool blinking = _sharedBuffer[4] >= 0.5f;
            int featCount = Mathf.Clamp((int)_sharedBuffer[5], 0, MaxSharedFeatures);

            if (!s_loggedFirstSample)
            {
                s_loggedFirstSample = true;
                Debug.Log($"UNITEYE_WEBGL_BRIDGE_OK first sample (HEAPF32): ({x:F1},{y:F1}) feats={featCount}");
            }

            if (facePresent && featCount > 0)
            {
                if (_reusedFeatures.Length != featCount)
                    _reusedFeatures = new float[featCount];
                Array.Copy(_sharedBuffer, SharedHeaderFloats, _reusedFeatures, 0, featCount);
                WebGLGazeProvider.ReportFeatures(_reusedFeatures);
            }
            else
            {
                WebGLGazeProvider.ReportFeatures(Array.Empty<float>());
            }

            WebGLGazeProvider.ReportGaze(x, y, facePresent, blinking);
        }

        // Called from JavaScript when the browser pipeline fails to start (camera denied, model 404, CDN down).
        public void OnWebGazeError(string message)
        {
            Debug.LogError($"UNITEYE_WEBGL_PIPELINE_ERROR: {message}");
        }

        private static bool s_loggedFirstSample;

        // Fallback called from JavaScript via SendMessage when HEAPF32 is unavailable.
        public void OnWebGaze(string csv)
        {
            if (string.IsNullOrEmpty(csv)) return;
            var p = csv.Split(',');
            if (p.Length < 4) return;

            if (!s_loggedFirstSample)
            {
                s_loggedFirstSample = true;
                Debug.Log($"UNITEYE_WEBGL_BRIDGE_OK first sample: {csv.Substring(0, Mathf.Min(48, csv.Length))}");
            }

            float x = ParseF(p[0]), y = ParseF(p[1]);
            bool facePresent = p[2] == "1";
            bool blinking = p[3] == "1";

            int featLen = p.Length - 4;
            if (facePresent && featLen > 0)
            {
                if (_reusedFeatures.Length != featLen)
                    _reusedFeatures = new float[featLen];
                for (int i = 0; i < featLen; i++) _reusedFeatures[i] = ParseF(p[4 + i]);
                WebGLGazeProvider.ReportFeatures(_reusedFeatures);
            }
            else
            {
                WebGLGazeProvider.ReportFeatures(Array.Empty<float>());
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
