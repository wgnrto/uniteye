// Excluded from WebGL player builds: depends on the native MediaPipe plugin.
#if !UNITY_WEBGL || UNITY_EDITOR
using UnityEngine;

namespace Mediapipe.Unity
{
    /// <summary>
    /// Minimal webcam source for the MediaPipe 0.16.3 Task-API path. Replaces the old Solution-era
    /// ImageSource/WebCamSource stack (which depended on the removed Solution API): it just opens a
    /// WebCamTexture, exposes the current frame + device selection, and lets FaceMeshSolution read a frame
    /// each Update to feed the FaceLandmarker. Kept in the Mediapipe.Unity namespace + same file (GUID) so
    /// the prefab's WebCamSource component and the UnitEye consumers keep working.
    /// </summary>
    public class WebCamSource : MonoBehaviour
    {
        public readonly struct Resolution
        {
            public readonly int width;
            public readonly int height;
            public Resolution(int width, int height) { this.width = width; this.height = height; }
        }

        [Tooltip("The available webcam whose width is closest to this value is chosen by default. Lower = faster.")]
        [SerializeField] private int _preferableDefaultWidth = 1280;
        [Tooltip("Preferred webcam device name; empty picks the closest to Preferable Default Width.")]
        [SerializeField] private string _name = "";
        //At ~60cm, 1° of gaze rotation moves the iris ~0.3px at 720p-class capture — the per-frame gaze
        //signal is SUB-PIXEL and landmark jitter is the pipeline's binding accuracy constraint. Higher
        //frame rates also double the samples per fixation (~sqrt(2) precision after aggregation) and
        //shorten exposure (less motion blur on the iris). The driver delivers the closest supported rate.
        [Tooltip("Requested camera frame rate. 60fps = 2x samples per fixation and shorter exposure (sharper iris); the driver falls back to the closest supported rate.")]
        [SerializeField] private int _requestedFps = 60;

        private WebCamTexture _webCamTexture;
        private int _deviceIndex = -1;

        /// <summary>True once the webcam has delivered a real frame.</summary>
        public bool isPrepared => _webCamTexture != null && _webCamTexture.width > 16;
        /// <summary>True only on frames where the camera delivered a NEW image. Lets consumers skip
        /// reprocessing the same frame when the display refresh outruns the camera rate.</summary>
        public bool didUpdateThisFrame => _webCamTexture != null && _webCamTexture.didUpdateThisFrame;
        public string sourceName => _webCamTexture != null ? _webCamTexture.deviceName : "";
        public bool isVerticallyFlipped => _webCamTexture != null && _webCamTexture.videoVerticallyMirrored;
        public bool isFrontFacing
        {
            get
            {
                var devices = WebCamTexture.devices;
                return _deviceIndex >= 0 && _deviceIndex < devices.Length && devices[_deviceIndex].isFrontFacing;
            }
        }
        public int rotation => _webCamTexture != null ? _webCamTexture.videoRotationAngle : 0;
        public int textureWidth => _webCamTexture != null ? _webCamTexture.width : 0;
        public int textureHeight => _webCamTexture != null ? _webCamTexture.height : 0;
        public Resolution resolution => new Resolution(textureWidth, textureHeight);

        /// <summary>
        /// The frame rate this source asked the device for. Exposed so consumers can compare it with
        /// the rate actually delivered: gaze updates are gated on new camera frames, so a camera
        /// running well below its requested rate silently caps the entire pipeline.
        /// </summary>
        public int requestedFps => _requestedFps;

        public Texture GetCurrentTexture() => _webCamTexture;

        // On mobile, the camera permission must be granted BEFORE WebCamTexture.Play(); otherwise the
        // texture silently stays at the 16x16 placeholder and the pipeline idles with no error (the old
        // Solution-era WebCamSource had this flow; the migration dropped it). Desktop needs no prompt.
        private System.Collections.IEnumerator Start()
        {
#if UNITY_ANDROID && !UNITY_EDITOR
            if (!UnityEngine.Android.Permission.HasUserAuthorizedPermission(UnityEngine.Android.Permission.Camera))
            {
                bool answered = false;
                var callbacks = new UnityEngine.Android.PermissionCallbacks();
                callbacks.PermissionGranted += _ => answered = true;
                callbacks.PermissionDenied += _ => answered = true;
                UnityEngine.Android.Permission.RequestUserPermission(UnityEngine.Android.Permission.Camera, callbacks);
                yield return new WaitUntil(() => answered);
                if (!UnityEngine.Android.Permission.HasUserAuthorizedPermission(UnityEngine.Android.Permission.Camera))
                {
                    Debug.LogWarning("WebCamSource: camera permission denied; no webcam input available.");
                    yield break;
                }
            }
#elif UNITY_IOS && !UNITY_EDITOR
            yield return Application.RequestUserAuthorization(UserAuthorization.WebCam);
            if (!Application.HasUserAuthorization(UserAuthorization.WebCam))
            {
                Debug.LogWarning("WebCamSource: camera permission denied; no webcam input available.");
                yield break;
            }
#endif
            SelectSource(-1);
            yield break;
        }

        private void OnDestroy() => StopCamera();

        private void StopCamera()
        {
            if (_webCamTexture == null) return;
            _webCamTexture.Stop();
            Destroy(_webCamTexture);
            _webCamTexture = null;
        }

        /// <summary>Index of the current device in WebCamTexture.devices, or -1 if none.</summary>
        public int GetCameraIndex() => _deviceIndex;

        /// <summary>
        /// Opens the webcam device at the given index (wrapped into range). index &lt; 0 picks the device
        /// whose reported name matches _name, else the one closest to _preferableDefaultWidth (or the first).
        /// </summary>
        public void SelectSource(int index)
        {
            var devices = WebCamTexture.devices;
            if (devices.Length == 0)
            {
                Debug.LogWarning("WebCamSource: no webcam devices found.");
                return;
            }

            if (index < 0)
            {
                index = 0;
                if (!string.IsNullOrEmpty(_name))
                {
                    for (int i = 0; i < devices.Length; i++)
                        if (devices[i].name == _name) { index = i; break; }
                }
            }
            index = ((index % devices.Length) + devices.Length) % devices.Length;

            StopCamera();
            _deviceIndex = index;
            _name = devices[index].name;
            // Request the preferred width + frame rate; the device delivers the closest it supports.
            _webCamTexture = new WebCamTexture(devices[index].name, _preferableDefaultWidth,
                Mathf.RoundToInt(_preferableDefaultWidth * 9f / 16f), Mathf.Max(1, _requestedFps));
            _webCamTexture.Play();
        }
    }
}
#endif
