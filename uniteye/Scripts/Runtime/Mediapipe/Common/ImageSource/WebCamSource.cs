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

        private WebCamTexture _webCamTexture;
        private int _deviceIndex = -1;

        /// <summary>True once the webcam has delivered a real frame.</summary>
        public bool isPrepared => _webCamTexture != null && _webCamTexture.width > 16;
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

        public Texture GetCurrentTexture() => _webCamTexture;

        private void Start() => SelectSource(-1);
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
            // Request the preferred width; the device delivers the closest resolution it supports.
            _webCamTexture = new WebCamTexture(devices[index].name, _preferableDefaultWidth, Mathf.RoundToInt(_preferableDefaultWidth * 9f / 16f));
            _webCamTexture.Play();
        }
    }
}
#endif
