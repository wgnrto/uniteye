using UnityEngine;
namespace UnitEye
{

    /// <summary>
    /// This class provides access to the Webcam. 
    /// It also allows downsizing the webcam texture and optionally renders the webcam output to an Unity UI Image.
    /// </summary>
    public class WebCamInput : MonoBehaviour
    {

        private int _webcamIndex;
        private Texture _rawImageBackup;
        private Texture _textureBackup;
        private bool _backupped;
        private bool _resumePlaybackOnEnable;

        //Default to 1080p, can be changed if neccessary
        public static Vector2 DefaultResolution = new Vector2(1920, 1080);

        [SerializeField] public string webCamName;
        [SerializeField] public Vector2 webCamResolution = DefaultResolution;
        [SerializeField] Texture staticInput;
        [SerializeField] UnityEngine.UI.RawImage rawImage;
        [SerializeField] int targetFramerate = 999;

        // Provide input image Texture.
        public Texture inputImageTexture {
            get {
                if (staticInput != null) return staticInput;
                return inputRT;
            }
        }

        public WebCamTexture webCamTexture;
        public RenderTexture inputRT;

        public bool mirrorImage;

        void Start()
        {
            //Limit Application.targetFrameRate to avoid a webcam freezing bug in Unity
            Application.targetFrameRate = targetFramerate;

            if (rawImage != null)
            {
                _textureBackup = rawImage.texture;
                _rawImageBackup = _textureBackup;
            }

            //Default to first device and DefaultResolution if webCamName is empty string or not in list of available webcams
            if (webCamName == "" || !WebCamInput.CheckValidWebCam(webCamName))
            {
                if (WebCamTexture.devices.Length > 0)
                {
                    webCamName = WebCamTexture.devices[0].name;
                    _webcamIndex = 0;
                }
                webCamResolution = DefaultResolution;
            }
            else
            {
                _webcamIndex = GetCameraIndex();
            }

            //If no staticInput is set use webcam
            if (staticInput == null)
            {
                webCamTexture = new WebCamTexture(webCamName, (int)webCamResolution.x, (int)webCamResolution.y);
                webCamTexture.Play();
                //Size the RT from the requested resolution: right after Play() webCamTexture.width often
                //still reports the 16x16 placeholder, which would give a tiny inputRT. Only the webcam path
                //needs an inputRT (inputImageTexture returns staticInput directly on the static path).
                inputRT = new RenderTexture((int)webCamResolution.x, (int)webCamResolution.y, 0);
            }
            //Else use staticInput (no webcam, no inputRT — previously this NRE'd dereferencing webCamTexture)
            else
            {
                if (rawImage != null) rawImage.texture = staticInput;
            }
        }

        void Update()
        {
            //Return if staticInput or no webcam update in this frame
            if (staticInput != null) return;
            if (!webCamTexture.didUpdateThisFrame) return;

            //Lazily (re)create inputRT once the camera's REAL resolution is known. The LoadCamera paths
            //defer sizing to here: right after Play() a WebCamTexture still reports the 16x16 placeholder
            //(the Start() path already worked around this — see its comment), so sizing there produced a
            //16x16 RT the whole pipeline then silently ran on. didUpdateThisFrame guarantees real dims.
            if (inputRT == null)
                inputRT = new RenderTexture(webCamTexture.width, webCamTexture.height, 0);

            var aspect1 = (float)webCamTexture.width / webCamTexture.height;
            var aspect2 = (float)inputRT.width / inputRT.height;
            var aspectGap = aspect2 / aspect1;

            //Scale and offset webCamTexture
            var vMirrored = webCamTexture.videoVerticallyMirrored;
            var scale = new Vector2(aspectGap, vMirrored ? -1 : 1);
            var offset = new Vector2((1 - aspectGap) / 2, vMirrored ? 1 : 0);

            //Copy to inputRT
            Graphics.Blit(webCamTexture, inputRT, scale, offset);
            //Draw webcam on rawImage if set
            if (rawImage != null && !_backupped) rawImage.texture = inputRT;
            //Mirror rawImage if desired
            if (rawImage != null && !_backupped) rawImage.transform.localScale = new Vector3((mirrorImage ? -1f : 1f), 1f, 1f);
        }

        void OnDisable()
        {
            if (staticInput != null) return;

            if (webCamTexture != null)
            {
                _resumePlaybackOnEnable = webCamTexture.isPlaying;
                if (webCamTexture.isPlaying)
                {
                    webCamTexture.Stop();
                }
            }
            else
            {
                _resumePlaybackOnEnable = false;
            }
        }

        void OnEnable()
        {
            if (staticInput != null) return;

            if (_resumePlaybackOnEnable && webCamTexture != null && !webCamTexture.isPlaying)
            {
                webCamTexture.Play();
            }

            _resumePlaybackOnEnable = false;
        }

        public void Stop()
        {
            if (webCamTexture != null)
                webCamTexture.Stop();
        }

        /// <summary>
        /// Stops and destroys the current WebCamTexture and inputRT so a camera switch does not leak them.
        /// </summary>
        private void ReleaseCurrentCamera()
        {
            if (webCamTexture != null)
            {
                webCamTexture.Stop();
                Destroy(webCamTexture);
                webCamTexture = null;
            }
            if (inputRT != null)
            {
                inputRT.Release();
                Destroy(inputRT);
                inputRT = null;
            }
        }

        void OnDestroy()
        {
            Stop();
            if (webCamTexture != null) Destroy(webCamTexture);
            if (inputRT != null) Destroy(inputRT);
        }


        /// <summary>
        /// Check if webCamName is in WebCamTexture.devices.
        /// </summary>
        /// <param name="webCamName">name to check</param>
        /// <returns>true if webCamName is valid, false if webCamName does not exist</returns>
        public static bool CheckValidWebCam(string webCamName)
        {
            foreach (var device in WebCamTexture.devices)
            {
                if (device.name == webCamName)
                    return true;
            }

            return false;
        }

        /// <summary>
        /// Returns first index in the webcam devices list that matches the currently used webcam.
        /// </summary>
        /// <returns>int Camera index in WebCamTexture.devices, -1 if no cameras are plugged in</returns>
        public int GetCameraIndex()
        {
            int index = -1;
            var webCamDevices = WebCamTexture.devices;

            for (int i = 0; i < webCamDevices.Length; i++)
            {
                if (webCamDevices[i].name == webCamName)
                {
                    index = i;
                    break;
                }
            }

            return index;
        }

        /// <summary>
        /// Loads the first camera in WebCamTexture.devices that matches frontFacing boolean. Only works on Android devices.
        /// If no resolution is provided, the WebCamTexture default for the camera will be used.
        /// </summary>
        /// <param name="frontFacing">true to load a front facing camera, false to load a back facing camera</param>
        /// <param name="requestedWidth">requested width of the camera in pixels</param>
        /// <param name="requestedHeight">requested height of the camera in pixels</param>
        /// <returns>true if loaded successfully, false if no matching camera was found</returns>
        public bool LoadCamera(bool frontFacing = true, int requestedWidth = -1, int requestedHeight = -1)
        {
            var found = false;

            var webCamDevices = WebCamTexture.devices;

            for (int i = 0; i < webCamDevices.Length; i++)
            {
                if (webCamDevices[i].isFrontFacing == frontFacing)
                {
                    //Release the previous webcam + RT before replacing them, otherwise every camera
                    //switch leaks the old WebCamTexture and RenderTexture (both native resources).
                    ReleaseCurrentCamera();

                    //If both values are not -1 use them, otherwise use WebCamTexture default
                    if (requestedWidth > 0 && requestedHeight > 0)
                        webCamTexture = new WebCamTexture(webCamDevices[i].name, requestedWidth, requestedHeight);
                    else
                        webCamTexture = new WebCamTexture(webCamDevices[i].name);
                    webCamTexture.Play();
                    //inputRT sizing is deferred to Update(): right after Play() webCamTexture.width still
                    //reports the 16x16 placeholder, which used to bake a 16x16 inputRT. With a requested
                    //resolution we can size immediately (mirrors Start()).
                    inputRT = requestedWidth > 0 && requestedHeight > 0
                        ? new RenderTexture(requestedWidth, requestedHeight, 0)
                        : null;

                    if (rawImage != null) rawImage.texture = webCamTexture;
                    webCamName = webCamDevices[i].name;
                    _webcamIndex = i;
                    found = true;
                    break;
                }
            }

            return found;
        }

        /// <summary>
        /// Loads the camera in WebCamTexture.devices with index and resolution. If no resolution is provided, the WebCamTexture default for the camera will be used.
        /// </summary>
        /// <param name="index">int camera index to load</param>
        /// <param name="requestedWidth">requested width of the camera in pixels</param>
        /// <param name="requestedHeight">requested height of the camera in pixels</param>
        /// <returns>true if camera was found, false if not</returns>
        public bool LoadCamera(int index, int requestedWidth = -1, int requestedHeight = -1)
        {
            var found = false;

            var webCamDevices = WebCamTexture.devices;

            if (index >= 0 && webCamDevices.Length > index)
            {
                //Release the previous webcam + RT before replacing them, otherwise every camera
                //switch leaks the old WebCamTexture and RenderTexture (both native resources).
                ReleaseCurrentCamera();

                //If both values are not -1 use them, otherwise use WebCamTexture default
                if (requestedWidth > 0 && requestedHeight > 0)
                    webCamTexture = new WebCamTexture(webCamDevices[index].name, requestedWidth, requestedHeight);
                else
                    webCamTexture = new WebCamTexture(webCamDevices[index].name);
                webCamTexture.Play();
                //inputRT sizing deferred to Update() (see the front-facing overload above): right after
                //Play() the WebCamTexture still reports the 16x16 placeholder.
                inputRT = requestedWidth > 0 && requestedHeight > 0
                    ? new RenderTexture(requestedWidth, requestedHeight, 0)
                    : null;

                webCamName = webCamDevices[index].name;
                if (rawImage != null) rawImage.texture = webCamTexture;
                _webcamIndex = index;
                found = true;
            }

            return found;
        }

        /// <summary>
        /// Loads the next camera in WebCamTexture.devices. Can also set a requested Resolution for the webcam.
        /// </summary>
        /// <param name="requestedWidth">requested width of the camera in pixels</param>
        /// <param name="requestedHeight">requested height of the camera in pixels</param>
        /// <returns>true if found, false if not</returns>
        public bool NextCamera(int requestedWidth = -1, int requestedHeight = -1)
        {
            //Wrap past the last device so repeated presses cycle (matches WebCamSource.SelectSource)
            int count = WebCamTexture.devices.Length;
            if (count == 0) return false;
            return LoadCamera((_webcamIndex + 1) % count, requestedWidth, requestedHeight);
        }

        /// <summary>
        /// Loads the previous camera in WebCamTexture.devices. Can also set a requested Resolution for the webcam.
        /// </summary>
        /// <param name="requestedWidth">requested width of the camera in pixels</param>
        /// <param name="requestedHeight">requested height of the camera in pixels</param>
        /// <returns>true if found, false if not</returns>
        public bool PreviousCamera(int requestedWidth = -1, int requestedHeight = -1)
        {
            //Wrap below the first device so repeated presses cycle (matches WebCamSource.SelectSource)
            int count = WebCamTexture.devices.Length;
            if (count == 0) return false;
            return LoadCamera(((_webcamIndex - 1) % count + count) % count, requestedWidth, requestedHeight);
        }

        /// <summary>
        /// Stops the webcam from being drawn on the RawImage and saves reference to a backup.
        /// </summary>
        public void RemoveImage()
        {
            if (rawImage != null && !_backupped)
            {
                _rawImageBackup = rawImage.texture;
                rawImage.texture = _textureBackup;
                _backupped = true;
            }
        }

        /// <summary>
        /// Restores webcam to RawImage backup.
        /// </summary>
        public void RestoreImage()
        {
            if (rawImage != null)
            {
                rawImage.texture = _rawImageBackup;
                _backupped = false;
            }
        }
    }
}
