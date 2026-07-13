/// Code based on <see cref="Gaze"/>.
/// Updated by Tobias Wagner 07/2023 to integrate <see cref="Mediapipe.Unity"/> package.

#if !UNITY_WEBGL || UNITY_EDITOR
using Mediapipe.Unity;
using Mediapipe.Unity.FaceMesh;
#endif
using System.Collections.Generic;
using UnitEye;
using UnityEngine;
using Screen = UnityEngine.Screen;
namespace UnitEye
{

    public class HomulerGaze : MonoBehaviour
    {

        const int IMG_SIZE = 128;
        const int CROSSHAIR_SIZE = 80;

        #region Private values
        private AOIManager _aoiManager = new AOIManager();
        private List<string> aoiNameList = new List<string>();

        private bool _drowsy;
        private bool _blinking;
        private float _distance;

        private Rect gazeUI = new Rect(Screen.height * 0.05f, Screen.height * 0.08f, Screen.width * 0.5f, Screen.height * 0.82f);

        private GUIStyle style = new GUIStyle();

        // Cached debug-UI styles + enum arrays, rebuilt only when the resolution changes. This replaces
        // (a) mutating the SHARED GUI.skin styles in place every frame — which permanently leaked
        // bold/white/scaled text into every other IMGUI in the host game — and (b) allocating a GUIStyle
        // and two Enum.GetValues arrays on every OnGUI pass.
        private static readonly Calibrations[] _calibrationValues = (Calibrations[])System.Enum.GetValues(typeof(Calibrations));
        private static readonly Filtering[] _filteringValues = (Filtering[])System.Enum.GetValues(typeof(Filtering));
        private static readonly GazeBackbone[] _backboneValues = (GazeBackbone[])System.Enum.GetValues(typeof(GazeBackbone));
        private GUIStyle _uiStyleBox, _uiStyleButton, _uiStyleLabel, _uiStyleHSThumb, _toggleStyle;
        private int _uiStyleW = -1, _uiStyleH = -1;

        private AOIBox _offscreenAOI;

        //Owns the calibration models + raw->calibrated refinement (extracted out of this class).
        private readonly CalibrationModelStore _modelStore = new CalibrationModelStore();
        //Platform seam: native MediaPipe+Inference Engine on desktop, or a browser-JS provider on WebGL.
        private IGazeProvider _provider;
        private KalmanFilter kalmanFilter;
        private EaseSmoothing easeSmoothing;
        private OneEuroFilter<Vector2> oneEuroFilter;

        [SerializeField] private HomulerGazeCalibration _calibrationScript;
        private HomulerGazeEvaluation _evaluationScript;

        private bool _drawDotBackup = true;
        private bool _showEyesBackup = true;
        private bool _visualizeAOIBackup = false;
        private bool _showGazeUIBackup = false;
        private Calibrations _calibrationBackup;
        private bool _backupped;
        #endregion

        #region Public accessors
        //The platform-specific gaze producer (CV). Consumers (calibration, API) go through this, not the concrete runner.
        public IGazeProvider Provider => _provider;
        //A user is considered present while the provider is tracking a face
        public bool IsUserPresent => _provider != null && _provider.IsFacePresent;
        public AOIManager AOIManager { get => _aoiManager; }
        public AOI OffscreenAOI { get => _offscreenAOI; }
        public CSVLogger CSVLogger { get => _csvLogger; }
        public bool Drowsy { get => _drowsy; }
        public bool Blinking { get => _blinking; }
        public float Distance { get => _distance; }
        public bool PauseCSVLogging { get; set; }
        public long LastGazeLocationTimeUnix { get; private set; }
        /// <summary>Increments once for every fresh provider gaze sample consumed by this component.</summary>
        public long GazeSampleSequence { get; private set; }
        #endregion

        #region Serialized values
        [SerializeField] private GameObject _mediaPipeGO;
        [SerializeField] private CSVLogger _csvLogger;

        public Vector2 gazeLocation = Vector2.zero;

        public Texture2D dot;
        public bool drawDot = true;
        public bool showEyes = true;
        public bool visualizeAOI = false;
        public bool showGazeUI = false;
        //Debug MediaPipe face-mesh landmark overlay (the 468 points on the face). Off = a small perf win
        //(no per-frame point draw). Toggleable from the Gaze UI; honoured across calibration restores.
        public bool showFaceMesh = true;

        //Which gaze model the native provider runs (EyeMU by default; the GazeEstimation models need an
        //ONNX + hand-test, see docs/GAZE-BACKBONES.md). Read at Start; can also be switched at runtime via
        //SetBackbone / the Gaze UI. The current value is kept in sync when switched.
        [SerializeField] private GazeBackbone _gazeBackbone = GazeBackbone.EyeMU;
        public GazeBackbone GazeBackbone => _gazeBackbone;
        //The direction-based backbones feed the model a FACE crop (shown as one thumbnail), not eye crops.
        private bool UsesFaceCrop => _gazeBackbone != GazeBackbone.EyeMU;

        /// <summary>
        /// Switch the gaze model at runtime. Rebuilds the provider's backbone; the shared face-mesh /
        /// blink / distance stack is unchanged. Calibration is per-backbone (the feature vector differs),
        /// so gaze falls back to raw (uncalibrated) until you recalibrate for the new model.
        /// </summary>
        public void SetBackbone(GazeBackbone backbone)
        {
            //Refuse to swap the model while a calibration/evaluation is running: the backbones produce
            //DIFFERENT feature-vector lengths (EyeMU 12 vs GazeEstimation 8), so a mid-run switch would mix
            //jagged rows into the capture (training then throws inside the coroutine) and the result would
            //be saved under the NEW backbone's calibration file, silently corrupting it.
            if ((_calibrationScript != null && _calibrationScript.enabled) ||
                (_evaluationScript != null && _evaluationScript.enabled))
            {
                UnitEyeLog.Warn("SetBackbone ignored: finish or cancel the running calibration/evaluation first.");
                return;
            }

            _gazeBackbone = backbone;
            _provider?.SetBackbone(backbone);
            //Load this backbone's own calibration (per-backbone files). If it hasn't been calibrated yet,
            //the models load as null and RefineGazeLocation falls back to raw gaze until you calibrate.
            _modelStore.Load(_calibrations, _gazeBackbone);
        }

        [System.NonSerialized]
        public bool gazeUIActivated;

        [SerializeField]
        private Calibrations _calibrations = Calibrations.RidgeRegression;
        public Calibrations Calibrations
        {
            get => _calibrations;
            set
            {
                //Append a note to csv entry if calibration changed
                if (Application.isPlaying && _csvLogger != null && _csvLogger.isActiveAndEnabled && value != _calibrations)
                    _csvLogger.AppendNote($"Changed calibration type to {_calibrations}");

                _calibrations = value;
                _modelStore.Load(_calibrations, _gazeBackbone);
            }
        }
        [SerializeField]
        private Filtering _filtering = Filtering.OneEuro;
        public Filtering Filtering
        {
            get => _filtering;
            set
            {
                //Append a note to csv entry if filtering changed
                if (Application.isPlaying && _csvLogger != null && _csvLogger.isActiveAndEnabled && value != _filtering)
                    _csvLogger.AppendNote($"Changed filtering type to {_filtering}");

                _filtering = value;
            }
        }

        [SerializeField, Range(0, 1)] public float easefactor = 0.4f;

        [SerializeField, Range(1e-10f, 1.0f)] public float Q = 1e-5f;
        [SerializeField, Range(1e-10f, 1.0f)] public float R = 1e-4f;

        //Smooths rapid big movement
        [SerializeField, Range(1e-10f, 0.05f)] public float beta = 0.001f;
        //Smooths fixation jitter
        [SerializeField, Range(1e-10f, 1.0f)] public float mincutoff = 0.001f;
        [SerializeField, Range(1e-10f, 10.0f)] public float dcutoff = 1.0f;

        //Hold the last gaze location while blinking instead of feeding unreliable eye crops through calibration/filtering
        [SerializeField] public bool holdGazeDuringBlink = true;
        [SerializeField, Range(0.1f, 2.0f)] public float maxBlinkHoldSeconds = 0.5f;
        private float _blinkHoldStartedAt = -1f;

        [SerializeField, Range(30, 120)] private int _frameRate = 30;

        private bool _isRendering;
        public bool IsRendering {
            get => _isRendering;
            private set
            {
                _isRendering = value;
                //Route through the provider seam (as AnnotateFaceMesh already does) instead of a direct
                //GetComponent<FaceMeshSolution>: the native provider forwards to its cached solution and
                //WebGL no-ops, so rendering + annotate state always reach the same object via one path.
                if (_provider != null)
                {
                    //Respect the user's showFaceMesh choice when rendering resumes (calibration turns
                    //everything off, but must not force the mesh overlay back on afterwards).
                    _provider.AnnotateFaceMesh = _isRendering && showFaceMesh;
                    _provider.SetRendering(_isRendering);
                }
            }
        }

        #endregion

        public virtual void Start()
        {
            Application.targetFrameRate = _frameRate;

            //Create the platform gaze provider. The seam keeps everything below (calibration/filter/AOI/CSV)
            //identical across platforms; only the webcam->raw-gaze producer differs.
    #if UNITY_WEBGL && !UNITY_EDITOR
            _provider = new WebGLGazeProvider();
    #else
            _provider = new NativeGazeProvider(_mediaPipeGO, _gazeBackbone);
    #endif

            //Apply the initial face-mesh overlay preference
            _provider.AnnotateFaceMesh = showFaceMesh;

            _modelStore.Load(_calibrations, _gazeBackbone);

            //Create filters
            kalmanFilter = new KalmanFilter(Q, R);
            easeSmoothing = new EaseSmoothing(easefactor);
            oneEuroFilter = new OneEuroFilter<Vector2>(60f, mincutoff, beta, dcutoff);

            //Add offscreen AOI by default
            _offscreenAOI = new AOIBox("Offscreen", new Vector2(0f, 0f), new Vector2(1f, 1f), true, true, true);
            _aoiManager.AddAOI(_offscreenAOI);

            //Prepare GUI style for AOI string
            style.fontSize = 30;
            style.fontStyle = FontStyle.Bold;
            style.normal.textColor = Color.magenta;

            //Apply AOI visualization
            if (_aoiManager != null)
            {
                if (visualizeAOI)
                    _aoiManager.EnableVisualize();
                else
                    _aoiManager.DisableVisualize();
            }
        }

        public virtual void OnValidate()
        {
            //Update filter values
            if (kalmanFilter != null)
            {
                kalmanFilter.Q = Q;
                kalmanFilter.R = R;
            }
            if (easeSmoothing != null)
            {
                easeSmoothing.Factor = easefactor;
            }
            if (oneEuroFilter != null)
            {
                oneEuroFilter.UpdateParams(60f, mincutoff, beta, dcutoff);
            }
            //Update calibration and filtering type
            Calibrations = _calibrations;
            Filtering = _filtering;
            if (Application.isPlaying && _aoiManager != null)
            {
                //If this throws a warning, ignore it, is a Unity bug with attaching GameObjects to Main camera in OnValidate()
                if (visualizeAOI)
                    _aoiManager.EnableVisualize();
                else
                    _aoiManager.DisableVisualize();
            }
        }

        public virtual void LateUpdate()
        {
            //Peform neural network inference through entire eye tracking pipeline
            if (!_provider.Tick())
                return;

            GazeSampleSequence++;

            //Drowsy, blinking and distance
            _drowsy = _provider.IsDrowsy;
            _blinking = _provider.IsBlinking;
            _distance = _provider.DistanceMm;

            //While blinking the eye crops are unreliable, so optionally hold the last gaze location
            //instead of feeding the resulting spike through calibration and filtering. Capped so a
            //miscalibrated blinking threshold cannot freeze the gaze location.
            bool holdGaze = false;
            if (holdGazeDuringBlink && _blinking)
            {
                if (_blinkHoldStartedAt < 0f)
                    _blinkHoldStartedAt = Time.unscaledTime;
                holdGaze = Time.unscaledTime - _blinkHoldStartedAt <= maxBlinkHoldSeconds;
            }
            else
            {
                _blinkHoldStartedAt = -1f;
            }

            Vector2 unfilteredGaze;
            if (holdGaze)
            {
                unfilteredGaze = gazeLocation;
            }
            else
            {
                //Get raw gaze location from the provider
                gazeLocation.x = _provider.RawGaze.x;
                gazeLocation.y = _provider.RawGaze.y;

                //Apply calibration
                gazeLocation = RefineGazeLocation(gazeLocation, _calibrations);

                //Apply filtering
                unfilteredGaze = gazeLocation;
                gazeLocation = SmoothGazeLocation(gazeLocation, _filtering);
            }

            //Update last gaze location timestamp
            var now = System.DateTime.Now;
            LastGazeLocationTimeUnix = ((System.DateTimeOffset)now).ToUnixTimeMilliseconds();

            //AOI updating (refills the reused scratch list; no per-frame allocation)
            _aoiManager.CheckAOIList(new Vector2(gazeLocation.x / Screen.width, gazeLocation.y / Screen.height), aoiNameList);

            //CSV Logging. ShouldLog is checked BEFORE building the row: with logsPerSecond below the frame
            //rate the limiter drops most frames, so skipping the CSVData + AOI-list copy on those frames
            //avoids steady per-frame garbage. CSVData retains its AOI list by reference until the queue is
            //flushed, so accepted rows get their OWN copy (the scratch list is refilled every frame).
            if (!PauseCSVLogging && _csvLogger != null && _csvLogger.isActiveAndEnabled && _csvLogger.ShouldLog)
                _csvLogger.Append(new CSVData(gazeLocation.x, gazeLocation.y, gazeLocation.x / Screen.width, gazeLocation.y / Screen.height, unfilteredGaze.x / Screen.width, unfilteredGaze.y / Screen.height, _distance, _provider.EyeFeature, _blinking, now, new List<string>(aoiNameList)));

            //Drowsy calibration
            if (_provider.IsCalibratingDrowsy)
                _provider.CalibrateDrowsy(false);

            //Unload Calibration if calibration is done
            if (_calibrationScript != null && _calibrationScript.Returned)
            {
                //Add one entry for the note because PauseCSVLogging is currently true (owned AOI copy:
                //CSVData keeps the list reference until the queue flush)
                if (_csvLogger != null && _csvLogger.isActiveAndEnabled)
                    _csvLogger.Append(new CSVData(gazeLocation.x, gazeLocation.y, gazeLocation.x / Screen.width, gazeLocation.y / Screen.height, unfilteredGaze.x / Screen.width, unfilteredGaze.y / Screen.height, _distance, _provider.EyeFeature, _blinking, now, new List<string>(aoiNameList)));
                UnloadCalibration();
            }

            //Unload Evaluation if evaluation is done
            if (_evaluationScript != null && _evaluationScript.Returned)
                UnloadEvaluation();
        }

        public virtual void OnGUI()
        {
            //Draw the debug crop textures if they exist. EyeMU produces two eye crops (left drawn
            //top-right mirrored, right top-left); the direction-based backbones produce a single FACE
            //crop, so draw just one thumbnail for those instead of the same face twice.
            if (showEyes && _provider?.LeftEyeTexture != null && _provider?.RightEyeTexture != null)
            {
                if (UsesFaceCrop)
                {
                    GUI.DrawTexture(new Rect(10, 10, IMG_SIZE, IMG_SIZE), _provider.LeftEyeTexture);
                }
                else
                {
                    GUI.DrawTexture(new Rect(Screen.width - 10, 10, -IMG_SIZE, IMG_SIZE), _provider.LeftEyeTexture);
                    GUI.DrawTexture(new Rect(10, 10, IMG_SIZE, IMG_SIZE), _provider.RightEyeTexture);
                }
            }

            //Draw crosshair on the GUI if one is selected. The size scales with screen height (a fixed
            //80px dot is a speck on a high-DPI display like 3200x2000), and the position is clamped so the
            //WHOLE crosshair stays on-screen: an uncalibrated or badly calibrated gaze location can be far
            //outside the window, and an invisible crosshair is indistinguishable from a broken pipeline.
            if (drawDot && dot != null && !float.IsNaN(gazeLocation.x) && !float.IsNaN(gazeLocation.y))
            {
                //Scale relative to a 1080p baseline where 80px looked right; never shrink below the baseline
                float crosshairSize = Mathf.Max(CROSSHAIR_SIZE, CROSSHAIR_SIZE * Screen.height / 1080f);
                float half = crosshairSize / 2f;
                float dotX = Mathf.Clamp(gazeLocation.x, half, Screen.width - half);
                float dotY = Mathf.Clamp(gazeLocation.y, half, Screen.height - half);
                GUI.DrawTexture(new Rect(dotX - half, dotY - half, crosshairSize, crosshairSize), dot);
            }

            //Font scale relative to a 1080p baseline (== 1.0 at 1920x1080). Fixed-pixel IMGUI text is a
            //speck on high-DPI displays (e.g. 3200x2000), so scale it up; never shrink below the baseline.
            float uiScale = Mathf.Max(1f, Mathf.Sqrt(0.001f * Screen.width * Screen.height / 2073.6f));

            //Gaze UI. The toggle button used Unity's default built-in font, which is tiny at high DPI.
            if (showGazeUI)
            {
                EnsureGazeUIStyles(Screen.width, Screen.height);
                if (GUI.Button(new Rect(Screen.height * 0.05f, Screen.height - Screen.height * 0.1f, Screen.width * 0.1f, Screen.height * 0.05f), $"{(gazeUIActivated ? "Hide" : "Show")} Gaze UI", _toggleStyle))
                    gazeUIActivated = !gazeUIActivated;
            }

            if (gazeUIActivated)
                gazeUI = GUI.Window(0, gazeUI, GazeUI, "");

            //Draw text
            if (visualizeAOI && aoiNameList != null && aoiNameList.Count > 0)
            {
                style.fontSize = Mathf.RoundToInt(30f * uiScale);
                GUI.Label(new Rect(200, 100, 500, 50), string.Join(", ", aoiNameList), style);
            }
        }

        public virtual void OnDestroy()
        {
            // Must call Dispose method when no longer in use.
            _provider?.Dispose();
            _provider = null;
        }

        /// <summary>
        /// Refines the EyeMU gaze location by applying the calibrated model (delegates to the model store).
        /// </summary>
        /// <param name="calibrations">The calibrated model type to use</param>
        /// <returns>The calibrated gaze location</returns>
        public Vector2 RefineGazeLocation(Vector2 rawGaze, Calibrations calibrations)
        {
            return _modelStore.Refine(rawGaze, calibrations, _provider.GetFeatures(), Screen.width, Screen.height);
        }

        /// <summary>
        /// Smooths the specified gaze location by applying special filters
        /// </summary>
        /// <param name="unfilteredGaze">The unfiltered gaze location</param>
        /// <param name="filtering">The filter to apply</param>
        /// <returns>The smoothed gaze location</returns>
        public Vector2 SmoothGazeLocation(Vector2 unfilteredGaze, Filtering filtering)
        {
            Vector2 smoothedGaze = Vector2.zero;

            switch (filtering)
            {
                case Filtering.Kalman:
                    smoothedGaze = kalmanFilter.Update(unfilteredGaze);
                    break;
                case Filtering.Easing:
                    smoothedGaze = easeSmoothing.Update(unfilteredGaze);
                    break;
                case Filtering.KalmanEasing:
                    smoothedGaze = kalmanFilter.Update(easeSmoothing.Update(unfilteredGaze));
                    break;
                case Filtering.EasingKalman:
                    smoothedGaze = easeSmoothing.Update(kalmanFilter.Update(unfilteredGaze));
                    break;
                case Filtering.OneEuro:
                    //FilterVector2: allocation-free equivalent of Filter<Vector2> (no per-frame boxing)
                    smoothedGaze = oneEuroFilter.FilterVector2(unfilteredGaze, Time.realtimeSinceStartup);
                    break;
                default:
                    smoothedGaze = unfilteredGaze;
                    break;
            }

            return smoothedGaze;
        }

        /// <summary>
        /// Attaches Calibration script and backs up settings.
        /// </summary>
        /// <param name="speed">Speed of the calibration dot</param>
        /// <param name="padding">Padding in pixels around the edges</param>
        /// <param name="rounds">Number of Rounds</param>
        public void LoadCalibration(float speed = 9.0f, float padding = 20.0f, int rounds = 2)
        {
            //Return if we have no Calibration to calibrate for
            if (_calibrations == Calibrations.None) 
                return;

            IsRendering = false;

            //Backup settings
            BackupSettings();

            //Hide everything for calibration
            showEyes = false;
            showGazeUI = false;
            visualizeAOI = false;
            drawDot = false;
            gazeUIActivated = false;

            //Append a note to csv entry
            if (_csvLogger != null && _csvLogger.isActiveAndEnabled && _calibrationScript == null)
                _csvLogger.AppendNote("Started calibration");

            //Unpause CSVLogging
            PauseCSVLogging = true;

            //Attach calibration to same gameObject
            _calibrationScript.enabled = true;

            //Set _calibrations to none for a bit of performance gain
            _calibrationScript.calibrationType = _calibrations;
            _calibrations = Calibrations.None;

            //Calibration settings
            _calibrationScript.quitAfterCalibration = false;
            _calibrationScript.returnAfter = true;
            _calibrationScript.speed = speed;
            _calibrationScript.padding = padding;
            _calibrationScript.maxRoundsPerPreset = rounds;
        }

        /// <summary>
        /// Destroys Calibration script and restores settings.
        /// </summary>
        private void UnloadCalibration()
        {
            //Restore settings
            RestoreSettings();

            //Unpause CSVLogging
            PauseCSVLogging = false;

            //Append a note to csv entry
            if (_csvLogger != null && _csvLogger.isActiveAndEnabled)
                _csvLogger.AppendNote(_calibrationScript.ReturnMessage);

            _calibrationScript.enabled = false;
            //Consume the return so LateUpdate does not call UnloadCalibration again next frame
            _calibrationScript.ClearReturned();

            //Reload calibration file
            Calibrations = _calibrations;

            IsRendering = true;
        }

        /// <summary>
        /// Attaches GazeEvaluation script and backs up settings.
        /// </summary>
        /// <param name="rows">Number of rows in the dot grid</param>
        /// <param name="columns">Number of columns in the dot grid</param>
        public void LoadEvaluation(int rows = 5, int columns = 5)
        {
            IsRendering = false;

            //Backup settings
            BackupSettings();

            //Hide everything for calibration
            showEyes = false;
            showGazeUI = false;
            visualizeAOI = false;
            drawDot = false;
            gazeUIActivated = false;

            //Append a note to csv entry
            if (_csvLogger != null && _csvLogger.isActiveAndEnabled && _calibrationScript == null)
                _csvLogger.AppendNote("Started evaluation");

            //Attach calibration to same gameObject
            _evaluationScript = GetComponent<HomulerGazeEvaluation>();
            _evaluationScript.enabled = true;

            //Evaluation settings
            _evaluationScript.quitAfterEvaluation = false;
            _evaluationScript.returnAfter = true;
            _evaluationScript.rows = rows;
            _evaluationScript.columns = columns;
        }

        /// <summary>
        /// Destroys GazeEvaluation script and restores settings.
        /// </summary>
        private void UnloadEvaluation()
        {
            //Restore settings
            RestoreSettings();

            //Append a note to csv entry
            if (_csvLogger != null && _csvLogger.isActiveAndEnabled)
                _csvLogger.AppendNote(_evaluationScript.ReturnMessage);

            //Destroy calibration script
            _evaluationScript.enabled = false;
            //Consume the return so LateUpdate does not call UnloadEvaluation again next frame
            _evaluationScript.ClearReturned();

            IsRendering = true;
        }

        /// <summary>
        /// Backup relevant settings.
        /// </summary>
        private void BackupSettings()
        {
            //Only backup if not already backupped. _backupped MUST be set here: without it a second Load
            //(e.g. LoadEvaluation while a calibration is still up) re-captured the already-hidden state —
            //including _calibrations == None — and RestoreSettings then made that corruption permanent.
            if (!_backupped)
            {
                _showEyesBackup = showEyes;
                _showGazeUIBackup = showGazeUI;
                _visualizeAOIBackup = visualizeAOI;
                _drawDotBackup = drawDot;
                _calibrationBackup = _calibrations;
                _backupped = true;
            }
        }

        /// <summary>
        /// Restore settings backup.
        /// </summary>
        private void RestoreSettings()
        {
            //Restore settings
            showEyes = _showEyesBackup;
            showGazeUI = _showGazeUIBackup;
            visualizeAOI = _visualizeAOIBackup;
            drawDot = _drawDotBackup;
            _calibrations = _calibrationBackup;
            //Allow the next Load to take a fresh backup
            _backupped = false;
        }

        #region GazeUI GUI

        /// <summary>
        /// Creates the draggable Gaze UI overlay.
        /// </summary>
        /// <param name="windowID"></param>
        /// <summary>
        /// Builds the cached debug-UI GUIStyles once per resolution (rebuilt when the screen size changes so
        /// text stays legible on high-DPI displays). Must be called from OnGUI: GUI.skin is only valid there.
        /// Replaces the old per-pass in-place mutation of the shared GUI.skin styles.
        /// </summary>
        void EnsureGazeUIStyles(int width, int height)
        {
            if (_uiStyleBox != null && _uiStyleW == width && _uiStyleH == height)
                return;
            _uiStyleW = width;
            _uiStyleH = height;

            //Font scale relative to a 1080p baseline (== 1.0 at 1920x1080). Uses the LIMITING axis, not
            //sqrt(w*h): the window's button rects scale with width/height directly, so a geometric-mean
            //font scale outgrows the buttons on non-16:9 displays (e.g. 3200x2000: font x1.76 vs button
            //width x1.67) and text that fit at 1080p starts spilling onto a second line.
            float resolutionScale = Mathf.Min(width / 1920f, height / 1080f);
            int fontSize = (int)(14f * resolutionScale);

            _uiStyleBox = new GUIStyle(GUI.skin.box) { wordWrap = true, fontStyle = FontStyle.Bold, fontSize = fontSize };
            _uiStyleBox.normal.textColor = Color.white;
            _uiStyleButton = new GUIStyle(GUI.skin.button) { wordWrap = true, fontStyle = FontStyle.Bold, fontSize = fontSize };
            _uiStyleButton.normal.textColor = Color.white;
            _uiStyleLabel = new GUIStyle(GUI.skin.label) { wordWrap = true, fontStyle = FontStyle.Bold, fontSize = fontSize };
            _uiStyleLabel.normal.textColor = Color.white;
            _uiStyleHSThumb = new GUIStyle(GUI.skin.horizontalSliderThumb) { fontSize = fontSize };

            //The "Show/Hide Gaze UI" toggle button never shrinks below the baseline (matches the old uiScale)
            _toggleStyle = new GUIStyle(GUI.skin.button) { fontSize = Mathf.RoundToInt(14f * Mathf.Max(1f, resolutionScale)) };
        }

        //Short display names for the Gaze UI's narrow buttons. The raw enum names are single long words
        //("RidgeRegression", "KalmanEasing", "GazeMobileNetV2") that cannot word-wrap, so in a narrow
        //button IMGUI breaks them mid-word and the tail characters spill onto a second line.
        private static string DisplayName(Calibrations cal)
        {
            switch (cal)
            {
                case Calibrations.RidgeRegression: return "Ridge";
                case Calibrations.MLCalibration: return "MLP";
                default: return cal.ToString();
            }
        }

        private static string DisplayName(Filtering filt)
        {
            switch (filt)
            {
                case Filtering.KalmanEasing: return "Kal+Ease";
                case Filtering.EasingKalman: return "Ease+Kal";
                default: return filt.ToString();
            }
        }

        private static string DisplayName(GazeBackbone backbone)
        {
            switch (backbone)
            {
                case GazeBackbone.GazeMobileOne: return "MobileOne";
                case GazeBackbone.GazeMobileNetV2: return "MobileNetV2";
                case GazeBackbone.GazeResNet34: return "ResNet34";
                default: return backbone.ToString();
            }
        }

        void GazeUI(int windowID)
        {
            //This method of GUI drawing is not very efficient but it works for now

            //Runtime rescaling in case the screen size changes
            var width = Screen.width;
            gazeUI.width = width * 0.5f;
            var height = Screen.height;
            gazeUI.height = height * 0.82f;

            //Opaque backdrop: Unity's IMGUI window is semi-transparent by default, so the live webcam feed
            //behind it shows through and washes out the text (unreadable regardless of font size). Paint a
            //solid dark panel over the whole window before drawing any content.
            var prevGuiColor = GUI.color;
            GUI.color = new Color(0.12f, 0.12f, 0.13f, 1f);
            GUI.DrawTexture(new Rect(0f, 0f, gazeUI.width, gazeUI.height), Texture2D.whiteTexture);
            GUI.color = prevGuiColor;

            //Use the cached styles (built once per resolution) instead of mutating the shared GUI.skin
            //styles in place every pass. High-contrast bold white text on the dark panel: the default skin
            //text is grey and, on high-DPI editor Game Views (which render at game resolution then upscale),
            //a little soft — bold white reads far more clearly.
            EnsureGazeUIStyles(width, height);
            var gazeUIStyleBox = _uiStyleBox;
            var gazeUIStyleButton = _uiStyleButton;
            var gazeUIStyleLabel = _uiStyleLabel;
            var gazeUIStyleHSThumb = _uiStyleHSThumb;

            //Make header draggable
            GUI.DragWindow(new Rect(0, 0, width, height * 0.02f));

            //Webcam + model controls
            // This might be broken (TW 07/2023)
            // We may have to restart the new webcam if we change the source.
            GUI.BeginGroup(new Rect(width * 0.01f, height * 0.02f, width * 0.48f, height * 0.08f));

            GUI.Box(new Rect(0, 0, width * 0.48f, height * 0.08f), "Webcam & Model controls", gazeUIStyleBox);

            if (GUI.Button(new Rect(width * 0.02f, height * 0.025f, width * 0.1f, height * 0.05f), $"Prev Cam", gazeUIStyleButton))
            {
                _provider.PreviousCamera();
            }

            //Truncate long device names ("Integrated Webcam FHD (04f2:b6d9)" etc.) — the label is narrow
            //and long parenthesized IDs break mid-word onto extra lines over the controls below.
            var camName = _provider.CurrentCameraName;
            if (camName != null && camName.Length > 22)
                camName = camName.Substring(0, 21) + "…";
            GUI.Label(new Rect(width * 0.125f, height * 0.025f, width * 0.11f, height * 0.05f), $"Cam: {camName}", gazeUIStyleLabel);

            if (GUI.Button(new Rect(width * 0.24f, height * 0.025f, width * 0.1f, height * 0.05f), $"Next Cam", gazeUIStyleButton))
            {
                _provider.NextCamera();
            }

            //Switch the gaze model at runtime (cycles EyeMU -> MobileOne -> MobileNetV2). The pipeline
            //falls back to raw gaze until you recalibrate for the newly selected model.
            if (GUI.Button(new Rect(width * 0.35f, height * 0.025f, width * 0.12f, height * 0.05f), $"Model: {DisplayName(_gazeBackbone)}", gazeUIStyleButton))
            {
                int i = System.Array.IndexOf(_backboneValues, _gazeBackbone);
                SetBackbone(_backboneValues[(i + 1) % _backboneValues.Length]);
            }

            GUI.EndGroup();

            //Toggle buttons
            GUI.BeginGroup(new Rect(width * 0.01f, height * 0.11f, width * 0.48f, height * 0.08f));

            GUI.Box(new Rect(0, 0, width * 0.48f, height * 0.08f), "Toggle UI Overlays", gazeUIStyleBox);

            if (GUI.Button(new Rect(width * 0.02f, height * 0.025f, width * 0.105f, height * 0.05f), $"{(visualizeAOI ? "Hide" : "Show")} AOIs", gazeUIStyleButton))
                if (_aoiManager != null)
                {
                    visualizeAOI = !visualizeAOI;
                    if (visualizeAOI)
                        _aoiManager.EnableVisualize();
                    else
                        _aoiManager.DisableVisualize();
                }

            if (GUI.Button(new Rect(width * 0.135f, height * 0.025f, width * 0.105f, height * 0.05f), $"{(drawDot ? "Hide" : "Show")} GazeDot", gazeUIStyleButton))
                drawDot = !drawDot;

            if (GUI.Button(new Rect(width * 0.25f, height * 0.025f, width * 0.105f, height * 0.05f), $"{(showEyes ? "Hide" : "Show")} {(UsesFaceCrop ? "FaceCrop" : "Eyecrops")}", gazeUIStyleButton))
                showEyes = !showEyes;

            if (GUI.Button(new Rect(width * 0.365f, height * 0.025f, width * 0.105f, height * 0.05f), $"{(showFaceMesh ? "Hide" : "Show")} FaceMesh", gazeUIStyleButton))
            {
                showFaceMesh = !showFaceMesh;
                if (_provider != null)
                    _provider.AnnotateFaceMesh = showFaceMesh;
            }

            GUI.EndGroup();

            //Distance calibration
            GUI.BeginGroup(new Rect(width * 0.01f, height * 0.2f, width * 0.48f, height * 0.13f));

            GUI.Box(new Rect(0, 0, width * 0.48f, height * 0.13f), "Distance to camera Calibration", gazeUIStyleBox);

            GUI.Label(new Rect(width * 0.025f, height * 0.025f, width * 0.455f, height * 0.05f), "Calibrate Distance to Camera by pressing the button when your eyes are 50cm away from the camera. After calibration the calculated distance should match the real life distance. This value is saved between runs.", gazeUIStyleLabel);

            if (GUI.Button(new Rect(width * 0.025f, height * 0.07f, width * 0.2f, height * 0.05f), $"Calibrate Distance to Camera", gazeUIStyleButton))
                _provider.CalibrateDistance();

            GUI.Label(new Rect(width * 0.26f, height * 0.085f, width * 0.195f, height * 0.05f), $"Calculated distance: {_distance:F1} mm", gazeUIStyleLabel);

            GUI.EndGroup();

            //Drowsy and blinking calibration
            GUI.BeginGroup(new Rect(width * 0.01f, height * 0.34f, width * 0.48f, height * 0.13f));

            GUI.Box(new Rect(0, 0, width * 0.48f, height * 0.13f), "Blinking and Drowsiness Calibration", gazeUIStyleBox);

            GUI.Label(new Rect(width * 0.025f, height * 0.025f, width * 0.455f, height * 0.05f), "Calibrate blinking and drowsiness thresholds based on the eye aspect ratio. These values are saved between runs.", gazeUIStyleLabel);

            if (GUI.Button(new Rect(width * 0.025f, height * 0.07f, width * 0.1f, height * 0.05f), $"Calibrate Blinking Threshold", gazeUIStyleButton))
                _provider.CalibrateBlinking();

            GUI.Label(new Rect(width * 0.15f, height * 0.085f, width * 0.08f, height * 0.05f), $"{(_blinking ? "Eyes are closed" : "Eyes are open")}", gazeUIStyleLabel);

            //Short progress text: the old "Calibrating Drowsiness based on N values" needed 3+ wrapped
            //lines and overflowed the button vertically while calibrating.
            if (GUI.Button(new Rect(width * 0.255f, height * 0.07f, width * 0.1f, height * 0.05f), $"{(_provider.IsCalibratingDrowsy ? $"Calibrating… ({_provider.DrowsyCalibrationCount})" : "Calibrate Drowsiness Baseline")}", gazeUIStyleButton))
                _provider.CalibrateDrowsy(true);

            GUI.Label(new Rect(width * 0.38f, height * 0.085f, width * 0.08f, height * 0.05f), $"{(_drowsy ? "Drowsy" : "Alert")}", gazeUIStyleLabel);

            GUI.EndGroup();

            //These values are not saved yet, might be TODO
            //Filtering and calibration selection buttons
            GUI.BeginGroup(new Rect(width * 0.01f, height * 0.48f, width * 0.48f, height * 0.24f));

            GUI.Box(new Rect(0, 0, width * 0.48f, height * 0.24f), "Used filtering and calibration type selection", gazeUIStyleBox);

            //Unity doesn't have an easy way to create a Dropdownlist in OnGUI(), so we use loops.
            //Short display names throughout: the raw enum names are single unbreakable words that spill
            //characters onto a second line in these narrow buttons.
            //Calibration types
            GUI.Label(new Rect(width * 0.025f, height * 0.03f, width * 0.08f, height * 0.06f), $"Calibration type\n(current: {DisplayName(_calibrations)})", gazeUIStyleLabel);

            for (int i = 0; i < _calibrationValues.Length; i++)
            {
                var cal = _calibrationValues[i];
                if (GUI.Button(new Rect(i * width * 0.06f + width * 0.11f, height * 0.025f, width * 0.05f, height * 0.05f), DisplayName(cal), gazeUIStyleButton))
                    Calibrations = cal;
            }

            //Filtering types
            GUI.Label(new Rect(width * 0.025f, height * 0.09f, width * 0.08f, height * 0.06f), $"Filtering type\n(current: {DisplayName(Filtering)})", gazeUIStyleLabel);

            for (int i = 0; i < _filteringValues.Length; i++)
            {
                var filt = _filteringValues[i];
                if (GUI.Button(new Rect(i * width * 0.06f + width * 0.11f, height * 0.080f, width * 0.05f, height * 0.05f), DisplayName(filt), gazeUIStyleButton))
                    Filtering = filt;
            }

            //Filtering sliders
            var minfloat = 0.000001f;
            switch (Filtering)
            {
                case Filtering.Kalman:
                    GUI.Label(new Rect(width * 0.025f, height * 0.14f, width * 0.08f, height * 0.06f), $"Q: {Q}", gazeUIStyleLabel);
                    kalmanFilter.Q = Q = GUI.HorizontalSlider(new Rect(width * 0.11f, height * 0.15f, width * 0.3f, height * 0.02f), Q, minfloat, 0.0001f, GUI.skin.horizontalSlider, gazeUIStyleHSThumb);
                    GUI.Label(new Rect(width * 0.025f, height * 0.17f, width * 0.08f, height * 0.06f), $"R: {R}", gazeUIStyleLabel);
                    kalmanFilter.R = R = GUI.HorizontalSlider(new Rect(width * 0.11f, height * 0.18f, width * 0.3f, height * 0.02f), R, minfloat, 0.001f, GUI.skin.horizontalSlider, gazeUIStyleHSThumb);
                    break;
                case Filtering.Easing:
                    GUI.Label(new Rect(width * 0.025f, height * 0.14f, width * 0.08f, height * 0.06f), $"easefactor: {easefactor}", gazeUIStyleLabel);
                    easeSmoothing.Factor = easefactor = GUI.HorizontalSlider(new Rect(width * 0.11f, height * 0.15f, width * 0.3f, height * 0.02f), easefactor, minfloat, 1f, GUI.skin.horizontalSlider, gazeUIStyleHSThumb);
                    break;
                case Filtering.KalmanEasing:
                case Filtering.EasingKalman:
                    GUI.Label(new Rect(width * 0.025f, height * 0.14f, width * 0.08f, height * 0.06f), $"easefactor: {easefactor}", gazeUIStyleLabel);
                    easeSmoothing.Factor = easefactor = GUI.HorizontalSlider(new Rect(width * 0.11f, height * 0.15f, width * 0.3f, height * 0.02f), easefactor, minfloat, 1f, GUI.skin.horizontalSlider, gazeUIStyleHSThumb);
                    GUI.Label(new Rect(width * 0.025f, height * 0.17f, width * 0.08f, height * 0.06f), $"Q: {Q}", gazeUIStyleLabel);
                    kalmanFilter.Q = Q = GUI.HorizontalSlider(new Rect(width * 0.11f, height * 0.18f, width * 0.3f, height * 0.02f), Q, minfloat, 0.0001f, GUI.skin.horizontalSlider, gazeUIStyleHSThumb);
                    GUI.Label(new Rect(width * 0.025f, height * 0.20f, width * 0.08f, height * 0.06f), $"R: {R}", gazeUIStyleLabel);
                    kalmanFilter.R = R = GUI.HorizontalSlider(new Rect(width * 0.11f, height * 0.21f, width * 0.3f, height * 0.02f), R, minfloat, 0.001f, GUI.skin.horizontalSlider, gazeUIStyleHSThumb);
                    break;
                case Filtering.OneEuro:
                    GUI.Label(new Rect(width * 0.025f, height * 0.14f, width * 0.08f, height * 0.06f), $"Beta: {beta}", gazeUIStyleLabel);
                    beta = GUI.HorizontalSlider(new Rect(width * 0.11f, height * 0.15f, width * 0.3f, height * 0.02f), beta, minfloat, 0.05f, GUI.skin.horizontalSlider, gazeUIStyleHSThumb);
                    GUI.Label(new Rect(width * 0.025f, height * 0.17f, width * 0.08f, height * 0.06f), $"Mincutoff: {mincutoff}", gazeUIStyleLabel);
                    mincutoff = GUI.HorizontalSlider(new Rect(width * 0.11f, height * 0.18f, width * 0.3f, height * 0.02f), mincutoff, minfloat, 0.05f, GUI.skin.horizontalSlider, gazeUIStyleHSThumb);
                    GUI.Label(new Rect(width * 0.025f, height * 0.20f, width * 0.08f, height * 0.06f), $"Dcutoff: {dcutoff}", gazeUIStyleLabel);
                    dcutoff = GUI.HorizontalSlider(new Rect(width * 0.11f, height * 0.21f, width * 0.3f, height * 0.02f), dcutoff, minfloat, 10f, GUI.skin.horizontalSlider, gazeUIStyleHSThumb);
                    oneEuroFilter.UpdateParams(60f, mincutoff, beta, dcutoff);
                    break;
                default:
                    break;
            }

            GUI.EndGroup();

            //Calibration and evaluation buttons
            GUI.BeginGroup(new Rect(width * 0.01f, height * 0.73f, width * 0.48f, height * 0.08f));

            GUI.Box(new Rect(0, 0, width * 0.48f, height * 0.08f), "Calibration and evaluation controls", gazeUIStyleBox);

            if (GUI.Button(new Rect(width * 0.025f, height * 0.025f, width * 0.2f, height * 0.05f), $"Start calibration", gazeUIStyleButton))
            {
                LoadCalibration();
            }

            if (GUI.Button(new Rect(width * 0.255f, height * 0.025f, width * 0.2f, height * 0.05f), $"Start evaluation", gazeUIStyleButton))
            {
                LoadEvaluation();
            }

            GUI.EndGroup();
        }

        #endregion
    }
}
