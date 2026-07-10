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

    private AOIBox _offscreenAOI;

    private RidgeRegression _xModel, _yModel;
    private SimpleMLP _mlp;
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
            //Load calibration files, suppress exceptions as they are handled internally
            switch (_calibrations)
            {
                case Calibrations.RidgeRegression:
                    try { _xModel = RidgeRegression.LoadX("Reg_X.json"); } catch { }
                    try { _yModel = RidgeRegression.LoadY("Reg_Y.json"); } catch { }
                    break;
                case Calibrations.MLCalibration:
                    //Fix: the loaded model was previously discarded (never assigned to _mlp)
                    try { _mlp = SimpleMLP.Load("MLP.json"); } catch { }
                    break;
            }
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
#if !UNITY_WEBGL || UNITY_EDITOR
            //FaceMeshSolution only exists on non-WebGL players (native MediaPipe); on WebGL the browser owns rendering
            var solution = _mediaPipeGO.GetComponent<FaceMeshSolution>();
            if (solution != null)
            {
                solution.Annotate = _isRendering;
                solution.IsRendering = _isRendering;
            }
#endif
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
        _provider = new NativeGazeProvider(_mediaPipeGO);
#endif

        //Load calibration files, suppress exceptions as they are handled internally
        switch (_calibrations)
        {
            case Calibrations.RidgeRegression:
                try { _xModel = RidgeRegression.LoadX("Reg_X.json"); } catch { }
                try { _yModel = RidgeRegression.LoadY("Reg_Y.json"); } catch { }
                break;
            case Calibrations.MLCalibration:
                try { _mlp = SimpleMLP.Load("MLP.json"); } catch { }
                break;
        }

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

        //AOI updating
        aoiNameList = _aoiManager.CheckAOIList(new Vector2(gazeLocation.x / Screen.width, gazeLocation.y / Screen.height));

        //CSV Logging
        if (!PauseCSVLogging && _csvLogger != null && _csvLogger.isActiveAndEnabled)
            _csvLogger.Append(new CSVData(gazeLocation.x, gazeLocation.y, gazeLocation.x / Screen.width, gazeLocation.y / Screen.height, unfilteredGaze.x / Screen.width, unfilteredGaze.y / Screen.height, _distance, _provider.EyeFeature, _blinking, now, aoiNameList));

        //Drowsy calibration
        if (_provider.IsCalibratingDrowsy)
            _provider.CalibrateDrowsy(false);

        //Unload Calibration if calibration is done
        if (_calibrationScript != null && _calibrationScript.Returned)
        {
            //Add one entry for the note because PauseCSVLogging is currently true
            if (_csvLogger != null && _csvLogger.isActiveAndEnabled)
                _csvLogger.Append(new CSVData(gazeLocation.x, gazeLocation.y, gazeLocation.x / Screen.width, gazeLocation.y / Screen.height, unfilteredGaze.x / Screen.width, unfilteredGaze.y / Screen.height, _distance, _provider.EyeFeature, _blinking, now, aoiNameList));
            UnloadCalibration();
        }

        //Unload Evaluation if evaluation is done
        if (_evaluationScript != null && _evaluationScript.Returned)
            UnloadEvaluation();
    }

    public virtual void OnGUI()
    {
        //Draw eye textures on the GUI if they exist
        if (showEyes && _provider?.LeftEyeTexture != null && _provider?.RightEyeTexture != null)
        {
            GUI.DrawTexture(new Rect(Screen.width - 10, 10, -IMG_SIZE, IMG_SIZE), _provider.LeftEyeTexture);

            GUI.DrawTexture(new Rect(10, 10, IMG_SIZE, IMG_SIZE), _provider.RightEyeTexture);
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
            var toggleStyle = new GUIStyle(GUI.skin.button) { fontSize = Mathf.RoundToInt(14f * uiScale) };
            if (GUI.Button(new Rect(Screen.height * 0.05f, Screen.height - Screen.height * 0.1f, Screen.width * 0.1f, Screen.height * 0.05f), $"{(gazeUIActivated ? "Hide" : "Show")} Gaze UI", toggleStyle))
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
    /// Refines the EyeMU gaze location by applying a calibrated model
    /// </summary>
    /// <param name="calibrations">The calibrated model type to use</param>
    /// <returns>The calibrated gaze location</returns>
    public Vector2 RefineGazeLocation(Vector2 rawGaze, Calibrations calibrations)
    {
        var features = _provider.GetFeatures();
        Vector2 refinedGaze = Vector2.zero;

        //No feature vector (e.g. a browser provider streaming only raw gaze) -> calibration cannot
        //apply, use the raw gaze location directly
        if (features == null || features.Length == 0)
            return rawGaze;

        //Switch by calibration type
        switch (calibrations)
        {
            case Calibrations.None:
                refinedGaze = rawGaze;
                break;
            case Calibrations.RidgeRegression:
                //Fall back to the raw gaze location if no calibration model is loaded
                if (_xModel == null || _yModel == null)
                {
                    refinedGaze = rawGaze;
                    break;
                }
                refinedGaze.x = _xModel.Predict(features);
                refinedGaze.y = _yModel.Predict(features);

                refinedGaze.x *= Screen.width;
                refinedGaze.y *= Screen.height;
                break;
            case Calibrations.MLCalibration:
                //Fall back to the raw gaze location if no calibration model is loaded
                if (_mlp == null)
                {
                    refinedGaze = rawGaze;
                    break;
                }
                refinedGaze = _mlp.Predict(features);
                break;
        }
        //If calibration produced no usable value (NaN, e.g. model/feature mismatch), fall back to raw gaze
        if (float.IsNaN(refinedGaze.x) || float.IsNaN(refinedGaze.y))
            return rawGaze;

        return refinedGaze;
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
                smoothedGaze = oneEuroFilter.Filter(unfilteredGaze, Time.realtimeSinceStartup);
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
        //Only backup if not already backupped
        if (!_backupped)
        {
            _showEyesBackup = showEyes;
            _showGazeUIBackup = showGazeUI;
            _visualizeAOIBackup = visualizeAOI;
            _drawDotBackup = drawDot;
            _calibrationBackup = _calibrations;
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
    }

    #region GazeUI GUI

    /// <summary>
    /// Creates the draggable Gaze UI overlay.
    /// </summary>
    /// <param name="windowID"></param>
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

        //Set GUIStyles
        var gazeUIStyleBox = GUI.skin.box;
        var gazeUIStyleButton = GUI.skin.button;
        var gazeUIStyleLabel = GUI.skin.label;
        var gazeUIStyleHSThumb = GUI.skin.horizontalSliderThumb;
        gazeUIStyleButton.wordWrap = gazeUIStyleLabel.wordWrap = true;
        //High-contrast text on the dark panel (default skin text is grey and hard to read). Bold too:
        //the editor Game View renders at game resolution and upscales on high-DPI displays, so the text
        //is inherently a little soft there, and bold white reads much more clearly than thin grey.
        gazeUIStyleLabel.normal.textColor = Color.white;
        gazeUIStyleBox.normal.textColor = Color.white;
        gazeUIStyleButton.normal.textColor = Color.white;
        gazeUIStyleLabel.fontStyle = gazeUIStyleBox.fontStyle = gazeUIStyleButton.fontStyle = FontStyle.Bold;

        //Scale font based on Resolution comparison to 1080p
        var resolutionScale = Mathf.Sqrt((0.001f * (float)width * (float)height) / 2073.6f);
        gazeUIStyleHSThumb.fontSize = gazeUIStyleBox.fontSize = gazeUIStyleButton.fontSize = gazeUIStyleLabel.fontSize = (int)(14f * resolutionScale);

        //Make header draggable
        GUI.DragWindow(new Rect(0, 0, width, height * 0.02f));

        //Webcam controls
        // This might be broken (TW 07/2023)
        // We may have to restart the new webcam if we change the source.
        GUI.BeginGroup(new Rect(width * 0.01f, height * 0.02f, width * 0.48f, height * 0.08f));

        GUI.Box(new Rect(0, 0, width * 0.48f, height * 0.08f), "Webcam controls", gazeUIStyleBox);

        if (GUI.Button(new Rect(width * 0.025f, height * 0.025f, width * 0.12f, height * 0.05f), $"Previous Webcam", gazeUIStyleButton))
        {
            _provider.NextCamera();
        }

        GUI.Label(new Rect(width * 0.18f, height * 0.025f, width * 0.12f, height * 0.05f), $"Current Webcam: {_provider.CurrentCameraName}", gazeUIStyleLabel);

        if (GUI.Button(new Rect(width * 0.335f, height * 0.025f, width * 0.12f, height * 0.05f), $"Next Webcam", gazeUIStyleButton))
        {
            _provider.PreviousCamera();
        }

        GUI.EndGroup();

        //Toggle buttons
        GUI.BeginGroup(new Rect(width * 0.01f, height * 0.11f, width * 0.48f, height * 0.08f));

        GUI.Box(new Rect(0, 0, width * 0.48f, height * 0.08f), "Toggle UI Overlays", gazeUIStyleBox);

        if (GUI.Button(new Rect(width * 0.025f, height * 0.025f, width * 0.12f, height * 0.05f), $"{(visualizeAOI ? "Hide" : "Show")} AOIs", gazeUIStyleButton))
            if (_aoiManager != null)
            {
                visualizeAOI = !visualizeAOI;
                if (visualizeAOI)
                    _aoiManager.EnableVisualize();
                else
                    _aoiManager.DisableVisualize();
            }

        if (GUI.Button(new Rect(width * 0.18f, height * 0.025f, width * 0.12f, height * 0.05f), $"{(drawDot ? "Hide" : "Show")} GazeDot", gazeUIStyleButton))
            drawDot = !drawDot;

        if (GUI.Button(new Rect(width * 0.335f, height * 0.025f, width * 0.12f, height * 0.05f), $"{(showEyes ? "Hide" : "Show")} Eyecrops", gazeUIStyleButton))
            showEyes = !showEyes;

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

        if (GUI.Button(new Rect(width * 0.255f, height * 0.07f, width * 0.1f, height * 0.05f), $"{(_provider.IsCalibratingDrowsy ? $"Calibrating Drowsiness based on {_provider.DrowsyCalibrationCount} values" : "Calibrate Drowsiness Baseline")}", gazeUIStyleButton))
            _provider.CalibrateDrowsy(true);

        GUI.Label(new Rect(width * 0.38f, height * 0.085f, width * 0.08f, height * 0.05f), $"{(_drowsy ? "Drowsy" : "Alert")}", gazeUIStyleLabel);

        GUI.EndGroup();

        //These values are not saved yet, might be TODO
        //Filtering and calibration selection buttons
        GUI.BeginGroup(new Rect(width * 0.01f, height * 0.48f, width * 0.48f, height * 0.24f));

        GUI.Box(new Rect(0, 0, width * 0.48f, height * 0.24f), "Used filtering and calibration type selection", gazeUIStyleBox);

        //Unity doesn't have an easy way to create a Dropdownlist in OnGUI(), so we use loops
        //Calibration types
        GUI.Label(new Rect(width * 0.025f, height * 0.03f, width * 0.08f, height * 0.06f), $"Calibration type\n(current: {_calibrations})", gazeUIStyleLabel);

        var calibrationsEnumArray = System.Enum.GetValues(typeof(Calibrations));
        for (int i = 0; i < calibrationsEnumArray.Length; i++)
        {
            if (GUI.Button(new Rect(i * width * 0.06f + width * 0.11f, height * 0.025f, width * 0.05f, height * 0.05f), $"{(Calibrations)i}", gazeUIStyleButton))
                Calibrations = (Calibrations)i;
        }

        //Filtering types
        GUI.Label(new Rect(width * 0.025f, height * 0.09f, width * 0.08f, height * 0.06f), $"Filtering type\n(current: {Filtering})", gazeUIStyleLabel);

        var filteringEnumArray = System.Enum.GetValues(typeof(Filtering));
        for (int i = 0; i < filteringEnumArray.Length; i++)
        {
            if (GUI.Button(new Rect(i * width * 0.06f + width * 0.11f, height * 0.080f, width * 0.05f, height * 0.05f), $"{(Filtering)i}", gazeUIStyleButton))
                Filtering = (Filtering)i;
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
