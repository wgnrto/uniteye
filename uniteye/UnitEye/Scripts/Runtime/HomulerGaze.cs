/// Code based on <see cref="Gaze"/>.
/// Updated by Tobias Wagner 07/2023 to integrate <see cref="Mediapipe.Unity"/> package.

using Mediapipe.Unity;
using Mediapipe.Unity.FaceMesh;
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

    private HomulerEyeHelper _eyeHelper;
    private WebCamSource _webcam;

    private AOIBox _offscreenAOI;

    private RidgeRegression _xModel, _yModel;
    private MLP _mlp;
    private HomulerEyeMURunner _modelRunner;
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
    public HomulerEyeMURunner ModelRunner { get => _modelRunner; }
    public HomulerEyeHelper EyeHelper { get => _eyeHelper; }
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
    private Calibrations _calibrations = Calibrations.MLCalibration;
    public Calibrations Calibrations
    {
        get => _calibrations;
        set
        {
            //Append a note to csv entry if calibration changed
            if (Application.isPlaying && _csvLogger != null && _csvLogger.isActiveAndEnabled && value != _calibrations)
                _csvLogger.AppendNote($"Changed calibration type to {_calibrations}");

            _calibrations = value;
            switch (_calibrations)
            {
                case Calibrations.RidgeRegression:
                    _xModel = RidgeRegression.LoadX("Reg_X.json");
                    _yModel = RidgeRegression.LoadY("Reg_Y.json");
                    break;
                case Calibrations.MLCalibration:
                    MLP.Load("MLP.json");
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

    [SerializeField, Range(30, 120)] private int _frameRate = 30;

    private bool _isRendering;
    public bool IsRendering {
        get => _isRendering;
        private set
        {
            _isRendering = value;
            var solution = _mediaPipeGO.GetComponent<FaceMeshSolution>();
            solution.Annotate = _isRendering;
            solution.IsRendering = _isRendering;
        }
    }

    #endregion

    public virtual void Start()
    {
        Application.targetFrameRate = _frameRate;

        _webcam = _mediaPipeGO.GetComponent<WebCamSource>();
        //FaceMeshSolution instance
        FaceMeshSolution faceMesh = _mediaPipeGO.GetComponent<FaceMeshSolution>();

        //Create new EyeHelper
        _eyeHelper = new HomulerEyeHelper(faceMesh, _webcam.name);

        //Create new EyeMURunner
        _modelRunner = new HomulerEyeMURunner(faceMesh);

        //Load calibration files, suppress exceptions as they are handled internally
        switch (_calibrations)
        {
            case Calibrations.RidgeRegression:
                try { _xModel = RidgeRegression.LoadX("Reg_X.json"); } catch { }
                try { _yModel = RidgeRegression.LoadY("Reg_Y.json"); } catch { }
                break;
            case Calibrations.MLCalibration:
                try { _mlp = MLP.Load("MLP.json"); } catch { }
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
        if (!_modelRunner.PerformInference(_webcam)) 
            return;

        //Get gaze location from network output
        gazeLocation.x = _modelRunner.NetworkOutput[0];
        gazeLocation.y = _modelRunner.NetworkOutput[1];

        //Apply calibration
        gazeLocation = RefineGazeLocation(gazeLocation, _calibrations);

        //Apply filtering
        Vector2 unfilteredGaze = gazeLocation;
        gazeLocation = SmoothGazeLocation(gazeLocation, _filtering);

        //Update last gaze location timestamp
        var now = System.DateTime.Now;
        LastGazeLocationTimeUnix = ((System.DateTimeOffset)now).ToUnixTimeMilliseconds();

        //AOI updating
        aoiNameList = _aoiManager.CheckAOIList(new Vector2(gazeLocation.x / Screen.width, gazeLocation.y / Screen.height));

        //Drowsy, blinking and distance
        _drowsy = _eyeHelper.IsDrowsy();
        _blinking = _eyeHelper.IsBlinking();
        _distance = _eyeHelper.CalculateCamDistanceFocal();

        //CSV Logging
        if (!PauseCSVLogging && _csvLogger != null && _csvLogger.isActiveAndEnabled)
            _csvLogger.Append(new CSVData(gazeLocation.x, gazeLocation.y, gazeLocation.x / Screen.width, gazeLocation.y / Screen.height, unfilteredGaze.x / Screen.width, unfilteredGaze.y / Screen.height, _distance, _eyeHelper.EyeFeature(), _blinking, now, aoiNameList));

        //Drowsy calibration
        if (_eyeHelper.Calibrating)
            _eyeHelper.CalibrateDrowsyStats(false);

        //Unload Calibration if calibration is done
        if (_calibrationScript != null && _calibrationScript.Returned)
        {
            //Add one entry for the note because PauseCSVLogging is currently true
            if (_csvLogger != null && _csvLogger.isActiveAndEnabled)
                _csvLogger.Append(new CSVData(gazeLocation.x, gazeLocation.y, gazeLocation.x / Screen.width, gazeLocation.y / Screen.height, unfilteredGaze.x / Screen.width, unfilteredGaze.y / Screen.height, _distance, _eyeHelper.EyeFeature(), _blinking, now, aoiNameList));
            UnloadCalibration();
        }

        //Unload Evaluation if evaluation is done
        if (_evaluationScript != null && _evaluationScript.Returned)
            UnloadEvaluation();
    }

    public virtual void OnGUI()
    {
        //Draw eye textures on the GUI if they exist
        if (showEyes && _modelRunner?.LeftEyeTexture != null && _modelRunner?.RightEyeTexture != null)
        {
            GUI.DrawTexture(new Rect(Screen.width - 10, 10, -IMG_SIZE, IMG_SIZE), _modelRunner.LeftEyeTexture);

            GUI.DrawTexture(new Rect(10, 10, IMG_SIZE, IMG_SIZE), _modelRunner.RightEyeTexture);
        }

        //Draw crosshair on the GUI if one is selected
        if (drawDot && dot != null)
            GUI.DrawTexture(new Rect(gazeLocation.x - CROSSHAIR_SIZE / 2, gazeLocation.y - CROSSHAIR_SIZE / 2, CROSSHAIR_SIZE, CROSSHAIR_SIZE), dot);

        //Gaze UI
        if (showGazeUI && GUI.Button(new Rect(Screen.height * 0.05f, Screen.height - Screen.height * 0.1f, Screen.width * 0.1f, Screen.height * 0.05f), $"{(gazeUIActivated ? "Hide" : "Show")} Gaze UI"))
            gazeUIActivated = !gazeUIActivated;

        if (gazeUIActivated)
            gazeUI = GUI.Window(0, gazeUI, GazeUI, "");

        //Draw text
        if (visualizeAOI && aoiNameList != null && aoiNameList.Count > 0)
            GUI.Label(new Rect(200, 100, 500, 50), string.Join(", ", aoiNameList), style);
    }

    public virtual void OnDestroy()
    {
        // Must call Dispose method when no longer in use.
        _modelRunner?.Dispose();
        _modelRunner = null;
    }

    /// <summary>
    /// Refines the EyeMU gaze location by applying a calibrated model
    /// </summary>
    /// <param name="calibrations">The calibrated model type to use</param>
    /// <returns>The calibrated gaze location</returns>
    public Vector2 RefineGazeLocation(Vector2 rawGaze, Calibrations calibrations)
    {
        var features = _modelRunner.Features.ToArray();
        Vector2 refinedGaze = Vector2.zero;

        //Switch by calibration type
        switch (calibrations)
        {
            case Calibrations.None:
                refinedGaze = rawGaze;
                break;
            case Calibrations.RidgeRegression:
                if (_xModel == null || _yModel == null)
                    break;
                refinedGaze.x = _xModel.Predict(features);
                refinedGaze.y = _yModel.Predict(features);

                refinedGaze.x *= Screen.width;
                refinedGaze.y *= Screen.height;
                break;
            case Calibrations.MLCalibration:
                refinedGaze = _mlp.Predict(features);
                break;
        }
        //Default if calibration fails
        if (float.IsNaN(refinedGaze.x)) refinedGaze.x = 0.0f;
        if (float.IsNaN(refinedGaze.y)) refinedGaze.y = 0.0f;

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

        //Set GUIStyles
        var gazeUIStyleBox = GUI.skin.box;
        var gazeUIStyleButton = GUI.skin.button;
        var gazeUIStyleLabel = GUI.skin.label;
        var gazeUIStyleHSThumb = GUI.skin.horizontalSliderThumb;
        gazeUIStyleButton.wordWrap = gazeUIStyleLabel.wordWrap = true;

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
            _webcam.SelectSource(_webcam.GetCameraIndex() + 1);
            _eyeHelper.CameraChanged(_webcam.sourceName);
        }

        GUI.Label(new Rect(width * 0.18f, height * 0.025f, width * 0.12f, height * 0.05f), $"Current Webcam: {_webcam.sourceName}", gazeUIStyleLabel);

        if (GUI.Button(new Rect(width * 0.335f, height * 0.025f, width * 0.12f, height * 0.05f), $"Next Webcam", gazeUIStyleButton))
        {
            _webcam.SelectSource(_webcam.GetCameraIndex() - 1);
            _eyeHelper.CameraChanged(_webcam.sourceName);
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
            _eyeHelper.CalibrateFocalLength();

        GUI.Label(new Rect(width * 0.26f, height * 0.085f, width * 0.195f, height * 0.05f), $"Calculated distance: {_distance:F1} mm", gazeUIStyleLabel);

        GUI.EndGroup();

        //Drowsy and blinking calibration
        GUI.BeginGroup(new Rect(width * 0.01f, height * 0.34f, width * 0.48f, height * 0.13f));

        GUI.Box(new Rect(0, 0, width * 0.48f, height * 0.13f), "Blinking and Drowsiness Calibration", gazeUIStyleBox);

        GUI.Label(new Rect(width * 0.025f, height * 0.025f, width * 0.455f, height * 0.05f), "Calibrate blinking and drowsiness thresholds based on the eye aspect ratio. These values are saved between runs.", gazeUIStyleLabel);

        if (GUI.Button(new Rect(width * 0.025f, height * 0.07f, width * 0.1f, height * 0.05f), $"Calibrate Blinking Threshold", gazeUIStyleButton))
            _eyeHelper.CalibrateBlinking();

        GUI.Label(new Rect(width * 0.15f, height * 0.085f, width * 0.08f, height * 0.05f), $"{(_blinking ? "Eyes are closed" : "Eyes are open")}", gazeUIStyleLabel);

        if (GUI.Button(new Rect(width * 0.255f, height * 0.07f, width * 0.1f, height * 0.05f), $"{(_eyeHelper.Calibrating ? $"Calibrating Drowsiness based on {_eyeHelper.CalibrationCount} values" : "Calibrate Drowsiness Baseline")}", gazeUIStyleButton))
            _eyeHelper.CalibrateDrowsyStats(true);

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
