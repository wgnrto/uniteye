using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using UnitEye;
using UnityEngine;
using UnityEngine.InputSystem;

/// <summary>
/// This component is responsible for providing calibration to achieve better eye tracking performance.
/// Without calibration, UnitEye is only able to provide gaze location from EyeMU which was trained on portrait mode smartphones.
/// Therefore, uncalibrated gaze location is unprecise on desktop computers.
/// Multiple calibration presets are used in this class to ensure as many areas as possible from the screen are used for training.
/// </summary>
public class HomulerGazeCalibration : MonoBehaviour
{
    #region Private

    private HomulerGaze _gaze;

    private List<float[]> _xData = new List<float[]>();

    private List<float> _yXData = new List<float>();

    private List<float> _yYData = new List<float>();

    private List<Vector2> _yData = new List<Vector2>();

    private int _currentPoint = 0;

    private Vector2 _crossHairPos = Vector2.zero;

    private bool _isYielding = false;
    private float _currentTime;

    private GUIStyle _guiStyle = new GUIStyle();
    private GUIStyle _timerStyle = new GUIStyle();

    private List<CalibrationPreset> _presets;
    private int _currentPreset = 0;

    private bool _started = false;
    private bool _finished = false;
    private bool _finishedRound = false;
    private bool _earlyStop = false;
    private bool _showMessage = true;
    private bool _stop = false;

    private string _guiMessage = "Follow the dot with your eyes!\nClick to start calibration";

    #endregion

    #region Public

    public List<Vector2> points = new List<Vector2>();

    [NonSerialized]
    public bool returnAfter;
    public bool Returned { get; private set; }
    //Cleared by the owner (HomulerGaze) once it has handled the return. Without this, Returned stays
    //true forever and HomulerGaze.LateUpdate re-runs UnloadCalibration every frame, whose RestoreSettings
    //stomps the UI toggles ~30x/second (they appear to flip back instantly when clicked).
    public void ClearReturned() => Returned = false;
    //Default return message for cancellation
    public string ReturnMessage { get; private set; } = "Cancelled calibration";

    public Texture2D calibrationDot;

    public float speed = 6.0f;

    public float padding = 10.0f;

    public bool drawCheckpoints;

    public int currentRound = 0;

    public int maxRoundsPerPreset = 2;

    public Calibrations calibrationType = Calibrations.RidgeRegression;
    public bool save = true;

    public bool stopAfterPoints = true;
    public bool quitAfterCalibration = false;

    public GameObject screen;

    public LineRenderer path;

    #endregion

    private void OnEnable()
    {
        //Reset per-session state so a repeat calibration (e.g. switching calibration type and running
        //again in the same play session) starts fresh instead of inheriting the previous run's
        //finished/returned flags (which would make it abort immediately). On the very first enable this
        //runs before Start(), when _presets is still null, so the point reset is guarded and Start()
        //performs the initial point setup.
        _started = false;
        _finished = false;
        _finishedRound = false;
        _earlyStop = false;
        _showMessage = true;
        Returned = false;
        currentRound = 0;
        _guiMessage = "Follow the dot with your eyes!\nClick to start calibration";
        if (_presets != null)
        {
            _currentPreset = 0;
            ResetPoints(0);
        }
    }

    void Start()
    {
        //Get HomulerGaze reference (features come from its platform gaze provider)
        _gaze = GetComponent<HomulerGaze>();

        //If no crosshair is selected load the CalibrationDot Resource
        if (calibrationDot == null)
            calibrationDot = (Texture2D)Resources.Load("CalibrationDot");

        //If can return after calibration append string to gui
        if (returnAfter)
            _guiMessage += "\nRight click to cancel and return";

        //Create new preset with all the wanted rounds
        _presets = new List<CalibrationPreset>
        {
            new ZigZagPreset(padding, true, 4),
            new VerticalWavyPreset(padding, out _stop),
            new HorizontalWavyPreset(padding, out _stop)
            //new CornerPreset(padding),
            //new ZigZagPreset(padding, true, 4),
            //new ZigZagPreset(padding, false, 4),
            //new CornerPreset(padding, mirrored: true) 
        };

        //Reset for first point
        stopAfterPoints = _stop;
        ResetPoints(0);

        path = Instantiate(path, new Vector3(Screen.width/2, -Screen.height/2), Quaternion.identity, screen.transform);

        DrawPath(points);
    }

    private void ResetPoints(int currentPreset)
    {
        points = _presets[currentPreset].GetPoints();
        _crossHairPos = points[0];
    }

    /// <summary>
    /// Renders a line along the given waypoints. TODO: adjust coordinate conversion, currently not working correctly
    /// </summary>
    /// <param name="waypoints">A list of Vector2 objects containing the local screen coordinates.</param>
    private void DrawPath(List<Vector2> waypoints)
    {
        path.transform.localPosition = new Vector3(Screen.width * 0.5f, -(Screen.height * 0.5f), -1f);

        var lineRendererPositions = new Vector3[waypoints.Count];
        for (int i = 0; i < waypoints.Count; i++)
            lineRendererPositions[i] = new Vector3(waypoints[i].x, waypoints[i].y, transform.position.z);

        path.positionCount = lineRendererPositions.Length;
        path.SetPositions(lineRendererPositions);
    }

    void Update()
    {
        //If no HomulerGaze reference yet, get one, this is in case of Start() racing conditions
        if (_gaze == null)
            _gaze = GetComponent<HomulerGaze>();

        //If finished and leftclick, signal Returned
        if (Mouse.current.leftButton.wasPressedThisFrame && returnAfter && _finished)
            Returned = true;
        //If rightclick, signal Returned
        if (Mouse.current.rightButton.wasPressedThisFrame && returnAfter)
            Returned = true;
        //Start on leftclick
        if (Mouse.current.leftButton.wasPressedThisFrame && !_finished)
        {
            _started = true;
            _showMessage = false;
            _finishedRound = false;
        }
        //Stop calibration early when clicking S
        if (Keyboard.current[Key.S].wasPressedThisFrame && _started && !_finished)
        {
            _earlyStop = true;
        }
    }

    void LateUpdate()
    {
        //Abort if finished or not started
        if (!_started) return;
        if (_finished) return;

        // Only move if we are not currently pausing
        if (!_isYielding)
        {
            var pointReached = _crossHairPos.Equals(points[_currentPoint]);

            if (pointReached)
            {
                _currentPoint++;

                if (_currentPoint > 1 && stopAfterPoints)
                {
                    //Wait at the location for 2 seconds
                    _isYielding = true;
                    _currentTime = 2;
                }

                if (_currentPoint >= points.Count)
                {
                    //If finished with round reset for next
                    _currentPoint = 0;
                    _finishedRound = true;
                }
            }

            if (!_isYielding)
            {
                //Move dot on screen
                _crossHairPos =
                    Vector2.MoveTowards(_crossHairPos, points[_currentPoint], speed);
            }
        }
        else
        {
            //If waiting reduce time until 0
            _currentTime -= Time.deltaTime;
            if (_currentTime <= 0)
            {
                _isYielding = false;
            }
        }

        //Add data from raw neural network output
        CaptureNetworkOutput();

        if (_finishedRound)
        {
            //If finished current round reset for next round
            currentRound++;
            _currentPreset = _currentPreset >= _presets.Count - 1 ? 0 : _currentPreset + 1;
            ResetPoints(_currentPreset);
            _started = false;
            _showMessage = true;
            _guiMessage = $"Click to start next round\nRound {currentRound}/{_presets.Count * maxRoundsPerPreset}\nRight click to cancel calibration and return";
            _isYielding = false;
            DrawPath(points);
        }

        //If done with all rounds or if we want to stop early, finish calibration
        if (currentRound == _presets.Count * maxRoundsPerPreset || _earlyStop)
        {
            //If all rounds are done start training with GUI message
            _guiMessage = "Starting training. This can take a while, please be patient!";
            _showMessage = true;
            _finished = true;

            //Use a coroutine to start in the next frame to allow OnGUI() to run once.
            StartCoroutine(Training());
        }
    }

    System.Collections.IEnumerator Training()
    {
        //Yield until next frame
        yield return 0;

        //Prepare GUI message
        var message = $"Calibration done!\nScreen size: {Functions.PixelsToMm(Screen.width) * 0.1f}x{Functions.PixelsToMm(Screen.height) * 0.1f}cm. Unity's built in DPI value might be wrong!\n";

        //Process data by calibration type
        switch (calibrationType)
        {
            case Calibrations.RidgeRegression:
                ReturnMessage = $"{ProcessData()} ";
                message += $"{ReturnMessage}\n";
                break;
            case Calibrations.MLCalibration:
                ReturnMessage = $"{ProcessDataNeural()} ";
                message += $"{ReturnMessage}\n";
                break;
            default:
                break;
        }

        //Append return hint to GUI
        if (returnAfter)
            message += $"Click to return.";

        //Write message to debug
        _guiMessage = message;
        Debug.Log(_guiMessage);

        //Quit if wanted
        if (quitAfterCalibration) Functions.Quit();

        GetComponent<HomulerGaze>().showGazeUI = true;
    }

    private void CaptureNetworkOutput()
    {
        _xData.Add(_gaze.Provider.GetFeatures());
        _yXData.Add(_crossHairPos.x / Screen.width);
        _yYData.Add(_crossHairPos.y / Screen.height);
        _yData.Add(new Vector2(_crossHairPos.x /*/ Screen.width*/, _crossHairPos.y /*/ Screen.height*/));
    }

    private string ProcessDataNeural()
    {
        Debug.Log("Starting MLP training");
        Debug.Log($"Total Count: {_xData.Count}");

        var mlp = new SimpleMLP();
        string MLPstring = mlp.Train(_xData.ToArray(), _yData.ToArray());

        if (save)
        {
            mlp.Save("MLP.json");
        }

        return MLPstring;
    }

    private string ProcessData()
    {
        Debug.Log("Starting RidgeRegression training");

        var result = RidgeCalibrationTrainer.Train(
            _xData, _yXData, _yYData,
            rmseScaleX: Functions.PixelsToMm(Screen.width) * 0.1f,
            rmseScaleY: Functions.PixelsToMm(Screen.height) * 0.1f);

        Debug.Log($"Total Count: {_xData.Count}, Train Count: {result.TrainCount}, Test Count: {result.TestCount}, " +
                  $"Lambda X: {result.BestLambdaX}, Lambda Y: {result.BestLambdaY}");

        if (save)
        {
            Debug.Log("Saving best models");
            result.XModel.Save("Reg_X.json");
            result.YModel.Save("Reg_Y.json");
        }

        return $"RidgeRegression Training done. Best RMSE X: {result.XRmse}cm | Best RMSE Y: {result.YRmse}cm.";
    }

    private void OnGUI()
    {
        //Show message on screen. Scale the font with resolution so the message (and the final RMSE)
        //stays legible on high-DPI displays; == baseline at 1080p, larger above it.
        if (_showMessage)
        {
            float uiScale = Mathf.Max(1f, Mathf.Sqrt(0.001f * Screen.width * Screen.height / 2073.6f));
            _guiStyle.fontSize = Mathf.RoundToInt((_finished ? 16 : 36) * uiScale);
            GUI.Label(new Rect(Screen.width / 2 - Screen.width * (_finished ? 0.15f : 0.1f), Screen.height / 2 - 20, 100, 60), $"{_guiMessage}", _guiStyle);
        }

        var size = 36;
        if (calibrationDot != null)
        {
            // Draw faded out checkpoints
            var oldColor = GUI.color;
            //GUI.color = new Color(oldColor.r, oldColor.g, oldColor.b, 0.2f);
            if (drawCheckpoints)
            {
                foreach (var point in points)
                {
                    GUI.DrawTexture(new Rect(point.x - 0.5f * size,
                        point.y - 0.5f * size,
                        size,
                        size),
                    calibrationDot);
                }
            }
            GUI.color = oldColor;

            // Draw calibration dot
            GUI.DrawTexture(new Rect(_crossHairPos.x - 0.5f * size,
                    _crossHairPos.y - 0.5f * size,
                    size,
                    size),
                calibrationDot);

            // Draw countdown
            if (_isYielding)
            {
                _timerStyle.fixedHeight = _timerStyle.fixedWidth = size;
                _timerStyle.normal.textColor = Color.red;
                _timerStyle.alignment = TextAnchor.MiddleCenter;
                GUI.Label(new Rect(_crossHairPos.x - 0.5f * size,
                        _crossHairPos.y - 0.5f * size,
                        size,
                        size), String.Format("{0}s", Mathf.FloorToInt((_currentTime + 1) % 60)), _timerStyle);
            }
        }
    }
}
