using System;
using System.Collections.Generic;
using UnitEye;
using UnityEngine;

/// <summary>
/// This component is responsible for evaluating the UnitEye eye tracking.
/// The user is supposed to look at each appearing dot.
/// </summary>
public class HomulerGazeEvaluation : MonoBehaviour
{
    #region Private

    private bool _isTimerRunning;
    private float _timeRemaining;

    private Vector2 _targetLocation = Vector2.zero;
    private GUIStyle _guiStyle = new GUIStyle();
    private GUIStyle _timerStyle = new GUIStyle();

    private List<Vector2> _points = new List<Vector2>();
    private List<CalibrationPreset> _presets;

    private List<Vector2> _predMLPData = new List<Vector2>();
    private List<Vector2> _predRidgeData = new List<Vector2>();
    private List<Vector2> _targetData = new List<Vector2>();

    private int _currentPoint;

    private bool _started = false;
    private bool _finished = false;
    private bool _earlyStop = false;
    private bool _showMessage = true;

    private string _guiMessage = "Click to start evaluation";

    private HomulerGaze _gaze;

    #endregion

    #region Public

    [NonSerialized]
    public bool returnAfter;
    public bool Returned { get; private set; }
    //Default return message for cancellation
    public string ReturnMessage { get; private set; } = "Cancelled evaluation";

    public Texture2D evaluationDot;

    public int duration = 4;

    public int padding = 40;
    public int dotSize = 46;

    public int rows = 5;
    public int columns = 5;

    public bool showAllPoints = false;
    public bool quitAfterEvaluation = false;

    #endregion

    void Start()
    {
        //Get Gaze reference
        _gaze = GetComponent<HomulerGaze>();

        //If no crosshair is selected load the CalibrationDot Resource
        if (evaluationDot == null)
            evaluationDot = (Texture2D)Resources.Load("CalibrationDot");

        //If can return after evaluation append string to GUI
        if (returnAfter)
            _guiMessage += "\nRight click to cancel and return";

        //Create new preset with padding, rows and columns
        _presets = new List<CalibrationPreset> { new EvaluationPreset(padding, rows, columns) };

        //Add points
        foreach (var preset in _presets)
        {
            _points.AddRange(preset.GetPoints());
        }

        //Randomly shuffle list
        _points.Shuffle();

        _targetLocation = new Vector2(_points[_currentPoint].x, _points[_currentPoint].y);
    }

    void Update()
    {
        //If finished and leftclick, signal Returned
        if (Input.GetKeyDown(KeyCode.Mouse0) && returnAfter && _finished)
            Returned = true;
        //If rightclick, signal Returned
        if (Input.GetKeyDown(KeyCode.Mouse1) && returnAfter)
            Returned = true;
        //If finished don't run through evaluation anymore
        if (_finished) return;

        //Start on leftclick
        if (Input.GetKeyDown(KeyCode.Mouse0) && !_started)
        {
            _started = true;
            _showMessage = false;
            _isTimerRunning = true;
            _timeRemaining = duration;
        }

        //Stop evaluation early when clicking S
        if (Input.GetKeyDown(KeyCode.S) && _started)
        {
            _earlyStop = true;
        }

        if (_isTimerRunning)
        {
            if (_timeRemaining > 0)
            {
                //Reduce time by frametime
                _timeRemaining -= Time.deltaTime;

                var threeQuarterDuration = duration * 0.75f;
                var oneQuarterDuration = duration * 0.25f;
                //Only take data between one and three quarter duration
                if (_timeRemaining >= oneQuarterDuration && _timeRemaining <= threeQuarterDuration)
                {
                    // Calculate gaze for MLP and RidgeRegression and current filtering method
                    var rawGaze = _gaze.gazeLocation;
                    _predMLPData.Add(CalculateGaze(rawGaze, Calibrations.MLCalibration, _gaze.Filtering));
                    _predRidgeData.Add(CalculateGaze(rawGaze, Calibrations.RidgeRegression, _gaze.Filtering));
                    _targetData.Add(_targetLocation);
                }
            }
            else
            {
                //Reset for next point
                _timeRemaining = duration;
                _currentPoint++;
                _targetLocation = new Vector2(_points[_currentPoint].x, _points[_currentPoint].y);
            }
        }

        //If done with all the points or if we want to stop early, finish evaluation
        if (_currentPoint == _points.Count - 1 || _earlyStop)
        {
            _isTimerRunning = false;
            _finished = true;
            _showMessage = true;

            //Calculate errors
            _guiMessage = Evaluate();

            //Append return hint to GUI
            if (returnAfter)
                _guiMessage += $"Click to return.";

            //Write message to debug
            Debug.Log(_guiMessage);

            //Quit if wanted
            if (quitAfterEvaluation) Functions.Quit();

            _gaze.showGazeUI = true;
        }
    }

    private Vector2 CalculateGaze(Vector2 gazeLocation, Calibrations calibrations, Filtering filtering)
    {
        var result = gazeLocation;
        result = _gaze.RefineGazeLocation(result, calibrations);
        result = _gaze.SmoothGazeLocation(result, filtering);

        return result;
    }

    private string Evaluate()
    {
        var mlpError = CalculateRMSE(_predMLPData, _targetData);
        var regError = CalculateRMSE(_predRidgeData, _targetData);

        string message = $"Evaluation done.\nScreen size: {Functions.PixelsToMm(Screen.width) * 0.1f}x{Functions.PixelsToMm(Screen.height) * 0.1f}cm. Unity's built in DPI value might be wrong!\n";
        ReturnMessage = "";

        ReturnMessage += $"MLP Evaluation: RMSE X: {Functions.PixelsToMm(mlpError.x) * 0.1f}cm | RMSE Y: {Functions.PixelsToMm(mlpError.y) * 0.1f}cm. ";
        message += $"MLP Evaluation: RMSE X: {Functions.PixelsToMm(mlpError.x) * 0.1f}cm | RMSE Y: {Functions.PixelsToMm(mlpError.y) * 0.1f}cm.\n";
        ReturnMessage += $"RidgeRegression Evaluation: RMSE X: {Functions.PixelsToMm(regError.x) * 0.1f}cm | RMSE Y: {Functions.PixelsToMm(regError.y) * 0.1f}cm. ";
        message += $"RidgeRegression Evaluation: RMSE X: {Functions.PixelsToMm(regError.x) * 0.1f}cm | RMSE Y: {Functions.PixelsToMm(regError.y) * 0.1f}cm.\n";

        return message;
    }

    private (float x, float y) CalculateRMSE(List<Vector2> predData, List<Vector2> targetData)
    {
        float errorX = 0.0f, errorY = 0.0f;
        for (int i = 0; i < _targetData.Count; i++)
        {
            var errX = Mathf.Pow(predData[i].x - targetData[i].x, 2);
            var errY = Mathf.Pow(predData[i].y - targetData[i].y, 2);

            // Since the dot is a circle check its radius
            if (errX > Mathf.Pow(dotSize * 0.5f, 2))
                errorX += errX;
            if (errY > Mathf.Pow(dotSize * 0.5f, 2))
                errorY += errY;
        }

        return (Mathf.Sqrt(errorX / _targetData.Count), Mathf.Sqrt(errorY / _targetData.Count));
    }

    void OnGUI()
    {
        //Show message on screen
        if (_showMessage)
        {
            _guiStyle.fontSize = _finished ? 16 : 36;
            GUI.Label(new Rect(Screen.width / 2 - Screen.width * (_finished ? 0.15f : 0.1f), Screen.height / 2 - 20, 100, 60), $"{_guiMessage}", _guiStyle);
        }

        if (evaluationDot != null)
        {
            // Draw faded out points
            if (showAllPoints)
            {
                var oldColor = GUI.color;
                GUI.color = new Color(oldColor.r, oldColor.g, oldColor.b, 0.2f);
                foreach (var point in _points)
                {
                    GUI.DrawTexture(new Rect(point.x - 0.5f * dotSize,
                        point.y - 0.5f * dotSize,
                        dotSize,
                        dotSize),
                    evaluationDot);
                }
                GUI.color = oldColor;
            }

            // Draw calibration dot
            GUI.DrawTexture(new Rect(_targetLocation.x - 0.5f * dotSize,
                    _targetLocation.y - 0.5f * dotSize,
                    dotSize,
                    dotSize),
                evaluationDot);

            // Draw countdown
            if (_isTimerRunning)
            {
                _timerStyle.fixedHeight = _timerStyle.fixedWidth = dotSize;
                _timerStyle.normal.textColor = Color.red;
                _timerStyle.alignment = TextAnchor.MiddleCenter;
                GUI.Label(new Rect(_targetLocation.x - 0.5f * dotSize,
                        _targetLocation.y - 0.5f * dotSize,
                        dotSize,
                        dotSize), String.Format("{0}s", Mathf.FloorToInt((_timeRemaining + 1) % 60)), _timerStyle);
            }
        }
    }
}
