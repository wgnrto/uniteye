using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using UnitEye;
using UnityEngine;

/// <summary>
/// This component is responsible for providing calibration to achieve better eye tracking performance.
/// Without calibration, UnitEye is only able to provide gaze location from EyeMU which was trained on portrait mode smartphones.
/// Therefore, uncalibrated gaze location is unprecise on desktop computers.
/// Multiple calibration presets are used in this class to ensure as many areas as possible from the screen are used for training.
/// </summary>
public class HomulerGazeCalibration : MonoBehaviour
{
    #region Private

    private HomulerEyeMURunner _modelRunner;

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
    //Default return message for cancellation
    public string ReturnMessage { get; private set; } = "Cancelled calibration";

    public Texture2D calibrationDot;

    public float speed = 6.0f;

    public float padding = 10.0f;

    public bool drawCheckpoints;

    public int currentRound = 0;

    public int maxRoundsPerPreset = 2;

    public Calibrations calibrationType = Calibrations.MLCalibration;
    public bool save = true;

    public bool stopAfterPoints = true;
    public bool quitAfterCalibration = false;

    public GameObject screen;

    public LineRenderer path;

    #endregion

    void Start()
    {
        //Get ModelRunner reference
        _modelRunner = GetComponent<HomulerGaze>().ModelRunner;

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
        //If no ModelRunner get a new reference, this is in case of Start() racing conditions
        if (_modelRunner == null)
            _modelRunner = GetComponent<HomulerGaze>().ModelRunner;

        //If finished and leftclick, signal Returned
        if (Input.GetKeyDown(KeyCode.Mouse0) && returnAfter && _finished)
            Returned = true;
        //If rightclick, signal Returned
        if (Input.GetKeyDown(KeyCode.Mouse1) && returnAfter)
            Returned = true;
        //Start on leftclick
        if (Input.GetKeyDown(KeyCode.Mouse0) && !_finished)
        {
            _started = true;
            _showMessage = false;
            _finishedRound = false;
        }
        //Stop calibration early when clicking S
        if (Input.GetKeyDown(KeyCode.S) && _started && !_finished)
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
        _xData.Add(_modelRunner.Features.ToArray());
        _yXData.Add(_crossHairPos.x / Screen.width);
        _yYData.Add(_crossHairPos.y / Screen.height);
        _yData.Add(new Vector2(_crossHairPos.x /*/ Screen.width*/, _crossHairPos.y /*/ Screen.height*/));
    }

    private string ProcessDataNeural()
    {
        Debug.Log("Starting MLP training");
        Debug.Log($"Total Count: {_xData.Count}");

        var mlp = new MLP();
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

        // Split data into training and test set
        var testCount = Mathf.FloorToInt(_xData.Count * 0.2f);
        Debug.Log($"Total Count: {_xData.Count}, Train Count: {_xData.Count - testCount}, Test Count: {testCount}");

        var rand = new System.Random();
        var randIndices = Enumerable.Range(0, testCount)
                                     .Select(i => new Tuple<int, int>(rand.Next(testCount), i))
                                     .OrderBy(i => i.Item1)
                                     .Select(i => i.Item2);

        var xTest = new List<float[]>(testCount);
        var yXTest = new List<float>(testCount);
        var yYTest = new List<float>(testCount);

        // Extract test data
        foreach (var index in randIndices)
        {
            xTest.Add(_xData[index]);
            yXTest.Add(_yXData[index]);
            yYTest.Add(_yYData[index]);
        }

        // Remove test data from full data
        foreach (var index in randIndices.OrderByDescending(v => v))
        {
            _xData.RemoveAt(index);
            _yXData.RemoveAt(index);
            _yYData.RemoveAt(index);
        }

        float bestXRMSE = float.MaxValue, bestYRMSE = float.MaxValue;
        RidgeRegression bestXModel = null, bestYModel = null;

        float[] lambdas = { 0.01f, 0.05f, 0.1f, 1.0f, 5.0f, 10.0f };

        var screenWidthToCm = Functions.PixelsToMm(Screen.width) * 0.1f;
        var screenHeightToCm = Functions.PixelsToMm(Screen.height) * 0.1f;

        // Find best x and y model
        foreach (var lambda in lambdas)
        {
            var xModel = new RidgeRegression(lambda);
            xModel.Train(_xData.ToArray(), _yXData.ToArray());
            var xRMSE = CalculateRMSE(xModel, xTest.ToArray(), yXTest.ToArray(), screenWidthToCm);

            var yModel = new RidgeRegression(lambda);
            yModel.Train(_xData.ToArray(), _yYData.ToArray());
            var yRMSE = CalculateRMSE(yModel, xTest.ToArray(), yYTest.ToArray(), screenHeightToCm);

            Debug.Log($"Lambda: {lambda}, MSE X: {xRMSE}, MSE Y: {yRMSE}");

            if (xRMSE < bestXRMSE)
            {
                bestXRMSE = xRMSE;
                bestXModel = xModel;
            }
            if (yRMSE < bestYRMSE)
            {
                bestYRMSE = yRMSE;
                bestYModel = yModel;
            }
        }

        if (save)
        {
            Debug.Log("Saving best models");
            bestXModel.Save("Reg_X.json");
            bestYModel.Save("Reg_Y.json");
        }

        return $"RidgeRegression Training done. Best RMSE X: {bestXRMSE}cm | Best RMSE Y: {bestYRMSE}cm.";
    }

    private float CalculateRMSE(RidgeRegression model, float[][] X, float[] Y, float factor)
    {
        var error = 0.0f;
        for (int i = 0; i < Y.Length; i++)
        {
            var yhat = model.Predict(X[i]);
            error += MathF.Pow(Y[i] * factor - yhat * factor, 2);
        }

        return MathF.Sqrt(error / Y.Length);
    }

    private void OnGUI()
    {
        //Show message on screen
        if (_showMessage)
        {
            _guiStyle.fontSize = _finished ? 16 : 36;
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
