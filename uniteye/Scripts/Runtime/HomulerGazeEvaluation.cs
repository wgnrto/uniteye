using System;
using System.Collections.Generic;
using UnitEye;
using UnityEngine;
using UnityEngine.InputSystem;
namespace UnitEye
{

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
        //Cleared by the owner (HomulerGaze) once it has handled the return, so LateUpdate does not re-run
        //UnloadEvaluation every frame (which would stomp the UI toggles via RestoreSettings).
        public void ClearReturned() => Returned = false;
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
            //If finished and leftclick, signal Returned (new Input System, matching HomulerGazeCalibration)
            if (Mouse.current.leftButton.wasPressedThisFrame && returnAfter && _finished)
                Returned = true;
            //If rightclick, signal Returned
            if (Mouse.current.rightButton.wasPressedThisFrame && returnAfter)
                Returned = true;
            //If finished don't run through evaluation anymore
            if (_finished) return;

            //Start on leftclick
            if (Mouse.current.leftButton.wasPressedThisFrame && !_started)
            {
                _started = true;
                _showMessage = false;
                _isTimerRunning = true;
                _timeRemaining = duration;
            }

            //Stop evaluation early when pressing S
            if (Keyboard.current[Key.S].wasPressedThisFrame && _started)
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

        /// <summary>
        /// Root-mean-square gaze error in pixels, per axis.
        /// Metric note: a hit that lands within the calibration dot's radius counts as zero error (you
        /// cannot be more accurate than the dot itself), but that sample IS still counted in the average.
        /// So this is "mean error treating within-dot hits as perfect", NOT a plain RMSE — it reads lower
        /// than a plain RMSE by design. Change the denominator to the above-threshold count if you instead
        /// want the RMSE over only the misses.
        /// Uses the passed-in lists (not the _targetData field) so pred/target lengths stay in lockstep.
        /// </summary>
        private (float x, float y) CalculateRMSE(List<Vector2> predData, List<Vector2> targetData)
        {
            int count = Mathf.Min(predData.Count, targetData.Count);
            if (count == 0)
                return (0f, 0f);

            float errorX = 0.0f, errorY = 0.0f;
            float radiusSq = (dotSize * 0.5f) * (dotSize * 0.5f);
            for (int i = 0; i < count; i++)
            {
                float dx = predData[i].x - targetData[i].x;
                float dy = predData[i].y - targetData[i].y;
                float errX = dx * dx;
                float errY = dy * dy;

                // Since the dot is a circle, only count the error beyond its radius
                if (errX > radiusSq)
                    errorX += errX;
                if (errY > radiusSq)
                    errorY += errY;
            }

            return (Mathf.Sqrt(errorX / count), Mathf.Sqrt(errorY / count));
        }

        void OnGUI()
        {
            //Show message on screen. Scale the font with resolution so the message (and the final RMSE)
            //stays legible on high-DPI displays; == baseline at 1080p, larger above it.
            if (_showMessage)
            {
                float uiScale = Mathf.Max(1f, Mathf.Sqrt(0.001f * Screen.width * Screen.height / 2073.6f));
                _guiStyle.fontSize = Mathf.RoundToInt((_finished ? 16 : 36) * uiScale);
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
}
