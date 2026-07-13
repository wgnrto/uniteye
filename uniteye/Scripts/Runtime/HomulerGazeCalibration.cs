using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using UnitEye;
using UnityEngine;
using UnityEngine.InputSystem;

namespace UnitEye
{

    /// <summary>
    /// This component is responsible for providing calibration to achieve better eye tracking performance.
    /// Without calibration, UnitEye is only able to provide gaze location from EyeMU which was trained on portrait mode smartphones.
    /// Therefore, uncalibrated gaze location is unprecise on desktop computers.
    /// Multiple calibration presets are used in this class to ensure as many areas as possible from the screen are used for training.
    /// </summary>
    // HomulerGaze has the default execution order (0). A small positive order (100) deliberately runs
    // calibration afterwards without imposing an order on unrelated host-game scripts, so each target uses
    // that frame's gaze/features rather than the prior frame's values.
    [DefaultExecutionOrder(100)]
    public class HomulerGazeCalibration : MonoBehaviour
    {
        #region Private

        //Frame rate the pixels-per-frame `speed` tuning assumed (same reference EaseSmoothing/KalmanFilter use).
        private const float ReferenceFrameRate = 30f;

        private HomulerGaze _gaze;

        private List<float[]> _xData = new List<float[]>();

        private List<float> _yXData = new List<float>();

        private List<float> _yYData = new List<float>();

        private List<Vector2> _yData = new List<Vector2>();
        private List<Vector2> _sampleTargets = new List<Vector2>();
        private List<bool> _sampleCapturedAtDwell = new List<bool>();

        private int _currentPoint = 0;

        private Vector2 _crossHairPos = Vector2.zero;

        private bool _isYielding = false;
        private float _currentTime;
        private long _lastCapturedGazeSample = -1;

        private GUIStyle _guiStyle = new GUIStyle();
        private GUIStyle _timerStyle = new GUIStyle();

        private List<CalibrationPreset> _presets;
        private int _currentPreset = 0;

        private bool _started = false;
        private bool _finished = false;
        private bool _finishedRound = false;
        private bool _earlyStop = false;
        private bool _showMessage = true;

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
        [Range(0f, 0.25f)]
        public float normalizedSafeMargin = 0.08f;
        [Range(1, 4)]
        public int cornerVisits = 2;
        [Range(1f, 6f)]
        public float cornerDwellSeconds = 3f;
        [Range(0.1f, 1f)]
        public float settleSeconds = 0.5f;
        [Range(5, 60)]
        public int minimumCornerSamples = 15;
        [Range(1f, 6f)]
        public float cornerOutlierZScore = 3f;

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
            //Also reset the run-position state and DISCARD any samples from a previous (possibly
            //cancelled) run. Without this, a mid-preset cancel left _currentPoint pointing into a longer
            //preset (points[_currentPoint] then throws every frame once the short first preset is
            //reloaded) and the old run's samples — possibly captured at a different seating position —
            //were silently mixed into the next training set, contaminating the model and its RMSE.
            _currentPoint = 0;
            _isYielding = false;
            _currentTime = 0f;
            _lastCapturedGazeSample = -1;
            _xData.Clear();
            _yXData.Clear();
            _yYData.Clear();
            _yData.Clear();
            _sampleTargets.Clear();
            _sampleCapturedAtDwell.Clear();
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

            //Calibration presets. CornerPreset leads and DWELLS at the four corners + four edge midpoints
            //(it's a StopAtWaypoints preset) so the extremes get sustained-fixation samples — without that
            //the fit had almost no leverage at the corners and collapsed predictions toward the centre. The
            //ZigZag + wavy presets follow as continuous sweeps for full-screen coverage (no dwell). Dwell is
            //now decided per preset (StopAtWaypoints), not one global flag.
            _presets = new List<CalibrationPreset>
            {
                new CornerPreset(padding, cornerVisits, cornerDwellSeconds, normalizedSafeMargin),
                new ZigZagPreset(padding, true, 4),
                new VerticalWavyPreset(padding),
                new HorizontalWavyPreset(padding),
            };

            //Master enable for dwelling; each preset's StopAtWaypoints then decides whether IT dwells.
            stopAfterPoints = true;
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

                    //Dwell only for presets that mark their waypoints as fixation targets (corners/edges),
                    //not the continuous sweeps whose ~150 path points would each pause 2s.
                    if (_currentPoint > 1 && stopAfterPoints && _presets[_currentPreset].StopAtWaypoints)
                    {
                        //Wait at the location so the eye settles and samples accumulate on the target
                        _isYielding = true;
                        _currentTime = _presets[_currentPreset].DwellSeconds;
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
                    //Move dot on screen. Frame-rate independent: `speed` is tuned in pixels-per-frame at
                    //the 30 fps reference (HomulerGaze sets targetFrameRate = 30, but vsync can override
                    //it), so scale by real elapsed time — otherwise a 144 Hz monitor sweeps the dot ~5x
                    //faster and each screen region contributes ~5x fewer training samples.
                    _crossHairPos =
                        Vector2.MoveTowards(_crossHairPos, points[_currentPoint], speed * Time.deltaTime * ReferenceFrameRate);
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
            //Only capture frames whose features are trustworthy. When the face is lost the provider's
            //feature buffer freezes at the last successful frame, so capturing would pair stale features
            //with a far-away label and contaminate the training set — gate on IsFacePresent + non-empty
            //features. NOTE: we deliberately do NOT gate on IsBlinking here: the blink test is an
            //eye-aspect-ratio threshold that false-positives on DOWNWARD gaze (the upper lid lowers), so
            //it was silently dropping bottom-edge / bottom-corner samples and starving exactly the region
            //that needs data. A few genuine blink frames out of thousands are negligible noise the fit
            //absorbs; systematically losing a screen region is not.
            var provider = _gaze != null ? _gaze.Provider : null;
            if (provider == null || !provider.IsFacePresent)
                return;
            if (_gaze.GazeSampleSequence <= 0 || _gaze.GazeSampleSequence == _lastCapturedGazeSample)
                return;
            var features = provider.GetFeatures();
            if (features == null || features.Length == 0)
                return;

            //During a dwell, skip the first ~0.3s: the dot just jumped to the waypoint and the eye is still
            //saccading to it, so those frames would pair the new (corner) label with mid-flight gaze.
            if (_isYielding && _currentTime > _presets[_currentPreset].DwellSeconds - settleSeconds)
                return;

            //Clone: GetFeatures() returns the provider's reused per-frame buffer, so the retained training
            //sample must be an owned copy (otherwise every captured sample would alias the latest frame).
            _xData.Add((float[])features.Clone());
            _yXData.Add(_crossHairPos.x / Screen.width);
            _yYData.Add(_crossHairPos.y / Screen.height);
            _yData.Add(new Vector2(_crossHairPos.x /*/ Screen.width*/, _crossHairPos.y /*/ Screen.height*/));
            _sampleTargets.Add(_crossHairPos);
            _sampleCapturedAtDwell.Add(_isYielding);
            _lastCapturedGazeSample = _gaze.GazeSampleSequence;
        }

        private string ProcessDataNeural()
        {
            Debug.Log("Starting MLP training");
            Debug.Log($"Total Count: {_xData.Count}");

            BuildBalancedTrainingData(out var features, out _, out _, out var targets);
            var mlp = new SimpleMLP();
            string MLPstring = mlp.Train(features, targets);

            if (save)
            {
                //Save under the active backbone's name so each gaze model keeps its own calibration.
                mlp.Save(CalibrationModelStore.FileName("MLP.json", _gaze.GazeBackbone));
            }

            return MLPstring;
        }

        private string ProcessData()
        {
            Debug.Log("Starting RidgeRegression training");

            BuildBalancedTrainingData(out var features, out var targetsX, out var targetsY, out _);
            var result = RidgeCalibrationTrainer.Train(
                features, targetsX, targetsY,
                rmseScaleX: Functions.PixelsToMm(Screen.width) * 0.1f,
                rmseScaleY: Functions.PixelsToMm(Screen.height) * 0.1f);

            Debug.Log($"Total Count: {_xData.Count}, Train Count: {result.TrainCount}, Test Count: {result.TestCount}, " +
                      $"Lambda X: {result.BestLambdaX}, Lambda Y: {result.BestLambdaY}");

            if (save)
            {
                Debug.Log("Saving best models");
                //Save under the active backbone's name so each gaze model keeps its own calibration.
                result.XModel.Save(CalibrationModelStore.FileName("Reg_X.json", _gaze.GazeBackbone));
                result.YModel.Save(CalibrationModelStore.FileName("Reg_Y.json", _gaze.GazeBackbone));
            }

            return $"RidgeRegression Training done. Best RMSE X: {result.XRmse}cm | Best RMSE Y: {result.YRmse}cm.";
        }

        private void BuildBalancedTrainingData(out float[][] features, out float[] targetsX,
            out float[] targetsY, out Vector2[] targets)
        {
            var keep = new bool[_xData.Count];
            for (var i = 0; i < keep.Length; i++)
                keep[i] = true;
            var dwellGroups = new Dictionary<Vector2, List<int>>();
            for (var i = 0; i < _xData.Count; i++)
            {
                if (!_sampleCapturedAtDwell[i] || !IsBoundaryTarget(_sampleTargets[i]))
                    continue;
                if (!dwellGroups.TryGetValue(_sampleTargets[i], out var group))
                {
                    group = new List<int>();
                    dwellGroups.Add(_sampleTargets[i], group);
                }
                group.Add(i);
            }

            var rejected = 0;
            foreach (var group in dwellGroups.Values)
            {
                if (group.Count < minimumCornerSamples)
                {
                    foreach (var index in group) keep[index] = false;
                    rejected += group.Count;
                    continue;
                }

                var featureCount = _xData[group[0]].Length;
                var mean = new double[featureCount];
                var variance = new double[featureCount];
                foreach (var index in group)
                    for (var feature = 0; feature < featureCount; feature++)
                        mean[feature] += _xData[index][feature];
                for (var feature = 0; feature < featureCount; feature++)
                    mean[feature] /= group.Count;
                foreach (var index in group)
                    for (var feature = 0; feature < featureCount; feature++)
                    {
                        var delta = _xData[index][feature] - mean[feature];
                        variance[feature] += delta * delta;
                    }
                for (var feature = 0; feature < featureCount; feature++)
                    variance[feature] = Math.Max(1e-8, variance[feature] / group.Count);

                foreach (var index in group)
                {
                    double meanZSquared = 0;
                    for (var feature = 0; feature < featureCount; feature++)
                    {
                        var delta = _xData[index][feature] - mean[feature];
                        meanZSquared += delta * delta / variance[feature];
                    }
                    if (Math.Sqrt(meanZSquared / featureCount) > cornerOutlierZScore)
                    {
                        keep[index] = false;
                        rejected++;
                    }
                }
            }

            var acceptedFeatures = new List<float[]>();
            var acceptedX = new List<float>();
            var acceptedY = new List<float>();
            var acceptedTargets = new List<Vector2>();
            for (var i = 0; i < _xData.Count; i++)
            {
                if (!keep[i]) continue;
                acceptedFeatures.Add(_xData[i]);
                acceptedX.Add(_yXData[i]);
                acceptedY.Add(_yYData[i]);
                acceptedTargets.Add(_yData[i]);
            }
            if (acceptedFeatures.Count == 0)
                throw new InvalidOperationException("No valid calibration samples remain after corner quality checks.");

            var selected = RidgeCalibrationTrainer.SpatiallyBalancedIndices(
                acceptedX, acceptedY, new System.Random(12345));
            features = new float[selected.Length][];
            targetsX = new float[selected.Length];
            targetsY = new float[selected.Length];
            targets = new Vector2[selected.Length];
            for (var i = 0; i < selected.Length; i++)
            {
                var index = selected[i];
                features[i] = acceptedFeatures[index];
                targetsX[i] = acceptedX[index];
                targetsY[i] = acceptedY[index];
                targets[i] = acceptedTargets[index];
            }
            Debug.Log($"Calibration kept {acceptedFeatures.Count}/{_xData.Count} raw samples " +
                $"({rejected} unstable/undersampled boundary samples rejected), spatially balanced to {features.Length}.");
        }

        private bool IsBoundaryTarget(Vector2 target)
        {
            var x = target.x / Screen.width;
            var y = target.y / Screen.height;
            return x <= 1f / 3f || x >= 2f / 3f || y <= 1f / 3f || y >= 2f / 3f;
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
}
