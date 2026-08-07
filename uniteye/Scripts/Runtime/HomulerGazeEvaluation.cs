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
    // HomulerGaze has the default execution order (0). A small positive order (100) deliberately runs
    // evaluation afterwards without imposing an order on unrelated host-game scripts, so it scores the
    // fresh sample that corresponds to the displayed target.
    [DefaultExecutionOrder(100)]
    public class HomulerGazeEvaluation : MonoBehaviour
    {
        #region Private

        private bool _isTimerRunning;
        private float _timeRemaining;

        private Vector2 _targetLocation = Vector2.zero;
        private GUIStyle _guiStyle = new GUIStyle();
        private GUIStyle _timerStyle = new GUIStyle();
        private GUIStyle _heatmapStyle = new GUIStyle();
        //Per-target heatmap entries, aggregated ONCE when the evaluation finishes. OnGUI runs 2+ passes per
        //frame for as long as the results screen is up, so aggregating there allocated two dictionaries and
        //re-grouped every sample on every pass.
        private readonly List<(Vector2 target, Vector2 mean, Color color)> _heatmapEntries =
            new List<(Vector2 target, Vector2 mean, Color color)>();

        private List<Vector2> _points = new List<Vector2>();
        private List<CalibrationPreset> _presets;

        private List<Vector2> _predMLPData = new List<Vector2>();
        private List<Vector2> _predRidgeData = new List<Vector2>();
        private List<Vector2> _targetData = new List<Vector2>();
        private List<ScreenRegion> _targetRegions = new List<ScreenRegion>();

        private int _currentPoint;
        private long _lastCapturedGazeSample = -1;

        private bool _started = false;
        private bool _finished = false;
        private bool _earlyStop = false;
        private bool _showMessage = true;

        private string _guiMessage = "Click to start evaluation";

        private HomulerGaze _gaze;

        //Evaluation-owned model store with BOTH calibration types loaded (HomulerGaze's store only holds
        //the ACTIVE type, so evaluating "the other" model through it silently fell back to the input gaze
        //and its RMSE row was mislabeled). Loaded fresh on each evaluation start.
        private readonly CalibrationModelStore _evalStore = new CalibrationModelStore();
        private bool _hasMlpModel;
        private bool _hasRidgeModel;

        //Results-screen palette. The screen is drawn on an opaque WHITE backdrop (the scene behind it is
        //arbitrary game content, which used to make the error colours impossible to judge and the summary
        //hard to read), so every colour here is picked for contrast against white rather than against a
        //dark scene.
        private static readonly Color ResultsBackdrop = Color.white;
        private static readonly Color ResultsText = new Color(0.10f, 0.10f, 0.12f);
        private static readonly Color TargetMarker = new Color(0.20f, 0.20f, 0.22f, 0.9f);

        private enum ScreenRegion { Corner, Edge, Center }
        private const float RegionBoundaryThreshold = 1f / 3f;
        private const float RegionBoundaryUpperThreshold = 1f - RegionBoundaryThreshold;
        //Evaluation dot pulse rate (Hz); one pulse per second, matching the calibration dot animation.
        private const float EvalPulseHz = 1f;

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
        //Set when Evaluate() switched the active calibration to the better corner model. HomulerGaze must
        //re-apply it after RestoreSettings, which otherwise rewinds to the pre-evaluation backup and
        //silently undoes the model the results screen just reported as "Applied.".
        public Calibrations? AppliedCalibration { get; private set; }

        public Texture2D evaluationDot;

        public int duration = 4;

        public int padding = 40;
        [Range(0f, 0.25f)]
        public float normalizedSafeMargin = 0.08f;
        public int dotSize = 46;

        public int rows = 5;
        public int columns = 5;

        public bool showAllPoints = false;
        public bool applyBestCornerModel = true;
        public bool quitAfterEvaluation = false;
        [Tooltip("After the evaluation, draw arrows from each target to the mean measured gaze, colored by error, so you can see WHERE accuracy is good or bad.")]
        public bool showHeatmap = true;

        #endregion

        private void OnEnable()
        {
            //Reset per-session state so a repeat evaluation in the same play session starts fresh instead
            //of inheriting the previous run's finished/returned flags (which made the first click return
            //immediately) and its recorded samples (which would contaminate the new run's RMSE). Mirrors
            //HomulerGazeCalibration.OnEnable, which was added for exactly this bug. On the very first
            //enable this runs before Start(), which does the one-time setup.
            _started = false;
            _finished = false;
            _earlyStop = false;
            _showMessage = true;
            Returned = false;
            AppliedCalibration = null;
            _isTimerRunning = false;
            _timeRemaining = 0f;
            _currentPoint = 0;
            _lastCapturedGazeSample = -1;
            _predMLPData.Clear();
            _predRidgeData.Clear();
            _targetData.Clear();
            _targetRegions.Clear();
            _heatmapEntries.Clear();
            _guiMessage = "Click to start evaluation" + (returnAfter ? "\nRight click to cancel and return" : "");
        }

        void Start()
        {
            //OnGUI here draws only labels/textures (no GUILayout); skipping the layout pass halves the
            //per-frame IMGUI overhead during a run and while the results/heatmap screen is up.
            useGUILayout = false;

            //Get Gaze reference
            _gaze = GetComponent<HomulerGaze>();

            //If no crosshair is selected load the CalibrationDot Resource. Load it TYPED: the untyped
            //overload returns whichever asset named "CalibrationDot" it finds first (the folder holds both
            //a .png imported as a Sprite and a .svg), and casting that threw an InvalidCastException out of
            //Start() — which also skipped BuildPoints() below, leaving the whole evaluation dead.
            if (evaluationDot == null)
                evaluationDot = Resources.Load<Texture2D>("CalibrationDot");

            //If can return after evaluation append string to GUI
            if (returnAfter)
                _guiMessage += "\nRight click to cancel and return";

            //Initial point grid (rebuilt on start-click, when rows/columns are final)
            BuildPoints();
        }

        /// <summary>
        /// (Re)builds the evaluation dot grid from the CURRENT padding/rows/columns. Called on the
        /// start-click rather than only in Start(): LoadEvaluation sets rows/columns AFTER enabling the
        /// component, and Start() does not re-run on later enables, so building here is the only way those
        /// values are honored on every run.
        /// </summary>
        private void BuildPoints()
        {
            _points.Clear();
            _presets = new List<CalibrationPreset>
            {
                new EvaluationPreset(padding, rows, columns, normalizedSafeMargin)
            };
            foreach (var preset in _presets)
                _points.AddRange(preset.GetPoints());

            //Randomly shuffle list
            _points.Shuffle();

            _currentPoint = 0;
            _targetLocation = new Vector2(_points[0].x, _points[0].y);
        }

        // This must run after HomulerGaze.LateUpdate: both the fresh-sample sequence and the provider
        // values are updated there. Running in Update could pair a newly moved target with stale gaze.
        void LateUpdate()
        {
            //Mouse.current/Keyboard.current are NULL when no such device exists (headless players,
            //touch-only devices) and dereferencing them threw a NullReferenceException EVERY frame.
            var mouse = Mouse.current;
            bool leftClick = mouse != null && mouse.leftButton.wasPressedThisFrame;
            bool rightClick = mouse != null && mouse.rightButton.wasPressedThisFrame;

            //If finished and leftclick, signal Returned (new Input System, matching HomulerGazeCalibration)
            if (leftClick && returnAfter && _finished)
                Returned = true;
            //If rightclick, signal Returned
            if (rightClick && returnAfter)
                Returned = true;
            //If finished don't run through evaluation anymore
            if (_finished) return;

            //Start on leftclick
            if (leftClick && !_started)
            {
                //rows/columns/padding are final by now (LoadEvaluation sets them after enabling)
                BuildPoints();
                //Load BOTH calibration models for the active backbone into the evaluation's own store, so
                //each RMSE row measures the model it claims to (HomulerGaze's store only holds the active
                //type; the previous code silently evaluated the fallback for the other row).
                _evalStore.Load(Calibrations.RidgeRegression, _gaze.GazeBackbone);
                _evalStore.Load(Calibrations.MLCalibration, _gaze.GazeBackbone);
                _hasRidgeModel = _evalStore.HasModel(Calibrations.RidgeRegression);
                _hasMlpModel = _evalStore.HasModel(Calibrations.MLCalibration);

                _started = true;
                _showMessage = false;
                _isTimerRunning = true;
                _timeRemaining = duration;
            }

            //Stop evaluation early when pressing S (null when no keyboard device exists)
            var keyboard = Keyboard.current;
            if (keyboard != null && keyboard[Key.S].wasPressedThisFrame && _started)
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
                        //Evaluate each model on the provider's RAW gaze + features (what the models
                        //actually run on), NOT on _gaze.gazeLocation — that is already calibrated by the
                        //ACTIVE model and filtered, so it measured the live pipeline, not the model under
                        //test. No SmoothGazeLocation here either: it mutated the SAME stateful filter
                        //instances the live pipeline uses (three interleaved signals per frame corrupted
                        //both the on-screen gaze and these numbers). Samples are gated like the
                        //Calibration capture: no-face frames pair stale features with the dot's position
                        //and skew the RMSE. Do not gate on the EAR blink heuristic: downward gaze can
                        //look like a blink, and calibration intentionally retains those edge samples.
                        var provider = _gaze.Provider;
                        if (provider != null && provider.IsFacePresent &&
                            _gaze.GazeSampleSequence > 0 &&
                            _gaze.GazeSampleSequence != _lastCapturedGazeSample)
                        {
                            var features = provider.GetFeatures();
                            var raw = provider.RawGaze;
                            _predMLPData.Add(_evalStore.Refine(raw, Calibrations.MLCalibration, features, Screen.width, Screen.height));
                            _predRidgeData.Add(_evalStore.Refine(raw, Calibrations.RidgeRegression, features, Screen.width, Screen.height));
                            _targetData.Add(_targetLocation);
                            _targetRegions.Add(ClassifyRegion(_targetLocation));
                            _lastCapturedGazeSample = _gaze.GazeSampleSequence;
                        }
                    }
                }
                else
                {
                    //Reset for next point
                    _currentPoint++;
                    if (_currentPoint < _points.Count)
                    {
                        _timeRemaining = duration;
                        _targetLocation = new Vector2(_points[_currentPoint].x, _points[_currentPoint].y);
                    }
                }
            }

            //Only finish after the final target's duration increments _currentPoint from Count - 1 to Count.
            //The bounds guard above therefore protects the lookup after the final target; it does not skip it.
            if (_currentPoint >= _points.Count || _earlyStop)
            {
                _isTimerRunning = false;
                _finished = true;
                _showMessage = true;

                //Calculate errors
                _guiMessage = Evaluate();

                //Accuracy/precision decomposition + AOI-hit rates + persisted per-region error model.
                //This is the measurement that RANKS improvement work: bias-dominated error wants
                //calibration/drift/geometry effort, jitter-dominated error wants resolution/aggregation.
                _guiMessage += BuildErrorStatistics();

                //Aggregate the per-target heatmap once, here — not in OnGUI, which repeats 2+ passes/frame.
                BuildHeatmapEntries();

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

        private string Evaluate()
        {
            string message = $"Evaluation done.\nScreen size: {Functions.PixelsToMm(Screen.width) * 0.1f}x{Functions.PixelsToMm(Screen.height) * 0.1f}cm. Unity's built in DPI value might be wrong!\n";
            ReturnMessage = "";

            //All samples gated out (face never tracked) -> no data to score.
            if (_targetData.Count == 0)
            {
                ReturnMessage = "Evaluation captured no valid samples (was the face tracked?). ";
                return message + "No valid samples captured (was the face tracked?).\n";
            }

            //Only report an RMSE for models that were actually loaded — without a model, Refine falls
            //back to the raw gaze and the number would be the RAW pipeline's error mislabeled as the model's.
            string mlpLine = _hasMlpModel
                ? FormatRmseLine("MLP", _predMLPData)
                : $"MLP Evaluation: not calibrated for {_gaze.GazeBackbone} (no model file).";
            string ridgeLine = _hasRidgeModel
                ? FormatRmseLine("RidgeRegression", _predRidgeData)
                : $"RidgeRegression Evaluation: not calibrated for {_gaze.GazeBackbone} (no model file).";

            ReturnMessage += $"{mlpLine} {ridgeLine} ";
            message += $"{mlpLine}\n{ridgeLine}\n";

            if (_hasMlpModel && _hasRidgeModel)
            {
                var mlpCorner = CalculateRMSE(_predMLPData, ScreenRegion.Corner);
                var ridgeCorner = CalculateRMSE(_predRidgeData, ScreenRegion.Corner);
                var best = EuclideanError(mlpCorner) <= EuclideanError(ridgeCorner)
                    ? Calibrations.MLCalibration : Calibrations.RidgeRegression;
                var selection = $"Best corner model: {best}.";
                if (applyBestCornerModel)
                {
                    _gaze.Calibrations = best;
                    //Remembered so UnloadEvaluation can re-apply it after RestoreSettings; otherwise the
                    //user is told the better model is in use while the pre-evaluation one actually runs.
                    AppliedCalibration = best;
                    selection += " Applied.";
                }
                ReturnMessage += $" {selection}";
                message += $"{selection}\n";
            }

            return message;
        }

        private string FormatRmseLine(string label, List<Vector2> predictions)
        {
            var all = CalculateRMSE(predictions, null);
            var corner = CalculateRMSE(predictions, ScreenRegion.Corner);
            var edge = CalculateRMSE(predictions, ScreenRegion.Edge);
            var center = CalculateRMSE(predictions, ScreenRegion.Center);
            return $"{label} Evaluation: all {FormatError(all)}; corners {FormatError(corner)}; " +
                $"edges {FormatError(edge)}; center {FormatError(center)}.";
        }

        private static string FormatError((float x, float y) error)
            => $"X {Functions.PixelsToMm(error.x) * 0.1f:F2}cm, Y {Functions.PixelsToMm(error.y) * 0.1f:F2}cm";

        /// <summary>
        /// Root-mean-square gaze error in pixels, per axis. This intentionally matches the calibration
        /// holdout metric, independent of the visual dot size.
        /// An optional screen region exposes corner, edge, and centre accuracy independently.
        /// </summary>
        private (float x, float y) CalculateRMSE(List<Vector2> predData, ScreenRegion? region)
        {
            var count = Mathf.Min(predData.Count, _targetData.Count);
            var included = 0;
            if (count == 0)
                return (0f, 0f);

            float errorX = 0.0f, errorY = 0.0f;
            for (int i = 0; i < count; i++)
            {
                if (region.HasValue && _targetRegions[i] != region.Value)
                    continue;
                float dx = predData[i].x - _targetData[i].x;
                float dy = predData[i].y - _targetData[i].y;
                errorX += dx * dx;
                errorY += dy * dy;
                included++;
            }

            return included == 0 ? (0f, 0f) :
                (Mathf.Sqrt(errorX / included), Mathf.Sqrt(errorY / included));
        }

        private static float EuclideanError((float x, float y) error)
            => Mathf.Sqrt(error.x * error.x + error.y * error.y);

        private static ScreenRegion ClassifyRegion(Vector2 target)
        {
            var x = target.x / Screen.width;
            var y = target.y / Screen.height;
            var horizontalEdge = x <= RegionBoundaryThreshold || x >= RegionBoundaryUpperThreshold;
            var verticalEdge = y <= RegionBoundaryThreshold || y >= RegionBoundaryUpperThreshold;
            if (horizontalEdge && verticalEdge) return ScreenRegion.Corner;
            return horizontalEdge || verticalEdge ? ScreenRegion.Edge : ScreenRegion.Center;
        }

        void OnGUI()
        {
            //Draw behind HomulerGaze's own overlays (depth 0) but in front of the webcam preview (depth 5):
            //this component has the later execution order, so without a depth of its own the results
            //backdrop below would paint over the "Show Gaze UI" button and the crosshair.
            GUI.depth = 1;

            //The results screen gets an opaque WHITE backdrop. It used to be drawn straight over the live
            //scene, so the heatmap colours and the summary competed with whatever the game was rendering.
            if (_finished)
                GUIShapes.FillRect(new Rect(0f, 0f, Screen.width, Screen.height), ResultsBackdrop);

            //After the run, draw the accuracy heatmap (arrows from each target to the mean measured gaze,
            //colored by error) behind the summary text so you can see WHERE it is accurate vs off.
            if (_finished && showHeatmap)
                DrawResultsHeatmap();

            //Show message on screen. Scale the font with resolution so the message (and the final RMSE)
            //stays legible on high-DPI displays; == baseline at 1080p, larger above it.
            if (_showMessage)
            {
                float uiScale = Mathf.Max(1f, Mathf.Sqrt(0.001f * Screen.width * Screen.height / 2073.6f));
                _guiStyle.fontSize = Mathf.RoundToInt((_finished ? 16 : 36) * uiScale);
                if (_finished)
                    _guiStyle.normal.textColor = ResultsText;
                GUI.Label(new Rect(Screen.width / 2 - Screen.width * (_finished ? 0.15f : 0.1f), Screen.height / 2 - 20, 100, 60), $"{_guiMessage}", _guiStyle);
            }

            //Nothing below is part of the results screen: the dot marks the target the participant should
            //be looking at right now, and after the run there is none (the heatmap marks every target).
            if (evaluationDot != null && !_finished)
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

                // Expanding, fading pulse ring while a point is active, cueing the participant to hold a
                // steady fixation on the dot (matches the calibration dot animation and the reference HTML).
                if (_isTimerRunning)
                {
                    var prev = GUI.color;
                    float phase = Mathf.Repeat(Time.time * EvalPulseHz, 1f);
                    float ringSize = dotSize * (1f + phase * 1.6f);
                    GUI.color = new Color(prev.r, prev.g, prev.b, (1f - phase) * 0.55f);
                    GUI.DrawTexture(new Rect(_targetLocation.x - 0.5f * ringSize,
                            _targetLocation.y - 0.5f * ringSize, ringSize, ringSize), evaluationDot);
                    GUI.color = prev;
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

        /// <summary>
        /// The predictions to score, and the calibration they belong to. Every consumer (the heatmap, the
        /// accuracy/precision decomposition and the persisted error model's tag) must agree, and all must
        /// describe the calibration that will ACTUALLY be running once the evaluation unloads — not merely
        /// whichever model file happens to exist. The old "ridge if a ridge file exists" test meant that
        /// with MLCalibration active and a stale ridge file still on disk, the error model was measured
        /// from ridge predictions and tagged SourceCalibration=RidgeRegression; GazeErrorModel.AppliesTo is
        /// strict equality, so HomulerGaze could never apply it, and the heatmap described a model the user
        /// was not looking through. Reading _gaze.Calibrations is right here because Evaluate() has already
        /// applied the best corner model, and UnloadEvaluation now preserves that choice.
        /// </summary>
        private (List<Vector2> data, Calibrations source) SelectPredictions()
        {
            var active = _gaze != null ? _gaze.Calibrations : Calibrations.None;
            if (active == Calibrations.MLCalibration && _hasMlpModel)
                return (_predMLPData, Calibrations.MLCalibration);
            if (active == Calibrations.RidgeRegression && _hasRidgeModel)
                return (_predRidgeData, Calibrations.RidgeRegression);
            //Active type has no model of its own (e.g. Calibrations.None): fall back to whatever exists so
            //the results screen still shows something, tagged with the type it genuinely measures.
            if (_hasRidgeModel) return (_predRidgeData, Calibrations.RidgeRegression);
            if (_hasMlpModel) return (_predMLPData, Calibrations.MLCalibration);
            return (null, active);
        }

        /// <summary>
        /// Draws the post-evaluation accuracy heatmap: for each evaluated target, a line from the target to
        /// the MEAN measured gaze for that target (a hollow-ish marker at the target, a filled marker at the
        /// mean), colored green/amber/red by error as a fraction of the screen diagonal. Scores the
        /// calibration that will actually be running — see SelectPredictions.
        /// </summary>
        /// <summary>
        /// Aggregates the evaluation samples into per-target (target, mean gaze, error color) entries.
        /// Called once when the evaluation finishes; DrawResultsHeatmap then just renders the cached list.
        /// </summary>
        private void BuildHeatmapEntries()
        {
            _heatmapEntries.Clear();
            var (predictions, _) = SelectPredictions();
            if (predictions == null || predictions.Count == 0 || _targetData.Count == 0)
                return;

            int count = Mathf.Min(predictions.Count, _targetData.Count);
            var sum = new Dictionary<Vector2, Vector2>();
            var counts = new Dictionary<Vector2, int>();
            for (int i = 0; i < count; i++)
            {
                var target = _targetData[i];
                sum.TryGetValue(target, out var s);
                sum[target] = s + predictions[i];
                counts.TryGetValue(target, out var c);
                counts[target] = c + 1;
            }

            float diagonal = Mathf.Sqrt((float)Screen.width * Screen.width + (float)Screen.height * Screen.height);
            foreach (var pair in sum)
            {
                var mean = pair.Value / counts[pair.Key];
                _heatmapEntries.Add((pair.Key, mean, ErrorColor(Vector2.Distance(mean, pair.Key) / diagonal)));
            }
        }

        /// <summary>
        /// Computes the standard accuracy/precision/RMS-S2S decomposition per evaluation target on the
        /// model the heatmap shows (see SelectPredictions), reports AOI-hit rates at representative
        /// AOI sizes, and persists the per-region error model (bias + covariance per target) next to the
        /// calibration so the runtime AOI layer can turn hits into calibrated probabilities.
        /// </summary>
        private string BuildErrorStatistics()
        {
            var (predictions, source) = SelectPredictions();
            if (predictions == null || predictions.Count == 0 || _targetData.Count == 0)
                return "";

            //Group samples per target (targets repeat identically per dwell, so Vector2 keys are exact).
            int count = Mathf.Min(predictions.Count, _targetData.Count);
            var perTarget = new Dictionary<Vector2, List<Vector2>>();
            for (int i = 0; i < count; i++)
            {
                if (!perTarget.TryGetValue(_targetData[i], out var list))
                    perTarget[_targetData[i]] = list = new List<Vector2>();
                list.Add(predictions[i]);
            }

            var stats = new List<GazeStatistics.FixationStats>(perTarget.Count);
            //Tag the model with the calibration whose predictions it measures — its bias field must only
            //be applied at runtime while THAT calibration is active.
            var errorModel = new GazeErrorModel
            {
                SourceCalibration = source
            };
            float w = Screen.width, h = Screen.height;
            foreach (var pair in perTarget)
            {
                var s = GazeStatistics.Compute(pair.Value, pair.Key);
                stats.Add(s);
                //Anchor in normalized coords so the persisted model is resolution-independent.
                errorModel.AddAnchor(
                    new Vector2(pair.Key.x / w, pair.Key.y / h),
                    new Vector2(s.bias.x / w, s.bias.y / h),
                    (s.sd.x / w) * (s.sd.x / w), s.cov / (w * h), (s.sd.y / h) * (s.sd.y / h));
            }

            GazeStatistics.Aggregate(stats, out float accuracyPx, out _, out float precisionPx, out float whiteness);

            //AOI-hit rates: fraction of targets whose MEAN gaze lands inside a square AOI of the given
            //size centred on the target — the metric the shipped product (AOI logging) actually lives on.
            float pxPerCm = 10f / Mathf.Max(1e-3f, Functions.PixelsToMm(1f));
            string aoiLine = "AOI hit rate (mean-gaze in square AOI): ";
            foreach (float sizeCm in new[] { 3f, 5f, 8f })
            {
                float halfPx = sizeCm * pxPerCm * 0.5f;
                int hits = 0;
                foreach (var s in stats)
                    if (Mathf.Abs(s.bias.x) <= halfPx && Mathf.Abs(s.bias.y) <= halfPx)
                        hits++;
                aoiLine += $"{sizeCm:F0}cm {(100f * hits / stats.Count):F0}%  ";
            }

            //Persist the error model for the runtime AOI-probability layer + session-quality reporting.
            try { errorModel.Save(_gaze.GazeBackbone); }
            catch (System.Exception e) { UnitEyeLog.Exception(e); }

            //Whiteness ≈ 1.41 = white noise (fixation averaging pays ~sqrt(N)); much lower = colored
            //noise/drift (averaging saturates — spend effort on bias/drift instead).
            return $"Decomposition: accuracy(bias) {Functions.PixelsToMm(accuracyPx) * 0.1f:F2}cm, " +
                   $"precision(SD) {Functions.PixelsToMm(precisionPx) * 0.1f:F2}cm, " +
                   $"RMS-S2S/SD {whiteness:F2} (1.41=white noise).\n{aoiLine}\n";
        }

        private void DrawResultsHeatmap()
        {
            if (_heatmapEntries.Count == 0)
                return;

            float diagonal = Mathf.Sqrt((float)Screen.width * Screen.width + (float)Screen.height * Screen.height);
            float uiScale = Mathf.Max(1f, Mathf.Sqrt(0.001f * Screen.width * Screen.height / 2073.6f));
            float marker = 12f * uiScale;

            var previousColor = GUI.color;
            foreach (var (target, mean, color) in _heatmapEntries)
            {
                GUIShapes.DrawLine(target, mean, color, Mathf.Max(2f, 3f * uiScale));
                DrawMarker(target, marker, TargetMarker);  // where they were asked to look
                DrawMarker(mean, marker * 0.9f, color);    // where the gaze actually landed
            }
            GUI.color = previousColor;

            //Legend, top-left, its own style so it doesn't disturb the summary text's style.
            _heatmapStyle.fontSize = Mathf.RoundToInt(14 * uiScale);
            _heatmapStyle.normal.textColor = ResultsText;
            _heatmapStyle.wordWrap = true;
            GUI.Label(new Rect(Screen.width * 0.02f, Screen.height * 0.03f, Screen.width * 0.6f, Screen.height * 0.08f),
                $"Accuracy heatmap — line = target → mean gaze.  " +
                $"green < {0.02f * diagonal:F0}px   amber < {0.04f * diagonal:F0}px   red = worse", _heatmapStyle);
        }

        private static Color ErrorColor(float fractionOfDiagonal)
        {
            //Darker than the usual traffic-light triple: these are drawn on the white results backdrop,
            //where the light amber in particular was barely visible.
            if (fractionOfDiagonal < 0.02f) return new Color(0.10f, 0.60f, 0.20f);  // good
            if (fractionOfDiagonal < 0.04f) return new Color(0.85f, 0.50f, 0.00f);  // ok
            return new Color(0.80f, 0.12f, 0.12f);                                  // poor
        }

        private void DrawMarker(Vector2 center, float size, Color color)
        {
            if (evaluationDot == null) return;
            GUI.color = color;
            GUI.DrawTexture(new Rect(center.x - 0.5f * size, center.y - 0.5f * size, size, size), evaluationDot);
        }
    }
}
