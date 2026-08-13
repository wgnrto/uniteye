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
        //Prevents zero-variance fixation features from producing undefined z-scores.
        //Keeps spatial resampling reproducible across the Ridge and MLP calibration paths.

        private HomulerGaze _gaze;

        //Optional consented dataset recording. Both stay null unless a CalibrationRecordingConsent component
        //is present AND the participant opted in, so the ordinary calibration path is untouched.
        private CalibrationRecordingConsent _consentGate;
        private GazeSessionRecorder _recorder;
        private double _recordingStartedAt;

        private List<float[]> _xData = new List<float[]>();

        private List<float> _yXData = new List<float>();

        private List<float> _yYData = new List<float>();

        private List<Vector2> _yData = new List<Vector2>();
        private List<Vector2> _sampleTargets = new List<Vector2>();
        private List<bool> _sampleCapturedAtDwell = new List<bool>();
        //True for samples captured during the head-movement stage. They are DELIBERATELY high-variance
        //(head yaw/pitch/roll swing while the eye fixates one target), so they must bypass the corner
        //stability rejection in BuildBalancedTrainingData — that rejection exists to drop unstable corner
        //fixations, and would otherwise throw away exactly the head-pose variance this stage collects.
        private List<bool> _sampleFromHeadRotation = new List<bool>();

        //Recent dot positions (time-stamped) for pursuit-lag label correction: while the dot sweeps, the eye
        //trails it by ~100ms, so pairing the CURRENT dot position with the current gaze bakes a systematic
        //error along the sweep direction into every moving sample. Sweep samples are instead labeled with
        //the dot position pursuitLagSeconds ago. Kept trimmed to well under a second of history.
        private readonly List<(float time, Vector2 pos)> _dotTrail = new List<(float time, Vector2 pos)>();
        //Recent raw-gaze samples (time-stamped) for the dwell fixation gate.
        private readonly List<(float time, Vector2 pos)> _gazeTrail = new List<(float time, Vector2 pos)>();
        //Per-dwell fixation-gate accounting; the gate bypasses itself mid-dwell rather than starve a target.
        private int _dwellGateAccepts, _dwellGateRejects;
        private bool _dwellGateBypassed;
        private bool _warnedGateBypass;

        private int _currentPoint = 0;

        private Vector2 _crossHairPos = Vector2.zero;

        private bool _isYielding = false;
        private float _currentTime;
        private long _lastCapturedGazeSample = -1;

        private GUIStyle _guiStyle = new GUIStyle();
        private GUIStyle _timerStyle = new GUIStyle();
        private GUIStyle _headRotationStyle = new GUIStyle();
        //Dot pulse rate (Hz). One pulse per second matches the reference HTML's 1s target animation.
        private const float PulseHz = 1f;

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

        /// <summary>
        /// Full-screen backdrop drawn behind the calibration dot.
        ///
        /// Calibration previously drew straight over whatever the application was
        /// already rendering. In a transparent overlay (VIP-Sim) that is the user's
        /// live desktop, so the dot competes with arbitrary content and is genuinely
        /// hard to find — which defeats the point, since calibration accuracy depends
        /// on the user actually fixating the dot.
        ///
        /// A plain backdrop is also standard practice in eye tracking: a uniform,
        /// mid-to-light field keeps pupil size stable across the whole calibration,
        /// whereas a background that jumps between bright and dark changes pupil
        /// diameter and adds noise to the very samples being fitted.
        ///
        /// Alpha 0 restores the old see-through behaviour.
        /// </summary>
        public Color backgroundColor = new Color(0.92f, 0.92f, 0.92f, 1f);

        public float speed = 6.0f;

        public float padding = 10.0f;
        [Range(0f, 0.25f)]
        public float normalizedSafeMargin = 0.08f;
        [Range(1, 4)]
        public int cornerVisits = 2;
        [Range(1f, 6f)]
        public float cornerDwellSeconds = 3f;
        [Tooltip("Seconds to dwell at each head-movement target while the user rotates their head. Longer = more head-pose coverage but a longer calibration.")]
        [Range(2f, 10f)]
        public float headRotationDwellSeconds = 5f;
        [Tooltip("Seconds to dwell at each INTERIOR target (centre + quadrant points) — these give the TPS warp interior anchors. ~8s total at the default.")]
        [Range(0.5f, 4f)]
        public float interiorDwellSeconds = 1.5f;
        [Range(0.1f, 1f)]
        public float settleSeconds = 0.5f;
        [Range(5, 60)]
        public int minimumCornerSamples = 15;
        [Range(1f, 6f)]
        public float cornerOutlierZScore = 3f;
        [Tooltip("Smooth-pursuit latency compensation: while the dot SWEEPS, the eye trails it by roughly this long, so moving samples are labeled with the dot position this many seconds AGO. 0 disables.")]
        [Range(0f, 0.3f)]
        public float pursuitLagSeconds = 0.1f;
        [Tooltip("During dwells, only capture once the raw gaze has been stable (fixation detected). Auto-bypasses within a dwell if it would starve the target of samples.")]
        public bool fixationGate = true;
        [Range(0.05f, 1f)]
        public float fixationWindowSeconds = 0.3f;
        [Tooltip("Maximum raw-gaze spread (fraction of the screen diagonal) still counted as a fixation.")]
        [Range(0.01f, 0.15f)]
        public float fixationDispersionFraction = 0.035f;

        public bool drawCheckpoints;

        [Tooltip("Draw the route the calibration dot will take. Drawn in GUI space while a calibration is running, so it disappears with the rest of the calibration overlay.")]
        public bool drawPath = true;

        public int currentRound = 0;

        public int maxRoundsPerPreset = 2;

        public Calibrations calibrationType = Calibrations.RidgeRegression;
        public bool save = true;
        [Tooltip("Optional bounded jitter of numerical calibration features during training only. Image augmentation is not label-preserving for screen targets.")]
        public CalibrationFeatureAugmentationSettings featureAugmentation = new CalibrationFeatureAugmentationSettings();
        [Tooltip("After ridge training, fit a thin-plate-spline LOCAL correction on the dwell anchors; it is kept only when it improves the held-out samples, otherwise discarded (overfit guard).")]
        public bool enableRidgeWarp = true;

        public bool stopAfterPoints = true;
        public bool quitAfterCalibration = false;

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
            //Append here (not only in Start) so repeat runs keep the cancel hint. LoadCalibration sets
            //returnAfter BEFORE enabling.
            if (returnAfter)
                _guiMessage += "\nRight click to cancel and return";
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
            _sampleFromHeadRotation.Clear();
            _dotTrail.Clear();
            _gazeTrail.Clear();
            _dwellGateAccepts = _dwellGateRejects = 0;
            _dwellGateBypassed = false;
            _warnedGateBypass = false;
            //A cancelled-and-retried calibration clears the training arrays above, so any recorder opened for
            //the abandoned attempt is now keyed to indices that no longer exist. Close it out rather than let
            //it keep appending rows that can never be joined to a trained model.
            if (_recorder != null)
            {
                _recorder.Finish("abandoned", -1f);
                _recorder = null;
            }
            _consentGate = GetComponent<CalibrationRecordingConsent>();
            _consentGate?.BeginIfNeeded();
            if (_presets != null)
            {
                //REBUILD the presets, not just reset the position: Start() runs once per component, so a
                //repeat calibration in the same play session otherwise kept the FIRST run's preset geometry
                //and dwell times — inspector/API changes to padding, cornerVisits, cornerDwellSeconds,
                //headRotationDwellSeconds or normalizedSafeMargin between runs were silently ignored.
                //(LoadCalibration assigns its parameters before enabling, so the values here are current.)
                BuildPresets();
                _currentPreset = 0;
                ResetPoints(0);
            }
        }

        void Start()
        {
            //OnGUI here draws only labels/textures (no GUILayout); skipping the layout pass halves the
            //per-frame IMGUI overhead during a run, which is exactly when frame pacing matters most.
            useGUILayout = false;

            //Get HomulerGaze reference (features come from its platform gaze provider)
            _gaze = GetComponent<HomulerGaze>();

            //If no crosshair is selected load the CalibrationDot Resource. Load it TYPED: the untyped
            //overload returns whichever asset named "CalibrationDot" it finds first (the folder holds both
            //a .png imported as a Sprite and a .svg), and casting that threw an InvalidCastException out of
            //Start(), skipping the rest of the setup below.
            if (calibrationDot == null)
                calibrationDot = Resources.Load<Texture2D>("CalibrationDot");

            //(The returnAfter cancel hint is appended in OnEnable, which has already run.)

            BuildPresets();

            //Master enable for dwelling; each preset's StopAtWaypoints then decides whether IT dwells.
            stopAfterPoints = true;
            ResetPoints(0);
        }

        /// <summary>
        /// (Re)builds the calibration presets from the CURRENT parameters. CornerPreset leads and DWELLS at
        /// the four corners + four edge midpoints (StopAtWaypoints) so the extremes get sustained-fixation
        /// samples — without that the fit had almost no leverage at the corners and collapsed predictions
        /// toward the centre. HeadRotationPreset then dwells at the centre + corners while prompting the
        /// user to rotate their head, giving the head-pose features real variance so the fit compensates
        /// for head movement (see HeadRotationPreset / CalibrationPreset.IsHeadMovement). The ZigZag + wavy
        /// presets follow as continuous sweeps for full-screen coverage (no dwell). Called from Start and
        /// from OnEnable on repeat runs, so parameter changes between runs take effect.
        /// </summary>
        private void BuildPresets()
        {
            _presets = new List<CalibrationPreset>
            {
                new CornerPreset(padding, cornerVisits, cornerDwellSeconds, normalizedSafeMargin),
                //Interior dwells (centre + the 25%/75% quadrant points): gives the thin-plate-spline warp
                //INTERIOR anchors — with boundary-only anchors its interior behaviour was pure affine
                //extrapolation, unable to correct mid-screen residuals where a game's AOIs actually live.
                new InteriorPreset(padding, interiorDwellSeconds),
                new HeadRotationPreset(padding, headRotationDwellSeconds, normalizedSafeMargin),
                new ZigZagPreset(padding, true, 4),
                new VerticalWavyPreset(padding),
                new HorizontalWavyPreset(padding),
            };
        }

        private void ResetPoints(int currentPreset)
        {
            points = _presets[currentPreset].GetPoints();
            _crossHairPos = points[0];
        }

        /// <summary>
        /// Renders the route the dot will take, in GUI space (the same space the waypoints and the dot
        /// itself live in), so the line lands exactly on the path the participant is asked to follow.
        /// </summary>
        /// <remarks>
        /// This used to be a world-space <c>LineRenderer</c> (Prefabs/Path.prefab) instantiated under the
        /// MediaPipe annotation Canvas and positioned with raw <c>Screen</c> pixel values. That Canvas is
        /// Screen-Space-Camera with a CanvasScaler, so its units are pixels/scaleFactor: on anything other
        /// than the scaler's 2436x1125 reference resolution the whole line was offset and shrunk (at
        /// 1920x876 it sat off the right-hand edge of the screen). It was also spawned once in Start() and
        /// never hidden, so the stray line stayed on screen through the evaluation and its results.
        /// </remarks>
        private void DrawPath(List<Vector2> waypoints)
        {
            //Light grey, matching the old LineRenderer's gradient, thin enough not to compete with the dot.
            GUIShapes.DrawPolyline(waypoints, new Color(0.74f, 0.74f, 0.74f, 0.5f),
                Mathf.Max(2f, 2f * Screen.height / 1080f));
        }

        void Update()
        {
            //If no HomulerGaze reference yet, get one, this is in case of Start() racing conditions
            if (_gaze == null)
                _gaze = GetComponent<HomulerGaze>();

            //Mouse.current/Keyboard.current are NULL when no such device exists (headless players,
            //touch-only devices) and dereferencing them threw a NullReferenceException EVERY frame.
            var mouse = Mouse.current;
            var keyboard = Keyboard.current;
            bool leftClick = mouse != null && mouse.leftButton.wasPressedThisFrame;
            bool rightClick = mouse != null && mouse.rightButton.wasPressedThisFrame;

            //If finished and leftclick, signal Returned
            if (leftClick && returnAfter && _finished)
                Returned = true;
            //If rightclick, signal Returned
            if (rightClick && returnAfter)
                Returned = true;
            //Escape aborts too. Right-click already did this, but calibration takes over the whole
            //screen and nothing on it says so, so a user who wants out has no discoverable way back —
            //in an overlay app that is click-through outside its own panel, "just right-click" is not
            //something anyone guesses. Escape is the near-universal convention for "get me out of this
            //modal thing". Same clean path: Returned makes HomulerGaze run UnloadCalibration, which
            //restores the settings BackupSettings captured.
            if (keyboard != null && keyboard.escapeKey.wasPressedThisFrame && returnAfter)
                Returned = true;
            //Start on leftclick. Blocked while the consent screens are up: recording has to be answered
            //BEFORE any sample exists, or the first samples would be captured without an answer.
            if (leftClick && !_finished && !(_consentGate != null && _consentGate.Blocking))
            {
                if (!_started)
                {
                    //Warned at the moment the run starts, not in OnEnable: this is when the geometry that the
                    //whole calibration will be bound to becomes fixed, and it is the last point at which
                    //stopping to maximise the Game view costs nothing.
                    var geometryWarning = ScreenGeometry.PhysicalScaleWarning();
                    if (geometryWarning.Length > 0) UnitEyeLog.Warn(geometryWarning);
                    BeginRecordingIfConsented();
                }
                _started = true;
                _showMessage = false;
                _finishedRound = false;
            }
            //Stop calibration early when clicking S
            if (keyboard != null && keyboard[Key.S].wasPressedThisFrame && _started && !_finished)
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
                        //Fresh fixation-gate state for this dwell (see CaptureNetworkOutput).
                        _gazeTrail.Clear();
                        _dwellGateAccepts = _dwellGateRejects = 0;
                        _dwellGateBypassed = false;
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

            //Record the dot's position history for the pursuit-lag label correction, trimmed to a short
            //window (the lag lookup never needs more than pursuitLagSeconds of history).
            _dotTrail.Add((Time.unscaledTime, _crossHairPos));
            while (_dotTrail.Count > 0 && _dotTrail[0].time < Time.unscaledTime - 0.6f)
                _dotTrail.RemoveAt(0);

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
                //Heads-up when the upcoming round is the head-movement stage (_currentPreset already points
                //at it), so the user expects the "turn your head" prompt before clicking in.
                var headTurnNote = _presets[_currentPreset].IsHeadMovement
                    ? "\nHead-turn round: keep looking at the dot and slowly rotate your head"
                    : "";
                _guiMessage = $"Click to start next round\nRound {currentRound}/{_presets.Count * maxRoundsPerPreset}{headTurnNote}\nRight click to cancel calibration and return";
                _isYielding = false;
                //The dot teleports to the next preset's start; stale trail entries would give the first
                //sweep samples of the new round a wrong (pre-jump) pursuit-lag label.
                _dotTrail.Clear();
                _gazeTrail.Clear();
                //No DrawPath here: it used to push the new preset's waypoints into the LineRenderer, which
                //had to be refreshed once per round. It now draws in immediate mode from OnGUI, which
                //re-reads `points` every frame — and calling it here threw "You can only call GUI functions
                //from inside OnGUI" on every round transition.
            }

            //If done with all rounds or if we want to stop early, finish calibration
            if (currentRound == _presets.Count * maxRoundsPerPreset || _earlyStop)
            {
                //If all rounds are done start training with GUI message
                _guiMessage = "Starting training. This can take a while, please be patient!";
                _showMessage = true;
                _finished = true;
                //An early stop can land mid-dwell, and no later frame can clear this (LateUpdate returns on
                //_finished from here on), leaving the state machine claiming a dwell that will never end.
                _isYielding = false;

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

            //Tell the feature augmentation which feature slots are head yaw/pitch/roll for the active
            //backbone, so its extra head-pose jitter targets them (the layout differs per backbone).
            featureAugmentation.headPoseFeatureIndices =
                _gaze != null ? HeadPoseFeatureIndices(_gaze.GazeBackbone) : null;

            //Process data by calibration type. Training legitimately throws on a degenerate capture — no
            //samples at all (face never tracked, or an early stop seconds in), or every boundary dwell group
            //below minimumCornerSamples. An exception escaping here kills the coroutine, and since _finished
            //was latched BEFORE StartCoroutine, LateUpdate is already inert: the overlay would sit on
            //"Starting training…" forever, with no result, no return hint and the Gaze UI still hidden.
            //Report the failure instead and let the rest of this method restore the UI.
            try
            {
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
            }
            catch (Exception e)
            {
                UnitEyeLog.Exception(e);
                ReturnMessage = $"Calibration training failed: {e.Message} ";
                message += $"{ReturnMessage}\n";
            }

            //Validation-gate advice (empty when the holdout accuracy is fine).
            var advice = ConfidenceAdvice(LastHoldoutRmseCm);
            if (advice.Length > 0)
                message += advice.TrimStart('\n') + "\n";

            //Close the recording with the session's own quality number, so a dataset folder carries the
            //accuracy it was collected at and a bad session can be excluded without re-deriving it. Before
            //_guiMessage is set, so the withdrawal code reaches the screen the participant is looking at.
            if (_recorder != null)
            {
                _recorder.Finish("completed", LastHoldoutRmseCm);
                if (_recorder.ImagesDropped > 0)
                    UnitEyeLog.Warn($"Recording: {_recorder.ImagesDropped} image(s) were dropped (disk or GPU " +
                                    "could not keep up); see summary.json. Sample rows are unaffected.");
                message += $"Recorded {_recorder.SampleCount} samples. Withdrawal code: {_recorder.ParticipantToken}\n";
                _recorder = null;
                _consentGate?.ShowCompletionScreen();
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

        //A calibration abandoned by closing play mode or destroying the object must still flush and close its
        //files; otherwise the last rows and any in-flight images are lost and the folder looks truncated with
        //no explanation.
        private void OnDisable()
        {
            if (_recorder != null)
            {
                _recorder.Finish("interrupted", -1f);
                _recorder = null;
            }
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
            //Never mix vector lengths in one training set: a backbone swap right before the run (or the
            //embedding output publishing its runtime-sized tail a frame late) can produce one row of a
            //different length, and a jagged feature matrix throws inside training. Drop the odd row out.
            if (_xData.Count > 0 && features.Length != _xData[0].Length)
                return;

            //Consume the sample NOW (not only on a successful capture): every return below is a decision
            //about THIS camera sample. Leaving it unconsumed made repeated render frames (render faster
            //than camera) re-run the fixation gate with the identical RawGaze, and those duplicates have
            //zero spread — deflating the measured dispersion and letting an unstable eye pass the gate.
            _lastCapturedGazeSample = _gaze.GazeSampleSequence;

            //During a dwell, skip the first ~0.3s: the dot just jumped to the waypoint and the eye is still
            //saccading to it, so those frames would pair the new (corner) label with mid-flight gaze.
            if (_isYielding && _currentTime > _presets[_currentPreset].DwellSeconds - settleSeconds)
                return;

            //Label selection. While the dot SWEEPS, the eye pursues it with ~100ms latency — the gaze
            //measured NOW corresponds to where the dot was pursuitLagSeconds ago, so that older position is
            //the honest label. During a dwell the dot is parked (current == delayed) and the settle-skip
            //above already discards the saccade, so the current position is used directly.
            Vector2 label = _crossHairPos;
            if (!_isYielding && pursuitLagSeconds > 0f)
                label = DelayedDotPosition(_dotTrail, Time.unscaledTime, pursuitLagSeconds, _crossHairPos);

            //Dwell fixation gate: commercial calibrations only accept samples once the eye has actually
            //settled ON the target. Gate on the raw-gaze dispersion over a short window; if the raw signal
            //is too noisy to ever pass (dispersion threshold is in screen-diagonal fractions), the gate
            //bypasses itself for the rest of the dwell rather than starve the target — the post-hoc
            //z-score rejection in BuildBalancedTrainingData still guards those samples. Head-movement
            //dwells are exempt: raw gaze legitimately wanders there while the head turns.
            if (_isYielding && fixationGate && !_dwellGateBypassed && !_presets[_currentPreset].IsHeadMovement)
            {
                _gazeTrail.Add((Time.unscaledTime, provider.RawGaze));
                while (_gazeTrail.Count > 0 && _gazeTrail[0].time < Time.unscaledTime - fixationWindowSeconds)
                    _gazeTrail.RemoveAt(0);

                float diagonal = Mathf.Sqrt((float)Screen.width * Screen.width + (float)Screen.height * Screen.height);
                if (!IsFixationStable(_gazeTrail, fixationDispersionFraction * diagonal))
                {
                    _dwellGateRejects++;
                    //Bypass when the gate has rejected most of the dwell so far and the dwell is half over.
                    var dwellSeconds = _presets[_currentPreset].DwellSeconds;
                    if (_currentTime < dwellSeconds * 0.5f && _dwellGateRejects > 3 * Mathf.Max(1, _dwellGateAccepts))
                    {
                        _dwellGateBypassed = true;
                        if (!_warnedGateBypass)
                        {
                            _warnedGateBypass = true;
                            UnitEyeLog.Warn("Calibration fixation gate: raw gaze too unstable at a dwell target; " +
                                "capturing ungated for the rest of that dwell (warned once per run — consider better lighting/camera position).");
                        }
                    }
                    return;
                }
                _dwellGateAccepts++;
            }

            //Mirror into the dataset recording BEFORE the Add, using the index this sample is about to take.
            //Hooked here at the successful tail — past every rejection above — so a recorded row can never
            //exist for a sample the training set does not contain.
            RecordSampleIfRecording(_xData.Count, features, label);

            //Clone: GetFeatures() returns the provider's reused per-frame buffer, so the retained training
            //sample must be an owned copy (otherwise every captured sample would alias the latest frame).
            _xData.Add((float[])features.Clone());
            _yXData.Add(label.x / Screen.width);
            _yYData.Add(label.y / Screen.height);
            _yData.Add(new Vector2(label.x /*/ Screen.width*/, label.y /*/ Screen.height*/));
            _sampleTargets.Add(label);
            _sampleCapturedAtDwell.Add(_isYielding);
            _sampleFromHeadRotation.Add(_presets[_currentPreset].IsHeadMovement);
        }

        /// <summary>
        /// Opens a recording session if the participant consented. Clamps imagery tiers away when the
        /// provider cannot deliver imagery that belongs to the same camera frame as the features — under
        /// async GPU readback the crops are one frame ahead of the label, which would produce a dataset whose
        /// pixels and targets disagree with nothing downstream able to notice.
        /// </summary>
        private void BeginRecordingIfConsented()
        {
            _recorder = null;
            if (_consentGate == null || !_consentGate.ShouldRecord) return;

            var provider = _gaze != null ? _gaze.Provider : null;
            var source = provider as IGazeRecordingSource;
            var tier = _consentGate.Tier;

            if (source == null && tier > GazeRecordingTier.Features)
            {
                tier = GazeRecordingTier.Features;
                UnitEyeLog.Warn("Recording: this provider exposes no landmarks or imagery; recording features only.");
            }
            if (source != null && tier >= GazeRecordingTier.EyeCrops && !source.ImageryInSyncWithFeatures)
            {
                tier = GazeRecordingTier.Landmarks;
                UnitEyeLog.Warn("Recording: async GPU readback puts the eye crops one camera frame ahead of the " +
                                "features they would be labelled with, so imagery is disabled for this session. " +
                                "Turn off HomulerGaze._asyncGpuReadback to record imagery.");
            }

            try
            {
                _recorder = new GazeSessionRecorder(_consentGate.Record, tier,
                    source != null ? Mathf.Max(1, source.LandmarkCount) : 1);
                _consentGate.ActiveRecorder = _recorder;
                _recordingStartedAt = Time.unscaledTimeAsDouble;
                _recorder.WriteSessionHeader(
                    backbone: _gaze != null ? _gaze.GazeBackbone.ToString() : "unknown",
                    screenWidth: Screen.width, screenHeight: Screen.height,
                    screenWidthCm: Functions.PixelsToMm(Screen.width) * 0.1f,
                    screenHeightCm: Functions.PixelsToMm(Screen.height) * 0.1f,
                    frameWidth: source != null ? source.FrameWidth : 0,
                    frameHeight: source != null ? source.FrameHeight : 0,
                    flipH: source != null && source.FrameFlippedHorizontally,
                    flipV: source != null && source.FrameFlippedVertically,
                    landmarksSmoothed: source != null && source.LandmarksSmoothed,
                    landmarkCount: source != null ? source.LandmarkCount : 0,
                    rollNormalizeCrops: true, flipAugmentation: false,
                    //Nothing in this flow asks the participant about glasses, so recording HomulerGaze's
                    //default would claim an answer that was never given.
                    glassesState: "not_asked");
            }
            catch (Exception e)
            {
                UnitEyeLog.Error("Could not start the calibration recording; calibrating without it.");
                UnitEyeLog.Exception(e);
                _recorder = null;
            }
        }

        /// <summary>
        /// Mirrors an accepted training sample into the recording. Called from the tail of the capture path
        /// with the index the sample is about to take, so rows and images are 1:1 with training data by
        /// construction rather than by a join that can drift.
        /// </summary>
        private void RecordSampleIfRecording(int sampleIndex, float[] features, Vector2 label)
        {
            if (_recorder == null || !_recorder.Recording) return;
            var provider = _gaze != null ? _gaze.Provider : null;
            if (provider == null) return;
            _recorder.RecordSample(
                sampleIndex, features, label, _crossHairPos,
                Time.unscaledTimeAsDouble - _recordingStartedAt,
                _isYielding, _presets[_currentPreset].IsHeadMovement, _currentPreset, currentRound,
                provider.DistanceMm, provider.HeadPoseEuler, provider.BinocularIrisDisagreement,
                provider.IsBlinking, provider as IGazeRecordingSource, provider);
        }

        private string ProcessDataNeural()
        {
            Debug.Log("Starting MLP training");
            Debug.Log($"Total Count: {_xData.Count}");

            //LastHoldoutRmseCm is static and shared with the ridge path: clear it FIRST so a throw below
            //can never leave a previous RidgeRegression run's number standing in for this session.
            LastHoldoutRmseCm = -1f;

            BuildBalancedTrainingData(out var features, out _, out _, out var targets);
            var mlp = new SimpleMLP();
            string MLPstring = mlp.Train(features, targets, featureAugmentation);

            //Session confidence from the MLP's OWN untouched holdout, same Euclidean convention as the
            //ridge path — otherwise the validation gate (ConfidenceAdvice, the CSV tag and any host-game
            //gating) described whichever calibration happened to be trained last, or never fired at all
            //when MLCalibration was the first calibration of a fresh install.
            LastHoldoutRmseCm = Mathf.Sqrt(mlp.LastHoldoutRmseXCm * mlp.LastHoldoutRmseXCm +
                                           mlp.LastHoldoutRmseYCm * mlp.LastHoldoutRmseYCm);

            if (save)
            {
                //Save under the active backbone's name so each gaze model keeps its own calibration.
                mlp.Save(CalibrationModelStore.FileName("MLP.json", _gaze.GazeBackbone));
                //A persisted error model measured the OLD fit's residuals — stale after retraining.
                GazeErrorModel.Delete(_gaze.GazeBackbone);
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
                rmseScaleY: Functions.PixelsToMm(Screen.height) * 0.1f,
                augmentation: featureAugmentation);

            Debug.Log($"Total Count: {_xData.Count}, Train Count: {result.TrainCount}, Test Count: {result.TestCount}, " +
                      $"Lambda X: {result.BestLambdaX}, Lambda Y: {result.BestLambdaY}");

            //Optional thin-plate-spline local correction, validated on the holdout (kept only if better).
            //The reported ridge RMSE stays warp-free; the warp's own holdout numbers go in the note.
            ThinPlateSplineWarp warp = null;
            var warpNote = "";
            if (enableRidgeWarp)
                warp = TryBuildValidatedWarp(result, out warpNote);

            if (save)
            {
                Debug.Log("Saving best models");
                //Save under the active backbone's name so each gaze model keeps its own calibration.
                result.XModel.Save(CalibrationModelStore.FileName("Reg_X.json", _gaze.GazeBackbone));
                result.YModel.Save(CalibrationModelStore.FileName("Reg_Y.json", _gaze.GazeBackbone));
                //A warp is only valid for the ridge pair it was fitted on: save the kept one, and DELETE
                //any previous warp otherwise, so a stale warp never pairs with this fresh ridge.
                var warpFile = CalibrationModelStore.FileName("Warp.json", _gaze.GazeBackbone);
                if (warp != null)
                    warp.Save(warpFile);
                else
                    ThinPlateSplineWarp.Delete(warpFile);
                //The persisted per-region error model measured the OLD fit's residuals — applying it to
                //this fresh calibration would corrupt the AOI stream. Delete; the next evaluation rebuilds it.
                GazeErrorModel.Delete(_gaze.GazeBackbone);
            }

            //Session confidence from the holdout RMSE — the calibration VALIDATION gate. Sessions in the
            //"poor" band produce AOI logs that are largely noise; telling the user (and tagging the CSV)
            //converts unknown-quality data into known-quality data, which is how commercial webcam
            //trackers earn their reported numbers (session gating), applied here honestly.
            LastHoldoutRmseCm = Mathf.Sqrt(result.XRmse * result.XRmse + result.YRmse * result.YRmse);

            return $"RidgeRegression Training done. Best RMSE X: {result.XRmse}cm | Best RMSE Y: {result.YRmse}cm.{warpNote}";
        }

        /// <summary>Euclidean holdout RMSE (cm) of the most recent calibration training; -1 before any.
        /// The session-quality headline number — consumers (Gaze UI, CSV notes, host game) can gate or
        /// tag their data on it.</summary>
        public static float LastHoldoutRmseCm { get; private set; } = -1f;

        /// <summary>Human advice line for the validation gate; empty when accuracy is fine.</summary>
        public static string ConfidenceAdvice(float rmseCm)
        {
            if (rmseCm < 0f) return "";
            if (rmseCm < 2.0f) return "";
            if (rmseCm < 3.5f) return "\nAccuracy is MODERATE - consider a re-run if precise AOIs matter.";
            return "\nAccuracy is POOR - please recalibrate (check lighting, camera height, seating).";
        }

        /// <summary>
        /// Fits the thin-plate-spline correction on the sit-still dwell anchors (mean ridge prediction →
        /// true target, normalized) and keeps it only when a LEAVE-ONE-ANCHOR-OUT check shows it helps at
        /// screen locations it was not anchored to. With ~9-13 anchors an unvalidated spline can bend the
        /// space between anchors in wrong ways — this gate is what makes the "local calibration" lever safe
        /// to ship enabled, so the gate itself has to be honest.
        /// </summary>
        /// <remarks>
        /// Two leaks used to make this gate self-fulfilling, and both had to go:
        ///
        /// 1. SAME SAMPLE. The anchor loop walked all of _xData, which INCLUDES the trainer's holdout — the
        ///    very samples the warp is then scored on. A holdout sample at target T helped define the anchor
        ///    at T whose destination is exactly T, so the warp was fitted toward the answer it was about to
        ///    be graded on. (The arrays are literally the same objects: _xData -> acceptedFeatures ->
        ///    BuildBalancedTrainingData's output -> xTest -> Result.HoldoutFeatures, all by reference, so
        ///    reference identity is an exact test for membership.)
        /// 2. SAME LOCATION. Even with those samples removed, scoring the full warp on holdout samples at T
        ///    still asks "does the warp help at a location it has an anchor for?" — which it always does, by
        ///    construction. That is not the deployment question. Gaze lands everywhere, so what matters is
        ///    whether the spline helps BETWEEN its anchors.
        ///
        /// So: anchors come only from training samples, and each holdout sample is scored against a warp
        /// fitted WITHOUT the anchor at its own target. Fewer warps survive this than the old gate — the
        /// ones that stop surviving were never earning their keep.
        /// </remarks>
        private ThinPlateSplineWarp TryBuildValidatedWarp(RidgeCalibrationTrainer.Result result, out string note)
        {
            if (result.HoldoutFeatures == null || result.HoldoutFeatures.Length < 10)
            {
                note = " Corner warp skipped (holdout too small).";
                return null;
            }

            //Reference identity, not value equality: these are the same float[] objects the trainer held out.
            var holdoutSamples = new HashSet<float[]>(ReferenceEqualityComparer<float[]>.Instance);
            foreach (var f in result.HoldoutFeatures)
                if (f != null) holdoutSamples.Add(f);

            //Anchors: one per unique sit-still dwell target, from TRAINING samples only. Head-movement dwells
            //are excluded — their deliberate head-pose variance would smear the anchor's mean prediction.
            var sums = new Dictionary<Vector2, Vector2>();
            var counts = new Dictionary<Vector2, int>();
            for (var i = 0; i < _xData.Count; i++)
            {
                if (!_sampleCapturedAtDwell[i] || _sampleFromHeadRotation[i])
                    continue;
                if (holdoutSamples.Contains(_xData[i]))
                    continue;
                var prediction = new Vector2(result.XModel.Predict(_xData[i]), result.YModel.Predict(_xData[i]));
                if (float.IsNaN(prediction.x) || float.IsNaN(prediction.y))
                    continue;
                var key = new Vector2(_sampleTargets[i].x / Screen.width, _sampleTargets[i].y / Screen.height);
                sums.TryGetValue(key, out var s);
                sums[key] = s + prediction;
                counts.TryGetValue(key, out var c);
                counts[key] = c + 1;
            }
            if (sums.Count < ThinPlateSplineWarp.MinimumAnchors)
            {
                note = " Corner warp skipped (too few dwell anchors).";
                return null;
            }

            var anchorKeys = new List<Vector2>(sums.Keys);
            var source = new Vector2[anchorKeys.Count];
            var destination = new Vector2[anchorKeys.Count];
            for (var i = 0; i < anchorKeys.Count; i++)
            {
                destination[i] = anchorKeys[i];
                source[i] = sums[anchorKeys[i]] / counts[anchorKeys[i]];
            }

            var warp = ThinPlateSplineWarp.Fit(source, destination);
            if (warp == null)
            {
                note = " Corner warp skipped (fit failed).";
                return null;
            }

            //Leave-one-anchor-out warps. Dropping one anchor must still leave a fittable spline; if it does
            //not, we cannot validate honestly and so discard rather than ship an unvalidated spline.
            if (anchorKeys.Count - 1 < ThinPlateSplineWarp.MinimumAnchors)
            {
                note = " Corner warp discarded (too few anchors to validate without leaking).";
                return null;
            }
            var withoutAnchor = new Dictionary<Vector2, ThinPlateSplineWarp>();
            for (var j = 0; j < anchorKeys.Count; j++)
            {
                var src = new Vector2[anchorKeys.Count - 1];
                var dst = new Vector2[anchorKeys.Count - 1];
                var w = 0;
                for (var i = 0; i < anchorKeys.Count; i++)
                {
                    if (i == j) continue;
                    src[w] = source[i];
                    dst[w] = destination[i];
                    w++;
                }
                withoutAnchor[anchorKeys[j]] = ThinPlateSplineWarp.Fit(src, dst);
            }

            //Score each holdout sample against a warp that never saw an anchor at that sample's own target.
            double before = 0, after = 0;
            var n = 0;
            for (var i = 0; i < result.HoldoutFeatures.Length; i++)
            {
                var p = new Vector2(result.XModel.Predict(result.HoldoutFeatures[i]),
                    result.YModel.Predict(result.HoldoutFeatures[i]));
                if (float.IsNaN(p.x) || float.IsNaN(p.y)) continue;

                var key = new Vector2(result.HoldoutTargetsX[i], result.HoldoutTargetsY[i]);
                var scoringWarp = withoutAnchor.TryGetValue(key, out var held) ? held : warp;
                //A dropped anchor whose remaining spline would not fit leaves nothing honest to score with.
                if (scoringWarp == null) continue;

                var warped = scoringWarp.Apply(p);
                double ex = p.x - key.x, ey = p.y - key.y;
                before += ex * ex + ey * ey;
                ex = warped.x - key.x;
                ey = warped.y - key.y;
                after += ex * ex + ey * ey;
                n++;
            }
            if (n < 10)
            {
                note = " Corner warp discarded (too few scorable holdout samples).";
                return null;
            }

            //Require a clear (>2% MSE) improvement to keep.
            if (after < before * 0.98)
            {
                note = $" Corner warp kept (leave-one-anchor-out error {Math.Sqrt(before / n):F4}→{Math.Sqrt(after / n):F4} normalized).";
                return warp;
            }
            note = " Corner warp discarded (no leave-one-anchor-out improvement).";
            return null;
        }

        /// <summary>
        /// Identity comparison for reference types. Needed because the warp anchors must exclude exactly the
        /// arrays the trainer held out, and two distinct samples can hold numerically equal feature vectors.
        /// (System.Runtime.CompilerServices.ReferenceEqualityComparer is .NET 5+; this keeps the package on
        /// the Unity-supported surface.)
        /// </summary>
        private sealed class ReferenceEqualityComparer<T> : IEqualityComparer<T> where T : class
        {
            public static readonly ReferenceEqualityComparer<T> Instance = new ReferenceEqualityComparer<T>();
            public bool Equals(T a, T b) => ReferenceEquals(a, b);
            public int GetHashCode(T obj) => System.Runtime.CompilerServices.RuntimeHelpers.GetHashCode(obj);
        }

        //Thin wrapper over CalibrationSampleBalancer, which holds the actual logic so the offline benchmark
        //trains on exactly the samples a real calibration would. Do not reimplement either side.
        private void BuildBalancedTrainingData(out float[][] features, out float[] targetsX,
            out float[] targetsY, out Vector2[] targets)
        {
            CalibrationSampleBalancer.Build(
                _xData, _yXData, _yYData, _yData, _sampleTargets, _sampleCapturedAtDwell, _sampleFromHeadRotation,
                Screen.width, Screen.height, minimumCornerSamples, cornerOutlierZScore,
                out features, out targetsX, out targetsY, out targets, out var report);
            Debug.Log(report);
        }

        private bool IsBoundaryTarget(Vector2 target)
            => CalibrationSampleBalancer.IsBoundaryTarget(target, Screen.width, Screen.height);

        /// <summary>
        /// The newest dot position at least <paramref name="lagSeconds"/> old — the pursuit-lag corrected
        /// label for a sweep sample (see CaptureNetworkOutput). Falls back to <paramref name="fallback"/>
        /// (the current position) when the trail has no entry that old yet, e.g. right after a round starts.
        /// </summary>
        public static Vector2 DelayedDotPosition(IReadOnlyList<(float time, Vector2 pos)> trail,
            float now, float lagSeconds, Vector2 fallback)
        {
            var cutoff = now - lagSeconds;
            for (var i = trail.Count - 1; i >= 0; i--)
                if (trail[i].time <= cutoff)
                    return trail[i].pos;
            return fallback;
        }

        /// <summary>
        /// Simple dispersion-based fixation test (I-DT style): the window counts as a stable fixation when
        /// it holds at least 3 samples and the bounding-box diagonal of the gaze points is within
        /// <paramref name="maxDispersionPixels"/>. The caller trims the sample window by time.
        /// </summary>
        public static bool IsFixationStable(IReadOnlyList<(float time, Vector2 pos)> samples, float maxDispersionPixels)
        {
            if (samples.Count < 3)
                return false;
            float minX = float.MaxValue, maxX = float.MinValue, minY = float.MaxValue, maxY = float.MinValue;
            for (var i = 0; i < samples.Count; i++)
            {
                var p = samples[i].pos;
                if (p.x < minX) minX = p.x;
                if (p.x > maxX) maxX = p.x;
                if (p.y < minY) minY = p.y;
                if (p.y > maxY) maxY = p.y;
            }
            var dx = maxX - minX;
            var dy = maxY - minY;
            return dx * dx + dy * dy <= maxDispersionPixels * maxDispersionPixels;
        }

        /// <summary>
        /// The indices of the head yaw/pitch/roll slots in the active backbone's feature vector, used to
        /// aim the augmentation's extra head-pose jitter. EyeMU emits [embedding4, gaze polynomial (gx, gy,
        /// gx², gy², gx·gy, gx³, gy³), headYaw, headPitch, headRoll, headArea] so head pose is at 11/12/13;
        /// the direction backbones emit [yaw, pitch, yaw², pitch², yaw·pitch, yaw³, pitch³, headYaw, headPitch,
        /// headRoll, headArea] so it is at 7/8/9. Head area is excluded (it is not a rotation). Kept in sync
        /// with HomulerEyeMURunner.FillEyeMUFeatures and GazeEstimationRunner.FillGazeFeatures.
        /// </summary>
        private static int[] HeadPoseFeatureIndices(GazeBackbone backbone)
            => CalibrationSampleBalancer.HeadPoseFeatureIndices(backbone);

        private void OnGUI()
        {
            //Backdrop first, so everything else in this OnGUI draws on top of it.
            if (backgroundColor.a > 0f)
            {
                var prevBg = GUI.color;
                GUI.color = backgroundColor;
                GUI.DrawTexture(new Rect(0f, 0f, Screen.width, Screen.height), Texture2D.whiteTexture);
                GUI.color = prevBg;
            }

            //Yield the screen entirely while the consent screens are up. Draw order between two components'
            //OnGUI is not defined, so without this the calibration dot and the "click to start" message paint
            //over (or under) a decision the participant is in the middle of making.
            if (_consentGate != null && _consentGate.Blocking) return;

            //Show message on screen. Scale the font with resolution so the message (and the final RMSE)
            //stays legible on high-DPI displays; == baseline at 1080p, larger above it.
            if (_showMessage)
            {
                float uiScale = Mathf.Max(1f, Mathf.Sqrt(0.001f * Screen.width * Screen.height / 2073.6f));
                _guiStyle.fontSize = Mathf.RoundToInt((_finished ? 16 : 36) * uiScale);
                GUI.Label(new Rect(Screen.width / 2 - Screen.width * (_finished ? 0.15f : 0.1f), Screen.height / 2 - 20, 100, 60), $"{_guiMessage}", _guiStyle);
            }

            //During the head-movement stage, prompt the user to rotate their head so the head-pose features
            //gain variance (see HeadRotationPreset). Shown as a banner near the top so it never covers the dot.
            if (IsHeadRotationStage())
            {
                float uiScale = Mathf.Max(1f, Mathf.Sqrt(0.001f * Screen.width * Screen.height / 2073.6f));
                _headRotationStyle.fontSize = Mathf.RoundToInt(26 * uiScale);
                _headRotationStyle.fontStyle = FontStyle.Bold;
                _headRotationStyle.alignment = TextAnchor.MiddleCenter;
                _headRotationStyle.wordWrap = true;
                _headRotationStyle.normal.textColor = Color.white;
                var prompt = _isYielding
                    ? "Keep looking at the dot —\nslowly turn your head:  left · right · up · down"
                    : "Head-turn round: follow the dot, then turn your head when it stops";
                var rect = new Rect(Screen.width * 0.1f, Screen.height * 0.04f, Screen.width * 0.8f, Screen.height * 0.16f);
                //Dark backing so the prompt stays readable over whatever the scene renders behind it.
                var prev = GUI.color;
                GUI.color = new Color(0f, 0f, 0f, 0.5f);
                GUI.DrawTexture(rect, Texture2D.whiteTexture);
                GUI.color = prev;
                GUI.Label(rect, prompt, _headRotationStyle);
            }

            //Route preview for the current preset. Drawn here (rather than as a scene object) so it exists
            //only while this overlay does. Repaint-only: the sweep presets have ~150 waypoints, and the
            //other OnGUI passes (one per input event) would redo that geometry for nothing.
            if (drawPath && !_finished && Event.current.type == EventType.Repaint)
                DrawPath(points);

            var size = 36;
            //Nothing below belongs on the results screen: the dot marks the target the participant should be
            //looking at right now, and once the run is over there is none. Without the _finished term the
            //breathing dot, its dwell pulse ring and the countdown label all keep drawing over the RMSE
            //readout — and after an early stop (S) the countdown freezes, because _isYielding is still set
            //when LateUpdate latches _finished and every later frame returns before it can be cleared.
            //(HomulerGazeEvaluation.cs got exactly this guard in f78f0ce; the calibration side was missed.)
            if (calibrationDot != null && !_finished)
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

                // Draw the calibration dot, animated to hold the participant's attention: a gentle breathing
                // pulse always, plus an expanding/fading ring during the dwell ("collect") phase so it is
                // obvious WHEN to hold a steady fixation — the same cue the reference GazeCloud HTML uses.
                DrawAnimatedDot(_crossHairPos, size, _isYielding);

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

        //True while the active preset is the head-movement stage (used to show the rotate-your-head prompt).
        private bool IsHeadRotationStage() =>
            _started && !_finished && _presets != null && _currentPreset >= 0 &&
            _currentPreset < _presets.Count && _presets[_currentPreset].IsHeadMovement;

        /// <summary>
        /// Draws the calibration dot with an attention-holding animation: a small always-on "breathing"
        /// scale, plus an expanding, fading ring behind it while <paramref name="collecting"/> (the dwell /
        /// measurement phase). The animation is driven by Time.time so it runs at the same rate regardless
        /// of frame rate. Restores GUI.color before returning.
        /// </summary>
        private void DrawAnimatedDot(Vector2 pos, float baseSize, bool collecting)
        {
            var oldColor = GUI.color;

            //Expanding, fading pulse ring during the collect/dwell phase — the "hold your fixation here" cue.
            if (collecting)
            {
                float phase = Mathf.Repeat(Time.time * PulseHz, 1f);   // 0..1 each pulse
                float ringSize = baseSize * (1f + phase * 1.6f);
                GUI.color = new Color(oldColor.r, oldColor.g, oldColor.b, (1f - phase) * 0.55f);
                GUI.DrawTexture(new Rect(pos.x - 0.5f * ringSize, pos.y - 0.5f * ringSize, ringSize, ringSize), calibrationDot);
            }

            //Gentle breathing of the dot itself so it keeps drawing the eye while it moves.
            float breathe = 1f + 0.12f * Mathf.Sin(Time.time * 2f * Mathf.PI * PulseHz);
            float dotSize = baseSize * breathe;
            GUI.color = oldColor;
            GUI.DrawTexture(new Rect(pos.x - 0.5f * dotSize, pos.y - 0.5f * dotSize, dotSize, dotSize), calibrationDot);
        }
    }
}
