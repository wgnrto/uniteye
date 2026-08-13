/// Code based on <see cref="Gaze"/>.
/// Updated by Tobias Wagner 07/2023 to integrate <see cref="Mediapipe.Unity"/> package.

#if !UNITY_WEBGL || UNITY_EDITOR
using Mediapipe.Unity;
using Mediapipe.Unity.FaceMesh;
#endif
using System.Collections.Generic;
using UnitEye;
using UnityEngine;
using UnityEngine.InputSystem;
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

        //Taller than before (0.96 vs 0.82) and higher up so the Calibration profiles panel at the bottom is
        //inside the window instead of clipped past its lower edge. Draggable at runtime.
        private Rect gazeUI = new Rect(Screen.height * 0.05f, Screen.height * 0.02f, Screen.width * 0.5f, Screen.height * 0.96f);

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

        //Calibration profile save/load UI state (see CalibrationProfileStore). Calibration is slow, so the
        //Gaze UI lets the user snapshot the current calibration under a name and restore it later.
        private string _profileName = "";
        private string _profileStatus = "";
        private List<string> _profileList;
        private int _profileIndex;

        //Drift re-centering: a constant offset added to the calibrated gaze so slow seating/posture drift
        //(the gaze creeping off after calibration) can be corrected in seconds without a full recalibration.
        //_recentCalibratedGaze is a smoothed PRE-offset calibrated gaze used as the reference when re-centering.
        private Vector2 _driftOffset = Vector2.zero;
        private Vector2 _recentCalibratedGaze;
        private bool _hasRecentGaze;
        private float _recenterArmedUntil = -1f;
        private const float RecenterArmSeconds = 1.5f;
        private GUIStyle _recenterStyle;
        /// <summary>True while a drift re-center is counting down (a marker is shown at screen center).</summary>
        public bool IsRecentering => _recenterArmedUntil > 0f;

        private bool _drawDotBackup = true;
        private bool _showEyesBackup = true;
        private bool _visualizeAOIBackup = false;
        private bool _showGazeUIBackup = false;
        private Calibrations _calibrationBackup;
        private bool _backupped;

        //---- Online drift stack (see DriftCorrector): a 6-DOF affine correction on top of the frozen
        //calibration, fed by validated anchors (clicks, pursuit of registered game objects, attention
        //events). Holds calibrated accuracy across the session instead of letting it decay.
        private readonly DriftCorrector _driftCorrector = new DriftCorrector();
        //Ring buffer of recent PRE-corrector calibrated gaze (normalized) for the pre-click fixation
        //window: gaze leads a click by 100-200ms, so the honest sample is the fixation BEFORE the click.
        private readonly List<(double t, Vector2 gazeNorm)> _gazeTrail = new List<(double, Vector2)>(64);
        private const double GazeTrailSeconds = 0.6;
        private const double ClickWindowStart = 0.35;   // seconds before the click
        private const double ClickWindowEnd = 0.08;
        //Registered pursuit targets (game objects the player may track) + pending attention events.
        private readonly Dictionary<string, PursuitCorrelator> _pursuitTargets = new Dictionary<string, PursuitCorrelator>();
        private readonly List<(double t, Vector2 posNorm)> _attentionEvents = new List<(double, Vector2)>();
        private const double AttentionEventWindow = 0.45;   // capture happens within ~100-350ms of onset

        //---- Fixation-level AOI stream (see FixationAggregator): AOI hit-testing consumes the running
        //fixation centroid of the UNFILTERED calibrated gaze (sqrt(N) less noise, no filter phase-lag);
        //the visible cursor keeps the responsive One-Euro output.
        private readonly FixationAggregator _fixationAggregator = new FixationAggregator();

        //---- Per-user, per-region error model (bias + covariance), written by the evaluation. Powers
        //(a) runtime BIAS CORRECTION of the AOI stream (the measured systematic offset per region is
        //subtracted — a free local correction on top of ridge+TPS, measured rather than fitted) and
        //(b) the probabilistic AOI layer (P(AOI|fixation) under the region's error ellipse).
        private GazeErrorModel _errorModel;
        private readonly List<(string uID, float probability)> _aoiProbabilities = new List<(string, float)>();
        private double _lastProbabilityTime;
        private const double ProbabilityInterval = 0.25;   // seconds; 32-sample MC per AOI, so throttled

        //Latest camera-frame capture time + derived pipeline latency (see IGazeProvider.CaptureTimestamp).
        private double _captureTimestamp;
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
        /// <summary>Capture time (Time.unscaledTimeAsDouble) of the camera frame behind the current gaze.</summary>
        public double CaptureTimestamp => _captureTimestamp;
        /// <summary>Measured pipeline latency of the current sample (consume-time to now), seconds. A
        /// moving object crossing at 500px/s under 150ms latency is a 75px systematic AOI error — pair
        /// logged gaze with world state at CaptureTimestamp, not at log time.</summary>
        public float MeasuredLatencySeconds { get; private set; }
        /// <summary>The online drift corrector (read-only access for session-health telemetry).</summary>
        public DriftCorrector DriftCorrector => _driftCorrector;
        /// <summary>True while the fixation aggregator classifies the current gaze as a fixation.</summary>
        public bool InFixation => _fixationAggregator.InFixation;
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
        //EXPERIMENTAL: pipeline the inference result readback instead of stalling the CPU on the GPU every
        //frame. Gaze is published one camera frame later (~33-66ms extra latency at 15-30fps camera rates)
        //but the main thread no longer blocks in DownloadToArray. Verify gaze quality after enabling.
        [Tooltip("EXPERIMENTAL: async GPU readback — higher throughput, gaze arrives one camera frame later. Verify with a webcam before shipping.")]
        [SerializeField] private bool _asyncGpuReadback = false;
        public GazeBackbone GazeBackbone => _gazeBackbone;
        //The direction-based backbones feed the model a FACE crop (shown as one thumbnail), not eye crops.
        //Only the PURE direction backbones show a single face-crop thumbnail; EyeMU and the ensemble
        //(whose debug textures are EyeMU's) show the two eye crops.
        private bool UsesFaceCrop => _gazeBackbone == GazeBackbone.GazeMobileOne ||
                                     _gazeBackbone == GazeBackbone.GazeMobileNetV2 ||
                                     _gazeBackbone == GazeBackbone.GazeResNet34;

        /// <summary>
        /// Switch the gaze model at runtime. Rebuilds the provider's backbone; the shared face-mesh /
        /// blink / distance stack is unchanged. Calibration is per-backbone (the feature vector differs),
        /// so gaze falls back to raw (uncalibrated) until you recalibrate for the new model.
        /// </summary>
        public void SetBackbone(GazeBackbone backbone)
        {
            //Refuse to swap the model while a calibration/evaluation is running: the backbones produce
            //DIFFERENT feature-vector lengths (EyeMU 15 vs GazeEstimation 11), so a mid-run switch would mix
            //jagged rows into the capture (training then throws inside the coroutine) and the result would
            //be saved under the NEW backbone's calibration file, silently corrupting it.
            if ((_calibrationScript != null && _calibrationScript.enabled) ||
                (_evaluationScript != null && _evaluationScript.enabled))
            {
                UnitEyeLog.Warn("SetBackbone ignored: finish or cancel the running calibration/evaluation first.");
                return;
            }

            //Persist the OLD backbone's drift state before switching (it belongs to that backbone's
            //calibrated signal), then load the NEW backbone's saved state instead of deleting it —
            //ClearDrift() here used to wipe the switched-TO backbone's cross-session warm start, because
            //DriftStateKey already pointed at the new name.
            if (_driftCorrector.AcceptedAnchors > 0)
                PlayerPrefs.SetString(DriftStateKey, _driftCorrector.SaveToJson());

            _gazeBackbone = backbone;
            _provider?.SetBackbone(backbone);
            //Load this backbone's own calibration (per-backbone files). If it hasn't been calibrated yet,
            //the models load as null and RefineGazeLocation falls back to raw gaze until you calibrate.
            _modelStore.Load(_calibrations, _gazeBackbone);
            _errorModel = GazeErrorModel.Load(_gazeBackbone);

            //Fresh in-memory state for the new backbone's signal, warm-started from ITS saved session.
            _driftOffset = Vector2.zero;
            _recenterArmedUntil = -1f;
            _hasRecentGaze = false;
            _driftCorrector.Reset();
            _driftCorrector.LoadFromJson(PlayerPrefs.GetString(DriftStateKey, ""));
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

        //Re-reads the calibration model files for the current type + backbone from disk. Used after a
        //calibration profile is loaded (which overwrites those files) so the change takes effect live. A
        //new calibration supersedes any drift correction, so clear it.
        public void ReloadCalibration()
        {
            _modelStore.Load(_calibrations, _gazeBackbone);
            _errorModel = GazeErrorModel.Load(_gazeBackbone);
            ClearDrift();
        }

        /// <summary>
        /// Arms a drift re-center: a marker appears at screen centre for a moment; the user looks at it and
        /// the calibrated gaze is nudged so its smoothed value maps exactly to centre, correcting slow
        /// seating/posture drift without a full recalibration. Safe to call from host-game code / a custom key.
        /// </summary>
        public void RecenterDrift() => _recenterArmedUntil = Time.unscaledTime + RecenterArmSeconds;

        /// <summary>Clears any drift correction — the manual re-center offset AND the online affine
        /// corrector state (and cancels a pending re-center). Called on (re)calibration/profile load.</summary>
        public void ClearDrift()
        {
            _driftOffset = Vector2.zero;
            _recenterArmedUntil = -1f;
            _driftCorrector.Reset();
            PlayerPrefs.DeleteKey(DriftStateKey);
        }

        private string DriftStateKey => $"UnitEyeDriftState_{_gazeBackbone}";

        private void OnApplicationQuit()
        {
            //Persist the drift state for a warm start next session (same backbone; reset on recalibration).
            if (_driftCorrector.AcceptedAnchors > 0)
            {
                PlayerPrefs.SetString(DriftStateKey, _driftCorrector.SaveToJson());
                PlayerPrefs.Save();
            }
        }

        // ------------------------------------------------------------------------------------------
        // Drift-anchor sources. All observe the PRE-corrector calibrated gaze (the trail) so the
        // corrector learns the full residual; all go through DriftCorrector's outlier gate.
        // ------------------------------------------------------------------------------------------

        /// <summary>
        /// Click anchors: at a mouse click, the median PRE-CLICK-window gaze (gaze leads a click by
        /// 100-200ms and has settled by then) is anchored to the click position. The corrector's robust
        /// residual gate rejects the ~1/3 of clicks users make without looking.
        /// </summary>
        private void CollectClickAnchor()
        {
            if (!clickAnchors) return;
            var mouse = Mouse.current;                       // null when no mouse device exists
            if (mouse == null || !mouse.leftButton.wasPressedThisFrame) return;

            var mp = mouse.position.ReadValue();
            //Input system positions are bottom-left; gaze space is top-left.
            var clickNorm = new Vector2(
                Mathf.Clamp01(mp.x / Screen.width),
                Mathf.Clamp01(1f - mp.y / Screen.height));

            if (TryMedianTrailGaze(ClickWindowStart, ClickWindowEnd, out var gazeNorm))
                _driftCorrector.AddAnchor(gazeNorm, clickNorm, 1f);
        }

        /// <summary>
        /// Attention-event anchors: the game declares "something attention-grabbing appeared at P"
        /// (<see cref="ReportAttentionEvent"/>); if the gaze settles near P within the capture window
        /// (~100-350ms, abrupt onsets capture attention that fast), it becomes a low-weight anchor.
        /// </summary>
        private void CollectAttentionEventAnchors(Vector2 currentGazeNorm)
        {
            for (int i = _attentionEvents.Count - 1; i >= 0; i--)
            {
                var (t, pos) = _attentionEvents[i];
                double age = _captureTimestamp - t;
                if (age > AttentionEventWindow)
                {
                    _attentionEvents.RemoveAt(i);
                    continue;
                }
                //Settled near the event (within ~10% of the screen) after the saccade window opened.
                if (age > 0.1 && _fixationAggregator.InFixation &&
                    (currentGazeNorm - pos).magnitude < 0.1f)
                {
                    _driftCorrector.AddAnchor(currentGazeNorm, pos, 0.3f);
                    _attentionEvents.RemoveAt(i);
                }
            }
        }

        /// <summary>Median of the gaze-trail samples captured between [click-start .. click-end] ago.</summary>
        private bool TryMedianTrailGaze(double windowStart, double windowEnd, out Vector2 median)
        {
            median = default;
            double now = Time.unscaledTimeAsDouble;
            var xs = new List<float>(8);
            var ys = new List<float>(8);
            foreach (var (t, g) in _gazeTrail)
            {
                double age = now - t;
                if (age <= windowStart && age >= windowEnd)
                {
                    xs.Add(g.x);
                    ys.Add(g.y);
                }
            }
            if (xs.Count < 3) return false;
            xs.Sort();
            ys.Sort();
            median = new Vector2(xs[xs.Count / 2], ys[ys.Count / 2]);
            return true;
        }

        /// <summary>
        /// Game-facing: report the CURRENT position (normalized 0..1, top-left origin) of a moving object
        /// the player plausibly tracks, once per frame per object. When the gaze trajectory provably
        /// pursues it (per-axis correlation gate), dense drift anchors are emitted along the path —
        /// including screen regions clicks never visit. Route reward-flight animations through corners
        /// occasionally and the corners stay calibrated for free.
        /// </summary>
        public void FeedPursuitTarget(string id, Vector2 normalizedPosition)
        {
            if (!onlineDriftCorrection || _provider == null) return;
            if (!_pursuitTargets.TryGetValue(id, out var correlator))
                _pursuitTargets[id] = correlator = new PursuitCorrelator();

            //Current pre-corrector gaze = last trail entry (this frame's sample). The gaze carries its
            //CAPTURE time; the object position carries the RENDER clock — the correlator needs both (the
            //pipeline latency between them would otherwise invert the pursuit-lag compensation).
            if (_gazeTrail.Count == 0) return;
            var (gazeTime, gazeNorm) = _gazeTrail[_gazeTrail.Count - 1];
            if (correlator.Feed(gazeNorm, gazeTime, normalizedPosition, Time.unscaledTimeAsDouble,
                    out var pairedGaze, out var pairedTarget))
                _driftCorrector.AddAnchor(pairedGaze, pairedTarget, 0.5f);
        }

        /// <summary>Stop tracking a pursuit target (e.g. the object despawned).</summary>
        public void RemovePursuitTarget(string id) => _pursuitTargets.Remove(id);

        /// <summary>
        /// Game-facing: declare that something attention-grabbing just appeared/happened at the given
        /// normalized screen position (enemy spawn, explosion, dialog popup). If the player's gaze settles
        /// there within ~350ms it becomes a low-weight drift anchor. Costs nothing when they don't look.
        /// </summary>
        public void ReportAttentionEvent(Vector2 normalizedPosition)
        {
            if (!onlineDriftCorrection) return;
            _attentionEvents.Add((Time.unscaledTimeAsDouble, normalizedPosition));
            if (_attentionEvents.Count > 16)
                _attentionEvents.RemoveAt(0);
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

        //Reference display the 1€ filter is normalized to, so `beta` behaves the same on every resolution
        //(see SmoothGazeLocation). Gaze is scaled into this space before filtering and back afterward.
        private const float OneEuroReferenceWidth = 1920f;
        private const float OneEuroReferenceHeight = 1080f;
        //1€ filter speed coefficient: how much fast movement raises the cutoff (less lag while the gaze
        //moves). The old default 0.001 barely adapted, so quick glances to the corners lagged far behind.
        //Now interpreted in the 1920x1080 reference space above (resolution-independent); 0.012 there matches
        //the 0.007 that tested well on a 3200-wide panel.
        [SerializeField, Range(1e-10f, 0.05f)] public float beta = 0.012f;
        //1€ filter minimum cutoff (Hz): the responsiveness floor while fixating. The old default 0.001 Hz
        //(and even the old 0.05 slider ceiling) over-smoothed ~1000x — the dot could not reach a corner
        //before the eye moved on, which reads as poor accuracy. ~1.0 Hz is the 1€ paper's pointing baseline.
        [SerializeField, Range(1e-10f, 3.0f)] public float mincutoff = 1.0f;
        [SerializeField, Range(1e-10f, 10.0f)] public float dcutoff = 1.0f;

        [Tooltip("Feed AOI hit-testing the running FIXATION CENTROID of the calibrated gaze instead of the per-frame filtered sample. Fixation-level aggregation cuts the noise component ~sqrt(N) and is what commercial trackers report; the visible cursor keeps the responsive filtered signal either way.")]
        [SerializeField] public bool fixationAOILogging = true;

        [Tooltip("Use the per-region error model measured by the EVALUATION (run one after calibrating!) to (a) subtract the measured systematic bias from the AOI stream and (b) log P(AOI|fixation) probabilities alongside boolean hits. No effect until an evaluation has been run for the active backbone.")]
        [SerializeField] public bool useErrorModel = true;

        [Tooltip("Continuously correct slow drift with a small affine layer fed by validated interaction anchors (clicks, registered pursuit targets, attention events). Sits on top of the frozen calibration; worst case it converges to identity. Webcam trackers without this lose ~50% accuracy per 20-minute session.")]
        [SerializeField] public bool onlineDriftCorrection = true;

        [Tooltip("Treat mouse clicks as gaze anchors for drift correction: the pre-click fixation (gaze leads a click by 100-200ms) is anchored to the click position, gated for outliers (users look at their click only ~2/3 of the time).")]
        [SerializeField] public bool clickAnchors = true;

        [Tooltip("Horizontal-flip test-time augmentation for the direction backbones (MobileOne/MobileNetV2/ResNet34): infer the mirrored crop too and average (~3-8% accuracy gain, 2x inference). Sync readback only. HAND-TEST before shipping: if gaze collapses toward screen centre horizontally, the mirror convention is wrong on this setup - turn it off.")]
        [SerializeField] private bool _flipAugmentation = false;

        [Tooltip("Roll-normalize the direction backbones' face crop (rotate sampling by -headRoll so the model always sees an upright face — 2D data normalization; laptop users tilt constantly). HAND-TEST: if gaze degrades when you tilt your head, the sign convention is wrong on this setup - turn it off.")]
        [SerializeField] private bool _rollNormalizeCrops = true;

        [Tooltip("Whether the user currently wears glasses. Saved with calibration profiles; loading a profile made with the other state warns (glasses are worth ~1cm+ of error to appearance models). Toggleable in the Gaze UI profiles panel.")]
        public bool userWearsGlasses = false;

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
            //A provider that cannot be built (unassigned/incomplete MediaPipe GameObject) must disable this
            //component: every callback below dereferences _provider, so carrying on would replace one
            //actionable error with an NRE per frame from LateUpdate/OnGUI.
            try
            {
    #if UNITY_WEBGL && !UNITY_EDITOR
                _provider = new WebGLGazeProvider();
    #else
                _provider = new NativeGazeProvider(_mediaPipeGO, _gazeBackbone, _asyncGpuReadback, _flipAugmentation, _rollNormalizeCrops);
    #endif
            }
            catch (System.Exception e)
            {
                Debug.LogError($"UnitEye: gaze provider setup failed, disabling {nameof(HomulerGaze)} on '{name}'. {e.Message}", this);
                enabled = false;
                return;
            }

            //Warm-start the drift correction from the previous session (same backbone): seating drift is
            //largely affine, so last session's correction is a better prior than identity. It keeps
            //adapting from anchors either way, and a recalibration resets it.
            _driftCorrector.LoadFromJson(PlayerPrefs.GetString(DriftStateKey, ""));

            //Per-region error model measured by the evaluation (null until one has been run).
            _errorModel = GazeErrorModel.Load(_gazeBackbone);

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
            //No provider = setup failed (Start logged why) or the component was torn down; everything below
            //dereferences it.
            if (_provider == null)
                return;

            //Unload calibration/evaluation the moment they signal Returned -- deliberately ABOVE the
            //Tick() gate. These checks used to sit below it, which made LEAVING the fullscreen modal
            //depend on the camera still delivering frames: a webcam that stalls (USB hiccup, lid closed,
            //another app grabbing the device) or a run started with nobody in front of the camera left
            //the calibration overlay up with Returned set and nothing ever consuming it -- no click,
            //Escape or right-click could exit. Exiting a modal must not require working eye tracking.
            if (_calibrationScript != null && _calibrationScript.Returned)
            {
                //CSV note marking the calibration end (PauseCSVLogging is still true here). No fresh
                //sample exists on this path, so the last known gaze stands in for both columns.
                if (_csvLogger != null && _csvLogger.isActiveAndEnabled)
                    _csvLogger.Append(new CSVData(gazeLocation.x, gazeLocation.y, gazeLocation.x / Screen.width, gazeLocation.y / Screen.height, gazeLocation.x / Screen.width, gazeLocation.y / Screen.height, _distance, _provider.EyeFeature, _blinking, System.DateTime.Now, new List<string>(aoiNameList)));
                UnloadCalibration();
            }
            if (_evaluationScript != null && _evaluationScript.Returned)
                UnloadEvaluation();

            //Click anchors are collected on EVERY render frame, BEFORE the fresh-sample gate below: the
            //camera runs at ~30fps while the display runs 60-144, so `wasPressedThisFrame` is true on
            //exactly one render frame that usually carries NO new camera sample — gating clicks on fresh
            //samples silently dropped most of them. The pre-click fixation window reads the gaze TRAIL,
            //which exists regardless of whether this frame produced a sample.
            if (onlineDriftCorrection && !PauseCSVLogging)
                CollectClickAnchor();

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

            //Capture-time bookkeeping for THIS sample (a fresh sample was consumed even when the blink
            //hold below freezes the reported position — the timestamp must not go stale for the fixation
            //aggregator / CSV latency column / pursuit pairing).
            _captureTimestamp = _provider.CaptureTimestamp;
            MeasuredLatencySeconds = (float)(Time.unscaledTimeAsDouble - _captureTimestamp);

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

                //Per-region error-model bias correction — the systematic offset the EVALUATION measured
                //at this screen region, subtracted UPSTREAM of the drift corrector so the two layers see
                //disjoint residuals (the corrector's anchors observe the error-model-corrected signal and
                //therefore learn only what remains — stacking them the other way around subtracted the
                //same bias twice once anchors accumulated). Applied only when the model was measured on
                //the calibration type that is actually active.
                if (useErrorModel && _errorModel != null && _errorModel.AppliesTo(_calibrations))
                {
                    var gazeNormForBias = new Vector2(gazeLocation.x / Screen.width, gazeLocation.y / Screen.height);
                    _errorModel.Query(gazeNormForBias, out var regionBias, out _, out _, out _);
                    gazeLocation -= new Vector2(regionBias.x * Screen.width, regionBias.y * Screen.height);
                }

                //The pre-corrector gaze trail the anchor sources sample from (normalized).
                var preCorrectorNorm = new Vector2(gazeLocation.x / Screen.width, gazeLocation.y / Screen.height);
                _gazeTrail.Add((_captureTimestamp, preCorrectorNorm));
                while (_gazeTrail.Count > 0 && _captureTimestamp - _gazeTrail[0].t > GazeTrailSeconds)
                    _gazeTrail.RemoveAt(0);

                //Online drift correction: a slowly-adapted affine layer fed by validated interaction
                //anchors. The anchor sources observe the PRE-corrector signal, so the corrector always
                //learns the full residual.
                if (onlineDriftCorrection)
                {
                    var corrected = _driftCorrector.Apply(preCorrectorNorm);
                    gazeLocation = new Vector2(corrected.x * Screen.width, corrected.y * Screen.height);
                }

                //Manual drift re-center, applied and captured AFTER the corrector: it corrects what the
                //corrector has not learned (capturing it pre-corrector made the two translations stack).
                if (!_hasRecentGaze) { _recentCalibratedGaze = gazeLocation; _hasRecentGaze = true; }
                else _recentCalibratedGaze = Vector2.Lerp(_recentCalibratedGaze, gazeLocation, 0.15f);
                if (_recenterArmedUntil > 0f && Time.unscaledTime >= _recenterArmedUntil)
                {
                    _driftOffset = new Vector2(Screen.width * 0.5f, Screen.height * 0.5f) - _recentCalibratedGaze;
                    _recenterArmedUntil = -1f;
                }
                gazeLocation += _driftOffset;

                //Attention-event anchors (click anchors are collected before the fresh-sample gate — see
                //the top of LateUpdate — because clicks land on render frames, most of which carry no new
                //camera sample).
                if (onlineDriftCorrection && !PauseCSVLogging)
                    CollectAttentionEventAnchors(preCorrectorNorm);

                //Apply filtering
                unfilteredGaze = gazeLocation;
                gazeLocation = SmoothGazeLocation(gazeLocation, _filtering);
            }

            //Update last gaze location timestamp
            var now = System.DateTime.Now;
            LastGazeLocationTimeUnix = ((System.DateTimeOffset)now).ToUnixTimeMilliseconds();

            //AOI updating (refills the reused scratch list; no per-frame allocation). The AOI stream uses
            //the fixation CENTROID of the unfiltered calibrated gaze (sqrt(N) noise reduction, no filter
            //phase lag) — the cursor above keeps the filtered signal; the two consumers deliberately differ.
            var aoiPoint = fixationAOILogging
                ? _fixationAggregator.Add(unfilteredGaze, _captureTimestamp)
                : gazeLocation;
            var aoiNorm = new Vector2(aoiPoint.x / Screen.width, aoiPoint.y / Screen.height);

            //Error ellipse for this screen region (the BIAS was already subtracted upstream, before the
            //drift corrector — subtracting it here too double-corrected); the covariance feeds P(AOI|...).
            float covXX = 0f, covXY = 0f, covYY = 0f;
            if (useErrorModel && _errorModel != null)
                _errorModel.Query(aoiNorm, out _, out covXX, out covXY, out covYY);

            _aoiManager.CheckAOIList(aoiNorm, aoiNameList);

            //Probabilistic AOI logging: P(AOI | fixation) under the region's error ellipse, throttled
            //(32-sample MC per AOI) and only while fixating (a saccade sample has no meaningful ellipse).
            if (useErrorModel && _errorModel != null &&
                (!fixationAOILogging || _fixationAggregator.InFixation) &&
                Time.unscaledTimeAsDouble - _lastProbabilityTime >= ProbabilityInterval)
            {
                _lastProbabilityTime = Time.unscaledTimeAsDouble;
                _aoiManager.CheckAOIProbabilities(aoiNorm, covXX, covXY, covYY, _aoiProbabilities);
                //Fold the calibrated probabilities into the logged AOI strings: "uID p=0.87". A top-2 gap
                //under 0.2 is flagged AMBIGUOUS — those fixations are coin flips and analysts must know.
                if (_aoiProbabilities.Count > 0)
                {
                    for (int i = 0; i < _aoiProbabilities.Count && i < 3; i++)
                        aoiNameList.Add($"{_aoiProbabilities[i].uID} p={_aoiProbabilities[i].probability:F2}");
                    //Only meaningful when the leader is a real candidate — two near-zero probabilities are
                    //"looking at neither", not an ambiguous hit.
                    if (_aoiProbabilities.Count >= 2 && _aoiProbabilities[0].probability >= 0.2f &&
                        _aoiProbabilities[0].probability - _aoiProbabilities[1].probability < 0.2f)
                        aoiNameList.Add("AMBIGUOUS");
                }
            }

            //CSV Logging. ShouldLog is checked BEFORE building the row: with logsPerSecond below the frame
            //rate the limiter drops most frames, so skipping the CSVData + AOI-list copy on those frames
            //avoids steady per-frame garbage. CSVData retains its AOI list by reference until the queue is
            //flushed, so accepted rows get their OWN copy (the scratch list is refilled every frame).
            if (!PauseCSVLogging && _csvLogger != null && _csvLogger.isActiveAndEnabled && _csvLogger.ShouldLog)
                _csvLogger.Append(new CSVData(gazeLocation.x, gazeLocation.y, gazeLocation.x / Screen.width, gazeLocation.y / Screen.height, unfilteredGaze.x / Screen.width, unfilteredGaze.y / Screen.height, _distance, _provider.EyeFeature, _blinking, now, new List<string>(aoiNameList), MeasuredLatencySeconds * 1000f));

            //Drowsy calibration
            if (_provider.IsCalibratingDrowsy)
                _provider.CalibrateDrowsy(false);

            //(Calibration/evaluation unload runs at the TOP of LateUpdate, above the Tick() gate --
            //see the comment there. Keeping a second copy here would be dead code.)
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
                //Top-right, clear of the (now taller) window on the left — and far easier to find than the
                //old bottom-left spot, which users routinely missed.
                if (GUI.Button(new Rect(Screen.width - Screen.width * 0.12f, Screen.height * 0.02f, Screen.width * 0.11f, Screen.height * 0.045f), $"{(gazeUIActivated ? "Hide" : "Show")} Gaze UI", _toggleStyle))
                    gazeUIActivated = !gazeUIActivated;

                //Drift re-center controls, top-right below the toggle (outside the left-half panel). Corrects
                //slow seating/posture drift in seconds: click Re-center, then look at the centre marker.
                if (gazeUIActivated)
                {
                    float bx = Screen.width - Screen.width * 0.12f;
                    float bw = Screen.width * 0.11f;
                    float bh = Screen.height * 0.045f;
                    if (GUI.Button(new Rect(bx, Screen.height * 0.075f, bw, bh), IsRecentering ? "Look centre…" : "Re-center drift", _toggleStyle))
                        RecenterDrift();
                    if (GUI.Button(new Rect(bx, Screen.height * 0.125f, bw, bh), "Clear drift", _toggleStyle))
                        ClearDrift();
                }
            }

            if (gazeUIActivated)
                gazeUI = GUI.Window(0, gazeUI, GazeUI, "");

            //Drift re-center marker: while armed, show a target at screen centre for the user to look at.
            if (IsRecentering)
            {
                var center = new Vector2(Screen.width * 0.5f, Screen.height * 0.5f);
                float size = Mathf.Max(48f, 48f * Screen.height / 1080f);
                float phase = Mathf.Repeat(Time.time * 1.5f, 1f);
                var prev = GUI.color;
                if (dot != null)
                {
                    GUI.color = new Color(1f, 1f, 1f, 1f - phase * 0.6f);
                    float ring = size * (1f + phase);
                    GUI.DrawTexture(new Rect(center.x - ring * 0.5f, center.y - ring * 0.5f, ring, ring), dot);
                    GUI.color = prev;
                    GUI.DrawTexture(new Rect(center.x - size * 0.5f, center.y - size * 0.5f, size, size), dot);
                }
                //Dedicated style: mutating the shared `style` here leaked MiddleCenter/white into the AOI
                //label below (it only resets fontSize), restyling it after the first re-center.
                _recenterStyle ??= new GUIStyle { alignment = TextAnchor.MiddleCenter, fontStyle = FontStyle.Bold };
                _recenterStyle.fontSize = Mathf.RoundToInt(24f * uiScale);
                _recenterStyle.normal.textColor = Color.white;
                //Clamp: if the face is lost while armed, the capture waits for the next tracked frame and
                //the raw countdown would run negative ("(-3)") — hold at 0 instead.
                int remaining = Mathf.Max(0, Mathf.CeilToInt(_recenterArmedUntil - Time.unscaledTime));
                GUI.Label(new Rect(center.x - Screen.width * 0.2f, center.y + size, Screen.width * 0.4f, size),
                    $"Look at the dot to re-center… ({remaining})", _recenterStyle);
                GUI.color = prev;
            }

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
                    //FilterVector2: allocation-free equivalent of Filter<Vector2> (no per-frame boxing).
                    //Resolution-normalize first: the 1€ cutoff rises with beta*|velocity|, and velocity is in
                    //pixels/second, so the same beta reacts far more strongly on a 3200-wide panel than at
                    //1080p (a given eye movement covers ~1.7x more pixels). Scale the gaze into a fixed
                    //reference space so beta means the same thing on every display, filter, then scale back.
                    //mincutoff/dcutoff are frequencies and are already resolution-independent.
                    float sx = OneEuroReferenceWidth / Mathf.Max(1, Screen.width);
                    float sy = OneEuroReferenceHeight / Mathf.Max(1, Screen.height);
                    var scaledIn = new Vector2(unfilteredGaze.x * sx, unfilteredGaze.y * sy);
                    var scaledOut = oneEuroFilter.FilterVector2(scaledIn, Time.realtimeSinceStartup);
                    smoothedGaze = new Vector2(scaledOut.x / sx, scaledOut.y / sy);
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

            //Same reasoning as LoadEvaluation: bail BEFORE mutating anything. _calibrationScript is
            //deliberately unwired in HomulerGazeCalibration.unity (that scene drives its own calibration
            //standalone), so the Gaze UI's Calibrate button reached this method with a null field and threw
            //below — after IsRendering, ClearDrift, BackupSettings and the overlay hiding had already run,
            //none of which RestoreSettings can undo unless UnloadCalibration gets to run.
            if (_calibrationScript == null)
            {
                UnitEyeLog.Warn($"LoadCalibration ignored: no {nameof(HomulerGazeCalibration)} assigned to " +
                                $"_calibrationScript on '{name}'.");
                return;
            }

            IsRendering = false;

            //A fresh calibration supersedes any drift re-centering correction.
            ClearDrift();

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

            //Configure BEFORE enabling: OnEnable rebuilds the presets from the current parameters, so
            //assigning padding/rounds after `enabled = true` (the old order) meant a repeat calibration in
            //the same session ran with the previous run's preset geometry.
            //Set _calibrations to none for a bit of performance gain
            _calibrationScript.calibrationType = _calibrations;
            _calibrations = Calibrations.None;

            //Calibration settings
            _calibrationScript.quitAfterCalibration = false;
            _calibrationScript.returnAfter = true;
            _calibrationScript.speed = speed;
            _calibrationScript.padding = padding;
            _calibrationScript.maxRoundsPerPreset = rounds;

            //Attach calibration to same gameObject
            _calibrationScript.enabled = true;
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
            //A fresh calibration deleted the old per-region error model (its biases measured the OLD fit)
            //— drop the in-memory copy too; the next evaluation rebuilds it.
            _errorModel = GazeErrorModel.Load(_gazeBackbone);
            //The corrector's anchors were collected against the old calibration's signal.
            ClearDrift();

            IsRendering = true;
        }

        /// <summary>
        /// Attaches GazeEvaluation script and backs up settings.
        /// </summary>
        /// <param name="rows">Number of rows in the dot grid</param>
        /// <param name="columns">Number of columns in the dot grid</param>
        public void LoadEvaluation(int rows = 5, int columns = 5)
        {
            //Resolve BEFORE touching any state. HomulerGazeEvaluation is optional — every other use of
            //_evaluationScript null-guards, and neither the UnitEyeUsingHomulerMediapipe prefab nor the
            //HomulerGazeCalibration scene carries one — so dereferencing it here threw AFTER the hide +
            //backup below had already run, and only UnloadEvaluation (gated on a non-null _evaluationScript)
            //can undo those: the scene was left with every overlay off and no runtime way back.
            var evaluation = GetComponent<HomulerGazeEvaluation>();
            if (evaluation == null)
            {
                UnitEyeLog.Warn($"LoadEvaluation ignored: no {nameof(HomulerGazeEvaluation)} component on '{name}'. " +
                                "Add one to this GameObject to run an evaluation from the Gaze UI.");
                return;
            }
            _evaluationScript = evaluation;

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

            //RestoreSettings rewinds _calibrations to the pre-evaluation backup, which would silently undo
            //the model the evaluation just reported as "Applied." on its results screen. Re-assign through
            //the PROPERTY (not the field RestoreSettings writes) so the store is loaded for the surviving
            //type. Must come after RestoreSettings, or the restore clobbers it again.
            if (_evaluationScript.AppliedCalibration.HasValue)
                Calibrations = _evaluationScript.AppliedCalibration.Value;

            //Append a note to csv entry
            if (_csvLogger != null && _csvLogger.isActiveAndEnabled)
                _csvLogger.AppendNote(_evaluationScript.ReturnMessage);

            //Destroy calibration script
            _evaluationScript.enabled = false;
            //Consume the return so LateUpdate does not call UnloadEvaluation again next frame
            _evaluationScript.ClearReturned();

            //The evaluation just measured + saved a fresh per-region error model — pick it up live.
            _errorModel = GazeErrorModel.Load(_gazeBackbone);

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
                case GazeBackbone.EyeMUPlusResNet34: return "Ensemble";
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
            //Tall enough to contain the Calibration profiles group (its controls sit at ~0.90 of this height).
            gazeUI.height = height * 0.96f;

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
                    mincutoff = GUI.HorizontalSlider(new Rect(width * 0.11f, height * 0.18f, width * 0.3f, height * 0.02f), mincutoff, minfloat, 3f, GUI.skin.horizontalSlider, gazeUIStyleHSThumb);
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

            //Calibration profiles: calibration is slow, so let the user SAVE the current calibration (for the
            //active backbone) under a name and LOAD it back later. Saved profiles go to StreamingAssets;
            //profiles committed to the repo (package Resources) are listed too. See CalibrationProfileStore.
            GUI.BeginGroup(new Rect(width * 0.01f, height * 0.82f, width * 0.48f, height * 0.15f));
            GUI.Box(new Rect(0, 0, width * 0.48f, height * 0.15f), "Calibration profiles (save/load)", gazeUIStyleBox);

            GUI.Label(new Rect(width * 0.02f, height * 0.03f, width * 0.08f, height * 0.04f), "Name:", gazeUIStyleLabel);
            _profileName = GUI.TextField(new Rect(width * 0.08f, height * 0.03f, width * 0.19f, height * 0.035f), _profileName ?? "");
            //Glasses state travels with the profile: glasses are worth ~1cm+ to appearance models, so a
            //calibration made with them silently degrades without them (and vice versa) — Load warns on
            //a mismatch against this toggle.
            userWearsGlasses = GUI.Toggle(new Rect(width * 0.275f, height * 0.03f, width * 0.055f, height * 0.035f),
                userWearsGlasses, "Glasses", gazeUIStyleButton);
            if (GUI.Button(new Rect(width * 0.335f, height * 0.028f, width * 0.12f, height * 0.04f), $"Save ({DisplayName(_gazeBackbone)})", gazeUIStyleButton))
            {
                _profileStatus = CalibrationProfileStore.Save(_profileName, _gazeBackbone, userWearsGlasses);
                _profileList = CalibrationProfileStore.List();
            }

            //Browse the available profiles and load the shown one.
            _profileList ??= CalibrationProfileStore.List();
            var hasProfiles = _profileList.Count > 0;
            var current = hasProfiles ? _profileList[Mathf.Clamp(_profileIndex, 0, _profileList.Count - 1)] : "(none)";
            if (GUI.Button(new Rect(width * 0.02f, height * 0.08f, width * 0.035f, height * 0.04f), "<", gazeUIStyleButton) && hasProfiles)
                _profileIndex = (_profileIndex - 1 + _profileList.Count) % _profileList.Count;
            GUI.Label(new Rect(width * 0.06f, height * 0.08f, width * 0.19f, height * 0.04f), current, gazeUIStyleLabel);
            if (GUI.Button(new Rect(width * 0.255f, height * 0.08f, width * 0.035f, height * 0.04f), ">", gazeUIStyleButton) && hasProfiles)
                _profileIndex = (_profileIndex + 1) % _profileList.Count;
            if (GUI.Button(new Rect(width * 0.30f, height * 0.08f, width * 0.09f, height * 0.04f), "Load", gazeUIStyleButton) && hasProfiles)
            {
                _profileStatus = CalibrationProfileStore.Load(current, userWearsGlasses);
                ReloadCalibration();
                _profileName = current;
            }
            if (GUI.Button(new Rect(width * 0.395f, height * 0.08f, width * 0.06f, height * 0.04f), "Refresh", gazeUIStyleButton))
                _profileList = CalibrationProfileStore.List();

            GUI.Label(new Rect(width * 0.02f, height * 0.125f, width * 0.44f, height * 0.02f), _profileStatus, gazeUIStyleLabel);

            GUI.EndGroup();
        }

        #endregion
    }
}
