using System;
using UnityEngine;
using UnityEngine.InputSystem;

namespace UnitEye
{
    /// <summary>
    /// The participant-facing consent gate for recording a calibration session. Put it on the same
    /// GameObject as <see cref="HomulerGazeCalibration"/>; when <see cref="askBeforeCalibration"/> is on, the
    /// calibration will not start until the participant has answered.
    ///
    /// Everything here is deliberately boring IMGUI drawn from the verbatim strings in
    /// <see cref="GazeConsentTexts"/>. No prefab, no localisation indirection, no editable text field: what
    /// is on screen has to be exactly what the stored SHA-256 covers, or the record proves nothing.
    ///
    /// The component NEVER uploads. It writes a folder and shows the participant where it is. Publication is
    /// a human decision made later, out of band, against the consent record it leaves behind.
    /// </summary>
    //Grouped under UnitEye in Add Component: without this the class is reachable only by typing its exact
    //name, which is a poor way to find a component most users will not know exists.
    [AddComponentMenu("UnitEye/Calibration Recording Consent")]
    [DisallowMultipleComponent]
    public class CalibrationRecordingConsent : MonoBehaviour
    {
        public enum State { NotAsked, Asking, Granted, Declined }

        [Tooltip("Ask the participant before every calibration. Off = never record, and no screens are shown.")]
        public bool askBeforeCalibration = false;

        [Tooltip("Highest tier the participant may choose. Lower it for studies that must not collect imagery.")]
        public GazeRecordingTier maxTierOffered = GazeRecordingTier.EyeCrops;

        [Tooltip("How a participant reaches you to withdraw their data. Shown verbatim and stored in consent.json.")]
        public string withdrawalContact = "";

        [Tooltip("Label for this study/session set. Never a participant name.")]
        public string studyLabel = "";

        public State Status { get; private set; } = State.NotAsked;
        public GazeConsentRecord Record { get; private set; }
        public GazeRecordingTier Tier { get; private set; } = GazeRecordingTier.Off;

        /// <summary>Set by the calibration once a session has been written, so the final screen can offer deletion.</summary>
        public GazeSessionRecorder ActiveRecorder { get; set; }

        private int _screen;                       // 0 intro, 1 tier, 2 full-frame confirm, 3 publication, 4 done
        private GazeRecordingTier _pendingTier = GazeRecordingTier.Landmarks;
        private int _publishChoice = -1;           // -1 unanswered, 0 no, 1 yes
        private GUIStyle _body, _heading;
        private bool _showDoneScreen;
        private bool _deleted;

        /// <summary>True while the participant still owes an answer — the calibration must not start.</summary>
        public bool Blocking => askBeforeCalibration && Status == State.Asking;

        /// <summary>Whether a recording should be created for the run about to start.</summary>
        public bool ShouldRecord => Status == State.Granted && Tier != GazeRecordingTier.Off;

        /// <summary>
        /// Begins the flow. Called by the calibration on enable. Recording is unavailable on WebGL — the
        /// browser build loads its CV from third-party CDNs, so a "nothing leaves this computer" promise
        /// would not be true there, and the provider exposes no landmarks or imagery anyway.
        /// </summary>
        public void BeginIfNeeded()
        {
            Reset();
#if UNITY_WEBGL && !UNITY_EDITOR
            askBeforeCalibration = false;
#endif
            if (!askBeforeCalibration) { Status = State.NotAsked; return; }
            Status = State.Asking;
            _screen = 0;
        }

        public void Reset()
        {
            Status = State.NotAsked;
            Record = null;
            Tier = GazeRecordingTier.Off;
            ActiveRecorder = null;
            _screen = 0;
            _publishChoice = -1;
            _pendingTier = (GazeRecordingTier)Mathf.Min((int)GazeRecordingTier.Landmarks, (int)maxTierOffered);
            _showDoneScreen = false;
            _deleted = false;
        }

        /// <summary>Shows the closing screen with the withdrawal code once the session has been written.</summary>
        public void ShowCompletionScreen() { if (Status == State.Granted) { _showDoneScreen = true; _screen = 4; } }

        private void EnsureStyles()
        {
            if (_body != null) return;
            _body = new GUIStyle(GUI.skin.label) { wordWrap = false, richText = false, alignment = TextAnchor.UpperLeft };
            _heading = new GUIStyle(GUI.skin.label) { fontStyle = FontStyle.Bold, alignment = TextAnchor.UpperLeft };
            float scale = Mathf.Max(1f, Screen.height / 1080f);
            _body.fontSize = Mathf.RoundToInt(15 * scale);
            _heading.fontSize = Mathf.RoundToInt(20 * scale);
        }

        private void OnGUI()
        {
            if (Status != State.Asking && !_showDoneScreen) return;
            EnsureStyles();

            //Opaque backdrop: the calibration overlay must not show through a consent decision.
            GUIShapes.FillRect(new Rect(0, 0, Screen.width, Screen.height), new Color(0.06f, 0.06f, 0.08f, 0.97f));

            float m = Mathf.Round(Screen.width * 0.08f);
            var area = new Rect(m, m * 0.5f, Screen.width - 2 * m, Screen.height - m);
            GUILayout.BeginArea(area);

            switch (_screen)
            {
                case 0: DrawIntro(); break;
                case 1: DrawTierChoice(); break;
                case 2: DrawFullFrameConfirm(); break;
                case 3: DrawPublication(); break;
                case 4: DrawDone(); break;
            }

            GUILayout.EndArea();
        }

        private void DrawIntro()
        {
            GUILayout.Label(GazeConsentTexts.Intro, _body);
            GUILayout.Space(20);
            GUILayout.BeginHorizontal();
            if (GUILayout.Button("Continue", GUILayout.Height(40))) _screen = 1;
            GUILayout.Space(20);
            if (GUILayout.Button("No thanks - just calibrate", GUILayout.Height(40))) Decline();
            GUILayout.EndHorizontal();
        }

        private void DrawTierChoice()
        {
            GUILayout.Label(GazeConsentTexts.TierChoice, _body);
            GUILayout.Space(16);
            for (var t = GazeRecordingTier.Features; t <= maxTierOffered; t++)
            {
                bool selected = _pendingTier == t;
                if (GUILayout.Toggle(selected, $"  {(int)t} - {GazeConsentTexts.Describe(t)}", GUILayout.Height(26)) && !selected)
                    _pendingTier = t;
            }
            GUILayout.Space(16);
            GUILayout.BeginHorizontal();
            if (GUILayout.Button("Continue", GUILayout.Height(40)))
                _screen = _pendingTier == GazeRecordingTier.FullFrames ? 2 : 3;
            GUILayout.Space(20);
            if (GUILayout.Button("Back", GUILayout.Height(40))) _screen = 0;
            GUILayout.Space(20);
            if (GUILayout.Button("Cancel - record nothing", GUILayout.Height(40))) Decline();
            GUILayout.EndHorizontal();
        }

        private void DrawFullFrameConfirm()
        {
            GUILayout.Label(GazeConsentTexts.FullFrameConfirm, _body);
            //Live preview: people agree to "video of the room" without picturing their actual room. Showing
            //the real feed is the only way this confirmation means anything.
            var provider = GetComponent<HomulerGaze>() != null ? GetComponent<HomulerGaze>().Provider : null;
            var source = provider as IGazeRecordingSource;
            var cam = source != null ? source.CameraTexture : null;
            if (cam != null)
            {
                float w = Mathf.Min(Screen.width * 0.5f, 640f);
                float h = w * cam.height / Mathf.Max(1, cam.width);
                GUILayout.Space(10);
                GUI.DrawTexture(GUILayoutUtility.GetRect(w, h, GUILayout.Width(w), GUILayout.Height(h)), cam, ScaleMode.ScaleToFit);
            }
            GUILayout.Space(16);
            GUILayout.BeginHorizontal();
            if (GUILayout.Button("Yes, save video of my face and my room", GUILayout.Height(40))) _screen = 3;
            GUILayout.Space(20);
            if (GUILayout.Button("Go back and choose less", GUILayout.Height(40))) _screen = 1;
            GUILayout.EndHorizontal();
        }

        private void DrawPublication()
        {
            GUILayout.Label(GazeConsentTexts.Publication, _body);
            GUILayout.Space(10);
            GUILayout.Label($"Withdrawal contact: {(string.IsNullOrWhiteSpace(withdrawalContact) ? "(not configured)" : withdrawalContact)}", _body);
            GUILayout.Space(16);
            //Neither preselected: a default here would be a nudge on the one question that is hardest to undo.
            if (GUILayout.Toggle(_publishChoice == 1, "  Yes, this may be published under the terms above", GUILayout.Height(26))) _publishChoice = 1;
            if (GUILayout.Toggle(_publishChoice == 0, "  No - keep it on this computer only", GUILayout.Height(26))) _publishChoice = 0;
            GUILayout.Space(16);
            GUILayout.BeginHorizontal();
            GUI.enabled = _publishChoice >= 0;
            if (GUILayout.Button("Start calibration", GUILayout.Height(40))) Grant();
            GUI.enabled = true;
            GUILayout.Space(20);
            if (GUILayout.Button("Cancel - record nothing", GUILayout.Height(40))) Decline();
            GUILayout.EndHorizontal();
        }

        private void DrawDone()
        {
            if (_deleted)
            {
                GUILayout.Label("Deleted.\n\nYour recording has been removed from this computer.", _heading);
                GUILayout.Space(20);
                if (GUILayout.Button("Done", GUILayout.Height(40))) { _showDoneScreen = false; }
                return;
            }
            GUILayout.Label("Saved.", _heading);
            GUILayout.Space(10);
            GUILayout.Label($"Your code:   {Record?.participantToken}", _body);
            GUILayout.Label($"Saved:       {GazeConsentTexts.Describe(Tier)}", _body);
            GUILayout.Label($"Published?   {(Record != null && Record.mayPublish ? "you agreed it may be" : "no - this computer only")}", _body);
            if (Record != null && Record.mayPublish)
                GUILayout.Label($"Not before:  {Record.publicationHoldUntilUtcDate}", _body);
            GUILayout.Label($"To withdraw: {Record?.withdrawalContact}", _body);
            GUILayout.Space(20);
            GUILayout.BeginHorizontal();
            if (GUILayout.Button("Delete my recording now", GUILayout.Height(40)))
            {
                ActiveRecorder?.DeleteEverything();
                _deleted = true;
            }
            GUILayout.Space(20);
            if (GUILayout.Button("Done", GUILayout.Height(40))) _showDoneScreen = false;
            GUILayout.EndHorizontal();
        }

        private void Grant()
        {
            Tier = _pendingTier;
            //DateTime.UtcNow only here, and only its DATE is stored — see GazeConsentRecord.
            Record = GazeConsentRecord.Create(Tier, _publishChoice == 1, DateTime.UtcNow, withdrawalContact, studyLabel);
            Status = State.Granted;
        }

        private void Decline()
        {
            Tier = GazeRecordingTier.Off;
            Record = null;
            Status = State.Declined;
        }
    }
}
