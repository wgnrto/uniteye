using System;
using System.Collections.Generic;
using System.Globalization;
using System.IO;
using System.IO.Compression;
using System.Linq;
using UnitEye;
using UnityEditor;
using UnityEngine;

/// <summary>
/// Browse recorded calibration sessions and package the shareable ones into a single zip.
///
/// This is the "how do I actually send this?" half of the recorder. It lives in the EDITOR on purpose. The
/// runtime that a participant runs has no network code at all — a smoke test enforces that — so the promise
/// on the consent screen stays literally true, and no GitHub credential can end up inside a shipped build.
/// The one irreversible step, publishing, stays a deliberate act by a person who can see what is in the file.
///
/// Open with: UnitEye > Recorded Sessions
/// </summary>
public class RecordedSessionBrowser : EditorWindow
{
    private const string UploadPageUrl = "https://github.com/wgnrto/uniteye/issues/new";

    private class Session
    {
        public string Folder;
        public string Token;
        public GazeConsentRecord Consent;
        public int Samples = -1;
        public string Outcome = "";
        public float HoldoutRmseCm = -1f;
        public int ImagesDropped;
        public long Bytes;
        public bool Selected;
        public string BlockedReason;   // null = eligible to publish
    }

    private List<Session> _sessions;
    private Vector2 _scroll;
    private string _status = "";

    [MenuItem("UnitEye/Recorded Sessions")]
    public static void Open()
    {
        var w = GetWindow<RecordedSessionBrowser>(false, "Recorded Sessions");
        w.minSize = new Vector2(760, 380);
        w.Refresh();
    }

    private void OnEnable() { if (_sessions == null) Refresh(); }

    private void Refresh()
    {
        _sessions = new List<Session>();
        foreach (var publish in new[] { true, false })
        {
            var root = GazeSessionRecorder.RootFor(publish);
            if (!Directory.Exists(root)) continue;
            foreach (var dir in Directory.GetDirectories(root))
                _sessions.Add(Load(dir));
        }
        _sessions = _sessions.OrderByDescending(s => s.Consent?.consentedOnUtcDate ?? "").ToList();
        _status = _sessions.Count == 0
            ? $"No recordings found under {Path.Combine(Application.persistentDataPath, "UnitEyeRecordings")}."
            : $"{_sessions.Count} session(s).";
    }

    private static Session Load(string dir)
    {
        var s = new Session { Folder = dir, Token = Path.GetFileName(dir) };
        try
        {
            var consentPath = Path.Combine(dir, "consent.json");
            if (File.Exists(consentPath))
                s.Consent = JsonUtility.FromJson<GazeConsentRecord>(File.ReadAllText(consentPath));

            var summaryPath = Path.Combine(dir, "summary.json");
            if (File.Exists(summaryPath))
            {
                var text = File.ReadAllText(summaryPath);
                s.Samples = (int)ReadNumber(text, "samples", -1);
                s.ImagesDropped = (int)ReadNumber(text, "imagesDropped", 0);
                s.HoldoutRmseCm = ReadNumber(text, "holdoutRmseCm", -1f);
                s.Outcome = ReadString(text, "outcome");
            }

            foreach (var f in Directory.GetFiles(dir, "*", SearchOption.AllDirectories))
                s.Bytes += new FileInfo(f).Length;
        }
        catch (Exception e) { UnitEyeLog.Exception(e); }

        //Eligibility is decided here, once, so every button below agrees on it.
        var now = DateTime.UtcNow;
        if (s.Consent == null)
            //No consent record = no terms. Treat as unconsented and say so loudly; the recorder writes
            //consent.json before anything else, so this means the folder was tampered with or half-created.
            s.BlockedReason = "no consent.json - do not share; delete it";
        else if (!s.Consent.mayPublish)
            s.BlockedReason = "participant said local-only";
        else if (!s.Consent.PublicationHoldElapsed(now))
            s.BlockedReason = $"hold until {s.Consent.publicationHoldUntilUtcDate}";
        else if (s.Samples == 0)
            s.BlockedReason = "no samples captured";
        return s;
    }

    private void OnGUI()
    {
        EditorGUILayout.Space(6);
        using (new EditorGUILayout.HorizontalScope())
        {
            if (GUILayout.Button("Refresh", GUILayout.Width(90))) Refresh();
            if (GUILayout.Button("Open recordings folder", GUILayout.Width(180)))
                EditorUtility.RevealInFinder(Path.Combine(Application.persistentDataPath, "UnitEyeRecordings") + Path.DirectorySeparatorChar);
            GUILayout.FlexibleSpace();
            GUILayout.Label(_status, EditorStyles.miniLabel);
        }

        EditorGUILayout.HelpBox(
            "Only sessions whose participant agreed to publication AND whose 14-day hold has elapsed can be " +
            "packaged. Everything else is listed but cannot be selected - that hold is the promise the consent " +
            "screen made, and this window is the only thing enforcing it.",
            MessageType.Info);

        if (_sessions == null || _sessions.Count == 0)
        {
            EditorGUILayout.LabelField("Nothing recorded yet.");
            return;
        }

        _scroll = EditorGUILayout.BeginScrollView(_scroll);
        foreach (var s in _sessions)
        {
            using (new EditorGUILayout.VerticalScope(EditorStyles.helpBox))
            using (new EditorGUILayout.HorizontalScope())
            {
                bool eligible = s.BlockedReason == null;
                using (new EditorGUI.DisabledScope(!eligible))
                    s.Selected = EditorGUILayout.Toggle(s.Selected && eligible, GUILayout.Width(18));

                var tier = s.Consent != null ? s.Consent.tier.ToString() : "?";
                EditorGUILayout.LabelField($"{s.Token}", EditorStyles.boldLabel, GUILayout.Width(130));
                EditorGUILayout.LabelField(tier, GUILayout.Width(90));
                EditorGUILayout.LabelField($"{(s.Samples < 0 ? "?" : s.Samples.ToString())} samples", GUILayout.Width(95));
                EditorGUILayout.LabelField(FormatBytes(s.Bytes), GUILayout.Width(75));
                EditorGUILayout.LabelField(
                    s.HoldoutRmseCm >= 0f ? $"{s.HoldoutRmseCm.ToString("F2", CultureInfo.InvariantCulture)} cm" : "-",
                    GUILayout.Width(65));

                if (eligible)
                    EditorGUILayout.LabelField("publishable", EditorStyles.miniLabel);
                else
                {
                    var prev = GUI.color;
                    GUI.color = new Color(1f, 0.75f, 0.4f);
                    EditorGUILayout.LabelField(s.BlockedReason, EditorStyles.miniLabel);
                    GUI.color = prev;
                }

                if (s.ImagesDropped > 0)
                    EditorGUILayout.LabelField($"{s.ImagesDropped} imgs dropped", EditorStyles.miniLabel, GUILayout.Width(115));

                if (GUILayout.Button("Show", GUILayout.Width(50)))
                    EditorUtility.RevealInFinder(s.Folder + Path.DirectorySeparatorChar);
                if (GUILayout.Button("Delete", GUILayout.Width(60)) &&
                    EditorUtility.DisplayDialog("Delete recording?",
                        $"Permanently delete {s.Token}?\n\nUse this to honour a withdrawal request.", "Delete", "Cancel"))
                {
                    try { Directory.Delete(s.Folder, true); } catch (Exception e) { UnitEyeLog.Exception(e); }
                    Refresh();
                    GUIUtility.ExitGUI();
                }
            }
        }
        EditorGUILayout.EndScrollView();

        var selected = _sessions.Where(s => s.Selected && s.BlockedReason == null).ToList();
        EditorGUILayout.Space(4);
        using (new EditorGUILayout.HorizontalScope())
        {
            if (GUILayout.Button("Select all publishable", GUILayout.Width(160)))
                foreach (var s in _sessions) s.Selected = s.BlockedReason == null;
            if (GUILayout.Button("Select none", GUILayout.Width(100)))
                foreach (var s in _sessions) s.Selected = false;
            GUILayout.FlexibleSpace();
            using (new EditorGUI.DisabledScope(selected.Count == 0))
            {
                if (GUILayout.Button($"Package {selected.Count} session(s) into a zip…", GUILayout.Height(26), GUILayout.Width(250)))
                    Package(selected, andOpenUpload: false);
                if (GUILayout.Button("Package and upload…", GUILayout.Height(26), GUILayout.Width(170)))
                    Package(selected, andOpenUpload: true);
            }
        }
        EditorGUILayout.Space(4);
    }

    private void Package(List<Session> sessions, bool andOpenUpload)
    {
        //Say exactly what is about to leave the machine. "3 sessions" is not informed; tiers are, because
        //that is the difference between numbers and a video of someone's room.
        var tiers = string.Join(", ", sessions.Select(s => s.Consent.tier.ToString()).Distinct());
        long total = sessions.Sum(s => s.Bytes);
        bool hasImagery = sessions.Any(s => s.Consent.tier >= GazeRecordingTier.EyeCrops);
        var warning = hasImagery
            ? "\n\nThis includes IMAGES of people. Once published it cannot be fully retracted."
            : "";
        if (!EditorUtility.DisplayDialog("Package recordings?",
                $"{sessions.Count} session(s), {FormatBytes(total)}.\nTiers: {tiers}{warning}\n\n" +
                "Every one of these consented to publication and is past its hold date.",
                "Create zip", "Cancel"))
            return;

        var path = EditorUtility.SaveFilePanel("Save dataset zip", "",
            $"uniteye-sessions-{sessions.Count}.zip", "zip");
        if (string.IsNullOrEmpty(path)) return;

        try
        {
            if (File.Exists(path)) File.Delete(path);
            //ZipArchive rather than ZipFile.CreateFromDirectory: the latter lives in an assembly Unity does
            //not reference by default, and this way each entry path is explicit rather than inherited from a
            //machine-specific absolute path (which would embed the account name in the archive).
            using (var fs = new FileStream(path, FileMode.CreateNew))
            using (var zip = new ZipArchive(fs, ZipArchiveMode.Create))
            {
                foreach (var s in sessions)
                {
                    var files = Directory.GetFiles(s.Folder, "*", SearchOption.AllDirectories);
                    for (int i = 0; i < files.Length; i++)
                    {
                        if (EditorUtility.DisplayCancelableProgressBar("Packaging", $"{s.Token} ({i + 1}/{files.Length})",
                                (float)i / Mathf.Max(1, files.Length)))
                            throw new OperationCanceledException();
                        var relative = files[i].Substring(s.Folder.Length).TrimStart('\\', '/').Replace('\\', '/');
                        //Qualified: UnityEngine defines a CompressionLevel too, and the two are unrelated.
                        var entry = zip.CreateEntry($"{s.Token}/{relative}",
                            System.IO.Compression.CompressionLevel.Optimal);
                        using (var src = File.OpenRead(files[i]))
                        using (var dst = entry.Open())
                            src.CopyTo(dst);
                    }
                }
            }
            EditorUtility.ClearProgressBar();
            _status = $"Wrote {FormatBytes(new FileInfo(path).Length)} to {Path.GetFileName(path)}";
            EditorUtility.RevealInFinder(path);

            if (andOpenUpload)
            {
                //Deliberately hands off to the browser instead of POSTing. No token to create or store, the
                //file is visible before it is attached, and the click that actually publishes belongs to a
                //human looking at what they are publishing.
                Application.OpenURL(UploadPageUrl);
                EditorUtility.DisplayDialog("Ready to upload",
                    "The zip is selected in your file browser and the upload page is open.\n\n" +
                    "Drag the zip into the box to attach it.\n\n" +
                    "Nothing has been sent yet - attaching and submitting is your call.",
                    "Got it");
            }
        }
        catch (OperationCanceledException)
        {
            EditorUtility.ClearProgressBar();
            try { if (File.Exists(path)) File.Delete(path); } catch { }
            _status = "Cancelled.";
        }
        catch (Exception e)
        {
            EditorUtility.ClearProgressBar();
            UnitEyeLog.Exception(e);
            _status = $"Failed: {e.Message}";
        }
    }

    private static string FormatBytes(long b)
    {
        if (b >= 1L << 30) return $"{b / (float)(1L << 30):F1} GB";
        if (b >= 1L << 20) return $"{b / (float)(1L << 20):F1} MB";
        if (b >= 1L << 10) return $"{b / (float)(1L << 10):F0} KB";
        return $"{b} B";
    }

    //Minimal scalar readers: summary.json is written by us and is flat, so a full JSON dependency here would
    //be more machinery than the two fields warrant.
    private static float ReadNumber(string json, string key, float fallback)
    {
        var needle = $"\"{key}\":";
        int i = json.IndexOf(needle, StringComparison.Ordinal);
        if (i < 0) return fallback;
        int start = i + needle.Length;
        int end = start;
        while (end < json.Length && json[end] != ',' && json[end] != '}') end++;
        return float.TryParse(json.Substring(start, end - start).Trim(), NumberStyles.Float,
            CultureInfo.InvariantCulture, out var v) ? v : fallback;
    }

    private static string ReadString(string json, string key)
    {
        var needle = $"\"{key}\":\"";
        int i = json.IndexOf(needle, StringComparison.Ordinal);
        if (i < 0) return "";
        int start = i + needle.Length;
        int end = json.IndexOf('"', start);
        return end > start ? json.Substring(start, end - start) : "";
    }
}
