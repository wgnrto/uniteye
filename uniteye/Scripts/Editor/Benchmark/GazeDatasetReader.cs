using System;
using System.Collections.Generic;
using System.Globalization;
using System.IO;
using UnityEngine;

namespace UnitEye.Benchmark
{
    /// <summary>
    /// Reads a recorded calibration session back off disk — the exact inverse of GazeSessionRecorder.
    ///
    /// Editor-only by placement: this is analysis tooling, and keeping it out of the runtime assembly means
    /// nothing here can end up in a shipped player.
    /// </summary>
    public static class GazeDatasetReader
    {
        public class Sample
        {
            public int Index;
            public double SessionSeconds;
            public Vector2 LabelPx;
            public Vector2 TargetPx;
            public bool Dwell;
            public bool HeadRotation;
            public int Preset;
            public int Round;
            public bool Blinking;
            public float[] Features;
            public float DistanceMm = float.NaN;   // NaN = the row omitted it
        }

        public class Session
        {
            public string Folder;
            public string Token;
            public GazeConsentRecord Consent;
            public string Backbone = "";
            public int ScreenWidthPx, ScreenHeightPx;
            public float ScreenWidthCm, ScreenHeightCm;
            /// <summary>
            /// Whether this session's centimetre figures describe real centimetres. False for anything
            /// recorded in a windowed Editor Game view, where Screen.dpi describes the monitor while
            /// Screen.width describes a smaller render surface. % of diagonal stays valid either way (it is
            /// a ratio of the same units); degrees of visual angle does not.
            /// </summary>
            public bool PhysicalScaleTrustworthy;
            public List<Sample> Samples = new List<Sample>();
            public float AppHoldoutRmseCm = -1f;
            public string Status = "ok";

            public bool Usable => Status == "ok" && Samples.Count > 0;
            /// <summary>Feature layout key. Models must never be trained across differing layouts.</summary>
            public string GroupKey => $"{Backbone}/{(Samples.Count > 0 ? Samples[0].Features.Length : 0)}";
        }

        /// <summary>
        /// Loads every session under both recording roots. Sessions that cannot be used are returned with a
        /// Status explaining why rather than skipped — silent exclusion is how a benchmark ends up quietly
        /// reporting on a different population than you think.
        /// </summary>
        public static List<Session> LoadAll()
        {
            var sessions = new List<Session>();
            foreach (var publish in new[] { true, false })
            {
                var root = GazeSessionRecorder.RootFor(publish);
                if (!Directory.Exists(root)) continue;
                foreach (var dir in Directory.GetDirectories(root))
                    sessions.Add(Load(dir));
            }
            return sessions;
        }

        public static Session Load(string folder)
        {
            var s = new Session { Folder = folder, Token = Path.GetFileName(folder) };
            try
            {
                //Consent gating on READ as well as on publish: a folder with no terms must not feed anything.
                var consentPath = Path.Combine(folder, "consent.json");
                if (!File.Exists(consentPath)) { s.Status = "excluded:no-consent"; return s; }
                s.Consent = JsonUtility.FromJson<GazeConsentRecord>(File.ReadAllText(consentPath));
                if (s.Consent == null) { s.Status = "excluded:unreadable-consent"; return s; }

                var sessionPath = Path.Combine(folder, "session.json");
                if (File.Exists(sessionPath))
                {
                    var j = File.ReadAllText(sessionPath);
                    s.Backbone = Str(j, "backbone");
                    s.ScreenWidthPx = (int)Num(j, "screenWidthPx", 0);
                    s.ScreenHeightPx = (int)Num(j, "screenHeightPx", 0);
                    s.ScreenWidthCm = Num(j, "screenWidthCm", 0f);
                    s.ScreenHeightCm = Num(j, "screenHeightCm", 0f);
                    //Absent in sessions recorded before this field existed. Defaulting to FALSE is the safe
                    //direction: an old session's physical scale is genuinely unknown, and treating unknown as
                    //trustworthy is how a bad centimetre figure ends up averaged into a headline number.
                    s.PhysicalScaleTrustworthy = j.Contains("\"physicalScaleTrustworthy\":true");
                }
                if (s.ScreenWidthPx <= 0 || s.ScreenHeightPx <= 0) { s.Status = "excluded:no-screen-geometry"; return s; }

                var summaryPath = Path.Combine(folder, "summary.json");
                if (File.Exists(summaryPath))
                    s.AppHoldoutRmseCm = Num(File.ReadAllText(summaryPath), "holdoutRmseCm", -1f);

                var rowsPath = Path.Combine(folder, "samples.jsonl");
                var blobPath = Path.Combine(folder, "features.f32");
                if (!File.Exists(rowsPath) || !File.Exists(blobPath)) { s.Status = "excluded:missing-data"; return s; }

                var blob = File.ReadAllBytes(blobPath);
                var lines = File.ReadAllLines(rowsPath);
                foreach (var line in lines)
                {
                    //A crash mid-session truncates the final line. Skip malformed rows rather than abort the
                    //whole session — the preceding rows are perfectly good data.
                    if (string.IsNullOrWhiteSpace(line) || !line.StartsWith("{") || !line.EndsWith("}")) continue;

                    var offset = (long)Num(line, "featureOffset", -1);
                    var count = (int)Num(line, "featureCount", 0);
                    if (offset < 0 || count <= 0) continue;
                    if (offset + count * 4L > blob.Length) continue;   // truncated blob

                    var f = new float[count];
                    Buffer.BlockCopy(blob, (int)offset, f, 0, count * 4);

                    s.Samples.Add(new Sample
                    {
                        Index = (int)Num(line, "\"i\"", 0, keyIsRaw: true),
                        SessionSeconds = Num(line, "t", 0f),
                        LabelPx = new Vector2(Num(line, "labelX", 0f), Num(line, "labelY", 0f)),
                        TargetPx = new Vector2(Num(line, "targetX", 0f), Num(line, "targetY", 0f)),
                        Dwell = Flag(line, "dwell"),
                        HeadRotation = Flag(line, "headRotation"),
                        Preset = (int)Num(line, "preset", 0),
                        Round = (int)Num(line, "round", 0),
                        Blinking = Flag(line, "blinking"),
                        DistanceMm = Num(line, "distanceMm", float.NaN),
                        Features = f,
                    });
                }

                if (s.Samples.Count == 0) { s.Status = "excluded:no-samples"; return s; }

                //Jagged feature vectors cannot be trained together. The capture path drops odd-length rows
                //before the recorder ever sees them, so this means the file is damaged - exclude the whole
                //session rather than silently dropping rows and changing what is being measured.
                var len = s.Samples[0].Features.Length;
                foreach (var sample in s.Samples)
                    if (sample.Features.Length != len) { s.Status = "excluded:jagged-features"; return s; }
            }
            catch (Exception e)
            {
                UnitEyeLog.Exception(e);
                s.Status = "excluded:error";
            }
            return s;
        }

        //Hand-rolled scalar readers matching GazeSessionRecorder's hand-built JSON. InvariantCulture on the
        //way in as well as out; a de-DE parse of "0.4193" would otherwise silently yield 4193.
        private static float Num(string json, string key, float fallback, bool keyIsRaw = false)
        {
            var needle = keyIsRaw ? key + ":" : $"\"{key}\":";
            int i = json.IndexOf(needle, StringComparison.Ordinal);
            if (i < 0) return fallback;
            int start = i + needle.Length, end = start;
            while (end < json.Length && json[end] != ',' && json[end] != '}') end++;
            var text = json.Substring(start, end - start).Trim();
            if (text == "null") return float.NaN;
            return float.TryParse(text, NumberStyles.Float, CultureInfo.InvariantCulture, out var v) ? v : fallback;
        }

        private static bool Flag(string json, string key)
        {
            var needle = $"\"{key}\":";
            int i = json.IndexOf(needle, StringComparison.Ordinal);
            return i >= 0 && json.Substring(i + needle.Length).StartsWith("true", StringComparison.Ordinal);
        }

        private static string Str(string json, string key)
        {
            var needle = $"\"{key}\":\"";
            int i = json.IndexOf(needle, StringComparison.Ordinal);
            if (i < 0) return "";
            int start = i + needle.Length, end = json.IndexOf('"', start);
            return end > start ? json.Substring(start, end - start) : "";
        }
    }
}
