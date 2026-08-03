using System;
using System.Collections.Generic;
using System.IO;
using Newtonsoft.Json;
using UnityEngine;

namespace UnitEye
{
    /// <summary>
    /// A saved calibration profile: a named, self-contained snapshot of the calibration files under
    /// StreamingAssets/Calibration Files/ for one gaze backbone (the RidgeRegression X/Y JSONs and/or the
    /// MLP JSON). Calibration is slow, so a profile lets a good calibration be kept and restored later
    /// instead of redone. One profile serializes to ONE JSON file, which makes it trivial to keep, share, or
    /// commit to the repo.
    /// </summary>
    [Serializable]
    public class CalibrationProfile
    {
        public string name;
        public string createdUtc;
        public string backbone;   // informational: the active gaze backbone when the profile was saved
        //Whether the user wore glasses when this calibration was made. Glasses are worth ~1cm+ of error
        //to appearance models, and a profile calibrated WITH glasses silently degrades without them (and
        //vice versa) — consumers should warn on a mismatch with the current session.
        public bool wearingGlasses;
        //Relative path under "Calibration Files" (e.g. "RidgeRegression/Reg_X_EyeMU.json") -> raw JSON content.
        public Dictionary<string, string> files = new Dictionary<string, string>();
    }

    /// <summary>
    /// Saves / loads / lists <see cref="CalibrationProfile"/>s. User-saved profiles are written to a
    /// writable StreamingAssets folder; profiles shipped in the package (committed to the repo) live under
    /// Resources/<see cref="ResourcesFolder"/> and are read-only. Both appear in <see cref="List"/>.
    /// </summary>
    public static class CalibrationProfileStore
    {
        public const string Extension = ".json";
        //Read-only profiles shipped in the package Resources (so they travel with the repo).
        public const string ResourcesFolder = "CalibrationProfiles";
        //The calibration-file subfolders a profile snapshots (mirrors RidgeRegression.Save / SimpleMLP.Save).
        private static readonly string[] Subfolders = { "RidgeRegression", "MLP" };

        private static string CalibrationRoot =>
            Path.Combine(Application.streamingAssetsPath, "Calibration Files");
        //Writable location for user-saved profiles (a sibling of the RidgeRegression/MLP folders).
        private static string UserProfilesDir => Path.Combine(CalibrationRoot, "Profiles");

        /// <summary>
        /// Snapshots the current calibration files for <paramref name="backbone"/> into a user profile file.
        /// Returns a short status string for the UI.
        /// </summary>
        public static string Save(string name, GazeBackbone backbone, bool wearingGlasses = false)
        {
            name = Sanitize(name);
            if (string.IsNullOrEmpty(name))
                return "Enter a profile name first.";

            var files = CollectCurrentFiles(backbone);
            if (files.Count == 0)
                return $"No {backbone} calibration to save — calibrate first.";

            var profile = new CalibrationProfile
            {
                name = name,
                createdUtc = DateTime.UtcNow.ToString("o"),
                backbone = backbone.ToString(),
                wearingGlasses = wearingGlasses,
                files = files,
            };

            try
            {
                Directory.CreateDirectory(UserProfilesDir);
                File.WriteAllText(Path.Combine(UserProfilesDir, name + Extension),
                    JsonConvert.SerializeObject(profile, Formatting.Indented));
            }
            catch (Exception e)
            {
                UnitEyeLog.Exception(e);
                return $"Could not save profile '{name}'.";
            }
            return $"Saved profile '{name}' ({files.Count} file(s)).";
        }

        /// <summary>
        /// Restores a named profile's files over the active calibration files. The caller reloads the
        /// calibration model afterward. Returns a short status string for the UI.
        /// </summary>
        public static string Load(string name) => Load(name, null);

        /// <summary>
        /// Load with a glasses check: pass the CURRENT session's glasses state and the status warns when
        /// it mismatches the state the profile was calibrated with (worth ~1cm+ of silent error).
        /// </summary>
        public static string Load(string name, bool? currentlyWearingGlasses)
        {
            name = Sanitize(name);
            var json = ReadProfileJson(name);
            if (json == null)
                return $"Profile '{name}' not found.";

            CalibrationProfile profile;
            try { profile = JsonConvert.DeserializeObject<CalibrationProfile>(json); }
            catch (Exception e) { UnitEyeLog.Exception(e); return $"Profile '{name}' is corrupt."; }

            if (profile?.files == null || profile.files.Count == 0)
                return $"Profile '{name}' has no calibration files.";

            var written = 0;
            foreach (var kv in profile.files)
                if (WriteCalibrationFile(kv.Key, kv.Value))
                    written++;
            if (written == 0)
                return $"Profile '{name}' contained no usable calibration files.";

            //Companion files NOT in the profile must not survive the restore: a TPS warp or per-region
            //error model is only valid for the exact ridge fit it was measured on — leaving the previous
            //calibration's on disk cross-pairs it with the restored fit and corrupts gaze/AOI output.
            if (!string.IsNullOrEmpty(profile.backbone))
            {
                foreach (var companion in new[] { "Warp", "ErrorModel" })
                {
                    var rel = $"RidgeRegression/{companion}_{profile.backbone}{Extension}";
                    if (profile.files.ContainsKey(rel)) continue;
                    try
                    {
                        var stale = Path.Combine(CalibrationRoot, "RidgeRegression", $"{companion}_{profile.backbone}{Extension}");
                        if (File.Exists(stale)) File.Delete(stale);
                    }
                    catch (Exception e) { UnitEyeLog.Exception(e); }
                }
            }

            var glassesWarning = currentlyWearingGlasses.HasValue && profile.wearingGlasses != currentlyWearingGlasses.Value
                ? (profile.wearingGlasses
                    ? " WARNING: profile was calibrated WITH glasses — accuracy degrades without them."
                    : " WARNING: profile was calibrated WITHOUT glasses — accuracy degrades with them.")
                : "";
            return $"Loaded profile '{name}' ({written} file(s)).{glassesWarning}";
        }

        /// <summary>All available profile names: user-saved (StreamingAssets) + shipped (Resources), deduped and sorted.</summary>
        public static List<string> List()
        {
            var names = new SortedSet<string>(StringComparer.OrdinalIgnoreCase);
            if (Directory.Exists(UserProfilesDir))
                foreach (var f in Directory.GetFiles(UserProfilesDir, "*" + Extension))
                    names.Add(Path.GetFileNameWithoutExtension(f));
            foreach (var ta in Resources.LoadAll<TextAsset>(ResourcesFolder))
                names.Add(ta.name);
            return new List<string>(names);
        }

        private static Dictionary<string, string> CollectCurrentFiles(GazeBackbone backbone)
        {
            //Only this backbone's files (they are named "..._<backbone>.json"), so a profile is one model's
            //calibration and does not drag in another backbone's or the legacy no-suffix files.
            var suffix = $"_{backbone}{Extension}";
            var map = new Dictionary<string, string>();
            foreach (var sub in Subfolders)
            {
                var dir = Path.Combine(CalibrationRoot, sub);
                if (!Directory.Exists(dir)) continue;
                foreach (var f in Directory.GetFiles(dir, "*" + Extension))
                {
                    var file = Path.GetFileName(f);
                    if (file.EndsWith(suffix, StringComparison.OrdinalIgnoreCase))
                        map[$"{sub}/{file}"] = File.ReadAllText(f);
                }
            }
            return map;
        }

        private static bool WriteCalibrationFile(string relative, string content)
        {
            if (!IsSafeRelativePath(relative))
                return false;
            var parts = relative.Replace('\\', '/').Split('/');
            var dir = Path.Combine(CalibrationRoot, parts[0]);
            try
            {
                Directory.CreateDirectory(dir);
                File.WriteAllText(Path.Combine(dir, parts[1]), content);
                return true;
            }
            catch (Exception e)
            {
                UnitEyeLog.Exception(e);
                return false;
            }
        }

        private static string ReadProfileJson(string name)
        {
            var userPath = Path.Combine(UserProfilesDir, name + Extension);
            if (File.Exists(userPath))
                return File.ReadAllText(userPath);
            var ta = Resources.Load<TextAsset>($"{ResourcesFolder}/{name}");
            return ta != null ? ta.text : null;
        }

        //Characters that must never appear in a profile-entry file name. Explicit (not
        //Path.GetInvalidFileNameChars) so the rule is identical on every platform — on Windows a ':' would
        //otherwise silently write an NTFS alternate data stream ("Reg_X:evil.json").
        private static readonly char[] ReservedNameChars = { ':', '*', '?', '"', '<', '>', '|' };

        /// <summary>
        /// A profile entry key must be "&lt;knownSubfolder&gt;/&lt;file&gt;.json" with no path traversal and no
        /// reserved characters, so a crafted profile can never write outside the calibration folder.
        /// </summary>
        public static bool IsSafeRelativePath(string relative)
        {
            if (string.IsNullOrEmpty(relative)) return false;
            var parts = relative.Replace('\\', '/').Split('/');
            if (parts.Length != 2) return false;
            if (Array.IndexOf(Subfolders, parts[0]) < 0) return false;
            if (parts[1].Length == 0 || parts[1].Contains("..") || !parts[1].EndsWith(Extension, StringComparison.OrdinalIgnoreCase))
                return false;
            if (parts[1].IndexOfAny(ReservedNameChars) >= 0)
                return false;
            return true;
        }

        /// <summary>Strips characters not allowed in a file name so a profile name is always a safe file stem.</summary>
        public static string Sanitize(string name)
        {
            if (string.IsNullOrWhiteSpace(name)) return "";
            name = name.Trim();
            foreach (var c in Path.GetInvalidFileNameChars())
                name = name.Replace(c.ToString(), "");
            return name;
        }
    }
}
