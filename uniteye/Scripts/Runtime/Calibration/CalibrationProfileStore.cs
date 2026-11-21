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

        private static string CalibrationRoot => CalibrationModelStore.PersistentCalibrationRoot;
        //Writable location for user-saved profiles (a sibling of the RidgeRegression/MLP folders).
        private static string UserProfilesDir => Path.Combine(CalibrationRoot, "Profiles");
        private static string LegacyStreamingProfilesDir =>
            Path.Combine(CalibrationModelStore.StreamingCalibrationRoot, "Profiles");

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

            //Files NOT in the profile for this backbone must not survive the restore: a TPS warp, error model,
            //or opposite model family (MLP vs Ridge) from an earlier calibration would otherwise remain on disk.
            if (!string.IsNullOrEmpty(profile.backbone))
            {
                var managedFiles = new (string sub, string stem)[]
                {
                    ("RidgeRegression", "Reg_X"),
                    ("RidgeRegression", "Reg_Y"),
                    ("RidgeRegression", "Warp"),
                    ("RidgeRegression", "ErrorModel"),
                    ("MLP", "MLP"),
                };
                foreach (var (sub, stem) in managedFiles)
                {
                    var fileName = $"{stem}_{profile.backbone}{Extension}";
                    var rel = $"{sub}/{fileName}";
                    if (profile.files.ContainsKey(rel)) continue;
                    CalibrationModelStore.DeleteCalibrationFile(sub, fileName);
                }
            }

            var glassesWarning = currentlyWearingGlasses.HasValue && profile.wearingGlasses != currentlyWearingGlasses.Value
                ? (profile.wearingGlasses
                    ? " WARNING: profile was calibrated WITH glasses — accuracy degrades without them."
                    : " WARNING: profile was calibrated WITHOUT glasses — accuracy degrades with them.")
                : "";
            return $"Loaded profile '{name}' ({written} file(s)).{glassesWarning}";
        }

        /// <summary>
        /// Inspects a saved profile's metadata without writing files so callers can synchronize the active
        /// GazeBackbone and Calibrations mode when loading it.
        /// </summary>
        public static bool TryInspectProfile(string name, out GazeBackbone backbone, out bool hasRidge, out bool hasMlp)
        {
            backbone = GazeBackbone.EyeMU;
            hasRidge = false;
            hasMlp = false;
            name = Sanitize(name);
            var json = ReadProfileJson(name);
            if (json == null) return false;
            try
            {
                var profile = JsonConvert.DeserializeObject<CalibrationProfile>(json);
                if (profile == null) return false;
                bool parsedBackbone = !string.IsNullOrEmpty(profile.backbone) &&
                                      Enum.TryParse(profile.backbone, out backbone);
                if (profile.files != null)
                {
                    foreach (var key in profile.files.Keys)
                    {
                        if (key.StartsWith("RidgeRegression/Reg_X_", StringComparison.OrdinalIgnoreCase))
                            hasRidge = true;
                        else if (key.StartsWith("MLP/MLP_", StringComparison.OrdinalIgnoreCase))
                            hasMlp = true;
                    }
                }
                return parsedBackbone || hasRidge || hasMlp;
            }
            catch
            {
                return false;
            }
        }

        /// <summary>All available profile names: user-saved (persistent + StreamingAssets) + shipped (Resources), deduped and sorted.</summary>
        public static List<string> List()
        {
            var names = new SortedSet<string>(StringComparer.OrdinalIgnoreCase);
            if (Directory.Exists(UserProfilesDir))
                foreach (var f in Directory.GetFiles(UserProfilesDir, "*" + Extension))
                    names.Add(Path.GetFileNameWithoutExtension(f));
            try
            {
                if (Directory.Exists(LegacyStreamingProfilesDir))
                    foreach (var f in Directory.GetFiles(LegacyStreamingProfilesDir, "*" + Extension))
                        names.Add(Path.GetFileNameWithoutExtension(f));
            }
            catch { }
            foreach (var ta in Resources.LoadAll<TextAsset>(ResourcesFolder))
                names.Add(ta.name);
            return new List<string>(names);
        }

        private static Dictionary<string, string> CollectCurrentFiles(GazeBackbone backbone)
        {
            //Only this backbone's files (they are named "..._<backbone>.json"), so a profile is one model's
            //calibration and does not drag in another backbone's or the legacy no-suffix files.
            var suffix = $"_{backbone}{Extension}";
            var map = new Dictionary<string, string>(StringComparer.OrdinalIgnoreCase);
            foreach (var root in new[] { CalibrationModelStore.StreamingCalibrationRoot, CalibrationRoot })
            {
                try
                {
                    foreach (var sub in Subfolders)
                    {
                        var dir = Path.Combine(root, sub);
                        if (!Directory.Exists(dir)) continue;
                        foreach (var f in Directory.GetFiles(dir, "*" + Extension))
                        {
                            var file = Path.GetFileName(f);
                            if (file.EndsWith(suffix, StringComparison.OrdinalIgnoreCase))
                                map[$"{sub}/{file}"] = File.ReadAllText(f);
                        }
                    }
                }
                catch { }
            }
            return map;
        }

        private static bool WriteCalibrationFile(string relative, string content)
        {
            if (!IsSafeRelativePath(relative))
                return false;
            var parts = relative.Replace('\\', '/').Split('/');
            var dir = CalibrationModelStore.GetWritableDirectory(parts[0]);
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
            try
            {
                var legacyPath = Path.Combine(LegacyStreamingProfilesDir, name + Extension);
                if (File.Exists(legacyPath))
                    return File.ReadAllText(legacyPath);
            }
            catch { }
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
