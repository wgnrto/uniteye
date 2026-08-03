using System.Collections.Generic;
using System.IO;
using UnityEngine;

namespace UnitEye
{
    /// <summary>
    /// Per-user, per-screen-region gaze ERROR model: bias vector + covariance ellipse at each evaluation
    /// target, fitted from the evaluation run and persisted alongside the calibration. Accuracy and
    /// precision vary MORE THAN SIX-FOLD between users and between screen positions (Feit et al. CHI'17),
    /// so a single global sigma misstates most fixations; this model is what turns the probabilistic AOI
    /// layer's P(AOI | fixation) from a guess into a calibrated number, and it powers per-session quality
    /// reporting ("your tracking error is ~X cm; AOIs smaller than Y are coin flips").
    ///
    /// Query = inverse-distance-weighted interpolation over the anchor grid (smooth, no fitting step,
    /// exact at the anchors). Coordinates are NORMALIZED (0..1) so the model is resolution-independent.
    /// </summary>
    public class GazeErrorModel
    {
        [System.Serializable]
        public class Anchor
        {
            public float x, y;            // target position (normalized)
            public float biasX, biasY;    // mean gaze - target (normalized)
            public float covXX, covXY, covYY; // sample covariance about the mean (normalized^2)
        }

        [System.Serializable]
        private class Persisted
        {
            public string sourceCalibration = "";
            public List<Anchor> anchors = new List<Anchor>();
        }

        private readonly List<Anchor> _anchors = new List<Anchor>();
        public IReadOnlyList<Anchor> Anchors => _anchors;
        public bool IsEmpty => _anchors.Count == 0;

        /// <summary>The calibration type whose predictions this model was measured on. The bias field is
        /// only valid for THAT model's output — applying ridge-measured biases while the MLP is active
        /// corrects errors the active model does not have.</summary>
        public Calibrations SourceCalibration { get; set; } = Calibrations.RidgeRegression;
        /// <summary>Whether the bias field may be applied to the given active calibration type.</summary>
        public bool AppliesTo(Calibrations active) => active == SourceCalibration;

        public void Clear() => _anchors.Clear();

        public void AddAnchor(Vector2 target, Vector2 bias, float covXX, float covXY, float covYY)
        {
            _anchors.Add(new Anchor
            {
                x = target.x, y = target.y,
                biasX = bias.x, biasY = bias.y,
                covXX = covXX, covXY = covXY, covYY = covYY,
            });
        }

        /// <summary>
        /// Interpolated error at a screen position (normalized): expected bias and covariance.
        /// Inverse-distance-squared weighting over all anchors; exact at an anchor.
        /// </summary>
        public void Query(Vector2 position, out Vector2 bias, out float covXX, out float covXY, out float covYY)
        {
            bias = Vector2.zero;
            //Conservative default when no model exists: ~2cm-class sigma on a laptop screen
            //(0.04 normalized SD per axis), uncorrelated.
            covXX = covYY = 0.04f * 0.04f;
            covXY = 0f;
            if (_anchors.Count == 0) return;

            float wSum = 0f;
            float bx = 0f, by = 0f, cxx = 0f, cxy = 0f, cyy = 0f;
            for (int i = 0; i < _anchors.Count; i++)
            {
                var a = _anchors[i];
                float d2 = (position - new Vector2(a.x, a.y)).sqrMagnitude;
                float w = 1f / (d2 + 1e-4f);   // epsilon keeps the weight finite at the anchor itself
                wSum += w;
                bx += w * a.biasX;
                by += w * a.biasY;
                cxx += w * a.covXX;
                cxy += w * a.covXY;
                cyy += w * a.covYY;
            }
            float inv = 1f / wSum;
            bias = new Vector2(bx * inv, by * inv);
            covXX = Mathf.Max(1e-8f, cxx * inv);
            covXY = cxy * inv;
            covYY = Mathf.Max(1e-8f, cyy * inv);
        }

        /// <summary>Mean per-anchor error magnitude (normalized) — the session-quality headline number.</summary>
        public float MeanErrorNormalized()
        {
            if (_anchors.Count == 0) return -1f;
            float sum = 0f;
            foreach (var a in _anchors)
                sum += new Vector2(a.biasX, a.biasY).magnitude;
            return sum / _anchors.Count;
        }

        // ---- Persistence: rides the same per-backbone "Calibration Files" convention as the calibration
        // models (and therefore ships in builds + is picked up by calibration profiles' filename filter).
        private static string PathFor(GazeBackbone backbone) =>
            Application.streamingAssetsPath + "/Calibration Files/RidgeRegression/" +
            CalibrationModelStore.FileName("ErrorModel.json", backbone);

        public void Save(GazeBackbone backbone)
        {
            var dir = Application.streamingAssetsPath + "/Calibration Files/RidgeRegression/";
            if (!Directory.Exists(dir))
                Directory.CreateDirectory(dir);
            var persisted = new Persisted { sourceCalibration = SourceCalibration.ToString(), anchors = _anchors };
            File.WriteAllText(PathFor(backbone), JsonUtility.ToJson(persisted));
        }

        /// <summary>Deletes the persisted error model for a backbone. Called on RE-CALIBRATION: the model
        /// measured the OLD calibration's residuals, and applying them to a fresh fit corrupts the AOI
        /// stream until the next evaluation replaces it.</summary>
        public static void Delete(GazeBackbone backbone)
        {
            try
            {
                var path = PathFor(backbone);
                if (File.Exists(path)) File.Delete(path);
            }
            catch (System.Exception e) { UnitEyeLog.Exception(e); }
        }

        public static GazeErrorModel Load(GazeBackbone backbone)
        {
            try
            {
                var path = PathFor(backbone);
                if (!File.Exists(path)) return null;
                var persisted = JsonUtility.FromJson<Persisted>(File.ReadAllText(path));
                if (persisted?.anchors == null || persisted.anchors.Count == 0) return null;
                var model = new GazeErrorModel();
                model._anchors.AddRange(persisted.anchors);
                if (System.Enum.TryParse(persisted.sourceCalibration, out Calibrations source))
                    model.SourceCalibration = source;
                return model;
            }
            catch (System.Exception e)
            {
                UnitEyeLog.Exception(e);
                return null;
            }
        }
    }
}
