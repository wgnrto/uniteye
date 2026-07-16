using MathNet.Numerics.LinearAlgebra;
using Newtonsoft.Json;
using System;
using System.IO;
using UnityEngine;

namespace UnitEye
{
    /// <summary>
    /// A regularized 2D thin-plate-spline warp used as a LOCAL residual correction on top of the ridge
    /// calibration: fitted on (mean ridge prediction at a dwell anchor → true anchor target) pairs, it bends
    /// the prediction space so region-specific errors (e.g. one bad corner) are corrected without disturbing
    /// the rest — the global polynomial features cannot represent such local distortions. All coordinates
    /// are NORMALIZED screen positions (0..1), which also conditions the kernel.
    ///
    /// Overfit control: the fit is regularized (lambda smooths toward the pure affine map) and the CALLER
    /// must validate on held-out samples and discard the warp unless it actually improves them
    /// (HomulerGazeCalibration.ProcessData does exactly that) — with only ~9-13 anchors an unvalidated
    /// spline can easily bend the space between anchors in wrong ways.
    /// </summary>
    public class ThinPlateSplineWarp
    {
        //Control points (the anchor SOURCE positions, i.e. ridge predictions) and the fitted parameters of
        //the two coordinate maps. Public for JSON serialization.
        public float[] SourceX { get; set; }
        public float[] SourceY { get; set; }
        public float[] WeightsX { get; set; }
        public float[] WeightsY { get; set; }
        //Affine part per map: [constant, x coefficient, y coefficient].
        public float[] AffineX { get; set; }
        public float[] AffineY { get; set; }
        public float Lambda { get; set; }

        /// <summary>Minimum anchors for a meaningful fit (TPS needs 3 non-collinear; require a margin).</summary>
        public const int MinimumAnchors = 5;

        /// <summary>
        /// Fits the warp mapping <paramref name="source"/> points to <paramref name="destination"/> points.
        /// Returns null when there are too few anchors or the system fails to solve.
        /// </summary>
        public static ThinPlateSplineWarp Fit(Vector2[] source, Vector2[] destination, float lambda = 0.01f)
        {
            if (source == null || destination == null || source.Length != destination.Length ||
                source.Length < MinimumAnchors)
                return null;

            int n = source.Length;
            var l = Matrix<float>.Build.Dense(n + 3, n + 3);
            for (int i = 0; i < n; i++)
            {
                for (int j = 0; j < n; j++)
                {
                    float dx = source[i].x - source[j].x;
                    float dy = source[i].y - source[j].y;
                    l[i, j] = Kernel(dx * dx + dy * dy) + (i == j ? lambda : 0f);
                }
                l[i, n] = l[n, i] = 1f;
                l[i, n + 1] = l[n + 1, i] = source[i].x;
                l[i, n + 2] = l[n + 2, i] = source[i].y;
            }

            var rhsX = Vector<float>.Build.Dense(n + 3);
            var rhsY = Vector<float>.Build.Dense(n + 3);
            for (int i = 0; i < n; i++)
            {
                rhsX[i] = destination[i].x;
                rhsY[i] = destination[i].y;
            }

            Vector<float> solutionX, solutionY;
            try
            {
                var lu = l.LU();
                solutionX = lu.Solve(rhsX);
                solutionY = lu.Solve(rhsY);
            }
            catch (Exception)
            {
                return null; //singular system (e.g. collinear anchors) -> no warp rather than garbage
            }

            var warp = new ThinPlateSplineWarp
            {
                SourceX = new float[n],
                SourceY = new float[n],
                WeightsX = new float[n],
                WeightsY = new float[n],
                AffineX = new[] { solutionX[n], solutionX[n + 1], solutionX[n + 2] },
                AffineY = new[] { solutionY[n], solutionY[n + 1], solutionY[n + 2] },
                Lambda = lambda,
            };
            for (int i = 0; i < n; i++)
            {
                warp.SourceX[i] = source[i].x;
                warp.SourceY[i] = source[i].y;
                warp.WeightsX[i] = solutionX[i];
                warp.WeightsY[i] = solutionY[i];
                if (float.IsNaN(solutionX[i]) || float.IsNaN(solutionY[i]))
                    return null;
            }
            return warp;
        }

        /// <summary>
        /// Applies the warp to a normalized point. Falls back to the input (identity) if the warp is
        /// malformed or produces a non-finite value — never worse than no warp.
        /// </summary>
        public Vector2 Apply(Vector2 point)
        {
            if (SourceX == null || WeightsX == null || AffineX == null || AffineX.Length < 3 ||
                SourceX.Length != WeightsX.Length || SourceY == null || WeightsY == null ||
                AffineY == null || AffineY.Length < 3)
                return point;

            float x = AffineX[0] + AffineX[1] * point.x + AffineX[2] * point.y;
            float y = AffineY[0] + AffineY[1] * point.x + AffineY[2] * point.y;
            for (int i = 0; i < SourceX.Length; i++)
            {
                float dx = point.x - SourceX[i];
                float dy = point.y - SourceY[i];
                float u = Kernel(dx * dx + dy * dy);
                x += WeightsX[i] * u;
                y += WeightsY[i] * u;
            }

            if (float.IsNaN(x) || float.IsInfinity(x) || float.IsNaN(y) || float.IsInfinity(y))
                return point;
            return new Vector2(x, y);
        }

        //TPS radial kernel U(r) = r^2 ln r, written on the squared radius: 0.5 * r^2 * ln(r^2).
        private static float Kernel(float squaredRadius)
            => squaredRadius <= 1e-12f ? 0f : 0.5f * squaredRadius * Mathf.Log(squaredRadius);

        //Stored next to the ridge files (same folder + per-backbone suffix), so calibration profiles bundle
        //the warp together with the ridge model it belongs to.
        private static string PathFor(string filename)
            => Application.streamingAssetsPath + $"/Calibration Files/RidgeRegression/{filename}";

        public void Save(string filename)
        {
            var directory = Application.streamingAssetsPath + "/Calibration Files/RidgeRegression/";
            if (!Directory.Exists(directory))
                Directory.CreateDirectory(directory);
            File.WriteAllText(PathFor(filename), JsonConvert.SerializeObject(this));
        }

        /// <summary>Loads a warp, or null when none was saved (the expected no-warp case).</summary>
        public static ThinPlateSplineWarp Load(string filename)
        {
            var path = PathFor(filename);
            if (!File.Exists(path))
                return null;
            return JsonConvert.DeserializeObject<ThinPlateSplineWarp>(File.ReadAllText(path));
        }

        /// <summary>Removes a saved warp (called when a retrain decides against keeping one, so a stale
        /// warp can never pair with a newer ridge fit).</summary>
        public static void Delete(string filename)
        {
            var path = PathFor(filename);
            if (File.Exists(path))
                File.Delete(path);
        }
    }
}
