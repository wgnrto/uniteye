using System.Collections.Generic;
using UnityEngine;

namespace UnitEye
{
    /// <summary>
    /// Minimal IMGUI shape helpers shared by the calibration and evaluation overlays.
    /// Both overlays place everything in GUI coordinates, so drawing their guide lines here keeps them
    /// pixel-exact against the dots they annotate. (The calibration path used to be a world-space
    /// LineRenderer parented under the annotation Canvas and positioned with raw pixel values; as soon as
    /// the CanvasScaler's scale factor was not exactly 1 — i.e. on any resolution other than its 2436x1125
    /// reference — the line drifted off-target and onto the screen edge.)
    /// </summary>
    public static class GUIShapes
    {
        //Reused 1x1 white texture. Not tied to any scene/asset: survives scene loads and never shows up
        //as a leaked asset.
        private static Texture2D _pixel;

        /// <summary>A 1x1 opaque white texture, tint it via GUI.color.</summary>
        public static Texture2D Pixel
        {
            get
            {
                if (_pixel == null)
                {
                    _pixel = new Texture2D(1, 1) { hideFlags = HideFlags.HideAndDontSave };
                    _pixel.SetPixel(0, 0, Color.white);
                    _pixel.Apply();
                }
                return _pixel;
            }
        }

        /// <summary>Fills a GUI-space rectangle with a solid colour. Restores GUI.color.</summary>
        public static void FillRect(Rect rect, Color color)
        {
            var previous = GUI.color;
            GUI.color = color;
            GUI.DrawTexture(rect, Pixel);
            GUI.color = previous;
        }

        /// <summary>
        /// Draws a straight line between two GUI-space points. Restores GUI.color and GUI.matrix.
        /// </summary>
        public static void DrawLine(Vector2 a, Vector2 b, Color color, float width)
        {
            var delta = b - a;
            float length = delta.magnitude;
            //Written as "not long enough" rather than "shorter than": that also rejects a NaN length, which
            //would otherwise poison GUI.matrix through RotateAroundPivot for the rest of the pass.
            if (!(length >= 1f)) return;
            float angle = Mathf.Atan2(delta.y, delta.x) * Mathf.Rad2Deg;

            var matrix = GUI.matrix;
            var previous = GUI.color;
            GUI.color = color;
            GUIUtility.RotateAroundPivot(angle, a);
            GUI.DrawTexture(new Rect(a.x, a.y - width * 0.5f, length, width), Pixel);
            GUI.matrix = matrix;
            GUI.color = previous;
        }

        /// <summary>Draws a polyline through consecutive GUI-space points.</summary>
        public static void DrawPolyline(IReadOnlyList<Vector2> points, Color color, float width)
        {
            if (points == null) return;
            for (int i = 1; i < points.Count; i++)
                DrawLine(points[i - 1], points[i], color, width);
        }
    }
}
