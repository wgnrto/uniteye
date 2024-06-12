using Unity.Mathematics;
using UnityEngine;

namespace UnitEye
{
    /// <summary>
    /// This class provides a circle-shaped AOI
    /// </summary>
    public class AOICircle : AOI
    {
        public Vector2 center;
        public float radius;

        public AOICircle(string uID, Vector2 center, float radius, bool inverted = false, bool enabled = true, bool visualized = true) : base(uID, inverted, enabled, visualized)
        {
            this.center = center;
            this.radius = radius;
        }

        /// <summary>
        /// Check if point is in the AOICircle.
        /// </summary>
        /// <param name="point"></param>
        /// <returns>True if point is in the AOICircle, false if not. If this.inverted is true the return value is inverted.</returns>
        public override bool CheckAOI(Vector2 point)
        {
            //inclusion is true if point is inside Circle around center with radius
            var inclusion = CheckCircle(center, radius, point);
            //Invert if AOI is inverted
            return this.inverted ? !inclusion : inclusion;
        }

        /// <summary>
        /// Visualize an AOICircle using GL.LINE_STRIP.
        /// </summary>
        /// <param name="color"></param>
        public override void Visualize(Color color)
        {
            GL.Begin(GL.LINE_STRIP);
            GL.Color(color);
            DrawGLCircle(new Vector2(center.x, 1 - center.y), radius);
            GL.End();
        }
    }
}