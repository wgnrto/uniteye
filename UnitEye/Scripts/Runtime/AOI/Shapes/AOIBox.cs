using Unity.Mathematics;
using UnityEngine;

namespace UnitEye
{
    /// <summary>
    /// This class provides a box-shaped AOI
    /// </summary>
    public class AOIBox : AOI
    {
        public Vector2 startpoint;
        public Vector2 endpoint;

        public AOIBox(string uID, Vector2 startpoint, Vector2 endpoint, bool inverted = false, bool enabled = true, bool visualized = true) : base(uID, inverted, enabled, visualized)
        {
            this.startpoint = startpoint;
            this.endpoint = endpoint;
        }

        /// <summary>
        /// Check if point is in the AOIBox.
        /// </summary>
        /// <param name="point"></param>
        /// <returns>True if point is in the AOIBox, false if not. If this.inverted is true the return value is inverted.</returns>
        public override bool CheckAOI(Vector2 point)
        {
            //inclusion is true if point is inside the rectangle with startpoint and endpoint as opposite corners
            var inclusion = CheckBox(startpoint, endpoint, point);
            //Invert if AOI is inverted
            return this.inverted ? !inclusion : inclusion;
        }

        /// <summary>
        /// Visualize an AOIBox using GL.LINE_STRIP.
        /// </summary>
        /// <param name="color"></param>
        public override void Visualize(Color color)
        {
            GL.Begin(GL.LINE_STRIP);
            GL.Color(color);
            GL.Vertex(new Vector3(startpoint.x, 1 - startpoint.y, 0));
            GL.Vertex(new Vector3(endpoint.x, 1 - startpoint.y, 0));
            GL.Vertex(new Vector3(endpoint.x, 1 - endpoint.y, 0));
            GL.Vertex(new Vector3(startpoint.x, 1 - endpoint.y, 0));
            GL.Vertex(new Vector3(startpoint.x, 1 - startpoint.y, 0));
            GL.End();
        }
    }
}
