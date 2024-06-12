using Unity.Mathematics;
using UnityEngine;

namespace UnitEye
{
    /// <summary>
    /// This class provides a capsulebox-shaped AOI
    /// </summary>
    public class AOICapsuleBox : AOI
    {
        public Vector2 startpoint;
        public Vector2 endpoint;
        public float radius;

        public AOICapsuleBox(string uID, Vector2 startpoint, Vector2 endpoint, float radius, bool inverted = false, bool enabled = true, bool visualized = true) : base(uID, inverted, enabled, visualized)
        {
            this.startpoint = startpoint;
            this.endpoint = endpoint;
            this.radius = radius;
        }

        /// <summary>
        /// Check if point is in the AOICapsuleBox.
        /// </summary>
        /// <param name="point"></param>
        /// <returns>True if point is in the AOICapsuleBox, false if not. If this.inverted is true the return value is inverted.</returns>
        public override bool CheckAOI(Vector2 point)
        {
            //inclusion is true if point is inside CapsuleBox in the middle
            var inclusion = CheckCapsuleBox(startpoint, endpoint, radius, point);
            //Invert if AOI is inverted
            return this.inverted ? !inclusion : inclusion;
        }

        /// <summary>
        /// Visualize an AOICapsuleBox using GL.LINE_STRIP.
        /// </summary>
        /// <param name="color"></param>
        public override void Visualize(Color color)
        {
            float X = endpoint.x - startpoint.x;
            float Y = endpoint.y - startpoint.y;
            float distance = Mathf.Sqrt(X * X + Y * Y);

            //Calculate corner points
            Vector2 directionVector = (endpoint - startpoint) * (radius/distance);

            directionVector = RotateFloat2CCW(directionVector, 90);
            Vector2 start2 = startpoint + directionVector;
            Vector2 end2 = endpoint + directionVector;

            directionVector = RotateFloat2CCW(directionVector, 180);
            Vector2 start1 = startpoint + directionVector;
            Vector2 end1 = endpoint + directionVector;

            //Connect cornerpoints
            GL.Begin(GL.LINE_STRIP);
            GL.Color(color);
            GL.Vertex(new Vector3(start1.x, 1 - start1.y, 0));
            GL.Vertex(new Vector3(start2.x, 1 - start2.y, 0));
            GL.Vertex(new Vector3(end2.x, 1 - end2.y, 0));
            GL.Vertex(new Vector3(end1.x, 1 - end1.y, 0));
            GL.Vertex(new Vector3(start1.x, 1 - start1.y, 0));
            GL.End();
        }
    }
}
