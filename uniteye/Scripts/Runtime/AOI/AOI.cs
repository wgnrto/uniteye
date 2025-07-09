using System.Collections.Generic;
using Unity.Mathematics;
using UnityEngine;

namespace UnitEye
{
    /// <summary>
    /// This is the base class for all AOI shapes. New shapes should inherit from this class.
    /// </summary>
    public abstract class AOI
    {
        public readonly string uID;
        public bool inverted;
        public bool enabled = true;
        public bool visualized = true;

        public bool focused = false;

        protected LineRenderer _lineRenderer;

        public AOI(string uID, bool inverted = false, bool enabled = true, bool visualized = true)
        {
            this.uID = uID;
            this.inverted = inverted;
            this.enabled = enabled;
            this.visualized = visualized;
        }

        public abstract bool CheckAOI(Vector2 point);
        public abstract void Visualize(Color color);

        /// <summary>
        /// Check point inclusion for a Box between startpoint and endpoint.
        /// </summary>
        /// <param name="startpoint"></param>
        /// <param name="endpoint"></param>
        /// <param name="point"></param>
        /// <returns>True if point is in the Box, false if not.</returns>
        public bool CheckBox(Vector2 startpoint, Vector2 endpoint, Vector2 point)
        {
            //If startpoint.x <= point.x <= endpoint.x (or flipped if start and end are flipped)
            //and startpoint.y <= point.y <= endpoint.y (or flipped) the point is in the Box
            return (startpoint.x <= point.x && point.x <= endpoint.x || startpoint.x >= point.x && point.x >= endpoint.x)
                && (startpoint.y <= point.y && point.y <= endpoint.y || startpoint.y >= point.y && point.y >= endpoint.y);
        }

        /// <summary>
        /// Check point inclusion for Circle by checking if Euclidean distance between center and point is <= radius.
        /// </summary>
        /// <param name="center"></param>
        /// <param name="radius"></param>
        /// <param name="point"></param>
        /// <returns>True if point is in the Circle, false if not.</returns>
        public bool CheckCircle(Vector2 center, float radius, Vector2 point)
        {
            var distancePoint = Mathf.Sqrt((point.x - center.x) * (point.x - center.x) + (point.y - center.y) * (point.y - center.y));
            return distancePoint <= radius;
        }

        /// <summary>
        /// Calculate orthopoint on line between startpoint and endpoint where vector between
        /// orthopoint and point is orthogonal to vector between startpoint and endpoint.
        /// CheckCircle() with new orthopoint for point inclusion.
        /// </summary>
        /// <param name="startpoint"></param>
        /// <param name="endpoint"></param>
        /// <param name="radius"></param>
        /// <param name="point"></param>
        /// <returns>True if point is in the Capsule, false if not.</returns>
        public bool CheckCapsule(Vector2 startpoint, Vector2 endpoint, float radius, Vector2 point)
        {
            Vector2 vector = endpoint - startpoint;
            Vector2 pointstart = point - startpoint;

            //Calculate distance from startpoint to orthopoint on line created by startpoint and endpoint
            float dot = (pointstart.x * vector.x + pointstart.y * vector.y) / (vector.x * vector.x + vector.y * vector.y);

            //Calculate orthopoint on line between startpoint and endpoint
            Vector2 orthopoint = startpoint + Mathf.Min(Mathf.Max(dot, 0), 1) * vector;

            //Debug.Log($"startpoint: {startpoint} | endpoint: {endpoint} | radius: {radius} | point: {point} | vector: {vector} | pointstart: {pointstart} | dot: {dot} | orthopoint: {orthopoint}");

            //Return if point is in the circle around orthopoint with radius
            return CheckCircle(orthopoint, radius, point);
        }

        /// <summary>
        /// Return false when vector from startpoint to point is more than +-90 degrees flipped to vector from startpoint to endpoint
        /// or vector from endpoint to point is less than +-90 degrees flipped to vector from startpoint to endpoint.
        /// Calculate orthopoint on line between startpoint and endpoint where vector between
        /// orthopoint and point is orthogonal to vector between startpoint and endpoint.
        /// CheckCircle with new orthopoint for point inclusion.
        /// </summary>
        /// <param name="startpoint"></param>
        /// <param name="endpoint"></param>
        /// <param name="radius"></param>
        /// <param name="point"></param>
        /// <returns>True if point is in the CapsuleBox, false if not.</returns>
        public bool CheckCapsuleBox(Vector2 startpoint, Vector2 endpoint, float radius, Vector2 point)
        {
            Vector2 vector = endpoint - startpoint;
            Vector2 pointstart = point - startpoint;
            //If pointstart is more than +-90 degrees turned from vector the dot product is negative
            //and point is outside of the CapsuleBox
            if ((pointstart.x * vector.x + pointstart.y * vector.y) < 0) return false;

            Vector2 pointend = point - endpoint;
            //If pointend is less than +-90 degrees turned from vector the dot product is positive
            //and point is outside of the CapsuleBox
            if ((pointend.x * vector.x + pointend.y * vector.y) > 0) return false;

            //Calculate distance from startpoint to orthopoint on line created by startpoint and endpoint
            float dot = (pointstart.x * vector.x + pointstart.y * vector.y) / (vector.x * vector.x + vector.y * vector.y);
            //Calculate orthopoint on line between startpoint and endpoint
            Vector2 orthopoint = startpoint + Mathf.Min(Mathf.Max(dot, 0), 1) * vector;

            //Debug.Log($"startpoint: {startpoint} | endpoint: {endpoint} | radius: {radius} | point: {point} | vector: {vector} | pointstart: {pointstart} | dot: {dot} | orthopoint: {orthopoint}");
            
            //Return if point is in the circle around orthopoint with radius
            return CheckCircle(orthopoint, radius, point);
        }

        /// <summary>
        /// Check if point is in polygon.
        /// Source: https://stackoverflow.com/questions/4243042/c-sharp-point-in-polygon
        /// </summary>
        /// <param name="points">Polygon points</param>
        /// <param name="point">Point to check</param>
        /// <returns>True if point is within Polygon</returns>
        public bool CheckPolygon(List<Vector2> points, Vector2 point)
        {
            bool result = false;
            int j = points.Count - 1;
            Vector2 pointI, pointJ;
            for (int i = 0; i < points.Count; i++)
            {
                pointI = points[i];
                pointJ = points[j];
                if (pointI.y < point.y && pointJ.y >= point.y || pointJ.y < point.y && pointI.y >= point.y)
                {
                    if (pointI.x + (point.y - pointI.y) / (pointJ.y - pointI.y) * (pointJ.x - pointI.x) < point.x)
                    {
                        result = !result;
                    }
                }
                j = i;
            }
            return result;
        }

        /// <summary>
        /// Rotate a Vector2 vector counterclockwise by degrees.
        /// Source: https://answers.unity.com/questions/661383/whats-the-most-efficient-way-to-rotate-a-vector2-o.html
        /// </summary>
        /// <param name="vector"></param>
        /// <param name="degrees"></param>
        /// <returns>Counterclockwise rotated vector.</returns>
        public Vector2 RotateFloat2CCW(Vector2 vector, float degrees)
        {
            float sin = Mathf.Sin(degrees * Mathf.Deg2Rad);
            float cos = Mathf.Cos(degrees * Mathf.Deg2Rad);

            float vX = vector.x;
            float vY = vector.y;
            vector.x = (vX * cos) - (vY * sin);
            vector.y = (vX * sin) + (vY * cos);

            return vector;
        }

        /// <summary>
        /// Add GL.Vertex() to form a circle with radius around center point.
        /// </summary>
        /// <param name="center"></param>
        /// <param name="radius"></param>
        public static void DrawGLCircle(Vector2 center, float radius)
        {
            //Calculate steps based on radius
            int steps = (int)(radius * 300f);
            
            //For each step add a vertex on a circle with radius around center
            for (int currentStep = 0; currentStep < steps; currentStep++)
            {
                float floatProgress = (float)currentStep / steps;
            
                //2 * pi for a complete circle
                float currentRadian = floatProgress * 2 * Mathf.PI;

                GL.Vertex(new Vector3(center.x + Mathf.Cos(currentRadian) * radius, center.y + Mathf.Sin(currentRadian) * radius, 0));
            }

            //Add last vertex to connect to the first point and close the circle
            GL.Vertex(new Vector3(center.x + radius, center.y, 0));
        }

        /// <summary>
        /// Add GL.Vertex() to form a semicircle around the center between startpoint and endpoint
        /// with the diameter equal to their distance.
        /// </summary>
        /// <param name="startpoint"></param>
        /// <param name="endpoint"></param>
        public static void DrawGLSemicircle(Vector2 startpoint, Vector2 endpoint)
        {
            //Voodoo magic to draw a counter clockwise semicircle
            float x = startpoint.x - endpoint.x;
            float y = startpoint.y - endpoint.y;
            
            //Radius equals half the distance between the points
            float radius = Mathf.Sqrt(x * x + y * y) / 2;

            //Calculate steps based on radius
            int steps = (int)(radius * 150f);

            //Center is startpoint + half the vector between startpoint and endpoint
            Vector2 center = startpoint + 0.5f * (endpoint - startpoint);

            //Calculate starting radian for semicircle since we can't start to the right of center if the startpoint is somewhere else
            float baseRadian;
            baseRadian = Mathf.Atan2(y ,x);

            //For each step add a vertex on a semicircle with radius around center
            for (int currentStep = 0; currentStep < steps; currentStep++)
            {
                float floatProgress = (float)currentStep / steps;

                //1 * pi for a semicircle
                float currentRadian = floatProgress * Mathf.PI + baseRadian;

                GL.Vertex(new Vector3(center.x + Mathf.Cos(currentRadian) * radius, center.y + Mathf.Sin(currentRadian) * radius, 0));
            }

            //Add last vertex to connect to the endpoint and close the semicircle
            GL.Vertex(new Vector3(endpoint.x, endpoint.y, 0));
        }
    }
}


