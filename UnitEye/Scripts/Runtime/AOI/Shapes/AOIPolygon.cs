using System.Collections.Generic;
using Unity.Mathematics;
using UnityEngine;

namespace UnitEye
{
    /// <summary>
    /// This class allows for arbitarly shaped polygon AOIs
    /// </summary>
    public class AOIPolygon : AOI
    {
        public List<Vector2> points;

        public AOIPolygon(string uID, bool inverted = false, bool enabled = true, bool visualized = true) : base(uID, inverted, enabled, visualized)
        {
        }

        public AOIPolygon(string uID, List<Vector2> points, bool inverted = false, bool enabled = true, bool visualized = true) : base(uID, inverted, enabled, visualized)
        {
            this.points = points;
        }

        /// <summary>
        /// Add point to AOIPolygon.
        /// </summary>
        /// <param name="point"></param>
        public void AddPoint(Vector2 point)
        {
            points.Add(point);
        }

        /// <summary>
        /// Insert point to AOIPolygon at index i.
        /// </summary>
        /// <param name="point"></param>
        public void InsertPoint(Vector2 point, int i)
        {
            points.Insert(i, point);
        }

        /// <summary>
        /// Remove first matching point from AOIPolygon.
        /// </summary>
        /// <param name="point"></param>
        public void RemovePoint(Vector2 point)
        {
            points.Remove(point);
        }

        /// <summary>
        /// Remove all matching points from AOIPolygon.
        /// </summary>
        /// <param name="point"></param>
        public void RemoveAllPoints(Vector2 point)
        {
            foreach (Vector2 pointList in points)
            {
                if (pointList.x == point.x && pointList.y == point.y) points.Remove(point);
            }
        }

        /// <summary>
        /// Check if point is in the AOIPolygon.
        /// </summary>
        /// <param name="point"></param>
        /// <returns>True if point is in the AOIPolygon, false if not. If this.inverted is true the return value is inverted.</returns>
        public override bool CheckAOI(Vector2 point)
        {
            //inclusion is true if point is inside the polygon defined by points
            var inclusion = CheckPolygon(points, point);
            //Invert if AOI is inverted
            return this.inverted ? !inclusion : inclusion;
        }

        /// <summary>
        /// Visualize an AOIPolygon using GL.LINE_STRIP.
        /// </summary>
        /// <param name="color"></param>
        public override void Visualize(Color color)
        {
            GL.Begin(GL.LINE_STRIP);
            GL.Color(color);
            //Start from first point
            GL.Vertex(new Vector3(points[0].x, 1 - points[0].y, 0));
            //Draw all points until the last point
            for (int i = 1; i < points.Count; i++)
            {
                GL.Vertex(new Vector3(points[i].x, 1 - points[i].y, 0));
            }
            //Close the shape to first point
            GL.Vertex(new Vector3(points[0].x, 1 - points[0].y, 0));
            GL.End();
        }
    }
}