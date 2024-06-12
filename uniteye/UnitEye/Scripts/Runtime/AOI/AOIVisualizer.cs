using UnityEngine;

namespace UnitEye
{
    /// <summary>
    /// This class is used to visualize the AOIs present in the AOIManager.
    /// </summary>
    public class AOIVisualizer : MonoBehaviour
    {
        public Material mat;
        public AOIManager aoiManager;

        /// <summary>
        /// Assign material for the GL drawer
        /// </summary>
        void Awake()
        {
            mat = Resources.Load<Material>("Materials/AOIVisualizerMaterial");
            mat.color = Color.white;
        }

        /// <summary>
        /// Visualize all AOIs contained in the corresponding _aoiManager
        /// </summary>
        void OnPostRender()
        {
            if (!mat)
            {
                //Should not happen unless Unity breaks
                Debug.LogError("Material broke, pls fix");
                return;
            }
            GL.PushMatrix();
            mat.SetPass(0);
            GL.LoadOrtho();

            aoiManager.VisualizeAOIList();

            GL.PopMatrix();
        }

    }
}