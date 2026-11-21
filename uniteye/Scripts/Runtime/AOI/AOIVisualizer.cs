using UnityEngine;
using UnityEngine.Rendering;

namespace UnitEye
{
    /// <summary>
    /// This class is used to visualize the AOIs present in the AOIManager across both Built-in RP and SRP (URP/HDRP).
    /// </summary>
    public class AOIVisualizer : MonoBehaviour
    {
        public Material mat;
        public AOIManager aoiManager;
        private Camera _camera;
        private bool _ownsMaterial;

        void Awake()
        {
            _camera = GetComponent<Camera>();
            var sharedMat = Resources.Load<Material>("Materials/AOIVisualizerMaterial");
            if (sharedMat != null)
            {
                mat = new Material(sharedMat) { color = Color.white, hideFlags = HideFlags.HideAndDontSave };
                _ownsMaterial = true;
            }
        }

        void OnEnable()
        {
            RenderPipelineManager.endCameraRendering += OnEndCameraRendering;
        }

        void OnDisable()
        {
            RenderPipelineManager.endCameraRendering -= OnEndCameraRendering;
        }

        void OnDestroy()
        {
            if (_ownsMaterial && mat != null)
                Destroy(mat);
        }

        private void OnEndCameraRendering(ScriptableRenderContext context, Camera cam)
        {
            if (_camera != null && cam == _camera)
                DrawAOIs();
        }

        void OnPostRender()
        {
            DrawAOIs();
        }

        private void DrawAOIs()
        {
            if (aoiManager == null) return;
            if (!mat)
            {
                Debug.LogError("AOIVisualizer: Material missing.");
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