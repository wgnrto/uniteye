// Excluded from WebGL player builds: depends on the native MediaPipe plugin (Mediapipe.Runtime
// has no wasm library, so IL2CPP linking fails). Kept for the Editor regardless of build target.
#if !UNITY_WEBGL || UNITY_EDITOR
using Mediapipe.Unity;
using Mediapipe.Unity.FaceMesh;
using UnityEngine;
namespace UnitEye
{

    public class LandmarkVisualizer : MonoBehaviour
    {
        [SerializeField] private FaceMeshSolution _faceMesh;
        [SerializeField] private int _landmark;
        [SerializeField] private GameObject _pointGO;

        public Vector3 Position
        {
            get
            {
                var landmark = _faceMesh.FaceLandmarks[_landmark];
                return new Vector3(landmark.X, landmark.Y, landmark.Z);
            }
        }

        private WebCamSource _webCamSource;

        public Vector2 ScreenSize
        {
            get
            {
                if (_webCamSource == null)
                    _webCamSource = _faceMesh.gameObject.GetComponent<WebCamSource>();
                var resolution = _webCamSource.resolution;
                return new Vector2(resolution.width, resolution.height);
            }
        }

        void Start()
        {
            if (_faceMesh == null)
                enabled = false;
        }

        void Update()
        {
            if (_pointGO == null)
                _pointGO = GameObject.CreatePrimitive(PrimitiveType.Sphere);

            //Cache Position/ScreenSize (each read hits the face mesh / a GetComponent); no per-frame log.
            var pos = Position;
            var size = ScreenSize;
            _pointGO.transform.position = new Vector3(pos.x * size.x, pos.y * size.y, 0);
        }
    }
}
#endif
