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
                //FaceLandmarks is null while no face is tracked (face loss now propagates); hold zero.
                var landmarks = _faceMesh.FaceLandmarks;
                if (landmarks == null || _landmark < 0 || _landmark >= landmarks.Count)
                    return Vector3.zero;
                var landmark = landmarks[_landmark];
                return new Vector3(landmark.X, landmark.Y, landmark.Z);
            }
        }

        private WebCamSource _webCamSource;
        private bool _spawnedPointGO;

        public Vector2 ScreenSize
        {
            get
            {
                if (_faceMesh == null)
                    return new Vector2(Screen.width, Screen.height);
                if (_webCamSource == null)
                    _webCamSource = _faceMesh.gameObject.GetComponent<WebCamSource>();
                if (_webCamSource == null)
                    return new Vector2(Screen.width, Screen.height);
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
            if (_faceMesh == null) return;
            if (_pointGO == null)
            {
                _pointGO = GameObject.CreatePrimitive(PrimitiveType.Sphere);
                _spawnedPointGO = true;
                var col = _pointGO.GetComponent<Collider>();
                if (col != null) Destroy(col);
            }

            //Cache Position/ScreenSize (MediaPipe Y is top-down; flip to Unity Y-up).
            var pos = Position;
            var size = ScreenSize;
            _pointGO.transform.position = new Vector3(pos.x * size.x, (1f - pos.y) * size.y, 0f);
        }

        void OnDestroy()
        {
            if (_spawnedPointGO && _pointGO != null)
                Destroy(_pointGO);
        }
    }
}
#endif
