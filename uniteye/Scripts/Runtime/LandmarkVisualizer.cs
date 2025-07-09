using Mediapipe.Unity;
using Mediapipe.Unity.FaceMesh;
using UnityEngine;

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

    public Vector2 ScreenSize
    {
        get
        {
            var resolution = _faceMesh.gameObject.GetComponent<WebCamSource>().resolution;
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

        _pointGO.transform.position = new Vector3(Position.x * ScreenSize.x, Position.y * ScreenSize.y, 0);
        Debug.Log(_pointGO.transform.position);
    }
}
