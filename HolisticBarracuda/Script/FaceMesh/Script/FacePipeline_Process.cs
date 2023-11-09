using System.Linq;
using UnityEngine;
using Unity.Mathematics;
using System.Collections.Generic;
using MediaPipe.BlazeFace;

namespace MediaPipe.FaceMesh {

//
// Image processing part of the face pipeline class
//

partial class FacePipeline
{
    // Face/eye region trackers
    FaceRegion _faceRegion = new FaceRegion();
    EyeRegion _leyeRegion = new EyeRegion();
    EyeRegion _reyeRegion = new EyeRegion(true);
    float _faceDetectionScore;
    float _faceDetectionThreshold = 0.5f;

    // Vertex retrieval from the face landmark detector
    float4 GetFaceVertex(int index)
      => _landmarkDetector.face.VertexArray.ElementAt(index);

    IEnumerable<Vector4> FaceVertexArray()
      => _landmarkDetector.face.VertexArray;

    //Flipped because eyes are flipped to real life
    float4 GetEyeLVertex(int index)
      => _landmarkDetector.eyeR.VertexArray.ElementAt(index);

    //Flipped because eyes are flipped to real life
    float4 GetEyeRVertex(int index)
      => _landmarkDetector.eyeL.VertexArray.ElementAt(index);

    FaceDetector.Detection GetFace()
      => _faceDetector.Detections.FirstOrDefault();

    /// <summary>
    /// Get Face Yaw, Pitch and Roll using the same calculation in https://github.com/FIGLAB/EyeMU/blob/master/trainingSetup/landmarks3D.py
    /// Yaw is the angle in the x-z plane with the vertical axis at the origin
    /// Turning the head left is a positive angle, right is a negative angle, 0 is head on.
    /// </summary>
    /// <returns>The yaw in radians</returns>
    float GetFaceYaw()
    {
        var vertex50 = GetFaceVertex(50);
        var vertex280 = GetFaceVertex(280);
        return Mathf.Atan((vertex50.z - vertex280.z)/(vertex50.x - vertex280.x));
    }

    /// <summary>
    /// Pitch is the angle in the z-y plane with the horizontal axis at the origin
    /// Turning the head up is a positive angle, down is a negative angle, 0 is head on.
    /// </summary>
    /// <returns>The pitch in radians</returns>
    float GetFacePitch()
    {
        var vertex10 = GetFaceVertex(10);
        var vertex168 = GetFaceVertex(168);
        return -Mathf.Atan((vertex10.z - vertex168.z) / (vertex168.y - vertex10.y));
    }

    /// <summary>
    /// Roll is the angle in the x-y plane (face plane)
    /// Rotating the head clockwise is a negative angle, counterclockwise is a positive angle, 0 is no rotation.
    /// </summary>
    /// <returns>The roll in radians divided by 2</returns>
    float GetFaceRoll()
    {
        var vertex6 = GetFaceVertex(6);
        var vertex151 = GetFaceVertex(151);
        float roll = Mathf.Atan2(vertex151.x - vertex6.x, vertex6.y - vertex151.y);

        //Divide by 2 to lessen roll impact
        if (roll >= 0) return (roll - 3.141592f)/2;
        else return (roll + 3.141592f)/2;
    }

    /// <summary>
    /// Get Face Area in float by calculating the size of the bounding box
    /// </summary>
    /// <returns>The face area</returns>
    float GetFaceArea()
    {
        var bbox = _computeBuffer.bbox.GetBoundingBoxData();
        return (bbox.Max.x - bbox.Min.x) * (bbox.Max.y - bbox.Min.y);
    }

    /// <summary>
    /// Checks whether or not a user is detected.
    /// </summary>
    /// <param name="face">The face</param>
    /// <returns>Is a face detected</returns>
    public bool IsUserPresent()
    {
        return _faceDetectionScore >= _faceDetectionThreshold;
    }

    /// <summary>
    /// Sets face detection threshold
    /// <param name="threshold">new threshold</param>
    /// </summary>
    public void SetFaceDetectionThreshold(float threshold)
    {
        _faceDetectionThreshold = threshold;
    }

    void RunPipeline(Texture input)
    {
        // Face detection
        _faceDetector.ProcessImage(input);

        // Cancel if the face detection score is too low.
        var face = _faceDetector.Detections.FirstOrDefault();
        _faceDetectionScore = face.score;
        if (!IsUserPresent()) return;

        // Try updating the face region with the detection result. It's
        // actually updated only when there is a noticeable jump from the last
        // frame.
        _faceRegion.TryUpdateWithDetection(face);

        // Face region cropping
        _preprocess.SetMatrix("_Xform", _faceRegion.CropMatrix);
        Graphics.Blit(input, _cropRT.face, _preprocess, 0);

        // Face landmark detection
        _landmarkDetector.face.ProcessImage(_cropRT.face);

        // Key points from the face landmark
        var mouth    = _faceRegion.Transform(GetFaceVertex( 13)).xy;
        var mid_eyes = _faceRegion.Transform(GetFaceVertex(168)).xy;
        var eye_l0   = _faceRegion.Transform(GetFaceVertex( 33)).xy;
        var eye_l1   = _faceRegion.Transform(GetFaceVertex(133)).xy;
        var eye_r0   = _faceRegion.Transform(GetFaceVertex(362)).xy;
        var eye_r1   = _faceRegion.Transform(GetFaceVertex(263)).xy;

        // Eye region update
        _leyeRegion.Update(eye_l0, eye_l1, _faceRegion.RotationMatrix);
        _reyeRegion.Update(eye_r0, eye_r1, _faceRegion.RotationMatrix);

        // Eye region cropping
        _preprocess.SetMatrix("_Xform", _leyeRegion.CropMatrix);
        Graphics.Blit(input, _cropRT.eyeL, _preprocess, 0);

        _preprocess.SetMatrix("_Xform", _reyeRegion.CropMatrix);
        Graphics.Blit(input, _cropRT.eyeR, _preprocess, 0);

        // Eye landmark detection
        _landmarkDetector.eyeL.ProcessImage(_cropRT.eyeL);
        _landmarkDetector.eyeR.ProcessImage(_cropRT.eyeR);

        // Postprocess for face mesh construction
        var post = _resources.postprocessCompute;

        post.SetMatrix("_fx_xform", _faceRegion.CropMatrix);
        post.SetBuffer(0, "_fx_input", _landmarkDetector.face.VertexBuffer);
        post.SetBuffer(0, "_fx_output", _computeBuffer.post);
        post.SetBuffer(0, "_fx_bbox", _computeBuffer.bbox);
        post.Dispatch(0, 1, 1, 1);

        post.SetBuffer(1, "_e2f_index_table", _computeBuffer.eyeToFace);
        post.SetBuffer(1, "_e2f_eye_l", _landmarkDetector.eyeL.VertexBuffer);
        post.SetBuffer(1, "_e2f_eye_r", _landmarkDetector.eyeR.VertexBuffer);
        post.SetMatrix("_e2f_xform_l", _leyeRegion.CropMatrix);
        post.SetMatrix("_e2f_xform_r", _reyeRegion.CropMatrix);
        post.SetBuffer(1, "_e2f_face", _computeBuffer.post);
        post.Dispatch(1, 1, 1, 1);

        post.SetBuffer(2, "_lpf_input", _computeBuffer.post);
        post.SetBuffer(2, "_lpf_output", _computeBuffer.filter);
        post.SetFloat("_lpf_beta", 30.0f);
        post.SetFloat("_lpf_cutoff_min", 1.5f);
        post.SetFloat("_lpf_t_e", Time.deltaTime);
        post.Dispatch(2, 468 / 52, 1, 1);

        // Face region update based on the postprocessed face mesh
        _faceRegion.Step
          (_computeBuffer.bbox.GetBoundingBoxData(), mid_eyes - mouth);
    }

    /// <summary>
    /// Calculates the eye corners based on the face landmarks
    /// </summary>
    /// <returns>The list of eye corners</returns>
    float[] GetCorners()
    {
        var left_leftcorner = _faceRegion.Transform(GetFaceVertex(263));
        var left_rightcorner = _faceRegion.Transform(GetFaceVertex(362));
        var right_rightcorner = _faceRegion.Transform(GetFaceVertex(33));
        var right_leftcorner = _faceRegion.Transform(GetFaceVertex(133));

        return new float[] { left_leftcorner.x, left_leftcorner.y, left_rightcorner.x, left_rightcorner.y,
                             right_rightcorner.x, right_rightcorner.y, right_leftcorner.x, right_leftcorner.y };
    }
}

} // namespace MediaPipe.FaceMesh
