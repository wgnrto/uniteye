using UnityEngine;
using Unity.Mathematics;
using System.Collections.Generic;

namespace MediaPipe.FaceMesh {

//
// Public part of the face pipeline class
//

public partial class FacePipeline
{
    #region Accessors for vertex buffers

    public ComputeBuffer RawFaceVertexBuffer
      => _landmarkDetector.face.VertexBuffer;

    public ComputeBuffer RawLeftEyeVertexBuffer
      => _landmarkDetector.eyeL.VertexBuffer;

    public ComputeBuffer RawRightEyeVertexBuffer
      => _landmarkDetector.eyeR.VertexBuffer;

    public ComputeBuffer RefinedFaceVertexBuffer
      => _computeBuffer.filter;

    #endregion

    #region Accessors for cropped textures

    public Texture CroppedFaceTexture
      => _cropRT.face;

    public Texture CroppedLeftEyeTexture
      => _cropRT.eyeL;

    public Texture CroppedRightEyeTexture
      => _cropRT.eyeR;

    #endregion

    #region Accessors for crop region matrices

    public float4x4 FaceCropMatrix
      => _faceRegion.CropMatrix;

    public float4x4 FaceRotationMatrix
      => _faceRegion.RotationMatrix;

    public float4x4 LeftEyeCropMatrix
      => _leyeRegion.CropMatrix;

    public float4x4 RightEyeCropMatrix
      => _reyeRegion.CropMatrix;

    #endregion
    
    public float FaceDetectionScore
      => _faceDetectionScore;
    public float FaceDetectionThreshold
      => _faceDetectionThreshold;

    #region Accessors for face region getFaceVertex

    public FaceRegion FaceRegion
            => _faceRegion;

    #endregion

    #region Public methods

    public float4 GetFaceRegionVertex(int index)
    {
        return _faceRegion.Transform(GetFaceVertex(index));
    }

    public float4 GetRawFaceRegionVertex(int index)
    {
        return GetFaceVertex(index);
    }

    public IEnumerable<Vector4> GetFaceVertexArray()
        => FaceVertexArray();

    public float4 GetEyeLRegionVertex(int index)
    {
        return _faceRegion.Transform(GetEyeLVertex(index));
    }

    public float4 GetEyeRRegionVertex(int index)
    {
        return _faceRegion.Transform(GetEyeRVertex(index));
    }

    public float GetFaceRegionYaw()
      => GetFaceYaw();

    public float GetFaceRegionPitch()
      => GetFacePitch();

    public float GetFaceRegionRoll()
      => GetFaceRoll();

    public float GetFaceRegionArea()
      => GetFaceArea();

    public float[] GetEyeCorners()
      => GetCorners();

    public FacePipeline(ResourceSet resources)
      => AllocateObjects(resources);

    public void Dispose()
      => DeallocateObjects();

    public void ProcessImage(Texture image)
      => RunPipeline(image);

    #endregion
}

} // namespace MediaPipe.FaceMesh
