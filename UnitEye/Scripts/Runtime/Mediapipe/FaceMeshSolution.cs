// Updated by Tobias Wagner for the UnitEye Project 07/2023
// Ulm University, Institute of Media Informatics
//
// Based on Assets/MediaPipeUnity/Samples/Scenes/Face Mesh/FaceMeshSolution.cs
// Copyright (c) 2021 homuler
//
// Use of this source code is governed by an MIT-style
// license that can be found in the LICENSE file or at
// https://opensource.org/licenses/MIT.

using System;
using System.Collections;
using System.Collections.Generic;
using System.Linq;
using UnityEngine;

namespace Mediapipe.Unity.FaceMesh
{
    public class FaceMeshSolution : ImageSourceSolution<FaceMeshGraph>
    {
        [SerializeField] private MultiFaceLandmarkListAnnotationController _multiFaceLandmarksAnnotationController;
        [SerializeField] private NormalizedRectListAnnotationController _faceRectsFromLandmarksAnnotationController;
        [SerializeField] private bool _annotate = true;

        public bool IsFaceDetected { get; private set; } = false;

        public bool Annotate
        {
            get => _annotate;

            set
            {
                _annotate = value;
                _multiFaceLandmarksAnnotationController.gameObject.SetActive(_annotate);
            }
        }

        public bool IsRendering { get => screen.IsRendering; set => screen.IsRendering = value; }

        public int maxNumFaces
        {
            get => graphRunner.maxNumFaces;
            set => graphRunner.maxNumFaces = value;
        }

        public bool refineLandmarks
        {
            get => graphRunner.refineLandmarks;
            set => graphRunner.refineLandmarks = value;
        }

        public float minDetectionConfidence
        {
            get => graphRunner.minDetectionConfidence;
            set => graphRunner.minDetectionConfidence = value;
        }

        public float minTrackingConfidence
        {
            get => graphRunner.minTrackingConfidence;
            set => graphRunner.minTrackingConfidence = value;
        }

        public IList<NormalizedLandmark> FaceLandmarks { get; private set; }

        public IList<NormalizedLandmark> LeftIrisLandmarks { get; private set; }

        public IList<NormalizedLandmark> RightIrisLandmarks { get; private set; }

        public IList<NormalizedRect> FaceRects { get; private set; }

        public float[] EyeCorners
        {
            get => LeftEyeCorners.Concat(RightEyeCorners).ToArray();

        }

        protected float[] LeftEyeCorners
        {
            get => new float[] {
                FaceLandmarks[263].X, FaceLandmarks[263].Y,
                FaceLandmarks[362].X, FaceLandmarks[362].Y
            };
        }

        protected float[] RightEyeCorners
        {
            get => new float[] {
                FaceLandmarks[33].X, FaceLandmarks[33].Y,
                FaceLandmarks[133].X, FaceLandmarks[133].Y
            };
        }

        public float HeadYaw
        {
            get
            {
                var l50 = FaceLandmarks[50];
                var l280 = FaceLandmarks[280];
                return Mathf.Atan((l50.Z - l280.Z) / (l50.X - l280.X));
            }
        }

        public float HeadPitch
        {
            get
            {
                var l10 = FaceLandmarks[10];
                var l168 = FaceLandmarks[168];
                return Mathf.Atan((l10.Z - l168.Z) / (l168.Y - l10.Y));
            }
        }

        public float HeadRoll
        {
            get
            {
                var l6 = FaceLandmarks[6];
                var l151 = FaceLandmarks[6];
                float roll = Mathf.Atan2(l151.X - l6.X, l6.Y - l151.Y);

                //Divide by 2 to lessen roll impact
                if (roll >= 0)
                    return (roll - Mathf.PI) / 2;
                else
                    return (roll + Mathf.PI) / 2;
            }
        }

        public float HeadArea
        {
            get
            {
                if (FaceRects == null)
                    return 0.0f;
                return FaceRects[0].Width * FaceRects[0].Height;
            }
        }

        public float[] HeadGeom
        {
            get => new float[4] { HeadYaw, HeadPitch, HeadRoll, HeadArea };
        }

        protected override void OnStartRun()
        {
            if (!runningMode.IsSynchronous())
            {
                graphRunner.OnMultiFaceLandmarksOutput += OnMultiFaceLandmarksOutput;
                graphRunner.OnFaceRectsFromLandmarksOutput += OnFaceRectsFromLandmarksOutput;
                graphRunner.OnFaceDetectionsOutput += OnFaceDetectionOutput;
            }

            var imageSource = ImageSourceProvider.ImageSource;

            if (!_annotate)
                return;

            SetupAnnotationController(_faceRectsFromLandmarksAnnotationController, imageSource);
            SetupAnnotationController(_multiFaceLandmarksAnnotationController, imageSource);
        }

        protected override void AddTextureFrameToInputStream(TextureFrame textureFrame)
        {
            graphRunner.AddTextureFrameToInputStream(textureFrame);
        }

        protected override IEnumerator WaitForNextValue()
        {
            List<Detection> faceDetections = null;
            List<NormalizedLandmarkList> multiFaceLandmarks = null;
            List<NormalizedRect> faceRectsFromLandmarks = null;
            List<NormalizedRect> faceRectsFromDetections = null;

            if (runningMode == RunningMode.Sync)
            {
                var _ = graphRunner.TryGetNext(out faceDetections, out multiFaceLandmarks, out faceRectsFromLandmarks, out faceRectsFromDetections, true);
                if (multiFaceLandmarks == null)
                    yield break;
                (FaceLandmarks, LeftIrisLandmarks, RightIrisLandmarks) = FaceLandmarkListWithIrisAnnotation.PartitionLandmarkList(multiFaceLandmarks[0].Landmark);
            }
            else if (runningMode == RunningMode.NonBlockingSync)
            {
                yield return new WaitUntil(() => graphRunner.TryGetNext(out faceDetections, out multiFaceLandmarks, out faceRectsFromLandmarks, out faceRectsFromDetections, false));
                if (multiFaceLandmarks == null)
                    yield break;
                (FaceLandmarks, LeftIrisLandmarks, RightIrisLandmarks) = FaceLandmarkListWithIrisAnnotation.PartitionLandmarkList(multiFaceLandmarks[0].Landmark);
            }

            if (_annotate)
            {
                _faceRectsFromLandmarksAnnotationController.DrawNow(faceRectsFromLandmarks);
                _multiFaceLandmarksAnnotationController.DrawNow(multiFaceLandmarks);
            }
        }

        private void OnMultiFaceLandmarksOutput(object stream, OutputEventArgs<List<NormalizedLandmarkList>> eventArgs)
        {
            if (_annotate)
                _multiFaceLandmarksAnnotationController.DrawLater(eventArgs.value);

            if (eventArgs.value == null)
                return;
            
            (FaceLandmarks, LeftIrisLandmarks, RightIrisLandmarks) = FaceLandmarkListWithIrisAnnotation.PartitionLandmarkList(eventArgs.value[0].Landmark);
        }

        private void OnFaceRectsFromLandmarksOutput(object stream, OutputEventArgs<List<NormalizedRect>> eventArgs)
        {
            if (_annotate)
                _faceRectsFromLandmarksAnnotationController.DrawLater(eventArgs.value);

            FaceRects = eventArgs.value;
        }

        private void OnFaceDetectionOutput(object stream, OutputEventArgs<List<Detection>> eventArgs)
        {
            if (eventArgs.value == null)
            {
                IsFaceDetected = false;
                return;
            }

            IsFaceDetected = true;
        }
    }
}
