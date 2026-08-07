using System;
using System.Collections.Concurrent;
using System.Collections.Generic;
using System.Globalization;
using System.IO;
using System.Text;
using System.Threading;
using UnityEngine;
using UnityEngine.Rendering;

namespace UnitEye
{
    /// <summary>
    /// Writes a consented calibration session to disk as a dataset, so gaze accuracy work can be measured
    /// against recorded sessions instead of re-run by hand each time.
    ///
    /// Design constraints that shaped this, all of them load-bearing:
    ///
    /// 1. It must not change WHICH samples the calibration captures. The fixation gate accepts or rejects a
    ///    sample from the dispersion of recent raw gaze, so anything that stalls the main thread (a
    ///    synchronous ReadPixels, a PNG encode) changes frame timing, changes dispersion, and therefore
    ///    changes the data it was supposed to be observing. All pixel work goes through AsyncGPUReadback
    ///    and all encoding happens on a writer thread.
    /// 2. Rows and images must be 1:1 BY CONSTRUCTION. It is hooked at the tail of the capture path, after
    ///    every rejection, and keyed on the training-sample index — so an image can never exist for a sample
    ///    that was not trained on, and offline joins cannot silently drift.
    /// 3. It never uploads anything, and contains no network code. Consent to publish is not publication;
    ///    a human decides that out of band.
    /// 4. Numbers are written with InvariantCulture. On a de-DE machine the ambient culture renders 0.4193f
    ///    as "0,4193", which turns a comma-separated dataset into garbage on someone else's computer.
    /// </summary>
    public class GazeSessionRecorder : IDisposable
    {
        public const string DatasetFormatVersion = "1";

        //Bounded so a slow disk costs frames of imagery rather than unbounded RAM; drops are counted and
        //surfaced in the summary, because a dataset that silently lost 30% of its images is a trap.
        private const int MaxQueuedImages = 64;

        private readonly string _root;
        private readonly GazeConsentRecord _consent;
        private readonly GazeRecordingTier _tier;
        private StreamWriter _rows;
        private FileStream _features, _landmarks;
        private readonly BlockingCollection<PendingImage> _imageQueue;
        private readonly Thread _writer;
        private readonly float[] _landmarkBuffer;
        private readonly float[] _blendshapeBuffer = new float[10];

        private RenderTexture _leftCopy, _rightCopy, _frameCopy;
        private int _samples, _imagesWritten, _imagesDropped, _readbackFailures;
        private int _inFlight;

        public bool Recording { get; private set; }
        public GazeRecordingTier Tier => _tier;
        public string SessionFolder => _root;
        public int SampleCount => _samples;
        public int ImagesDropped => _imagesDropped;
        public string ParticipantToken => _consent != null ? _consent.participantToken : "";

        private struct PendingImage
        {
            public string RelativePath;
            public byte[] Pixels;      // RGBA32, bottom-up (GPU origin)
            public int Width, Height;
            public bool Lossless;
        }

        /// <summary>
        /// Root for all recordings. persistentDataPath, never streamingAssetsPath: the latter is read-only
        /// on Android/WebGL, is inside the Unity project, and is exactly where participant data must not be
        /// so that nobody commits it by reflex. Publication-consented and local-only sessions go to separate
        /// roots so excluding a set is a directory operation, not a flag a script can forget to read.
        /// </summary>
        public static string RootFor(bool mayPublish) => Path.Combine(Application.persistentDataPath,
            "UnitEyeRecordings", mayPublish ? "publication-consented" : "local-only");

        /// <summary>
        /// Opens a session. Writes consent.json FIRST and throws if that fails: data whose terms could not be
        /// recorded must not exist. Caller is responsible for having obtained the consent it passes in.
        /// </summary>
        public GazeSessionRecorder(GazeConsentRecord consent, GazeRecordingTier tier, int landmarkCapacity)
        {
            if (consent == null) throw new ArgumentNullException(nameof(consent));
            _consent = consent;
            _tier = tier;

            //Folder name is the participant token, so a withdrawal request maps to a directory with no lookup
            //table — and no date in the path, which would leak session timing.
            _root = Path.Combine(RootFor(consent.mayPublish), consent.participantToken);
            Directory.CreateDirectory(_root);

            File.WriteAllText(Path.Combine(_root, "consent.json"),
                JsonUtility.ToJson(consent, prettyPrint: true), new UTF8Encoding(false));

            _rows = new StreamWriter(Path.Combine(_root, "samples.jsonl"), append: false, new UTF8Encoding(false));
            _features = new FileStream(Path.Combine(_root, "features.f32"), FileMode.Create, FileAccess.Write);
            if (tier >= GazeRecordingTier.Landmarks)
            {
                _landmarks = new FileStream(Path.Combine(_root, "landmarks.f32"), FileMode.Create, FileAccess.Write);
                _landmarkBuffer = new float[Mathf.Max(1, landmarkCapacity) * 3];
            }

            if (tier >= GazeRecordingTier.EyeCrops)
            {
                _imageQueue = new BlockingCollection<PendingImage>(MaxQueuedImages);
                _writer = new Thread(WriterLoop) { IsBackground = true, Name = "UnitEye recording writer" };
                _writer.Start();
                Directory.CreateDirectory(Path.Combine(_root, "eyes"));
                if (tier >= GazeRecordingTier.FaceVideo) Directory.CreateDirectory(Path.Combine(_root, "frames"));
            }

            Recording = true;
        }

        /// <summary>
        /// Writes the session header. Separate from the constructor because most of it is only knowable once
        /// the calibration has started (backbone, feature length, screen geometry).
        /// </summary>
        /// <remarks>
        /// Deliberately carries no feature-vector length. It is not knowable when the session opens — the
        /// direction backbones append a runtime-sized embedding tail only after their first inference — so a
        /// header field here would be a confident 0. Each row carries its own authoritative featureCount.
        /// </remarks>
        public void WriteSessionHeader(string backbone, int screenWidth, int screenHeight,
            float screenWidthCm, float screenHeightCm, int frameWidth, int frameHeight,
            bool flipH, bool flipV, bool landmarksSmoothed, int landmarkCount,
            bool rollNormalizeCrops, bool flipAugmentation, string glassesState)
        {
            var sb = new StringBuilder();
            sb.Append("{");
            Str(sb, "datasetFormatVersion", DatasetFormatVersion); sb.Append(",");
            Str(sb, "tier", _tier.ToString()); sb.Append(",");
            Str(sb, "backbone", backbone); sb.Append(",");
            Num(sb, "screenWidthPx", screenWidth); sb.Append(",");
            Num(sb, "screenHeightPx", screenHeight); sb.Append(",");
            Num(sb, "screenWidthCm", screenWidthCm); sb.Append(",");
            Num(sb, "screenHeightCm", screenHeightCm); sb.Append(",");
            //Physical-scale provenance. screenWidthCm is derived from Screen.dpi assuming one render pixel
            //covers one physical pixel; when that is false (the usual case for a windowed Editor Game view)
            //every centimetre and degree figure downstream is wrong by the same ratio. Recorded rather than
            //corrected, because it cannot be corrected after the fact — but it CAN be detected, so a pooled
            //dataset does not silently mix incompatible physical units.
            Num(sb, "displayWidthPx", ScreenGeometry.DisplayWidth); sb.Append(",");
            Num(sb, "displayHeightPx", ScreenGeometry.DisplayHeight); sb.Append(",");
            Num(sb, "screenDpi", Screen.dpi); sb.Append(",");
            Bool(sb, "renderMatchesDisplay", ScreenGeometry.RenderMatchesDisplay); sb.Append(",");
            Bool(sb, "recordedInEditor", Application.isEditor); sb.Append(",");
            //The single field a consumer should gate on before trusting cm or degrees from this session.
            Bool(sb, "physicalScaleTrustworthy",
                ScreenGeometry.RenderMatchesDisplay && !Application.isEditor && Screen.dpi > 1f); sb.Append(",");
            Num(sb, "cameraFrameWidth", frameWidth); sb.Append(",");
            Num(sb, "cameraFrameHeight", frameHeight); sb.Append(",");
            Bool(sb, "frameFlippedHorizontally", flipH); sb.Append(",");
            Bool(sb, "frameFlippedVertically", flipV); sb.Append(",");
            Bool(sb, "gazeLandmarksSmoothed", landmarksSmoothed); sb.Append(",");
            Num(sb, "landmarkCount", landmarkCount); sb.Append(",");
            Bool(sb, "rollNormalizeCrops", rollNormalizeCrops); sb.Append(",");
            Bool(sb, "flipAugmentation", flipAugmentation); sb.Append(",");
            Str(sb, "glassesState", glassesState); sb.Append(",");
            //Recorded so a consumer knows the landmark block's coordinate convention without guessing.
            Str(sb, "landmarkSpace", "mediapipe-normalized-0-1-ydown-cameraframe"); sb.Append(",");
            Str(sb, "imageOrigin", "bottom-left-gpu");
            sb.Append("}");
            File.WriteAllText(Path.Combine(_root, "session.json"), sb.ToString(), new UTF8Encoding(false));
        }

        /// <summary>
        /// Records one ACCEPTED training sample. <paramref name="sampleIndex"/> must be the index this sample
        /// takes in the calibration's own training arrays — that is what makes rows joinable to the trained
        /// model, and it restarts at 0 when a cancelled calibration is re-run, exactly as the arrays do.
        /// Never throws into the caller: a recording failure must not take down a calibration.
        /// </summary>
        public void RecordSample(int sampleIndex, float[] features, Vector2 labelPx, Vector2 targetPx,
            double sessionSeconds, bool atDwell, bool headRotation, int preset, int round,
            float distanceMm, Vector3 headPose, float irisDisagreement, bool blinking,
            IGazeRecordingSource source, IGazeProvider provider)
        {
            if (!Recording) return;
            try
            {
                long featureOffset = _features.Position;
                WriteFloats(_features, features, features.Length);

                long landmarkOffset = -1;
                int landmarkFloats = 0;
                if (_landmarks != null && source != null)
                {
                    landmarkFloats = source.TryCopyLandmarks(_landmarkBuffer);
                    if (landmarkFloats > 0)
                    {
                        landmarkOffset = _landmarks.Position;
                        WriteFloats(_landmarks, _landmarkBuffer, landmarkFloats);
                    }
                }

                bool haveBlendshapes = source != null && source.TryCopyEyeBlendshapes(_blendshapeBuffer);

                var sb = new StringBuilder(512);
                sb.Append("{");
                Num(sb, "i", sampleIndex); sb.Append(",");
                Num(sb, "t", sessionSeconds); sb.Append(",");
                Num(sb, "labelX", labelPx.x); sb.Append(",");
                Num(sb, "labelY", labelPx.y); sb.Append(",");
                Num(sb, "targetX", targetPx.x); sb.Append(",");
                Num(sb, "targetY", targetPx.y); sb.Append(",");
                Bool(sb, "dwell", atDwell); sb.Append(",");
                Bool(sb, "headRotation", headRotation); sb.Append(",");
                Num(sb, "preset", preset); sb.Append(",");
                Num(sb, "round", round); sb.Append(",");
                Bool(sb, "blinking", blinking); sb.Append(",");
                Num(sb, "irisDisagreement", irisDisagreement); sb.Append(",");
                Num(sb, "featureOffset", featureOffset); sb.Append(",");
                Num(sb, "featureCount", features.Length);
                //Omit rather than zero-fill anything the platform could not supply: a sentinel written as a
                //measurement is indistinguishable from a real one downstream.
                if (distanceMm > -999f) { sb.Append(","); Num(sb, "distanceMm", distanceMm); }
                if (headPose != Vector3.zero)
                {
                    sb.Append(","); Num(sb, "headPitch", headPose.x);
                    sb.Append(","); Num(sb, "headYaw", headPose.y);
                    sb.Append(","); Num(sb, "headRoll", headPose.z);
                }
                if (landmarkOffset >= 0)
                {
                    sb.Append(","); Num(sb, "landmarkOffset", landmarkOffset);
                    sb.Append(","); Num(sb, "landmarkCount", landmarkFloats / 3);
                }
                if (haveBlendshapes)
                {
                    sb.Append(",\"eyeBlendshapes\":[");
                    for (int k = 0; k < _blendshapeBuffer.Length; k++)
                    {
                        if (k > 0) sb.Append(",");
                        sb.Append(_blendshapeBuffer[k].ToString("R", CultureInfo.InvariantCulture));
                    }
                    sb.Append("]");
                }
                sb.Append("}");
                _rows.WriteLine(sb.ToString());
                _samples++;

                if (_tier >= GazeRecordingTier.EyeCrops)
                    CaptureImagery(sampleIndex, source, provider);
            }
            catch (Exception e)
            {
                UnitEyeLog.Error("Calibration recording failed; continuing the calibration without it.");
                UnitEyeLog.Exception(e);
                Recording = false;
            }
        }

        private void CaptureImagery(int sampleIndex, IGazeRecordingSource source, IGazeProvider provider)
        {
            if (provider != null)
            {
                //Blit into recorder-owned copies FIRST. The backbone's crop textures are the destination of a
                //Graphics.Blit on the very next camera frame, so an async readback issued straight against
                //them can complete holding a later frame's pixels — silently pairing the wrong image with
                //this row. A copy costs one GPU blit and removes the race entirely.
                Snapshot(provider.LeftEyeTexture, ref _leftCopy, $"eyes/{sampleIndex:D6}_L.png", lossless: true);
                Snapshot(provider.RightEyeTexture, ref _rightCopy, $"eyes/{sampleIndex:D6}_R.png", lossless: true);
            }

            if (_tier >= GazeRecordingTier.FaceVideo && source != null && source.CameraTexture != null)
                SnapshotFrame(sampleIndex, source);
        }

        private void Snapshot(RenderTexture src, ref RenderTexture copy, string relativePath, bool lossless)
        {
            if (src == null) return;
            if (copy == null || copy.width != src.width || copy.height != src.height)
            {
                if (copy != null) copy.Release();
                copy = new RenderTexture(src.width, src.height, 0, RenderTextureFormat.ARGB32);
                copy.Create();
            }
            Graphics.Blit(src, copy);
            RequestReadback(copy, relativePath, lossless);
        }

        private void SnapshotFrame(int sampleIndex, IGazeRecordingSource source)
        {
            var cam = source.CameraTexture;
            int w = cam.width, h = cam.height;
            if (w <= 0 || h <= 0) return;

            //FaceVideo crops to the face box so the room (and any bystander) never enters the dataset; only
            //FullFrames keeps the background, and only behind its own confirmation screen.
            var rect = new Rect(0f, 0f, 1f, 1f);
            if (_tier == GazeRecordingTier.FaceVideo)
            {
                var b = source.FaceBoundsNormalized;
                if (b.width <= 0f || b.height <= 0f) return;
                //Pad so the crop keeps the whole head rather than clipping at the landmark hull.
                float padX = b.width * 0.35f, padY = b.height * 0.35f;
                rect = Rect.MinMaxRect(
                    Mathf.Clamp01(b.xMin - padX), Mathf.Clamp01(b.yMin - padY),
                    Mathf.Clamp01(b.xMax + padX), Mathf.Clamp01(b.yMax + padY));
                if (rect.width <= 0f || rect.height <= 0f) return;
            }

            int outW = Mathf.Max(16, Mathf.RoundToInt(w * rect.width));
            int outH = Mathf.Max(16, Mathf.RoundToInt(h * rect.height));
            if (_frameCopy == null || _frameCopy.width != outW || _frameCopy.height != outH)
            {
                if (_frameCopy != null) _frameCopy.Release();
                _frameCopy = new RenderTexture(outW, outH, 0, RenderTextureFormat.ARGB32);
                _frameCopy.Create();
            }
            //Landmarks are y-down while the GPU is y-up; the crop rect is flipped to match so the saved frame
            //corresponds to the region the (y-down) face box describes.
            var scale = new Vector2(rect.width, rect.height);
            var offset = new Vector2(rect.xMin, 1f - rect.yMax);
            Graphics.Blit(cam, _frameCopy, scale, offset);
            RequestReadback(_frameCopy, $"frames/{sampleIndex:D6}.jpg", lossless: false);
        }

        private void RequestReadback(RenderTexture rt, string relativePath, bool lossless)
        {
            if (!SystemInfo.supportsAsyncGPUReadback)
            {
                //No silent fallback to a synchronous ReadPixels: it would stall the main thread and change
                //which samples the fixation gate accepts, i.e. corrupt the very data being recorded.
                _imagesDropped++;
                return;
            }
            if (_inFlight >= MaxQueuedImages) { _imagesDropped++; return; }

            int w = rt.width, h = rt.height;
            Interlocked.Increment(ref _inFlight);
            AsyncGPUReadback.Request(rt, 0, TextureFormat.RGBA32, req =>
            {
                Interlocked.Decrement(ref _inFlight);
                if (req.hasError) { _readbackFailures++; return; }
                //The NativeArray is only valid inside this callback, so copy before handing it off.
                var pixels = req.GetData<byte>().ToArray();
                if (_imageQueue == null || _imageQueue.IsAddingCompleted) return;
                if (!_imageQueue.TryAdd(new PendingImage
                {
                    RelativePath = relativePath, Pixels = pixels,
                    Width = w, Height = h, Lossless = lossless,
                }))
                {
                    //Bounded queue full: drop this image rather than block the main thread or grow without limit.
                    _imagesDropped++;
                }
            });
        }

        private void WriterLoop()
        {
            try
            {
                foreach (var img in _imageQueue.GetConsumingEnumerable())
                {
                    try
                    {
                        //Array-based encoders work on raw pixels and are safe off the main thread, unlike the
                        //Texture2D overloads. Eye crops are PNG: they are tiny and they are the model's actual
                        //input, so JPEG ringing around the iris edge is not worth the bytes.
                        var bytes = img.Lossless
                            ? ImageConversion.EncodeArrayToPNG(img.Pixels,
                                UnityEngine.Experimental.Rendering.GraphicsFormat.R8G8B8A8_UNorm,
                                (uint)img.Width, (uint)img.Height)
                            : ImageConversion.EncodeArrayToJPG(img.Pixels,
                                UnityEngine.Experimental.Rendering.GraphicsFormat.R8G8B8A8_UNorm,
                                (uint)img.Width, (uint)img.Height, 0, 90);
                        if (bytes != null && bytes.Length > 0)
                        {
                            File.WriteAllBytes(Path.Combine(_root, img.RelativePath), bytes);
                            Interlocked.Increment(ref _imagesWritten);
                        }
                    }
                    catch (Exception e)
                    {
                        Interlocked.Increment(ref _imagesDropped);
                        UnitEyeLog.Exception(e);
                    }
                }
            }
            catch (Exception e)
            {
                UnitEyeLog.Exception(e);
            }
        }

        /// <summary>
        /// Closes the session and writes summary.json. Reports dropped images explicitly — a dataset that
        /// quietly lost frames looks complete and is not.
        /// </summary>
        public void Finish(string outcome, float holdoutRmseCm)
        {
            if (!Recording && _rows == null) return;
            Recording = false;
            try
            {
                _rows?.Flush();
                _features?.Flush();
                _landmarks?.Flush();

                //Give in-flight readbacks a chance to land before the queue closes. Bounded: a wedged GPU
                //must not hang the calibration's completion screen.
                if (_imageQueue != null)
                {
                    AsyncGPUReadback.WaitAllRequests();
                    _imageQueue.CompleteAdding();
                    _writer?.Join(TimeSpan.FromSeconds(10));
                }

                var sb = new StringBuilder();
                sb.Append("{");
                Str(sb, "outcome", outcome); sb.Append(",");
                Num(sb, "samples", _samples); sb.Append(",");
                Num(sb, "imagesWritten", _imagesWritten); sb.Append(",");
                Num(sb, "imagesDropped", _imagesDropped); sb.Append(",");
                Num(sb, "readbackFailures", _readbackFailures); sb.Append(",");
                Num(sb, "holdoutRmseCm", holdoutRmseCm);
                sb.Append("}");
                File.WriteAllText(Path.Combine(_root, "summary.json"), sb.ToString(), new UTF8Encoding(false));
            }
            catch (Exception e) { UnitEyeLog.Exception(e); }
            finally { Dispose(); }
        }

        /// <summary>
        /// Deletes everything recorded in this session. Backs the "delete my recording now" button, which has
        /// to exist: a participant who changes their mind thirty seconds later should not have to email anyone.
        /// </summary>
        public void DeleteEverything()
        {
            Recording = false;
            Dispose();
            try { if (Directory.Exists(_root)) Directory.Delete(_root, recursive: true); }
            catch (Exception e) { UnitEyeLog.Exception(e); }
        }

        public void Dispose()
        {
            try { _rows?.Dispose(); } catch { } finally { _rows = null; }
            try { _features?.Dispose(); } catch { } finally { _features = null; }
            try { _landmarks?.Dispose(); } catch { } finally { _landmarks = null; }
            try { if (_imageQueue != null && !_imageQueue.IsAddingCompleted) _imageQueue.CompleteAdding(); } catch { }
            if (_leftCopy != null) { _leftCopy.Release(); _leftCopy = null; }
            if (_rightCopy != null) { _rightCopy.Release(); _rightCopy = null; }
            if (_frameCopy != null) { _frameCopy.Release(); _frameCopy = null; }
        }

        private static void WriteFloats(FileStream fs, float[] values, int count)
        {
            var bytes = new byte[count * 4];
            Buffer.BlockCopy(values, 0, bytes, 0, bytes.Length);
            fs.Write(bytes, 0, bytes.Length);
        }

        //InvariantCulture everywhere: the ambient culture would emit "0,4193" on a German machine and make
        //the dataset unreadable (or worse, subtly mis-parsed) anywhere else. "R" round-trips exactly.
        private static void Num(StringBuilder sb, string key, float v)
            => sb.Append('"').Append(key).Append("\":").Append(
                float.IsNaN(v) || float.IsInfinity(v) ? "null" : v.ToString("R", CultureInfo.InvariantCulture));
        private static void Num(StringBuilder sb, string key, double v)
            => sb.Append('"').Append(key).Append("\":").Append(
                double.IsNaN(v) || double.IsInfinity(v) ? "null" : v.ToString("R", CultureInfo.InvariantCulture));
        private static void Num(StringBuilder sb, string key, long v)
            => sb.Append('"').Append(key).Append("\":").Append(v.ToString(CultureInfo.InvariantCulture));
        private static void Bool(StringBuilder sb, string key, bool v)
            => sb.Append('"').Append(key).Append("\":").Append(v ? "true" : "false");
        private static void Str(StringBuilder sb, string key, string v)
            => sb.Append('"').Append(key).Append("\":\"").Append((v ?? "").Replace("\\", "\\\\").Replace("\"", "\\\"")).Append('"');
    }
}
