# homuler MediaPipe Unity Plugin upgrade — 0.12.0 → 0.16.3 (DONE)

UnitEye's vendored [MediaPipe Unity Plugin](https://github.com/homuler/MediaPipeUnityPlugin) was
upgraded from **0.12.0** (legacy Solution API) to **0.16.3** (Task API, `FaceLandmarker`) and the
face-mesh seam was rewritten to match. Completed July 2026 — this document records what changed and
which upgrades were deliberately deferred.

## What was done (Option B — Task-API rewrite)

- **Re-vendored** `com.github.homuler.mediapipe` 0.12.0 → **0.16.3** (native binaries + the
  self-contained `face_landmarker_v2.bytes` `.task` bundle). The `Mediapipe.Runtime` asmdef GUID is
  unchanged, so `UniUlm.UnitEye.Runtime` / `UniUlm.UnitEye.MediapipeVendored` kept resolving without
  edits. `package.json` dependency bumped to `0.16.3`.
- **Rewrote `FaceMeshSolution.cs`** as a self-contained facade over
  `Mediapipe.Tasks.Vision.FaceLandmarker` (`RunningMode.VIDEO`). It loads the `.task` bundle from
  `StreamingAssets`, runs `TryDetectForVideo` per frame, and **keeps the exact same public surface**
  the pipeline consumes (`FaceLandmarks`, `Left/RightIrisLandmarks`, `HeadYaw/Pitch/Roll/Area`,
  `HeadGeom`, `EyeCorners`, `Annotate`/`IsRendering`) — so `NativeGazeProvider` and the runners were
  untouched. The 478 landmarks are copied into reused `Mediapipe.NormalizedLandmark` objects each
  frame (zero per-frame allocation).
- **Rewrote `WebCamSource.cs`** as a lean `WebCamTexture` wrapper (the old Solution-era
  `ImageSource`/`TextureFrame` stack was deleted). Same file GUID + `Mediapipe.Unity` namespace, so
  the prefab component still binds.
- **Deleted** the whole Solution-era subset: `FaceMeshGraph`, `GraphRunner`, `ImageSourceSolution`,
  `Screen`, `TextureFramePool`, `Bootstrap`, `GlobalConfigManager`, `AssetLoader`, `WaitForResult`,
  `MemoizedLogger`, `StartSceneController`, `AutoFit`, the `ImageSource`/`StaticImageSource`/
  `VideoSource` sources, and the dead `face_mesh_{cpu,gpu,opengles}.txt` graph configs.
- **`MediaPipeAssetInstaller`** now installs a single file, `face_landmarker_v2.bytes` (the `.task`
  bundle is self-contained: detector + 478-landmark model with iris), instead of the three legacy
  `.bytes` models.
- **`MigrationCleanup.cs`** (menu `UnitEye ▸ Cleanup Missing Scripts`) stripped the missing-script
  placeholders the deleted components left on the scenes/prefabs. One-time tool; safe to delete once
  the scenes are re-saved everywhere.
- Compile-verified headless on 6000.5.3f1: the **smoke suite passes (103 checks)**.

## Landmark remap (the key numeric change)

The Task API returns `result.faceLandmarks[0].landmarks` as the **478-point list with iris already
appended** — no `PartitionLandmarkList` step. The partition is fixed:

- face mesh **0–467**
- **left iris 468–472**, **right iris 473–477**

`FaceMeshSolution` slices `Left/RightIrisLandmarks` from those ranges. All the hardcoded indices the
downstream math relies on (eye corners 263/362/33/133, head-pose 50/280/10/168/6/151, and the EAR
indices in `HomulerEyeHelper`) are unchanged from the old face-mesh output and were preserved.

**`HeadArea` changed meaning.** The old Solution path read `face_rects_from_landmarks`, which was
`null` in synchronous mode, so `HeadArea` was effectively always `0`. The Task API has no such
stream, so `HeadArea` is now derived from the landmark bounding box — a **nonzero** value. Because
`HeadArea` is part of the calibration feature vector, **existing calibrations must be redone** after
the upgrade.

## Consumers of the FaceMesh output (unchanged surface)

The rewrite preserved this surface, so these first-party consumers did not change:

| Consumer | What it reads |
|---|---|
| `NativeGazeProvider` | constructs `FaceMeshSolution` + `WebCamSource`; drives the per-frame tick |
| `HomulerEyeMURunner` | `FaceLandmarks`, `EyeCorners` (263/362/33/133), `HeadGeom` → EyeMU inputs |
| `GazeEstimationRunner` | eye-crop rect from `FaceLandmarks` bbox → mobileone/mobilenetv2 inputs |
| `HomulerEyeHelper` | `FaceLandmarks[i]` for EAR (blink/drowsy), iris landmarks, `HeadYaw`/`HeadPitch` |
| `HomulerFunctions` | eye-crop rect from `FaceLandmarks` corners |
| `LandmarkVisualizer` | single `FaceLandmarks[i]` for a debug sphere |

Everything downstream of `IGazeProvider` (calibration, filtering, AOI, CSV) and the WebGL browser
path (`webgl/`, never used homuler native) were untouched — the point of the provider seam.

## Still owed — webcam validation

The migration is compile-verified but not yet validated against a live camera. On a rig with a
webcam, run `HomulerGazeScene` and check:

1. **Face detected / eyes tracked** — the crosshair should follow gaze, not sit in a corner.
2. **Orientation** — if gaze is mirrored or upside-down, flip `_flipHorizontally` / `_flipVertically`
   on `FaceMeshSolution` (serialized hand-test knobs; currently `_flipVertically = true`).
3. **Iris indices** — if iris-based signals look swapped, the 468–472 / 473–477 split is the knob.
4. **Recalibrate** — because `HeadArea` changed from 0 to bbox area (see above).

## Deferred follow-ups (optional accuracy wins, not part of the port)

The Task API exposes two better-conditioned signals that were intentionally **not** adopted, because
each changes the calibration feature vector (now 19 terms for EyeMU) and would force a recalibration +
re-verification of the shipped contract:

- **`eyeBlink` blendshapes** instead of the hand-rolled EAR for blink/drowsy detection.
- **Facial transformation matrix** instead of the landmark-difference `HeadYaw/Pitch/Roll` math.

Both are cleaner; treat them as a separate, deliberately-scoped change if pursued.

## MediaPipe 1.0.0 (July 2026) — where each path stands

Google shipped [MediaPipe **v1.0.0**](https://github.com/google-ai-edge/mediapipe/releases/tag/v1.0.0),
its first stable release. UnitEye consumes MediaPipe through **two independent paths**, and only one of
them can move today:

| Path | Consumes MediaPipe via | Status |
| --- | --- | --- |
| **Web / WebGL** (`webgl/uniteye-cv.js`) | npm `@mediapipe/tasks-vision` (Google, direct) | **Updated to 1.0.0** and verified — see [`webgl/README.md`](../webgl/README.md) |
| **Native** (Unity desktop/Android) | [homuler MediaPipe Unity Plugin](https://github.com/homuler/MediaPipeUnityPlugin), vendored | **Stays on plugin 0.16.3 → MediaPipe v0.10.22** |

### Why the native path cannot follow yet

The native path does not depend on MediaPipe directly — it depends on homuler's plugin, which vendors
a **prebuilt** MediaPipe (`mediapipe_c.dll` / `.so` / `.dylib` + Android/iOS binaries) plus generated
C# protobuf bindings. As of this writing homuler's latest release is **v0.16.3** (8 Nov 2025), which
bundles **MediaPipe v0.10.22**. There is no plugin release built against MediaPipe 1.0.0, so there is
nothing to vendor.

Moving the native path would mean building homuler's plugin from source against MediaPipe 1.0.0: a
Bazel build per target platform (Windows/macOS/Linux/Android/iOS), against an upstream that in 1.0.0
also changed C/Python API namespacing, migrated Apple rules to `MODULE.bazel`, dropped macOS
prebuilts, and bumped Abseil. That is a port of the plugin, not a version bump, and it would leave
UnitEye maintaining a fork of a large native build. **Recommendation: wait for an upstream homuler
release.** Watch [homuler/MediaPipeUnityPlugin releases](https://github.com/homuler/MediaPipeUnityPlugin/releases).

### Is anything lost by staying on 0.10.22?

For UnitEye's use — `FaceLandmarker`, 478 landmarks with iris — no. MediaPipe 1.0.0's headline changes
are in text/LLM tasks (Gecko, TextSummarizer/TextProofreader), audio calculators, WebGPU plumbing and
build/packaging; the face-landmark model and its Task API surface are unchanged. This was checked
empirically on the web path: the same `face_landmarker.task` model run through tasks-vision **1.0.0**
and **0.10.35** returns landmark coordinates that are **bit-identical**. The version gap between the
two paths therefore carries no accuracy difference for face landmarking.

(The two paths have always been independently versioned and already require separate calibration —
different head-area and blink scaling — so this adds no new divergence.)
