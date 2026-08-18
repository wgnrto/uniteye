# Gaze backbones (EyeMU vs. a direction-based model)

UnitEye's native provider runs a swappable **gaze backbone** behind the shared MediaPipe FaceMesh +
blink/drowsy/distance stack. Pick it on the `HomulerGaze` component → **Gaze Backbone (model)**, or in
code via the `GazeBackbone` enum. Everything downstream (calibration, filtering, AOI, CSV) consumes the
backbone's `RawGaze` + `Features` unchanged, so switching backbones is transparent to your game — **but
each backbone has its own feature vector, so you must recalibrate after switching.**

Seam: [`IGazeBackbone`](../uniteye/Scripts/Runtime/GazeProvider/IGazeBackbone.cs) — `PerformInference` →
`RawGaze` (pixels) + `Features` (for calibration) + debug textures.

## EyeMU (default, verified)

[`HomulerEyeMURunner`](../uniteye/Scripts/Runtime/Utility/HomulerEyeMURunner.cs). Eye crops + eye corners
+ head geometry → a **screen point** directly (19-feature vector: embedding 4, gaze polynomial 7, head
pose 4, iris offsets 4). Ships with the package; this is the tested path.

## GazeMobileOne / GazeMobileNetV2 (yakhyo/gaze-estimation — integrated; needs a webcam accuracy check)

[`GazeEstimationRunner`](../uniteye/Scripts/Runtime/GazeProvider/GazeEstimationRunner.cs) runs a
direction-based model from [yakhyo/gaze-estimation](https://github.com/yakhyo/gaze-estimation) (the same
weights uniface serves as "MobileGaze"). Three models ship in
`Resources/ONNX/GazeEstimation/` — `mobileone_s0_gaze.onnx` (fastest), `mobilenetv2_gaze.onnx`, and
`resnet34_gaze.onnx` (largest / most accurate; uniface's default) — all **already wired and selectable**:
pick **GazeMobileOne**, **GazeMobileNetV2**, or **GazeResNet34** on `HomulerGaze → Gaze Backbone (model)`
before play, **or switch at runtime** with
the **Model:** button in the Gaze UI (Webcam & Model controls) / `HomulerGaze.SetBackbone(...)`. Switching
rebuilds only the model (the shared face-mesh/blink/distance stack stays); calibration is per-backbone, so
gaze falls back to raw until you recalibrate. With a direction model the debug crop is a **face** crop
(one thumbnail), so the Gaze UI shows a **FaceCrop** toggle instead of Eyecrops.

The I/O was introspected from the actual ONNX (menu **UnitEye ▸ Inspect Gaze Models**), so the runner is
coded to the real contract, not a guess — both models are identical:

| | |
|---|---|
| input | `input` — `(1, 3, 448, 448)` NCHW, RGB, **ImageNet-normalized** |
| outputs | `yaw`, `pitch` — each `(1, 90)` per-bin logits |
| decode | softmax over bins → expectation → `index*4° − 180°` → radians (L2CS/Gaze360) |

Per frame the runner: GPU-crops the face from the FaceMesh rect → ImageNet-normalizes via
`PreprocessGazeEstimation.compute` into the `(1,3,448,448)` tensor → runs the model → decodes both heads
→ maps to a rough on-screen `RawGaze` and emits a **polynomial** calibration feature vector for the
calibrator to refine (the rough raw mapping needn't be accurate).

### Why the features are a polynomial (fixing "crosshair stuck in the centre, corners bad")

A gaze *direction* maps to a flat screen non-linearly — `screen_x ≈ distance · tan(yaw + headYaw)`, with a
genuine yaw–pitch coupling toward the corners. UnitEye's calibrators are **per-axis linear** (RidgeRegression
fits X and Y separately; even the MLP trains best on a good basis), so feeding them the *raw* angles let them
fit only the centre slope and **compress the corners inward** — the classic regression-to-the-centre symptom.
The direction backbones therefore emit a low-order polynomial basis instead of the raw angles:

`Features = [yaw, pitch, yaw², pitch², yaw·pitch, yaw³, pitch³, headYaw, headPitch, headRoll, headArea]`
([`GazeEstimationRunner.FillGazeFeatures`](../uniteye/Scripts/Runtime/GazeProvider/GazeEstimationRunner.cs))

…followed by the iris block, the shared context block, and — since the softmax-breadth work below — a
four-term **σ block**:

`… + [σ_yaw, σ_pitch, σ_yaw·yaw, σ_pitch·pitch]`
([`FillSigmaFeatures`](../uniteye/Scripts/Runtime/GazeProvider/GazeEstimationRunner.cs))

**36 engineered terms in total** (+ the 64-d projected embedding).

The `yaw²/pitch²` + `yaw·pitch` terms are the standard 2nd-order eye-tracking calibration polynomial (they
give the off-centre asymmetry and the diagonal coupling); the `yaw³/pitch³` terms are a pole-free stand-in for
`tan` (`tan x ≈ x + x³/3`) that extends reach at the corners. The two old constant `screenW/screenH` features
were dropped (they standardize to zero — no signal). Nothing in the calibration math changed: RidgeRegression
sizes to the vector length and the MLP infers its input count, and both capture and inference read this same
vector, so train and predict always agree. **This changes the direction backbones' feature length, so
recalibrate them once after updating.**

Sampling was rebalanced to match: the calibration now leads with a **CornerPreset that dwells** (~2 s) at the
four corners + four edge midpoints so the extremes get sustained-fixation samples (previously two of three
presets never left a central box and no point dwelled, so the fit had almost no corner leverage), and the
capture no longer drops downward-gaze frames as false "blinks".

**Verified headlessly** (smoke suite): both models import with exactly that I/O and execute on CPU, and
the decode math is correct.

**Verified against the reference implementation (uniface v4.0.0, Aug 2026):** these releases live on as
**"MobileGaze"** inside [yakhyo/uniface](https://github.com/yakhyo/uniface) — the author's successor
library that consolidates the earlier repos and downloads the *same* ONNX files from the gaze-estimation
releases. Its `uniface/gaze/models.py` pins the exact conventions this runner uses: RGB, ImageNet
mean/std, 90 bins, softmax expectation `* 4° − 180°` → radians, pitch/yaw. So the preprocessing and
decode are no longer guesses. Their demo passes the **raw rectangular detector bbox** (no padding or
squaring — the resize stretch is tolerated); our squared `FACE_CROP_SCALE`-padded landmark bbox emulates
that detector framing without the stretch, and remains the one webcam-tuning knob alongside crop
orientation.

> **uniface v4.0.0 changes nothing here.** The `v3.7.1...v4.0.0` diff touches **no gaze file at all** —
> the major bump is Python API ergonomics (factory functions removed, keyword-only constructors, stricter
> input validation) plus new *detection* models (RetinaFace-R50, BlazeFace, CenterFace), a FaceMesh
> landmarker and FaceAttribNet, all of which duplicate what MediaPipe FaceLandmarker already provides in
> this pipeline. There is **no dependency to upgrade** — uniface is Python and UnitEye consumes only the
> ONNX weights. Two things it does confirm: the full-90-bin decode this runner reverted to in `8c66d89`
> is still exactly what the reference does, and the ResNet-18/50 weight URLs are now pinned in
> `uniface/constants.py` (see below).
>
> One divergence worth knowing: uniface derives the input side from the ONNX
> (`input_size = input_shape[2:4][::-1]`) where `GazeEstimationRunner` hardcodes `INPUT_SIZE = 448`.
> Fine for the three shipped exports, a silent mis-preprocess for any future export at another
> resolution.

uniface also lists **ResNet-18/34/50** gaze variants (larger, more accurate; ResNet-34 is uniface's
default) served from the same releases page — `resnet18_gaze.onnx`, `resnet34_gaze.onnx` and
`resnet50_gaze.onnx` under `.../gaze-estimation/releases/download/weights/`. They share this exact I/O
contract, so adding one to UnitEye is: drop the `.onnx` into `Resources/ONNX/GazeEstimation/`, add a
`GazeBackbone` enum entry, and map it in `NativeGazeProvider.CreateBackbone` — **plus the `embedding`
graph edit**, which the stock releases do not have. The shipped three were edited to expose their
pre-logit GAP vector as a third output for embedding-head personalization, and `UnitEyeSmokeTests`
asserts all three outputs, so an unedited drop-in fails the suite and silently loses the personalization
features. Its other modules (RetinaFace detection, PIPNet landmarks, head pose) duplicate what
MediaPipe FaceLandmarker already provides here, and the Python library itself is not consumable from
Unity.

### Softmax breadth as a per-frame confidence (σ)

The decode collapses 90 logits per axis into one angle and used to throw the rest away. It now also returns
the **standard deviation of the softmax distribution**, per axis, in radians — free, since the expectation
loop already has the probabilities
([`DecodeAngleRadians(float[], out float)`](../uniteye/Scripts/Runtime/GazeProvider/GazeEstimationRunner.cs)).
A Gaussian head of σ=3 bins reports exactly 12.00° (3 × the 4° bin width); a one-hot spike reports 0; the
bimodal shape that made the windowed decode hop reports ~105°.

That number does two jobs.

**1. It is the per-frame quality signal the pipeline never had.** Everything downstream ran on *static*
uncertainty: the AOI error ellipse is measured once at evaluation time, the drift corrector gates only on
residual outliers, and calibration weighted every sample equally. (`BinocularIrisDisagreement`, the one
existing per-frame proxy, is still only written to a CSV column — nothing consumes it.) The raw σ is not
comparable across models or people, so [`GazeConfidence`](../uniteye/Scripts/Runtime/Utility/GazeConfidence.cs)
scores each frame against **this session's own running baseline**, which needs no tuned per-backbone
constant. Consumers:

| Where | What it does |
|---|---|
| AOI probabilities (`HomulerGaze`) | inflates the error ellipse per frame (variances × the squared spread ratio, capped at 9×) so `P(AOI\|fixation)` widens when the estimate is genuinely shakier |
| Drift anchors (`AnchorWeight`) | scales click / pursuit / attention anchor weights — the residual gate cannot see a badly-observed frame that landed plausibly anyway |
| Calibration capture | drops frames at ~4× the session's typical spread, with the same per-dwell bypass the fixation gate uses |

Deliberate limitation, stated so nobody reads more into it: because the baseline adapts, a **uniformly** bad
setup converges back to "confident". This catches transient degradation, not a globally poor configuration —
absolute quality is what the calibration holdout and the evaluation error model measure.

Backbones with no output distribution (EyeMU, and the whole WebGL path) report `NaN`, which every consumer
treats as "no opinion". Nothing about those paths changes.

**2. It parameterizes the decode's compression.** The full-range soft-argmax is systematically compressed —
a true +80° reads +43.7° on a broad head. That alone would be harmless, since a constant gain error is
exactly what calibration absorbs. The catch is that the compression factor is a function of the *breadth*,
and the breadth moves frame to frame with crop framing, head pose and lighting; a fixed polynomial in
(yaw, pitch) cannot represent a gain that changes underneath it. Given σ it can, because σ·angle is the
first-order term of that varying gain — the same move as the gaze×head-pose cross terms, applied to the
decode instead of the geometry. Hence the σ block in the feature vector above.

### Flip TTA: measured offline, and the catastrophe risk is ruled out

Flip TTA computes `(yaw - yawMirrored) * 0.5`. That is worth reading as algebra rather than as "an
average". Writing the two readings as

`yaw(x) = b + a` and `yaw(mirror(x)) = b − a`

the expression returns exactly **a**, the mirror-**antisymmetric** component, and cancels **b**, the
model's mirror-symmetric bias. Flip TTA is a bias-cancelling estimator. The one way it can fail
catastrophically is `a ≈ 0` for every input — a yaw head that ignores the mirror — because then the
estimate is identically zero and gaze pins to the screen centre, which reads as a bad calibration rather
than a bug. That is what kept the toggle default-off, and it is now **ruled out by measurement** against
the shipped weights:

| model | antisymmetric `a` | bias `b` cancelled | pitch swing | verdict |
|---|---|---|---|---|
| `mobilenetv2_gaze` | **60.07°** | −32.33° | 22.57° | SAFE-TO-ENABLE |
| `mobileone_s0_gaze` | **34.14°** | −12.18° | 8.68° | SAFE-TO-ENABLE |
| `resnet34_gaze` | **21.16°** | −35.58° | 3.75° | SAFE-TO-ENABLE |

(`TestFlipAugmentationSignConvention`, threshold 2°. Cross-checked independently through onnxruntime
against the same `.onnx` files, outside Unity.)

Note the size of `b`: these models carry a **large mirror-symmetric bias** on off-distribution input, and
flip TTA removes it. That is a plausible mechanism for the 3–8 % the literature reports, beyond simple
variance reduction.

Pitch is mirror-*invariant*, so flip TTA averages it — the right operation for a symmetric component
whatever its size. The swing is reported, not asserted on.

**Still open, and the reason the default stays off:** none of this shows flip TTA *improves* accuracy on
real faces — that needs a webcam — and it costs **2× inference per frame** (sync mode only; the async
pipeline holds one in-flight inference). Safe to turn on, not yet shown to be worth it.

#### A wrong version of this test shipped first — worth keeping written down

The original check scored `ratio = |yaw(x) + yaw(mirror(x))| / (|yaw(x)| + |yaw(mirror(x))|)`, on the
reasoning that ~0 meant the sign convention held and ~1 meant it was inverted. Substituting the
decomposition above, that ratio is `2|b| / (|b+a| + |b−a|)`, which is **exactly 1 whenever `|b| ≥ |a|`,
regardless of the convention**. It measures bias dominance, not sign. Run against the shipped exports it
returned 1.000 for all three — purely because of `b` — and would have condemned a working configuration.

The stimulus was wrong too. A random field is statistically mirror-symmetric, so the models respond to it
and its mirror almost identically (L1 between the two output distributions ≈ 0.03–0.24 out of a possible
2.0, i.e. near-blind to the flip) and `a` is ~0 for want of signal, not for want of equivariance. The test
now uses two deliberately maximally mirror-asymmetric stimuli, a horizontal ramp and a hard bright/dark
split.

**Not yet verified: runtime gaze accuracy** — the face-crop framing/orientation on a live webcam.

### Hand-test

1. Select **GazeMobileOne** (fastest) on `HomulerGaze` and enter play mode.
2. **Show Eyecrops** → the thumbnail should show your **face**, upright.
3. Move your gaze around and confirm the raw dot tracks the right direction (with `Calibration → None`).
4. **Calibrate** (Ridge or ML) — accuracy comes from calibration, not the raw output. Each backbone has
   its own feature vector, so **recalibrate after switching**.
5. If the uncalibrated dot moves the wrong way or barely/too much: the knobs in `GazeEstimationRunner`
   are `BIN_WIDTH_DEG` / `ANGLE_OFFSET_DEG` (if these weights use a different binning than Gaze360's
   4°/180°) and `ANGLE_TO_SCREEN_GAIN` (pre-calibration travel only). If the face crop looks wrong,
   check the FaceMesh face rect / the Blit UV in `PerformInference`.

### Trade-off

Direction-based models can generalize better across people/distances, but they need the face-crop +
angle→screen step above, and their own calibration. EyeMU's direct screen-point regression is why it
drops straight into this pipeline. If accuracy (not model novelty) is the goal, better calibration data
with EyeMU is the cheaper lever.
