# Accuracy Roadmap — how to make UnitEye a substantially better eye tracker

> **STATUS (August 2026): implemented.** Everything below that is implementable client-side has been
> built — see the checklist at the bottom (§6) for exactly what shipped, what is opt-in, and what was
> deliberately deferred (with reasons). **All existing calibrations are stale (the feature vector
> changed twice over): RECALIBRATE every backbone, then run an EVALUATION — the evaluation now
> measures and persists the per-region error model that powers runtime bias correction and
> probabilistic AOI logging.**

*Compiled August 2026 from a multi-angle research sweep (SOTA models 2021–2026, personalization
literature, implicit-calibration literature, geometric methods, commercial-tracker teardowns,
temporal/AOI post-processing, and a line-level audit of this codebase), adversarially cross-checked.
Sources are linked per item; numbers are from the cited papers, not guesses.*

---

## 1. Where UnitEye actually stands

Measured on Mark's setup (calibrated EyeMU + ridge, 1920×876 window): **RMSE ~4 cm X / 2.7 cm Y**
all-screen, corners ~4 cm. At a typical 55–65 cm viewing distance that is **~2.5–3.5°**.

Honest, like-for-like tiers (angular, fixation-level, independent measurements — not vendor claims):

| Tier | Systems | Accuracy | Notes |
|---|---|---|---|
| Floor | WebGazer | ~4° (4.06 cm; 207 px online) | ridge over raw eye pixels, no head model; degrades 49 %/20 min |
| Commercial mid-pack | RealEye, iMotions WebET, Sticky by Tobii, GazeRecorder | **2–3°** | RealEye's "~113 px" claim is a median, fixation-filtered, closest-fixation metric; their own honest metric is 149 px ≈ 3.9 cm. GazeRecorder claims 1.05°, independently measured 2.65° (4.1° with head motion) |
| Verified best (webcam) | Labvanced | **1.45°** (vs EyeLink 1000, N=19) | ~5-min, ~120-target, 7-head-pose calibration that fine-tunes the network per user + drift check every 6–7 trials |
| Phone (close range) | Google NatComm 2020 | 0.6–1.0° (0.46 cm) | ~1 cm only because the screen is 25–40 cm away; angularly = Labvanced tier |

**UnitEye is one tier below the commercial mid-pack — and already at/near the state of practice for
its architecture class** (frozen generic backbone + feature regression; the EMC-Gaze 2026 paper shows
landmark-feature pipelines bottom out at ~2.9° head-still / 5.8° free-head no matter how clever the
features get). No webcam system has independently validated below ~1.3°. **A realistic end-state is
~2–2.5 cm point accuracy, held for the whole session — and effectively better than that at the AOI/
fixation level, which is what the product (AOI logging in a game) actually needs.** The industry
leader for exactly that product shape (Lumen, 650k+ sessions) sells *binary AOI-hit detection at
70 %* — a calibrated 2–2.5° tracker with fixation-level AOI logic comfortably beats that.

### Why it underperforms today (causal order, from the code audit)

1. **The signal is sub-pixel.** At 60 cm, 1° of gaze rotation moves the iris ~0.21 mm ≈ 0.3 px at
   720p-class capture. MediaPipe iris jitter of ~0.5–1 px is therefore worth 2–4° — which *is* the
   measured error band. The pipeline is **landmark-jitter limited, not model-IQ limited**. Native
   capture is already 1920 (prefab), but **the WebGL path captures at 640×480** (`uniteye-cv.js`) —
   3× less px/° than native. Nothing controls autofocus/exposure/fps on either platform.
2. **Off-domain backbones.** EyeMU is phone-trained (portrait, 25–35 cm). The yakhyo L2CS models are
   trained *only on Gaze360* (11–13° MAE within Gaze360 → ~7–9° raw on desktop frontal faces). The
   per-user ridge is currently doing domain adaptation *and* person adaptation at once — that's why
   corners and head motion break it.
3. **The calibration only sees 2 decoded angles** (plus engineered features). The backbone's
   penultimate embedding (512-d ResNet34 / 1280-d MobileNetV2) — where the person-specific appearance
   information lives — is thrown away. EyeMU's 4-d embedding is too small to carry it.
4. **Discarded MediaPipe outputs.** `FaceMeshSolution` never requests `outputFacialTransformationMatrixes`
   (metric head rotation + **translation** — no current feature carries lateral head shift) or
   `outputFaceBlendshapes` (8 eyeLook* coefficients = a second, independent gaze estimate; eyeBlink
   scores that would replace the EAR heuristic that false-positives on downward gaze). Head pose is
   an atan-of-relative-Z hack with a magic `/2` on roll.
5. **Drift is unmanaged.** Published webcam baselines lose ~50 % accuracy per 20-min session without
   correction; UnitEye's only tool is a manual, translation-only re-center (can't fix gain drift —
   uncalibrated models show 0.3–0.7× per-axis *scale* errors, not just offsets).
6. **The product metric is computed wrong.** AOI hits are per-frame at whatever the pipeline lag is.
   No capture timestamps: 100–300 ms lag × a moving object's speed = 30–150 px of systematic AOI
   error — comparable to or larger than the entire calibration error. Fixation-level aggregation
   (what every commercial tool quotes) is not used at runtime, though `IsFixationStable` (I-DT)
   already exists in the calibration code.
7. Smaller structural caps: axis-aligned crops (head roll rotates the eye inside the crop; also
   `GetEyeCropRect` mixes width/height-normalized units in `yShift`), TPS warp has only 8 anchors all
   on the screen boundary (interior correction is affine-only), softmax-expectation decode over all
   90 bins pulls estimates toward center (corner compression), no gaze×headpose / gaze×distance
   interaction terms (the head-rotation calibration stage collects variance the linear-in-pose model
   family cannot use).

---

## 2. The roadmap

Phases are ordered by evidence-adjusted return per effort. Every item names the seam it lands in.
Native + WebGL parity noted per item. "Gain" figures cite the source literature; treat them as
relative-improvement evidence, not promises.

### Phase 0 — Measure first (days; gates everything else)

| # | What | Where | Why first |
|---|---|---|---|
| 0.1 | **Accuracy/precision decomposition** in the evaluation: per-region mean offset (bias), within-fixation SD + BCEA (precision), and the RMS-S2S/STD ratio (white vs colored noise) | `HomulerGazeEvaluation` | Decides everything: if the 4 cm is mostly *bias*, filtering/averaging is capped at a few %; if *jitter*, resolution + aggregation pay hugely. One hour of work that re-ranks the whole roadmap |
| 0.2 | **Capture-timestamp + latency discipline**: stamp gaze samples with camera-frame capture time, measure end-to-end lag per session, hit-test AOIs against their transforms *at capture time* (ring buffer) | `IGazeProvider` (+`CaptureTimestamp`), `AOIManager`, `CSVLogger`; WebGL via `requestVideoFrameCallback` mediaTime | Removes a systematic (lag × object-speed) error of 30–150 px on moving AOIs — likely *larger than the calibration error*. Also replaces the hard-coded 100 ms pursuit-lag constant with a measured value |
| 0.3 | **AOI-hit evaluation mode**: report AOI-hit precision/recall per AOI size alongside RMSE | evaluation scene | The go/no-go metric for the game is AOI-level, not cm RMSE |
| 0.4 | (Recommended) a **reference measurement** on 3–5 people with a cheap IR tracker (Tobii-class, ~€200) + a frozen benchmark protocol (fixed grid; head-still, head-moving, and 15-min drift runs) | — | Session-to-session variance otherwise swallows the 5–15 % effects half this roadmap claims |

### Phase 1 — Quick wins (each S effort; sum is a realistic 20–40 % + session stability)

**Camera & signal (attacks the actual bottleneck):**
1. **Fix WebGL capture: 640×480 → 1920×1080** + `frameRate` constraint in `getUserMedia`
   (`uniteye-cv.js`). Native is already 1920; the browser path is running at 3× less px/° today.
2. **Lock autofocus/auto-exposure where possible; prefer 60 fps.** AF hunting changes effective
   focal length mid-session (a drift confound no calibration can model); long exposure = motion blur
   that destroys the sub-pixel iris signal; 60 fps = 2× samples per fixation = √2 precision.
   (UVC controls natively; `getUserMedia` advanced constraints on web, where supported.)
3. **Photometric normalization of crops** (per-crop mean/variance or histogram equalization) before
   inference — stabilizes the embedding under uncontrolled home lighting. Lands in the two
   preprocess compute shaders / JS crop path.
4. **Horizontal-flip test-time augmentation**: run the face crop and its mirror, negate yaw, average.
   Standard 3–8 % angular improvement, zero training, 2× inference (fine at 30 Hz native; optional
   web). `GazeEstimationRunner`.

**Unlock the discarded MediaPipe outputs:**
5. **Enable `outputFacialTransformationMatrixes` + `outputFaceBlendshapes`** in `FaceMeshSolution`
   (two ctor args + accessors; same options exist in `@mediapipe/tasks-vision` for web parity). Feed:
   metric head rotation (replaces the atan-Z hack), **head translation tx/ty** (closes the
   lateral-head-shift blindness), the 8 eyeLook* blendshapes as direct gaze features, and eyeBlink
   scores as the blink gate (recovers clean bottom-edge calibration samples). Extend both feature
   fillers + `HeadPoseFeatureIndices`; stale calibrations already fall back safely. Literature: 10–30 %
   RMSE + markedly better head-movement robustness for feature-rich pose-aware regression.
6. **Iris-diameter metric distance** (physiological iris ≈ 11.7 mm; Z = f·11.7 mm/iris_px; Google
   reports 4.3 % mean depth error with a fixed-FOV focal prior). Replaces the "head area" proxy,
   which conflates distance with head size. ~50 lines, shared C#.

**Calibration-layer upgrades:**
7. **Interaction terms in the feature basis**: yaw×headYaw, pitch×headPitch, yaw×tx, pitch×ty,
   yaw×dist, pitch×dist + bbox-center. The physical map is `x ≈ eyePos + D·tan(yaw+κ)` — the current
   basis is polynomial in gaze but *linear* in head pose, so the head-rotation stage collects
   variance the model family can't exploit. Ridge's standardization + CV-λ absorb the extra columns.
8. **Interior TPS anchors**: one new dwell preset (center + 4 points at 25 %/75 %, ~1.5 s each,
   +8–12 s calibration) → 13 anchors instead of 8 boundary-only. The existing holdout gate makes it
   strictly-no-worse. Targets mid-screen, where the game's AOIs actually live.
9. **Windowed (top-k) softmax decode** for the 90-bin heads: expectation over ±5 bins around argmax
   instead of all 90 (the far bins' softmax mass drags estimates toward center — worst at corners).
   ~10 lines in `DecodeAngleRadians` + smoke test.
10. **Calibration validation gate with retry** (RealEye/Labvanced practice): after calibration, a
    short validation sweep; fail → targeted redo. Also produces a **per-session confidence score**
    to tag AOI logs (Tobii's webcam numbers rest on a 63 % session-pass gate — quality gating is how
    the industry reports good numbers *and* it converts unknown-quality data into known-quality data).

**AOI/product layer:**
11. **Fixation-level AOI logging**: run I-DT (reuse `IsFixationStable`) on the calibrated stream; feed
    the fixation centroid to `AOIManager.CheckAOIList`, keep One-Euro only for the visible cursor.
    Precision component shrinks ~√N (6–30 samples/fixation); commercial tools quote exactly this.
    Gain is bounded by the bias/precision split from item 0.1 — measure first.
12. **AOI margins + hysteresis + minimum dwell (100–120 ms)** generalized from the existing
    `GazeGridQuantizer` pattern to all AOI shapes; editor warning when two AOIs sit closer than 2×
    the user's error radius. Published: +1.0° AOI enlargement takes hit-any-AOI from <70 % to >80 %.

### Phase 2 — The two big levers (M each; the evidence-backed route from ~4 cm toward ~2–2.5 cm)

**2A. Embedding-head personalization (the Google NatComm recipe).**
ONNX graph surgery (no retraining) adds a second output to the L2CS models tapping the pre-logit GAP
vector (ResNet34 512-d / MobileNetV2 1280-d) — plus the 90-bin logits, whose entropy is a free
per-frame uncertainty proxy. Extend the existing Huber-IRLS ridge/MLP to consume
[embedding (PCA→~64) + engineered features]. This is "fine-tune the last layer" done as a closed-form
solve — the only personalization the no-backprop constraint allows, and the best-evidenced one:
- Google (Nature Comm 2020): SVR on penultimate features from ~30 s of pursuit → 1.92 → 0.46 cm on
  phones; personalization roughly halved error.
- FAZE's own baseline study: polynomial-on-outputs (≈ UnitEye today) *converges* to meta-learned
  performance at k≥128 samples — but feature-space heads keep improving; UnitEye's pursuit
  calibration collects hundreds of samples, exactly the regime where the embedding head wins
  (Liu TPAMI: feature-space beats output-space only at k≥25).
Realistic: **10–30 % calibrated-RMSE reduction** on the current backbone; compounds with any future
backbone. Works identically native (Inference Engine multi-output) and web (onnxruntime-web).
Risks: d≫n overfitting (PCA + the existing CV-λ + jitter augmentation), lighting sensitivity
(mitigated by Phase 1.3).

**2B. Online drift stack (fast/slow architecture).**
Published webcam baselines lose ~50 %/session; holding calibrated accuracy all session is worth more
delivered centimeters than any backbone change. All pure C#, above the provider seam, native+web:
1. **RLS affine drift layer** (6-param, exponential forgetting ~0.99, per-DOF gating: translation
   fast, scale/shear only with ≥6–8 well-spread anchors, magnitude caps). Sits *after* the frozen
   ridge+TPS — worst case it converges to identity. The re-center button becomes "reset drift state".
   Also upgrade re-center itself from translation-only to a 3-point gain+offset refit (uncalibrated
   models show 0.3–0.7× per-axis scale errors; translation can't fix that).
2. **Click anchors with 4-gate validation** (WebGazer/PACE lineage): sample the *pre-click fixation
   window* (gaze leads clicks by 100–200 ms), require an I-DT fixation, reject residuals > 2–3×
   running MAD (users look at their click only ~⅔ of the time), let Huber IRLS downweight the rest.
   Median gaze-cursor distance at click ≈ 74 px → a validated click is a ~0.7–1° label. The game can
   snap anchors to the clicked AOI's center — cleaner labels than WebGazer ever had.
3. **Correlation-gated pursuit anchors from moving game objects** (Pursuit Calibration, UIST 2013):
   sliding-window Pearson correlation between gaze and each registered object trajectory; r > ~0.75
   on both axes → dense labeled samples along the path (with the existing pursuit-lag shift).
   Route reward-flight animations through corners occasionally = corner anchors several times/min.
   Weight ~0.5× vs clicks. This is what keeps *corners* at calibrated accuracy.
4. **Game-event priors**: the engine knows where the attention-grabbing event is (spawn, explosion,
   dialog). Extend the existing AOI declaration API with "became salient at t"; treat like click
   anchors at ~0.3–0.5× weight. Do **not** run a saliency CNN (see do-not-build).
5. **Session persistence + warm-start revalidation**: persist calibration + drift state locally;
   next session, first 30–60 s of anchors decide: accept / silently refit affine / trigger a 20 s
   booster. Kills the 2–3 min recalibration for returning players — the biggest funnel drop for an
   online game. (PACE's cross-session accumulation reached 2.56° with *no* explicit calibration.)
6. **Rehearsal-buffered slow refit**: every N validated anchors, re-run the existing trainer on
   {explicit samples + weighted anchors}, keep only if explicit-sample holdout doesn't regress
   (generalizes the TPS keep-if-better gate). Rehearsal prevents the classic failure where a flood of
   bottom-UI clicks drags the whole map down-screen.

### Phase 3 — Structural upgrades (L; do after Phases 1–2 prove out)

- **3A. Per-user error model + probabilistic AOI layer.** Fit per-region bias + covariance (BCEA) from
  the validation sweep, persist next to the calibration; replace boolean `CheckAOI` with
  P(AOI | fixation) = ∫ fixation-Gaussian over AOI (closed forms for box/circle; BayesGaze-style
  posterior accumulation). Log top-2 AOIs + probabilities + a "coin-flip" flag instead of
  confidently-wrong binary labels. Between-user accuracy varies >6× (Feit et al.) — this is what
  makes the research data honest, and it converts the residual error from a correctness problem into
  a quantified-uncertainty problem.
- **3B. Roll-normalized crops** (2D data normalization): rotate crops so the inter-corner line is
  horizontal (exact from landmarks, no intrinsics). Fix the `GetEyeCropRect` yShift axis-mixing bug
  while in there. Zhang/Sugano: normalization is worth ~1°-class improvements under pose variation;
  laptop users tilt constantly.
- **3C. Hybrid geometric mapping**: with the transformation matrix (1.5) + iris depth (1.6), build the
  3D gaze-ray → screen-plane intersection with <10 physical unknowns (κ per eye, screen pose/offset)
  solved by a small LM fit on the existing corner+pursuit data; keep ridge/TPS as a residual layer on
  top (what commercial trackers do). Head-pose-invariant *by construction* — corners stop being
  extrapolation, re-center becomes a principled 1-glance κ refresh, and the head-rotation calibration
  stage finally becomes load-bearing (it decorrelates κ from screen pose). Prior art in exactly this
  stack: WebEyeTrack (2025, Apache-2.0) — MediaPipe + iris-scale metric pose + ray intersection in
  the browser. Caveat: with an 11–13° Gaze360 backbone the geometric solve is noise-dominated — do
  it together with 3D/3E.
- **3D. Full data normalization + a normalized-trained backbone** (must ship together — warping input
  to a Gaze360-crop-trained model *hurts*): Zhang/Sugano virtual-camera warp (focal prior is fine;
  intrinsics sensitivity is forgiving) + swap to ETH-XGaze/MPIIFaceGaze-normalized weights.
  Reference implementation: hysts/pytorch_mpiigaze_demo.
- **3E. Dedicated iris pass on the hi-res eye crops**: run MediaPipe Iris (or equivalent) on the
  already-extracted eye crops instead of relying on full-frame FaceLandmarker precision — decouples
  iris precision from capture resolution entirely; the most direct attack on the jitter floor.
- **3F. Binocular consistency**: per-eye rays/predictions where available; disagreement = free
  per-frame uncertainty + blink/occlusion detector; averaging both eyes in screen space was worth
  16 % in EVE. Feeds the uncertainty weighting (2A's logit entropy + 3A's error model).
- **3G. Glasses handling**: detect eyewear (specular highlights on the crop or a tiny classifier),
  maintain separate calibration profiles glasses-on/off, warn on mismatch (glasses are worth
  ~12+ mm in the 2026 open-domains paper; a profile calibrated with glasses silently invalidates
  without them).

### Phase 4 — Research-tier (XL / strategic; only after the above)

- **4A. Desktop-domain backbone in the existing ONNX slot** — *the* backbone move worth making:
  fine-tune MobileNetV2/ResNet34-class weights on ETH-XGaze + GazeCapture + synthetic
  (glasses/lighting augmentation), or distill a strong teacher (UniGaze-B) into the MobileNetV2 slot.
  Fixes the real problem (Gaze360-only domain mismatch), ships inside the current web budget with
  zero runtime changes, license-controllable if trained on synthetic (UnitEye is a Unity shop —
  rendering a desktop-geometry synthetic corpus with domain randomization is in-house work; the
  2026 open-domains paper's 3.8 M-param model beats UniGaze-H at <1 % of the parameters).
  Datasets caveat: ETH-XGaze/GazeCapture/EVE are research-only licenses.
- **4B. UniGaze-B as a native "high-accuracy" backbone experiment** (M): current cross-domain SOTA
  (~4.7–5.9° raw on unseen desktop-style data vs ~7–9° for the shipped models), 224×224 normalized
  input, direct regression, HuggingFace weights. **License is non-commercial** (ModelGo NC-RAI-2.0) —
  fine for research deployments, not for a commercial game. 170 MB fp16; needs 3D's warp anyway.
  The adversarial review rates its *calibrated* gain well under the raw delta — treat as experiment,
  not cornerstone.
- **4C. The data flywheel**: the shipped game generates the dataset the whole field lacks
  (desktop-webcam faces with pursuit-quality labels at scale). A consented, opt-in collection path —
  even features-only — makes 4A self-funding and is the single biggest *long-term* lever. Requires
  real privacy/consent design (current constraint is "nothing leaves the machine"; this would be an
  explicit, separate opt-in).
- **4D. Gamified calibration with active sampling**: wrap the existing collectors in game mechanics
  (charge-a-shot dwells, whack-a-mole grids, chase-the-firefly pursuits), spawn targets where CV
  residuals are highest. Same 2–3 min, more and better-distributed samples; repeat-play studies show
  calibration quality *improves* across sessions. Do whenever the calibration UX is next touched.
- **4E. Pose-binned local calibration mixtures** (Sugano-style): 2–4 head-pose-clustered ridge maps
  with soft assignment — the non-parametric alternative if 1.7's cross terms underfit the true pose
  interaction. Evaluate against 1.7 with the Phase-0 harness.

---

## 3. Do-not-build list (documented so it isn't re-litigated)

| Idea | Verdict | Why |
|---|---|---|
| Bigger frozen backbone as a cure-all | ✗ | Per-user ridge absorbs person-independent bias; post-calibration residual is jitter- and head-motion-dominated. Calibrated deltas between frozen backbones largely wash out |
| Naive per-user CNN fine-tuning / on-device TTA (PnP-GA, TPGaze…) | ✗ | Needs on-device backprop — unavailable in Unity Inference Engine / impractical in ORT-web. FAZE ablation: naive fine-tuning at low k is *worse* than no adaptation. The implementable subset IS 2A |
| Differential/siamese gaze net as primary estimator | ✗ | Few-shot advantage evaporates at k>25 without fine-tuning; UnitEye lives at k in the hundreds. (Concept survives only as a drift corrector — dominated by the S-effort RLS affine layer) |
| Limbus-ellipse eyeball fitting (EyeTab/Swirski) | ✗ | Needs eccentricity of a 30–60 px eye — unmeasurable; MediaPipe gives exactly 5 near-circular iris points/eye. EyeTab managed only 6.88° on a *tablet* |
| Generic saliency-map recalibration | ✗ | EVE benchmark: helps images, *doubles* error on text/UI-like content (+43–84 %). Games are full of UI. Use game-event priors instead (2B.4) — exact and free |
| Naive cursor-following supervision | ✗ | Gaze-cursor alignment only ~66 % of the time; WebGazer's weakest signal. Salvage only correlation-gated drags/aims |
| Temporal GRU/LSTM backbone | ✗ (now) | Measured gain 4–10 % and inconsistent per eye (EVE); Unity Inference Engine doesn't import GRU/Scan/Loop. Cheap 90 % substitute: feed a short temporal window of features to the existing ridge |
| Screen-content-conditioned refiner (GazeRefineNet analogue) | ✗ (now) | Needs own ground-truth dataset + second per-frame CNN; EVE shows 30–84 % *worse* out-of-distribution (text/UI). Degenerate safe version: aggregate-logs-as-prior inside the Bayesian AOI layer |
| AOI snapping/magnetism for the research log | ✗ | Manufactures confident labels from ambiguous data; users adapt and absorb the assistance. Fine for *interaction*; never for the logged stream (log probabilities instead — 3A) |
| Screen-glint / corneal reflection of the monitor | ✗ | Sub-pixel and content-dependent at 50–70 cm; ScreenGlint-class results required phone-range close-ups |
| Multi-camera / depth sensor (Eyeware Beam route) | ✗ | Setup friction kills an online game; the webcam-only Beam product notably ships *no* accuracy claims. The 2D-available version of its idea = iris-scale metric pose (1.6) |
| Full camera-intrinsics calibration (checkerboard etc.) | ✗ | Per-user screen-space calibration absorbs fixed projective distortion; FOV prior + iris refinement suffices (Google: 4.3 % depth error on a fixed focal). Only matters for absolute calibration-free gaze |
| GazeTR / MCGaze / iTracker / AFF-Net adoption | ✗ | L2CS-class accuracy (GazeTR), wrong domain (MCGaze: Gaze360), phone-domain PoG (iTracker/AFF-Net = EyeMU's problem again) |

---

## 4. Recommended execution order (opinionated)

1. **Phase 0** (0.1 decomposition + 0.2 timestamps first — days, re-ranks everything).
2. **Phase 1** items 1, 5, 6, 7, 8, 9, 10 (one calibration-format change, one recalibration), then
   11–12 on the AOI layer. Items 2–4 opportunistically.
3. **2B.1–2B.2** (drift layer + click anchors) — best gain-per-effort in the whole sweep.
4. **2A** (embedding head) — biggest single expected accuracy jump.
5. **2B.3–2B.6**, then **3A** (probabilistic AOI) — this pair turns a ~2.5–3 cm tracker into a
   trustworthy AOI logger.
6. **3B–3G** as a coherent "geometry release".
7. **4A/4C** as the strategic backbone play; 4B as a native experiment when idle; 4D with the next
   calibration UX rework.

**Expectation to hold:** per-frame 1–2 cm from a 30 Hz webcam is beyond what anyone has independently
validated; per-*fixation*, drift-corrected, uncertainty-tagged 1–2 cm-effective AOI decisions are
reachable with the stack above — and that is the metric the product ships.

---

## 5. Key sources

Models: [UniGaze](https://arxiv.org/abs/2502.02307) · [3DGazeNet](https://arxiv.org/abs/2212.02997) ·
[open-domains MobileNetV2 2026](https://arxiv.org/html/2603.26945) · [EFE](https://arxiv.org/abs/2305.05526) ·
[yakhyo/gaze-estimation](https://github.com/yakhyo/gaze-estimation)
Personalization: [Google NatComm 2020](https://www.nature.com/articles/s41467-020-18360-5) ·
[FAZE](https://arxiv.org/abs/1905.01941) · [Liu TPAMI differential](https://arxiv.org/abs/1904.09459) ·
[SPAZE](https://arxiv.org/abs/1807.00664) · [Chen & Shi decomposition](https://arxiv.org/abs/1905.04451)
Implicit calibration: [WebGazer](https://webgazer.cs.brown.edu/) · [PACE](https://ira.lib.polyu.edu.hk/bitstream/10397/64378/1/Huang_Building_Personalized_Auto-Calibrating.pdf) ·
[Pursuit Calibration](https://dl.acm.org/doi/10.1145/2501988.2501998) · [Online-EYE CHI'25](https://dl.acm.org/doi/10.1145/3706598.3713461) ·
[COMETIC](https://dl.acm.org/doi/full/10.1145/3706598.3713936)
Geometry: [Zhang/Sugano normalization](https://dl.acm.org/doi/10.1145/3204493.3204548) ·
[MediaPipe Iris depth](https://research.google/blog/mediapipe-iris-real-time-iris-tracking-depth-estimation/) ·
[WebEyeTrack](https://arxiv.org/abs/2508.19544) · [hysts demo](https://github.com/hysts/pytorch_mpiigaze_demo)
Benchmarks/teardowns: [Labvanced vs EyeLink](https://pmc.ncbi.nlm.nih.gov/articles/PMC11289017/) ·
[RealEye white paper](https://www.realeye.io/whitepaper) · [GazeRecorder independent eval](https://www.frontiersin.org/journals/robotics-and-ai/articles/10.3389/frobt.2024.1369566/full) ·
[iMotions WebET validation](https://imotions.com/blog/learning/product-news/webcam-eye-tracking-validation-study/) ·
[Tobii Sticky white paper](https://connect.tobii.com/s/article/White-Paper-Webcam-eye-tracking-accuracy?language=en_US) ·
[EMC-Gaze ceiling](https://arxiv.org/abs/2603.12388) · [online DL benchmark 2024](https://pmc.ncbi.nlm.nih.gov/articles/PMC11133145/)
Temporal/AOI: [EVE](https://www.ecva.net/papers/eccv_2020/papers_ECCV/papers/123570732.pdf) ·
[Blignaut filter study](https://bop.unibe.ch/JEMR/article/download/JEMR_12.2.3/7837/19309) ·
[I2MC](https://github.com/royhessels/I2MC) · [I-BDT](https://arxiv.org/abs/1511.07732) ·
[BayesGaze](https://pmc.ncbi.nlm.nih.gov/articles/PMC8853835/) · [Feit et al.](https://dl.acm.org/doi/10.1145/3025453.3025599) ·
[Wang et al. AOI uncertainty](https://dl.acm.org/doi/abs/10.1145/3517031.3531166)

---

## 6. Implementation status (August 2026)

Everything client-side implementable was built in one pass. **Recalibrate every backbone, then run an
evaluation** (feature vectors changed; the evaluation now feeds the runtime error model).

### Shipped (on by default)
- **0.1 Decomposition**: evaluation reports accuracy(bias) / precision(SD) / RMS-S2S-whiteness + AOI-hit
  rates @3/5/8 cm (`GazeStatistics`, `HomulerGazeEvaluation.BuildErrorStatistics`).
- **0.2 Timestamps**: `IGazeProvider.CaptureTimestamp` (async-readback-consistent), measured pipeline
  latency exposed (`HomulerGaze.MeasuredLatencySeconds`) + new CSV column "Capture Latency In
  Milliseconds". The calibration's pursuit-lag constant remains (measured per-session value = follow-up).
- **1.1/1.2 Camera**: WebGL `getUserMedia` 640×480 → **1920×1080 + 60 fps ideal**; native
  `WebCamSource` now requests **60 fps** (was driver default). AF/AE lock not reachable from Unity/browser APIs — documented.
- **1.5 MediaPipe unlock**: `outputFacialTransformationMatrixes` + `outputFaceBlendshapes` enabled on
  BOTH platforms. Metric head pose (matrix) replaces the landmark-Z hack (toggle `_useMatrixHeadPose`),
  head **translation** + depth + 8 **eyeLook blendshapes** feed the features, **eyeBlink blendshapes**
  replace the EAR blink gate (EAR fallback kept).
- **1.7 Interaction terms**: shared 17-feature context block [tx, ty, dist, eyeLook×8, gaze×pose,
  gaze×translation, gaze×distance] on every backbone. EyeMU 19→**36** features, direction 15→**32**(+embedding), ensemble 43(+…).
- **1.8 Interior TPS anchors**: new `InteriorPreset` (centre + 4 quadrant dwells, ~8 s) in every calibration.
- **1.9 Windowed decode**: softmax expectation over argmax±5 bins (kills far-bin centre drag).
- **1.10 Validation gate**: holdout-RMSE confidence bands with recalibrate advice on the results screen
  (`HomulerGazeCalibration.LastHoldoutRmseCm` for host-game gating).
- **1.11/1.12 AOI layer**: fixation-centroid AOI stream (runtime I-DT `FixationAggregator`) with the
  cursor keeping One-Euro; per-AOI `margin`; manager-level minimum-dwell (100 ms) + exit hysteresis.
- **2A Embedding head**: the three direction ONNX files now expose the pre-logit GAP vector as an
  `embedding` output (graph edit; MobileOne 1024-d, MobileNetV2 1280-d, ResNet34 512-d) and it is
  concatenated into the calibration features — closed-form last-layer personalization (Google NatComm
  recipe). Ridge's standardization + CV-λ absorb the width; MLP also sees it (holdout-gated).
- **2B Drift stack**: 6-DOF RLS affine `DriftCorrector` (staged DOFs, magnitude caps, MAD outlier gate)
  after the frozen calibration; **click anchors** (pre-click fixation window); **pursuit anchors**
  (`HomulerGaze.FeedPursuitTarget` + Pearson gate); **attention events** (`ReportAttentionEvent`);
  state persists per backbone for a warm start; `ClearDrift`/recalibration resets.
- **3A Error model + probabilistic AOI**: evaluation persists per-region bias+covariance
  (`GazeErrorModel`, `ErrorModel_<backbone>.json`); runtime subtracts the measured regional bias from
  the AOI stream and logs `P(AOI|fixation)` (32-sample deterministic MC per shape) with an AMBIGUOUS
  flag when top-2 are within 0.2.
- **3B Roll normalization**: direction-model face crop samples rotated by −headRoll (toggle
  `_rollNormalizeCrops`, hand-test note in tooltip). EyeMU eye crops deliberately untouched (tight
  crops would clip; its corners input already encodes roll).
- **3F Binocular signal**: `IGazeProvider.BinocularIrisDisagreement` (per-frame quality proxy).
- **3G Glasses**: profile metadata + Gaze-UI toggle + load-mismatch warning.
- **Robustness**: `Mouse.current`/`Keyboard.current` null guards (headless/touch devices crashed per-frame).

### Opt-in (default off — needs one webcam sanity check first)
- **Flip TTA** (`_flipAugmentation`): mirrored-crop average for the direction models. If the mirror
  convention is wrong on a device the average collapses toward centre — verify once, then enable.

### Deferred, with reasons
- **Photometric crop normalization**: changing input statistics under a FROZEN ImageNet-normalized
  backbone is a train/test mismatch; belongs with backbone retraining (4A).
- **`GetEyeCropRect` yShift axis-mixing**: documented in-code instead of changed — it silently reframes
  EyeMU's input (stales all calibrations + needs webcam verify + JS port sync).
- **Full 3D gaze-ray→screen solve (3C)**: gated on a normalized-trained backbone (3D) — with the
  Gaze360-domain models the geometric solve is direction-noise-dominated. The enabling pieces (metric
  pose, iris depth, translation features) are in.
- **3D normalized backbone swap / 3E hi-res iris pass / 4A retraining / 4C data flywheel**: external
  weights/models/datasets/consent programs — not code-only. 4B (UniGaze) additionally **non-commercial license**.
- **2B slow refit (rehearsal-buffered re-train)**: needs persisted explicit-sample stores; the fast
  affine layer covers session drift. Clean follow-up.
- **Pursuit-lag measured per session / I-BDT event detection / pose-binned mixtures (4E)**: follow-ups;
  current I-DT + cross terms cover the bulk.

### Post-review fix round (same day)

A 12-agent adversarial review of the implementation (6 dimension reviewers + per-finding verification)
confirmed 18 unique defects; all were fixed and re-verified:

- **Double bias correction**: the error-model bias is now subtracted UPSTREAM of the drift corrector
  (whole pipeline benefits; anchors observe the corrected signal so the two layers learn disjoint
  residuals); the AOI branch no longer subtracts it a second time. The manual re-center is likewise
  captured/applied POST-corrector.
- **Pursuit timing**: the correlator now takes the gaze CAPTURE time and the object's RENDER time as
  separate clocks (stamping the object with the gaze clock shifted the lag pairing by the full pipeline
  latency), and adds at most one correlation sample/anchor per camera frame (was per render frame).
- **Click anchors**: collected before the fresh-camera-sample gate — clicks land on render frames, most
  of which carry no new camera sample, so the old placement silently dropped most clicks.
- **DriftCorrector hardening**: warm-up ramp (first anchors move the correction gradually instead of one
  click jumping it to the caps) and a latched affine unlock (momentary anchor-spread decay no longer
  snaps learned gains back to identity in one frame). Blink-hold frames keep timestamps fresh.
- **Staleness**: recalibration and profile loads now delete/replace the per-region error model and any
  companion Warp file not in the profile (a warp/error model is only valid for the exact fit it was
  measured on); `SetBackbone` saves the old backbone's drift state and warm-starts the new one's
  (previously it deleted the switched-TO backbone's saved state); the error model records which
  calibration type it measured and is only applied while that type is active.
- **Training freeze**: the raw 512–1280-d embedding is compressed to **64 dims by a fixed sparse-JL
  (±1) projection** before entering the features — CV training stays seconds instead of minutes, and
  the in-code xorshift keeps saved calibrations valid across runtimes. Jagged feature rows (backbone
  swap race) are dropped at capture.
- **AOI state machine**: exit hysteresis only extends AOIs that actually REGISTERED (no phantom hits
  from saccades sweeping through), disabled/removed AOIs drop their dwell state, raycast AOIs
  (AOITagList) are excluded from hysteresis (stale hit lists / NREs), margins no longer shrink inverted
  AOIs, and AMBIGUOUS requires a real leading candidate (p ≥ 0.2).
- **Web**: the facial transformation matrix is read COLUMN-major (`data[12..14]`; the row-major read
  produced all-zero translation features), `RidgeModel.predict` gained the native NaN dimension guard
  (stale pre-36-feature calibrations fall back to raw gaze instead of silent garbage), and out-of-frame
  eye crops are rejected like native (drawImage silently clips and left stale pixels in the canvas).
