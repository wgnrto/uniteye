# Third-party notices

UnitEye redistributes the components below. Their licenses are listed here as required by those licenses;
this file does not replace them. UnitEye itself is licensed under [GPL-3.0](LICENSE) — see
[Why UnitEye is GPL](#why-uniteye-is-gpl).

Nothing here is legal advice. If you plan to redistribute UnitEye, especially commercially, check each entry
against your own use and take advice — several of these are more restrictive than they first look.

---

## Gaze models

### EyeMU — **GPL-2.0**
- <https://github.com/FIGLAB/EyeMU>, CMU Future Interfaces Group
- Ships as: `uniteye/Resources/ONNX/EyeMUEmbedding.onnx`, `webgl/models/eyemu.onnx`
- These are format conversions of FIGLAB's trained weights, i.e. derivative works of a GPL-2.0 work.
- **This is why UnitEye as a whole is GPL.** The upstream grant appears to be GPL-2.0 *only* — no
  "or any later version" notice — which makes UnitEye's own use of GPL-3.0 worth confirming with the authors.

### yakhyo/gaze-estimation — MIT **code**, non-commercial **weights**
- <https://github.com/yakhyo/gaze-estimation> — the repository is MIT.
- Ships as: `uniteye/Resources/ONNX/GazeEstimation/mobileone_s0_gaze.onnx`, `mobilenetv2_gaze.onnx`,
  `resnet34_gaze.onnx`
- **The MIT license covers the source code, not the provenance of the weights.** That repository states:
  *"All models are trained only on Gaze360 dataset."* The
  [Gaze360](https://github.com/erkil1452/gaze360) dataset states: *"The usage of the dataset and the code is
  for non-commercial research use only."*
- Treat these weights as **non-commercial research use only**, regardless of the MIT badge on the code.
  Anyone shipping UnitEye commercially needs their own permission from the Gaze360 authors, or different
  weights.

---

## Computer vision

### MediaPipe Unity Plugin (homuler) — MIT
- <https://github.com/homuler/MediaPipeUnityPlugin>
- Vendored in full at `com.github.homuler.mediapipe/`, including native binaries
  (`mediapipe_c.dll`, `libmediapipe_c.so`, `libmediapipe_c.dylib`, `mediapipe_android.aar`).
- That package carries its own [Third Party Notices](com.github.homuler.mediapipe/Third%20Party%20Notices.md)
  covering its 29 upstream dependencies; it is part of this distribution and must travel with it.

### MediaPipe models (Google) — Apache-2.0
- Ships as `com.github.homuler.mediapipe/PackageResources/MediaPipe/*.bytes`, of which UnitEye loads
  `face_landmarker_v2_with_blendshapes.bytes`.

---

## Libraries

### Math.NET Numerics — MIT
- <https://github.com/mathnet/mathnet-numerics>
- Ships as `uniteye/Plugins/MathNet.Numerics.dll`

### Google.Protobuf — BSD-3-Clause
- <https://github.com/protocolbuffers/protobuf>
- Ships as `com.github.homuler.mediapipe/Runtime/Plugins/Protobuf/Google.Protobuf.dll`

### System.Runtime.CompilerServices.Unsafe — MIT
- <https://github.com/dotnet/runtime>
- Ships as `com.github.homuler.mediapipe/Runtime/Plugins/Protobuf/System.Runtime.CompilerServices.Unsafe.dll`

### Unity Inference Engine (`com.unity.ai.inference`)
- [Unity Companion License](https://unity.com/legal/licenses/unity-companion-license). A package dependency,
  resolved by the customer's Unity — not redistributed here.

---

## Source adapted into UnitEye

### OneEuroFilterUnity — MIT, Copyright (c) 2017 DarioMazzanti
- <https://github.com/DarioMazzanti/OneEuroFilterUnity>
- Adapted into `uniteye/Scripts/Runtime/Smoothing/OneEuroFilter.cs` and `LowPassFilter.cs`, which carry the
  full MIT notice inline as that license requires.
- Underlying algorithm: Casiez, Roussel and Vogel, *"1 euro filter: a simple speed-based low-pass filter for
  noisy input in interactive systems"*, CHI 2012.

### Point-in-polygon helper — Stack Overflow, CC BY-SA
- Adapted in `uniteye/Scripts/Runtime/AOI/AOI.cs` (see the `Source:` comment at the method).
- Stack Overflow contributions are licensed CC BY-SA. The share-alike obligation is compatible with UnitEye's
  GPL-3.0 licensing, but **would need re-implementation before any permissively licensed or
  proprietary-EULA redistribution.** It is a few lines of standard ray-casting and is straightforward to
  rewrite from the algorithm rather than the post.

---

## Why UnitEye is GPL

The EyeMU weights above are GPL-2.0, and UnitEye loads them in-process against their specific tensor names.
That makes UnitEye a derivative work, so it is distributed under the GPL.

Practical consequence: **UnitEye cannot currently be redistributed under a restrictive EULA** — including the
Unity Asset Store's, which forbids buyers from modifying or redistributing the asset, exactly the rights the
GPL grants. Removing EyeMU does not by itself resolve this, because the remaining gaze weights carry the
Gaze360 non-commercial restriction. A commercially redistributable build needs gaze weights whose license
permits it: a commercial grant from the respective authors, or weights trained on a dataset that allows it.
