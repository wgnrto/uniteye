using System;
using System.Collections.Generic;
using System.Text;
using Unity.InferenceEngine;
using UnityEditor;
using UnityEngine;
using UnitEye;

/// <summary>
/// Logs the input/output layout of the gaze-estimation ONNX models under Resources/ONNX/GazeEstimation
/// so GazeEstimationRunner can be wired to the real I/O (input size + layout, output shape/decoding).
/// Menu: UnitEye ▸ Inspect Gaze Models, or -executeMethod GazeModelInspector.Inspect (headless).
/// </summary>
public static class GazeModelInspector
{
    static readonly string[] Models = {
        "ONNX/GazeEstimation/mobileone_s0_gaze",
        "ONNX/GazeEstimation/mobilenetv2_gaze",
    };

    [MenuItem("UnitEye/Inspect Gaze Models")]
    public static void Inspect()
    {
        foreach (var name in Models)
        {
            try { InspectOne(name); }
            catch (Exception e) { Debug.LogError($"GAZE_MODEL_INSPECT_FAILED {name}: {e}"); }
        }
        Debug.Log("GAZE_MODEL_INSPECT_DONE");
    }

    static void InspectOne(string name)
    {
        var asset = Resources.Load<ModelAsset>(name);
        if (asset == null) { Debug.LogError($"GAZE_MODEL_MISSING {name}"); return; }

        var model = ModelLoader.Load(asset);
        var sb = new StringBuilder();
        sb.AppendLine($"GAZE_MODEL_IO {name}");
        sb.AppendLine($"  inputs={model.inputs.Count} outputs={model.outputs.Count}");
        foreach (var inp in model.inputs)
            sb.AppendLine($"  IN  name={inp.name} shape={inp.shape} dtype={inp.dataType}");
        foreach (var outp in model.outputs)
            sb.AppendLine($"  OUT name={outp.name}");

        // Dummy inference to reveal concrete OUTPUT shapes (decoded angles vs per-bin logits). Build the
        // input from the model's static input shape (these image models have a fixed input size).
        try
        {
            using var worker = new Worker(model, BackendType.CPU);
            var dummies = new List<Tensor>();
            foreach (var inp in model.inputs)
                dummies.Add(new Tensor<float>(inp.shape.ToTensorShape()));

            if (dummies.Count == 1)
            {
                worker.Schedule(dummies[0]);
            }
            else
            {
                for (int i = 0; i < model.inputs.Count; i++)
                    worker.SetInput(model.inputs[i].name, dummies[i]);
                worker.Schedule();
            }

            foreach (var outp in model.outputs)
            {
                var o = worker.PeekOutput(outp.name) as Tensor<float>;
                sb.AppendLine($"  OUT-SHAPE name={outp.name} shape={(o != null ? o.shape.ToString() : "null")}");
            }
            foreach (var t in dummies) t.Dispose();
        }
        catch (Exception e)
        {
            sb.AppendLine($"  (dummy inference failed: {e.Message})");
        }

        Debug.Log(sb.ToString());
    }
}
