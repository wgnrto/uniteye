using System.Collections.Generic;
using System;

namespace UnitEye
{
    public class CSVData
    {
        public float gazePixelX;
        public float gazePixelY;
        public float gazeNormalizedX;
        public float gazeNormalizedY;
        public float gazeNormalizedUnfilteredX;
        public float gazeNormalizedUnfilteredY;
        public float distanceToCamera;
        public float eyeAspectRatio;
        public bool blinking;
        public List<string> AOIList;
        public DateTime timestamp;
        public string notes;
        //Measured pipeline latency of this sample (camera-frame consumption -> logging), in ms. Analysts
        //re-aligning gaze against MOVING stimuli must shift by this: 150ms x a 500px/s object = 75px of
        //systematic error if ignored. 0 for rows created without latency information (e.g. notes).
        public float captureLatencyMs;

        //Constructor to set all values in one line of code
        public CSVData(float gazePixelX, float gazePixelY, float gazeNormalizedX, float gazeNormalizedY, float gazeNormalizedUnfilteredX, float gazeNormalizedUnfilteredY, float distanceToCamera, float eyeAspectRatio, bool blinking, DateTime timestamp, List<string> AOIList, float captureLatencyMs = 0f)
        {
            this.gazePixelX = gazePixelX;
            this.gazePixelY = gazePixelY;
            this.gazeNormalizedX = gazeNormalizedX;
            this.gazeNormalizedY = gazeNormalizedY;
            this.gazeNormalizedUnfilteredX = gazeNormalizedUnfilteredX;
            this.gazeNormalizedUnfilteredY = gazeNormalizedUnfilteredY;
            this.distanceToCamera = distanceToCamera;
            this.eyeAspectRatio = eyeAspectRatio;
            this.blinking = blinking;
            this.timestamp = timestamp;
            this.AOIList = AOIList;
            this.captureLatencyMs = captureLatencyMs;
        }

        //Serialize all the CSVData properties into one semicolon-delimited row (InvariantCulture, sanitized)
        public string SerializeCSVData()
        {
            var inv = System.Globalization.CultureInfo.InvariantCulture;
            var sb = new System.Text.StringBuilder(192);

            int px = float.IsNaN(gazePixelX) || float.IsInfinity(gazePixelX) ? 0 : UnityEngine.Mathf.RoundToInt(gazePixelX);
            int py = float.IsNaN(gazePixelY) || float.IsInfinity(gazePixelY) ? 0 : UnityEngine.Mathf.RoundToInt(gazePixelY);
            float nx = FiniteOrZero(gazeNormalizedX);
            float ny = FiniteOrZero(gazeNormalizedY);
            float unx = FiniteOrZero(gazeNormalizedUnfilteredX);
            float uny = FiniteOrZero(gazeNormalizedUnfilteredY);
            float dist = FiniteOrZero(distanceToCamera);
            float ear = FiniteOrZero(eyeAspectRatio);
            float lat = FiniteOrZero(captureLatencyMs);

            sb.Append(px.ToString(inv)).Append(';')
              .Append(py.ToString(inv)).Append(';')
              .Append(nx.ToString(inv)).Append(';')
              .Append(ny.ToString(inv)).Append(';')
              .Append(unx.ToString(inv)).Append(';')
              .Append(uny.ToString(inv)).Append(';')
              .Append(dist.ToString("F1", inv)).Append(';')
              .Append(ear.ToString(inv)).Append(';')
              .Append(blinking ? "True" : "False").Append(';')
              .Append(timestamp.ToString("yyyy-MM-dd HH:mm:ss", inv)).Append(':').Append(timestamp.Millisecond.ToString("D3", inv)).Append(';')
              .Append(((DateTimeOffset)timestamp).ToUnixTimeMilliseconds().ToString(inv)).Append(';')
              .Append(lat.ToString("F0", inv)).Append(';');

            if (AOIList != null && AOIList.Count > 0)
            {
                for (int i = 0; i < AOIList.Count; i++)
                {
                    if (i > 0) sb.Append(" | ");
                    AppendSanitized(sb, AOIList[i]);
                }
            }

            sb.Append(';');
            AppendSanitized(sb, notes);

            return sb.ToString();
        }

        private static float FiniteOrZero(float v) => float.IsNaN(v) || float.IsInfinity(v) ? 0f : v;

        private static void AppendSanitized(System.Text.StringBuilder sb, string text)
        {
            if (string.IsNullOrEmpty(text)) return;
            for (int i = 0; i < text.Length; i++)
            {
                char c = text[i];
                if (c == ';') sb.Append(',');
                else if (c == '\r' || c == '\n') sb.Append(' ');
                else sb.Append(c);
            }
        }
    }
}