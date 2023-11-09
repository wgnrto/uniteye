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

        //Constructor to set all values in one line of code
        public CSVData(float gazePixelX, float gazePixelY, float gazeNormalizedX, float gazeNormalizedY, float gazeNormalizedUnfilteredX, float gazeNormalizedUnfilteredY, float distanceToCamera, float eyeAspectRatio, bool blinking, DateTime timestamp, List<string> AOIList)
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
        }

        //Seralize all the CSVData properties into one string
        public string SerializeCSVData()
        {
            System.Text.StringBuilder bobTheStringBuilder = new System.Text.StringBuilder();

            //Gaze location both in pixels and normalized
            bobTheStringBuilder.Append($"{(int)gazePixelX};{(int)gazePixelY};{gazeNormalizedX};{gazeNormalizedY}");

            //Unfiltered normalized gaze location 
            bobTheStringBuilder.Append($";{gazeNormalizedUnfilteredX};{gazeNormalizedUnfilteredY}");

            //Distance to camera
            bobTheStringBuilder.Append($";{(float.IsPositiveInfinity(distanceToCamera) ? 0f : distanceToCamera):F1}");

            //Eye Aspect Ratio and blinking
            bobTheStringBuilder.Append($";{(float.IsNaN(eyeAspectRatio) ? 0f : eyeAspectRatio)};{blinking}");

            //Add timestamp with milliseconds at the end
            bobTheStringBuilder.Append($";{timestamp}:{timestamp.Millisecond}");

            //Add unix timestamp in milliseconds
            bobTheStringBuilder.Append($";{((DateTimeOffset)timestamp).ToUnixTimeMilliseconds()}");

            //Replace decimal separator from comma to period (for locales with comma)
            bobTheStringBuilder.Replace(",", ".");

            //Write AOIList if not empty
            if (AOIList.Count > 0)
            {
                //Might need changing depending on the .ToString() behaviour of the AOIList
                bobTheStringBuilder.Append($";{string.Join(" | ", AOIList)}");
            }
            else
            {
                //If empty write empty value
                bobTheStringBuilder.Append($";");
            }

            //Write notes
            bobTheStringBuilder.Append($";{notes}");

            return bobTheStringBuilder.ToString();
        }
    }
}