using System;
using System.Collections;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using UnityEngine;
using UnityEngine.Assertions;
using UnityEngine.Events;

namespace UnitEye
{
    public class CSVLogger : MonoBehaviour
    {
        [SerializeField] private string _baseFolderPath;
        [SerializeField] private string _baseFileName;

        [SerializeField] public bool useDefaultFolder = true;

        //Fire PathChange Event on set
        public string BaseFolderPath
        {
            get => _baseFolderPath;
            set
            {
                if (string.Equals(value, _baseFolderPath, StringComparison.Ordinal)) return;

                _onPathChangeEvent?.Invoke();
                _baseFolderPath = value;
            }
        }
        //Fire PathChange Event on set
        public string BaseFileName
        {
            get => _baseFileName;
            set
            {
                if (string.Equals(value, _baseFileName, StringComparison.Ordinal)) return;

                _onPathChangeEvent?.Invoke();
                _baseFileName = value;
            }
        }
        [SerializeField] float timeUntilWrite = 10f;
        [SerializeField] float logsPerSecond = 999f;

        //Timestamp of the last accepted row; negative infinity so the very first Append always logs.
        private float _lastLogTime = float.NegativeInfinity;

        //Unity Event to handle path changes (initialized eagerly so SetFilePath before Start never NREs)
        private UnityEvent _onPathChangeEvent = new UnityEvent();

        //Private queue
        private readonly List<CSVData> _queue = new List<CSVData>();

        //Header row to name columns in csv file
        private static readonly string _Header = "X Filtered Pixel; Y Filtered Pixel; X Filtered Normalized; Y Filtered Normalized; X Raw Normalized; Y Raw Normalized; Distance to Camera in mm; Eye Aspect Ratio (Height/Width); Blinking; Timestamp With Milliseconds; Unix Timestamp In Milliseconds; Capture Latency In Milliseconds; AOI List; Additional Notes";

        private void Start()
        {
            //Current Time
            DateTime now = DateTime.Now;

            _onPathChangeEvent.RemoveListener(PathChange);
            _onPathChangeEvent.AddListener(PathChange);

            //Use writable persistentDataPath in deployed player builds; keep Application.dataPath/CSVLogs in Editor
#if UNITY_EDITOR
            string defaultDir = $"{Application.dataPath}/CSVLogs";
#else
            string defaultDir = $"{Application.persistentDataPath}/CSVLogs";
#endif
            BaseFolderPath = string.IsNullOrWhiteSpace(BaseFolderPath) || useDefaultFolder
                ? defaultDir
                : BaseFolderPath;

            if (string.IsNullOrWhiteSpace(BaseFileName))
                BaseFileName = "GazeLog";

            //Append timestamp with leading zeroes to filename
            BaseFileName += $"_{now.Year:D4}{now.Month:D2}{now.Day:D2}_{now.Hour:D2}{now.Minute:D2}{now.Second:D2}";

            //Start timed coroutine to write data every timeUntilWrite seconds
            StartCoroutine(WriteData());
        }

        //Write when pause is triggered
        private void OnApplicationPause(bool pauseStatus)
        {
            if (pauseStatus) WriteQueue();
        }

        //Write residual queue when application is about to quit
        private void OnApplicationQuit()
        {
            WriteQueue();
        }

        private void OnDestroy()
        {
            WriteQueue();
        }

        //Return file path
        public string GetFilePath()
        {
            return $"{BaseFolderPath}/{BaseFileName}.csv";
        }
        //Set file path externally
        public void SetFilePath(string filepath)
        {
            if (string.IsNullOrWhiteSpace(filepath)) return;
            BaseFileName = Path.GetFileNameWithoutExtension(filepath);
            var dir = Path.GetDirectoryName(filepath);
            if (!string.IsNullOrEmpty(dir))
                BaseFolderPath = dir.Replace("\\", "/");
        }

        private bool TryEnsureDirectory()
        {
            if (string.IsNullOrWhiteSpace(BaseFolderPath) || string.IsNullOrWhiteSpace(BaseFileName))
            {
                Debug.LogWarning("CSVLogger: File path is not valid; buffered rows remain queued until a valid path is set.");
                return false;
            }
            if (!Directory.Exists(BaseFolderPath))
                Directory.CreateDirectory(BaseFolderPath);
            return true;
        }

        //Write full queue on path change
        private void PathChange()
        {
            WriteQueue();
        }

        //Write the Queue to Disk
        public void WriteQueue()
        {
            if (_queue.Count == 0) return;

            try
            {
                if (!TryEnsureDirectory()) return;

                string csvPath = GetFilePath();
                bool exists = File.Exists(csvPath);

                using (StreamWriter csvWriter = new StreamWriter(csvPath, exists))
                {
                    if (!exists) csvWriter.WriteLine(_Header);

                    foreach (var csvdata in _queue)
                        csvWriter.WriteLine(csvdata.SerializeCSVData());
                }
                _queue.Clear();
            }
            catch (Exception e) when (e is IOException || e is UnauthorizedAccessException)
            {
                Debug.LogWarning("CSVLogger: Could not access CSV file (locked or read-only); retaining queued rows.");
                Debug.LogException(e);
            }
        }

        /// <summary>
        /// True when the rate limiter would accept a row right now. Uses unscaledTime so logging continues
        /// cleanly when host games pause Time.timeScale.
        /// </summary>
        public bool ShouldLog => logsPerSecond > 0f && Time.unscaledTime - _lastLogTime >= 1f / logsPerSecond;

        //Append csvdata to _queue (rate-limited to logsPerSecond)
        public void Append(CSVData csvdata)
        {
            if (csvdata == null || !ShouldLog) return;
            _queue.Add(csvdata);
            _lastLogTime = Time.unscaledTime;
        }

        /// <summary>
        /// Appends a note to last CSVData entry in Queue.
        /// </summary>
        /// <param name="note">string note to append</param>
        public void AppendNote(string note)
        {
            var lastData = _queue.Count > 0 ? _queue[_queue.Count - 1] : null;
            if (lastData != null)
                lastData.notes += $"{note} ";
        }

        /// <summary>
        /// Prepends a line to the CSV file atomically via a temporary file so an I/O error can never destroy
        /// existing recorded data.
        /// </summary>
        /// <param name="line">Line to prepend</param>
        public void PrependLine(string line)
        {
            try
            {
                WriteQueue();
                if (!TryEnsureDirectory()) return;

                string csvPath = GetFilePath();
                string tempPath = csvPath + ".tmp";
                bool exists = File.Exists(csvPath);
                string sanitizedLine = (line ?? "").Replace("\r", " ").Replace("\n", " ");

                using (var writer = new StreamWriter(tempPath, append: false))
                {
                    writer.WriteLine(sanitizedLine);
                    if (exists)
                    {
                        using (var reader = new StreamReader(csvPath))
                        {
                            string existingLine;
                            while ((existingLine = reader.ReadLine()) != null)
                                writer.WriteLine(existingLine);
                        }
                    }
                    else
                    {
                        writer.WriteLine(_Header);
                    }
                }

                if (exists)
                    File.Delete(csvPath);
                File.Move(tempPath, csvPath);
            }
            catch (Exception e) when (e is IOException || e is UnauthorizedAccessException)
            {
                Debug.LogWarning("CSVLogger: Could not prepend line to CSV file.");
                Debug.LogException(e);
            }
        }

        //Write Data every timeUntilWrite seconds by calling WriteQueue()
        public IEnumerator WriteData()
        {
            while (Application.isPlaying)
            {
                WriteQueue();
                yield return new WaitForSecondsRealtime(Mathf.Max(0.1f, timeUntilWrite));
            }
        }
    }
}
