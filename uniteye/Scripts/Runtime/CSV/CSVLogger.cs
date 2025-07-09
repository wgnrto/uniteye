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
                //Return if no change
                if (value.Equals(_baseFolderPath)) return;

                //Invoke Event to write queue
                _onPathChangeEvent.Invoke();
                //Change path
                _baseFolderPath = value;
            }
        }
        //Fire PathChange Event on set
        public string BaseFileName
        {
            get => _baseFileName;
            set
            {
                //Return if no change
                if (value.Equals(_baseFileName)) return;

                //Invoke Event to write queue
                _onPathChangeEvent.Invoke();
                //Change path
                _baseFileName = value;
            }
        }
        [SerializeField] float timeUntilWrite = 10f;
        [SerializeField] float logsPerSecond = 999f;

        private float timeSinceLastLog = 0f;

        //Unity Event to handle path changes
        private UnityEvent _onPathChangeEvent;

        //Private queue
        private readonly List<CSVData> _queue = new List<CSVData>();

        //Header row to name columns in csv file
        private static readonly string _Header = "X Filtered Pixel; Y Filtered Pixel; X Filtered Normalized; Y Filtered Normalized; X Raw Normalized; Y Raw Normalized; Distance to Camera in mm; Eye Aspect Ratio (Height/Width); Blinking; Timestamp With Milliseconds; Unix Timestamp In Milliseconds; AOI List; Additional Notes";

        private void Start()
        {
            //Current Time
            DateTime now = DateTime.Now;

            //Set up Event Listener
            if (_onPathChangeEvent == null)
                _onPathChangeEvent = new UnityEvent();
            _onPathChangeEvent.AddListener(PathChange);

            //Default to Application.persistentDataPath/CSVLogs for Android
            //Else if useDefaultFolder default to Application.dataPath/CSVLogs or keep the path if useDefaultFolder is false
            BaseFolderPath = Application.platform == RuntimePlatform.Android ? $"{Application.persistentDataPath}/CSVLogs" : BaseFolderPath == "" || useDefaultFolder ? $"{Application.dataPath}/CSVLogs" : BaseFolderPath;

            //Append timestamp with leading zeroes to filename
            BaseFileName += $"_{now.Year}" +
                $"{((now.Month < 10) ? "0"+now.Month : now.Month)}" +
                $"{((now.Day < 10) ? "0" + now.Day : now.Day)}" +
                $"_{((now.Hour < 10) ? "0" + now.Hour : now.Hour)}" +
                $"{((now.Minute < 10) ? "0" + now.Minute : now.Minute)}" +
                $"{((now.Second < 10) ? "0" + now.Second : now.Second)}";

            //Start timed coroutine to write data every timeUntilWrite seconds
            StartCoroutine(WriteData());
        }

        //Write when pause is triggered
        private void OnApplicationPause(bool pauseStatus)
        {
            WriteQueue();
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
            //Might need tweaking for different slashes in path
            BaseFileName = Path.GetFileNameWithoutExtension(filepath);
            BaseFolderPath = Path.GetDirectoryName(filepath).Replace("\\", "/");
        }
        //Check FilePath
        private void IsFilePathValid()
        {
            //Assert that the path is not null and not empty
            Assert.AreNotEqual(null, BaseFolderPath);
            Assert.AreNotEqual(null, BaseFileName);
            Assert.AreNotEqual("", BaseFolderPath);
            Assert.AreNotEqual("", BaseFileName);

            //Create Folder if it does not exist already
            if (!Directory.Exists(BaseFolderPath)) Directory.CreateDirectory(BaseFolderPath);
        }
        //Write full queue on path change
        private void PathChange()
        {
            WriteQueue();
        }

        //Write the Queue to Disk
        public void WriteQueue()
        {
            //If queue empty return
            if (_queue.Count == 0) return;

            //Try Catch possible AssertionException
            try
            {
                IsFilePathValid();
            } catch (AssertionException e)
            {
                Debug.Log("File Path is not valid, please ensure a valid file path is provided.");
                Debug.LogException(e);
            }

            //Check if file exists already
            var exists = File.Exists(GetFilePath());

            //Open a StreamWriter to write data, creates a new file or appends depending on exists
            try
            {
                using (StreamWriter csvWriter = new StreamWriter($"{BaseFolderPath}/{BaseFileName}.csv", exists))
                {
                    //Write _Header row if first write to file
                    if (!exists) csvWriter.WriteLine(_Header);

                    //Write each CSVData in the _queue and then remove it
                    foreach (var csvdata in _queue.ToArray())
                    {
                        //Debug.Log(csvdata.SerializeCSVData());
                        csvWriter.WriteLine(csvdata.SerializeCSVData());
                        _queue.Remove(csvdata);
                    }
                }
            } catch (IOException e)
            {
                Debug.Log("Could not access file, a different program might be locking it!");
                Debug.LogException(e);
            }
        }

        //Append csvdata to _queue
        public void Append(CSVData csvdata)
        {
            //Add deltaTime from last to current frame to timeSinceLastLog
            timeSinceLastLog += Time.deltaTime;

            //Only log if enough time from the last log has elapsed, provided by 1 / logPerSecond
            if (timeSinceLastLog >= 1 / logsPerSecond)
            {
                _queue.Add(csvdata);
                //Reset timeSinceLastLog
                timeSinceLastLog = 0f;
            }
        }

        /// <summary>
        /// Appends a note to last CSVData entry in Queue.
        /// </summary>
        /// <param name="note">string note to append</param>
        public void AppendNote(string note)
        {
            var lastData = _queue.LastOrDefault();
            //It's possible that a bad script execution order can write the queue between the CSVData creation and AppendNote
            if (lastData != null)
                lastData.notes += $"{note} ";
        }

        /// <summary>
        /// Prepends a line to the CSV file. This rewrites the entire file and can freeze the program for a while if the csv file is large!
        /// </summary>
        /// <param name="line">Line to prepend</param>
        public void PrependLine(string line)
        {
            //Try Catch possible AssertionException
            try
            {
                IsFilePathValid();
            }
            catch (AssertionException e)
            {
                Debug.Log("File Path is not valid, please ensure a valid file path is provided.");
                Debug.LogException(e);
            }

            //Check if file exists already
            var exists = File.Exists(GetFilePath());

            //Open a StreamWriter to write data, creates a new file or appends depending on exists
            try
            {
                //Initialize strings
                string fileContent = "";
                string csvPath = ($"{BaseFolderPath}/{BaseFileName}.csv");

                //Read contents of file if it exists
                if (exists)
                {
                    using (StreamReader csvReader = new StreamReader(csvPath))
                    {
                        fileContent = csvReader.ReadToEnd();
                    }
                }

                //Delete file
                File.Delete(csvPath);

                //Write contents with line prepended
                using (StreamWriter csvWriter = new StreamWriter(csvPath, exists))
                {
                    //Prepend line
                    csvWriter.WriteLine(line);

                    //Write contents of file
                    csvWriter.Write(fileContent);

                    //Write _Header row if first write to file
                    if (!exists) csvWriter.WriteLine(_Header);
                }
            }
            catch (IOException e)
            {
                Debug.Log("Could not access file, it might not exist or a different program might be locking it!");
                Debug.LogException(e);
            }
        }

        //Write Data every timeUntilWrite seconds by calling WriteQueue()
        public IEnumerator WriteData()
        {
            //While Application is running, call WriteQueue every timeUntilWrite seconds
            while (Application.isPlaying)
            {
                WriteQueue();
                yield return new WaitForSeconds(timeUntilWrite);
            }
        }
    }
}
