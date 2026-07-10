using UnityEngine;

namespace UnitEye
{
    /// <summary>
    /// Quantizes a continuous gaze location into a coarse grid of screen cells with
    /// hysteresis and dwell time. This is the recommended way to consume webcam gaze
    /// data for coarse targets (for example "which fifth of the screen is the user
    /// looking at"): sub-cell jitter disappears entirely and cell changes only happen
    /// after the gaze has clearly settled in another cell.
    ///
    /// Usage:
    /// <code>
    /// var quantizer = new GazeGridQuantizer(columns: 3, rows: 3);
    ///
    /// void Update()
    /// {
    ///     var gaze = UnitEyeAPI.GetGazeLocationInGUI();
    ///     var normalized = new Vector2(gaze.x / Screen.width, gaze.y / Screen.height);
    ///     if (quantizer.Update(normalized, Time.unscaledTime))
    ///         Debug.Log($"Now looking at cell {quantizer.CurrentColumn}, {quantizer.CurrentRow}");
    /// }
    /// </code>
    /// </summary>
    public class GazeGridQuantizer
    {
        /// <summary>Number of grid columns.</summary>
        public int Columns { get; }

        /// <summary>Number of grid rows.</summary>
        public int Rows { get; }

        /// <summary>
        /// Extra margin around the active cell, as a fraction of one cell (0 to 0.49).
        /// The gaze has to leave the active cell by more than this margin before a
        /// switch is even considered, which suppresses flicker on cell borders.
        /// </summary>
        public float HysteresisMargin { get; }

        /// <summary>
        /// Time in seconds the gaze has to stay inside the same new cell before the
        /// active cell switches. Suppresses switches from single outlier samples.
        /// </summary>
        public float DwellSeconds { get; }

        /// <summary>Linear index (row * Columns + column) of the active cell, -1 before the first sample.</summary>
        public int CurrentCell { get; private set; } = -1;

        /// <summary>Column of the active cell, -1 before the first sample.</summary>
        public int CurrentColumn => CurrentCell < 0 ? -1 : CurrentCell % Columns;

        /// <summary>Row of the active cell, -1 before the first sample.</summary>
        public int CurrentRow => CurrentCell < 0 ? -1 : CurrentCell / Columns;

        private int _candidateCell = -1;
        private float _candidateSince;

        /// <param name="columns">Number of grid columns, at least 1</param>
        /// <param name="rows">Number of grid rows, at least 1</param>
        /// <param name="hysteresisMargin">Extra margin around the active cell as a fraction of one cell, 0 to 0.49</param>
        /// <param name="dwellSeconds">Time the gaze has to stay in a new cell before switching, 0 switches immediately</param>
        public GazeGridQuantizer(int columns, int rows, float hysteresisMargin = 0.15f, float dwellSeconds = 0.1f)
        {
            Columns = Mathf.Max(1, columns);
            Rows = Mathf.Max(1, rows);
            HysteresisMargin = Mathf.Clamp(hysteresisMargin, 0f, 0.49f);
            DwellSeconds = Mathf.Max(0f, dwellSeconds);
        }

        /// <summary>
        /// Feeds one gaze sample into the quantizer.
        /// </summary>
        /// <param name="normalizedGaze">Gaze location normalized to the screen, (0,0) top left to (1,1) bottom right</param>
        /// <param name="time">Current time in seconds, usually Time.unscaledTime</param>
        /// <returns>true when the active cell changed with this sample</returns>
        public bool Update(Vector2 normalizedGaze, float time)
        {
            //Ignore invalid samples
            if (float.IsNaN(normalizedGaze.x) || float.IsNaN(normalizedGaze.y)) return false;

            var sampleCell = CellAt(normalizedGaze);

            //Adopt the very first sample immediately
            if (CurrentCell < 0)
            {
                CurrentCell = sampleCell;
                _candidateCell = -1;
                return true;
            }

            //While the gaze is still inside the active cell plus hysteresis margin, stay put
            if (IsInsideCellWithMargin(normalizedGaze, CurrentCell, HysteresisMargin))
            {
                _candidateCell = -1;
                return false;
            }

            //The gaze clearly left the active cell, track the new cell as a candidate
            //and only switch after it held the same candidate for the dwell time
            if (sampleCell != _candidateCell)
            {
                _candidateCell = sampleCell;
                _candidateSince = time;
            }

            if (time - _candidateSince >= DwellSeconds)
            {
                CurrentCell = _candidateCell;
                _candidateCell = -1;
                return true;
            }

            return false;
        }

        /// <summary>
        /// Resets the quantizer to its initial state.
        /// </summary>
        public void Reset()
        {
            CurrentCell = -1;
            _candidateCell = -1;
        }

        /// <summary>
        /// Returns the linear cell index for a normalized gaze location, clamped to the grid.
        /// </summary>
        public int CellAt(Vector2 normalizedGaze)
        {
            var column = Mathf.Clamp((int)(normalizedGaze.x * Columns), 0, Columns - 1);
            var row = Mathf.Clamp((int)(normalizedGaze.y * Rows), 0, Rows - 1);
            return row * Columns + column;
        }

        /// <summary>
        /// Returns the normalized screen rect covered by a cell.
        /// </summary>
        public Rect GetCellRect(int cell)
        {
            var column = cell % Columns;
            var row = cell / Columns;
            return new Rect((float)column / Columns, (float)row / Rows, 1f / Columns, 1f / Rows);
        }

        private bool IsInsideCellWithMargin(Vector2 normalizedGaze, int cell, float margin)
        {
            var column = cell % Columns;
            var row = cell / Columns;

            var minX = (column - margin) / Columns;
            var maxX = (column + 1 + margin) / Columns;
            var minY = (row - margin) / Rows;
            var maxY = (row + 1 + margin) / Rows;

            return normalizedGaze.x >= minX && normalizedGaze.x <= maxX
                && normalizedGaze.y >= minY && normalizedGaze.y <= maxY;
        }
    }
}
