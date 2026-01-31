"""
CSV data reader for system identification.

Reads experimental data from CSV files containing input/output measurements
and PID controller data.
"""

from typing import Dict, List, Optional, Tuple
from pathlib import Path
import csv
import numpy as np
from dataclasses import dataclass


@dataclass
class ExperimentalData:
    """Container for experimental data from CSV."""
    time: np.ndarray
    input: np.ndarray
    output: np.ndarray
    setpoint: Optional[np.ndarray] = None
    error: Optional[np.ndarray] = None
    kp: Optional[float] = None
    ki: Optional[float] = None
    kd: Optional[float] = None
    sample_time: Optional[float] = None
    metadata: Optional[Dict] = None
    
    def __post_init__(self):
        if len(self.time) != len(self.input) or len(self.time) != len(self.output):
            raise ValueError("Time, input, and output arrays must have the same length")
        
        if len(self.time) < 10:
            raise ValueError("Need at least 10 data points for system identification")


class CSVDataReader:
    """
    Read and parse CSV files containing experimental control data.
    
    Expected CSV format:
    - Header row with column names
    - Columns: time, input, output (required)
    - Optional: setpoint, error, kp, ki, kd
    
    Example CSV:
        time,input,output,setpoint
        0.0,0.0,0.0,1.0
        0.01,5.2,0.05,1.0
        0.02,8.1,0.12,1.0
        ...
    """
    
    def __init__(self, file_path: str):
        """
        Initialize CSV reader.
        
        Args:
            file_path: Path to CSV file
        """
        self._file_path = Path(file_path)
        if not self._file_path.exists():
            raise FileNotFoundError(f"CSV file not found: {file_path}")
    
    def read(
        self,
        time_col: str = 'timestamp',
        input_col: str = 'output',
        output_col: str = 'measurement',
        setpoint_col: Optional[str] = 'setpoint',
        error_col: Optional[str] = None,
        skip_rows: int = 0,
        max_rows: Optional[int] = None
    ) -> ExperimentalData:
        """
        Read CSV file and extract experimental data.
        
        Args:
            time_col: Name of time column
            input_col: Name of input (control signal) column
            output_col: Name of output (process variable) column
            setpoint_col: Name of setpoint column (optional)
            error_col: Name of error column (optional)
            skip_rows: Number of data rows to skip after header
            max_rows: Maximum number of rows to read (None = all)
        
        Returns:
            ExperimentalData object with parsed data
        """
        data = {
            'time': [],
            'input': [],
            'output': [],
            'setpoint': [],
            'error': []
        }
        
        with open(self._file_path, 'r') as f:
            reader = csv.DictReader(f)
            
            if time_col not in reader.fieldnames:
                raise ValueError(f"Time column '{time_col}' not found in CSV")
            if input_col not in reader.fieldnames:
                raise ValueError(f"Input column '{input_col}' not found in CSV")
            if output_col not in reader.fieldnames:
                raise ValueError(f"Output column '{output_col}' not found in CSV")
            
            has_setpoint = setpoint_col and setpoint_col in reader.fieldnames
            has_error = error_col and error_col in reader.fieldnames
            
            row_count = 0
            for i, row in enumerate(reader):
                if i < skip_rows:
                    continue
                
                if max_rows and row_count >= max_rows:
                    break
                
                try:
                    data['time'].append(float(row[time_col]))
                    data['input'].append(float(row[input_col]))
                    data['output'].append(float(row[output_col]))
                    
                    if has_setpoint:
                        data['setpoint'].append(float(row[setpoint_col]))
                    if has_error:
                        data['error'].append(float(row[error_col]))
                    
                    row_count += 1
                except (ValueError, KeyError) as e:
                    print(f"Warning: Skipping row {i+1} due to error: {e}")
                    continue
        
        if len(data['time']) == 0:
            raise ValueError("No valid data rows found in CSV file")
        
        time_array = np.array(data['time'])
        sample_time = self._estimate_sample_time(time_array)
        
        return ExperimentalData(
            time=time_array,
            input=np.array(data['input']),
            output=np.array(data['output']),
            setpoint=np.array(data['setpoint']) if has_setpoint else None,
            error=np.array(data['error']) if has_error else None,
            sample_time=sample_time,
            metadata={'file_path': str(self._file_path), 'num_samples': len(time_array)}
        )
    
    def read_with_pid_params(
        self,
        time_col: str = 'timestamp',
        input_col: str = 'input',
        output_col: str = 'output',
        setpoint_col: Optional[str] = 'setpoint',
        kp_value: Optional[float] = None,
        ki_value: Optional[float] = None,
        kd_value: Optional[float] = None
    ) -> ExperimentalData:
        """
        Read CSV and include PID parameters used during data collection.
        
        Args:
            time_col: Name of time column
            input_col: Name of input column
            output_col: Name of output column
            setpoint_col: Name of setpoint column
            kp_value: Proportional gain used
            ki_value: Integral gain used
            kd_value: Derivative gain used
        
        Returns:
            ExperimentalData with PID parameters
        """
        exp_data = self.read(time_col, input_col, output_col, setpoint_col)
        exp_data.kp = kp_value
        exp_data.ki = ki_value
        exp_data.kd = kd_value
        
        return exp_data
    
    @staticmethod
    def _estimate_sample_time(time_array: np.ndarray) -> float:
        """Estimate sample time from time array."""
        if len(time_array) < 2:
            return 0.01
        
        diffs = np.diff(time_array)
        median_dt = np.median(diffs)
        
        return float(median_dt)
    
    @staticmethod
    def detect_step_response(
        data: ExperimentalData,
        threshold: float = 0.1
    ) -> Tuple[int, int]:
        """
        Detect the start and end of a step response in the data.
        
        Args:
            data: Experimental data
            threshold: Threshold for detecting step change (fraction of input range)
        
        Returns:
            Tuple of (start_index, end_index) for step response
        """
        input_diff = np.abs(np.diff(data.input))
        input_range = np.max(data.input) - np.min(data.input)
        
        if input_range < 1e-6:
            return 0, len(data.time) - 1
        
        step_threshold = threshold * input_range
        step_indices = np.where(input_diff > step_threshold)[0]
        
        if len(step_indices) == 0:
            return 0, len(data.time) - 1
        
        start_idx = step_indices[0]
        
        settling_window = min(200, len(data.time) - start_idx - 1)
        if settling_window < 10:
            end_idx = len(data.time) - 1
        else:
            end_idx = min(start_idx + settling_window, len(data.time) - 1)
        
        return start_idx, end_idx
