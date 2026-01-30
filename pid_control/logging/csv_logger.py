"""
Efficient CSV logging for PID data.

Features:
- Buffered writes for performance
- Automatic flushing on buffer size or time interval
- Thread-safe operation
- Graceful handling of file errors
"""

from typing import List, Dict, Any, Optional, Sequence
from pathlib import Path
import csv
import time
import threading
from collections import deque
import os


class CSVLogger:
    """
    Efficient CSV logger with buffering for high-frequency data logging.
    
    Implements buffered writes to minimize disk I/O overhead during
    real-time control loops.
    
    Example:
        >>> logger = CSVLogger("data.csv", columns=["time", "value", "error"])
        >>> logger.log({"time": 0.0, "value": 1.5, "error": 0.1})
        >>> logger.close()
    """
    
    def __init__(
        self,
        file_path: str,
        columns: List[str],
        buffer_size: int = 100,
        flush_interval: float = 1.0,
        append: bool = False
    ):
        """
        Initialize CSV logger.
        
        Args:
            file_path: Path to CSV file
            columns: List of column names
            buffer_size: Number of rows to buffer before writing
            flush_interval: Maximum seconds between flushes
            append: If True, append to existing file
        """
        if not columns:
            raise ValueError("columns cannot be empty")
        if buffer_size < 1:
            raise ValueError("buffer_size must be at least 1")
        if flush_interval <= 0:
            raise ValueError("flush_interval must be positive")
        
        self._file_path = Path(file_path)
        self._columns = columns
        self._buffer_size = buffer_size
        self._flush_interval = flush_interval
        
        # Thread safety
        self._lock = threading.Lock()
        
        # Buffer
        self._buffer: deque = deque()
        self._last_flush_time = time.time()
        self._total_rows = 0
        
        # Create directory if needed
        self._file_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Open file and write header
        mode = 'a' if append else 'w'
        self._file = open(self._file_path, mode, newline='', buffering=1)
        self._writer = csv.DictWriter(self._file, fieldnames=columns)
        
        # Write header if new file or not appending
        if not append or self._file_path.stat().st_size == 0:
            self._writer.writeheader()
        
        self._closed = False
    
    def log(self, data: Dict[str, Any]) -> None:
        """
        Log a row of data.
        
        Args:
            data: Dictionary mapping column names to values
                  Missing columns will be filled with empty string
        """
        if self._closed:
            raise RuntimeError("Logger is closed")
        
        # Ensure all columns are present
        row = {col: data.get(col, '') for col in self._columns}
        
        with self._lock:
            self._buffer.append(row)
            self._total_rows += 1
            
            # Check if we should flush
            should_flush = (
                len(self._buffer) >= self._buffer_size or
                time.time() - self._last_flush_time >= self._flush_interval
            )
        
        if should_flush:
            self.flush()
    
    def log_batch(self, data_list: Sequence[Dict[str, Any]]) -> None:
        """
        Log multiple rows of data efficiently.
        
        Args:
            data_list: List of row dictionaries
        """
        if self._closed:
            raise RuntimeError("Logger is closed")
        
        rows = [
            {col: data.get(col, '') for col in self._columns}
            for data in data_list
        ]
        
        with self._lock:
            self._buffer.extend(rows)
            self._total_rows += len(rows)
        
        if len(self._buffer) >= self._buffer_size:
            self.flush()
    
    def flush(self) -> None:
        """Flush buffer to disk."""
        with self._lock:
            if not self._buffer or self._closed:
                return
            
            rows_to_write = list(self._buffer)
            self._buffer.clear()
            self._last_flush_time = time.time()
        
        try:
            for row in rows_to_write:
                self._writer.writerow(row)
            self._file.flush()
        except IOError as e:
            # Re-add rows to buffer on failure
            with self._lock:
                self._buffer.extendleft(reversed(rows_to_write))
            raise RuntimeError(f"Failed to write to CSV: {e}")
    
    def close(self) -> None:
        """Close logger and flush remaining data."""
        if self._closed:
            return
        
        self.flush()
        
        with self._lock:
            self._closed = True
            self._file.close()
    
    @property
    def file_path(self) -> Path:
        """Get file path."""
        return self._file_path
    
    @property
    def total_rows(self) -> int:
        """Get total number of logged rows."""
        return self._total_rows
    
    @property
    def buffer_count(self) -> int:
        """Get current buffer size."""
        return len(self._buffer)
    
    def __enter__(self):
        """Context manager entry."""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.close()
        return False
    
    def __del__(self):
        """Destructor - ensure file is closed."""
        if not self._closed:
            try:
                self.close()
            except Exception:
                pass


class DataBuffer:
    """
    In-memory circular buffer for real-time data storage.
    
    Useful for keeping recent history without unbounded memory growth.
    """
    
    def __init__(self, max_size: int = 10000, columns: Optional[List[str]] = None):
        """
        Initialize data buffer.
        
        Args:
            max_size: Maximum number of rows to store
            columns: Optional list of expected columns
        """
        if max_size < 1:
            raise ValueError("max_size must be at least 1")
        
        self._max_size = max_size
        self._columns = columns
        self._buffer: deque = deque(maxlen=max_size)
        self._lock = threading.Lock()
    
    def append(self, data: Dict[str, Any]) -> None:
        """Add a row to the buffer."""
        with self._lock:
            self._buffer.append(data.copy())
    
    def extend(self, data_list: Sequence[Dict[str, Any]]) -> None:
        """Add multiple rows to the buffer."""
        with self._lock:
            for data in data_list:
                self._buffer.append(data.copy())
    
    def get_all(self) -> List[Dict[str, Any]]:
        """Get all buffered data as a list."""
        with self._lock:
            return list(self._buffer)
    
    def get_last(self, n: int) -> List[Dict[str, Any]]:
        """Get the last n rows."""
        with self._lock:
            if n >= len(self._buffer):
                return list(self._buffer)
            return list(self._buffer)[-n:]
    
    def get_column(self, column: str) -> List[Any]:
        """Get all values for a specific column."""
        with self._lock:
            return [row.get(column) for row in self._buffer]
    
    def clear(self) -> None:
        """Clear the buffer."""
        with self._lock:
            self._buffer.clear()
    
    def __len__(self) -> int:
        """Get current buffer size."""
        return len(self._buffer)
    
    @property
    def is_full(self) -> bool:
        """Check if buffer is at max capacity."""
        return len(self._buffer) >= self._max_size
    
    def to_csv(self, file_path: str) -> None:
        """
        Export buffer contents to CSV file.
        
        Args:
            file_path: Output file path
        """
        data = self.get_all()
        if not data:
            return
        
        columns = self._columns or list(data[0].keys())
        
        with open(file_path, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=columns)
            writer.writeheader()
            writer.writerows(data)
