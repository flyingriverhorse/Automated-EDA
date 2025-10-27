"""
Simplified Secure Sandbox for EDA Custom Analysis

This module provides a streamlined secure sandbox environment for executing 
user-provided Python code with strict limitations to prevent access to system 
resources, other databases, or external networks.

Security Features:
- Import restrictions (regex pattern blocking)
- File system access restrictions (pandas w        except Exception as e:
            timeout_event.set()  # Signal monitor thread to stop
            return {
                "success": False,
                "error": f"Sandbox execution failed: {str(e)}",
                "outputs": []
            }
        finally:
            # Clean up temporary file
            try:
                temp_script_path.unlink()
            except:
                passopen override)
- Network access blocking
- Database connection blocking
- System command execution blocking
- Code pattern analysis and filtering
"""

import os
import sys
import tempfile
import subprocess
import re
import builtins
import time
import threading
import psutil
from typing import Dict, Any, List
from pathlib import Path
import logging

logger = logging.getLogger(__name__)

class SecurityError(Exception):
    """Raised when code violates security policies"""
    pass

class ResourceLimitError(Exception):
    """Raised when resource limits are exceeded"""
    pass

class SimplifiedSandbox:
    """Simplified secure sandbox for executing Python code with EDA restrictions"""
    
    # Resource limits
    DEFAULT_MAX_EXECUTION_TIME = 120  # seconds
    DEFAULT_MAX_MEMORY_MB = 512  # MB
    DEFAULT_MAX_CPU_PERCENT = 50  # % of one CPU core
    
    # Dangerous patterns to detect in code
    DANGEROUS_PATTERNS = [
        r'import\s+os\b',
        r'import\s+sys\b', 
        r'import\s+subprocess\b',
        r'import\s+socket\b',
        r'import\s+urllib\b',
        r'import\s+requests\b',
        r'import\s+sqlite3\b',
        r'import\s+psycopg2\b',
        r'import\s+pymongo\b',
        r'import\s+mysql\b',
        r'from\s+os\b',
        r'from\s+sys\b',
        r'from\s+subprocess\b',
        r'from\s+socket\b',
        r'from\s+urllib\b',
        r'from\s+requests\b',
        r'__import__\s*\(',
        r'eval\s*\(',
        r'exec\s*\(',
        r'compile\s*\(',
        # r'open\s*\(',  # Remove this - we handle open() in the sandbox itself
        r'\.system\s*\(',
        r'\.popen\s*\(',
        r'\.call\s*\(',
        r'\.run\s*\(',
        r'connect\s*\(',
        r'\.cursor\s*\(',
        r'\.execute\s*\(',
        r'\.commit\s*\(',
        r'\.rollback\s*\(',
        # Package installation patterns (various package managers)
        r'pip\s+install\b',
        r'pip3\s+install\b',
        r'uv\s+add\b',
        r'uv\s+install\b',
        r'conda\s+install\b',
        r'mamba\s+install\b',
        r'poetry\s+add\b',
        r'pipenv\s+install\b',
        r'easy_install\b',
        r'setup\.py\s+install\b',
        r'python\s+-m\s+pip\s+install\b',
        r'python3\s+-m\s+pip\s+install\b',
        r'!pip\s+install\b',  # Jupyter notebook magic command
        r'!conda\s+install\b',  # Jupyter notebook magic command
        r'!uv\s+add\b',  # Jupyter notebook magic command
        # System shell access patterns
        r'!\s*[a-zA-Z]',  # Any shell command starting with !
        r'%%bash\b',  # Jupyter notebook bash magic
        r'%%sh\b',    # Jupyter notebook shell magic
        # Additional bypass attempts
        r'importlib\.import_module\s*\(',  # Dynamic imports
        r'__builtins__',  # Accessing builtins (simplified pattern)
        r'globals\s*\(\s*\)\s*\[',  # Accessing globals
        r'locals\s*\(\s*\)\s*\[',   # Accessing locals
        r'vars\s*\(\s*\)',          # vars() function
        r'dir\s*\(\s*\)',           # dir() function
        r'getattr\s*\(',            # getattr() function
        r'setattr\s*\(',            # setattr() function
        r'delattr\s*\(',            # delattr() function
        r'hasattr\s*\(',            # hasattr() function
        # Environment manipulation
        r'os\.environ\s*\[',        # Environment variables
        r'sys\.path\s*\.',          # Python path manipulation
        r'sys\.modules\s*\[',       # Module manipulation
        # File system bypass attempts
        r'pathlib\.Path\s*\(',      # pathlib access
        r'tempfile\.',              # tempfile module usage
        r'shutil\.',                # shutil module usage
        r'glob\.',                  # glob module usage
        # Network/web access
        r'http\.',                  # http module
        r'urllib\.',               # urllib module
        r'ftplib\.',               # ftp access
        r'smtplib\.',              # email access
        r'telnetlib\.',            # telnet access
    ]
    
    def __init__(self, max_execution_time: int = None, max_memory_mb: int = None, 
                 max_cpu_percent: int = None, user_id: str = None):
        """Initialize sandbox with resource limits"""
        self.max_execution_time = max_execution_time or self.DEFAULT_MAX_EXECUTION_TIME
        self.max_memory_mb = max_memory_mb or self.DEFAULT_MAX_MEMORY_MB
        self.max_cpu_percent = max_cpu_percent or self.DEFAULT_MAX_CPU_PERCENT
        self.user_id = user_id or "default"
        
        # Create user-specific temp directory
        self.user_temp_dir = Path(tempfile.gettempdir()) / f"eda_user_{self.user_id}"
        self.user_temp_dir.mkdir(exist_ok=True)
        
        logger.info(f"Initialized sandbox for user {self.user_id} with limits: "
                   f"time={self.max_execution_time}s, memory={self.max_memory_mb}MB, "
                   f"cpu={self.max_cpu_percent}%")
        
    def validate_code_security(self, code: str) -> Dict[str, Any]:
        """Validate code against security policies"""
        violations = []
        
        # Check for dangerous patterns
        for pattern in self.DANGEROUS_PATTERNS:
            if re.search(pattern, code, re.IGNORECASE):
                violations.append(f"Dangerous pattern detected: {pattern}")
        
        return {
            "valid": len(violations) == 0,
            "violations": violations,
            "safe": len(violations) == 0
        }
    
    def _monitor_process_resources(self, process, timeout_event):
        """Monitor process resource usage and terminate if limits exceeded"""
        try:
            psutil_process = psutil.Process(process.pid)
            start_time = time.time()
            
            while not timeout_event.is_set():
                try:
                    if not psutil_process.is_running():
                        break
                        
                    # Check memory usage
                    memory_mb = psutil_process.memory_info().rss / (1024 * 1024)
                    if memory_mb > self.max_memory_mb:
                        logger.warning(f"Process {process.pid} exceeded memory limit: {memory_mb:.1f}MB > {self.max_memory_mb}MB")
                        process.terminate()
                        timeout_event.set()
                        break
                    
                    # Check CPU usage (averaged over 1 second)
                    cpu_percent = psutil_process.cpu_percent(interval=1)
                    if cpu_percent > self.max_cpu_percent:
                        logger.warning(f"Process {process.pid} exceeded CPU limit: {cpu_percent:.1f}% > {self.max_cpu_percent}%")
                        process.terminate()
                        timeout_event.set()
                        break
                        
                    # Check execution time
                    elapsed_time = time.time() - start_time
                    if elapsed_time > self.max_execution_time:
                        logger.warning(f"Process {process.pid} exceeded time limit: {elapsed_time:.1f}s > {self.max_execution_time}s")
                        process.terminate()
                        timeout_event.set()
                        break
                        
                except (psutil.NoSuchProcess, psutil.AccessDenied):
                    break
                    
        except Exception as e:
            logger.error(f"Error monitoring process resources: {e}")
            
    def _cleanup_user_temp_files(self):
        """Clean up old temporary files for this user"""
        try:
            cutoff_time = time.time() - 3600  # Remove files older than 1 hour
            for temp_file in self.user_temp_dir.glob("*.py"):
                if temp_file.stat().st_mtime < cutoff_time:
                    temp_file.unlink()
                    logger.debug(f"Cleaned up old temp file: {temp_file}")
        except Exception as e:
            logger.warning(f"Error cleaning up temp files: {e}")
    
    def execute_code_safely(self, code: str, data_path: str) -> Dict[str, Any]:
        """Execute code in secure subprocess with limited environment"""
        
        # First, validate the code
        validation = self.validate_code_security(code)
        if not validation["valid"]:
            return {
                "success": False,
                "error": f"Security violations detected: {', '.join(validation['violations'])}",
                "outputs": []
            }
        
        # Create execution script with limited environment and file access control
        # Properly indent user code
        indented_code = '\n'.join('    ' + line if line.strip() else line for line in code.split('\n'))
        
        # Get absolute path for comparison
        abs_data_path = os.path.abspath(data_path)
        
        execution_script = f'''
import sys
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Use non-GUI backend
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

# Disable any GUI elements
import os
os.environ['DISPLAY'] = ''

# Override pandas file reading functions to restrict access
_original_read_csv = pd.read_csv
_original_read_excel = pd.read_excel
_original_read_json = pd.read_json
_original_read_parquet = getattr(pd, 'read_parquet', None)

ALLOWED_FILE = r'{abs_data_path}'

class FileAccessError(Exception):
    pass

def secure_read_csv(filepath_or_buffer, **kwargs):
    # Allow only if it's the assigned dataset file
    if isinstance(filepath_or_buffer, str):
        abs_requested = os.path.abspath(filepath_or_buffer)
        if abs_requested != ALLOWED_FILE:
            raise FileAccessError(f"Access denied: Can only read assigned dataset file. Requested: {{filepath_or_buffer}}")
    return _original_read_csv(filepath_or_buffer, **kwargs)

def secure_read_excel(filepath_or_buffer, **kwargs):
    if isinstance(filepath_or_buffer, str):
        abs_requested = os.path.abspath(filepath_or_buffer)  
        if abs_requested != ALLOWED_FILE:
            raise FileAccessError(f"Access denied: Can only read assigned dataset file. Requested: {{filepath_or_buffer}}")
    return _original_read_excel(filepath_or_buffer, **kwargs)

def secure_read_json(filepath_or_buffer, **kwargs):
    if isinstance(filepath_or_buffer, str):
        abs_requested = os.path.abspath(filepath_or_buffer)
        if abs_requested != ALLOWED_FILE:
            raise FileAccessError(f"Access denied: Can only read assigned dataset file. Requested: {{filepath_or_buffer}}")
    return _original_read_json(filepath_or_buffer, **kwargs)

# Replace pandas functions with secure versions
pd.read_csv = secure_read_csv
pd.read_excel = secure_read_excel  
pd.read_json = secure_read_json
if _original_read_parquet:
    def secure_read_parquet(filepath_or_buffer, **kwargs):
        if isinstance(filepath_or_buffer, str):
            abs_requested = os.path.abspath(filepath_or_buffer)
            if abs_requested != ALLOWED_FILE:
                raise FileAccessError(f"Access denied: Can only read assigned dataset file. Requested: {{filepath_or_buffer}}")
        return _original_read_parquet(filepath_or_buffer, **kwargs)
    pd.read_parquet = secure_read_parquet

# Override the built-in open function to prevent file access
_original_open = open

def secure_open(filepath, mode='r', **kwargs):
    abs_requested = os.path.abspath(filepath)
    # Allow reading only the assigned dataset file
    if 'r' in mode and abs_requested == ALLOWED_FILE:
        return _original_open(filepath, mode, **kwargs)
    else:
        raise FileAccessError(f"File access denied: {{filepath}}")

# Override built-in open function
import builtins
builtins.open = secure_open

try:
    # Load dataset
    df = _original_read_csv(r'{data_path}')
    print(f"Dataset loaded: {{df.shape}}")
    
    # Execute user code
{indented_code}

except Exception as e:
    print(f"ERROR: {{e}}")
'''
        
        # Write execution script to user-specific temporary file
        temp_script_path = self.user_temp_dir / f"exec_{int(time.time())}_{os.getpid()}.py"
        try:
            with open(temp_script_path, 'w', encoding='utf-8') as temp_file:
                temp_file.write(execution_script)
            
            # Clean up old temp files
            self._cleanup_user_temp_files()
            
            # Get the current Python environment
            python_executable = sys.executable
            
            # Start process with Popen for better monitoring
            process = subprocess.Popen(
                [python_executable, str(temp_script_path)],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                cwd=os.path.dirname(data_path),
                env=os.environ.copy()
            )
            
            # Set up resource monitoring
            timeout_event = threading.Event()
            monitor_thread = threading.Thread(
                target=self._monitor_process_resources,
                args=(process, timeout_event),
                daemon=True
            )
            monitor_thread.start()
            
            # Wait for process to complete or timeout
            try:
                stdout, stderr = process.communicate(timeout=self.max_execution_time + 5)
                timeout_event.set()  # Signal monitor thread to stop
            except subprocess.TimeoutExpired:
                process.terminate()
                timeout_event.set()
                stdout, stderr = process.communicate(timeout=5)
                return {
                    "success": False,
                    "error": f"Code execution timed out after {self.max_execution_time} seconds",
                    "outputs": []
                }
            
            # Collect outputs
            outputs = []
            if stdout:
                outputs.append({"type": "text", "text": stdout})
            
            if stderr and process.returncode != 0:
                outputs.append({"type": "error", "text": stderr})
            
            return {
                "success": process.returncode == 0,
                "outputs": outputs,
                "execution_result": {
                    "stdout": stdout,
                    "stderr": stderr,
                    "returncode": process.returncode
                }
            }
            
        except subprocess.TimeoutExpired:
            return {
                "success": False,
                "error": f"Code execution timed out after {{self.max_execution_time}} seconds",
                "outputs": []
            }
        except Exception as e:
            return {
                "success": False,
                "error": f"Sandbox execution failed: {{str(e)}}",
                "outputs": []
            }
        finally:
            # Clean up temporary file
            try:
                os.unlink(temp_script_path)
            except:
                pass


# Update the factory function
def create_sandbox(sandbox_type: str = "simplified", user_id: str = None, **kwargs):
    """Create sandbox instance with user isolation.

    Legacy sandbox types ("process", "thread") now route to the simplified sandbox
    implementation while preserving the requested resource limits.
    """

    normalized_type = (sandbox_type or "simplified").lower()
    if normalized_type in {"simplified", "process", "thread"}:
        return SimplifiedSandbox(user_id=user_id, **kwargs)
    raise ValueError(f"Unsupported sandbox type: {sandbox_type}")