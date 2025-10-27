"""Legacy sandbox module.

This file is kept for compatibility but the actual sandbox implementation has
been migrated to :mod:`core.eda.security.simplified_sandbox`. Importing this
module will raise an informative error to steer callers toward the supported
API.
"""

raise ImportError(
    "The legacy code_sandbox module has been removed. "
    "Use core.eda.security.simplified_sandbox.create_sandbox instead."
)
"""
            restricted_globals = self.create_restricted_globals(data_path)
            restricted_locals = {}
            
            # Execute with timeout
            def target():
                exec(code, restricted_globals, restricted_locals)
            
            thread = threading.Thread(target=target)
            thread.daemon = True
            thread.start()
            thread.join(timeout=self.max_execution_time)
            
            if thread.is_alive():
                return {
                    "success": False,
                    "error": f"Code execution timeout after {self.max_execution_time} seconds",
                    "outputs": []
                }
            
            # Get outputs
            stdout_content = stdout_capture.getvalue()
            stderr_content = stderr_capture.getvalue()
            
            outputs = []
            if stdout_content:
                outputs.append({
                    "type": "stream",
                    "name": "stdout", 
                    "text": stdout_content
                })
            
            if stderr_content:
                outputs.append({
                    "type": "stream",
                    "name": "stderr",
                    "text": stderr_content
                })
            
            return {
                "success": True,
                "outputs": outputs,
                "execution_count": 1
            }
            
        except Exception as e:
            return {
                "success": False,
                "error": f"Execution error: {str(e)}",
                "outputs": []
            }
        
        finally:
            # Restore output streams
            sys.stdout = old_stdout
            sys.stderr = old_stderr


class ProcessSandbox:
    """Process-based sandbox using subprocess with additional restrictions"""
    
    def __init__(self, max_execution_time: int = 30):
        self.max_execution_time = max_execution_time
        self.code_sandbox = CodeSandbox()
    
    def execute_code_in_process(self, code: str, data_path: str) -> Dict[str, Any]:
        """Execute code in a separate process with additional restrictions"""
        
        # Validate code first
        validation = self.code_sandbox.validate_code_security(code)
        if not validation["valid"]:
            return {
                "success": False,
                "error": f"Security violations: {', '.join(validation['violations'])}",
                "outputs": []
            }
        
        # Create sandboxed execution script (Windows compatible)
        # Properly indent the user code
        indented_code = '\n'.join('    ' + line if line.strip() else line for line in code.split('\n'))
        
        sandbox_script = f'''
import sys
import os
import platform

class SecurityError(Exception):
    pass

# Remove dangerous modules from sys.modules AFTER we import what we need
dangerous_modules = ['subprocess', 'socket', 'urllib', 'requests', 'sqlite3', 'psycopg2', 'pymongo']

# Set up timeout handler (Unix-like systems only)
if platform.system() != 'Windows':
    import signal
    def timeout_handler(signum, frame):
        raise TimeoutError("Code execution timeout")
    signal.signal(signal.SIGALRM, timeout_handler)
    signal.alarm({self.max_execution_time})

try:

"""
