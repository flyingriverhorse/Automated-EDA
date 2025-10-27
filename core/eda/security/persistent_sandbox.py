"""Persistent sandbox manager providing stateful execution sessions."""

from __future__ import annotations

import json
import os
import queue
import subprocess
import sys
import threading
import time
from pathlib import Path
from typing import Dict, Optional, Tuple

import psutil

from .simplified_sandbox import SimplifiedSandbox


class SandboxExecutionError(RuntimeError):
    """Raised when the persistent sandbox cannot execute code."""


class PersistentSandboxSession:
    """Stateful sandbox session backed by a dedicated worker process."""

    def __init__(
        self,
        data_path: str,
        user_id: str,
        *,
        max_execution_time: int = 30,
        max_memory_mb: int = 512,
        max_cpu_percent: int = 50,
    ) -> None:
        self.data_path = Path(data_path)
        self.user_id = user_id or "default"
        self.max_execution_time = max_execution_time
        self.max_memory_mb = max_memory_mb
        self.max_cpu_percent = max_cpu_percent

        if not self.data_path.exists():
            raise FileNotFoundError(f"Dataset not found: {self.data_path}")

        self._validator = SimplifiedSandbox(
            max_execution_time=max_execution_time,
            max_memory_mb=max_memory_mb,
            max_cpu_percent=max_cpu_percent,
            user_id=self.user_id,
        )

        worker_path = Path(__file__).with_name("persistent_worker.py")
        python_executable = sys.executable

        self._process = subprocess.Popen(
            [python_executable, str(worker_path), str(self.data_path), str(self.max_execution_time)],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            bufsize=1,
        )

        if not self._process.stdin or not self._process.stdout:
            raise SandboxExecutionError("Unable to establish communication with sandbox worker")

        self._stdout_queue: "queue.Queue[str]" = queue.Queue()
        self._stderr_queue: "queue.Queue[str]" = queue.Queue()

        self._stdout_thread = threading.Thread(
            target=self._enqueue_stream,
            args=(self._process.stdout, self._stdout_queue),
            daemon=True,
        )
        self._stdout_thread.start()

        self._stderr_thread = threading.Thread(
            target=self._enqueue_stream,
            args=(self._process.stderr, self._stderr_queue),
            daemon=True,
        )
        self._stderr_thread.start()

        self._execution_event = threading.Event()
        self._execution_event.set()
        self._shutdown_event = threading.Event()
        self._execution_start_time: float = 0.0
        self._last_used: float = time.time()
        self._killed_reason: Optional[str] = None

        self._monitor_thread = threading.Thread(target=self._monitor_resources, daemon=True)
        self._monitor_thread.start()

        self._lock = threading.Lock()

    @property
    def last_used(self) -> float:
        return self._last_used

    def _enqueue_stream(self, stream, output_queue: "queue.Queue[str]") -> None:
        try:
            for line in iter(stream.readline, ""):
                if line:
                    output_queue.put(line)
                if self._shutdown_event.is_set():
                    break
        finally:
            stream.close()

    def _monitor_resources(self) -> None:
        try:
            ps_process = psutil.Process(self._process.pid)
        except psutil.Error:
            return

        while not self._shutdown_event.is_set():
            if not self._execution_event.is_set():
                try:
                    if not ps_process.is_running():
                        break

                    mem_mb = ps_process.memory_info().rss / (1024 * 1024)
                    if mem_mb > self.max_memory_mb:
                        self._terminate("memory limit exceeded")
                        break

                    cpu_percent = ps_process.cpu_percent(interval=1)
                    if cpu_percent > self.max_cpu_percent:
                        self._terminate("cpu limit exceeded")
                        break

                    if self._execution_start_time:
                        elapsed = time.time() - self._execution_start_time
                        if elapsed > self.max_execution_time:
                            self._terminate("execution time limit exceeded")
                            break
                except psutil.NoSuchProcess:
                    break
                except psutil.AccessDenied:
                    break
            else:
                time.sleep(0.5)

    def _terminate(self, reason: str) -> None:
        if self._shutdown_event.is_set():
            return
        self._killed_reason = reason
        self._execution_event.set()
        self._shutdown_event.set()
        try:
            self._process.terminate()
        except Exception:
            pass

    def is_active(self) -> bool:
        return not self._shutdown_event.is_set() and self._process.poll() is None

    def close(self) -> None:
        if self._shutdown_event.is_set():
            return
        self._shutdown_event.set()
        try:
            if self._process.poll() is None and self._process.stdin:
                try:
                    self._process.stdin.write(json.dumps({"command": "shutdown"}) + "\n")
                    self._process.stdin.flush()
                except Exception:
                    pass
            time.sleep(0.2)
            if self._process.poll() is None:
                self._process.terminate()
        finally:
            try:
                self._process.wait(timeout=3)
            except Exception:
                self._process.kill()
            self._execution_event.set()

    def execute_code(self, code: str) -> Dict[str, object]:
        with self._lock:
            self._ensure_active()
            validation = self._validator.validate_code_security(code)
            if not validation["valid"]:
                return {
                    "success": False,
                    "error": f"Security violations detected: {', '.join(validation['violations'])}",
                    "outputs": [],
                }

            payload = json.dumps({"command": "execute", "code": code})
            try:
                assert self._process.stdin is not None
                self._process.stdin.write(payload + "\n")
                self._process.stdin.flush()
            except Exception as exc:
                self._terminate(f"failed to send code: {exc}")
                raise SandboxExecutionError("Sandbox communication error") from exc

            self._execution_event.clear()
            self._execution_start_time = time.time()

            try:
                response_line = self._wait_for_response()
            finally:
                self._execution_event.set()
                self._execution_start_time = 0.0
                self._last_used = time.time()

            if response_line is None:
                error = self._consume_stderr()
                reason = self._killed_reason or "Unknown sandbox failure"
                return {
                    "success": False,
                    "error": f"Sandbox terminated: {reason}",
                    "outputs": [{"type": "error", "text": error}] if error else [],
                }

            try:
                response = json.loads(response_line)
            except json.JSONDecodeError as exc:
                error = self._consume_stderr()
                raise SandboxExecutionError("Invalid response from sandbox") from exc

            outputs = []
            stdout = response.get("stdout", "")
            stderr = response.get("stderr", "")

            if stdout:
                outputs.append({"type": "text", "text": stdout})
            if stderr:
                outputs.append({"type": "warning", "text": stderr})
            if not response.get("success") and response.get("error"):
                outputs.append({"type": "error", "text": response.get("error")})

            return {
                "success": bool(response.get("success")),
                "outputs": outputs,
                "execution_result": {
                    "stdout": stdout,
                    "stderr": stderr,
                    "returncode": 0 if response.get("success") else 1,
                },
                "execution_time": response.get("execution_time"),
                "plots": response.get("plots", []),
                "error": response.get("error"),
            }

    def _wait_for_response(self) -> Optional[str]:
        deadline = time.time() + self.max_execution_time + 5
        while time.time() < deadline:
            try:
                return self._stdout_queue.get(timeout=0.5)
            except queue.Empty:
                if self._process.poll() is not None:
                    break
        self._terminate("execution timed out waiting for response")
        return None

    def _ensure_active(self) -> None:
        if self._shutdown_event.is_set() or self._process.poll() is not None:
            raise SandboxExecutionError("Sandbox session has been terminated")

    def _consume_stderr(self) -> str:
        errors = []
        while True:
            try:
                errors.append(self._stderr_queue.get_nowait())
            except queue.Empty:
                break
        return "".join(errors).strip()


class PersistentSandboxManager:
    """Manager maintaining persistent sandbox sessions per user + dataset."""

    def __init__(self, idle_ttl_seconds: int = 600) -> None:
        self._sessions: Dict[Tuple[str, str], PersistentSandboxSession] = {}
        self._lock = threading.Lock()
        self._ttl = idle_ttl_seconds

    def get_or_create_session(
        self,
        user_id: str,
        source_id: str,
        data_path: str,
        *,
        max_execution_time: int,
        max_memory_mb: int,
        max_cpu_percent: int,
    ) -> PersistentSandboxSession:
        key = (user_id or "default", source_id)
        self._cleanup_expired()
        with self._lock:
            session = self._sessions.get(key)
            if session and session.is_active():
                return session
            if session:
                session.close()
            session = PersistentSandboxSession(
                data_path,
                key[0],
                max_execution_time=max_execution_time,
                max_memory_mb=max_memory_mb,
                max_cpu_percent=max_cpu_percent,
            )
            self._sessions[key] = session
            return session

    def reset_session(self, user_id: str, source_id: str) -> None:
        key = (user_id or "default", source_id)
        with self._lock:
            session = self._sessions.pop(key, None)
            if session:
                session.close()

    def shutdown_all(self) -> None:
        with self._lock:
            sessions = list(self._sessions.values())
            self._sessions.clear()
        for session in sessions:
            session.close()

    def _cleanup_expired(self) -> None:
        with self._lock:
            stale_keys = [
                key
                for key, session in self._sessions.items()
                if time.time() - session.last_used > self._ttl or not session.is_active()
            ]
        for key in stale_keys:
            self.reset_session(*key)


_GLOBAL_MANAGER: Optional[PersistentSandboxManager] = None
_GLOBAL_LOCK = threading.Lock()


def get_persistent_sandbox_manager() -> PersistentSandboxManager:
    global _GLOBAL_MANAGER
    if _GLOBAL_MANAGER is None:
        with _GLOBAL_LOCK:
            if _GLOBAL_MANAGER is None:
                _GLOBAL_MANAGER = PersistentSandboxManager()
    return _GLOBAL_MANAGER
