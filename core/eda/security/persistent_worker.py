"""Persistent sandbox worker process for custom analysis sessions."""

from __future__ import annotations

import ast
import base64
import builtins
import json
import os
import sys
import time
import traceback
from contextlib import redirect_stdout, redirect_stderr
from io import BytesIO, StringIO
from typing import Any, Dict, List, Set

import collections
import datetime as datetime_module
import functools
import itertools
import math
import pandas as pd
import numpy as np
import matplotlib
import re
import statistics

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import seaborn as sns  # noqa: E402


def _normalize_path(path: str) -> str:
    return os.path.normcase(os.path.abspath(path))


def _paths_match(path_a: str, path_b: str) -> bool:
    try:
        if os.path.exists(path_a) and os.path.exists(path_b):
            try:
                return os.path.samefile(path_a, path_b)
            except OSError:
                pass
    except Exception:
        pass
    return _normalize_path(path_a) == _normalize_path(path_b)

ALLOWED_MODULES: Set[str] = {
    "pandas",
    "numpy",
    "matplotlib",
    "seaborn",
    "math",
    "statistics",
    "collections",
    "itertools",
    "functools",
    "datetime",
    "re",
}

BANNED_CALL_NAMES: Set[str] = {
    "eval",
    "exec",
    "compile",
    "globals",
    "locals",
    "vars",
    "__import__",
    "input",
}

BANNED_ATTRIBUTE_NAMES: Set[str] = {
    "__subclasses__",
    "__mro__",
    "__bases__",
    "__globals__",
    "__getattribute__",
    "__setattr__",
    "__delattr__",
    "__reduce__",
    "__reduce_ex__",
    "__setstate__",
    "__getstate__",
}

BANNED_NAME_LITERALS: Set[str] = {
    "__subclasses__",
    "__globals__",
    "__getattribute__",
    "__setattr__",
    "__delattr__",
    "__builtins__",
}


def _build_safe_builtins(allowed_modules: Set[str], secure_open):
    original_import = builtins.__import__

    def safe_import(name, globals=None, locals=None, fromlist=(), level=0):
        module_root = name.split(".")[0]
        if module_root in allowed_modules:
            return original_import(name, globals, locals, fromlist, level)
        raise ImportError(f"Import '{name}' is not permitted in the sandbox")

    safe_names = {
        "abs": abs,
        "all": all,
        "any": any,
        "bool": bool,
        "complex": complex,
        "dict": dict,
        "enumerate": enumerate,
        "filter": filter,
        "float": float,
        "int": int,
        "len": len,
        "list": list,
        "map": map,
        "max": max,
        "min": min,
        "pow": pow,
        "print": print,
        "range": range,
        "round": round,
        "set": set,
        "sorted": sorted,
        "str": str,
        "sum": sum,
        "tuple": tuple,
        "zip": zip,
        "Exception": Exception,
        "ValueError": ValueError,
        "TypeError": TypeError,
        "RuntimeError": RuntimeError,
        "NameError": NameError,
        "open": secure_open,
        "__import__": safe_import,
    }

    return safe_names


def _write_json(message: Dict[str, Any]) -> None:
    sys.stdout.write(json.dumps(message) + "\n")
    sys.stdout.flush()


class SecurityViolation(Exception):
    """Raised when submitted code violates sandbox security policies."""


def _validate_code_ast(code: str) -> None:
    try:
        tree = ast.parse(code, mode="exec")
    except SyntaxError:
        # Let the normal execution pipeline raise the syntax error for consistency
        return

    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                root = alias.name.split(".")[0]
                if root not in ALLOWED_MODULES:
                    raise SecurityViolation(f"Import of '{alias.name}' is not permitted in the sandbox")

        elif isinstance(node, ast.ImportFrom):
            module = node.module or ""
            root = module.split(".")[0]
            if not module or root not in ALLOWED_MODULES:
                raise SecurityViolation(f"Import from '{module or '<relative>'}' is not permitted in the sandbox")

        elif isinstance(node, ast.Call):
            func = node.func
            if isinstance(func, ast.Name) and func.id in BANNED_CALL_NAMES:
                raise SecurityViolation(f"Call to '{func.id}' is not permitted in the sandbox")
            if isinstance(func, ast.Attribute) and func.attr in BANNED_ATTRIBUTE_NAMES:
                raise SecurityViolation(f"Call to attribute '{func.attr}' is not permitted in the sandbox")

        elif isinstance(node, ast.Attribute):
            if node.attr in BANNED_ATTRIBUTE_NAMES:
                raise SecurityViolation(f"Access to attribute '{node.attr}' is not permitted in the sandbox")

        elif isinstance(node, ast.Name):
            if node.id in BANNED_NAME_LITERALS:
                raise SecurityViolation(f"Usage of '{node.id}' is not permitted in the sandbox")


def main() -> None:
    if len(sys.argv) < 3:
        _write_json({"success": False, "error": "Sandbox worker missing arguments"})
        return

    data_path = sys.argv[1]
    max_execution_time = int(sys.argv[2])
    abs_data_path = os.path.abspath(data_path)

    os.environ.setdefault("DISPLAY", "")

    if not os.path.exists(abs_data_path):
        _write_json({"success": False, "error": f"Dataset path not found: {abs_data_path}"})
        return

    original_open = builtins.open

    # Preserve original pandas readers for secure wrappers
    _original_read_csv = pd.read_csv
    _original_read_excel = getattr(pd, "read_excel", None)
    _original_read_json = getattr(pd, "read_json", None)
    _original_read_parquet = getattr(pd, "read_parquet", None)

    class FileAccessError(Exception):
        pass

    def secure_open(filepath, mode="r", *args, **kwargs):
        requested = os.fspath(filepath)
        if "r" in mode and _paths_match(requested, abs_data_path):
            return original_open(filepath, mode, *args, **kwargs)
        raise FileAccessError(f"File access denied: {filepath}")

    def _ensure_allowed_path(path_like: Any) -> None:
        if isinstance(path_like, (str, os.PathLike)):
            requested_path = os.fspath(path_like)
            if not _paths_match(requested_path, abs_data_path):
                raise FileAccessError(f"File access denied: {path_like}")

    def secure_read_csv(filepath_or_buffer, *args, **kwargs):
        _ensure_allowed_path(filepath_or_buffer)
        return _original_read_csv(filepath_or_buffer, *args, **kwargs)

    def secure_read_excel(filepath_or_buffer, *args, **kwargs):
        _ensure_allowed_path(filepath_or_buffer)
        if _original_read_excel is None:
            raise FileAccessError("Excel support unavailable in sandbox environment")
        return _original_read_excel(filepath_or_buffer, *args, **kwargs)

    def secure_read_json(filepath_or_buffer, *args, **kwargs):
        _ensure_allowed_path(filepath_or_buffer)
        if _original_read_json is None:
            raise FileAccessError("JSON support unavailable in sandbox environment")
        return _original_read_json(filepath_or_buffer, *args, **kwargs)

    def secure_read_parquet(filepath_or_buffer, *args, **kwargs):
        _ensure_allowed_path(filepath_or_buffer)
        if _original_read_parquet is None:
            raise FileAccessError("Parquet support unavailable in sandbox environment")
        return _original_read_parquet(filepath_or_buffer, *args, **kwargs)

    # Override pandas file readers and built-in open
    pd.read_csv = secure_read_csv
    if _original_read_excel:
        pd.read_excel = secure_read_excel  # type: ignore[assignment]
    if _original_read_json:
        pd.read_json = secure_read_json  # type: ignore[assignment]
    if _original_read_parquet:
        pd.read_parquet = secure_read_parquet  # type: ignore[assignment]

    try:
        df = _original_read_csv(abs_data_path)
    except Exception as exc:
        _write_json({"success": False, "error": f"Failed to load dataset: {exc}"})
        return

    safe_builtins = _build_safe_builtins(ALLOWED_MODULES, secure_open)

    # Execution environment shared across requests
    exec_globals: Dict[str, Any] = {
        "__builtins__": safe_builtins,
        "pd": pd,
        "np": np,
        "plt": plt,
        "sns": sns,
        "df": df,
        "FileAccessError": FileAccessError,
        "SecurityViolation": SecurityViolation,
        "math": math,
        "statistics": statistics,
        "datetime": datetime_module,
        "re": re,
        "collections": collections,
        "itertools": itertools,
        "functools": functools,
    }

    last_code_time = time.time()

    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue

        try:
            payload = json.loads(line)
        except json.JSONDecodeError as exc:
            _write_json({"success": False, "error": f"Invalid payload: {exc}"})
            continue

        command = payload.get("command")

        if command == "shutdown":
            _write_json({"success": True, "message": "Sandbox shutdown"})
            break

        if command == "heartbeat":
            _write_json({"success": True, "message": "alive", "last_code_time": last_code_time})
            continue

        if command != "execute":
            _write_json({"success": False, "error": f"Unknown command: {command}"})
            continue

        code = payload.get("code", "")
        if not isinstance(code, str):
            _write_json({"success": False, "error": "Invalid code payload"})
            continue

        try:
            _validate_code_ast(code)
        except SecurityViolation as sec_exc:
            _write_json(
                {
                    "success": False,
                    "stdout": "",
                    "stderr": "",
                    "error": str(sec_exc),
                    "execution_time": 0.0,
                    "plots": [],
                }
            )
            continue

        stdout_buffer = StringIO()
        stderr_buffer = StringIO()
        success = True
        error_message = ""
        plots: List[str] = []

        start = time.time()
        try:
            with redirect_stdout(stdout_buffer), redirect_stderr(stderr_buffer):
                exec(code, exec_globals)
        except Exception as exc:  # noqa: PERF203 - sandbox safety
            success = False
            error_message = "".join(traceback.format_exception(type(exc), exc, exc.__traceback__))
        finally:
            execution_duration = time.time() - start
            last_code_time = time.time()

        try:
            for fig_num in plt.get_fignums():
                fig = plt.figure(fig_num)
                buffer = BytesIO()
                fig.savefig(buffer, format="png", bbox_inches="tight")
                buffer.seek(0)
                plots.append(base64.b64encode(buffer.getvalue()).decode("utf-8"))
                plt.close(fig)
        except Exception as plot_exc:  # noqa: PERF203 - ensure failure is surfaced but non-fatal
            stderr_buffer.write(f"Plot capture failed: {plot_exc}\n")
            traceback.print_exc(file=sys.stderr)
        finally:
            plt.close("all")

        response: Dict[str, Any] = {
            "success": success,
            "stdout": stdout_buffer.getvalue(),
            "stderr": stderr_buffer.getvalue(),
            "error": error_message,
            "execution_time": execution_duration,
            "plots": plots,
        }

        _write_json(response)

    # Ensure clean exit
    sys.stdout.flush()
    sys.stderr.flush()


if __name__ == "__main__":
    main()
