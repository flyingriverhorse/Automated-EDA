# EDA Security Implementation Guide

## Overview

This document describes the comprehensive security implementation for custom code execution in the EDA (Exploratory Data Analysis) system. The system ensures that users can only perform legitimate EDA operations on their assigned datasets while preventing access to system resources, other databases, and external networks.

## Security Architecture

### 1. Code Validation Layer
- **Static Analysis**: Code is analyzed using regex patterns and AST parsing
- **Pattern Detection**: Identifies dangerous imports, function calls, and operations
- **Whitelist Approach**: Only approved libraries and operations are allowed

### 2. Execution Isolation
- **Process Sandbox**: Code runs in isolated subprocess with restricted environment
- **Resource Limits**: CPU time and memory constraints prevent resource abuse
- **File System Restrictions**: Only the specific dataset file can be accessed

### 3. Network & Database Blocking
- **Import Restrictions**: Database and network libraries are blocked at import level
- **Pattern Matching**: Dangerous connection patterns are detected and prevented

## Allowed Libraries and Operations

### ✅ **Permitted (EDA-focused)**
```python
# Data manipulation and analysis
import pandas as pd
import numpy as np

# Visualization
import matplotlib.pyplot as plt
import seaborn as sns

# Statistical analysis
import scipy
import sklearn

# Utilities
import warnings
import math
import datetime
import json
import re
```

### ❌ **Blocked (Security risks)**
```python
# System access
import os
import sys
import subprocess

# Network operations
import urllib
import requests
import socket

# Database operations
import sqlite3
import psycopg2
import pymongo
import mysql

# File operations (write mode)
open("file.txt", "w")
with open("file.txt", "w") as f:
    f.write("data")

# Code execution
eval("code")
exec("code")
```

## Implementation Details

### 1. SimplifiedSandbox Class

Located in `core/eda/security/simplified_sandbox.py`, this class provides:

```python
class SimplifiedSandbox:
    def validate_code_security(self, code: str) -> Dict[str, Any]:
        """Validates code against security patterns"""
        
    def execute_code_safely(self, code: str, data_path: str) -> Dict[str, Any]:
        """Executes code in secure subprocess"""
```

### 2. Security Validation Patterns

The system uses regex patterns to detect dangerous operations:

```python
DANGEROUS_PATTERNS = [
    r'import\s+os\b',           # OS module access
    r'import\s+subprocess\b',   # System commands
    r'import\s+socket\b',       # Network access
    r'import\s+sqlite3\b',      # Database access
    r'\.system\s*\(',          # System calls
    r'open\s*\(\s*[\'"].*[\'"].*[\'"]w',  # File writing
    # ... more patterns
]
```

### 3. Process Isolation

Code execution happens in a subprocess with:

- **Restricted Environment**: Limited environment variables
- **Timeout Protection**: Maximum execution time enforced
- **Memory Limits**: Resource usage constraints
- **Working Directory**: Execution in temporary directory

### 4. Dataset Access Control

Users can only access their specific dataset:

```python
# Only this specific file path is accessible
df = pd.read_csv(r'{data_path}')  # Allowed

# All other file operations are blocked
open("other_file.txt")  # Blocked
```

## Frontend Security Notice

The custom analysis interface displays a security notice informing users about:

- **Secure Environment**: Code runs in isolated sandbox
- **Available Libraries**: Only EDA libraries are permitted
- **Access Restrictions**: No file system, network, or database access
- **Dataset Scope**: Only current dataset is accessible

## Testing and Validation

### Security Test Suite

```python
# Test dangerous code blocking
dangerous_codes = [
    "import os; os.system('echo hello')",
    "import subprocess; subprocess.run(['ls'])",
    "import sqlite3; conn = sqlite3.connect('test.db')",
    "with open('file.txt', 'w') as f: f.write('hack')"
]

# Test safe EDA code
safe_codes = [
    "df.head()",
    "df.describe()",
    "import pandas as pd; df.mean()",
    "import matplotlib.pyplot as plt; plt.figure()"
]
```

### Validation Results

- ✅ All dangerous operations are blocked
- ✅ All legitimate EDA operations work correctly
- ✅ Resource limits prevent abuse
- ✅ Only assigned dataset is accessible

## Usage Examples

### Safe EDA Code Examples

```python
# Dataset exploration
print("Dataset shape:", df.shape)
print(df.info())
print(df.describe())

# Data visualization
import matplotlib.pyplot as plt
import seaborn as sns

plt.figure(figsize=(10, 6))
sns.heatmap(df.corr(), annot=True)
plt.title("Correlation Heatmap")
plt.show()

# Statistical analysis
from scipy import stats
print("Normality test:", stats.normaltest(df['numeric_column']))

# Missing value analysis
print("Missing values:", df.isnull().sum())
```

### Blocked Code Examples

```python
# ❌ These will be blocked with security violations:

import os
os.system("rm -rf /")  # System access

import requests
requests.get("http://malicious.com")  # Network access

import sqlite3
conn = sqlite3.connect("other_database.db")  # Database access

with open("/etc/passwd", "r") as f:  # File system access
    sensitive_data = f.read()
```

## Security Benefits

1. **Data Isolation**: Users cannot access data outside their assigned dataset
2. **System Protection**: No system commands or file system access
3. **Network Security**: No external network connections possible
4. **Database Security**: No connections to other databases
5. **Resource Protection**: CPU and memory limits prevent abuse
6. **Code Validation**: Static analysis prevents dangerous patterns

## Integration

The security sandbox is integrated into the EDA service:

```python
# In AdvancedEDAService.execute_code()
if context == "custom_notebook":
    from core.eda.security import create_sandbox
    sandbox = create_sandbox("simplified", max_execution_time=30)
    return sandbox.execute_code_safely(code, str(data_file_path))
```

This ensures all custom user code goes through the security sandbox while generated analysis code continues to use the standard execution path.

## Monitoring and Logging

- Security violations are logged with details
- Failed attempts are tracked
- Resource usage is monitored
- Timeout events are recorded

The system provides comprehensive security while maintaining full EDA functionality for legitimate use cases.