# 🔒 EDA Security Module Documentation

## Overview

This module provides secure sandbox environments for executing user-provided Python code within the MLops EDA system. The primary goal is to allow users to perform comprehensive data analysis on their assigned datasets while preventing unauthorized access to system resources, other databases, files, or network connections.

## 🏗️ Architecture

```
User Code Input
      ↓
Code Validation (Regex Patterns)
      ↓
Sandbox Environment Creation
      ↓
Process Isolation (subprocess)
      ↓
Library Access Control (pandas wrapper)
      ↓
Built-in Function Override (open, etc.)
      ↓
Secure Execution
      ↓
Output Collection & Return
```

## 📂 Module Structure

```
core/eda/security/
├── __init__.py                 # Module initialization and exports
├── simplified_sandbox.py       # Main secure sandbox implementation
├── code_sandbox.py            # Legacy sandbox (for backwards compatibility)
└── README.md                  # This documentation file
```

## 🔧 Components

### 1. SimplifiedSandbox Class

**Location**: `simplified_sandbox.py`
**Purpose**: Primary secure sandbox for EDA code execution

#### Key Features:
- **Regex-based Code Validation**: Blocks dangerous import statements and function calls
- **Process Isolation**: Executes code in separate subprocess with timeout
- **Library Access Control**: Wraps pandas file reading functions to restrict access
- **Built-in Function Override**: Replaces dangerous built-ins like `open()`
- **Environment Control**: Limits available modules and system access

#### Security Layers:

1. **Pre-execution Validation**:
   - Scans code for dangerous patterns using regex
   - Blocks before execution if violations found
   - Patterns include: `import os`, `subprocess.call()`, `eval()`, etc.

2. **Pandas File Access Control**:
   - Overrides `pd.read_csv()`, `pd.read_excel()`, `pd.read_json()`, `pd.read_parquet()`
   - Only allows reading from the assigned dataset file
   - Blocks access to any other CSV/Excel files on the system

3. **Built-in Function Override**:
   - Replaces the built-in `open()` function with secure version
   - Only allows reading the assigned dataset file
   - Blocks access to all other files (config files, system files, etc.)

4. **Process Isolation**:
   - Runs code in separate subprocess with resource limits
   - 30-second timeout by default
   - Isolated environment prevents system contamination

### 2. CodeSandbox Class (Legacy)

**Location**: `code_sandbox.py`
**Purpose**: Original sandbox implementation (kept for backwards compatibility)

#### Features:
- More complex AST-based validation
- Thread-based and process-based execution options
- Similar security restrictions but with different implementation approach

## 🚫 Security Restrictions

### Blocked Operations

#### 1. System Access
```python
# ❌ BLOCKED
import os
os.system('dir')
os.listdir('.')

import sys
sys.exit()

import subprocess
subprocess.call(['pip', 'install', 'requests'])
```

#### 2. Package Installation (All Package Managers)
```python
# ❌ BLOCKED - Traditional pip
pip install requests
python -m pip install beautifulsoup4

# ❌ BLOCKED - Modern package managers
uv add fastapi
poetry add django
conda install scipy
mamba install numpy
pipenv install flask

# ❌ BLOCKED - Jupyter notebook magic commands
!pip install requests
!conda install pandas
!uv add streamlit
%%bash
pip install malicious_package
```

#### 3. Network Access
```python
# ❌ BLOCKED
import requests
requests.get('https://example.com')

import urllib
urllib.request.urlopen('https://google.com')

import socket
socket.create_connection(('google.com', 80))

import http.client
conn = http.client.HTTPConnection('example.com')
```

#### 4. Database Access
```python
# ❌ BLOCKED
import sqlite3
conn = sqlite3.connect('database.db')

import psycopg2
conn = psycopg2.connect(host='localhost')

import pymongo
client = pymongo.MongoClient()
```

#### 5. File System Access
```python
# ❌ BLOCKED - File access beyond assigned dataset
with open('config.py', 'r') as f:
    content = f.read()

# ❌ BLOCKED - Pandas access to other files  
pd.read_csv('other_dataset.csv')

# ❌ BLOCKED - System file access
with open('C:/Windows/System32/hosts', 'r') as f:
    content = f.read()

# ❌ BLOCKED - Advanced file system access
import pathlib
pathlib.Path('/etc/passwd').read_text()

import tempfile
tempfile.mkdtemp()

import shutil
shutil.copy('source.txt', 'dest.txt')
```

#### 6. Code Injection & Dynamic Execution
```python
# ❌ BLOCKED
eval("malicious_code")
exec("import os; os.system('harmful_command')")
__import__('os')
compile('dangerous_code', '<string>', 'exec')

# ❌ BLOCKED - Advanced bypass attempts
import importlib
importlib.import_module('os')

# ❌ BLOCKED - Builtins manipulation
__builtins__["eval"]("print('bypassed')")
globals()["__builtins__"]
getattr(__builtins__, 'eval')
```

#### 7. Environment & System Manipulation
```python
# ❌ BLOCKED
import os
os.environ["MALICIOUS_VAR"] = "value"

import sys
sys.path.append("/malicious/path")
sys.modules["os"] = malicious_module

# ❌ BLOCKED - Introspection functions
vars()  # Access variables
dir()   # Directory listing
globals()  # Global namespace
locals()   # Local namespace
```

### ✅ Allowed Operations

#### 1. Data Analysis Libraries
```python
# ✅ ALLOWED
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.stats
from sklearn.preprocessing import StandardScaler
```

#### 2. Dataset Operations
```python
# ✅ ALLOWED - Operations on assigned dataset
print(df.shape)
print(df.info())
print(df.describe())
df.head()
df.groupby('column').mean()
```

#### 3. Visualizations
```python
# ✅ ALLOWED - Plotting and visualization
plt.figure(figsize=(10, 6))
plt.hist(df['column'])
plt.show()

sns.scatterplot(data=df, x='col1', y='col2')
plt.show()
```

#### 4. Statistical Analysis
```python
# ✅ ALLOWED - Statistical operations
from scipy import stats
correlation = df.corr()
stats.normaltest(df['numeric_column'])
```

## 🔌 Integration Points

### 1. API Endpoints

#### Custom Analysis Endpoint
```python
# Route: /api/execute-custom-code/{source_id}
# File: core/eda/advanced_eda/routes.py

@router.post("/api/execute-custom-code/{source_id}")
async def execute_custom_code(source_id: str, request_data: CustomCodeRequest):
    eda_service = AdvancedEDAService(service.session)
    result = await eda_service.execute_code(
        source_id=source_id,
        code=request_data.code,
        context="custom_notebook"  # Triggers sandbox usage
    )
```

#### LLM Code Execution Endpoint
```python
# Route: /api/llm/execute-code
# File: core/eda/advanced_eda/routes.py

@router.post("/api/llm/execute-code")
async def execute_llm_code(request_data: dict):
    # Frontend must pass context="custom_notebook" to use sandbox
    result = await eda_service.execute_code(source_id, code, context)
```

### 2. Service Integration

```python
# File: core/eda/advanced_eda/services.py

async def execute_code(self, source_id: str, code: str, context: str = None):
    if context == "custom_notebook":
        # Use secure sandbox for custom code execution
        from core.eda.security import create_sandbox
        sandbox = create_sandbox("simplified", max_execution_time=30)
        return sandbox.execute_code_safely(code, str(data_file_path))
```

### 3. Frontend Integration

#### Custom Analysis Tab
```javascript
// File: static/js/eda/custom_analysis.js
fetch(`/advanced-eda/api/execute-custom-code/${currentSourceId}`, {
    method: 'POST',
    body: JSON.stringify({ code: code })
})
```

> **Note:** The standalone LLM tab has been retired. LLM-driven execution examples now live alongside the custom analysis flows.

## 🔧 Usage Examples

### Creating a Sandbox Instance

```python
from core.eda.security import create_sandbox

# Create simplified sandbox (recommended)
sandbox = create_sandbox("simplified", max_execution_time=30)

# Create legacy sandbox (backwards compatibility)
sandbox = create_sandbox("process", max_execution_time=30)
```

### Executing Code Safely

```python
# User's EDA code
user_code = '''
print("Dataset Shape:", df.shape)
print("First 5 rows:")
print(df.head())
'''

# Execute with dataset access
result = sandbox.execute_code_safely(user_code, '/path/to/dataset.csv')

# Check results
if result['success']:
    for output in result['outputs']:
        print(output['text'])
else:
    print(f"Error: {result['error']}")
```

### Response Format

```python
{
    "success": True,
    "outputs": [
        {
            "type": "text", 
            "text": "Dataset Shape: (1000, 10)\\nFirst 5 rows:\\n..."
        }
    ],
    "execution_result": {
        "stdout": "Dataset Shape: (1000, 10)...",
        "stderr": "",
        "returncode": 0
    }
}
```

## 🧪 Testing Security

### Test Suite Location
- Security tests: `test_security_vulnerabilities.py`
- Pandas tests: `test_pandas_vulnerability.py`

### Running Security Tests

```bash
# Test general security
python test_security_vulnerabilities.py

# Test pandas file access control
python test_pandas_vulnerability.py
```

### Security Violation Examples

```python
# These should all be blocked:

# 1. File System Violation
try:
    with open('config.py', 'r') as f:
        print(f.read())
except Exception as e:
    print("✅ Blocked:", e)

# 2. Database Violation  
try:
    import sqlite3
    conn = sqlite3.connect('db.sqlite')
except Exception as e:
    print("✅ Blocked:", e)

# 3. Network Violation
try:
    import requests
    requests.get('https://httpbin.org')
except Exception as e:
    print("✅ Blocked:", e)
```

## ⚡ Performance Considerations

### Resource Limits
- **Execution Timeout**: 30 seconds (configurable)
- **Memory**: Inherited from subprocess (can be limited via OS)
- **CPU**: Single subprocess, no parallelization

### Optimization Tips
1. **Pre-filter Code**: Regex validation happens before subprocess creation
2. **Environment Reuse**: Same Python environment reduces startup time
3. **Output Buffering**: Collect all outputs before returning

## 🔄 Maintenance

### Adding New Security Patterns

```python
# In SimplifiedSandbox.DANGEROUS_PATTERNS, add new regex:
DANGEROUS_PATTERNS = [
    # Existing patterns...
    r'import\s+new_dangerous_module\b',  # Block new dangerous module
    r'\.dangerous_method\s*\(',          # Block dangerous method calls
]
```

### Updating Allowed Libraries

```python
# In sandbox execution script, add new safe imports:
execution_script = f'''
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import new_safe_library  # Add new safe library
# ... rest of script
'''
```

### Monitoring and Logging

```python
# Security violations are logged:
logger = logging.getLogger(__name__)

# In validation:
if violations:
    logger.warning(f"Security violations detected: {violations}")
    
# In execution:
logger.info(f"Executing code for source_id: {source_id}")
```

## 🚨 Critical Security Notes

### 1. File Access Control
- **CRITICAL**: Both pandas wrapper AND built-in open() override are required
- Pandas functions can bypass built-in open() restrictions
- Built-in open() override catches direct file access attempts

### 2. Process Isolation
- Code runs in separate subprocess to prevent contamination
- Timeout prevents infinite loops or resource exhaustion
- Environment variables are passed to maintain library access

### 3. Pattern Matching Limitations
- Regex patterns can't catch all possible code injection attempts
- Focus on blocking import statements and known dangerous functions
- Process isolation provides additional security layer

### 4. Library Access
- Only EDA-related libraries should be available in execution environment
- Regularly audit available imports in sandbox execution script
- Monitor for new potentially dangerous libraries

## 🔮 Future Enhancements

### Potential Improvements
1. **AST-based Validation**: More sophisticated code analysis
2. **Resource Limits**: Memory and CPU usage constraints  
3. **Logging Enhancement**: Detailed security event logging
4. **Whitelist Approach**: Explicit approval of allowed operations
5. **Dynamic Policy Updates**: Runtime security policy modifications

### Known Limitations
1. **Regex Limitations**: Some obfuscated code might bypass pattern matching
2. **Library Evolution**: New dangerous libraries require pattern updates
3. **Performance Impact**: Subprocess creation adds execution overhead
4. **Environment Dependencies**: Requires proper Python environment setup

---

## 📞 Support

For security concerns or questions:
1. Check existing test files for examples
2. Review security patterns in `SimplifiedSandbox.DANGEROUS_PATTERNS`
3. Test new code patterns with the security test suite
4. Update documentation when adding new security features

**Remember**: Security is layered - no single mechanism is foolproof. The combination of validation, process isolation, and library access control provides comprehensive protection.