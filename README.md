# Advanced Data Analysis & LLM Integration

[![CI](https://github.com/flyingriverhorse/Automated-EDA/actions/workflows/ci.yml/badge.svg)](https://github.com/flyingriverhorse/Automated-EDA/actions/workflows/ci.yml)
[![Release](https://img.shields.io/github/v/tag/flyingriverhorse/Automated-EDA?label=release&color=blue)](https://github.com/flyingriverhorse/Automated-EDA/releases)
[![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)](https://python.org)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.104+-green.svg)](https://fastapi.tiangolo.com)
[![License](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Code Style](https://img.shields.io/badge/Code%20Style-Black-black.svg)](https://github.com/psf/black)

A comprehensive EDA platform built with FastAPI that combines advanced exploratory data analysis (EDA), secure code execution, and intelligent LLM integrations for modern data science workflows.

## 🚀 Key Features

### 📊 **Advanced Data Analysis**
- **Multi-format Data Ingestion**: Support for CSV, JSON, Excel, Parquet, and database connections
- **Intelligent EDA Engine**: Automated exploratory data analysis with domain-specific insights
- **Granular Analysis Components**: Modular analysis for categorical, numerical, geospatial, and text data
- **Real-time Data Quality Monitoring**: Comprehensive data profiling and quality metrics

### 🔒 **Secure Code Execution**
- **Sandboxed Environment**: Isolated code execution with resource limits and security controls
- **Pattern-based Security**: Static analysis and runtime protection against malicious operations
- **Process Isolation**: Multi-process architecture for safe user code execution
- **Resource Management**: CPU and memory limits to prevent abuse

### 🤖 **LLM Integration**
- **Multi-Provider Support**: OpenAI, Anthropic Claude, DeepSeek, and local models (Ollama)
- **Intelligent Model Switching**: Automatic model selection based on task type (code, math, analysis)
- **Context-Aware Queries**: Data-driven context injection for relevant AI responses
- **Specialized Models**: DeepSeek Coder for programming tasks, DeepSeek Math for analysis

### 👥 **Enterprise Features**
- **JWT Authentication**: Secure token-based authentication with role-based access control
- **Multi-user Support**: User management with admin dashboard and permissions
- **Async Architecture**: Modern FastAPI with async/await for high performance
- **Database Flexibility**: Support for SQLite, PostgreSQL with async operations

## 🏗️ Architecture

```
MLOps Platform
├── 🔐 Authentication Layer (JWT + RBAC)
├── 📊 Data Ingestion Engine
├── 🔍 EDA Analysis Engine
├── 🤖 LLM Service Layer
├── 🛡️ Security Sandbox
└── 👥 Admin Dashboard
```

### Core Components

- **FastAPI Backend**: Modern async web framework with automatic API documentation
- **Advanced EDA**: Domain-specific analysis with granular components
- **LLM Router**: Intelligent routing to appropriate AI models
- **Security Layer**: Multi-layered protection for code execution
- **Data Pipeline**: Robust ingestion and processing workflow

## Web Experience

- **Templates**: Jinja2 views in `templates/` power data ingestion, EDA dashboards, admin pages, and login flows
- **Static Assets**: Modular JavaScript and CSS bundles in `static/` for ingestion, preview, chat, and shared UI components
- **Page Routing**: `core.pages.routes` wires HTML routes, while `core.templates` centralizes template setup
- **Frontend Security**: `core.auth.page_security` enforces per-page access and dynamic context rendering

## 🧩 Module Overview

- **Authentication (`core/auth`)**: JWT issuance, OAuth2 dependencies, page security helpers, and template routes
- **Admin (`core/admin`)**: Async services for system stats, ingestion oversight, maintenance utilities, and admin APIs
- **Data Ingestion (`core/data_ingestion`)**: Async upload pipeline, metadata management, schema validation, and router APIs
- **EDA (`core/eda`)**: Preview services, text analytics, sandboxed execution, and advanced analysis orchestrators
- **LLM (`core/llm`)**: Provider abstractions (OpenAI, Claude, DeepSeek, local), context builders, and chat endpoints
- **Database (`core/database`)**: Async engine factory, migration utilities, repository helpers, and model declarations
- **Exceptions & Middleware**: `core.exceptions.handlers` plus `middleware/` for structured logging and global error handling
- **Utilities (`core/utils`)**: File helpers, logging adapters, maintenance routines, and audit logging

## �📋 Prerequisites

- **Python 3.10+** (Required for geospatial dependencies)
- **Node.js 16+** (For frontend build tools)
- **PostgreSQL 12+** (Optional, SQLite included)
- **Docker** (Optional, for containerized deployment)

## ⚡ Quick Start

### 1. Clone Repository
```bash
git clone https://github.com/flyingruverhorse/Automated-EDA.git
cd Automated-EDA
```

### 2. Environment Setup
```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements-fastapi.txt
```

### 3. Configuration
```bash
# Copy environment template
cp .env.example .env

# Configure your settings
# Required: Set SECRET_KEY, database credentials, LLM API keys
```

### 4. Database Setup
```bash
# Initialize database
alembic upgrade head

# Create admin user (optional)
python -c "from core.auth.auth_core import create_dummy_users; create_dummy_users()"
```

### 5. Run Application
```bash
# Development server
python run_fastapi.py

# Production server
uvicorn main:app --host 0.0.0.0 --port 8000 --workers 4
```

The application will be available at `http://localhost:8000`

## 🔧 Configuration

### Environment Variables

#### Core Settings
```bash
# Application
SECRET_KEY=your-secret-key-here
DEBUG=false
ENVIRONMENT=production

# Database
DATABASE_TYPE=postgresql  # or sqlite
DB_HOST=localhost
DB_NAME=db
DB_USER=username
DB_PASSWORD=password
```

#### LLM Configuration
```bash
# OpenAI
OPENAI_API_KEY=your-openai-key
OPENAI_DEFAULT_MODEL=gpt-3.5-turbo

# DeepSeek (Code-specialized)
DEEPSEEK_API_KEY=your-deepseek-key
DEEPSEEK_CODE_MODEL=deepseek-coder
DEEPSEEK_MATH_MODEL=deepseek-math

# Anthropic Claude
ANTHROPIC_API_KEY=your-claude-key
CLAUDE_DEFAULT_MODEL=claude-3-haiku-20240307

# Local LLM (Ollama)
LOCAL_LLM_URL=http://localhost:11434
LOCAL_LLM_MODEL=llama2
```

## ⚙️ Environment Profiles

- **Config Management**: `config.py` uses `pydantic-settings` with cached `get_settings()` accessor and environment validation
- **Profiles**: `DevelopmentSettings`, `ProductionSettings`, and `TestingSettings` toggle middleware, logging, caching, and debugging defaults
- **Feature Flags**: Data lineage, schema drift, retention, and LLM behaviour exposed as `ENABLE_*` toggles
- **Helper APIs**: Utility methods for pandas configuration, database URLs, logging setup, and ML feature sizing

## 📚 API Documentation

### Interactive Documentation
- **Swagger UI**: `http://localhost:8000/docs`
- **ReDoc**: `http://localhost:8000/redoc`

### Key Endpoints

#### Authentication
```http
POST /api/auth/login     # User login
POST /api/auth/refresh   # Token refresh
POST /api/auth/logout    # User logout
```

#### Data Management
```http
POST /data/upload        # Upload datasets
GET  /data/sources       # List data sources
GET  /data/sources/{id}  # Get source details
```

#### EDA & Analysis
```http
GET  /eda/api/sources/{id}/preview   # Data preview
POST /eda/api/sources/{id}/quality   # Quality report
POST /advanced-eda/analyze/{id}      # Advanced analysis
```

#### LLM Integration
```http
POST /llm/query              # Chat with AI
POST /llm/recommend-model    # Get model recommendation
POST /llm/context/{id}       # Data-aware queries
```

## 🛡️ Security Features

### Code Execution Security
- **Static Analysis**: AST parsing for dangerous pattern detection
- **Import Restrictions**: Whitelist of allowed libraries only
- **Process Isolation**: Subprocess execution with timeout limits
- **Resource Limits**: Memory and CPU constraints
- **Network Blocking**: Prevention of external connections

### Authentication & Authorization
- **JWT Tokens**: Secure token-based authentication
- **Role-Based Access**: User, admin, and custom permissions
- **Session Management**: Secure session handling
- **API Rate Limiting**: Protection against abuse

## 🔬 Advanced Features

### Domain-Specific Analysis
```python
# Categorical data analysis
from core.eda.advanced_eda.granular_components.categorical import CategoricalAnalysis

# Geospatial analysis
from core.eda.advanced_eda.granular_components.geospatial import GeospatialAnalysis

# Time series analysis
from core.eda.advanced_eda.granular_components.time_series import TimeSeriesAnalysis
```

### LLM Model Switching
```javascript
// Switch to code-optimized model
await window.LLMChat.switchToCodeModel();

// Get model recommendation
await window.LLMChat.getModelRecommendation('code', 'deepseek');
```

## 🚀 Deployment

### Docker Deployment
```bash
# Build image
docker build -t mlops-platform .

# Run container
docker run -p 8000:8000 \
  -e SECRET_KEY=your-secret \
  -e DATABASE_URL=your-db-url \
  mlops-platform
```

### Production Deployment
```bash
# Using Gunicorn + Uvicorn
gunicorn main:app -w 4 -k uvicorn.workers.UvicornWorker \
  --bind 0.0.0.0:8000 \
  --timeout 120 \
  --keepalive 2
```

## 📊 Monitoring & Observability

### Built-in Monitoring
- **Health Checks**: `/health` endpoint for service monitoring
- **Performance Metrics**: Request timing and resource usage
- **Error Tracking**: Comprehensive logging and error handling
- **Admin Dashboard**: Real-time system statistics

### Logging Configuration
```python
# Structured logging with rotation
LOGGING = {
    'level': 'INFO',
    'format': '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    'file': 'logs/mlops.log',
    'max_bytes': 10_000_000,
    'backup_count': 5
}
```

## 🧪 Testing

### Run Test Suite
```bash
# Unit tests
pytest tests/unit/

# Integration tests
pytest tests/integration/

# Security tests
pytest tests/security/

# Full test suite with coverage
pytest --cov=core --cov-report=html
```

### Test Categories
- **Unit Tests**: Core functionality testing
- **Integration Tests**: End-to-end workflow testing
- **Security Tests**: Sandbox and authentication testing
- **Performance Tests**: Load and stress testing

## 🛠️ Development

### Development Setup
```bash
# Install development dependencies
pip install -r requirements-dev.txt

# Pre-commit hooks
pre-commit install

# Code formatting
black .
isort .

# Type checking
mypy core/
```

### Project Structure
```
├── core/                   # Core application modules
│   ├── auth/              # Authentication & authorization
│   ├── data_ingestion/    # Data upload & management
│   ├── eda/               # Exploratory data analysis
│   ├── llm/               # LLM integration
│   ├── admin/             # Admin dashboard
│   └── database/          # Database models & connections
├── static/                # Frontend assets
├── templates/             # Jinja2 templates
├── tests/                 # Test suite
├── docs/                  # Documentation
└── config.py              # Application configuration
```

## 🧰 Tooling & Automation

- **Notebook Bundling**: `tools/build_notebook_bundle.py` compiles advanced EDA notebooks for deployment
- **Alembic**: `alembic.ini` preconfigures migration settings for managing schema upgrades
- **Middleware Stack**: `middleware/logging.py` and `middleware/error_handler.py` provide structured logging and global error capture
- **Background Tasks**: Lifespan hooks in `main.py` initialize connections, warm caches, and gracefully shut down resources

## 📈 Performance Optimization

### Database Optimization
- **Connection Pooling**: Async SQLAlchemy with connection pooling
- **Query Optimization**: Efficient queries with proper indexing
- **Caching Layer**: Redis for session and analysis caching

### Analysis Performance
- **Lazy Loading**: On-demand data loading for large datasets
- **Chunked Processing**: Memory-efficient processing for large files
- **Parallel Processing**: Multi-process analysis for heavy computations

## 🤝 Contributing

### Contribution Guidelines
1. **Fork** the repository
2. **Create** a feature branch (`git checkout -b feature/amazing-feature`)
3. **Commit** your changes (`git commit -m 'Add amazing feature'`)
4. **Push** to the branch (`git push origin feature/amazing-feature`)
5. **Open** a Pull Request

### Code Standards
- **Black**: Code formatting
- **isort**: Import sorting
- **MyPy**: Type checking
- **Pytest**: Testing framework
- **Pre-commit**: Git hooks for quality

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🆘 Support

### Documentation
- **API Docs**: Available at `/docs` when running in debug mode
- **User Guide**: See `docs/` directory for detailed guides
- **Security Guide**: `docs/EDA_SECURITY_GUIDE.md`

### Getting Help
- **Issues**: GitHub Issues for bug reports and feature requests
- **Discussions**: GitHub Discussions for questions and community
- **Wiki**: Comprehensive documentation and examples

## 📘 Documentation Index

- `LLM_MODEL_SWITCHING_GUIDE.md`: Provider configuration and client-side switching helpers
- `docs/EDA_SECURITY_GUIDE.md`: In-depth look at sandbox enforcement and code validation
- `docs/MULTI_USER_ENHANCEMENTS.md`: Role-based access control and admin UX improvements
- `docs/REAL_TIME_MONITORING_GUIDE.md`: Monitoring architecture and observability patterns
- `core/eda/advanced_eda/README.md`: Advanced analysis runtime, generators, and component catalog

## 🙏 Acknowledgments

- **FastAPI**: Modern web framework for Python APIs
- **Pandas**: Data manipulation and analysis library
- **SQLAlchemy**: SQL toolkit and ORM
- **Jupyter**: Interactive computing environment
- **All Contributors**: Thanks to everyone who has contributed to this project

---

**Made with ❤️ for the Data Science Community**

*Transform your data workflows with intelligent automation and secure analysis.*