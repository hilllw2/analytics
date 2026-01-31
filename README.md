# DataChat Analytics Platform

A natural language data analytics platform that lets you explore and analyze your data through conversation. Upload your files, ask questions in plain English, and get instant insights, visualizations, and reports.

## Features

### Data Ingestion
- **File Support**: CSV, TSV, XLSX, XLS
- **Auto-detection**: Delimiter, encoding, and data types
- **Smart Type Inference**: Dates, numbers, categories, booleans
- **Large File Handling**: Sample mode and partial loading options
- **Multi-sheet Excel**: Select specific sheets to analyze

### Data Profiling & Auto Insights
- **Health Report**: Missing values, duplicates, outliers
- **Column Analysis**: Type detection, cardinality, distributions
- **Correlation Matrix**: Identify relationships between variables
- **Auto-generated Insights**: Top 10+ human-readable findings
- **Smart Suggestions**: Recommended questions to ask

### Natural Language Interface
- **ChatGPT-style UI**: Conversational data exploration
- **Context Awareness**: Remembers dataset, filters, and definitions
- **Response Options**: Regenerate, make detailed/simple/technical
- **Pin Definitions**: Define metrics once, reuse everywhere
- **Methodology Display**: See how answers were computed

### Analytics Capabilities
- **KPI Analysis**: Totals, averages, MoM/YoY comparisons
- **Contribution Analysis**: Which segments drive your metrics
- **Time Series**: Trends, seasonality, change detection
- **Cohort Analysis**: Retention tables and matrices
- **Funnel Analysis**: Stage conversion rates
- **Anomaly Detection**: IQR and z-score methods
- **Driver Analysis**: Correlation and feature importance

### Visualizations
- **Chart Types**: Line, bar, histogram, box, scatter, heatmap, pie, area, funnel, Pareto
- **Auto-selection**: Intelligent chart type based on data
- **Export**: PNG, SVG, underlying data as CSV
- **Interactive**: Zoom, pan, hover details (Plotly)

### Export & Reports
- **Download Charts**: PNG, SVG formats
- **Export Data**: CSV tables
- **Generate Reports**: HTML and Markdown formats
- **Bundle Export**: ZIP with all charts, data, and reports

### Ephemeral Sessions
- **No Persistence**: Data exists only during your session
- **Auto-cleanup**: Sessions timeout after inactivity
- **Privacy First**: Nothing stored after you leave

## Deploy with Docker (single container)

Build and run the full app (frontend + backend) in one container:

```bash
# From project root
docker compose up --build
# Open http://localhost:8000
```

Set `GEMINI_API_KEY` in a `.env` file in the project root, or run:

```bash
GEMINI_API_KEY=your_key docker compose up --build
```

For **Digital Ocean App Platform**, connect your repo and use the included `Dockerfile`. Set `GEMINI_API_KEY` and optional env vars in the DO dashboard. See **[DEPLOY.md](DEPLOY.md)** for step-by-step instructions.

---

## Quick Start (local development)

### Prerequisites
- Python 3.10+
- Node.js 18+
- Google Gemini API key

### Backend Setup

```bash
cd backend

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Configure environment
cp .env.example .env
# Edit .env and add your GEMINI_API_KEY

# Run the server
python run.py
```

The API will be available at http://localhost:8000 with docs at http://localhost:8000/docs

### Frontend Setup

```bash
cd frontend

# Install dependencies
npm install

# Run development server
npm run dev
```

The UI will be available at http://localhost:5173

## Architecture

```
datachat/
├── backend/
│   ├── app/
│   │   ├── api/           # FastAPI routes
│   │   │   ├── upload.py     # File upload & ingestion
│   │   │   ├── chat.py       # Natural language chat
│   │   │   ├── analytics.py  # Analytics endpoints
│   │   │   ├── export.py     # Export functionality
│   │   │   └── session.py    # Session management
│   │   ├── core/          # Core modules
│   │   │   ├── config.py     # Configuration
│   │   │   └── session_manager.py  # Ephemeral sessions
│   │   ├── services/      # Business logic
│   │   │   ├── data_ingestion.py   # File parsing
│   │   │   ├── data_profiler.py    # Data quality analysis
│   │   │   ├── auto_insights.py    # Insight generation
│   │   │   ├── llm_service.py      # Gemini integration
│   │   │   ├── query_executor.py   # Safe code execution
│   │   │   ├── analytics_engine.py # Advanced analytics
│   │   │   └── visualization.py    # Chart generation
│   │   └── main.py        # FastAPI app
│   └── requirements.txt
├── frontend/
│   ├── src/
│   │   ├── components/    # React components
│   │   ├── services/      # API client
│   │   └── store/         # Zustand state
│   ├── package.json
│   └── vite.config.ts
└── README.md
```

## API Endpoints

### Session
- `POST /api/session/create` - Create new session
- `GET /api/session/info` - Get session status
- `DELETE /api/session/end` - End session and delete data

### Upload
- `POST /api/upload/file` - Upload data file
- `GET /api/upload/preview` - Get paginated data preview
- `POST /api/upload/profile` - Generate health report
- `POST /api/upload/insights` - Generate auto insights

### Chat
- `POST /api/chat/message` - Send natural language query
- `POST /api/chat/regenerate` - Regenerate last response
- `POST /api/chat/pin-definition` - Pin a metric definition
- `GET /api/chat/suggestions` - Get suggested questions

### Analytics
- `POST /api/analytics/kpi` - Compute KPI cards
- `POST /api/analytics/contribution` - Segment contribution
- `POST /api/analytics/time-series` - Time series analysis
- `POST /api/analytics/cohort` - Cohort retention
- `POST /api/analytics/funnel` - Funnel analysis
- `POST /api/analytics/anomalies` - Anomaly detection
- `POST /api/analytics/drivers` - Driver analysis

### Export
- `GET /api/export/chart/{id}/png` - Download chart as PNG
- `GET /api/export/table` - Export data as CSV
- `POST /api/export/report` - Generate report
- `GET /api/export/bundle` - Download everything as ZIP

## Configuration

Environment variables in `.env`:

| Variable | Description | Default |
|----------|-------------|---------|
| `GEMINI_API_KEY` | Google Gemini API key | (required) |
| `SESSION_TIMEOUT_MINUTES` | Session expiry time | 60 |
| `MAX_UPLOAD_SIZE_MB` | Maximum file size | 500 |
| `LLM_MODEL` | Gemini model to use | gemini-2.0-flash |
| `LLM_TEMPERATURE` | LLM creativity | 0.3 |

## Example Queries

```
"Summarize this dataset"
"What are the top 10 customers by revenue?"
"Show me the trend of sales over time"
"Find anomalies in the transaction amounts"
"Compare revenue by region"
"What drives conversion rate?"
"Create a cohort retention analysis"
"Show the funnel from lead to customer"
```

## Technology Stack

### Backend
- FastAPI - High-performance API framework
- Pandas - Data manipulation
- Plotly - Interactive charts
- Google Gemini - LLM for natural language
- SciPy/Scikit-learn - Statistical analysis

### Frontend
- React 18 - UI framework
- TypeScript - Type safety
- Tailwind CSS - Styling
- Zustand - State management
- Plotly.js - Chart rendering
- TanStack Table - Data grid

## License

MIT License - See LICENSE file for details.
