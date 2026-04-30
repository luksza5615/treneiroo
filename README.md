# Treneiro 🏃‍♂️📊🤖
Treneiro is a Python-based **data + AI application** that ingests sport activity data from Garmin, stores and models it in a relational database, computes training metrics, and uses **LLMs** to generate **training insights**. It also includes a simple **data app UI** for reviewing recent activities, trends, and AI-generated summaries.

## What problem it solves
Garmin gives you lots of data, but turning it into **decisions** is harder:
- What should I change next week given my history?
- Is the training load distributed well over time?
- Is the training execution aligned with intended purpose?

Treneiro turns raw activity logs into:
- structured metrics and trends, and
- context-based AI recommendations

## Key features
- **Garmin data ingestion & ETL** (API/fit files→ normalized storage → analytics-ready tables)
- **Relational DB** (SQL schema design, migrations, consistency)
- **LLM integration** (weekly summaries, recommendations, structured outputs, prompt design)
- **Data app UI** (decision-support dashboard with computed metrics)
- **Cloud deployment** (Azure-friendly setup)
- **Other** (config management, logging, error handling, CI-ready structure)

### Planned / In progress
- RAG over training history for contextual answers (“why is this week harder?”)
- Fatigue / readiness indicators

## Architecture
**Flow**
1. Ingest activities from Garmin API
2. Store processed data in a relational DB (Azure SQL)
3. Compute training metrics (e.g. weekly load, intensity distribution, trends)
4. Use AI layer (LLMs) to generate insights and planning suggestions
5. Present results in a simple UI (Streamlit)

**Components**
- **Ingestion**: Garmin API / FIT parsing
- **Analysis**: metrics, transformations, aggregation
- **Database**: schema + persistence + migrations
- **AI Engine**: prompts + structured output
- **UI**: Streamlit UI

**Deployment**
- Azure Container Apps (app runtime)
- Azure SQL Database (storage)
- GitHub Actions (CI/CD) 

**Diagram**
![Architecture](docs/02-architecture/architecture.drawio.svg)

## Project structure
```markdown
📂 treneiroo
├── 📄 Dockerfile
├── 📄 README.md
├── 📄 app.py
└── 📂 docs/
├── 📄 pyproject.toml
└── 📂 executions/
└── 📂 src/
│  └── 📂 treneiroo/
│    └── 📂 ai/
│    └── 📂 analysis/
│    └── 📂 database/
│    └── 📂 domain/
│    └── 📂 ingestion/
│    └── 📂 orchestration/
│    └── 📂 settings/
│    └── 📂 ui/    
│    └── 📂 utils/    
└── 📂 tests/
│  └── 📂 integration/
│  └── 📂 unit/
├── 📄 uv.lock
```

## Tech stack
- **Language:** Python
- **Data processing:** pandas
- **Database / ORM:** Azure SQL + SQLAlchemy
- **Ingestion:** Garmin Connect API client, FIT file parsing
- **UI:** Streamlit
- **LLM providers:** Google 
- **Packaging** Docker
- **CI/CD:** GitHub Actions 
## Setup

### Prerequisites
- Python 3.11+ recommended
- Access to a Garmin account
- Azure SQL database 
- Google LLM API key

## Run locally
### 1) Set .env
```bash
Copy `.env.example` → `.env` and fill in
```

### 2) Install dependencies
```bash
uv sync
```

### 2) Run the UI
```bash
uv run streamlit run app.py
```
