# TCS Food Formulator — Implementation Plan

This implementation plan details the full-stack SaaS-style redesign and deployment strategy for **TCS Food Formulator**. The platform bridges a complex backend LangGraph RAG pipeline with an intuitive, interactive frontend supporting both Guided and Expert user workflows, combined with regional compliance validation and interactive visualizations.

---

## User Review Required

> [!IMPORTANT]
> The primary LLM used in the prototype is **llama-3.3-70b-versatile** on Groq. Please ensure your Groq API key is configured in the environment. If it is missing or rate-limited, the application will fallback to **llama-3.1-8b-instant** as implemented in the python script.

> [!WARNING]
> By default, the application will use the `InMemoryVectorStore` fallback for cosine similarity and category-aware search if Pinecone credentials are not supplied or if Pinecone index creation fails. This ensures out-of-the-box functionality in any local environment.

---

## Proposed System Architecture

### Complete System Architecture Diagram

```mermaid
graph TD
    %% Frontend Components
    subgraph Frontend [Next.js App Router & TypeScript]
        UI[Pages & UI Components]
        ZF[Zustand State Store]
        RF[React Flow Diagram]
        RC[Recharts Visualizations]
    end

    %% API Gateway / Nginx
    NG[Nginx Reverse Proxy]

    %% Backend Services
    subgraph Backend [FastAPI Backend Service]
        API[REST API Handlers]
        LG[LangGraph Formulation Engine]
        VAL[Compliance Validation Engine]
        DB_M[SQLAlchemy ORM]
    end

    %% Storage and External APIs
    subgraph Storage [Databases & External Systems]
        PG[(PostgreSQL Database)]
        VS[(Pinecone Vector DB / In-Memory Store)]
        USDA[USDA FoodData Central API]
        Groq[Groq LLM Service]
        HF[HuggingFace Embeddings Model]
    end

    %% Connections
    UI --> ZF
    UI --> RF
    UI --> RC
    UI -->|HTTPS| NG
    NG --> API
    API --> DB_M
    DB_M --> PG
    API --> LG
    API --> VAL
    LG --> VS
    LG --> Groq
    VS --> HF
    LG --> USDA
```

### LangGraph Workflow Diagram

```mermaid
graph LR
    START([Start / User Request]) --> Retrieve[Node 1: Retrieve<br>Category-Aware Multi-Role]
    Retrieve --> Generate[Node 2: Generate<br>Groq LLM Report]
    Generate --> END([End / Formulated Plain Text])
```

---

## Database Schema (PostgreSQL)

We will use SQLAlchemy ORM with a PostgreSQL database to model users, projects, ingredients, formulations, and compliance logs.

```mermaid
erDiagram
    USERS {
        id UUID PK
        username VARCHAR
        email VARCHAR
        password_hash VARCHAR
        role VARCHAR
        created_at TIMESTAMP
    }
    PROJECTS {
        id UUID PK
        user_id UUID FK
        name VARCHAR
        category VARCHAR
        query TEXT
        status VARCHAR
        created_at TIMESTAMP
        updated_at TIMESTAMP
    }
    INGREDIENTS {
        id SERIAL PK
        name VARCHAR
        category VARCHAR
        protein FLOAT
        fat FLOAT
        carbs FLOAT
        calories FLOAT
        functional_role VARCHAR
        allergen VARCHAR
    }
    FORMULATIONS {
        id UUID PK
        project_id UUID FK
        version VARCHAR
        composition JSONB
        functional_coverage JSONB
        performance_prediction JSONB
        process_recommendations JSONB
        status VARCHAR
        created_at TIMESTAMP
    }
    COMPLIANCE_LOGS {
        id UUID PK
        formulation_id UUID FK
        region VARCHAR
        status VARCHAR
        alerts JSONB
        verified_at TIMESTAMP
    }

    USERS ||--o{ PROJECTS : owns
    PROJECTS ||--o{ FORMULATIONS : has
    FORMULATIONS ||--o{ COMPLIANCE_LOGS : validates
```

---

## Folder Structure

The folder structure organizes the frontend, backend, database migrations, and docker configurations into a clean workspace:

```text
tcs-food-formulator/
├── docker-compose.yml
├── nginx.conf
├── backend/
│   ├── Dockerfile
│   ├── requirements.txt
│   ├── .env.example
│   ├── main.py
│   ├── config.py
│   ├── database.py
│   ├── models.py
│   ├── schemas.py
│   ├── compliance.py
│   ├── rag_engine.py
│   └── food_ingredients_clean.csv
├── frontend/
│   ├── Dockerfile
│   ├── package.json
│   ├── tsconfig.json
│   ├── tailwind.config.js
│   ├── src/
│   │   ├── app/
│   │   │   ├── layout.tsx
│   │   │   ├── page.tsx
│   │   │   ├── login/page.tsx
│   │   │   ├── register/page.tsx
│   │   │   └── dashboard/
│   │   │       ├── page.tsx
│   │   │       ├── projects/page.tsx
│   │   │       ├── formulate/
│   │   │       │   ├── guided/page.tsx
│   │   │       │   └── expert/page.tsx
│   │   │       ├── ingredients/page.tsx
│   │   │       ├── compliance/page.tsx
│   │   │       ├── reports/page.tsx
│   │   │       └── settings/page.tsx
│   │   ├── components/
│   │   │   ├── ui/          # shadcn components
│   │   │   ├── dashboard-nav.tsx
│   │   │   ├── expert-canvas.tsx
│   │   │   ├── guided-wizard.tsx
│   │   │   ├── formulation-results.tsx
│   │   │   └── charts.tsx
│   │   └── store/
│   │       └── useStore.ts  # Zustand store
```

---

## Sequence Diagram: Formulation Lifecycle

```mermaid
sequenceDiagram
    autonumber
    actor User as User (Guided/Expert)
    participant FE as Frontend (Next.js)
    participant BE as Backend (FastAPI)
    participant LG as LangGraph Workflow
    participant VS as Vector Store (Pinecone/Memory)
    participant LLM as Groq LLM
    participant COMP as Compliance Engine
    participant DB as PostgreSQL

    User->>FE: Inputs constraints & clicks "Generate"
    FE->>BE: POST /api/formulate { query, constraints, project_id }
    BE->>LG: Execute Graph State Graph
    rect rgb(200, 220, 240)
        note right of LG: Node 1: Retrieve
        LG->>VS: category_aware_search(query, k_per_role=2)
        VS-->>LG: Returns 16 ingredients covering 8 roles
    end
    rect rgb(220, 200, 240)
        note right of LG: Node 2: Generate
        LG->>LLM: invoke(SYSTEM_PROMPT, context_ingredients)
        LLM-->>LG: Returns plain-text formulation report
    end
    LG-->>BE: Returns raw_response
    BE->>BE: Parse plain-text report into JSON schema
    BE->>COMP: Run regional audits (FSSAI, FDA, EFSA)
    COMP-->>BE: Returns compliance logs & validation alerts
    BE->>DB: Save project formulation, compliance log, & update status
    DB-->>BE: Commit success
    BE-->>FE: Return formulation JSON + compliance + metadata
    FE-->>User: Render visualizations (Recharts, React Flow composition)
```

---

## API Specifications

### 1. Authentication
* **POST `/api/auth/register`**
  * Request: `{"username": "...", "email": "...", "password": "...", "role": "Guided User"}`
  * Response: `{"access_token": "...", "token_type": "bearer", "role": "..."}`
* **POST `/api/auth/login`**
  * Request: `{"email": "...", "password": "..."}`
  * Response: `{"access_token": "...", "token_type": "bearer", "role": "..."}`

### 2. Projects
* **GET `/api/projects`**: Lists all projects.
* **POST `/api/projects`**: Creates a new formulation project.
  * Request: `{"name": "...", "category": "Beverage", "query": "..."}`
* **GET `/api/projects/{id}`**: Retrieves project detail including formulation history.

### 3. Ingredients
* **GET `/api/ingredients`**: Lists and filters database ingredients.
* **GET `/api/ingredients/search?query=...`**: Search ingredients using semantic terms.

### 4. Formulation
* **POST `/api/formulate`**
  * Request:
    ```json
    {
      "project_id": "optional-uuid",
      "query": "Generate a vegan high-protein chocolate beverage",
      "constraints": {
        "vegan": true,
        "gluten_free": true,
        "protein_target": 20.0,
        "sugar_target": 5.0,
        "calories_target": 250,
        "allergens_exclude": ["Soy"]
      }
    }
    ```
  * Response:
    ```json
    {
      "formulation": {
        "recipe_id": "REC_CHOC_001",
        "recipe_name": "Vegan Chocolate High Protein Drink",
        "category": "Beverage",
        "composition": {
          "bulk_ingredients": [
            {"name": "Pea Protein Isolate", "category": "Protein Source", "percentage": 15.5},
            {"name": "Oat Flour", "category": "Bulking Agent", "percentage": 84.5}
          ],
          "functional_ingredients": []
        },
        "functional_coverage": {
          "roles": {
            "Protein Source": "Pea Protein Isolate",
            "Bulking Agent": "Oat Flour"
          },
          "completeness_score": 100.0,
          "assessment": "FUNCTIONALLY COMPLETE"
        },
        "performance_prediction": {
          "protein": 12.4,
          "fat": 0.62,
          "carbohydrates": 71.4,
          "calories": 328.0,
          "texture": "Smooth powder",
          "mouthfeel": "Creamy when hydrated",
          "viscosity": "Medium-low",
          "shelf_life": 12
        },
        "process_recommendations": [
          "Dry blending of bulk and functional ingredients",
          "Homogenization at 2000 rpm x 5 min",
          "Heat treatment at 85C for 30 sec"
        ],
        "status": "READY FOR VIRTUAL VALIDATION"
      },
      "compliance": {
        "FSSAI": {"status": "Passed", "alerts": []},
        "FDA": {"status": "Warning", "alerts": ["Sugar content limit warning"]},
        "EFSA": {"status": "Passed", "alerts": []}
      }
    }
    ```

---

## UI Wireframes Description

1. **Homepage (`/`)**: A highly polished marketing lander. Prominently shows a dark-themed visual block of "AI-powered Food Formulation" with CTAs to "Start Formulating" or "Explore Ingredients", a interactive SVG diagram showing the LangGraph workflow, validation statistics, and problem statement section.
2. **Dashboard (`/dashboard`)**: A command center layout with statistical cards (Formulations Created, Compliance Passing Rate, Ingredients Used). Underneath, two lists: "Recent Formulation Projects" (status badges: `Ready`, `Needs Review`, `Archived`) and "Recent Compliance Alerts". Quick action cards to launch "Guided Wizard" or "Expert Workspace".
3. **Guided Formulator (`/dashboard/formulate/guided`)**: A multi-step stepper interface:
   * *Step 1*: Category grid (Beverage, Bakery, Energy Bar, Dairy Alternative).
   * *Step 2*: Natural language prompt input box.
   * *Step 3*: Strict parameter collection sliders (Targets for Protein, Calories, Sugar, and dietary checkboxes for Vegan, Gluten-Free).
   * *Step 4*: Ingredient Preview. Displays matching ingredients returned by category-aware search grouped by roles.
   * *Step 5*: Generating/Result View. Renders a dashboard layout showing the final formulation breakdown in tables, bar charts for composition, compliance gauges, process flowchart, and export buttons.
4. **Expert Formulator (`/dashboard/formulate/expert`)**: Dual-pane workspace.
   * *Left Pane*: A drag-and-drop or checklist catalog of all ingredients grouped by role.
   * *Right Pane (Canvas)*: Live interactive list where users adjust the percentages of chosen ingredients with a slider. A real-time calculated panel dynamically updates total protein, fat, carbs, calories, and checks if the total equals exactly 100.00%. Includes an "AI Optimization Suggestions" alert box and a "Run Compliance Check" trigger.
5. **Ingredient Library (`/dashboard/ingredients`)**: A clean tabular interface with semantic search. Clicking on an ingredient shows a detail sidebar listing USDA macros, functional properties, maximum limits, allergens, and common applications. Allows side-by-side comparison.

---

## Docker Deployment Architecture

```yaml
version: '3.8'

services:
  postgres:
    image: postgres:15-alpine
    container_name: tf_postgres
    environment:
      POSTGRES_DB: tcs_food_formulator
      POSTGRES_USER: admin
      POSTGRES_PASSWORD: supersecretpassword
    ports:
      - "5432:5432"
    volumes:
      - pgdata:/var/lib/postgresql/data

  backend:
    build: ./backend
    container_name: tf_backend
    environment:
      - DATABASE_URL=postgresql://admin:supersecretpassword@postgres:5432/tcs_food_formulator
      - GROQ_API_KEY=${GROQ_API_KEY}
      - PINECONE_API_KEY=${PINECONE_API_KEY}
      - USDA_API_KEY=${USDA_API_KEY}
    ports:
      - "8000:8000"
    depends_on:
      - postgres

  frontend:
    build: ./frontend
    container_name: tf_frontend
    ports:
      - "3000:3000"
    environment:
      - NEXT_PUBLIC_API_URL=http://localhost:8000
    depends_on:
      - backend

  nginx:
    image: nginx:alpine
    container_name: tf_nginx
    ports:
      - "80:80"
    volumes:
      - ./nginx.conf:/etc/nginx/nginx.conf:ro
    depends_on:
      - frontend
      - backend

volumes:
  pgdata:
```

---

## Verification Plan

### Automated Tests
1. **RAG Pipeline Test**: Run RAG pipeline tests using sample queries (`"vegan chocolate beverage"`, `"gluten-free bread"`) to verify the LangGraph completes execution and produces all sections of the formulation report.
2. **REST API endpoints test**: Validate authentication, projects, and formulation endpoints.
3. **Compliance logic test**: Unit test compliance validation code with inputs designed to trigger warnings or failures (e.g. non-vegan ingredient in a vegan prompt, excessive stabilizer concentrations).

---

## Implementation Roadmap

* [ ] **Phase 1: Environment & Base Setup** (FastAPI structure, PostgreSQL schema, SQLite/PG local init).
* [ ] **Phase 2: RAG Pipeline Porting** (Importing graph nodes, in-memory/Pinecone index init, LLM connection validation).
* [ ] **Phase 3: Backend API Integration** (Writing API routes, parsing plain-text RAG reports into structured JSON, regional compliance checks).
* [ ] **Phase 4: Frontend Development** (Next.js layout, Zustand global state, Guided stepper, Expert drag/slider composer, Recharts dashboards).
* [ ] **Phase 5: Dockerization & Orchestration** (Setting up Dockerfiles, docker-compose, and local reverse proxy).

---

# Code Skeletons

Below are the production-ready code skeletons for both backend and frontend.

## 1. Backend Skeletons

### 1.1 Backend Models (`backend/models.py`)
```python
import uuid
from sqlalchemy import Column, String, Float, DateTime, ForeignKey, JSON
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import relationship
import datetime

Base = declarative_base()

class User(Base):
    __tablename__ = "users"
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    username = Column(String, unique=True, nullable=False)
    email = Column(String, unique=True, nullable=False)
    password_hash = Column(String, nullable=False)
    role = Column(String, default="Guided User") # Guided User, Expert User, Administrator
    created_at = Column(DateTime, default=datetime.datetime.utcnow)

class Project(Base):
    __tablename__ = "projects"
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    user_id = Column(UUID(as_uuid=True), ForeignKey("users.id"))
    name = Column(String, nullable=False)
    category = Column(String, nullable=False)
    query = Column(String, nullable=True)
    status = Column(String, default="Pending") # Pending, Formulated, Validated
    created_at = Column(DateTime, default=datetime.datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.datetime.utcnow, onupdate=datetime.datetime.utcnow)
    
    formulations = relationship("Formulation", back_populates="project")

class Formulation(Base):
    __tablename__ = "formulations"
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    project_id = Column(UUID(as_uuid=True), ForeignKey("projects.id"))
    version = Column(String, default="V1.0")
    composition = Column(JSON, nullable=False) # Ingredients list + details
    functional_coverage = Column(JSON, nullable=False) # Coverage list & completeness score
    performance_prediction = Column(JSON, nullable=False) # Viscosity, calories, texture, shelf life
    process_recommendations = Column(JSON, nullable=False) # Processing steps
    status = Column(String, default="Draft")
    created_at = Column(DateTime, default=datetime.datetime.utcnow)

    project = relationship("Project", back_populates="formulations")
    compliance_logs = relationship("ComplianceLog", back_populates="formulation")

class ComplianceLog(Base):
    __tablename__ = "compliance_logs"
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    formulation_id = Column(UUID(as_uuid=True), ForeignKey("formulations.id"))
    region = Column(String, nullable=False) # FSSAI, FDA, EFSA
    status = Column(String, nullable=False) # Passed, Warning, Failed
    alerts = Column(JSON, nullable=False) # List of alert strings
    verified_at = Column(DateTime, default=datetime.datetime.utcnow)

    formulation = relationship("Formulation", back_populates="compliance_logs")
```

### 1.2 Compliance Engine (`backend/compliance.py`)
```python
from typing import Dict, Any, List

def evaluate_compliance(composition: Dict[str, Any], constraints: Dict[str, Any]) -> Dict[str, Any]:
    """
    Evaluates ingredient composition and constraints against FSSAI, FDA, and EFSA standards.
    """
    results = {
        "FSSAI": {"status": "Passed", "alerts": []},
        "FDA": {"status": "Passed", "alerts": []},
        "EFSA": {"status": "Passed", "alerts": []}
    }
    
    ingredients = composition.get("bulk_ingredients", []) + composition.get("functional_ingredients", [])
    
    # 1. Allergen & Strict Dietary Constraints Checks
    is_vegan = constraints.get("vegan", False)
    is_gluten_free = constraints.get("gluten_free", False)
    excluded_allergens = constraints.get("allergens_exclude", [])
    
    for ing in ingredients:
        name = ing.get("name", "").lower()
        allergen = ing.get("allergen", "None")
        
        # Vegan Check
        if is_vegan and allergen in ["Milk", "Egg"]:
            msg = f"Violation: Vegan product contains allergen {allergen} from ingredient '{ing.get('name')}'"
            results["EFSA"]["status"] = "Failed"
            results["EFSA"]["alerts"].append(msg)
            results["FDA"]["status"] = "Failed"
            results["FDA"]["alerts"].append(msg)
            results["FSSAI"]["status"] = "Failed"
            results["FSSAI"]["alerts"].append(msg)
            
        # Gluten Free Check
        if is_gluten_free and allergen == "Gluten":
            msg = f"Violation: Gluten-Free product contains gluten-based ingredient '{ing.get('name')}'"
            results["FDA"]["status"] = "Failed"
            results["FDA"]["alerts"].append(msg)
            results["FSSAI"]["status"] = "Failed"
            results["FSSAI"]["alerts"].append(msg)
            
        # User Allergen Exclusion
        if allergen in excluded_allergens:
            msg = f"Allergen warning: '{ing.get('name')}' contains excluded allergen '{allergen}'"
            for region in results:
                results[region]["status"] = "Warning"
                results[region]["alerts"].append(msg)

    # 2. Limits and Additive restrictions (Example rules)
    for ing in ingredients:
        name = ing.get("name", "").lower()
        pct = ing.get("percentage", 0.0)
        
        # Preservatives limit checks
        if ing.get("category") == "Preservation" or ing.get("functional_role") == "Preservation":
            if pct > 0.1: # Standard general limit for food additives
                msg = f"FSSAI/EFSA Alert: Preservative '{ing.get('name')}' concentration ({pct}%) exceeds standard limit of 0.10%"
                results["FSSAI"]["status"] = "Warning"
                results["FSSAI"]["alerts"].append(msg)
                results["EFSA"]["status"] = "Warning"
                results["EFSA"]["alerts"].append(msg)

        # Sweeteners limit checks
        if ing.get("functional_role") == "Sweetener":
            if name in ["stevia extract", "sucralose"] and pct > 0.05:
                msg = f"FDA/EFSA Warning: High-intensity sweetener '{ing.get('name')}' exceeds standard formula limits (0.05%) for consumer beverages."
                results["FDA"]["status"] = "Warning"
                results["FDA"]["alerts"].append(msg)
                results["EFSA"]["status"] = "Warning"
                results["EFSA"]["alerts"].append(msg)

    return results
```

### 1.3 LangGraph RAG Service & Parsing (`backend/rag_engine.py`)
```python
import os
import re
from typing import Dict, Any, List
import pandas as pd
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_groq import ChatGroq
from langchain_core.messages import SystemMessage, HumanMessage
from langgraph.graph import StateGraph, END
from typing import TypedDict
from langchain_core.documents import Document as LCDocument

# Import configuration/keys
GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")
USDA_API_KEY = os.getenv("USDA_API_KEY", "")

# Shared constants
ALL_FUNCTIONAL_ROLES = [
    "Protein Source", "Bulking Agent", "Emulsifier",
    "Stabilizer", "Sweetener", "Flavor System",
    "Acidity Control", "Preservation"
]

class AgentState(TypedDict):
    query: str
    context_docs: List[LCDocument]
    raw_response: str

# Setup models (lazily initialized)
_embeddings = None
_vector_store = None
_llm = None

def get_embeddings():
    global _embeddings
    if not _embeddings:
        _embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    return _embeddings

class SimpleVectorStore:
    def __init__(self, df: pd.DataFrame, embeddings):
        self.embeddings = embeddings
        self.docs = []
        for _, row in df.iterrows():
            content = f"Ingredient Name: {row['name']}\nCategory: {row['category']}\nFunctional Role: {row['functional_role']}\nAllergen: {row['allergen']}\nProtein: {row['protein']}g\nFat: {row['fat']}g\nCarbs: {row['carbs']}g\nCalories: {row['calories']} kcal\n"
            metadata = row.to_dict()
            self.docs.append(LCDocument(page_content=content, metadata=metadata))
        
        texts = [d.page_content for d in self.docs]
        import numpy as np
        self.doc_embeddings = np.array(self.embeddings.embed_documents(texts))

    def category_aware_search(self, query: str, k_per_role: int = 2) -> List[LCDocument]:
        import numpy as np
        q_emb = np.array(self.embeddings.embed_query(query))
        results = []
        seen = set()
        
        for role in ALL_FUNCTIONAL_ROLES:
            indices = [i for i, d in enumerate(self.docs) if d.metadata.get("functional_role") == role]
            if not indices:
                continue
            sub_embs = self.doc_embeddings[indices]
            dots = sub_embs @ q_emb
            norms = np.linalg.norm(sub_embs, axis=1) * np.linalg.norm(q_emb) + 1e-9
            sims = dots / norms
            top_indices = np.argsort(sims)[::-1][:k_per_role]
            for idx in top_indices:
                doc = self.docs[indices[idx]]
                if doc.metadata["name"] not in seen:
                    results.append(doc)
                    seen.add(doc.metadata["name"])
        return results

def get_vector_store():
    global _vector_store
    if not _vector_store:
        csv_path = os.path.join(os.path.dirname(__file__), "food_ingredients_clean.csv")
        if not os.path.exists(csv_path):
            # Fallback to local import or create base dataset
            raise FileNotFoundError(f"Missing base dataset: {csv_path}")
        df = pd.read_csv(csv_path)
        _vector_store = SimpleVectorStore(df, get_embeddings())
    return _vector_store

def get_llm():
    global _llm
    if not _llm:
        _llm = ChatGroq(
            model_name="llama-3.3-70b-versatile",
            temperature=0.1,
            groq_api_key=GROQ_API_KEY
        )
    return _llm

# Nodes
def retrieve_node(state: AgentState) -> Dict[str, Any]:
    query = state["query"]
    store = get_vector_store()
    docs = store.category_aware_search(query)
    return {"context_docs": docs}

SYSTEM_PROMPT = """\
You are a Senior Food Scientist at TCS Research designing formulation reports for clients.
Generate a recipe formulation report in the EXACT plain-text format below.
Rules:
  1. All percentages must sum to exactly 100.00%.
  2. Use ONLY ingredients from the provided context list.
  3. Allergen filtering: if the query says vegan, exclude Milk/Egg. If gluten-free, exclude Gluten.
  4. Calculate Protein (g), Sugar (g), and Calories (kcal) from the formulation percentages and
     the per-100g macros supplied in the ingredient context.
  5. Output raw plain text ONLY — no markdown, no asterisks, no backticks.

Format Template:
====================================================================
TCS FOOD FORMULATOR – GENERATED FORMULATION REPORT
Constraint-Driven Multi-Objective Recipe Generation
====================================================================

Recipe ID      : <e.g. CHOC_BEV_001>
Recipe Name    : <Descriptive Name>
Category       : <Product Category>
Version        : V1.0

====================================================================
1. COMPLETE FORMULATION COMPOSITION
====================================================================

BULK INGREDIENTS
--------------------------------------------------------------------
Ingredient                     Category           Conc. (%)

<Ingredient 1>                 <Category>         <XX.XX>
<Ingredient 2>                 <Category>         <XX.XX>

Subtotal (Bulk)                                   <XX.XX> %

--------------------------------------------------------------------

FUNCTIONAL INGREDIENTS
--------------------------------------------------------------------
Ingredient                     Functional Role    Conc. (%)

<Ingredient 1>                 <Role>             <XX.XX>
<Ingredient 2>                 <Role>             <XX.XX>

Subtotal (Functional)                              <XX.XX> %

====================================================================
TOTAL FORMULATION = 100.00 %
====================================================================

====================================================================
2. FUNCTIONAL ROLE COVERAGE
====================================================================

Role                     Ingredient               Coverage

Protein Source           <Ingredient>              [YES/NO]
Bulking Agent            <Ingredient>              [YES/NO]
Emulsifier               <Ingredient>              [YES/NO]
Stabilizer               <Ingredient>              [YES/NO]
Sweetener                <Ingredient>              [YES/NO]
Flavor System            <Ingredient>              [YES/NO]
Acidity Control          <Ingredient>              [YES/NO]
Preservation             <Ingredient>              [YES/NO]

Functional Completeness Score (FCS): <XX.X> % (<X>/8 roles covered)
Assessment: <FUNCTIONALLY COMPLETE (>=90%) / NEEDS REVIEW (<90%)>

====================================================================
3. PRODUCT PERFORMANCE PREDICTION
====================================================================

Protein                 : <XX.X> g / 100g product
Total Fat               : <XX.X> g / 100g product
Carbohydrates           : <XX.X> g / 100g product
Calories                : <XXX> kcal / 100g product

Texture                 : <Description>
Mouthfeel               : <Description>
Viscosity               : <Description>
Shelf-life              : ~<X> months

====================================================================
4. PROCESS RECOMMENDATION
====================================================================

STEP 1: <Description>
STEP 2: <Description>
STEP 3: <Description>
STEP 4: Homogenization: <RPM> rpm x <duration> min
STEP 5: Heat treatment: <temp>C x <duration> sec

====================================================================
FINAL STATUS
====================================================================

Recipe Status: READY FOR VIRTUAL VALIDATION
====================================================================
"""

def generate_node(state: AgentState) -> Dict[str, Any]:
    query = state["query"]
    docs = state["context_docs"]
    
    context_lines = []
    for i, doc in enumerate(docs, 1):
        m = doc.metadata
        context_lines.append(
            f"[{i}] Name: {m['name']} | Role: {m['functional_role']} | "
            f"Protein: {m['protein']}g | Fat: {m['fat']}g | "
            f"Carbs: {m['carbs']}g | Calories: {m['calories']} kcal | "
            f"Allergen: {m['allergen']}"
        )
    context_str = "\n".join(context_lines)
    
    user_msg = (
        f"Product Request: {query}\n\n"
        f"Available Ingredients (use ONLY these):\n{context_str}\n\n"
        f"Generate the formulation report now."
    )
    
    messages = [
        SystemMessage(content=SYSTEM_PROMPT),
        HumanMessage(content=user_msg)
    ]
    
    llm = get_llm()
    resp = llm.invoke(messages)
    return {"raw_response": resp.content}

# Compile Workflow Graph
def build_agent():
    graph = StateGraph(AgentState)
    graph.add_node("retrieve", retrieve_node)
    graph.add_node("generate", generate_node)
    graph.set_entry_point("retrieve")
    graph.add_edge("retrieve", "generate")
    graph.add_edge("generate", END)
    return graph.compile()

# Text Parser
def parse_formulation_report(report_text: str) -> Dict[str, Any]:
    """
    Regex parses the raw plain-text report from Groq LLM into clean JSON.
    """
    data = {
        "recipe_id": "REC_GEN_001",
        "recipe_name": "AI Formulation",
        "category": "General",
        "composition": {"bulk_ingredients": [], "functional_ingredients": []},
        "functional_coverage": {"roles": {}, "completeness_score": 0.0, "assessment": "NEEDS REVIEW"},
        "performance_prediction": {"protein": 0.0, "fat": 0.0, "carbohydrates": 0.0, "calories": 0.0, "texture": "", "mouthfeel": "", "viscosity": "", "shelf_life": 0},
        "process_recommendations": [],
        "status": "DRAFT"
    }

    try:
        # Parse basic metadata
        id_match = re.search(r"Recipe ID\s*:\s*([^\n]+)", report_text)
        name_match = re.search(r"Recipe Name\s*:\s*([^\n]+)", report_text)
        cat_match = re.search(r"Category\s*:\s*([^\n]+)", report_text)
        if id_match: data["recipe_id"] = id_match.group(1).strip()
        if name_match: data["recipe_name"] = name_match.group(1).strip()
        if cat_match: data["category"] = cat_match.group(1).strip()

        # Parse Bulk Ingredients
        bulk_section = re.search(r"BULK INGREDIENTS\s*[-]+\s*Ingredient\s+Category\s+Conc\.\s*\(\%\)\s*(.*?)\s*Subtotal", report_text, re.DOTALL)
        if bulk_section:
            for line in bulk_section.group(1).strip().split("\n"):
                parts = re.split(r"\s{2,}", line.strip())
                if len(parts) >= 3:
                    try:
                        data["composition"]["bulk_ingredients"].append({
                            "name": parts[0],
                            "category": parts[1],
                            "percentage": float(parts[2].replace("%", "").strip())
                        })
                    except: pass

        # Parse Functional Ingredients
        func_section = re.search(r"FUNCTIONAL INGREDIENTS\s*[-]+\s*Ingredient\s+Functional Role\s+Conc\.\s*\(\%\)\s*(.*?)\s*Subtotal", report_text, re.DOTALL)
        if func_section:
            for line in func_section.group(1).strip().split("\n"):
                parts = re.split(r"\s{2,}", line.strip())
                if len(parts) >= 3:
                    try:
                        data["composition"]["functional_ingredients"].append({
                            "name": parts[0],
                            "functional_role": parts[1],
                            "percentage": float(parts[2].replace("%", "").strip())
                        })
                    except: pass

        # Parse Coverage
        score_match = re.search(r"Functional Completeness Score\s*\(FCS\):\s*([\d\.]+)\s*%", report_text)
        assess_match = re.search(r"Assessment:\s*([^\n]+)", report_text)
        if score_match: data["functional_coverage"]["completeness_score"] = float(score_match.group(1))
        if assess_match: data["functional_coverage"]["assessment"] = assess_match.group(1).strip()

        # Parse Performance Predictions
        prot_m = re.search(r"Protein\s*:\s*([\d\.]+)", report_text)
        fat_m = re.search(r"Total Fat\s*:\s*([\d\.]+)", report_text)
        carb_m = re.search(r"Carbohydrates\s*:\s*([\d\.]+)", report_text)
        cal_m = re.search(r"Calories\s*:\s*([\d\.]+)", report_text)
        if prot_m: data["performance_prediction"]["protein"] = float(prot_m.group(1))
        if fat_m: data["performance_prediction"]["fat"] = float(fat_m.group(1))
        if carb_m: data["performance_prediction"]["carbohydrates"] = float(carb_m.group(1))
        if cal_m: data["performance_prediction"]["calories"] = float(cal_m.group(1))

        text_m = re.search(r"Texture\s*:\s*([^\n]+)", report_text)
        mouth_m = re.search(r"Mouthfeel\s*:\s*([^\n]+)", report_text)
        visc_m = re.search(r"Viscosity\s*:\s*([^\n]+)", report_text)
        shelf_m = re.search(r"Shelf-life\s*:\s*~?([\d]+)", report_text)
        if text_m: data["performance_prediction"]["texture"] = text_m.group(1).strip()
        if mouth_m: data["performance_prediction"]["mouthfeel"] = mouth_m.group(1).strip()
        if visc_m: data["performance_prediction"]["viscosity"] = visc_m.group(1).strip()
        if shelf_m: data["performance_prediction"]["shelf_life"] = int(shelf_m.group(1))

        # Parse Processing Steps
        steps = re.findall(r"STEP \d+:\s*([^\n]+)", report_text)
        data["process_recommendations"] = steps

        # Status
        status_match = re.search(r"Recipe Status:\s*([^\n]+)", report_text)
        if status_match: data["status"] = status_match.group(1).strip()

    except Exception as e:
        print(f"Error parsing formulation text: {e}")

    return data
```

### 1.4 Main Entrypoint (`backend/main.py`)
```python
import os
from fastapi import FastAPI, Depends, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Dict, Any, List, Optional
import uuid

from database import engine, get_db
from models import Base, Project, Formulation, ComplianceLog
from sqlalchemy.orm import Session
from compliance import evaluate_compliance
from rag_engine import build_agent, parse_formulation_report

app = FastAPI(title="TCS Food Formulator API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Build DB
Base.metadata.create_all(bind=engine)
agent_app = build_agent()

class FormulateRequest(BaseModel):
    project_id: Optional[str] = None
    query: str
    constraints: Dict[str, Any] = {}

@app.post("/api/formulate")
def post_formulate(req: FormulateRequest, db: Session = Depends(get_db)):
    try:
        # 1. Run RAG agentic workflow
        result = agent_app.invoke({"query": req.query})
        raw_response = result.get("raw_response", "")
        
        # 2. Parse text report to schema
        parsed_formulation = parse_formulation_report(raw_response)
        
        # 3. Assess Compliance
        compliance_results = evaluate_compliance(parsed_formulation["composition"], req.constraints)
        
        # 4. Save to DB
        project_id = req.project_id
        if not project_id:
            db_project = Project(
                name=parsed_formulation.get("recipe_name", "New Formulation"),
                category=parsed_formulation.get("category", "General"),
                query=req.query,
                status="Formulated"
            )
            db.add(db_project)
            db.commit()
            db.refresh(db_project)
            project_id = str(db_project.id)
        
        db_formulation = Formulation(
            project_id=uuid.UUID(project_id),
            composition=parsed_formulation["composition"],
            functional_coverage=parsed_formulation["functional_coverage"],
            performance_prediction=parsed_formulation["performance_prediction"],
            process_recommendations=parsed_formulation["process_recommendations"],
            status=parsed_formulation["status"]
        )
        db.add(db_formulation)
        db.commit()
        db.refresh(db_formulation)
        
        for region, status_val in compliance_results.items():
            db_log = ComplianceLog(
                formulation_id=db_formulation.id,
                region=region,
                status=status_val["status"],
                alerts=status_val["alerts"]
            )
            db.add(db_log)
        db.commit()
        
        return {
            "formulation": parsed_formulation,
            "compliance": compliance_results,
            "project_id": project_id,
            "formulation_id": str(db_formulation.id)
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/projects")
def get_projects(db: Session = Depends(get_db)):
    return db.query(Project).all()

@app.get("/api/ingredients")
def get_ingredients():
    import pandas as pd
    csv_path = os.path.join(os.path.dirname(__file__), "food_ingredients_clean.csv")
    if os.path.exists(csv_path):
        df = pd.read_csv(csv_path)
        return df.to_dict(orient="records")
    return []
```

---

## 2. Frontend Skeletons

### 2.1 Zustand Global Store (`frontend/src/store/useStore.ts`)
```typescript
import { create } from 'zustand';

interface User {
  username: string;
  email: string;
  role: string;
}

interface Project {
  id: string;
  name: string;
  category: string;
  status: string;
}

interface AppState {
  user: User | null;
  projects: Project[];
  token: string | null;
  loading: boolean;
  setUser: (user: User | null) => void;
  setToken: (token: string | null) => void;
  setProjects: (projects: Project[]) => void;
  fetchProjects: () => Promise<void>;
  logout: () => void;
}

export const useStore = create<AppState>((set, get) => ({
  user: null,
  projects: [],
  token: typeof window !== 'undefined' ? localStorage.getItem('token') : null,
  loading: false,
  setUser: (user) => set({ user }),
  setToken: (token) => {
    if (token) localStorage.setItem('token', token);
    else localStorage.removeItem('token');
    set({ token });
  },
  setProjects: (projects) => set({ projects }),
  fetchProjects: async () => {
    set({ loading: true });
    try {
      const response = await fetch(`${process.env.NEXT_PUBLIC_API_URL}/api/projects`, {
        headers: { Authorization: `Bearer ${get().token}` },
      });
      const data = await response.json();
      set({ projects: data, loading: false });
    } catch (err) {
      set({ loading: false });
    }
  },
  logout: () => {
    localStorage.removeItem('token');
    set({ user: null, token: null, projects: [] });
  },
}));
```

### 2.2 Next.js Guided Wizard Component (`frontend/src/components/guided-wizard.tsx`)
```tsx
'use client';

import React, { useState } from 'react';
import { Card, CardHeader, CardTitle, CardContent, CardFooter } from '@/components/ui/card';
import { Button } from '@/components/ui/button';
import { Input } from '@/components/ui/input';
import { Slider } from '@/components/ui/slider';
import { Checkbox } from '@/components/ui/checkbox';
import { Loader2 } from 'lucide-react';
import FormulationResults from './formulation-results';

const categories = ['Beverage', 'Bakery', 'Energy Bar', 'Dairy Alternative'];

export default function GuidedWizard() {
  const [step, setStep] = useState(1);
  const [category, setCategory] = useState('');
  const [query, setQuery] = useState('');
  const [constraints, setConstraints] = useState({
    vegan: false,
    gluten_free: false,
    protein_target: 20,
    sugar_target: 5,
    calories_target: 250,
  });
  const [loading, setLoading] = useState(false);
  const [results, setResults] = useState<any>(null);

  const handleGenerate = async () => {
    setLoading(true);
    try {
      const res = await fetch(`${process.env.NEXT_PUBLIC_API_URL}/api/formulate`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          query: `Category: ${category}. Product details: ${query}`,
          constraints,
        }),
      });
      const data = await res.json();
      setResults(data);
      setStep(5);
    } catch (err) {
      console.error(err);
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="max-w-4xl mx-auto py-8">
      {step === 1 && (
        <Card className="bg-slate-900 border-slate-800 text-white shadow-xl">
          <CardHeader>
            <CardTitle className="text-2xl font-bold bg-clip-text text-transparent bg-gradient-to-r from-blue-400 to-indigo-400">
              Step 1: Choose Product Category
            </CardTitle>
          </CardHeader>
          <CardContent className="grid grid-cols-2 gap-4">
            {categories.map((cat) => (
              <button
                key={cat}
                onClick={() => { setCategory(cat); setStep(2); }}
                className={`p-6 rounded-lg border-2 text-left font-medium transition-all ${
                  category === cat ? 'border-indigo-500 bg-indigo-500/10' : 'border-slate-800 hover:border-slate-700 bg-slate-950'
                }`}
              >
                {cat}
              </button>
            ))}
          </CardContent>
        </Card>
      )}

      {step === 2 && (
        <Card className="bg-slate-900 border-slate-800 text-white shadow-xl">
          <CardHeader>
            <CardTitle className="text-2xl font-bold">Step 2: Describe Your Goal</CardTitle>
          </CardHeader>
          <CardContent className="space-y-4">
            <p className="text-slate-400">Describe the target food product details in natural language.</p>
            <Input
              value={query}
              onChange={(e) => setQuery(e.target.value)}
              placeholder="e.g. A high-protein strawberry milk alternative with natural flavors..."
              className="bg-slate-950 border-slate-800 text-white"
            />
          </CardContent>
          <CardFooter className="flex justify-between">
            <Button variant="ghost" onClick={() => setStep(1)}>Back</Button>
            <Button onClick={() => setStep(3)} disabled={!query}>Continue</Button>
          </CardFooter>
        </Card>
      )}

      {step === 3 && (
        <Card className="bg-slate-900 border-slate-800 text-white shadow-xl">
          <CardHeader>
            <CardTitle className="text-2xl font-bold">Step 3: Define Macro & Ingredient Constraints</CardTitle>
          </CardHeader>
          <CardContent className="space-y-6">
            <div className="flex gap-6">
              <label className="flex items-center space-x-2">
                <Checkbox
                  checked={constraints.vegan}
                  onCheckedChange={(checked) => setConstraints({ ...constraints, vegan: !!checked })}
                />
                <span>Vegan</span>
              </label>
              <label className="flex items-center space-x-2">
                <Checkbox
                  checked={constraints.gluten_free}
                  onCheckedChange={(checked) => setConstraints({ ...constraints, gluten_free: !!checked })}
                />
                <span>Gluten-Free</span>
              </label>
            </div>
            
            <div className="space-y-2">
              <div className="flex justify-between text-sm">
                <span>Protein Target (g / 100g)</span>
                <span>{constraints.protein_target}g</span>
              </div>
              <Slider
                value={[constraints.protein_target]}
                max={100}
                onValueChange={(val) => setConstraints({ ...constraints, protein_target: val[0] })}
              />
            </div>

            <div className="space-y-2">
              <div className="flex justify-between text-sm">
                <span>Sugar Target (g / 100g)</span>
                <span>{constraints.sugar_target}g</span>
              </div>
              <Slider
                value={[constraints.sugar_target]}
                max={50}
                onValueChange={(val) => setConstraints({ ...constraints, sugar_target: val[0] })}
              />
            </div>
          </CardContent>
          <CardFooter className="flex justify-between">
            <Button variant="ghost" onClick={() => setStep(2)}>Back</Button>
            <Button onClick={handleGenerate} className="bg-indigo-600 hover:bg-indigo-700">
              {loading ? (
                <>
                  <Loader2 className="mr-2 h-4 w-4 animate-spin" />
                  Generating...
                </>
              ) : (
                'Generate Formulation'
              )}
            </Button>
          </CardFooter>
        </Card>
      )}

      {step === 5 && results && (
        <FormulationResults data={results} onReset={() => setStep(1)} />
      )}
    </div>
  );
}
```

### 2.3 Next.js Expert Canvas Component (`frontend/src/components/expert-canvas.tsx`)
```tsx
'use client';

import React, { useState, useEffect } from 'react';
import { Card, CardHeader, CardTitle, CardContent } from '@/components/ui/card';
import { Button } from '@/components/ui/button';
import { Slider } from '@/components/ui/slider';
import { Alert, AlertTitle, AlertDescription } from '@/components/ui/alert';
import { AlertCircle, PlusCircle, CheckCircle2 } from 'lucide-react';

export default function ExpertCanvas() {
  const [ingredients, setIngredients] = useState<any[]>([]);
  const [selected, setSelected] = useState<any[]>([]);
  const [macros, setMacros] = useState({ protein: 0, fat: 0, carbs: 0, calories: 0 });

  useEffect(() => {
    fetch(`${process.env.NEXT_PUBLIC_API_URL}/api/ingredients`)
      .then((res) => res.json())
      .then((data) => setIngredients(data));
  }, []);

  useEffect(() => {
    // Recalculate macros in real-time
    const totals = selected.reduce(
      (acc, ing) => {
        const factor = ing.percentage / 100;
        return {
          protein: acc.protein + ing.protein * factor,
          fat: acc.fat + ing.fat * factor,
          carbs: acc.carbs + ing.carbs * factor,
          calories: acc.calories + ing.calories * factor,
        };
      },
      { protein: 0, fat: 0, carbs: 0, calories: 0 }
    );
    setMacros(totals);
  }, [selected]);

  const addIngredient = (ing: any) => {
    if (selected.find((x) => x.name === ing.name)) return;
    setSelected([...selected, { ...ing, percentage: 5.0 }]);
  };

  const updatePercentage = (index: number, val: number) => {
    const next = [...selected];
    next[index].percentage = val;
    setSelected(next);
  };

  const totalPercentage = selected.reduce((sum, x) => sum + x.percentage, 0);

  return (
    <div className="grid grid-cols-12 gap-6 p-6">
      {/* Sidebar: Ingredient Selector */}
      <div className="col-span-4 space-y-4">
        <Card className="bg-slate-900 border-slate-800 text-white">
          <CardHeader>
            <CardTitle className="text-lg font-bold">Ingredient Catalog</CardTitle>
          </CardHeader>
          <CardContent className="h-[600px] overflow-y-auto space-y-2">
            {ingredients.map((ing) => (
              <div
                key={ing.name}
                className="flex items-center justify-between p-3 rounded-lg bg-slate-950 border border-slate-800 hover:border-slate-700 transition"
              >
                <div>
                  <div className="font-semibold text-sm">{ing.name}</div>
                  <div className="text-xs text-indigo-400 font-mono">{ing.functional_role}</div>
                </div>
                <Button variant="ghost" size="sm" onClick={() => addIngredient(ing)}>
                  <PlusCircle className="h-5 w-5 text-indigo-400" />
                </Button>
              </div>
            ))}
          </CardContent>
        </Card>
      </div>

      {/* Main Panel: Formulation Builder Canvas */}
      <div className="col-span-8 space-y-6">
        <Card className="bg-slate-900 border-slate-800 text-white">
          <CardHeader className="flex flex-row justify-between items-center">
            <CardTitle className="text-xl font-bold">Formulation Canvas</CardTitle>
            <div className="font-mono text-sm">
              Total Concentration:
              <span className={`ml-2 font-bold ${Math.abs(totalPercentage - 100) < 0.01 ? 'text-green-400' : 'text-yellow-500'}`}>
                {totalPercentage.toFixed(2)}%
              </span>
            </div>
          </CardHeader>
          <CardContent className="space-y-6">
            {selected.length === 0 ? (
              <div className="text-center py-12 text-slate-500">
                Click ingredients in the catalog to add them to your canvas.
              </div>
            ) : (
              selected.map((ing, idx) => (
                <div key={ing.name} className="p-4 rounded-lg bg-slate-950 border border-slate-800 space-y-2">
                  <div className="flex justify-between items-center">
                    <div>
                      <span className="font-bold text-sm">{ing.name}</span>
                      <span className="ml-2 text-xs px-2 py-0.5 rounded bg-slate-900 text-indigo-400 font-mono">
                        {ing.functional_role}
                      </span>
                    </div>
                    <span className="font-mono text-sm font-semibold">{ing.percentage.toFixed(1)}%</span>
                  </div>
                  <Slider
                    value={[ing.percentage]}
                    max={100}
                    step={0.1}
                    onValueChange={(val) => updatePercentage(idx, val[0])}
                  />
                </div>
              ))
            )}

            {/* Live Real-time Macros Summary */}
            <div className="grid grid-cols-4 gap-4 pt-6 border-t border-slate-800">
              <div className="p-4 rounded-lg bg-slate-950 text-center">
                <div className="text-xs text-slate-400">Protein</div>
                <div className="text-xl font-mono font-bold text-indigo-400">{macros.protein.toFixed(1)}g</div>
              </div>
              <div className="p-4 rounded-lg bg-slate-950 text-center">
                <div className="text-xs text-slate-400">Fat</div>
                <div className="text-xl font-mono font-bold text-pink-400">{macros.fat.toFixed(1)}g</div>
              </div>
              <div className="p-4 rounded-lg bg-slate-950 text-center">
                <div className="text-xs text-slate-400">Carbohydrates</div>
                <div className="text-xl font-mono font-bold text-yellow-500">{macros.carbs.toFixed(1)}g</div>
              </div>
              <div className="p-4 rounded-lg bg-slate-950 text-center">
                <div className="text-xs text-slate-400">Calories</div>
                <div className="text-xl font-mono font-bold text-emerald-400">{macros.calories.toFixed(0)} kcal</div>
              </div>
            </div>

            {Math.abs(totalPercentage - 100) > 0.01 && (
              <Alert variant="destructive" className="bg-yellow-950/20 border-yellow-800 text-yellow-400">
                <AlertCircle className="h-4 w-4" />
                <AlertTitle>Formulation Imbalance</AlertTitle>
                <AlertDescription>
                  Your total concentration must sum to exactly 100.00% for physical compounding simulation. Current: {totalPercentage.toFixed(2)}%
                </AlertDescription>
              </Alert>
            )}

            {Math.abs(totalPercentage - 100) <= 0.01 && (
              <div className="flex gap-4">
                <Button className="w-full bg-emerald-600 hover:bg-emerald-700">
                  <CheckCircle2 className="mr-2 h-4 w-4" /> Verify Compliance & Save
                </Button>
                <Button className="w-full bg-indigo-600 hover:bg-indigo-700">
                  Optimize via LLM RAG
                </Button>
              </div>
            )}
          </CardContent>
        </Card>
      </div>
    </div>
  );
}
```
