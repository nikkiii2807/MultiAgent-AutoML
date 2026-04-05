# 🤖 Multi-Agent AutoML Studio

An end-to-end, human-in-the-loop multi-agent system built to automate comprehensive Data Science workflows.

AutoML Studio uses a team of AI Agents (powered by Google's Gemini models and LangGraph) to analyze raw CSV datasets, perform data cleaning, execute exploratory data analysis (EDA), engineer features, and train machine learning models. Built with a premium Dark Mode Next.js UI, the system visualizes the dataset dynamically at every step of the workflow.

---

## ✨ Features

- **Multi-Agent Orchestration**: Specialized agents handle unique parts of the ML lifecycle (Ingestion, Preprocessing, EDA, Standardization, and Modeling).
- **Human-in-the-Loop (HITL)**: Control the AI. The LangGraph state machine pauses execution at every major decision point to ask for your approval or custom instructions via a chat interface.
- **Dynamic Visualizer Playground**: Real-time rendering of data tables, correlation matrices, pie charts, and distribution histograms using Recharts.
- **Two-Tier Architecture**: A robust Python backend (FastAPI, Pandas, Scikit-learn) paired with a blazingly fast Next.js App Router frontend.

---

## 🛠️ Tech Stack

**Frontend**
* [Next.js](https://nextjs.org/) (App Router, React)
* Premium Bespoke Vanilla CSS (Glassmorphism & CSS Variables)
* [Recharts](https://recharts.org/) (Dynamic chart rendering)
* React-Markdown (Structured LLM text formatting)

**Backend**
* [Python 3](https://www.python.org/)
* [FastAPI](https://fastapi.tiangolo.com/) (Async API layer)
* [LangGraph](https://python.langchain.com/docs/langgraph/) & [LangChain](https://www.langchain.com/) (Agent State Machine workflow)
* Google GenAI (`gemini-2.5-flash-lite`)
* Pandas & Scikit-learn (Data handling and simulated ML)

---

## 🚀 Getting Started

Follow these steps to run the complete stack locally using the integrated Concurrent runner.

### 1. Requirements
* Node.js (v18+)
* Python (3.9+)

### 2. Installation Setup

**Clone and Install Root Dependencies**
```bash
git clone <repository-url>
cd Multiagent-AutoML
npm install
```

**Install Frontend Dependencies**
```bash
cd frontend
npm install
cd ..
```

**Install Backend Dependencies & Virtual Environment**
```bash
cd backend
python -m venv venv

# Activate the venv (Windows):
.\venv\Scripts\Activate.ps1
# Or (Mac/Linux):
# source venv/bin/activate

pip install -r requirements.txt
cd ..
```

### 3. Environment Variables
Create a `.env` file in the `backend/` directory and add your Google Gemini API key:
```env
# backend/.env
GOOGLE_API_KEY="your_api_key_here"
```

### 4. Running the App
We use `concurrently` to spin up both the FastAPI backend and the Next.js frontend within a single terminal instance. 

From the **root directory**, simply run:
```bash
npm start
```
* **Frontend** will be available at: `http://localhost:3000`
* **Backend API** will be available at: `http://localhost:8000`

---

## 🧪 Testing the Workflow

1. Once the application is live, navigate to `http://localhost:3000`.
2. A `sample_data.csv` is provided in the root repository. Click on "Upload CSV Dataset" in the chat panel to upload it.
3. Chat with the AutoML Agent, approve its preprocessing suggestions, and observe the charts building out your data pipeline!
