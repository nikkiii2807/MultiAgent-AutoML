from typing import TypedDict, Annotated, List, Dict, Any, Optional
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage, SystemMessage
from langgraph.graph import StateGraph, START, END
import pandas as pd
import json
import os
from dotenv import load_dotenv

load_dotenv()



# Using a standard typed dict for the state
class AgentState(TypedDict):
    messages: List[Dict[str, str]]
    current_step: str
    status: str
    dataset_info: Dict[str, Any]
    user_action: Optional[str]
    user_message: Optional[str]
    metrics: Dict[str, Any]

llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash-lite", convert_system_message_to_human=True) # use flash or pro

# --- Nodes ---

def ingestion_node(state: AgentState) -> AgentState:
    # First step: Read dataset info and suggest preprocessing
    info = state["dataset_info"]
    prompt = f"""You are an Expert Data Scientist AI assistant. The user has uploaded a dataset. Analyze it and provide a structured report.

**Dataset Overview:**
- Rows: {info.get('rows')}, Columns: {info.get('cols')}
- Column names: {info.get('columns')}
- Sample data (first 5 rows): {info.get('head')}
- Statistical summary: {info.get('summary')}

Respond in **well-structured Markdown** with:

## Dataset Overview
A brief summary of the dataset shape, purpose guess, and column types.

## Data Quality Assessment
- Missing values per column (if any)
- Potential outliers
- Data type issues

## Recommended Preprocessing Steps
A numbered list of specific actions to take (e.g., handle nulls, drop columns, encode categoricals).

Be concise but thorough. Use bold, bullet points, and tables where appropriate.
"""
    response = llm.invoke([HumanMessage(content=prompt)])
    state["messages"].append({"role": "assistant", "content": response.content})
    state["current_step"] = "preprocessing"
    state["status"] = "awaiting_human"
    return state

def preprocessing_node(state: AgentState) -> AgentState:
    usr_msg = state.get("user_message", "")
    action = state.get("user_action")
    
    if action == "approve":
        # In a real app we parse LLM's recommended actions and run Pandas here
        state["messages"].append({"role": "assistant", "content": "Great, preprocessing completed! I handled the missing values and dropped those columns as agreed. Should we move to Exploratory Data Analysis (EDA) next?"})
        state["current_step"] = "eda"
        state["status"] = "awaiting_human"
    else:
        # Ask LLM for new instruction based on user feedback
        prompt = f"User feedback on preprocessing: {usr_msg}. Provide an updated recommended plan."
        response = llm.invoke([HumanMessage(content=prompt)])
        state["messages"].append({"role": "assistant", "content": response.content})
    return state

def eda_node(state: AgentState) -> AgentState:
    action = state.get("user_action")
    if action == "approve":
        state["messages"].append({"role": "assistant", "content": "I generated the Exploratory Data Analysis charts (visible in the dashboard). Notice the high correlation between X and Y. Next, I recommend standardizing the numerical features and creating polynomial features. Shall I proceed?"})
        state["current_step"] = "standardization"
        state["status"] = "awaiting_human"
    else:
        prompt = f"User wants to explore something else: {state.get('user_message', '')}. Answer them and suggest next steps."
        response = llm.invoke([HumanMessage(content=prompt)])
        state["messages"].append({"role": "assistant", "content": response.content})
    return state

def standardization_node(state: AgentState) -> AgentState:
    action = state.get("user_action")
    if action == "approve":
        state["messages"].append({"role": "assistant", "content": "Features are standardized. We are ready for modeling. Based on this tabular data, I recommend testing a Random Forest Classifier and a Gradient Boosting Classifier. Which model would you prefer, or should I auto-select the best?"})
        state["current_step"] = "modeling"
        state["status"] = "awaiting_human"
    else:
         state["messages"].append({"role": "assistant", "content": f"Understood, user says: {state.get('user_message')}"})
    return state

def modeling_node(state: AgentState) -> AgentState:
    action = state.get("user_action")
    if action == "approve" or state.get("user_message"):
        # Simulated ML Training
        state["metrics"] = {"accuracy": 0.92, "f1_score": 0.90, "model_used": "Random Forest"}
        state["messages"].append({"role": "assistant", "content": f"The model has been built successfully! We achieved an accuracy of 92%. Review the metrics in the dashboard. The workflow is complete!"})
        state["current_step"] = "evaluation"
        state["status"] = "completed"
    return state

# --- Build the Graph ---
workflow = StateGraph(AgentState)

workflow.add_node("ingestion", ingestion_node)
workflow.add_node("preprocessing", preprocessing_node)
workflow.add_node("eda", eda_node)
workflow.add_node("standardization", standardization_node)
workflow.add_node("modeling", modeling_node)

workflow.add_edge(START, "ingestion")

# Transitions: The graph flows linearly but pauses. Instead of complex conditional edges for this demo, 
# in FastAPI we will just route to the correct node based on 'current_step'.
# For pure LangGraph flow with human-in-the-loop, we use `interrupt`. Here, to keep the API stateless-ish,
# we manually jump to the node matching the `current_step` in `process_agent_workflow`.

compiled_graph = workflow.compile()

async def process_agent_workflow(session_id: str, current_state: dict, user_message: str, action: str, data_store_ref: dict) -> dict:
    """Wrapper to run the appropriate node based on the state."""
    
    # Inject user input into state
    agent_state: AgentState = {
        "messages": current_state.get("messages", []),
        "current_step": current_state.get("current_step", "ingestion"),
        "status": current_state.get("status", "working"),
        "dataset_info": current_state.get("dataset_info", {}),
        "user_action": action,
        "user_message": user_message,
        "metrics": current_state.get("metrics", {})
    }
    
    # Determine which node to execute
    step = agent_state["current_step"]
    
    # Very simple routing mechanism for our API-driven approach
    if step == "ingestion":
        new_state = ingestion_node(agent_state)
    elif step == "preprocessing":
         new_state = preprocessing_node(agent_state)
    elif step == "eda":
         new_state = eda_node(agent_state)
    elif step == "standardization":
         new_state = standardization_node(agent_state)
    elif step == "modeling":
         new_state = modeling_node(agent_state)
    else:
        new_state = agent_state # Done
        
    return dict(new_state)
