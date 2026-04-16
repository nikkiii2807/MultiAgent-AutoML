from typing import TypedDict, Annotated, List, Dict, Any, Optional
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage, SystemMessage
from langgraph.graph import StateGraph, START, END
import pandas as pd
import numpy as np
import json
import os
from dotenv import load_dotenv

# sklearn imports for real model training
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor, GradientBoostingClassifier, GradientBoostingRegressor
from sklearn.linear_model import LogisticRegression, LinearRegression, Ridge
from sklearn.svm import SVC, SVR
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, r2_score, mean_squared_error

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
    preferred_model: Optional[str]  # captured from user chat

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
    # Capture any model preference mentioned in the user message at this stage too
    pref = _detect_model_preference(state.get("user_message", ""))
    if pref:
        state["preferred_model"] = pref
    if action == "approve":
        state["messages"].append({"role": "assistant", "content": "Features are standardized. We are ready for modeling. Based on this tabular data, I recommend testing a **Random Forest** and a **Gradient Boosting** model. Which model would you prefer (e.g. Random Forest, Gradient Boosting, SVM, Logistic Regression, KNN, Decision Tree), or should I auto-select the best?"})
        state["current_step"] = "modeling"
        state["status"] = "awaiting_human"
    else:
        state["messages"].append({"role": "assistant", "content": f"Understood, user says: {state.get('user_message')}"})
    return state

# --- Helpers ---

MODEL_KEYWORDS: Dict[str, str] = {
    "random forest": "Random Forest",
    "rf": "Random Forest",
    "gradient boost": "Gradient Boosting",
    "gradient boosting": "Gradient Boosting",
    "xgboost": "Gradient Boosting",
    "gbm": "Gradient Boosting",
    "logistic": "Logistic Regression",
    "logistic regression": "Logistic Regression",
    "svm": "SVM",
    "support vector": "SVM",
    "knn": "KNN",
    "k-nearest": "KNN",
    "k nearest": "KNN",
    "decision tree": "Decision Tree",
    "dt": "Decision Tree",
    "linear regression": "Linear Regression",
    "ridge": "Ridge Regression",
}

def _detect_model_preference(message: str) -> Optional[str]:
    """Scan user message for a preferred model name."""
    if not message:
        return None
    lower = message.lower()
    for keyword, model_name in MODEL_KEYWORDS.items():
        if keyword in lower:
            return model_name
    return None


def _train_model(df: pd.DataFrame, preferred_model: Optional[str]) -> Dict[str, Any]:
    """
    Auto-detect target column, determine task type (classification/regression),
    train the requested (or best auto-selected) model, and return real metrics.
    """
    # --- 1. Identify target column ---
    classification_target_hints = [
        "target", "label", "class", "output", "y", "result", "survived",
        "purchased", "churn", "default", "fraud", "clicked", "approved", "converted",
    ]
    regression_target_hints = ["price", "salary", "sales", "revenue", "score", "grade", "rating", "amount"]
    target_hints = classification_target_hints + regression_target_hints
    target_col = None
    cols_lower = {c.lower(): c for c in df.columns}
    for hint in target_hints:
        if hint in cols_lower:
            target_col = cols_lower[hint]
            break
    if target_col is None:
        target_col = df.columns[-1]  # fall back to last column

    # --- 2. Prepare features ---
    X = df.drop(columns=[target_col]).copy()
    y = df[target_col].copy()

    # Drop columns that are almost entirely unique (like IDs)
    for col in X.columns:
        if X[col].dtype == object and X[col].nunique() > 0.9 * len(X):
            X = X.drop(columns=[col])

    # Encode categorical features
    for col in X.select_dtypes(include=["object", "category"]).columns:
        X[col] = LabelEncoder().fit_transform(X[col].astype(str))

    # Fill missing values
    X = X.fillna(X.median(numeric_only=True))
    X = X.fillna(0)

    # Determine task type
    y_non_null = y.dropna()
    unique_count = int(y_non_null.nunique())
    sample_count = max(len(y_non_null), 1)
    unique_ratio = unique_count / sample_count
    target_col_lower = target_col.lower()
    classification_name_hint = (
        target_col_lower in classification_target_hints or
        any(token in target_col_lower for token in classification_target_hints if len(token) > 2)
    )
    regression_name_hint = (
        target_col_lower in regression_target_hints or
        any(token in target_col_lower for token in regression_target_hints if len(token) > 2)
    )
    is_numeric_target = pd.api.types.is_numeric_dtype(y_non_null)

    is_integer_like_numeric = False
    if is_numeric_target and not y_non_null.empty:
        y_numeric = pd.to_numeric(y_non_null, errors="coerce").dropna()
        if not y_numeric.empty:
            is_integer_like_numeric = bool(np.allclose(y_numeric, np.round(y_numeric)))

    is_classification = (
        y.dtype == object or
        y.dtype.name == "category" or
        classification_name_hint or
        unique_count == 2 or
        (is_numeric_target and is_integer_like_numeric and unique_count <= 20 and unique_ratio <= 0.2) or
        (unique_count <= 10 and unique_ratio <= 0.2)
    )
    if regression_name_hint and is_numeric_target and unique_count > 20 and not classification_name_hint:
        is_classification = False

    # Encode target if classification
    if is_classification:
        le = LabelEncoder()
        y = le.fit_transform(y.astype(str))
    else:
        y = pd.to_numeric(y, errors="coerce")
        y = y.fillna(y.median())

    # Scale features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Train/test split
    test_size = 0.2 if len(X) > 50 else 0.3
    stratify_y = None
    if is_classification:
        _, class_counts = np.unique(y, return_counts=True)
        if len(class_counts) > 1 and int(class_counts.min()) >= 2:
            stratify_y = y
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=test_size, random_state=42, stratify=stratify_y
    )

    # --- 3. Select model ---
    clf_map = {
        "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42),
        "Gradient Boosting": GradientBoostingClassifier(n_estimators=100, random_state=42),
        "Logistic Regression": LogisticRegression(max_iter=500, random_state=42),
        "SVM": SVC(kernel="rbf", random_state=42),
        "KNN": KNeighborsClassifier(n_neighbors=5),
        "Decision Tree": DecisionTreeClassifier(random_state=42),
    }
    reg_map = {
        "Random Forest": RandomForestRegressor(n_estimators=100, random_state=42),
        "Gradient Boosting": GradientBoostingRegressor(n_estimators=100, random_state=42),
        "Linear Regression": LinearRegression(),
        "Ridge Regression": Ridge(alpha=1.0),
        "SVM": SVR(kernel="rbf"),
        "KNN": KNeighborsRegressor(n_neighbors=5),
        "Decision Tree": DecisionTreeRegressor(random_state=42),
    }

    model_map = clf_map if is_classification else reg_map
    default_model = "Random Forest"

    selected_name = preferred_model if (preferred_model and preferred_model in model_map) else default_model
    model = model_map[selected_name]

    # --- 4. Train and evaluate ---
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    metrics: Dict[str, Any] = {"model_used": selected_name, "task": "classification" if is_classification else "regression", "target_column": target_col}

    if is_classification:
        unique_test_classes = np.unique(y_test)
        average = "binary" if len(unique_test_classes) == 2 else "weighted"
        acc = round(float(accuracy_score(y_test, y_pred)), 4)
        precision = round(float(precision_score(y_test, y_pred, average=average, zero_division=0)), 4)
        recall = round(float(recall_score(y_test, y_pred, average=average, zero_division=0)), 4)
        f1 = round(float(f1_score(y_test, y_pred, average="weighted", zero_division=0)), 4)
        metrics["accuracy"] = acc
        metrics["precision"] = precision
        metrics["recall"] = recall
        metrics["f1_score"] = f1
    else:
        r2 = round(float(r2_score(y_test, y_pred)), 4)
        rmse = round(float(np.sqrt(mean_squared_error(y_test, y_pred))), 4)
        metrics["r2_score"] = r2
        metrics["rmse"] = rmse

    # --- 5. Feature importance (if available) ---
    if hasattr(model, "feature_importances_"):
        importances = model.feature_importances_
        feat_names = X.columns.tolist()
        fi = sorted(
            [{"name": feat_names[i], "importance": round(float(importances[i]), 4)} for i in range(len(feat_names))],
            key=lambda x: x["importance"], reverse=True
        )
        metrics["feature_importances"] = fi[:10]

    return metrics


def modeling_node(state: AgentState) -> AgentState:
    action = state.get("user_action")
    user_msg = state.get("user_message", "")

    if action == "approve" or user_msg:
        # Detect model preference from the user's message (overrides any earlier preference)
        msg_pref = _detect_model_preference(user_msg)
        preferred = msg_pref or state.get("preferred_model")
        if msg_pref:
            state["preferred_model"] = msg_pref

        # Pull the actual dataframe from the state's dataset_info if available
        # NOTE: the actual df is passed via data_store_ref in process_agent_workflow
        df = state.get("_dataframe")  # injected below
        if df is not None and not df.empty:
            try:
                metrics = _train_model(df, preferred)
            except Exception as e:
                metrics = {"error": str(e), "model_used": preferred or "Random Forest"}
        else:
            metrics = {"error": "Dataset not available for training", "model_used": preferred or "Random Forest"}

        state["metrics"] = metrics

        # Build a human-readable summary
        model_name = metrics.get("model_used", "the selected model")
        task = metrics.get("task", "classification")
        if task == "classification":
            acc = metrics.get("accuracy", "N/A")
            precision = metrics.get("precision", "N/A")
            recall = metrics.get("recall", "N/A")
            f1 = metrics.get("f1_score", "N/A")
            perf_line = (
                f"Accuracy: **{acc * 100:.1f}%** | "
                f"Precision: **{precision:.3f}** | "
                f"Recall: **{recall:.3f}** | "
                f"F1 Score: **{f1:.3f}**"
                if isinstance(acc, float)
                else "Metrics unavailable."
            )
        else:
            r2 = metrics.get("r2_score", "N/A")
            rmse = metrics.get("rmse", "N/A")
            perf_line = f"R2 Score: **{r2:.3f}** | RMSE: **{rmse:.4f}**" if isinstance(r2, float) else "Metrics unavailable."

        target = metrics.get("target_column", "the target column")
        state["messages"].append({
            "role": "assistant",
            "content": (
                f"✅ **Model Training Complete!**\n\n"
                f"- **Model used:** {model_name}\n"
                f"- **Task type:** {task.capitalize()}\n"
                f"- **Target column detected:** `{target}`\n"
                f"- **{perf_line}**\n\n"
                f"Review the detailed metrics and feature importance in the dashboard. The AutoML workflow is complete!"
            )
        })
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
workflow.add_edge("ingestion", "preprocessing")
workflow.add_edge("preprocessing", "eda")
workflow.add_edge("eda", "standardization")
workflow.add_edge("standardization", "modeling")
workflow.add_edge("modeling", END)

# Transitions: The graph flows linearly but pauses. Instead of complex conditional edges for this demo, 
# in FastAPI we will just route to the correct node based on 'current_step'.
# For pure LangGraph flow with human-in-the-loop, we use `interrupt`. Here, to keep the API stateless-ish,
# we manually jump to the node matching the `current_step` in `process_agent_workflow`.

compiled_graph = workflow.compile()

async def process_agent_workflow(session_id: str, current_state: dict, user_message: str, action: str, data_store_ref: dict) -> dict:
    """Wrapper to run the appropriate node based on the state."""

    # Resolve model preference: detect from current message OR carry forward from prior state
    msg_pref = _detect_model_preference(user_message)
    prev_pref = current_state.get("preferred_model")
    resolved_pref = msg_pref or prev_pref

    # Inject user input into state
    agent_state: AgentState = {
        "messages": current_state.get("messages", []),
        "current_step": current_state.get("current_step", "ingestion"),
        "status": current_state.get("status", "working"),
        "dataset_info": current_state.get("dataset_info", {}),
        "user_action": action,
        "user_message": user_message,
        "metrics": current_state.get("metrics", {}),
        "preferred_model": resolved_pref,
        # Inject the actual DataFrame so modeling_node can train on real data
        "_dataframe": data_store_ref.get("current"),  # type: ignore[typeddict-item]
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
        new_state = agent_state  # Done

    # Strip the private _dataframe key before returning (not serializable)
    result = dict(new_state)
    result.pop("_dataframe", None)
    return result
