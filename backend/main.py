from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional
import pandas as pd
import io
import uuid

from session_store import (
    build_session_payload,
    create_session_record,
    ensure_storage,
    list_sessions,
    load_session_data,
    persist_session_state,
)
from workflow import PIPELINE_ORDER
from workflow.charting import generate_charts
from workflow.helpers import timestamp


# Initialize FastAPI app
app = FastAPI(title="Multi-Agent AutoML API")

# Setup CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

DATA_STORE = {}
STATE_STORE = {}


@app.on_event("startup")
def startup() -> None:
    ensure_storage()


@app.get("/")
def read_root():
    return {"message": "Multi-Agent AutoML Backend is running."}

@app.post("/upload")
async def upload_dataset(file: UploadFile = File(...)):
    filename_lower = file.filename.lower()
    file_extension = "." + file.filename.split(".")[-1].lower()
    if not (filename_lower.endswith(".csv") or filename_lower.endswith(".xlsx")):
        raise HTTPException(
            status_code=400,
            detail="Unsupported file type. Please upload a .csv or .xlsx file."
        )

    try:
        contents = await file.read()
        if filename_lower.endswith(".xlsx"):
            df = pd.read_excel(io.BytesIO(contents))
        else:
            df = pd.read_csv(io.BytesIO(contents))
        
        session_id = str(uuid.uuid4())
        dataset_info = {
            "rows": df.shape[0],
            "cols": df.shape[1],
            "columns": df.columns.tolist(),
            "head": df.head(5).to_json(orient="records"),
            "summary": df.describe(include='all').to_json()
        }
        
        DATA_STORE[session_id] = {
            "raw": df,
            "current": df.copy(),
            "filename": file.filename,
            "artifacts": {
                "datasets": {"ingestion": df.copy()},
                "step_outputs": {},
            },
            "current_dataset_step": "ingestion",
        }
        
        STATE_STORE[session_id] = {
            "messages": [],
            "current_step": "ingestion",
            "dataset_info": dataset_info,
            "status": "awaiting_human",
            "step_results": {},
            "metrics": {},
        }
        session_row = create_session_record(
            session_id,
            filename=file.filename,
            file_extension=file_extension,
            dataset_info=dataset_info,
            df=df,
            source_bytes=contents,
        )
        STATE_STORE[session_id]["title"] = session_row["title"]
        STATE_STORE[session_id]["filename"] = session_row["filename"]
        
        return {
            "message": "File uploaded successfully",
            "session_id": session_id,
            "title": session_row["title"],
            "filename": file.filename,
            "columns": df.columns.tolist(),
            "rows": df.shape[0],
            "dataset_info": dataset_info,
            "current_step": "ingestion",
            "state": "ingestion",
            "status": "awaiting_human",
            "step_results": {},
            "messages": [],
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing file: {str(e)}")

class ChatMessage(BaseModel):
    message: str
    session_id: str
    action: Optional[str] = None

@app.post("/chat")
async def chat(data: ChatMessage):
    session_id = data.session_id
    if session_id not in STATE_STORE or session_id not in DATA_STORE:
        loaded = load_session_data(session_id)
        if loaded is None:
            raise HTTPException(status_code=404, detail="Session not found")
        STATE_STORE[session_id], DATA_STORE[session_id] = loaded

    if session_id not in STATE_STORE:
        raise HTTPException(status_code=404, detail="Session not found")
        
    state = STATE_STORE[session_id]
    state["messages"].append(
        {
            "role": "user",
            "content": data.message,
            "action": data.action,
            "created_at": timestamp(),
        }
    )
    
    from agent import process_agent_workflow
    
    new_state = await process_agent_workflow(session_id, state, data.message, data.action, DATA_STORE[session_id])
    STATE_STORE[session_id] = new_state
    persist_session_state(session_id, state=new_state, data_store=DATA_STORE[session_id])
    
    latest_msg = new_state["messages"][-1] if new_state["messages"] else {"role": "assistant", "content": "Error"}
    
    current_df = DATA_STORE[session_id]["current"]
    current_step = new_state["current_step"]
    
    step_results = new_state.get("step_results", {})
    result_step = current_step if current_step in step_results else current_step
    if current_step not in step_results and current_step in PIPELINE_ORDER:
        step_index = PIPELINE_ORDER.index(current_step)
        previous_step = PIPELINE_ORDER[step_index - 1] if step_index > 0 else current_step
        if previous_step in step_results:
            result_step = previous_step
    selected_result = step_results.get(
        result_step,
        {
            "analysis": latest_msg["content"],
            "data_preview": current_df.head(10).to_json(orient="records") if current_df is not None else None,
            "metrics": new_state.get("metrics", {}),
            "charts": generate_charts(current_df, result_step if result_step in PIPELINE_ORDER else "ingestion"),
        },
    )
    
    return {
        "session_id": session_id,
        "reply": selected_result.get("analysis", latest_msg["content"]),
        "state": new_state["current_step"],
        "current_step": new_state["current_step"],
        "status": new_state["status"],
        "data_preview": selected_result.get("data_preview"),
        "metrics": selected_result.get("metrics", {}),
        "charts": selected_result.get("charts", []),
        "step_results": step_results,
        "result_step": result_step,
        "messages": new_state.get("messages", []),
    }

@app.get("/sessions")
async def get_sessions():
    return list_sessions()

@app.get("/sessions/{session_id}")
async def get_session(session_id: str):
    try:
        payload = build_session_payload(session_id)
    except KeyError:
        raise HTTPException(status_code=404, detail="Session not found")
    return payload

@app.get("/state/{session_id}")
async def get_state(session_id: str):
    try:
        payload = build_session_payload(session_id)
    except KeyError:
        raise HTTPException(status_code=404, detail="Session not found")
    return payload
