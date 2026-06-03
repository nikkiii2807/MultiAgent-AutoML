from __future__ import annotations

from datetime import datetime, timezone
import json
import shutil
import sqlite3
from pathlib import Path
from typing import Any, Dict, List, Tuple

import pandas as pd


BASE_DIR = Path(__file__).resolve().parent
STORAGE_DIR = BASE_DIR / "storage"
SESSIONS_DIR = STORAGE_DIR / "sessions"
DB_PATH = STORAGE_DIR / "sessions.db"


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def ensure_storage() -> None:
    STORAGE_DIR.mkdir(parents=True, exist_ok=True)
    SESSIONS_DIR.mkdir(parents=True, exist_ok=True)
    with sqlite3.connect(DB_PATH) as connection:
        connection.execute(
            """
            CREATE TABLE IF NOT EXISTS sessions (
                id TEXT PRIMARY KEY,
                title TEXT NOT NULL,
                filename TEXT NOT NULL,
                file_extension TEXT NOT NULL,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL,
                current_step TEXT NOT NULL,
                status TEXT NOT NULL,
                dataset_info TEXT NOT NULL,
                metrics TEXT NOT NULL,
                step_results TEXT NOT NULL,
                current_dataset_step TEXT NOT NULL,
                artifact_dir TEXT NOT NULL
            )
            """
        )
        connection.execute(
            """
            CREATE TABLE IF NOT EXISTS messages (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id TEXT NOT NULL,
                role TEXT NOT NULL,
                content TEXT NOT NULL,
                action TEXT,
                created_at TEXT NOT NULL,
                FOREIGN KEY (session_id) REFERENCES sessions(id)
            )
            """
        )
        connection.execute(
            "CREATE INDEX IF NOT EXISTS idx_messages_session_id ON messages(session_id, id)"
        )
        connection.commit()


def get_session_dir(session_id: str) -> Path:
    return SESSIONS_DIR / session_id


def get_source_path(session_id: str, extension: str) -> Path:
    return get_session_dir(session_id) / f"source{extension}"


def get_datasets_dir(session_id: str) -> Path:
    return get_session_dir(session_id) / "datasets"


def get_artifacts_dir(session_id: str) -> Path:
    return get_session_dir(session_id) / "artifacts"


def build_session_title(filename: str) -> str:
    stem = Path(filename).stem.replace("_", " ").strip() or "Dataset"
    stamp = datetime.now().strftime("%Y-%m-%d %H:%M")
    return f"{stem} - {stamp}"


def _json_dump(data: Any) -> str:
    return json.dumps(data, ensure_ascii=True)


def _json_load(payload: str) -> Any:
    return json.loads(payload) if payload else {}


def create_session_record(
    session_id: str,
    *,
    filename: str,
    file_extension: str,
    dataset_info: Dict[str, Any],
    df: pd.DataFrame,
    source_bytes: bytes,
) -> Dict[str, Any]:
    ensure_storage()
    session_dir = get_session_dir(session_id)
    if session_dir.exists():
        shutil.rmtree(session_dir)
    get_datasets_dir(session_id).mkdir(parents=True, exist_ok=True)
    get_artifacts_dir(session_id).mkdir(parents=True, exist_ok=True)
    get_source_path(session_id, file_extension).write_bytes(source_bytes)
    df.to_csv(get_datasets_dir(session_id) / "ingestion.csv", index=False)

    created_at = utc_now()
    row = {
        "id": session_id,
        "title": build_session_title(filename),
        "filename": filename,
        "file_extension": file_extension,
        "created_at": created_at,
        "updated_at": created_at,
        "current_step": "ingestion",
        "status": "awaiting_human",
        "dataset_info": dataset_info,
        "metrics": {},
        "step_results": {},
        "current_dataset_step": "ingestion",
        "artifact_dir": str(session_dir),
    }
    with sqlite3.connect(DB_PATH) as connection:
        connection.execute(
            """
            INSERT INTO sessions (
                id, title, filename, file_extension, created_at, updated_at, current_step, status,
                dataset_info, metrics, step_results, current_dataset_step, artifact_dir
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                row["id"],
                row["title"],
                row["filename"],
                row["file_extension"],
                row["created_at"],
                row["updated_at"],
                row["current_step"],
                row["status"],
                _json_dump(row["dataset_info"]),
                _json_dump(row["metrics"]),
                _json_dump(row["step_results"]),
                row["current_dataset_step"],
                row["artifact_dir"],
            ),
        )
        connection.commit()
    return row


def persist_session_state(
    session_id: str,
    *,
    state: Dict[str, Any],
    data_store: Dict[str, Any],
) -> None:
    ensure_storage()
    updated_at = utc_now()
    artifact_dir = get_session_dir(session_id)
    datasets_dir = get_datasets_dir(session_id)
    artifacts_dir = get_artifacts_dir(session_id)
    datasets_dir.mkdir(parents=True, exist_ok=True)
    artifacts_dir.mkdir(parents=True, exist_ok=True)

    artifact_datasets = data_store.get("artifacts", {}).get("datasets", {})
    artifact_outputs = data_store.get("artifacts", {}).get("step_outputs", {})
    valid_dataset_files = set()
    valid_artifact_files = set()

    for stage, frame in artifact_datasets.items():
        dataset_path = datasets_dir / f"{stage}.csv"
        frame.to_csv(dataset_path, index=False)
        valid_dataset_files.add(dataset_path.name)

    for stage, payload in artifact_outputs.items():
        artifact_path = artifacts_dir / f"{stage}.json"
        artifact_path.write_text(_json_dump(payload), encoding="utf-8")
        valid_artifact_files.add(artifact_path.name)

    for existing_path in datasets_dir.glob("*.csv"):
        if existing_path.name not in valid_dataset_files and existing_path.name != "source.csv":
            existing_path.unlink(missing_ok=True)

    for existing_path in artifacts_dir.glob("*.json"):
        if existing_path.name not in valid_artifact_files:
            existing_path.unlink(missing_ok=True)

    messages = state.get("messages", [])
    with sqlite3.connect(DB_PATH) as connection:
        connection.execute(
            """
            UPDATE sessions
            SET updated_at = ?, current_step = ?, status = ?, dataset_info = ?, metrics = ?,
                step_results = ?, current_dataset_step = ?, artifact_dir = ?
            WHERE id = ?
            """,
            (
                updated_at,
                state.get("current_step", "ingestion"),
                state.get("status", "awaiting_human"),
                _json_dump(state.get("dataset_info", {})),
                _json_dump(state.get("metrics", {})),
                _json_dump(state.get("step_results", {})),
                data_store.get("current_dataset_step", "ingestion"),
                str(artifact_dir),
                session_id,
            ),
        )
        connection.execute("DELETE FROM messages WHERE session_id = ?", (session_id,))
        connection.executemany(
            """
            INSERT INTO messages (session_id, role, content, action, created_at)
            VALUES (?, ?, ?, ?, ?)
            """,
            [
                (
                    session_id,
                    message.get("role", "assistant"),
                    message.get("content", ""),
                    message.get("action"),
                    message.get("created_at", updated_at),
                )
                for message in messages
            ],
        )
        connection.commit()


def _load_messages(connection: sqlite3.Connection, session_id: str) -> List[Dict[str, Any]]:
    rows = connection.execute(
        """
        SELECT role, content, action, created_at
        FROM messages
        WHERE session_id = ?
        ORDER BY id ASC
        """,
        (session_id,),
    ).fetchall()
    return [
        {
            "role": row[0],
            "content": row[1],
            "action": row[2],
            "created_at": row[3],
        }
        for row in rows
    ]


def get_session_row(session_id: str) -> Dict[str, Any] | None:
    ensure_storage()
    with sqlite3.connect(DB_PATH) as connection:
        row = connection.execute(
            """
            SELECT id, title, filename, file_extension, created_at, updated_at, current_step, status,
                   dataset_info, metrics, step_results, current_dataset_step, artifact_dir
            FROM sessions
            WHERE id = ?
            """,
            (session_id,),
        ).fetchone()
        if row is None:
            return None
        messages = _load_messages(connection, session_id)

    return {
        "id": row[0],
        "title": row[1],
        "filename": row[2],
        "file_extension": row[3],
        "created_at": row[4],
        "updated_at": row[5],
        "current_step": row[6],
        "status": row[7],
        "dataset_info": _json_load(row[8]),
        "metrics": _json_load(row[9]),
        "step_results": _json_load(row[10]),
        "current_dataset_step": row[11],
        "artifact_dir": row[12],
        "messages": messages,
    }


def list_sessions() -> List[Dict[str, Any]]:
    ensure_storage()
    with sqlite3.connect(DB_PATH) as connection:
        rows = connection.execute(
            """
            SELECT s.id, s.title, s.filename, s.current_step, s.status, s.updated_at,
                   COALESCE((
                       SELECT substr(m.content, 1, 160)
                       FROM messages m
                       WHERE m.session_id = s.id
                       ORDER BY m.id DESC
                       LIMIT 1
                   ), '')
            FROM sessions s
            ORDER BY s.updated_at DESC
            """
        ).fetchall()
    return [
        {
            "id": row[0],
            "title": row[1],
            "filename": row[2],
            "current_step": row[3],
            "status": row[4],
            "updated_at": row[5],
            "preview": row[6],
        }
        for row in rows
    ]


def load_session_data(session_id: str) -> Tuple[Dict[str, Any], Dict[str, Any]] | None:
    session = get_session_row(session_id)
    if session is None:
        return None

    datasets: Dict[str, pd.DataFrame] = {}
    artifacts: Dict[str, Any] = {}
    datasets_dir = get_datasets_dir(session_id)
    artifacts_dir = get_artifacts_dir(session_id)

    for csv_path in datasets_dir.glob("*.csv"):
        datasets[csv_path.stem] = pd.read_csv(csv_path)

    for artifact_path in artifacts_dir.glob("*.json"):
        artifacts[artifact_path.stem] = _json_load(artifact_path.read_text(encoding="utf-8"))

    current_stage = session.get("current_dataset_step", "ingestion")
    current_df = datasets.get(current_stage)
    if current_df is None:
        source_path = get_source_path(session_id, session["file_extension"])
        if source_path.suffix.lower() == ".xlsx":
            current_df = pd.read_excel(source_path)
        else:
            current_df = pd.read_csv(source_path)

    data_store = {
        "raw": datasets.get("ingestion", current_df.copy()),
        "current": current_df.copy(),
        "filename": session["filename"],
        "artifacts": {
            "datasets": {key: value.copy() for key, value in datasets.items()},
            "step_outputs": artifacts,
        },
        "current_dataset_step": current_stage,
    }
    state = {
        "messages": session.get("messages", []),
        "current_step": session.get("current_step", "ingestion"),
        "status": session.get("status", "awaiting_human"),
        "dataset_info": session.get("dataset_info", {}),
        "metrics": session.get("metrics", {}),
        "step_results": session.get("step_results", {}),
        "title": session.get("title"),
        "filename": session.get("filename"),
    }
    return state, data_store


def build_session_payload(session_id: str) -> Dict[str, Any]:
    loaded = load_session_data(session_id)
    if loaded is None:
        raise KeyError(session_id)

    state, data_store = loaded
    current_df = data_store["current"]
    return {
        "session_id": session_id,
        "title": state.get("title"),
        "filename": state.get("filename"),
        "messages": state.get("messages", []),
        "dataset_info": state.get("dataset_info", {}),
        "current_step": state.get("current_step", "ingestion"),
        "state": state.get("current_step", "ingestion"),
        "status": state.get("status", "awaiting_human"),
        "metrics": state.get("metrics", {}),
        "step_results": state.get("step_results", {}),
        "data_preview": current_df.head(10).to_json(orient="records") if current_df is not None else None,
    }
