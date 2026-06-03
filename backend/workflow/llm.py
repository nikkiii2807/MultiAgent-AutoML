from __future__ import annotations

from typing import Optional

from dotenv import load_dotenv

load_dotenv()

try:
    from langchain_google_genai import ChatGoogleGenerativeAI
    from langchain_core.messages import HumanMessage
except Exception:  # pragma: no cover
    ChatGoogleGenerativeAI = None
    HumanMessage = None


_llm: Optional["ChatGoogleGenerativeAI"] = None


def get_llm() -> Optional["ChatGoogleGenerativeAI"]:
    global _llm
    if _llm is not None:
        return _llm

    if ChatGoogleGenerativeAI is None:
        return None

    try:
        _llm = ChatGoogleGenerativeAI(
            model="gemini-2.5-flash-lite",
            convert_system_message_to_human=True,
        )
    except Exception:
        _llm = None
    return _llm


def invoke_llm(prompt: str, fallback: str) -> str:
    llm = get_llm()
    if llm is None or HumanMessage is None:
        return fallback

    try:
        response = llm.invoke([HumanMessage(content=prompt)])
        return response.content if getattr(response, "content", None) else fallback
    except Exception:
        return fallback
