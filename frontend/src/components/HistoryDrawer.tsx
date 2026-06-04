"use client";

import { useEffect, type RefObject } from "react";
import SessionHistoryList from "./SessionHistoryList";
import type { SessionSummary } from "../lib/studio";

interface HistoryDrawerProps {
  activeSessionId: string | null;
  drawerRef: RefObject<HTMLElement | null>;
  isLoading?: boolean;
  isOpen: boolean;
  onClose: () => void;
  onNewSession: () => void;
  onSelectSession: (sessionId: string) => void;
  sessions: SessionSummary[];
  toggleButtonRef: RefObject<HTMLButtonElement | null>;
  workspaceRef: RefObject<HTMLDivElement | null>;
}

export default function HistoryDrawer({
  activeSessionId,
  drawerRef,
  isLoading = false,
  isOpen,
  onClose,
  onNewSession,
  onSelectSession,
  sessions,
  toggleButtonRef,
  workspaceRef,
}: HistoryDrawerProps) {
  useEffect(() => {
    if (!isOpen) {
      return;
    }

    const handlePointerDown = (event: MouseEvent) => {
      const target = event.target as Node;
      const drawer = drawerRef.current;
      const toggleAnchor = toggleButtonRef.current?.closest(".history-toggle-anchor");
      if (
        drawer?.contains(target) ||
        toggleAnchor?.contains(target) ||
        workspaceRef.current?.querySelector(".history-toggle-anchor")?.contains(target)
      ) {
        return;
      }

      onClose();
    };

    window.addEventListener("mousedown", handlePointerDown);
    return () => window.removeEventListener("mousedown", handlePointerDown);
  }, [drawerRef, isOpen, onClose, toggleButtonRef, workspaceRef]);

  return (
    <>
      <div
        className={`history-drawer-backdrop${isOpen ? " is-visible" : ""}`}
        aria-hidden={!isOpen}
        onClick={onClose}
      />

      <aside
        ref={drawerRef}
        id="history-drawer"
        className={`history-drawer${isOpen ? " is-open" : ""}`}
        role="dialog"
        aria-modal="true"
        aria-labelledby="history-drawer-title"
        aria-hidden={!isOpen}
        tabIndex={isOpen ? 0 : -1}
      >
        <button
          type="button"
          className="history-drawer-close chrome-button"
          aria-label="Close history drawer"
          onClick={onClose}
        >
          Close
        </button>

        <SessionHistoryList
          activeSessionId={activeSessionId}
          isLoading={isLoading}
          onNewSession={onNewSession}
          onSelectSession={onSelectSession}
          sessions={sessions}
        />
      </aside>
    </>
  );
}
