"use client";

import { forwardRef, type CSSProperties, type PointerEvent } from "react";

interface HistoryToggleButtonProps {
  isDragging: boolean;
  isOpen: boolean;
  onClick: () => void;
  onDragHandlePointerDown: (event: PointerEvent<HTMLButtonElement>) => void;
  position: { x: number; y: number };
  sessionCount: number;
}

const HistoryToggleButton = forwardRef<HTMLButtonElement, HistoryToggleButtonProps>(
  function HistoryToggleButton(
    { isDragging, isOpen, onClick, onDragHandlePointerDown, position, sessionCount },
    ref,
  ) {
    const style: CSSProperties = {
      left: `${position.x}px`,
      top: `${position.y}px`,
    };

    return (
      <div
        className={`history-toggle-anchor${isDragging ? " is-dragging" : ""}${isOpen ? " is-open" : ""}`}
        style={style}
      >
        <button
          type="button"
          className="history-toggle-drag-handle"
          aria-label="Drag to reposition history button"
          onPointerDown={(event) => {
            event.preventDefault();
            event.currentTarget.setPointerCapture(event.pointerId);
            onDragHandlePointerDown(event);
          }}
        >
          ⠿
        </button>

        <button
          ref={ref}
          type="button"
          className="history-toggle-button"
          aria-expanded={isOpen}
          aria-controls="history-drawer"
          aria-haspopup="dialog"
          onClick={onClick}
        >
          <span className="history-toggle-icon" aria-hidden="true">
            🕘
          </span>
          <span className="history-toggle-label">History</span>
          {sessionCount > 0 && (
            <span className="history-toggle-badge" aria-label={`${sessionCount} saved sessions`}>
              {sessionCount}
            </span>
          )}
        </button>
      </div>
    );
  },
);

export default HistoryToggleButton;
