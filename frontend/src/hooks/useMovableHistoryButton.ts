"use client";

import { useCallback, useEffect, useRef, useState } from "react";
import {
  DEFAULT_HISTORY_BUTTON_POS,
  readStoredHistoryButtonPos,
  writeStoredHistoryButtonPos,
} from "../lib/workspace-layout";

const DRAG_THRESHOLD_PX = 6;

export function useMovableHistoryButton() {
  const workspaceRef = useRef<HTMLDivElement>(null);
  const [position, setPosition] = useState(DEFAULT_HISTORY_BUTTON_POS);
  const [hasHydratedPosition, setHasHydratedPosition] = useState(false);
  const [isDragging, setIsDragging] = useState(false);
  const dragOffsetRef = useRef({ x: 0, y: 0 });
  const didDragRef = useRef(false);
  const pointerStartRef = useRef({ x: 0, y: 0 });

  useEffect(() => {
    setPosition(readStoredHistoryButtonPos());
    setHasHydratedPosition(true);
  }, []);

  useEffect(() => {
    if (!hasHydratedPosition || isDragging) {
      return;
    }

    writeStoredHistoryButtonPos(position);
  }, [hasHydratedPosition, isDragging, position]);

  const clampPosition = useCallback((nextX: number, nextY: number) => {
    const workspace = workspaceRef.current;
    if (!workspace) {
      return { x: nextX, y: nextY };
    }

    const bounds = workspace.getBoundingClientRect();
    const buttonSize = 72;
    const maxX = Math.max(0, bounds.width - buttonSize);
    const maxY = Math.max(0, bounds.height - buttonSize);

    return {
      x: Math.min(maxX, Math.max(0, nextX)),
      y: Math.min(maxY, Math.max(0, nextY)),
    };
  }, []);

  const beginDrag = useCallback(
    (clientX: number, clientY: number) => {
      const workspace = workspaceRef.current;
      if (!workspace) {
        return;
      }

      const bounds = workspace.getBoundingClientRect();
      dragOffsetRef.current = {
        x: clientX - bounds.left - position.x,
        y: clientY - bounds.top - position.y,
      };
      pointerStartRef.current = { x: clientX, y: clientY };
      didDragRef.current = false;
      setIsDragging(true);
    },
    [position.x, position.y],
  );

  useEffect(() => {
    if (!isDragging) {
      return;
    }

    const handlePointerMove = (event: PointerEvent) => {
      const deltaX = Math.abs(event.clientX - pointerStartRef.current.x);
      const deltaY = Math.abs(event.clientY - pointerStartRef.current.y);
      if (deltaX > DRAG_THRESHOLD_PX || deltaY > DRAG_THRESHOLD_PX) {
        didDragRef.current = true;
      }

      const workspace = workspaceRef.current;
      if (!workspace) {
        return;
      }

      const bounds = workspace.getBoundingClientRect();
      const next = clampPosition(
        event.clientX - bounds.left - dragOffsetRef.current.x,
        event.clientY - bounds.top - dragOffsetRef.current.y,
      );
      setPosition(next);
    };

    const handlePointerUp = () => {
      setIsDragging(false);
    };

    document.body.classList.add("is-dragging-history-button");
    window.addEventListener("pointermove", handlePointerMove);
    window.addEventListener("pointerup", handlePointerUp);

    return () => {
      document.body.classList.remove("is-dragging-history-button");
      window.removeEventListener("pointermove", handlePointerMove);
      window.removeEventListener("pointerup", handlePointerUp);
    };
  }, [clampPosition, isDragging]);

  const consumeDragClick = useCallback(() => {
    if (!didDragRef.current) {
      return false;
    }

    didDragRef.current = false;
    return true;
  }, []);

  return {
    workspaceRef,
    position,
    isDragging,
    beginDrag,
    consumeDragClick,
  };
}
