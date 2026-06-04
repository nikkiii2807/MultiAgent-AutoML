"use client";

import { useCallback, useEffect, useRef, useState } from "react";
import {
  LAYOUT_MAIN_SPLIT,
  MAIN_SPLIT_MAX,
  MAIN_SPLIT_MIN,
  type LayoutMode,
  clamp,
  getMainSplitForMode,
  isNotebookHidden,
  readStoredLayoutMode,
  readStoredNumber,
  writeStoredLayoutMode,
  writeStoredNumber,
  STORAGE_MAIN_SPLIT,
} from "../lib/workspace-layout";

export function useWorkspaceLayout() {
  const [layoutMode, setLayoutModeState] = useState<LayoutMode>("balanced");
  const [mainSplit, setMainSplit] = useState(LAYOUT_MAIN_SPLIT.balanced);
  const [hasHydratedLayout, setHasHydratedLayout] = useState(false);
  const [isResizingPanels, setIsResizingPanels] = useState(false);

  const mainContainerRef = useRef<HTMLElement>(null);

  useEffect(() => {
    const mode = readStoredLayoutMode("balanced");
    setLayoutModeState(mode);
    setMainSplit(
      readStoredNumber(STORAGE_MAIN_SPLIT, getMainSplitForMode(mode)),
    );
    setHasHydratedLayout(true);
  }, []);

  useEffect(() => {
    if (!hasHydratedLayout) {
      return;
    }

    writeStoredLayoutMode(layoutMode);
  }, [hasHydratedLayout, layoutMode]);

  useEffect(() => {
    if (!hasHydratedLayout || isResizingPanels) {
      return;
    }

    writeStoredNumber(STORAGE_MAIN_SPLIT, mainSplit);
  }, [hasHydratedLayout, isResizingPanels, mainSplit]);

  const setLayoutMode = useCallback((mode: LayoutMode) => {
    setLayoutModeState(mode);
    setMainSplit(getMainSplitForMode(mode));
  }, []);

  const toggleFocusMode = useCallback(() => {
    setLayoutModeState((current) => {
      const next = current === "focus" ? "balanced" : "focus";
      setMainSplit(getMainSplitForMode(next));
      return next;
    });
  }, []);

  const setMainSplitFromPointer = useCallback((clientX: number) => {
    const container = mainContainerRef.current;
    if (!container) {
      return;
    }

    const rect = container.getBoundingClientRect();
    const ratio = clamp((clientX - rect.left) / rect.width, MAIN_SPLIT_MIN, MAIN_SPLIT_MAX);
    setMainSplit(ratio);
  }, []);

  const beginPanelResize = useCallback(
    (clientX?: number) => {
      setIsResizingPanels(true);
      if (clientX !== undefined) {
        setMainSplitFromPointer(clientX);
      }
    },
    [setMainSplitFromPointer],
  );

  useEffect(() => {
    if (!isResizingPanels) {
      return;
    }

    const handlePointerMove = (event: PointerEvent) => {
      setMainSplitFromPointer(event.clientX);
    };

    const handlePointerUp = () => {
      setIsResizingPanels(false);
    };

    document.body.classList.add("is-resizing-panels");
    window.addEventListener("pointermove", handlePointerMove);
    window.addEventListener("pointerup", handlePointerUp);

    return () => {
      document.body.classList.remove("is-resizing-panels");
      window.removeEventListener("pointermove", handlePointerMove);
      window.removeEventListener("pointerup", handlePointerUp);
    };
  }, [isResizingPanels, setMainSplitFromPointer]);

  return {
    layoutMode,
    mainSplit,
    isNotebookCollapsed: isNotebookHidden(layoutMode),
    isResizingPanels,
    mainContainerRef,
    setLayoutMode,
    setMainSplit,
    toggleFocusMode,
    beginPanelResize,
  };
}
