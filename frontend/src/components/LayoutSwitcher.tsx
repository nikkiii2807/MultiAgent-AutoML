"use client";

import { useEffect, useId, useRef, useState } from "react";
import {
  LAYOUT_MODE_LABELS,
  type LayoutMode,
} from "../lib/workspace-layout";

const LAYOUT_OPTIONS: LayoutMode[] = ["balanced", "focus", "wide-chat", "wide-notebook"];

interface LayoutSwitcherProps {
  layoutMode: LayoutMode;
  onChange: (mode: LayoutMode) => void;
}

export default function LayoutSwitcher({ layoutMode, onChange }: LayoutSwitcherProps) {
  const [isOpen, setIsOpen] = useState(false);
  const menuId = useId();
  const rootRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    if (!isOpen) {
      return;
    }

    const handlePointerDown = (event: MouseEvent) => {
      if (!rootRef.current?.contains(event.target as Node)) {
        setIsOpen(false);
      }
    };

    const handleKeyDown = (event: KeyboardEvent) => {
      if (event.key === "Escape") {
        setIsOpen(false);
      }
    };

    window.addEventListener("mousedown", handlePointerDown);
    window.addEventListener("keydown", handleKeyDown);
    return () => {
      window.removeEventListener("mousedown", handlePointerDown);
      window.removeEventListener("keydown", handleKeyDown);
    };
  }, [isOpen]);

  return (
    <div className="layout-switcher" ref={rootRef}>
      <button
        type="button"
        className="chrome-button layout-switcher-trigger"
        aria-expanded={isOpen}
        aria-haspopup="menu"
        aria-controls={menuId}
        onClick={() => setIsOpen((current) => !current)}
      >
        <span aria-hidden="true">⚙</span>
        <span>Layout</span>
      </button>

      {isOpen && (
        <div
          id={menuId}
          className="layout-switcher-menu"
          role="menu"
          aria-label="Workspace layout"
        >
          {LAYOUT_OPTIONS.map((mode) => (
            <button
              key={mode}
              type="button"
              role="menuitemradio"
              aria-checked={layoutMode === mode}
              className={`layout-switcher-option${layoutMode === mode ? " is-active" : ""}`}
              onClick={() => {
                onChange(mode);
                setIsOpen(false);
              }}
            >
              {LAYOUT_MODE_LABELS[mode]}
            </button>
          ))}
        </div>
      )}
    </div>
  );
}
