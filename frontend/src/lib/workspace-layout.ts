export type LayoutMode = "balanced" | "focus" | "wide-chat" | "wide-notebook";

export const LAYOUT_MODE_LABELS: Record<LayoutMode, string> = {
  balanced: "Balanced",
  focus: "Focus Mode",
  "wide-chat": "Wide Chat",
  "wide-notebook": "Wide Notebook",
};

export const LAYOUT_MAIN_SPLIT: Record<Exclude<LayoutMode, "focus">, number> = {
  balanced: 0.52,
  "wide-chat": 0.68,
  "wide-notebook": 0.38,
};

export const STORAGE_LAYOUT_MODE = "automl-layout-mode";
export const STORAGE_MAIN_SPLIT = "automl-main-split";
export const STORAGE_HISTORY_BUTTON_POS = "automl-history-button-pos";

export const MAIN_SPLIT_MIN = 0.32;
export const MAIN_SPLIT_MAX = 0.72;

export const DEFAULT_HISTORY_BUTTON_POS = { x: 16, y: 16 };

export function clamp(value: number, min: number, max: number): number {
  return Math.min(max, Math.max(min, value));
}

export function readStoredNumber(key: string, fallback: number): number {
  if (typeof window === "undefined") {
    return fallback;
  }

  const raw = window.localStorage.getItem(key);
  if (!raw) {
    return fallback;
  }

  const parsed = Number(raw);
  return Number.isFinite(parsed) ? parsed : fallback;
}

export function writeStoredNumber(key: string, value: number): void {
  if (typeof window === "undefined") {
    return;
  }

  window.localStorage.setItem(key, String(value));
}

export function readStoredHistoryButtonPos(): { x: number; y: number } {
  if (typeof window === "undefined") {
    return DEFAULT_HISTORY_BUTTON_POS;
  }

  try {
    const raw = window.localStorage.getItem(STORAGE_HISTORY_BUTTON_POS);
    if (!raw) {
      return DEFAULT_HISTORY_BUTTON_POS;
    }

    const parsed = JSON.parse(raw) as { x?: number; y?: number };
    if (typeof parsed.x === "number" && typeof parsed.y === "number") {
      return { x: parsed.x, y: parsed.y };
    }
  } catch {
    return DEFAULT_HISTORY_BUTTON_POS;
  }

  return DEFAULT_HISTORY_BUTTON_POS;
}

export function writeStoredHistoryButtonPos(pos: { x: number; y: number }): void {
  if (typeof window === "undefined") {
    return;
  }

  window.localStorage.setItem(STORAGE_HISTORY_BUTTON_POS, JSON.stringify(pos));
}

export function isLayoutMode(value: string): value is LayoutMode {
  return value in LAYOUT_MODE_LABELS;
}

export function readStoredLayoutMode(fallback: LayoutMode = "balanced"): LayoutMode {
  if (typeof window === "undefined") {
    return fallback;
  }

  const raw = window.localStorage.getItem(STORAGE_LAYOUT_MODE);
  return raw && isLayoutMode(raw) ? raw : fallback;
}

export function writeStoredLayoutMode(mode: LayoutMode): void {
  if (typeof window === "undefined") {
    return;
  }

  window.localStorage.setItem(STORAGE_LAYOUT_MODE, mode);
}

export function getMainSplitForMode(mode: LayoutMode): number {
  if (mode === "focus") {
    return LAYOUT_MAIN_SPLIT.balanced;
  }

  return LAYOUT_MAIN_SPLIT[mode];
}

export function isNotebookHidden(mode: LayoutMode): boolean {
  return mode === "focus";
}
