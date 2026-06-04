"use client";

interface PanelResizeHandleProps {
  label?: string;
  onPointerDown: (event: React.PointerEvent<HTMLButtonElement>) => void;
}

export default function PanelResizeHandle({
  label = "Resize AutoML Copilot and Notebook panels",
  onPointerDown,
}: PanelResizeHandleProps) {
  return (
    <button
      type="button"
      className="panel-resize-handle"
      aria-label={label}
      onPointerDown={(event) => {
        event.preventDefault();
        event.currentTarget.setPointerCapture(event.pointerId);
        onPointerDown(event);
      }}
    />
  );
}
