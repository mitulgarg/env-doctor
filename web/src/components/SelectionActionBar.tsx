import { useState } from "react";
import type { MachineGroup } from "../types";
import GroupPicker from "./GroupPicker";

interface Props {
  count: number;
  groups: MachineGroup[];
  onAssign: (groupName: string | null) => Promise<void> | void;
  onClear: () => void;
  busy?: boolean;
  error?: string | null;
}

export default function SelectionActionBar({ count, groups, onAssign, onClear, busy, error }: Props) {
  const [pickerOpen, setPickerOpen] = useState(false);

  const containerStyle: React.CSSProperties = {
    position: "absolute",
    bottom: 24,
    left: "50%",
    transform: "translateX(-50%)",
    background: "rgba(22,27,34,0.95)",
    border: "1px solid rgba(88,166,255,0.4)",
    borderRadius: 10,
    padding: "10px 14px",
    display: "flex",
    alignItems: "center",
    gap: 12,
    boxShadow: "0 8px 24px rgba(0,0,0,0.5)",
    color: "#e6edf3",
    fontSize: 13,
    zIndex: 30,
    backdropFilter: "blur(12px)",
  };

  const pillStyle: React.CSSProperties = {
    padding: "4px 10px",
    background: "rgba(88,166,255,0.18)",
    border: "1px solid rgba(88,166,255,0.4)",
    borderRadius: 12,
    fontSize: 12,
    fontWeight: 600,
    color: "#58a6ff",
  };

  const buttonStyle = (variant: "primary" | "secondary" | "ghost"): React.CSSProperties => ({
    padding: "6px 12px",
    borderRadius: 6,
    fontSize: 12,
    fontWeight: 600,
    border: variant === "primary" ? "none" : "1px solid rgba(255,255,255,0.15)",
    background:
      variant === "primary" ? "#1f6feb" :
      variant === "ghost" ? "transparent" : "rgba(255,255,255,0.05)",
    color:
      variant === "primary" ? "#fff" :
      variant === "ghost" ? "rgba(255,255,255,0.6)" : "#e6edf3",
    cursor: busy ? "wait" : "pointer",
    opacity: busy ? 0.6 : 1,
  });

  return (
    <div style={containerStyle} onMouseDown={e => e.stopPropagation()}>
      <span style={pillStyle}>{count} selected</span>

      <div style={{ position: "relative" }}>
        <button
          type="button"
          disabled={busy}
          onClick={() => setPickerOpen(v => !v)}
          style={buttonStyle("primary")}
        >
          Assign group ▾
        </button>
        {pickerOpen && (
          <div style={{
            position: "absolute",
            bottom: "calc(100% + 6px)",
            left: 0,
            minWidth: 240,
          }}>
            <GroupPicker
              value={null}
              groups={groups}
              onChange={async (next) => {
                setPickerOpen(false);
                if (next) await onAssign(next);
              }}
              onClose={() => setPickerOpen(false)}
            />
          </div>
        )}
      </div>

      <button
        type="button"
        disabled={busy}
        onClick={() => onAssign(null)}
        style={buttonStyle("secondary")}
        title="Remove all selected machines from any group"
      >
        Ungroup
      </button>

      <button
        type="button"
        onClick={onClear}
        style={buttonStyle("ghost")}
        title="Clear selection (Esc)"
      >
        ✕
      </button>

      {error && (
        <span style={{ color: "#f85149", fontSize: 12, marginLeft: 4 }}>
          {error}
        </span>
      )}
    </div>
  );
}
