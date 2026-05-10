import { useEffect, useMemo, useRef, useState } from "react";
import type { MachineGroup } from "../types";

interface Props {
  value: string | null;
  onChange: (next: string | null) => void;
  groups: MachineGroup[];
  /** Focus the input on mount. Default true. */
  autoFocus?: boolean;
  /** Called when the user dismisses without selecting (Esc, click outside). */
  onClose?: () => void;
  placeholder?: string;
}

const UNGROUPED_LABEL = "ungrouped";

const DROPDOWN_MAX_HEIGHT = 240;

const inputStyle: React.CSSProperties = {
  width: "100%",
  padding: "8px 10px",
  background: "#0d1117",
  border: "1px solid rgba(88,166,255,0.5)",
  borderRadius: 6,
  color: "#e6edf3",
  fontSize: 13,
  outline: "none",
  boxSizing: "border-box",
};

function dropdownStyle(openUp: boolean): React.CSSProperties {
  return {
    position: "absolute",
    [openUp ? "bottom" : "top"]: "calc(100% + 4px)",
    left: 0,
    right: 0,
    minWidth: 200,
    background: "#161b22",
    border: "1px solid rgba(255,255,255,0.12)",
    borderRadius: 6,
    maxHeight: DROPDOWN_MAX_HEIGHT,
    overflowY: "auto",
    zIndex: 50,
    boxShadow: openUp ? "0 -4px 16px rgba(0,0,0,0.4)" : "0 4px 16px rgba(0,0,0,0.4)",
  };
}

const itemStyle = (active: boolean): React.CSSProperties => ({
  padding: "8px 12px",
  fontSize: 13,
  color: "#e6edf3",
  cursor: "pointer",
  background: active ? "rgba(88,166,255,0.15)" : "transparent",
  display: "flex",
  alignItems: "center",
  justifyContent: "space-between",
  gap: 8,
});

export default function GroupPicker({
  value,
  onChange,
  groups,
  autoFocus = true,
  onClose,
  placeholder = "Group name…",
}: Props) {
  const [text, setText] = useState(value ?? "");
  const [highlight, setHighlight] = useState(0);
  const [openUp, setOpenUp] = useState(false);
  const rootRef = useRef<HTMLDivElement>(null);

  // Auto-detect whether the dropdown should open upward — if there's not
  // enough vertical room below the input (e.g. picker rendered near the
  // bottom of the viewport, like inside SelectionActionBar), flip up.
  useEffect(() => {
    if (!rootRef.current) return;
    const rect = rootRef.current.getBoundingClientRect();
    const spaceBelow = window.innerHeight - rect.bottom;
    const spaceAbove = rect.top;
    if (spaceBelow < DROPDOWN_MAX_HEIGHT + 20 && spaceAbove > spaceBelow) {
      setOpenUp(true);
    }
  }, []);

  // Filter out the synthetic "ungrouped" entry — it's not a real group.
  const realGroups = useMemo(
    () => groups.filter(g => g.name.toLowerCase() !== UNGROUPED_LABEL),
    [groups]
  );

  const trimmed = text.trim();
  const lower = trimmed.toLowerCase();

  const filtered = useMemo(() => {
    if (!lower) return realGroups;
    return realGroups.filter(g => g.name.toLowerCase().includes(lower));
  }, [realGroups, lower]);

  const exactMatch = filtered.some(g => g.name.toLowerCase() === lower);
  const showCreate = trimmed.length > 0 && !exactMatch;
  const showUngroup = value !== null;

  const optionCount = filtered.length + (showCreate ? 1 : 0) + (showUngroup ? 1 : 0);

  // Reset highlight whenever the filter list changes shape.
  useEffect(() => { setHighlight(0); }, [text, realGroups.length]);

  // Click-outside dismissal.
  useEffect(() => {
    if (!onClose) return;
    const handler = (e: MouseEvent) => {
      if (rootRef.current && !rootRef.current.contains(e.target as Node)) {
        onClose();
      }
    };
    document.addEventListener("mousedown", handler);
    return () => document.removeEventListener("mousedown", handler);
  }, [onClose]);

  const commitAt = (index: number) => {
    if (index < filtered.length) {
      onChange(filtered[index].name);
      return;
    }
    const offset = index - filtered.length;
    if (showCreate && offset === 0) {
      onChange(trimmed);
      return;
    }
    if (showUngroup) {
      onChange(null);
    }
  };

  const handleKey = (e: React.KeyboardEvent) => {
    if (e.key === "ArrowDown") {
      e.preventDefault();
      setHighlight(h => (optionCount === 0 ? 0 : Math.min(h + 1, optionCount - 1)));
    } else if (e.key === "ArrowUp") {
      e.preventDefault();
      setHighlight(h => Math.max(h - 1, 0));
    } else if (e.key === "Enter") {
      e.preventDefault();
      if (optionCount === 0) return;
      commitAt(highlight);
    } else if (e.key === "Escape") {
      e.preventDefault();
      onClose?.();
    }
  };

  return (
    <div ref={rootRef} style={{ position: "relative", minWidth: 200 }}>
      <input
        autoFocus={autoFocus}
        type="text"
        value={text}
        onChange={e => setText(e.target.value)}
        onKeyDown={handleKey}
        placeholder={placeholder}
        style={inputStyle}
        aria-label="Group name"
      />
      <div style={dropdownStyle(openUp)} role="listbox">
        {filtered.length === 0 && !showCreate && !showUngroup && (
          <div style={{ padding: "10px 12px", fontSize: 12, color: "rgba(255,255,255,0.4)" }}>
            No groups yet — type a name to create one.
          </div>
        )}
        {filtered.map((g, i) => (
          <div
            key={g.name}
            role="option"
            aria-selected={highlight === i}
            onMouseEnter={() => setHighlight(i)}
            onMouseDown={(e) => { e.preventDefault(); commitAt(i); }}
            style={itemStyle(highlight === i)}
          >
            <span>{g.name}</span>
            <span style={{ fontSize: 11, color: "rgba(255,255,255,0.4)" }}>
              {g.machine_count}
            </span>
          </div>
        ))}
        {showCreate && (
          <div
            role="option"
            aria-selected={highlight === filtered.length}
            onMouseEnter={() => setHighlight(filtered.length)}
            onMouseDown={(e) => { e.preventDefault(); commitAt(filtered.length); }}
            style={{
              ...itemStyle(highlight === filtered.length),
              borderTop: filtered.length > 0 ? "1px solid rgba(255,255,255,0.06)" : "none",
              color: "#58a6ff",
            }}
          >
            <span>+ Create &ldquo;{trimmed}&rdquo;</span>
          </div>
        )}
        {showUngroup && (
          <div
            role="option"
            aria-selected={highlight === filtered.length + (showCreate ? 1 : 0)}
            onMouseEnter={() => setHighlight(filtered.length + (showCreate ? 1 : 0))}
            onMouseDown={(e) => { e.preventDefault(); commitAt(filtered.length + (showCreate ? 1 : 0)); }}
            style={{
              ...itemStyle(highlight === filtered.length + (showCreate ? 1 : 0)),
              borderTop: "1px solid rgba(255,255,255,0.06)",
              color: "rgba(255,255,255,0.55)",
              fontSize: 12,
            }}
          >
            <span>Ungroup</span>
          </div>
        )}
      </div>
    </div>
  );
}
