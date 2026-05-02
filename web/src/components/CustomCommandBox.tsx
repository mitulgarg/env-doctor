import { useEffect, useRef, useState } from "react";
import { getCommands, queueCommand } from "../api";
import type { CommandRecord } from "../types";

interface Props {
  machineId: string;
  onComplete?: () => void;
}

type RunState = "idle" | "queued" | "running" | "done" | "failed";

const STATUS_COLOR: Record<string, string> = {
  pending: "#d29922",
  running: "#58a6ff",
  done: "#238636",
  failed: "#da3633",
};

export default function CustomCommandBox({ machineId, onComplete }: Props) {
  const [text, setText] = useState("");
  const [runState, setRunState] = useState<RunState>("idle");
  const [cmdId, setCmdId] = useState<number | null>(null);
  const [output, setOutput] = useState<string | null>(null);
  const [error, setError] = useState<string | null>(null);
  const pollRef = useRef<ReturnType<typeof setInterval> | null>(null);

  useEffect(() => {
    if (cmdId === null || runState === "done" || runState === "failed") {
      if (pollRef.current) clearInterval(pollRef.current);
      return;
    }
    pollRef.current = setInterval(async () => {
      try {
        const cmds: CommandRecord[] = await getCommands(machineId);
        const cmd = cmds.find(c => c.id === cmdId);
        if (!cmd) return;
        if (cmd.status === "done" || cmd.status === "failed") {
          setRunState(cmd.status as RunState);
          setOutput(cmd.output ?? null);
          if (pollRef.current) clearInterval(pollRef.current);
          setTimeout(() => onComplete?.(), 1500);
        } else if (cmd.status === "running") {
          setRunState("running");
        }
      } catch {
        /* keep polling */
      }
    }, 3000);
    return () => { if (pollRef.current) clearInterval(pollRef.current); };
  }, [cmdId, machineId, runState]);

  const submit = async (e: React.FormEvent) => {
    e.preventDefault();
    const command = text.trim();
    if (!command || runState === "queued" || runState === "running") return;

    setError(null);
    setOutput(null);
    setCmdId(null);
    setRunState("queued");
    try {
      const result = await queueCommand(machineId, command);
      setCmdId(result.id);
    } catch (err: unknown) {
      const msg = err instanceof Error ? err.message : "Failed to queue";
      setError(msg);
      setRunState("idle");
    }
  };

  const reset = () => {
    setRunState("idle");
    setCmdId(null);
    setOutput(null);
    setError(null);
    setText("");
  };

  const inFlight = runState === "queued" || runState === "running";

  return (
    <div style={{
      background: "rgba(255,255,255,0.04)",
      border: "1px solid rgba(255,255,255,0.08)",
      borderRadius: 8,
      padding: 16,
      marginBottom: 16,
    }}>
      <div style={{ fontSize: 13, fontWeight: 600, color: "rgba(255,255,255,0.6)", marginBottom: 8 }}>
        Run a command on this machine
      </div>
      <div style={{ fontSize: 11, color: "rgba(255,255,255,0.35)", marginBottom: 10 }}>
        Only commands starting with <code>env-doctor</code> or <code>doctor</code> are accepted.
      </div>
      <form onSubmit={submit} style={{ display: "flex", gap: 8 }}>
        <input
          type="text"
          value={text}
          onChange={e => setText(e.target.value)}
          placeholder="env-doctor cuda-info"
          spellCheck={false}
          disabled={inFlight}
          style={{
            flex: 1,
            padding: "8px 12px",
            background: "#0d1117",
            border: "1px solid rgba(255,255,255,0.12)",
            borderRadius: 6,
            color: "#e6edf3",
            fontFamily: "monospace",
            fontSize: 13,
          }}
        />
        <button
          type="submit"
          disabled={inFlight || !text.trim()}
          style={{
            padding: "8px 16px",
            background: inFlight ? "#1f3a6e" : (text.trim() ? "#1f6feb" : "#21262d"),
            border: "none",
            color: "#fff",
            fontWeight: 600,
            fontSize: 12,
            borderRadius: 6,
            cursor: inFlight || !text.trim() ? "default" : "pointer",
          }}
        >
          {inFlight ? "Running…" : "Queue"}
        </button>
        {(runState === "done" || runState === "failed") && (
          <button
            type="button"
            onClick={reset}
            style={{
              padding: "8px 12px",
              background: "transparent",
              border: "1px solid rgba(255,255,255,0.12)",
              borderRadius: 6,
              color: "rgba(255,255,255,0.55)",
              fontSize: 12,
              cursor: "pointer",
            }}
          >
            New
          </button>
        )}
      </form>

      {error && (
        <div style={{ marginTop: 8, fontSize: 12, color: "#f47174" }}>
          {error}
        </div>
      )}

      {runState !== "idle" && !error && (
        <div style={{
          marginTop: 8,
          fontSize: 11,
          color: STATUS_COLOR[runState === "queued" ? "pending" : runState] ?? "#868e96",
        }}>
          {runState === "queued" && "⟳ Waiting for machine to check in…"}
          {runState === "running" && "⟳ Machine is executing…"}
          {runState === "done" && "✓ Completed successfully"}
          {runState === "failed" && "✗ Command failed on machine"}
        </div>
      )}

      {output && (runState === "done" || runState === "failed") && (
        <pre style={{
          marginTop: 10,
          background: "#010409",
          border: "1px solid rgba(255,255,255,0.08)",
          borderRadius: 6,
          padding: 12,
          fontSize: 11,
          lineHeight: 1.6,
          color: "rgba(255,255,255,0.7)",
          maxHeight: 280,
          overflow: "auto",
          whiteSpace: "pre-wrap",
          wordBreak: "break-word",
        }}>
          {output}
        </pre>
      )}
    </div>
  );
}
