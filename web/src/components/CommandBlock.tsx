import { useState, useEffect, useRef } from "react";
import { queueCommand, getCommands } from "../api";
import type { CommandRecord } from "../types";

interface Props {
  machineId: string;
  command: string;
  label?: string;
}

type RunState = "idle" | "queued" | "running" | "done" | "failed";

const STATUS_COLOR: Record<string, string> = {
  pending: "#d29922",
  running: "#58a6ff",
  done: "#238636",
  failed: "#da3633",
};

export default function CommandBlock({ machineId, command, label }: Props) {
  const [runState, setRunState] = useState<RunState>("idle");
  const [cmdId, setCmdId] = useState<number | null>(null);
  const [output, setOutput] = useState<string | null>(null);
  const [showOutput, setShowOutput] = useState(false);
  const pollRef = useRef<ReturnType<typeof setInterval> | null>(null);

  // Poll for command completion
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
        } else if (cmd.status === "running") {
          setRunState("running");
        }
      } catch {
        // keep polling
      }
    }, 3000);
    return () => { if (pollRef.current) clearInterval(pollRef.current); };
  }, [cmdId, machineId, runState]);

  const handleRun = async () => {
    if (runState !== "idle") return;
    setRunState("queued");
    try {
      const result: CommandRecord = await queueCommand(machineId, command);
      setCmdId(result.id);
    } catch {
      setRunState("failed");
      setOutput("Failed to queue command on server.");
    }
  };

  const stateLabel: Record<RunState, string> = {
    idle: "▶ Run",
    queued: "Queued…",
    running: "Running…",
    done: "✓ Done",
    failed: "✗ Failed",
  };

  const btnColor: Record<RunState, string> = {
    idle: "#1f6feb",
    queued: "#d29922",
    running: "#58a6ff",
    done: "#238636",
    failed: "#da3633",
  };

  return (
    <div style={{ marginBottom: 10 }}>
      {label && (
        <div style={{ fontSize: 11, color: "rgba(255,255,255,0.35)", marginBottom: 4 }}>
          {label}
        </div>
      )}
      <div style={{
        display: "flex",
        alignItems: "center",
        gap: 8,
        background: "#161b22",
        border: "1px solid rgba(255,255,255,0.1)",
        borderRadius: 8,
        padding: "8px 12px",
      }}>
        <code style={{
          flex: 1,
          fontFamily: "monospace",
          fontSize: 13,
          color: "#e6edf3",
          wordBreak: "break-all",
        }}>
          {command}
        </code>
        <button
          onClick={handleRun}
          disabled={runState !== "idle"}
          style={{
            background: btnColor[runState],
            border: "none",
            color: "#fff",
            padding: "5px 12px",
            borderRadius: 6,
            cursor: runState === "idle" ? "pointer" : "default",
            fontSize: 12,
            fontWeight: 600,
            whiteSpace: "nowrap",
            opacity: runState === "idle" ? 1 : 0.85,
            flexShrink: 0,
          }}
        >
          {stateLabel[runState]}
        </button>
      </div>

      {/* Status line */}
      {runState !== "idle" && (
        <div style={{
          fontSize: 11,
          color: STATUS_COLOR[runState === "queued" ? "pending" : runState === "running" ? "running" : runState] ?? "#868e96",
          marginTop: 4,
          paddingLeft: 4,
          display: "flex",
          alignItems: "center",
          gap: 6,
        }}>
          {(runState === "queued" || runState === "running") && (
            <span style={{ animation: "spin 1s linear infinite", display: "inline-block" }}>⟳</span>
          )}
          {runState === "queued" && "Waiting for machine to check in…"}
          {runState === "running" && "Machine is executing…"}
          {runState === "done" && "Completed successfully"}
          {runState === "failed" && "Command failed on machine"}
          {output && (runState === "done" || runState === "failed") && (
            <button
              onClick={() => setShowOutput(!showOutput)}
              style={{ background: "none", border: "none", color: "#58a6ff", cursor: "pointer", fontSize: 11, padding: 0 }}
            >
              {showOutput ? "Hide output" : "Show output"}
            </button>
          )}
        </div>
      )}

      {/* Output terminal */}
      {showOutput && output && (
        <pre style={{
          background: "#010409",
          border: "1px solid rgba(255,255,255,0.08)",
          borderRadius: 6,
          padding: 12,
          fontSize: 11,
          lineHeight: 1.6,
          color: "rgba(255,255,255,0.7)",
          overflowX: "auto",
          maxHeight: 240,
          marginTop: 6,
          whiteSpace: "pre-wrap",
          wordBreak: "break-word",
        }}>
          {output}
        </pre>
      )}

      <style>{`@keyframes spin { from { transform: rotate(0deg) } to { transform: rotate(360deg) } }`}</style>
    </div>
  );
}
