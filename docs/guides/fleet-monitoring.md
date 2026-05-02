# Fleet Monitoring

> **Dashboard host:** `pip install env-doctor[dashboard]`
> **GPU machines:** `pip install env-doctor` (core CLI is enough)

The fleet dashboard is an optional observability layer on top of the core CLI. If you're working on a single machine, you don't need it — `env-doctor check` already gives you everything locally.

If you're managing multiple GPU machines (a training cluster, cloud instances, or lab workstations), the dashboard gives you a single view of their health without SSH-ing into each one.

---

## How It Works

There are two roles: the **dashboard host** and the **GPU machines**. They communicate over a simple REST API.

```
  Dashboard Host                              GPU Machines
  ┌──────────────────────────┐
  │  env-doctor dashboard    │◄─── POST /api/report ───  GPU machine 1  (env-doctor check --report-to)
  │  (web UI + SQLite)       │◄─── POST /api/report ───  GPU machine 2  (env-doctor check --report-to)
  │  pip install             │◄─── POST /api/report ───  GPU machine N  (env-doctor check --report-to)
  │    env-doctor[dashboard] │
  └──────────────────────────┘                            pip install env-doctor (core only)
```

1. The **dashboard host** runs `env-doctor dashboard` — this starts a web server that receives reports and serves the UI. It needs `pip install env-doctor[dashboard]` for the FastAPI/SQLite dependencies. No GPU needed — a cheap CPU instance is enough.
2. Each **GPU machine** runs `env-doctor check --report-to <url>` — this runs the diagnostic check locally, then POSTs the JSON result to the dashboard's `/api/report` endpoint. GPU machines only need `pip install env-doctor` (the core CLI).
3. The dashboard stores every report in SQLite and displays them in the web UI.

The core CLI is unchanged — `env-doctor check` still works standalone. `--report-to` adds a side-effect that sends a copy of the result to the dashboard.

---

## Quick Start

### Step 1 — Start the dashboard

Pick any machine with a reachable IP (a cloud VM, jump box, or one of your GPU machines):

```bash
pip install env-doctor[dashboard]
env-doctor dashboard
# → Serving at http://0.0.0.0:8765
```

Open `http://<host-ip>:8765` in a browser. The fleet table is empty until machines start reporting.

### Step 2 — Report from each GPU machine

**One-time report:**
```bash
pip install env-doctor
env-doctor check --report-to http://<dashboard-host>:8765
```

**Automatic reporting every 2 minutes:**
```bash
pip install env-doctor
env-doctor report install --url http://<dashboard-host>:8765 --interval 2m
```

`report install` creates a scheduled task **on the GPU machine** — a cron job on Linux/macOS or a Windows Task Scheduler entry on Windows. That task runs `env-doctor check --report-to <url>` on the configured interval.

Machines appear in the dashboard immediately after their first report.

---

## Cloud Setup

### Same VPC / Private Network

If all machines are on the same private network, use the internal IP directly:

```bash
# Find the dashboard host's private IP
hostname -I | awk '{print $1}'

# On each GPU instance
env-doctor report install --url http://10.0.1.50:8765
```

Make sure port `8765` is open in your internal firewall rules.

### AWS

```bash
# On the dashboard EC2 instance
pip install env-doctor[dashboard]
env-doctor dashboard

# In the AWS console:
# Security Group → Inbound Rules → Add rule: TCP 8765, Source: your VPC CIDR (e.g. 10.0.0.0/16)

# On each GPU instance (same VPC)
env-doctor report install --url http://<dashboard-private-ip>:8765
```

### GCP

```bash
# Create a firewall rule
gcloud compute firewall-rules create allow-env-doctor \
  --allow tcp:8765 \
  --source-ranges 10.0.0.0/8 \
  --target-tags env-doctor-dashboard

# Tag your dashboard instance
gcloud compute instances add-tags <dashboard-vm> --tags env-doctor-dashboard

# On each GPU instance
env-doctor report install --url http://<dashboard-internal-ip>:8765
```

### Azure

```bash
# Add inbound rule in Network Security Group: Port 8765, Source: VirtualNetwork

# On each GPU VM (same VNet)
env-doctor report install --url http://<dashboard-private-ip>:8765
```

---

## Machines on Different Networks (NAT)

If your GPU machines are behind different NATs, they can't reach a private IP. Options:

### Option 1 — Public IP (simplest)

Run the dashboard on a VM with a public IP and lock it down with a firewall rule to allow only your machines' IPs:

```bash
# Allow only your GPU machine IPs on port 8765
# (AWS SG, GCP firewall, Azure NSG, or iptables)

env-doctor report install --url http://<public-ip>:8765
```

### Option 2 — Tailscale (recommended for mixed networks)

[Tailscale](https://tailscale.com) gives every machine a routable IP with zero port forwarding. Free tier covers small fleets.

```bash
# On every machine (dashboard host + all GPU machines)
curl -fsSL https://tailscale.com/install.sh | sh
sudo tailscale up

# Find the dashboard host's Tailscale IP
tailscale ip -4

# On each GPU machine
env-doctor report install --url http://100.x.x.x:8765
```

### Option 3 — SSH tunnel (per machine, no persistent setup)

```bash
# On the dashboard host — forward remote machine's traffic through SSH
ssh -R 8765:localhost:8765 user@gpu-node-01

# On gpu-node-01 — reports to localhost, tunneled to your machine
env-doctor report install --url http://localhost:8765
```

---

## Fleet-Wide Setup via SSH / Ansible

Install and configure reporting on all machines at once:

```bash
# SSH one-liner per machine
ssh gpu-node-01 "pip install env-doctor && env-doctor report install --url http://10.0.1.50:8765"
ssh gpu-node-02 "pip install env-doctor && env-doctor report install --url http://10.0.1.50:8765"
```

```yaml
# Ansible playbook
- name: Install env-doctor and set up reporting
  hosts: gpu_machines
  tasks:
    - name: Install env-doctor
      pip:
        name: env-doctor

    - name: Set up periodic reporting
      command: env-doctor report install --url http://10.0.1.50:8765 --interval 2m
```

---

## Dashboard Features

### Fleet Overview

The main page shows all machines in a sortable, filterable table:

| Column | Description |
|--------|-------------|
| Hostname | Machine name |
| Status | `pass` / `warning` / `fail` |
| GPU | Primary GPU name |
| Driver | NVIDIA driver version |
| CUDA | System CUDA toolkit version |
| PyTorch | Installed PyTorch version |
| Last Seen | Time since last report |

Filter by status (All / Pass / Warning / Fail). Click any row to see the full diagnostic breakdown.

### Machine Detail

The detail page shows exactly what `env-doctor check` would print on that machine:

- GPU driver status and max CUDA version
- CUDA toolkit version and path
- cuDNN status
- Python library versions (PyTorch, TensorFlow, JAX)
- Python version compatibility
- Compute capability check
- All issues and recommendations — with **click-to-copy fix commands**
- Snapshot history timeline

### Smart Reporting (Change Detection)

The scheduled task fires every 2 minutes, but it does **not** POST every 2 minutes. Each run:

1. Runs `env-doctor check` locally on the GPU machine
2. Hashes the result (status + checks, excluding timestamps)
3. Compares to the last sent hash stored in `~/.env-doctor/report-state.json`

| Condition | Action |
|-----------|--------|
| Hash changed (driver updated, library installed, new issue) | POST full report immediately |
| Hash unchanged, 30 min since last POST | POST lightweight heartbeat (confirms machine is alive) |
| Hash unchanged, heartbeat not due | Skip — no network call, sub-second no-op |
| `--force` flag used | Always POST regardless |

On a stable machine with `--interval 2m`, this means **~1 POST every 30 minutes** instead of 720 per day. Only actual state changes trigger immediate reports.

---

## Managing Reporting

All `report` commands run **on the GPU machine**, not on the dashboard host. They manage the scheduled task and read local state — no network calls.

```bash
# Check reporting status on this machine
env-doctor report status
# → Reporting to http://10.0.1.50:8765 every 2m (heartbeat: 30m)
# → Scheduler: cron (active)           # or: Windows Task Scheduler (active)
# → Last report: 3m ago
# → Last report hash: a1b2c3d4

# Stop reporting and remove the scheduled task
env-doctor report uninstall
# → Reporting stopped. Local state cleaned up.

# Override machine identity (useful for shared filesystems / NFS home dirs)
export ENV_DOCTOR_MACHINE_ID=gpu-node-01
env-doctor check --report-to http://10.0.1.50:8765
```

### Platform Support

| Platform | Scheduler used by `report install` |
|----------|------------------------------------|
| Linux | cron (`crontab`) |
| macOS | cron (`crontab`) |
| Windows | Task Scheduler (`schtasks`) |

---

## Remote Remediation

The dashboard can queue `env-doctor` commands to run on remote machines — no SSH required.

### How it works

1. **Operator** opens a machine's detail page in the dashboard and types a command in the "Run a command" box (e.g. `env-doctor install torch --execute`).
2. The command is stored in the database with status `pending`.
3. On the **next check-in** (`POST /api/report`), the server returns the pending command in the response.
4. The **GPU machine CLI** executes the command, captures stdout/stderr, and POSTs the result back to `/api/machines/{id}/commands/{cmd_id}/result`.
5. The CLI then re-runs `env-doctor check --report-to <url>` to verify the fix and post a fresh diagnostic snapshot.
6. The **dashboard auto-refreshes** the machine detail page 1.5 s after the command completes, showing updated diagnostics without a manual page reload.

### Security

Only commands prefixed with `env-doctor` or `doctor` are accepted by the server. Arbitrary shell commands are rejected with HTTP 400.

### Command lifecycle

| Status | Meaning |
|--------|---------|
| `pending` | Queued, waiting for machine to check in |
| `running` | Machine picked it up, executing now |
| `done` | Completed with exit code 0 |
| `failed` | Completed with non-zero exit code |

### Exit codes for `env-doctor check --json`

When `env-doctor check --json` is queued as a remote command, exit code 1 does **not** mean the check failed — it means there are installed components with warnings. Uninstalled libraries alone do not produce a non-zero exit code.

| Exit code | Meaning |
|-----------|---------|
| `0` | All detected components compatible; uninstalled libraries ignored |
| `1` | Installed components have warnings or version conflicts |
| `2` | One or more components in error state |

---

## Data Storage

**On the dashboard host:**

| File | Purpose |
|------|---------|
| `~/.env-doctor/dashboard.db` | SQLite database (machines + snapshots) |

**On each GPU machine:**

| File | Purpose |
|------|---------|
| `~/.env-doctor/machine-id` | Stable UUID identifying this machine |
| `~/.env-doctor/report-state.json` | Last report hash and heartbeat timestamp (for change detection) |
| `~/.env-doctor/report-config.json` | Dashboard URL and interval config (set by `report install`) |

No external database, no cloud dependencies.
