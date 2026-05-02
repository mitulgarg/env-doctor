# dashboard & report

> **Requires:** `pip install env-doctor[dashboard]`

The `dashboard` and `report` commands are part of the optional fleet monitoring add-on. The core CLI works without them.

---

## `env-doctor dashboard`

Starts the fleet monitoring web UI on the local machine.

```bash
env-doctor dashboard [--host HOST] [--port PORT]
```

| Flag | Default | Description |
|------|---------|-------------|
| `--host` | `0.0.0.0` | Network interface to bind to |
| `--port` | `8765` | Port to listen on |

### Example

```bash
env-doctor dashboard
# → Starting env-doctor dashboard at http://0.0.0.0:8765
```

Open `http://localhost:8765` in a browser to see the fleet overview.

The dashboard stores all machine data in `~/.env-doctor/dashboard.db` (SQLite). No external database required.

---

## `env-doctor report`

Manage periodic reporting from a GPU machine to a running dashboard instance. All `report` commands run **on the GPU machine**, not on the dashboard host.

### `report install`

Set up a scheduled task to send diagnostic reports automatically. Uses **cron** on Linux/macOS and **Windows Task Scheduler** on Windows.

```bash
env-doctor report install --url URL [--interval INTERVAL] [--heartbeat HEARTBEAT]
```

| Flag | Default | Description |
|------|---------|-------------|
| `--url` | *(required)* | Dashboard URL (e.g., `http://10.0.1.50:8765`) |
| `--interval` | `2m` | How often to run the check (cron interval or Task Scheduler interval) |
| `--heartbeat` | `30m` | Re-send even if nothing changed, to confirm machine is alive |

**What it does:**

1. Validates the dashboard is reachable
2. Creates a scheduled task:
      - **Linux/macOS:** cron job tagged `# env-doctor-report`
      - **Windows:** Task Scheduler entry named `env-doctor-report`
3. Saves config to `~/.env-doctor/report-config.json`
4. Sends the first report immediately

```bash
env-doctor report install --url http://10.0.1.50:8765 --interval 2m
# ✅ Reporting to http://10.0.1.50:8765 every 2m (heartbeat: 30m) via cron
# Sending first report...
```

**Smart change detection** — the scheduled task fires every 2 minutes, but does **not** POST every 2 minutes. Each run hashes the check result and compares it to the last sent hash in `~/.env-doctor/report-state.json`. If nothing changed and the heartbeat timer hasn't expired, it skips entirely (no network call). Only actual state changes (new driver, new library, new issue) trigger an immediate report. On a stable machine, this means ~1 POST every 30 minutes instead of 720 per day.

### `report uninstall`

Remove the scheduled task (cron job or Task Scheduler entry) and clean up local state.

```bash
env-doctor report uninstall
# Reporting stopped. Local state cleaned up.
```

### `report status`

Show the current reporting configuration, scheduler status, and last report time. This is purely local — reads `~/.env-doctor/` files on this machine, no network call.

```bash
env-doctor report status
# Reporting to http://10.0.1.50:8765 every 2m (heartbeat: 30m)
# Scheduler: cron (active)
# Last report: 3m ago
# Last report hash: a1b2c3d4
```

---

## One-time reporting (no cron)

Use `--report-to` on the `check` command to send a single report without setting up periodic reporting:

```bash
env-doctor check --report-to http://10.0.1.50:8765
```

Add `--force` to bypass change detection and always send:

```bash
env-doctor check --report-to http://10.0.1.50:8765 --force
```

---

## Machine Identity

Each machine gets a stable UUID stored in `~/.env-doctor/machine-id`. This is generated on first use and persists across reboots.

If you're on a shared filesystem (NFS home dirs), override it with an environment variable:

```bash
export ENV_DOCTOR_MACHINE_ID=gpu-node-01
env-doctor check --report-to http://dashboard:8765
```

---

## API Reference

The dashboard exposes a REST API that machines post to:

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/api/report` | POST | Receive a check report; returns `pending_commands[]` for the machine |
| `/api/machines` | GET | List all machines with latest status |
| `/api/machines/{id}` | GET | Full machine detail and latest report |
| `/api/machines/{id}/history` | GET | Snapshot timeline |
| `/api/machines/{id}/commands` | POST | Queue a remediation command (must start with `env-doctor` or `doctor`) |
| `/api/machines/{id}/commands` | GET | List recent commands with status and output |
| `/api/machines/{id}/commands/{cmd_id}/result` | POST | Machine posts execution result `{output, exit_code}` |

See the [Fleet Monitoring Guide](../guides/fleet-monitoring.md) for full setup instructions.
