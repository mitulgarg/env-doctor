# Fleet Monitoring

> **Requires:** `pip install env-doctor[dashboard]`

The fleet dashboard is an optional observability layer on top of the core CLI. If you're working on a single machine, you don't need it — `env-doctor check` already gives you everything locally.

If you're managing multiple GPU machines (a training cluster, cloud instances, or lab workstations), the dashboard gives you a single view of their health without SSH-ing into each one.

---

## How It Works

```
  GPU machine 1          ┐
  GPU machine 2          ├──  env-doctor check --report-to URL  ──→  dashboard
  GPU machine N          ┘                                            (web UI + SQLite)
```

1. A dashboard server runs on one machine with a reachable IP
2. Each GPU machine reports its diagnostic output to the dashboard
3. The dashboard stores reports and serves a web UI

The core CLI is unchanged — `env-doctor check` still works standalone. `--report-to` is a side-effect that sends a copy of the result to the dashboard.

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

That's it. Machines appear in the dashboard immediately after their first report.

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

### Smart Reporting

The `--report-to` flag only sends a new report when something changes:

- Driver updated → sends immediately
- New library installed → sends immediately
- Nothing changed → sends a heartbeat every 30 minutes to confirm the machine is alive
- `--force` → always sends regardless

This keeps network traffic minimal when many machines check in frequently.

---

## Managing Reporting

```bash
# Check status on a machine
env-doctor report status
# → Reporting to http://10.0.1.50:8765 every 2m (heartbeat: 30m)
# → Last report: 3m ago

# Stop reporting
env-doctor report uninstall
# → Reporting stopped. Local state cleaned up.

# Override machine identity (useful for shared filesystems)
export ENV_DOCTOR_MACHINE_ID=gpu-node-01
env-doctor check --report-to http://10.0.1.50:8765
```

---

## Data Storage

All data is stored locally on the dashboard host:

| File | Purpose |
|------|---------|
| `~/.env-doctor/dashboard.db` | SQLite database (machines + snapshots) |
| `~/.env-doctor/machine-id` | Per-machine stable UUID |
| `~/.env-doctor/report-state.json` | Last report hash and heartbeat timestamp |
| `~/.env-doctor/report-config.json` | Dashboard URL and interval config |

No external database, no cloud dependencies.
