export interface MachineListItem {
  id: string;
  hostname: string;
  platform: string | null;
  python_version: string | null;
  latest_status: string | null;
  first_seen: string | null;
  last_seen: string | null;
  gpu_name: string | null;
  driver_version: string | null;
  cuda_version: string | null;
  torch_version: string | null;
}

export interface MachineDetail extends MachineListItem {
  latest_report: Report | null;
}

export interface Report {
  machine: {
    machine_id: string;
    hostname: string;
    platform: string;
    platform_release: string;
    python_version: string;
    reported_at: string;
  };
  status: string;
  timestamp: string;
  summary: {
    driver: string;
    cuda: string;
    cudnn: string;
    issues_count: number;
  };
  checks: {
    wsl2: CheckResult | null;
    driver: CheckResult;
    cuda: CheckResult;
    cudnn: CheckResult | null;
    libraries: Record<string, CheckResult>;
    python_compat: CheckResult;
    compute_compatibility: ComputeCompat | null;
  };
}

export interface CheckResult {
  component: string;
  status: string;
  detected: boolean;
  version: string | null;
  path: string | null;
  metadata: Record<string, unknown>;
  issues: string[];
  recommendations: string[];
}

export interface ComputeCompat {
  gpu_name: string;
  compute_capability: string | null;
  arch_list: string[];
  cuda_available: boolean | null;
  sm: string;
  arch_name: string;
  status: string;
}

export interface CommandRecord {
  id: number;
  machine_id: string;
  command: string;
  status: string; // pending | running | done | failed
  output: string | null;
  exit_code: number | null;
  created_at: string | null;
  executed_at: string | null;
}

export interface SnapshotSummary {
  id: number;
  machine_id: string;
  status: string;
  timestamp: string;
  gpu_name: string | null;
  driver_version: string | null;
  cuda_version: string | null;
  torch_version: string | null;
  is_heartbeat: boolean;
}
