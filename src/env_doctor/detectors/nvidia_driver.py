# ← Refactored from get_nvidia_driver_version()

import subprocess
import re
from env_doctor.core.detector import Detector, DetectionResult, Status
from env_doctor.core.registry import DetectorRegistry

@DetectorRegistry.register("nvidia_driver")
class NvidiaDriverDetector(Detector):
    """Detects NVIDIA GPU driver version and GPU VRAM information."""

    def detect(self) -> DetectionResult:
        # 1. Try NVML (preferred - supports GPU detection)
        detection_data = self._try_nvml()
        if detection_data:
            return self._success_result(detection_data, method="pynvml")

        # 2. Try nvidia-smi (fallback - gets driver and attempts GPU info)
        detection_data = self._try_nvidia_smi()
        if detection_data:
            return self._success_result(detection_data, method="nvidia-smi")

        # 3. Not found
        return DetectionResult(
            component="nvidia_driver",
            status=Status.NOT_FOUND,
            recommendations=[
                "Install NVIDIA drivers from https://www.nvidia.com/drivers",
                "For Linux: Check if nouveau drivers are blocking NVIDIA"
            ]
        )

    def _try_nvml(self):
        """
        Detect NVIDIA driver version and GPU information using pynvml.

        Returns:
            Dict with keys: driver, gpus (list of GPU info dicts)
            None if detection fails
        """
        try:
            import pynvml
            pynvml.nvmlInit()

            # Get driver version
            driver = pynvml.nvmlSystemGetDriverVersion().decode()

            # Get GPU information
            gpus = []
            try:
                device_count = pynvml.nvmlDeviceGetCount()

                for i in range(device_count):
                    handle = pynvml.nvmlDeviceGetHandleByIndex(i)

                    # Get GPU name
                    name = pynvml.nvmlDeviceGetName(handle).decode()

                    # Get memory info
                    mem_info = pynvml.nvmlDeviceGetMemoryInfo(handle)

                    # Convert bytes to MB
                    gpu_info = {
                        "index": i,
                        "name": name,
                        "total_vram_mb": mem_info.total // (1024 * 1024),
                        "free_vram_mb": mem_info.free // (1024 * 1024),
                        "used_vram_mb": mem_info.used // (1024 * 1024),
                    }

                    # Get compute capability
                    try:
                        cc_major, cc_minor = pynvml.nvmlDeviceGetCudaComputeCapability(handle)
                        gpu_info["compute_capability"] = f"{cc_major}.{cc_minor}"
                    except Exception:
                        pass

                    gpus.append(gpu_info)
            except Exception:
                # If GPU enumeration fails, still return driver version with empty GPU list
                pass

            pynvml.nvmlShutdown()
            return {"driver": driver, "gpus": gpus}
        except ImportError:
            # pynvml not installed
            return None
        except Exception:
            # pynvml initialization failed (common on Windows)
            # Try fallback method using nvidia-smi
            return None
    
    def _try_nvidia_smi(self):
        """
        Detect NVIDIA driver version and GPU info using nvidia-smi command.

        Returns:
            Dict with keys: driver, gpus (list of GPU info dicts)
            None if detection fails
        """
        try:
            out = subprocess.check_output(["nvidia-smi"], encoding="utf-8")

            # Extract driver version
            match = re.search(r"Driver Version:\s+(\d+\.\d+)", out)
            if not match:
                return None

            driver = match.group(1)
            gpus = []

            # Try to extract GPU information using nvidia-smi --query-gpu
            try:
                gpu_query_out = subprocess.check_output(
                    ["nvidia-smi", "--query-gpu=index,name,memory.total,compute_cap", "--format=csv,noheader"],
                    encoding="utf-8"
                )

                for line in gpu_query_out.strip().split("\n"):
                    if not line.strip():
                        continue

                    parts = [p.strip() for p in line.split(",")]
                    if len(parts) >= 3:
                        try:
                            idx = int(parts[0])
                            name = parts[1]
                            # Parse memory: e.g., "4096 MiB" or "4 GB"
                            memory_str = parts[2]
                            memory_mb = self._parse_memory_string(memory_str)

                            if memory_mb > 0:
                                gpu_info = {
                                    "index": idx,
                                    "name": name,
                                    "total_vram_mb": memory_mb,
                                    "free_vram_mb": 0,  # We can't get free memory this way
                                    "used_vram_mb": 0,
                                }

                                # Parse compute capability (4th field)
                                if len(parts) >= 4 and parts[3]:
                                    gpu_info["compute_capability"] = parts[3]

                                gpus.append(gpu_info)
                        except (ValueError, IndexError):
                            pass
            except:
                # GPU query failed, but we still have the driver version
                pass

            return {"driver": driver, "gpus": gpus}
        except:
            pass

        return None

    def _parse_memory_string(self, mem_str: str) -> int:
        """
        Parse memory string like '4096 MiB' or '4 GB' to MB.

        Args:
            mem_str: Memory string from nvidia-smi

        Returns:
            Memory in MB, or 0 if parsing fails
        """
        try:
            mem_str = mem_str.strip()

            # Extract number and unit
            parts = mem_str.split()
            if len(parts) < 2:
                return 0

            value = float(parts[0])
            unit = parts[1].lower()

            if unit == "mib" or unit == "mb":
                return int(value)
            elif unit == "gib" or unit == "gb":
                return int(value * 1024)
            elif unit == "kib" or unit == "kb":
                return int(value / 1024)
            else:
                return 0
        except:
            return 0
    
    def _success_result(self, detection_data: dict, method: str) -> DetectionResult:
        """
        Create a successful detection result with GPU information.

        Args:
            detection_data: Dict with keys: driver, gpus
            method: Detection method (pynvml or nvidia-smi)

        Returns:
            DetectionResult with GPU information in metadata
        """
        from env_doctor.db import get_max_cuda_for_driver

        driver_version = detection_data["driver"]
        gpus = detection_data.get("gpus", [])

        max_cuda = get_max_cuda_for_driver(driver_version)

        # Build metadata
        metadata = {
            "detection_method": method,
            "max_cuda_version": max_cuda,
        }

        # Add GPU information if available
        if gpus:
            metadata["gpu_count"] = len(gpus)
            metadata["gpus"] = gpus

            # Calculate total VRAM across all GPUs
            total_vram_mb = sum(gpu["total_vram_mb"] for gpu in gpus)
            metadata["total_vram_mb"] = total_vram_mb

            # Store primary GPU info (first GPU)
            metadata["primary_gpu_name"] = gpus[0]["name"]
            metadata["primary_gpu_vram_mb"] = gpus[0]["total_vram_mb"]
            if gpus[0].get("compute_capability"):
                metadata["primary_gpu_compute_capability"] = gpus[0]["compute_capability"]
        else:
            # No GPU detected
            metadata["gpu_count"] = 0
            metadata["total_vram_mb"] = 0
            metadata["primary_gpu_name"] = None
            metadata["primary_gpu_vram_mb"] = 0

        return DetectionResult(
            component="nvidia_driver",
            status=Status.SUCCESS,
            version=driver_version,
            metadata=metadata,
        )