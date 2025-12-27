# ← Refactored from get_nvidia_driver_version()

import subprocess
import re
from env_doctor.core.detector import Detector, DetectionResult, Status
from env_doctor.core.registry import DetectorRegistry

@DetectorRegistry.register("nvidia_driver")
class NvidiaDriverDetector(Detector):
    """Detects NVIDIA GPU driver version."""
    
    def detect(self) -> DetectionResult:
        # 1. Try NVML
        driver = self._try_nvml()
        if driver:
            return self._success_result(driver, method="pynvml")
        
        # 2. Try nvidia-smi
        driver = self._try_nvidia_smi()
        if driver:
            return self._success_result(driver, method="nvidia-smi")
        
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
        try:
            import pynvml
            pynvml.nvmlInit()
            driver = pynvml.nvmlSystemGetDriverVersion().decode()
            pynvml.nvmlShutdown()
            return driver
        except:
            return None
    
    def _try_nvidia_smi(self):
        try:
            out = subprocess.check_output(["nvidia-smi"], encoding="utf-8")
            match = re.search(r"Driver Version:\s+(\d+\.\d+)", out)
            if match:
                return match.group(1)
        except:
            return None
        return None
    
    def _success_result(self, driver_version: str, method: str) -> DetectionResult:
        from env_doctor.db import get_max_cuda_for_driver
        max_cuda = get_max_cuda_for_driver(driver_version)
        
        return DetectionResult(
            component="nvidia_driver",
            status=Status.SUCCESS,
            version=driver_version,
            metadata={
                "detection_method": method,
                "max_cuda_version": max_cuda
            }
        )