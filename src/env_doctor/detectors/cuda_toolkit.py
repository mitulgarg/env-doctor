# ← Refactored from get_system_cuda_version()

import shutil
import subprocess
import os
import re
from env_doctor.core.detector import Detector, DetectionResult, Status
from env_doctor.core.registry import DetectorRegistry

@DetectorRegistry.register("cuda_toolkit")
class CudaToolkitDetector(Detector):
    """Detects system CUDA toolkit (nvcc compiler)."""
    
    def detect(self) -> DetectionResult:
        nvcc_path = shutil.which("nvcc")
        if not nvcc_path and os.path.exists("/usr/local/cuda/bin/nvcc"):
            nvcc_path = "/usr/local/cuda/bin/nvcc"
        
        if not nvcc_path:
            return DetectionResult(
                component="cuda_toolkit",
                status=Status.NOT_FOUND,
                recommendations=[
                    "Install CUDA Toolkit: https://developer.nvidia.com/cuda-downloads",
                    "Or use pip-installed libraries with bundled CUDA (no nvcc needed for inference)"
                ]
            )
        
        try:
            result = subprocess.check_output([nvcc_path, "--version"], encoding="utf-8")
            match = re.search(r"release (\d+\.\d+)", result)
            if match:
                version = match.group(1)
                return DetectionResult(
                    component="cuda_toolkit",
                    status=Status.SUCCESS,
                    version=version,
                    path=nvcc_path,
                    metadata={"compiler": "nvcc"}
                )
        except Exception as e:
            return DetectionResult(
                component="cuda_toolkit",
                status=Status.ERROR,
                path=nvcc_path,
                issues=[f"Failed to run nvcc: {str(e)}"]
            )
        
        return DetectionResult(
            component="cuda_toolkit",
            status=Status.ERROR,
            path=nvcc_path,
            issues=["Could not parse nvcc version output"]
        )