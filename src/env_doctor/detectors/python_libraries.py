# ← Refactored from get_installed_library_version()

import importlib
import importlib.metadata
from typing import Optional
from env_doctor.core.detector import Detector, DetectionResult, Status
from env_doctor.core.registry import DetectorRegistry

@DetectorRegistry.register("python_library")
class PythonLibraryDetector(Detector):
    """Detects installed Python AI libraries and their CUDA versions."""
    
    #making library_name optional so it can be registered as a detector
    def __init__(self, library_name=None):
        self.library_name = library_name
    
    def detect(self) -> DetectionResult:
        try:
            lib = importlib.import_module(self.library_name)
            version = getattr(lib, "__version__", "Unknown")
            cuda_ver = "Unknown"
            cudnn_ver = "Unknown"
            
            if self.library_name == "torch":
                cuda_ver, cudnn_ver = self._detect_torch_cuda(lib)
            elif self.library_name == "tensorflow":
                cuda_ver, cudnn_ver = self._detect_tensorflow_cuda(lib)
            elif self.library_name == "jax":
                cuda_ver = self._detect_jax_cuda()
            
            return DetectionResult(
                component=f"python_library_{self.library_name}",
                status=Status.SUCCESS,
                version=version,
                metadata={
                    "cuda_version": cuda_ver,
                    "cudnn_version": cudnn_ver
                }
            )
        
        except ImportError:
            return DetectionResult(
                component=f"python_library_{self.library_name}",
                status=Status.NOT_FOUND,
                recommendations=[
                    f"Install {self.library_name} using: env-doctor install {self.library_name}"
                ]
            )
    
    def _detect_torch_cuda(self, lib):
        cuda_ver = "Unknown"
        cudnn_ver = "Unknown"
        try:
            cuda_ver = lib.version.cuda
            if hasattr(lib.backends, 'cudnn'):
                raw_cudnn = lib.backends.cudnn.version()
                if raw_cudnn:
                    major = raw_cudnn // 1000
                    minor = (raw_cudnn % 1000) // 100
                    patch = raw_cudnn % 100
                    cudnn_ver = f"{major}.{minor}.{patch}"
        except AttributeError:
            pass
        return cuda_ver, cudnn_ver
    
    def _detect_tensorflow_cuda(self, lib):
        cuda_ver = "Unknown"
        cudnn_ver = "Unknown"
        try:
            sys_config = getattr(lib, "sysconfig", None)
            if sys_config and hasattr(sys_config, "get_build_info"):
                build_info = sys_config.get_build_info()
                cuda_ver = build_info.get("cuda_version", "Unknown")
                cudnn_ver = build_info.get("cudnn_version", "Unknown")
        except Exception:
            pass
        return cuda_ver, cudnn_ver
    
    def _detect_jax_cuda(self):
        try:
            import jaxlib
            if hasattr(jaxlib, "version"):
                packages = [d.metadata['Name'] for d in importlib.metadata.distributions()]
                if any("nvidia-cuda-runtime-cu12" in p for p in packages):
                    return "12.x (via pip)"
                elif any("nvidia-cuda-runtime-cu11" in p for p in packages):
                    return "11.x (via pip)"
        except ImportError:
            return "CPU Only (jaxlib not found)"
        return "Unknown"