"""
Unit tests for PythonLibraryDetector

Tests import handling, including ImportError and other exceptions (DLL errors, etc.)
"""
import pytest
from unittest.mock import patch, MagicMock

from env_doctor.detectors.python_libraries import PythonLibraryDetector
from env_doctor.core.detector import Status


class TestPythonLibraryDetector:
    """Test suite for PythonLibraryDetector."""

    # ===== Test: Successful torch detection =====
    def test_torch_detection_success(self):
        """Test successful PyTorch detection with CUDA info."""
        detector = PythonLibraryDetector("torch")

        # Mock torch module with version and CUDA info
        mock_torch = MagicMock()
        mock_torch.__version__ = "2.1.0"
        mock_torch.version.cuda = "12.1"
        mock_torch.backends.cudnn.version.return_value = 8900

        with patch('importlib.import_module', return_value=mock_torch):
            result = detector.detect()

        assert result.status == Status.SUCCESS
        assert result.version == "2.1.0"
        assert result.metadata["cuda_version"] == "12.1"
        assert result.metadata["cudnn_version"] == "8.9.0"

    # ===== Test: Library not installed (ImportError) =====
    def test_library_not_installed(self):
        """Test when library is not installed (ImportError)."""
        detector = PythonLibraryDetector("torch")

        with patch('importlib.import_module', side_effect=ImportError("No module named 'torch'")):
            result = detector.detect()

        assert result.status == Status.NOT_FOUND
        assert "Install torch using" in result.recommendations[0]

    # ===== Test: Library import fails with DLL error (non-ImportError) =====
    def test_library_dll_error(self):
        """Test when library is installed but fails to import due to DLL error."""
        detector = PythonLibraryDetector("torch")

        # Simulate Windows DLL loading error (OSError)
        dll_error = OSError("DLL load failed while importing torch: The specified module could not be found.")

        with patch('importlib.import_module', side_effect=dll_error):
            result = detector.detect()

        assert result.status == Status.ERROR
        assert len(result.issues) > 0
        assert "Failed to import torch" in result.issues[0]
        assert "installed but failed to load" in result.recommendations[0]

    # ===== Test: Library import fails with other exception =====
    def test_library_generic_error(self):
        """Test when library import fails with generic exception."""
        detector = PythonLibraryDetector("tensorflow")

        generic_error = RuntimeError("Something went wrong during initialization")

        with patch('importlib.import_module', side_effect=generic_error):
            result = detector.detect()

        assert result.status == Status.ERROR
        assert "Failed to import tensorflow" in result.issues[0]
        assert len(result.recommendations) > 0

    # ===== Test: TensorFlow detection =====
    def test_tensorflow_detection_success(self):
        """Test successful TensorFlow detection."""
        detector = PythonLibraryDetector("tensorflow")

        mock_tf = MagicMock()
        mock_tf.__version__ = "2.13.0"
        mock_tf.sysconfig.get_build_info.return_value = {
            "cuda_version": "11.8",
            "cudnn_version": "8.6"
        }

        with patch('importlib.import_module', return_value=mock_tf):
            result = detector.detect()

        assert result.status == Status.SUCCESS
        assert result.version == "2.13.0"
        assert result.metadata["cuda_version"] == "11.8"
        assert result.metadata["cudnn_version"] == "8.6"

    # ===== Test: JAX detection =====
    @patch('importlib.metadata.distributions')
    def test_jax_detection_cu12(self, mock_distributions):
        """Test JAX detection with CUDA 12 runtime."""
        import sys

        detector = PythonLibraryDetector("jax")

        mock_jax = MagicMock()
        mock_jax.__version__ = "0.4.13"

        # Mock jaxlib module in sys.modules so it can be imported
        mock_jaxlib = MagicMock()
        mock_jaxlib.version = "0.4.13"

        # Mock distribution with CUDA 12 runtime
        mock_dist = MagicMock()
        mock_dist.metadata = {"Name": "nvidia-cuda-runtime-cu12"}
        mock_distributions.return_value = [mock_dist]

        # Add jaxlib to sys.modules before the test
        sys.modules['jaxlib'] = mock_jaxlib

        try:
            with patch('importlib.import_module', return_value=mock_jax) as mock_import:
                result = detector.detect()
        finally:
            # Clean up
            if 'jaxlib' in sys.modules:
                del sys.modules['jaxlib']

        assert result.status == Status.SUCCESS
        assert result.metadata["cuda_version"] == "12.x (via pip)"

    # ===== Test: Torch detection with arch_list =====
    def test_torch_detection_with_arch_list(self):
        """Test that arch_list is captured from torch.cuda.get_arch_list()."""
        detector = PythonLibraryDetector("torch")

        mock_torch = MagicMock()
        mock_torch.__version__ = "2.5.0"
        mock_torch.version.cuda = "12.4"
        mock_torch.backends.cudnn.version.return_value = 9100
        mock_torch.cuda.get_arch_list.return_value = [
            "sm_50", "sm_60", "sm_70", "sm_80", "sm_90", "compute_90"
        ]

        with patch('importlib.import_module', return_value=mock_torch):
            result = detector.detect()

        assert result.status == Status.SUCCESS
        assert result.metadata["arch_list"] == [
            "sm_50", "sm_60", "sm_70", "sm_80", "sm_90", "compute_90"
        ]

    # ===== Test: Torch detection without get_arch_list =====
    def test_torch_detection_no_arch_list(self):
        """Test that arch_list is empty when get_arch_list is unavailable."""
        detector = PythonLibraryDetector("torch")

        mock_torch = MagicMock()
        mock_torch.__version__ = "2.1.0"
        mock_torch.version.cuda = "12.1"
        mock_torch.backends.cudnn.version.return_value = 8900
        # Remove get_arch_list
        del mock_torch.cuda.get_arch_list

        with patch('importlib.import_module', return_value=mock_torch):
            result = detector.detect()

        assert result.status == Status.SUCCESS
        assert "arch_list" not in result.metadata

    # ===== Test: Torch without CUDA (CPU-only) =====
    def test_torch_cpu_only(self):
        """Test PyTorch detection when installed without CUDA support."""
        detector = PythonLibraryDetector("torch")

        mock_torch = MagicMock()
        mock_torch.__version__ = "2.1.0"
        # No cuda attribute (CPU-only build) - returns None
        mock_torch.version.cuda = None

        with patch('importlib.import_module', return_value=mock_torch):
            result = detector.detect()

        # Should still succeed but cuda_version will be None
        assert result.status == Status.SUCCESS
        assert result.version == "2.1.0"
        assert result.metadata["cuda_version"] is None

    # ===== Test: Missing version attribute =====
    def test_library_no_version(self):
        """Test when library doesn't have __version__ attribute."""
        detector = PythonLibraryDetector("some_library")

        mock_lib = MagicMock()
        del mock_lib.__version__

        with patch('importlib.import_module', return_value=mock_lib):
            result = detector.detect()

        assert result.status == Status.SUCCESS
        assert result.version == "Unknown"


class TestPythonLibraryDetectorIntegration:
    """Integration tests for PythonLibraryDetector."""

    def test_detector_can_be_instantiated(self):
        """Test that detector can be created with library name."""
        detector = PythonLibraryDetector("torch")
        assert detector.library_name == "torch"

    def test_detector_registered(self):
        """Test that PythonLibraryDetector is properly registered."""
        from env_doctor.core.registry import DetectorRegistry

        detector = DetectorRegistry.get("python_library")
        assert detector is not None
        assert isinstance(detector, PythonLibraryDetector)