"""
Docker Compose validator for GPU/CUDA compatibility issues.

This module validates docker-compose.yml files for proper GPU configuration
including deploy.resources.reservations and deprecated runtime syntax.
"""
import os
import re
import shutil
import subprocess
from typing import List, Dict, Any, Optional, Set
from pathlib import Path

try:
    import yaml
except ImportError:
    yaml = None

from .models import ValidationResult, Severity


class ComposeValidator:
    """
    Validates docker-compose.yml files for GPU configuration issues.

    This validator checks for:
    - Missing or incorrect GPU device configuration
    - Deprecated 'runtime: nvidia' syntax
    - Multi-service GPU conflicts
    - Host system requirements (nvidia-container-toolkit)
    """

    def __init__(self, compose_path: str = "docker-compose.yml"):
        """
        Initialize the Docker Compose validator.

        Args:
            compose_path: Path to docker-compose.yml file
        """
        self.compose_path = compose_path
        self.compose_data: Optional[Dict[str, Any]] = None
        self.services: Dict[str, Any] = {}

    def validate(self) -> ValidationResult:
        """
        Validate the docker-compose.yml for GPU issues.

        Returns:
            ValidationResult: Validation result with all detected issues
        """
        result = ValidationResult(file_path=self.compose_path)

        # Check if PyYAML is available
        if yaml is None:
            result.add_issue(
                line_number=0,
                severity=Severity.ERROR,
                issue="PyYAML library not installed",
                recommendation="Install PyYAML: pip install pyyaml"
            )
            return result

        # Load the compose file
        try:
            self._load_compose_file()
        except FileNotFoundError:
            result.add_issue(
                line_number=0,
                severity=Severity.ERROR,
                issue=f"Compose file not found: {self.compose_path}",
                recommendation="Ensure the docker-compose.yml file exists at the specified path"
            )
            return result
        except PermissionError:
            result.add_issue(
                line_number=0,
                severity=Severity.ERROR,
                issue=f"Permission denied reading: {self.compose_path}",
                recommendation="Check file permissions"
            )
            return result
        except yaml.YAMLError as e:
            result.add_issue(
                line_number=0,
                severity=Severity.ERROR,
                issue=f"Invalid YAML syntax: {str(e)}",
                recommendation="Fix YAML syntax errors in the compose file"
            )
            return result
        except Exception as e:
            result.add_issue(
                line_number=0,
                severity=Severity.ERROR,
                issue=f"Error reading compose file: {str(e)}",
                recommendation="Ensure the file is a valid docker-compose.yml"
            )
            return result

        # Validate services
        if not self.services:
            result.add_issue(
                line_number=0,
                severity=Severity.WARNING,
                issue="No services found in compose file",
                recommendation="Add service definitions to the compose file"
            )
            return result

        # Validate each service
        for service_name, service_config in self.services.items():
            self._validate_service(service_name, service_config, result)

        # Check for multi-service GPU conflicts
        self._validate_multi_service_gpu(result)

        # Check host system requirements
        self._validate_host_system(result)

        # Sort issues
        result.sort_issues()

        return result

    def _load_compose_file(self) -> None:
        """
        Load and parse the docker-compose.yml file.

        Raises:
            FileNotFoundError: If file doesn't exist
            PermissionError: If file can't be read
            yaml.YAMLError: If YAML is invalid
        """
        with open(self.compose_path, 'r', encoding='utf-8') as f:
            self.compose_data = yaml.safe_load(f)

        # Handle both v2 and v3 formats
        if 'services' in self.compose_data:
            self.services = self.compose_data['services']
        else:
            # v1 format (less common)
            self.services = {k: v for k, v in self.compose_data.items()
                           if isinstance(v, dict) and k != 'version'}

    def _validate_service(self, service_name: str, service_config: Dict[str, Any],
                         result: ValidationResult) -> None:
        """
        Validate GPU configuration for a single service.

        Args:
            service_name: Name of the service
            service_config: Service configuration dictionary
            result: ValidationResult to add issues to
        """
        # Check if service uses GPU (has image that suggests GPU usage)
        uses_gpu = self._service_likely_uses_gpu(service_config)

        if not uses_gpu:
            # Skip GPU validation for non-GPU services
            return

        # Check for deprecated 'runtime: nvidia' syntax
        if 'runtime' in service_config and service_config['runtime'] == 'nvidia':
            result.add_issue(
                line_number=0,  # YAML line numbers are complex to track
                severity=Severity.WARNING,
                issue=f"Service '{service_name}': Deprecated 'runtime: nvidia' syntax",
                recommendation="Use the new 'deploy.resources.reservations.devices' syntax instead",
                corrected_command=self._generate_gpu_config_snippet()
            )

        # Check for proper GPU device configuration
        has_gpu_config = self._check_gpu_device_config(service_config)

        if not has_gpu_config:
            result.add_issue(
                line_number=0,
                severity=Severity.ERROR,
                issue=f"Service '{service_name}': Missing GPU device configuration",
                recommendation="Add GPU device configuration under deploy.resources.reservations.devices",
                corrected_command=self._generate_gpu_config_snippet()
            )
        else:
            # Validate the GPU config structure
            self._validate_gpu_config_structure(service_name, service_config, result)

    def _service_likely_uses_gpu(self, service_config: Dict[str, Any]) -> bool:
        """
        Heuristic to determine if a service likely uses GPU.

        Args:
            service_config: Service configuration dictionary

        Returns:
            True if service appears to need GPU
        """
        # Check image name for GPU indicators
        image = service_config.get('image', '').lower()
        gpu_indicators = ['cuda', 'gpu', 'nvidia', 'pytorch', 'tensorflow', 'ml', 'ai']

        for indicator in gpu_indicators:
            if indicator in image:
                return True

        # Check for explicit GPU configuration
        if 'runtime' in service_config and service_config['runtime'] == 'nvidia':
            return True

        if self._check_gpu_device_config(service_config):
            return True

        return False

    def _check_gpu_device_config(self, service_config: Dict[str, Any]) -> bool:
        """
        Check if service has proper GPU device configuration.

        Args:
            service_config: Service configuration dictionary

        Returns:
            True if GPU config is present
        """
        try:
            devices = (service_config.get('deploy', {})
                      .get('resources', {})
                      .get('reservations', {})
                      .get('devices', []))
            return len(devices) > 0
        except (AttributeError, TypeError):
            return False

    def _validate_gpu_config_structure(self, service_name: str,
                                       service_config: Dict[str, Any],
                                       result: ValidationResult) -> None:
        """
        Validate the structure of GPU device configuration.

        Args:
            service_name: Name of the service
            service_config: Service configuration dictionary
            result: ValidationResult to add issues to
        """
        try:
            devices = (service_config.get('deploy', {})
                      .get('resources', {})
                      .get('reservations', {})
                      .get('devices', []))

            for device in devices:
                # Check driver field
                driver = device.get('driver')
                if driver != 'nvidia':
                    result.add_issue(
                        line_number=0,
                        severity=Severity.ERROR,
                        issue=f"Service '{service_name}': GPU device driver must be 'nvidia', got '{driver}'",
                        recommendation="Set driver: nvidia in device configuration"
                    )

                # Check capabilities field
                capabilities = device.get('capabilities', [])
                if 'gpu' not in capabilities and 'GPU' not in capabilities:
                    result.add_issue(
                        line_number=0,
                        severity=Severity.ERROR,
                        issue=f"Service '{service_name}': Missing 'gpu' in device capabilities",
                        recommendation="Add 'gpu' to capabilities list: capabilities: [gpu]"
                    )

        except (AttributeError, TypeError) as e:
            result.add_issue(
                line_number=0,
                severity=Severity.ERROR,
                issue=f"Service '{service_name}': Invalid GPU config structure: {str(e)}",
                recommendation="Use the correct deploy.resources.reservations.devices structure"
            )

    def _validate_multi_service_gpu(self, result: ValidationResult) -> None:
        """
        Check for potential GPU conflicts in multi-service setups.

        Args:
            result: ValidationResult to add issues to
        """
        gpu_services = []

        for service_name, service_config in self.services.items():
            if self._service_likely_uses_gpu(service_config):
                gpu_services.append(service_name)

        if len(gpu_services) > 1:
            result.add_issue(
                line_number=0,
                severity=Severity.WARNING,
                issue=f"Multiple services use GPU: {', '.join(gpu_services)}",
                recommendation="Ensure GPU resources are properly allocated. Consider using 'count' or device IDs "
                             "to assign specific GPUs to services. Without limits, all services share all GPUs."
            )

    def _validate_host_system(self, result: ValidationResult) -> None:
        """
        Validate host system has required components for GPU support.

        Args:
            result: ValidationResult to add issues to
        """
        # Check for nvidia-container-toolkit
        has_toolkit = self._check_nvidia_container_toolkit()

        if not has_toolkit:
            result.add_issue(
                line_number=0,
                severity=Severity.WARNING,
                issue="nvidia-container-toolkit may not be installed on host",
                recommendation="Install nvidia-container-toolkit:\n"
                             "  Ubuntu/Debian: sudo apt-get install -y nvidia-container-toolkit\n"
                             "  RHEL/CentOS: sudo yum install -y nvidia-container-toolkit\n"
                             "  Then restart Docker: sudo systemctl restart docker"
            )

        # Check Docker daemon config (if accessible)
        daemon_config_ok = self._check_docker_daemon_config()
        if not daemon_config_ok:
            result.add_issue(
                line_number=0,
                severity=Severity.INFO,
                issue="Could not verify Docker daemon GPU configuration",
                recommendation="Ensure /etc/docker/daemon.json contains:\n"
                             '  {\n'
                             '    "runtimes": {\n'
                             '      "nvidia": {\n'
                             '        "path": "nvidia-container-runtime",\n'
                             '        "runtimeArgs": []\n'
                             '      }\n'
                             '    }\n'
                             '  }'
            )

    def _check_nvidia_container_toolkit(self) -> bool:
        """
        Check if nvidia-container-toolkit is installed.

        Returns:
            True if toolkit is detected
        """
        # Check if nvidia-container-runtime is available
        return shutil.which('nvidia-container-runtime') is not None

    def _check_docker_daemon_config(self) -> bool:
        """
        Check if Docker daemon is configured for GPU support.

        Returns:
            True if config appears correct (or inaccessible)
        """
        daemon_config_path = '/etc/docker/daemon.json'

        if not os.path.exists(daemon_config_path):
            return False

        try:
            with open(daemon_config_path, 'r') as f:
                config = yaml.safe_load(f)
                return 'runtimes' in config and 'nvidia' in config.get('runtimes', {})
        except (PermissionError, yaml.YAMLError, Exception):
            # If we can't read it, don't fail - just indicate we couldn't verify
            return True

    def _generate_gpu_config_snippet(self) -> str:
        """
        Generate correct GPU configuration snippet.

        Returns:
            YAML snippet for GPU configuration
        """
        return """deploy:
  resources:
    reservations:
      devices:
        - driver: nvidia
          count: all  # or 1, 2, etc.
          capabilities: [gpu]"""
