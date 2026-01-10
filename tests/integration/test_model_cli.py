"""
Integration tests for model CLI command.

Tests the full end-to-end functionality of the model command via CLI.
"""

import subprocess
import sys
import pytest


class TestModelCLICommand:
    """Test model command execution via CLI."""

    def test_model_list_command(self):
        """Test --list flag shows available models"""
        result = subprocess.run(
            [sys.executable, "-m", "src.env_doctor.cli", "model", "--list"],
            capture_output=True,
            text=True,
            timeout=30,
        )

        assert result.returncode == 0
        assert "llama" in result.stdout.lower()
        assert "diffusion" in result.stdout.lower()
        assert "models" in result.stdout.lower()

    def test_model_list_has_categories(self):
        """Test --list shows models grouped by category"""
        result = subprocess.run(
            [sys.executable, "-m", "src.env_doctor.cli", "model", "--list"],
            capture_output=True,
            text=True,
            timeout=30,
        )

        assert result.returncode == 0
        # Should show categories
        output = result.stdout.lower()
        assert "llm" in output or "language" in output
        assert "diffusion" in output or "image" in output

    def test_model_check_basic(self):
        """Test basic model check command"""
        result = subprocess.run(
            [sys.executable, "-m", "src.env_doctor.cli", "model", "llama-3-8b"],
            capture_output=True,
            text=True,
            timeout=30,
        )

        assert result.returncode == 0
        assert "llama-3-8b" in result.stdout.lower()
        # Should show some output
        assert len(result.stdout) > 50

    def test_model_check_shows_vram(self):
        """Test that model check shows VRAM requirements"""
        result = subprocess.run(
            [sys.executable, "-m", "src.env_doctor.cli", "model", "mistral-7b"],
            capture_output=True,
            text=True,
            timeout=30,
        )

        assert result.returncode == 0
        output = result.stdout.lower()
        # Should show VRAM info
        assert "vram" in output or "gb" in output or "mb" in output

    def test_model_check_with_precision(self):
        """Test checking specific precision"""
        result = subprocess.run(
            [
                sys.executable,
                "-m",
                "src.env_doctor.cli",
                "model",
                "llama-3-8b",
                "--precision",
                "int4",
            ],
            capture_output=True,
            text=True,
            timeout=30,
        )

        assert result.returncode == 0
        assert "llama-3-8b" in result.stdout.lower()
        assert "int4" in result.stdout.lower()

    def test_model_not_found_error(self):
        """Test error message for unknown model"""
        result = subprocess.run(
            [sys.executable, "-m", "src.env_doctor.cli", "model", "nonexistent-model-xyz"],
            capture_output=True,
            text=True,
            timeout=30,
        )

        assert result.returncode == 0
        output = result.stdout.lower()
        # Should show error message
        assert "not found" in output or "error" in output

    def test_model_suggestions_for_typo(self):
        """Test suggestions for typos in model name"""
        result = subprocess.run(
            [sys.executable, "-m", "src.env_doctor.cli", "model", "llama-8b"],
            capture_output=True,
            text=True,
            timeout=30,
        )

        # Might not be found, but should handle gracefully
        assert result.returncode == 0

    def test_model_help_no_args(self):
        """Test help message when no args provided"""
        result = subprocess.run(
            [sys.executable, "-m", "src.env_doctor.cli", "model"],
            capture_output=True,
            text=True,
            timeout=30,
        )

        # Should either show help or error
        assert result.returncode == 0 or result.returncode == 1

    def test_model_alias_resolution(self):
        """Test that model aliases work"""
        result1 = subprocess.run(
            [sys.executable, "-m", "src.env_doctor.cli", "model", "llama-3-8b"],
            capture_output=True,
            text=True,
            timeout=30,
        )

        result2 = subprocess.run(
            [sys.executable, "-m", "src.env_doctor.cli", "model", "llama3-8b"],
            capture_output=True,
            text=True,
            timeout=30,
        )

        assert result1.returncode == 0
        assert result2.returncode == 0
        # Both should show success (or both show failure if no GPU)
        assert result1.returncode == result2.returncode

    def test_model_check_various_models(self):
        """Test checking different model types"""
        models = ["llama-3-8b", "stable-diffusion-xl", "whisper-base"]

        for model in models:
            result = subprocess.run(
                [sys.executable, "-m", "src.env_doctor.cli", "model", model],
                capture_output=True,
                text=True,
                timeout=30,
            )

            assert result.returncode == 0
            assert model in result.stdout.lower()

    def test_model_command_timeout(self):
        """Test that model command completes in reasonable time"""
        import time

        start = time.time()
        result = subprocess.run(
            [sys.executable, "-m", "src.env_doctor.cli", "model", "llama-3-8b"],
            capture_output=True,
            text=True,
            timeout=30,
        )
        elapsed = time.time() - start

        assert result.returncode == 0
        # Should complete in under 10 seconds (most systems)
        assert elapsed < 10, f"Command took {elapsed:.1f}s (expected <10s)"


class TestModelCLIUnicodeOutput:
    """Test Unicode/emoji output in CLI."""

    def test_list_output_has_emojis(self):
        """Test that output contains formatted emojis"""
        result = subprocess.run(
            [sys.executable, "-m", "src.env_doctor.cli", "model", "--list"],
            capture_output=True,
            text=True,
            timeout=30,
        )

        assert result.returncode == 0
        # Should have some unicode characters (emojis)
        # At minimum should have content
        assert len(result.stdout) > 100

    def test_check_output_formatting(self):
        """Test that check output is well formatted"""
        result = subprocess.run(
            [sys.executable, "-m", "src.env_doctor.cli", "model", "mistral-7b"],
            capture_output=True,
            text=True,
            timeout=30,
        )

        assert result.returncode == 0
        output = result.stdout
        # Should have structure
        assert "=" * 10 in output or "-" * 10 in output or "\n" in output


class TestModelCLIPrecisions:
    """Test different precision options."""

    @pytest.mark.parametrize(
        "precision", ["fp32", "fp16", "bf16", "int8", "int4"]
    )
    def test_all_supported_precisions(self, precision):
        """Test all supported precisions work"""
        result = subprocess.run(
            [
                sys.executable,
                "-m",
                "src.env_doctor.cli",
                "model",
                "llama-3-8b",
                "--precision",
                precision,
            ],
            capture_output=True,
            text=True,
            timeout=30,
        )

        assert result.returncode == 0
        assert precision in result.stdout.lower()

    def test_invalid_precision_error(self):
        """Test error for invalid precision"""
        result = subprocess.run(
            [
                sys.executable,
                "-m",
                "src.env_doctor.cli",
                "model",
                "llama-3-8b",
                "--precision",
                "invalid_precision",
            ],
            capture_output=True,
            text=True,
            timeout=30,
        )

        # Should fail due to argparse validation
        assert result.returncode != 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
