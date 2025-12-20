import pytest
from env_doctor import checks, db

def test_imports():
    """Simple test to ensure modules are importable."""
    assert checks is not None
    assert db is not None

def test_database_structure():
    """Verify that compatibility.json is loaded and has correct structure."""
    data = db.DB_DATA
    assert "driver_to_cuda" in data
    assert "recommendations" in data
    assert isinstance(data["driver_to_cuda"], dict)

def test_driver_logic():
    """Test the logic for driver -> max cuda parsing."""
    # Test a known driver
    assert db.get_max_cuda_for_driver("535.129") == "12.2"
    
    # Test a driver that isn't exact match (should use lower bound)
    # 540 should fall back to 535 -> 12.2
    assert db.get_max_cuda_for_driver("540.00") == "12.2"
    
    # Test very old driver (fallback)
    assert db.get_max_cuda_for_driver("300.00") == "10.0"

def test_install_command_generation():
    """Test that we generate install strings correctly."""
    cmd = db.get_install_command("torch", "12.1")
    assert "pip install torch" in cmd
    assert "cu121" in cmd

def test_migration_check_regex():
    """Test the regex for scanning imports."""
    # This is slightly tricky to test without a file, but we can verify the function exists
    assert callable(checks.scan_imports_in_folder)
