import pytest
from env_doctor.core import Detector, DetectionResult, Status, DetectorRegistry
from env_doctor.core.exceptions import DetectorRegistrationError, DetectorNotFoundError

# --- Test Data & Classes ---

class TestDetectorSimple(Detector):
    def detect(self) -> DetectionResult:
        return DetectionResult("test", Status.SUCCESS)

# --- Detector Base Logic Tests ---

def test_detector_cannot_instantiate_abstract():
    """Ensure we cannot create an instance of the abstract Detector class directly."""
    with pytest.raises(TypeError):
        Detector()

def test_detector_subclass_must_implement_detect():
    """Ensure subclasses fail without detect method."""
    class BadDetector(Detector):
        pass
    with pytest.raises(TypeError):
        BadDetector()

def test_detector_name_logic():
    """Ensure .name property strips 'Detector' suffix."""
    det = TestDetectorSimple()
    assert det.name == "testsimple" # TestDetectorSimple -> testdetectorsimple -> replace -> testsimple? 
    # Logic is: self.__class__.__name__.replace("Detector", "").lower()
    # TestDetectorSimple -> "TestDetectorSimple" -> "TestSimple" -> "testsimple"
    
    class ComplexgpuDetector(Detector):
       def detect(self): return None
    
    det2 = ComplexgpuDetector()
    assert det2.name == "complexgpu"

def test_detector_can_run_default():
    """Ensure can_run returns True by default."""
    det = TestDetectorSimple()
    assert det.can_run() is True

# --- DetectionResult Tests ---

def test_detection_result_success_property():
    res = DetectionResult(component="test", status=Status.SUCCESS)
    assert res.detected is True

def test_detection_result_failure_property():
    res = DetectionResult(component="test", status=Status.ERROR)
    assert res.detected is False

def test_detection_result_helpers():
    res = DetectionResult(component="test", status=Status.WARNING)
    res.add_issue("issue1")
    res.add_recommendation("rec1")
    
    assert "issue1" in res.issues
    assert "rec1" in res.recommendations

def test_detection_result_defaults():
    res = DetectionResult(component="foo", status=Status.WARNING)
    assert res.metadata == {}
    assert res.issues == []
    assert res.recommendations == []

# --- Registry Tests ---

# We need a clean registry for each test or unique names, since registry is Singleton-ish on the class.
# We will use unique names to avoid collisions between tests.

def test_registry_registration_and_get():
    name = "unit_test_reg_1"
    
    @DetectorRegistry.register(name)
    class MyRegistryDetector(Detector):
        def detect(self): return None
        
    instance = DetectorRegistry.get(name)
    assert isinstance(instance, MyRegistryDetector)

def test_registry_duplicate_registration_raises_error():
    name = "unit_test_duplicate"
    
    @DetectorRegistry.register(name)
    class First(Detector):
        def detect(self): return None
        
    with pytest.raises(DetectorRegistrationError):
        @DetectorRegistry.register(name)
        class Second(Detector):
            def detect(self): return None

def test_registry_get_not_found():
    with pytest.raises(DetectorNotFoundError):
        DetectorRegistry.get("non_existent_detector_xyz")

def test_registry_all_returns_instances():
    # Helper lists contain instantiated objects
    # We don't know state of other tests, but we expect at least the ones we registered
    dets = DetectorRegistry.all()
    assert len(dets) >= 1
    
def test_registry_get_names():
    names = DetectorRegistry.get_names()
    assert isinstance(names, list)
    assert "unit_test_reg_1" in names


# --- DetectionResult.to_dict() Tests ---

def test_detection_result_to_dict_basic():
    """Test basic to_dict serialization."""
    result = DetectionResult(
        component="test_component",
        status=Status.SUCCESS,
        version="1.0",
        path="/test/path"
    )

    data = result.to_dict()

    assert data["component"] == "test_component"
    assert data["status"] == "success"
    assert data["detected"] is True
    assert data["version"] == "1.0"
    assert data["path"] == "/test/path"
    assert data["metadata"] == {}
    assert data["issues"] == []
    assert data["recommendations"] == []


def test_detection_result_to_dict_with_metadata():
    """Test to_dict with metadata."""
    result = DetectionResult(
        component="cuda_toolkit",
        status=Status.SUCCESS,
        version="12.2",
        metadata={
            "installation_count": 2,
            "cuda_home": {"status": "set", "value": "/usr/local/cuda"},
            "driver_compatibility": {"compatible": True}
        }
    )

    data = result.to_dict()

    assert data["metadata"]["installation_count"] == 2
    assert data["metadata"]["cuda_home"]["status"] == "set"
    assert data["metadata"]["driver_compatibility"]["compatible"] is True


def test_detection_result_to_dict_with_issues():
    """Test to_dict with issues and recommendations."""
    result = DetectionResult(
        component="cuda_toolkit",
        status=Status.WARNING,
        version="11.8"
    )
    result.add_issue("CUDA_HOME not set")
    result.add_issue("Multiple CUDA installations detected")
    result.add_recommendation("Set CUDA_HOME to /usr/local/cuda-11.8")
    result.add_recommendation("Remove old CUDA installations")

    data = result.to_dict()

    assert len(data["issues"]) == 2
    assert "CUDA_HOME not set" in data["issues"]
    assert "Multiple CUDA installations detected" in data["issues"]
    assert len(data["recommendations"]) == 2
    assert "Set CUDA_HOME to /usr/local/cuda-11.8" in data["recommendations"]


def test_detection_result_to_dict_not_found():
    """Test to_dict with NOT_FOUND status."""
    result = DetectionResult(
        component="nvidia_driver",
        status=Status.NOT_FOUND
    )

    data = result.to_dict()

    assert data["status"] == "not_found"
    assert data["detected"] is False
    assert data["version"] is None
    assert data["path"] is None


def test_detection_result_to_dict_error():
    """Test to_dict with ERROR status."""
    result = DetectionResult(
        component="test",
        status=Status.ERROR
    )
    result.add_issue("Critical failure detected")

    data = result.to_dict()

    assert data["status"] == "error"
    assert data["detected"] is False
    assert "Critical failure detected" in data["issues"]


def test_detection_result_to_dict_json_serializable():
    """Test that to_dict output is JSON-serializable."""
    import json

    result = DetectionResult(
        component="test_component",
        status=Status.SUCCESS,
        version="1.0",
        path="/test/path",
        metadata={"key": "value", "number": 42, "nested": {"a": 1}},
    )
    result.add_issue("test issue")
    result.add_recommendation("test rec")

    data = result.to_dict()

    # Should not raise exception
    json_str = json.dumps(data)

    # Verify we can parse it back
    parsed = json.loads(json_str)
    assert parsed["component"] == "test_component"
    assert parsed["status"] == "success"
    assert parsed["metadata"]["key"] == "value"


def test_detection_result_to_dict_all_statuses():
    """Test to_dict with all possible Status enum values."""
    statuses = [Status.SUCCESS, Status.WARNING, Status.ERROR, Status.NOT_FOUND]
    expected_values = ["success", "warning", "error", "not_found"]

    for status, expected_value in zip(statuses, expected_values):
        result = DetectionResult(component="test", status=status)
        data = result.to_dict()
        assert data["status"] == expected_value
