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
