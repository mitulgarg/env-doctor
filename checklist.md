# PR Checklist: Core Detector Architecture (#1)

## Pre-Development
- [ ] Branch created: `git checkout -b feature/core-detector-architecture`
- [ ] Issue #1 assigned to you
- [ ] Read through all acceptance criteria

---

## Implementation Checklist

### Part 1: Create Directory Structure
- [ ] Create `src/env_doctor/core/` directory
- [ ] Create `src/env_doctor/core/__init__.py`

### Part 2: Implement detector.py
- [ ] Create `src/env_doctor/core/detector.py`
- [ ] Add file docstring explaining the module
- [ ] Import required types:
  - [ ] `from abc import ABC, abstractmethod`
  - [ ] `from dataclasses import dataclass, field`
  - [ ] `from typing import Optional, List, Dict, Any`
  - [ ] `from enum import Enum`

#### Status Enum
- [ ] Create `Status` enum with 4 values:
  - [ ] `SUCCESS = "success"`
  - [ ] `WARNING = "warning"`
  - [ ] `ERROR = "error"`
  - [ ] `NOT_FOUND = "not_found"`
- [ ] Add docstring to `Status` enum

#### DetectionResult Dataclass
- [ ] Create `DetectionResult` dataclass with fields:
  - [ ] `component: str` (required)
  - [ ] `status: Status` (required)
  - [ ] `version: Optional[str] = None`
  - [ ] `path: Optional[str] = None`
  - [ ] `metadata: Dict[str, Any] = field(default_factory=dict)`
  - [ ] `issues: List[str] = field(default_factory=list)`
  - [ ] `recommendations: List[str] = field(default_factory=list)`
- [ ] Add comprehensive docstring with field descriptions
- [ ] Add `@property` method: `detected(self) -> bool`
  - [ ] Returns `True` if `status == Status.SUCCESS`
- [ ] Add helper method: `add_issue(self, issue: str) -> None`
- [ ] Add helper method: `add_recommendation(self, rec: str) -> None`

#### Detector Base Class
- [ ] Create `Detector` abstract base class inheriting from `ABC`
- [ ] Add class docstring with usage example
- [ ] Add abstract method: `detect(self) -> DetectionResult`
  - [ ] Mark with `@abstractmethod`
  - [ ] Add docstring
- [ ] Add concrete method: `can_run(self) -> bool`
  - [ ] Default implementation returns `True`
  - [ ] Add docstring explaining override usage
- [ ] Add property: `name(self) -> str`
  - [ ] Returns `self.__class__.__name__.replace("Detector", "").lower()`
  - [ ] Add docstring

### Part 3: Implement registry.py
- [ ] Create `src/env_doctor/core/registry.py`
- [ ] Add file docstring explaining plugin system
- [ ] Import required types:
  - [ ] `from typing import Dict, List, Type`
  - [ ] `from .detector import Detector`

#### DetectorRegistry Class
- [ ] Create `DetectorRegistry` class
- [ ] Add class docstring with usage example
- [ ] Add class variable: `_detectors: Dict[str, Type[Detector]] = {}`
- [ ] Implement `register(cls, name: str)` classmethod:
  - [ ] Takes detector name as parameter
  - [ ] Returns decorator function
  - [ ] Decorator adds detector class to `_detectors` dict
  - [ ] Decorator returns the class unchanged
  - [ ] Raises `ValueError` if name already registered
  - [ ] Add docstring with example
- [ ] Implement `get(cls, name: str) -> Detector` classmethod:
  - [ ] Returns instance of detector by name
  - [ ] Raises `KeyError` if detector not found
  - [ ] Add docstring
- [ ] Implement `all(cls) -> List[Detector]` classmethod:
  - [ ] Returns list of all detector instances
  - [ ] Add docstring
- [ ] Implement `get_names(cls) -> List[str]` classmethod:
  - [ ] Returns list of registered detector names
  - [ ] Add docstring

### Part 4: Update __init__.py
- [ ] Update `src/env_doctor/core/__init__.py` to export:
  - [ ] `from .detector import Detector, DetectionResult, Status`
  - [ ] `from .registry import DetectorRegistry`

### Part 5: Create exceptions.py (Optional but recommended)
- [ ] Create `src/env_doctor/core/exceptions.py`
- [ ] Add `DetectorError` base exception class
- [ ] Add `DetectorNotFoundError` exception
- [ ] Add `DetectorRegistrationError` exception
- [ ] Add docstrings to all exceptions
- [ ] Export in `__init__.py`

---

## Testing Checklist

### Create Test File
- [ ] Create `tests/unit/core/` directory
- [ ] Create `tests/unit/core/__init__.py`
- [ ] Create `tests/unit/core/test_detector.py`

### Test DetectionResult
- [ ] Test creating DetectionResult with required fields only
- [ ] Test creating DetectionResult with all fields
- [ ] Test `detected` property returns `True` for SUCCESS
- [ ] Test `detected` property returns `False` for NOT_FOUND
- [ ] Test `add_issue()` appends to issues list
- [ ] Test `add_recommendation()` appends to recommendations list
- [ ] Test default factory for `metadata`, `issues`, `recommendations`

### Test Detector Base Class
- [ ] Test cannot instantiate Detector directly (abstract class)
- [ ] Test `can_run()` returns `True` by default
- [ ] Test `name` property returns correct name
- [ ] Test `name` property strips "Detector" suffix
- [ ] Test subclass must implement `detect()` method

### Test DetectorRegistry
- [ ] Test registering a detector with `@register()` decorator
- [ ] Test `get()` returns correct detector instance
- [ ] Test `get()` raises `KeyError` for unknown detector
- [ ] Test `all()` returns all registered detectors
- [ ] Test `get_names()` returns list of detector names
- [ ] Test registering duplicate name raises `ValueError`
- [ ] Test registry is shared across all imports (singleton behavior)

### Create Example Detector for Testing
- [ ] Create dummy detector class for tests:
```python
@DetectorRegistry.register("test")
class TestDetector(Detector):
    def detect(self) -> DetectionResult:
        return DetectionResult(
            component="test",
            status=Status.SUCCESS,
            version="1.0.0"
        )
```
- [ ] Use this in registry tests

---

## Documentation Checklist

### Code Documentation
- [ ] All classes have docstrings (Google style)
- [ ] All methods have docstrings
- [ ] All public APIs have type hints
- [ ] Examples in docstrings are valid code
- [ ] Complex logic has inline comments

### Module Docstrings
- [ ] `detector.py` has module docstring explaining:
  - [ ] Purpose of the module
  - [ ] How to use Detector base class
  - [ ] Example of creating a detector
- [ ] `registry.py` has module docstring explaining:
  - [ ] Plugin system purpose
  - [ ] How to register a detector
  - [ ] Example usage

### Example Code in Docstrings
- [ ] `Detector` class docstring includes example:
```python
"""
Example:
    @DetectorRegistry.register("driver")
    class DriverDetector(Detector):
        def detect(self) -> DetectionResult:
            # Detection logic here
            return DetectionResult(
                component="driver",
                status=Status.SUCCESS
            )
"""
```

- [ ] `DetectorRegistry` docstring includes example:
```python
"""
Example:
    # Register a detector
    @DetectorRegistry.register("my_detector")
    class MyDetector(Detector):
        ...
    
    # Use registry to get all detectors
    detectors = DetectorRegistry.all()
    for detector in detectors:
        result = detector.detect()
"""
```

---

## Code Quality Checklist

### Type Hints
- [ ] All function parameters have type hints
- [ ] All return types specified
- [ ] Use `Optional[]` for nullable types
- [ ] Use `List[]`, `Dict[]` instead of `list`, `dict`
- [ ] Import types from `typing` module

### Code Style
- [ ] Run `black src/env_doctor/core/` (auto-format)
- [ ] Run `ruff check src/env_doctor/core/` (linting)
- [ ] No lines exceed 100 characters
- [ ] Imports organized (stdlib, third-party, local)
- [ ] Two blank lines between top-level definitions

### Best Practices
- [ ] Use `dataclass` instead of manual `__init__`
- [ ] Use `field(default_factory=dict)` for mutable defaults
- [ ] Use `@property` for computed attributes
- [ ] Use `@abstractmethod` for required overrides
- [ ] Use `classmethod` for registry methods

---

## Testing Execution Checklist

### Run Tests Locally
- [ ] Run all tests: `pytest tests/unit/core/test_detector.py -v`
- [ ] All tests pass
- [ ] Run with coverage: `pytest tests/unit/core/test_detector.py --cov=env_doctor.core`
- [ ] Coverage >90% for new files
- [ ] No warnings or errors

### Test Output Verification
- [ ] Test names are descriptive
- [ ] Failure messages are clear
- [ ] No skipped tests
- [ ] No flaky tests (run 3 times to verify)

---

## Integration Verification

### Manual Testing
- [ ] Start Python REPL and test imports:
```python
from env_doctor.core import Detector, DetectionResult, Status, DetectorRegistry

# Test Status enum
print(Status.SUCCESS)  # Should print: Status.SUCCESS

# Test DetectionResult creation
result = DetectionResult(component="test", status=Status.SUCCESS)
print(result.detected)  # Should print: True

# Test adding issues
result.add_issue("Test issue")
print(result.issues)  # Should print: ['Test issue']

# Test registry
@DetectorRegistry.register("example")
class ExampleDetector(Detector):
    def detect(self):
        return DetectionResult(component="example", status=Status.SUCCESS)

# Get detector
detector = DetectorRegistry.get("example")
print(type(detector))  # Should print: <class 'ExampleDetector'>

# Get all detectors
all_detectors = DetectorRegistry.all()
print(len(all_detectors))  # Should print: 1
```

### Verify Design Goals
- [ ] Can create new detector in <10 lines of code
- [ ] Registry automatically discovers detectors
- [ ] Adding detector requires zero changes to other files
- [ ] DetectionResult format is consistent
- [ ] Easy to understand for new contributors

---

## Git Workflow Checklist

### Commit Strategy
- [ ] Make atomic commits (one logical change per commit)
- [ ] Commit messages follow convention:
  - [ ] Format: `feat: add core detector architecture`
  - [ ] Body explains "why" not "what"

### Suggested Commits
1. [ ] `feat: add Status enum and DetectionResult dataclass`
2. [ ] `feat: add Detector abstract base class`
3. [ ] `feat: add DetectorRegistry plugin system`
4. [ ] `test: add unit tests for detector core`
5. [ ] `docs: add docstrings to detector core`

### Before Pushing
- [ ] All tests pass locally
- [ ] No uncommitted changes
- [ ] No debug prints or commented code
- [ ] Git history is clean (no "WIP" commits)

---

## Pull Request Checklist

### PR Description
- [ ] Title: `feat: add core detector architecture (#1)`
- [ ] Description references issue: `Closes #1`
- [ ] Description explains:
  - [ ] What was implemented
  - [ ] Design decisions made
  - [ ] How to test it
- [ ] Screenshots/examples included (if applicable)

### PR Content
- [ ] Only contains changes for this issue
- [ ] No unrelated formatting changes
- [ ] No merge commits (rebase if needed)
- [ ] Branch is up to date with main

### PR Metadata
- [ ] Assignees: yourself
- [ ] Reviewers: other co-founder
- [ ] Labels: `enhancement`, `architecture`
- [ ] Milestone: `v0.2.0`
- [ ] Linked to issue #1

---

## Code Review Preparation

### Self-Review
- [ ] Review your own PR diff on GitHub
- [ ] Check for typos in comments/docstrings
- [ ] Verify all acceptance criteria met
- [ ] Add comments explaining complex logic

### Review Points to Highlight
- [ ] "Plugin system enables zero-config detector addition"
- [ ] "All future detectors will use this pattern"
- [ ] "Type hints enable IDE autocomplete"
- [ ] "Tests cover all code paths"

### Questions for Reviewer
- [ ] "Does the API feel intuitive?"
- [ ] "Any edge cases I missed?"
- [ ] "Documentation clear enough?"

---

## Merge Checklist

### Before Merging
- [ ] All PR checks pass (tests, linting)
- [ ] Approved by reviewer
- [ ] All review comments addressed
- [ ] Branch up to date with main
- [ ] No merge conflicts

### After Merging
- [ ] Delete feature branch
- [ ] Verify main branch builds
- [ ] Close issue #1
- [ ] Update project board
- [ ] Celebrate! 🎉

---

## Files Changed Summary

Expected files in this PR:
```
src/env_doctor/core/__init__.py          (new)
src/env_doctor/core/detector.py          (new)
src/env_doctor/core/registry.py          (new)
src/env_doctor/core/exceptions.py        (new, optional)
tests/unit/core/__init__.py              (new)
tests/unit/core/test_detector.py         (new)
```

Total: 5-6 new files, 0 modified files

---

## Time Tracking

Estimated time: 3 hours
- [ ] 0.5 hrs: Create detector.py
- [ ] 0.5 hrs: Create registry.py  
- [ ] 1.0 hr: Write tests
- [ ] 0.5 hrs: Documentation
- [ ] 0.5 hrs: Testing & cleanup

Actual time: _____ hours

---

## Post-Merge Verification

After PR is merged to main:
- [ ] Pull latest main
- [ ] Run full test suite: `pytest`
- [ ] Verify imports work:
  ```python
  from env_doctor.core import Detector, DetectionResult, Status, DetectorRegistry
  ```
- [ ] Check coverage didn't drop: `pytest --cov=env_doctor`
- [ ] Ready for dependent issues (#2, #3, #4)

---

## Common Issues & Solutions

### Issue: "Cannot import Status"
**Solution**: Make sure `__init__.py` exports it:
```python
from .detector import Detector, DetectionResult, Status
```

### Issue: "DetectorRegistry.register not working"
**Solution**: Verify decorator returns the class:
```python
def decorator(detector_class):
    cls._detectors[name] = detector_class
    return detector_class  # Don't forget this!
return decorator
```

### Issue: "Tests failing with 'no module named env_doctor'"
**Solution**: Install in dev mode:
```bash
pip install -e .
```

### Issue: "mypy complaining about types"
**Solution**: Add type: ignore comments or fix type hints:
```python
_detectors: Dict[str, Type[Detector]] = {}
```

---

## Success Criteria Met?

Before submitting PR, verify all criteria from issue #1:

✅ **Core Requirements**
- [ ] `Status` enum with 4 values
- [ ] `DetectionResult` dataclass with 7 fields
- [ ] `Detector` abstract base class
- [ ] `detect()` abstract method
- [ ] `DetectorRegistry` class
- [ ] `@register()` decorator
- [ ] `get()` and `all()` methods

✅ **Quality Requirements**
- [ ] All code has type hints
- [ ] All classes/methods have docstrings
- [ ] Tests written (>90% coverage)
- [ ] Tests passing
- [ ] Documentation complete

✅ **Design Requirements**
- [ ] Easy to add new detectors (<10 lines)
- [ ] Plugin system works automatically
- [ ] Consistent API across all detectors
- [ ] Extensible for future features

---

## Ready to Submit?

Final checks:
- [ ] All checkboxes in this document checked
- [ ] All tests passing: `pytest`
- [ ] Linting passing: `ruff check src/`
- [ ] Type checking passing: `mypy src/env_doctor/core/`
- [ ] Branch pushed: `git push origin feature/core-detector-architecture`
- [ ] PR created on GitHub
- [ ] PR description complete

**If all checked, submit PR and request review!** 🚀

---

## Notes & Decisions

Document any important decisions made during implementation:

**Design Decision 1**: ___________________________________________

**Alternative Considered**: _______________________________________

**Reason for Choice**: ____________________________________________

---

**Estimated Completion**: _____ / _____ / _____
**Actually Completed**: _____ / _____ / _____
**Merged to Main**: _____ / _____ / _____