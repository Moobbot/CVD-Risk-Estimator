"""P4c — CVD's PER-CASE version (needs torch/fastapi: run inside the CVD image).

Inside the container:  cd /app && python tests/test_case_version.py
On a host without torch: pytest skips it (importorskip).
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    import pytest
    pytest.importorskip("torch")
    pytest.importorskip("fastapi")
except ImportError:  # run directly in the container (no pytest)
    pass

import numpy as np  # noqa: E402


def test_case_version_appends_det_simple_when_this_case_used_simple():
    import routes
    routes.MODEL_INFO = {"version": "cvd@iter700.src.x+w.y"}
    assert routes._case_version({"heart_detection": "model"}) == "cvd@iter700.src.x+w.y"
    assert routes._case_version({"heart_detection": "simple"}) == "cvd@iter700.src.x+w.y+det.simple"


def test_case_version_is_none_without_identity():
    import routes
    routes.MODEL_INFO = {"model": "cvd", "loaded": False, "version": None}
    assert routes._case_version({"heart_detection": "simple"}) is None


class _Boom:
    """Heart detection model that raises mid-case (e.g. a CUDA error)."""
    def __call__(self, *a, **k):
        raise RuntimeError("CUDA error: simulated")


def test_detect_error_mid_case_sets_last_method_simple():
    from heart_detector import HeartDetector
    d = HeartDetector()
    d.model = _Boom()
    img = np.random.rand(4, 64, 64).astype("float32")
    _, _, visual = d.detect(img)
    assert visual is None
    assert d.last_method == "simple"


def test_detect_model_not_loadable_sets_last_method_simple(monkeypatch=None):
    from heart_detector import HeartDetector
    d = HeartDetector()
    d.model = None
    d.load_model = lambda: False  # retinanet_heart.pt is missing
    img = np.random.rand(4, 64, 64).astype("float32")
    _, _, visual = d.detect(img)
    assert visual is None and d.last_method == "simple"


if __name__ == "__main__":
    fails = 0
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            try:
                fn()
                print(f"PASS {name}")
            except Exception as e:  # noqa: BLE001
                fails += 1
                print(f"FAIL {name}: {e!r}")
    sys.exit(1 if fails else 0)
