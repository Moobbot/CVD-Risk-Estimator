"""P4c — phien ban THEO TUNG CA cua CVD (can torch/fastapi: chay trong image CVD).

Trong container:  cd /app && python tests/test_case_version.py
Tren host khong co torch: pytest tu bo qua (importorskip).
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    import pytest
    pytest.importorskip("torch")
    pytest.importorskip("fastapi")
except ImportError:  # chay truc tiep trong container (khong co pytest)
    pass

import numpy as np  # noqa: E402


def test_case_version_them_det_simple_khi_ca_nay_do_tim_simple():
    import routes
    routes.MODEL_INFO = {"version": "cvd@iter700.src.x+w.y"}
    assert routes._case_version({"heart_detection": "model"}) == "cvd@iter700.src.x+w.y"
    assert routes._case_version({"heart_detection": "simple"}) == "cvd@iter700.src.x+w.y+det.simple"


def test_case_version_none_khi_chua_co_dinh_danh():
    import routes
    routes.MODEL_INFO = {"model": "cvd", "loaded": False, "version": None}
    assert routes._case_version({"heart_detection": "simple"}) is None


class _Boom:
    """Model phat hien tim ném loi giua ca (vd loi CUDA)."""
    def __call__(self, *a, **k):
        raise RuntimeError("CUDA error: gia lap")


def test_detect_loi_giua_ca_thi_last_method_simple():
    from heart_detector import HeartDetector
    d = HeartDetector()
    d.model = _Boom()
    img = np.random.rand(4, 64, 64).astype("float32")
    _, _, visual = d.detect(img)
    assert visual is None
    assert d.last_method == "simple"


def test_detect_khong_nap_duoc_model_thi_last_method_simple(monkeypatch=None):
    from heart_detector import HeartDetector
    d = HeartDetector()
    d.model = None
    d.load_model = lambda: False  # thieu retinanet_heart.pt
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
