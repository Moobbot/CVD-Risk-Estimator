"""P5f — known answer: the reference CT case must give the CVD score recorded in P4d.

The value holds for the SIMPLE heart detector only (no RetinaNet checkpoint: the service
reports `+det.simple`). With the RetinaNet detector the cropped slices differ and so does the
score (P0 measured 0.779 with it) — do not compare across detector methods.

The case (291 slices, one series) is patient data and is NOT distributed with the code.
Run inside the service image against a local copy of the case:

    docker run --rm -v <case>:/case:ro -v <encoder-only checkpoint dir>:/app/checkpoint \
        -e CVD_KNOWN_ANSWER_DIR=/case <cvd image> python -m pytest tests/test_known_answer.py

Skipped without the variable (CI) or without torch.
"""
import math
import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

CASE = os.environ.get("CVD_KNOWN_ANSWER_DIR")
pytestmark = pytest.mark.skipif(not CASE, reason="CVD_KNOWN_ANSWER_DIR not set")

EXPECTED_SIMPLE_DETECTOR = 0.5660447478294373  # P4d, cvd@iter700 ... +det.simple


def test_reference_case_score_with_the_simple_detector(tmp_path):
    pytest.importorskip("torch")
    from call_model import load_model, predict

    heart_detector, model = load_model()
    assert model is not None, "the CVD model did not load"
    pred_dict, _, _ = predict(CASE, str(tmp_path), heart_detector, model, "known-answer", create_gif=False)
    # load_model() returns a detector object even without its checkpoint; what counts is the
    # method THIS case used (the service reports it as `+det.simple`).
    if pred_dict.get("heart_detection") != "simple":
        pytest.skip("the RetinaNet detector ran: the reference value is for the simple detector")
    score = pred_dict["predictions"][0]["score"]
    assert math.isclose(score, EXPECTED_SIMPLE_DETECTOR, rel_tol=1e-6), score
