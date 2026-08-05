from pathlib import Path

import pytest

from app.predictor import get_class_name


def test_get_class_name_valid_index():
    assert get_class_name(0) == 'Apple___Apple_scab'


def test_get_class_name_invalid_index():
    assert get_class_name(999) == 'Unknown'
