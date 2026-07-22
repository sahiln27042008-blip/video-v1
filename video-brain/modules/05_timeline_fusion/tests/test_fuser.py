"""Tests for timeline fuser."""

import pytest
import json
from pathlib import Path
from src.fuser import TimelineFuser
from src.models import TimelineSegment, TimelineResult

def test_parse_time():
    fuser = TimelineFuser()
    assert fuser.parse_time("00:01:30.500") == 90.5
    assert fuser.parse_time("01:00:00.000") == 3600.0

def test_format_time():
    fuser = TimelineFuser()
    assert fuser.format_time(90.5) == "00:01:30.500"
    assert fuser.format_time(3600.0) == "01:00:00.000"