"""Tests for identity detector."""

import pytest
import numpy as np
from src.detector import IdentityDetector

def test_init():
    detector = IdentityDetector(eps=0.3, min_samples=3)
    assert detector.eps == 0.3
    assert detector.min_samples == 3

def test_clustering():
    detector = IdentityDetector()
    # Create dummy embeddings
    emb1 = np.random.rand(512)
    emb2 = emb1 + 0.01  # close
    emb3 = np.random.rand(512)  # far
    embeddings = [emb1, emb2, emb3]
    labels = detector.cluster_identities(embeddings)
    assert len(labels) == 3
    # emb1 and emb2 should cluster together
    assert labels[0] == labels[1]