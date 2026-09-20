import sys
import os
from pathlib import Path

# Ensure root directory and BIB-2 are on sys.path
_ROOT = str(Path(__file__).resolve().parent.parent.parent)
_BIB2_DIR = os.path.join(_ROOT, "BIB-2")
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)
if _BIB2_DIR not in sys.path:
    sys.path.insert(0, _BIB2_DIR)

import pytest
import numpy as np
from bib2.brain import BIB2NervousSystem
from neuro_chess.adapter import ChessAdapter
from neuro_chess.training import GrandmasterTrainer


def test_imprint_and_sleep():
    brain = BIB2NervousSystem(seed=42)
    adapter = ChessAdapter(brain)
    trainer = GrandmasterTrainer(brain, adapter)

    count = trainer.imprint_grandmaster_motifs(limit=10)
    assert count == 10
    assert len(brain.limbic.hippocampus.ca3.stored_patterns) >= 10

    stats = trainer.consolidate_sleep()
    assert stats["downscaled"] is True
    assert stats["replayed_count"] >= 0


def test_spar_match():
    brain = BIB2NervousSystem(seed=42)
    adapter = ChessAdapter(brain)
    trainer = GrandmasterTrainer(brain, adapter)

    result = trainer.spar_match(max_moves=20)
    assert "moves_played" in result
    assert result["moves_played"] > 0
    assert "winner" in result


def test_save_and_load_checkpoint():
    brain = BIB2NervousSystem(seed=42)
    adapter = ChessAdapter(brain)
    trainer = GrandmasterTrainer(brain, adapter)
    trainer.imprint_grandmaster_motifs(limit=5)

    save_file = os.path.join(os.path.dirname(__file__), "temp_test_brain.npz")
    try:
        trainer.save_checkpoint(save_file)
        assert os.path.exists(save_file)

        # Load into a new brain
        new_brain = BIB2NervousSystem(seed=123)
        new_adapter = ChessAdapter(new_brain)
        new_trainer = GrandmasterTrainer(new_brain, new_adapter)
        new_trainer.load_checkpoint(save_file)

        assert len(new_brain.limbic.hippocampus.ca3.stored_patterns) == 5
        assert np.allclose(
            new_brain.limbic.hippocampus.ca3.recurrent_matrix,
            brain.limbic.hippocampus.ca3.recurrent_matrix
        )
    finally:
        if os.path.exists(save_file):
            os.remove(save_file)
