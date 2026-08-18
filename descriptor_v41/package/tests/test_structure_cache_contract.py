from pathlib import Path

import pytest

from alignn_stage2.structure_cache import SHARD_SIZE, shard_number


def test_fixed_shard_mapping_boundaries():
    assert SHARD_SIZE == 512
    assert shard_number(0) == 0 and shard_number(511) == 0 and shard_number(512) == 1
    with pytest.raises(ValueError):
        shard_number(-1)


def test_calibration_uses_same_global_cache_not_per_seed_cache():
    root = Path(__file__).resolve().parents[1]
    source = (root / "alignn_stage2/calibrate_export.py").read_text(encoding="utf-8")
    assert 'Path(os.environ["ALIGNN_GRAPH_CACHE_ROOT"])' in source
    assert 'work / "graph_cache"' not in source
