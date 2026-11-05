#!/usr/bin/env python3
"""Print one top-level value from a predictor config, for use in scripts/common.sh.

The config is loaded the same way the trainers load it (``_base_`` merged, ``${root}`` resolved).

Usage:  print_config_value.py CONFIG KEY
"""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from omegaconf import OmegaConf  # noqa: E402

from recomb2027.config_io import load_config  # noqa: E402


def main() -> int:
    if len(sys.argv) != 3:
        print(__doc__.strip(), file=sys.stderr)
        return 2
    config_path, key = sys.argv[1], sys.argv[2]
    cfg = OmegaConf.to_container(load_config(config_path), resolve=True)
    value = cfg.get(key)
    if value is None:
        print(f"{config_path}: key '{key}' is not set", file=sys.stderr)
        return 1
    print(value)
    return 0


if __name__ == "__main__":
    sys.exit(main())
