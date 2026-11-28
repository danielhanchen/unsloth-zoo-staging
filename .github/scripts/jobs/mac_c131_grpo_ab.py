"""Apple Silicon leg of jobs/c131_grpo_ab.py (staging CI runs mac_* jobs on macOS arm64): torch CPU path, not MLX."""

import os
import runpy
import sys

os.environ.setdefault("UNSLOTH_FORCE_GPU_PATH", "1")
here = os.path.dirname(os.path.abspath(__file__))
sys.argv[0] = os.path.join(here, "c131_grpo_ab.py")
runpy.run_path(sys.argv[0], run_name="__main__")
