#!/usr/bin/env python3
"""Day-19: Gemini-Flash judge on Phi-3.5-vision 358-row predictions."""
import sys
sys.argv = [sys.argv[0],
            "--predictions", "results/day19_phi35vision_heldout/per_pair_base.jsonl",
            "--workers", "3",
            "--out-dir", "results/day19_phi35vision_semantic"]

# Re-use day19_judge_mtvqa_full.py's main()
exec(open("scripts/day19_judge_mtvqa_full.py").read())
