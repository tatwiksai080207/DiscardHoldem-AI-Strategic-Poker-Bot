"""Parallel runner: APEX DISCARD v11 Dominator vs Challenger."""
import sys, os
from benchmark_v11_suite import run_series

if __name__ == "__main__":
    bot_v11 = "sparring_bots/bot_v11.py"
    bot_chal = "sparring_bots/bot_challenger.py"
    seeds = list(range(1, 11))
    run_series("APEX DISCARD v11 Dominator vs Challenger", bot_v11, bot_chal, seeds)
