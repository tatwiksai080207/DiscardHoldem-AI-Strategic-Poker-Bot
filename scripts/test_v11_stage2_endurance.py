"""Stage 2 Finals Endurance Test: APEX DISCARD v11 Dominator vs Challenger (1,000 hands)."""
import subprocess, re

cmd = [
    "py", "-3.11", "Starter_Kit/engine.py",
    "sparring_bots/bot_v11.py", "sparring_bots/bot_challenger.py",
    "--seed", "1337",
    "--stage", "2",
    "--python", "py",
    "--log-dir", ""
]

print("Starting Stage 2 Endurance Match (1,000 hands)...", flush=True)
proc = subprocess.run(cmd, capture_output=True, text=True)
print(proc.stdout, flush=True)
