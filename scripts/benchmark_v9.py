import sys
import subprocess
import time

def run_match(bot_a, bot_b, seed, stage=1):
    cmd = [sys.executable, 'Starter_Kit/engine.py', bot_a, bot_b, '--seed', str(seed), '--stage', str(stage)]
    res = subprocess.run(cmd, capture_output=True, text=True)
    net_a, net_b = 0, 0
    time_a, time_b = 0.0, 0.0
    notices_a, notices_b = 0, 0
    winner = 'draw'
    for line in res.stdout.splitlines():
        if 'A  ' in line and 'net' in line:
            parts = line.split()
            net_a = int(parts[parts.index('net') + 1])
            time_a = float(parts[parts.index('used') + 1])
            notices_a = int(parts[parts.index('notices') + 1])
        elif 'B  ' in line and 'net' in line:
            parts = line.split()
            net_b = int(parts[parts.index('net') + 1])
            time_b = float(parts[parts.index('used') + 1])
            notices_b = int(parts[parts.index('notices') + 1])
        elif 'Result:' in line:
            if 'A wins' in line:
                winner = 'A'
            elif 'B wins' in line:
                winner = 'B'
            else:
                winner = 'draw'
    return {
        'seed': seed,
        'net_a': net_a,
        'net_b': net_b,
        'time_a': time_a,
        'time_b': time_b,
        'notices_a': notices_a,
        'notices_b': notices_b,
        'winner': winner
    }

print("=== BATTLE GAUNTLET: APEX DISCARD v9 vs APEX_DH_2 (20 Matches) ===", flush=True)
v9_vs_v2 = []
for seed in range(1, 21):
    r = run_match('bot/bot_v9.py', 'sparring_bots/bot_apex_dh_v2.py', seed)
    v9_vs_v2.append(r)
    print(f"Seed {seed:2d}: v9 net = {r['net_a']:+5d} | Winner = {r['winner']} | Time = {r['time_a']:.2f}s | Notices = {r['notices_a']}", flush=True)

v2_wins = sum(1 for r in v9_vs_v2 if r['winner'] == 'A')
v2_losses = sum(1 for r in v9_vs_v2 if r['winner'] == 'B')
v2_draws = sum(1 for r in v9_vs_v2 if r['winner'] == 'draw')
v2_tot_net = sum(r['net_a'] for r in v9_vs_v2)
print(f"--> Result vs APEX_DH_2: {v2_wins}W - {v2_losses}L - {v2_draws}D (Win Rate: {v2_wins/20*100:.1f}%), Total Net: {v2_tot_net:+d}", flush=True)

print("\n=== BATTLE GAUNTLET: APEX DISCARD v9 vs v8 Grandmaster (10 Matches) ===", flush=True)
v9_vs_v8 = []
for seed in range(1, 11):
    r = run_match('bot/bot_v9.py', 'bot/bot.py', seed)
    v9_vs_v8.append(r)
    print(f"Seed {seed:2d}: v9 net = {r['net_a']:+5d} | Winner = {r['winner']} | Time = {r['time_a']:.2f}s | Notices = {r['notices_a']}", flush=True)

v8_wins = sum(1 for r in v9_vs_v8 if r['winner'] == 'A')
v8_losses = sum(1 for r in v9_vs_v8 if r['winner'] == 'B')
v8_tot_net = sum(r['net_a'] for r in v9_vs_v8)
print(f"--> Result vs v8 Grandmaster: {v8_wins}W - {v8_losses}L (Win Rate: {v8_wins/10*100:.1f}%), Total Net: {v8_tot_net:+d}", flush=True)

print("\n=== BATTLE GAUNTLET: APEX DISCARD v9 vs Sparring Suite ===", flush=True)
opponents = [
    ('TAG', 'sparring_bots/bot_tag.py'),
    ('LAG', 'sparring_bots/bot_lag.py'),
    ('APEX_GTO', 'sparring_bots/bot_apex_gto.py'),
    ('NEMESIS', 'sparring_bots/bot_nemesis.py'),
    ('v7_GOLDEN', 'sparring_bots/bot_v7_golden.py'),
    ('v4_PREV', 'sparring_bots/bot_v4_prev.py'),
]

for name, path in opponents:
    for seed in [1, 2]:
        r = run_match('bot/bot_v9.py', path, seed)
        print(f"vs {name:<10} (Seed {seed}): v9 net = {r['net_a']:+5d} | Winner = {r['winner']} | Time = {r['time_a']:.2f}s | Notices = {r['notices_a']}", flush=True)

print("\nALL BATTLES COMPLETED SUCCESSFULLY!", flush=True)
