import subprocess

seeds = [11, 14, 16, 19, 20]
for s in seeds:
    cmd = ['py', '-3.11', 'Starter_Kit/engine.py', 'bot/bot.py', 'sparring_bots/bot_apex_dh_v2.py', '--hands', '400', '--stage', '1', '--seed', str(s)]
    res = subprocess.run(cmd, capture_output=True, text=True)
    out = res.stdout
    net_line = [l for l in out.splitlines() if 'bot\\bot.py' in l]
    result_line = [l for l in out.splitlines() if 'Result:' in l]
    nl = net_line[0].strip() if net_line else ""
    rl = result_line[0].strip() if result_line else ""
    print(f"Seed {s:2}: {nl} | {rl}")
