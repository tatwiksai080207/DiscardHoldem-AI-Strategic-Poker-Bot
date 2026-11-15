import subprocess, sys, re, json

PY = ["py", "-3.11"]

def run(bot_a, bot_b, seed, hands=400, stage=1):
    cmd = PY + ["Starter_Kit/engine.py", bot_a, bot_b, "--hands", str(hands), "--stage", str(stage), "--seed", str(seed)]
    p = subprocess.run(cmd, capture_output=True, text=True, timeout=900)
    return p.stdout + "\n" + p.stderr

def parse_net(text):
    m = re.search(r'^\s*A\s+.*?\s+net\s+([+-]?\d+)', text, re.MULTILINE)
    if m:
        return int(m.group(1))
    raise ValueError(f"Could not parse net chips from engine output:\n{text}")

if __name__ == "__main__":
    # usage: py -3.11 tools/compare_versions.py <botA_path> <botB_path> [more opponents ...]  (writes JSON lines)
    bot_a = sys.argv[1]
    seeds = [111111, 222222]
    for opp in sys.argv[2:]:
        for seed in seeds:
            txt = run(bot_a, opp, seed)
            net_val = parse_net(txt)
            print(json.dumps({"bot": bot_a, "opp": opp, "seed": seed, "net": net_val}), flush=True)
