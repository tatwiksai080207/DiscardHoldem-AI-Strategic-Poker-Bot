"""Tournament benchmark suite for APEX DISCARD v11:
1. v11 Dominator vs v10 Ultimate (10 matches, 4,000 hands)
2. v11 Dominator vs Challenger (10 matches, 4,000 hands)
"""

import sys, os, subprocess, re, time

def run_match(bot_a, bot_b, seed, stage=1):
    engine_path = "Starter_Kit/engine.py"
    cmd = [
        "py", "-3.11", engine_path,
        bot_a, bot_b,
        "--seed", str(seed),
        "--stage", str(stage),
        "--python", "py",
        "--log-dir", ""
    ]
    proc = subprocess.run(cmd, capture_output=True, text=True)
    out = proc.stdout
    m_a = re.search(r"A\s+([^\n]+?)\s+net\s+([+-]\d+)\s+time used\s+([\d\.]+)\s*s", out)
    m_b = re.search(r"B\s+([^\n]+?)\s+net\s+([+-]\d+)\s+time used\s+([\d\.]+)\s*s", out)
    res_m = re.search(r"Result:\s+([A-B])\s+wins|Result:\s+(tie)", out)

    if m_a and m_b:
        net_a = int(m_a.group(2))
        net_b = int(m_b.group(2))
        time_a = float(m_a.group(3))
        time_b = float(m_b.group(3))
        winner = res_m.group(1) if (res_m and res_m.group(1)) else ("Tie" if (res_m and res_m.group(2)) else "?")
        return {"seed": seed, "winner": winner, "net_a": net_a, "net_b": net_b, "time_a": time_a, "time_b": time_b}
    else:
        print(f"Error on seed {seed}: {proc.stderr[:300]}", flush=True)
        return None

def run_series(name, bot_a, bot_b, seeds):
    print("=" * 80)
    print(f"SERIES: {name}")
    print(f"  Bot A: {bot_a}")
    print(f"  Bot B: {bot_b}")
    print(f"  Seeds: {seeds}")
    print("=" * 80, flush=True)

    results = []
    for s in seeds:
        t0 = time.time()
        res = run_match(bot_a, bot_b, s)
        elapsed = time.time() - t0
        if res:
            results.append(res)
            cap_a = max(-400, min(400, res["net_a"]))
            print(f"Seed {s:3d}: Winner {res['winner']} | A: {res['net_a']:+5d} (Cap {cap_a:+4d}, {res['time_a']:5.2f}s) | B: {res['net_b']:+5d} ({res['time_b']:5.2f}s) [{elapsed:4.1f}s]", flush=True)
        else:
            print(f"Seed {s:3d}: FAILED", flush=True)

    w_a = sum(1 for r in results if r["winner"] == "A")
    w_b = sum(1 for r in results if r["winner"] == "B")
    ties = sum(1 for r in results if r["winner"] == "Tie")
    total_net = sum(r["net_a"] for r in results)
    total_hands = len(results) * 400
    bb_100 = (total_net / total_hands) * 100 / 2.0
    total_cap = sum(max(-400, min(400, r["net_a"])) for r in results)

    print("-" * 80)
    print(f"SERIES SUMMARY: {name}")
    print(f"  Bot A Match Score: {w_a}W - {w_b}L - {ties}T ({w_a / len(results) * 100:.1f}%)")
    print(f"  Bot A Total Net:   {total_net:+d} chips ({bb_100:+.2f} BB/100)")
    print(f"  Bot A Capped Delta:{total_cap:+d}")
    print("=" * 80, flush=True)
    return results

def main():
    bot_v11 = "sparring_bots/bot_v11.py"
    bot_v10 = "bot/bot.py"
    bot_chal = "sparring_bots/bot_challenger.py"

    seeds = list(range(1, 11))

    # Benchmark 1: v11 vs v10
    run_series("APEX DISCARD v11 Dominator vs APEX DISCARD v10 Ultimate", bot_v11, bot_v10, seeds)

    # Benchmark 2: v11 vs Challenger
    run_series("APEX DISCARD v11 Dominator vs Challenger", bot_v11, bot_chal, seeds)

if __name__ == "__main__":
    main()
