"""Head to head match runner with live progress and capped delta metrics.
Usage:
    py -3.11 scripts/run_head_to_head.py <bot_a> <bot_b> <seeds_count> [start_seed]
"""

import sys, os, subprocess, re

def run_matches(bot_a, bot_b, seeds_count=5, start_seed=1):
    engine_path = "Starter_Kit/engine.py"
    print(f"Starting Head-to-Head Benchmark:")
    print(f"  Bot A: {bot_a}")
    print(f"  Bot B: {bot_b}")
    print(f"  Seeds: {start_seed} to {start_seed + seeds_count - 1} ({seeds_count} matches, {seeds_count * 400} hands)")
    print("=" * 70, flush=True)

    results = []
    for s_idx in range(seeds_count):
        seed = start_seed + s_idx
        cmd = [
            "py", "-3.11", engine_path,
            bot_a, bot_b,
            "--seed", str(seed),
            "--python", "py",
            "--log-dir", ""  # disable file logging to avoid disk contention
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
            results.append((seed, winner, net_a, net_b, time_a, time_b))
            print(f"Seed {seed:3d}: Winner {winner} | A: {net_a:+6d} ({time_a:5.2f}s) | B: {net_b:+6d} ({time_b:5.2f}s)", flush=True)
        else:
            print(f"Seed {seed:3d}: Failed to parse match output. stderr: {proc.stderr[:200]}", flush=True)

    if not results:
        print("No match completed successfully.")
        return

    wins_a = sum(1 for r in results if r[1] == "A")
    wins_b = sum(1 for r in results if r[1] == "B")
    ties = sum(1 for r in results if r[1] == "Tie")
    total_net_a = sum(r[2] for r in results)
    total_net_b = sum(r[3] for r in results)
    capped_delta_a = sum(max(-400, min(400, r[2])) for r in results)
    capped_delta_b = sum(max(-400, min(400, r[3])) for r in results)
    bb_per_100_a = (total_net_a / (len(results) * 400)) * 100 / 2.0  # Big blind = 2 chips

    print("=" * 70)
    print(f"FINAL TOURNAMENT SUMMARY ({len(results)} matches, {len(results)*400} hands):")
    print(f"  Bot A ({os.path.basename(bot_a)}):")
    print(f"    Match Win Rate: {wins_a}/{len(results)} ({wins_a/len(results)*100:.1f}%) | Ties: {ties}")
    print(f"    Total Net Chips: {total_net_a:+d} ({bb_per_100_a:+.2f} BB/100)")
    print(f"    Total Capped Delta: {capped_delta_a:+d}")
    print(f"  Bot B ({os.path.basename(bot_b)}):")
    print(f"    Match Win Rate: {wins_b}/{len(results)} ({wins_b/len(results)*100:.1f}%) | Ties: {ties}")
    print(f"    Total Net Chips: {total_net_b:+d} ({-bb_per_100_a:+.2f} BB/100)")
    print(f"    Total Capped Delta: {capped_delta_b:+d}")
    print("=" * 70, flush=True)

if __name__ == "__main__":
    b_a = sys.argv[1] if len(sys.argv) > 1 else "sparring_bots/bot_v10_ultimate.py"
    b_b = sys.argv[2] if len(sys.argv) > 2 else "sparring_bots/bot_challenger.py"
    n_seeds = int(sys.argv[3]) if len(sys.argv) > 3 else 5
    s_seed = int(sys.argv[4]) if len(sys.argv) > 4 else 1
    run_matches(b_a, b_b, n_seeds, s_seed)
