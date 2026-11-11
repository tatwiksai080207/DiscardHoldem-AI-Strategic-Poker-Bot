import sys
import os
import argparse
import subprocess
import time
import re
from concurrent.futures import ThreadPoolExecutor, as_completed

def run_single_match(engine_path, bot_a, bot_b, hands, stage, seed, python_exe):
    cmd = [
        python_exe,
        engine_path,
        bot_a,
        bot_b,
        "--hands", str(hands),
        "--stage", str(stage),
        "--seed", str(seed)
    ]
    t0 = time.perf_counter()
    res = subprocess.run(cmd, capture_output=True, text=True)
    t1 = time.perf_counter()

    out = res.stdout
    err = res.stderr

    net_a = 0
    net_b = 0
    capped_a = 0
    time_a = 0.0
    notices_a = 0
    winner = "Draw"
    forfeit_a = "A forfeited" in out
    forfeit_b = "B forfeited" in out

    m_a = re.search(r'A\s+.*?\s+net\s+([+-]?\d+)\s+time used\s+([\d.]+)\s+s.*?\s+notices\s+(\d+)', out)
    if m_a:
        net_a = int(m_a.group(1))
        time_a = float(m_a.group(2))
        notices_a = int(m_a.group(3))

    m_b = re.search(r'B\s+.*?\s+net\s+([+-]?\d+)', out)
    if m_b:
        net_b = int(m_b.group(1))

    default_cap = 1000 if stage == 2 else 400
    m_cap = re.search(r'Capped delta:\s+A\s+([+-]?\d+),\s+B\s+([+-]?\d+)', out)
    if m_cap:
        capped_a = int(m_cap.group(1))
    else:
        m_forfeit_a = re.search(r"A's delta is set to \+([+-]?\d+)", out)
        m_forfeit_b = re.search(r"B's delta is set to \+([+-]?\d+)", out)
        if m_forfeit_a:
            capped_a = int(m_forfeit_a.group(1))
        elif m_forfeit_b:
            capped_a = -int(m_forfeit_b.group(1))
        else:
            capped_a = max(-default_cap, min(default_cap, net_a))

    # Strict Rule 8.1 scoring
    if forfeit_b and not forfeit_a:
        winner = "A"
    elif forfeit_a and not forfeit_b:
        winner = "B"
    elif forfeit_a and forfeit_b:
        winner = "Draw"
    else:
        if net_a > 0:
            winner = "A"
        elif net_a < 0:
            winner = "B"
        else:
            winner = "Draw"

    return {
        "seed": seed,
        "winner": winner,
        "net_a": net_a,
        "net_b": net_b,
        "capped_a": capped_a,
        "time_a": time_a,
        "notices_a": notices_a,
        "forfeit_a": forfeit_a,
        "forfeit_b": forfeit_b,
        "elapsed_wall": t1 - t0,
        "returncode": res.returncode,
        "stderr": err,
    }

def run_tournament(bot_a_path, opponent_paths, matches_per_opp=20, hands=400, stage=1, jobs=4, python_exe=sys.executable):
    workspace = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    engine_path = os.path.join(workspace, "Starter_Kit", "engine.py")

    print(f"\n================================================================================")
    print(f"POKERBOTS 2026 DUPLICATE DEAL ARENA")
    print(f"Hero Bot: {os.path.basename(bot_a_path)}")
    print(f"Opponents: {len(opponent_paths)} | Matches/Opp: {matches_per_opp} | Hands/Match: {hands} | Stage: {stage}")
    print(f"================================================================================\n")

    overall_mp = 0.0
    overall_matches = 0
    overall_net = 0
    overall_capped = 0
    overall_hands = 0
    overall_forfeits = 0

    for opp_path in opponent_paths:
        opp_name = os.path.basename(opp_path)
        print(f"--- Sparring vs {opp_name} ({matches_per_opp} matches) ---")

        tasks = []
        with ThreadPoolExecutor(max_workers=jobs) as executor:
            for m in range(matches_per_opp):
                seed = 100000 + m * 7919
                tasks.append(executor.submit(
                    run_single_match, engine_path, bot_a_path, opp_path, hands, stage, seed, python_exe
                ))

        results = [t.result() for t in tasks]

        wins = sum(1 for r in results if r["winner"] == "A")
        draws = sum(1 for r in results if r["winner"] == "Draw")
        losses = sum(1 for r in results if r["winner"] == "B")
        tot_net = sum(r["net_a"] for r in results)
        tot_cap = sum(r["capped_a"] for r in results)
        tot_hands = matches_per_opp * hands
        max_time = max(r["time_a"] for r in results)
        tot_notices = sum(r["notices_a"] for r in results)
        opp_forfeits = sum(1 for r in results if r["forfeit_b"])
        hero_forfeits = sum(1 for r in results if r["forfeit_a"])

        mp = (wins + 0.5 * draws) / matches_per_opp
        bb100 = (tot_net / float(tot_hands * 2)) * 100.0
        score_cap = tot_cap / float(tot_hands)

        overall_mp += (wins + 0.5 * draws)
        overall_matches += matches_per_opp
        overall_net += tot_net
        overall_capped += tot_cap
        overall_hands += tot_hands
        overall_forfeits += (hero_forfeits + opp_forfeits)

        avg_wall = sum(r["elapsed_wall"] for r in results) / max(1, matches_per_opp)
        max_wall = max(r["elapsed_wall"] for r in results)

        print(f"  Result: {wins}W - {losses}L - {draws}D  (MP: {mp*100:.1f}%)")
        print(f"  Net Chips: {tot_net:+d} ({bb100:+.1f} BB/100) | Capped Delta: {tot_cap:+d} ({score_cap:+.3f} chips/hand)")
        print(f"  Wall Duration: avg {avg_wall:.2f}s (max {max_wall:.2f}s) | Peak Engine Time: {max_time:.2f}s | Notices: {tot_notices} | Forfeits: Hero={hero_forfeits}, Opp={opp_forfeits}\n")

    tot_mp_pct = (overall_mp / max(1, overall_matches)) * 100.0
    tot_bb100 = (overall_net / float(max(1, overall_hands * 2))) * 100.0
    tot_score_cap = overall_capped / float(max(1, overall_hands))

    print(f"================================================================================")
    print(f"OVERALL ARENA SUMMARY across {overall_matches} matches ({overall_hands} duplicate hands):")
    print(f"  Match Point Win Rate: {tot_mp_pct:.2f}%")
    print(f"  Uncapped Win Rate:    {tot_bb100:+.2f} BB/100 ({overall_net:+d} chips)")
    print(f"  Capped Tie Breaker:   {tot_score_cap:+.4f} chips/hand ({overall_capped:+d} capped chips)")
    print(f"  Total Forfeits:       {overall_forfeits}")
    print(f"================================================================================\n")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--bot", default="Starter_Kit/bot.py", help="Hero bot path")
    parser.add_argument("--matches", type=int, default=10, help="Matches per opponent")
    parser.add_argument("--hands", type=int, default=400, help="Hands per match")
    parser.add_argument("--stage", type=int, default=1, help="Tournament stage (1 or 2)")
    parser.add_argument("--jobs", type=int, default=4, help="Parallel worker threads")
    parser.add_argument("--python", default=sys.executable, help="Python executable")
    parser.add_argument("--opp", nargs="*", default=None, help="Filter opponent name substrings")
    args = parser.parse_args()

    workspace = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    sparring_dir = os.path.join(workspace, "sparring_bots")
    all_opponents = [
        os.path.join(sparring_dir, f) for f in sorted(os.listdir(sparring_dir)) if f.endswith(".py") and f != "bot_v5_golden.py"
    ]
    all_opponents.append(os.path.join(workspace, "Starter_Kit", "example_bots", "starter_bot.py"))
    all_opponents.append(os.path.join(workspace, "Starter_Kit", "example_bots", "random_bot.py"))

    if args.opp:
        filtered = []
        for pat in args.opp:
            for opp in all_opponents:
                if pat.lower() in os.path.basename(opp).lower() and opp not in filtered:
                    filtered.append(opp)
        opponents = filtered
    else:
        opponents = all_opponents

    run_tournament(
        args.bot, opponents,
        matches_per_opp=args.matches,
        hands=args.hands,
        stage=args.stage,
        jobs=args.jobs,
        python_exe=args.python
    )
