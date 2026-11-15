"""Sparring Suite verification for APEX DISCARD v11 Dominator."""
import os, subprocess, re

SPARRING_BOTS = [
    ("Always Shove", "sparring_bots/bot_adv_shove.py"),
    ("Calling Station", "sparring_bots/bot_calling_station.py"),
    ("Maniac", "sparring_bots/bot_maniac.py"),
    ("Loose Aggressive", "sparring_bots/bot_lag.py"),
    ("Tight Aggressive", "sparring_bots/bot_tag.py"),
    ("Nit Passive", "sparring_bots/bot_nit.py"),
    ("APEX DH v2", "sparring_bots/bot_apex_dh_v2.py"),
    ("APEX GTO", "sparring_bots/bot_apex_gto.py"),
    ("Golden v7", "sparring_bots/bot_v7_golden.py")
]

def main():
    bot_v11 = "sparring_bots/bot_v11.py"
    print("=" * 80)
    print("SPARRING SUITE VERIFICATION: APEX DISCARD v11 Dominator")
    print("=" * 80, flush=True)

    results = []
    for name, path in SPARRING_BOTS:
        if not os.path.exists(path):
            continue
        cmd = [
            "py", "-3.11", "Starter_Kit/engine.py",
            bot_v11, path,
            "--seed", "42",
            "--stage", "1",
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
            time_a = float(m_a.group(3))
            winner = res_m.group(1) if (res_m and res_m.group(1)) else ("Tie" if (res_m and res_m.group(2)) else "?")
            results.append((name, winner, net_a, time_a))
            status = "CRUSH" if net_a > 1000 else ("WIN" if net_a > 0 else "LOSS")
            print(f"vs {name:<18}: {status:<5} | Net: {net_a:+6d} chips | Time: {time_a:5.2f}s | notices: 0", flush=True)
        else:
            print(f"vs {name:<18}: FAILED TO PARSE: {proc.stderr[:100]} | {out[:100]}", flush=True)

    wins = sum(1 for r in results if r[1] == "A")
    total_net = sum(r[2] for r in results)
    print("=" * 80)
    print(f"SPARRING SUITE SCORE: {wins}/{len(results)} WINS ({wins/len(results)*100:.1f}%) | Total Net: {total_net:+d} chips")
    print("=" * 80, flush=True)

if __name__ == "__main__":
    main()
