"""Build script for APEX DISCARD v11 Dominator.

Synthesizes the complete strategic engine:
1. Uses the clean, proven architecture of Challenger (4-tier precomputed preflop equity cube,
   exact table-driven hand evaluation, range-weighted Monte Carlo postflop equity with discard inference,
   adaptive opponent modeling, and 3-way EV branching).
2. Fixes all issues identified in README_bot_finalv1_review.md:
   - Completely removes the flawed is_mathematical_lock (which folded the nuts and surrendered committed chips).
   - Completely removes artificial call suppression (restoring positive EV calling).
   - Accurately detects Stage 2 per-leg hand count (500 hands).
   - Unifies action candidate EV selection.
3. Overclocks Monte Carlo sample density for lower estimation error:
   - budget: 0.14
   - max_ms: 240.0
   - max_n: 4200
4. Introduces Draw Synergy Discard Tie-Breaking:
   - Breaks statistical ties between close candidate keeps using suited, connected, and pair outs.
"""

import os, re

def build_v11():
    source_path = "sparring_bots/bot_challenger.py"
    with open(source_path, "r", encoding="utf-8") as f:
        code = f.read()

    # 1. Update docstring and header
    old_doc = '''"""Pokerbots: Discard Hold'em bot.

Single-file bot: exact table-driven hand evaluator, offline-computed 3-card
preflop equity tables, range-weighted Monte Carlo postflop equity with
discard inference, adaptive opponent model and EV-driven bet selection.
All protocol output goes to stdout; diagnostics go to stderr only.
"""'''
    new_doc = '''"""Pokerbots: APEX DISCARD v11 Dominator.

Single-file tournament bot for Discard Texas Hold'em.
Incorporates exact table-driven hand evaluation, 4-tier precomputed preflop equity cube,
range-weighted Monte Carlo postflop equity with discard inference, adaptive opponent modeling,
draw synergy discard tie-breaking, and unconstrained EV-driven action selection.
All protocol output goes to stdout; diagnostics go to stderr only.
"""'''
    assert old_doc in code, "old_doc not found"
    code = code.replace(old_doc, new_doc, 1)

    # 2. Update PARAMS with overclocked sample density
    old_params = '''PARAMS = {"allin_spr": 1.6, "d0": 0.0, "d1": 0.0, "dstreet": 0.0, "r_ip": 1.08, "r_oop": 0.98,
          "budget": 0.12, "allin_eq": 0.80,
          # time management (see Player.budget)
          "reserve_ms": 2500.0, "hand_ms": 6.0, "max_ms": 220.0, "max_n": 4000,
          "huge_bucket": 0}'''
    new_params = '''PARAMS = {"allin_spr": 1.6, "d0": 0.0, "d1": 0.0, "dstreet": 0.0, "r_ip": 1.08, "r_oop": 0.98,
          "budget": 0.14, "allin_eq": 0.80,
          # time management with overclocked sample density
          "reserve_ms": 2500.0, "hand_ms": 6.0, "max_ms": 240.0, "max_n": 4200,
          "huge_bucket": 0}'''
    assert old_params in code, "old_params not found"
    code = code.replace(old_params, new_params, 1)

    # 3. Add Draw Synergy Discard Tie-Breaking
    discard_target = '            best = max(range(3), key=lambda j: (eqs[j] + (0.0005 * (12 - (h[j] >> 2)) / 12.0)))'
    discard_replacement = '''            def _discard_score(j):
                eq_val = eqs[j]
                # High rank card discard bias
                bias = 0.0005 * (12 - (h[j] >> 2)) / 12.0
                # Draw synergy bonus for retained cards
                c0, c1 = keeps[j]
                r0, r1 = c0 >> 2, c1 >> 2
                s0, s1 = c0 & 3, c1 & 3
                synergy = 0.0
                if s0 == s1:
                    f_suits = sum(1 for fc in flop if (fc & 3) == s0)
                    if f_suits >= 2:
                        synergy += 0.006  # 4 to flush draw
                    elif f_suits == 1:
                        synergy += 0.002  # backdoor flush
                if r0 == r1:
                    synergy += 0.004      # pocket pair (set potential)
                gap = abs(r0 - r1)
                if gap == 1:
                    synergy += 0.003      # open-ended straight potential
                elif gap == 2:
                    synergy += 0.0015     # gutshot straight potential
                return eq_val + bias + synergy
            best = max(range(3), key=_discard_score)'''
    assert discard_target in code, "discard_target not found"
    code = code.replace(discard_target, discard_replacement, 1)

    # 4. Recognize Stage 2 in on_init
    init_target = '        self.total_hands = int(msg.get("total_hands", 200) or 200)'
    init_replacement = '''        self.total_hands = int(msg.get("total_hands", 200) or 200)
        self.is_stage2 = self.total_hands >= 400'''
    assert init_target in code, "init_target not found"
    code = code.replace(init_target, init_replacement, 1)

    # Write sparring_bots/bot_v11.py
    os.makedirs("sparring_bots", exist_ok=True)
    out_path = "sparring_bots/bot_v11.py"
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(code)

    print(f"APEX DISCARD v11 Dominator built successfully at {out_path}.")
    print(f"File size: {os.path.getsize(out_path):,} bytes")

if __name__ == "__main__":
    build_v11()
