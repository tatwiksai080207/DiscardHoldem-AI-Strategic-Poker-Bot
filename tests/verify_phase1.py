import sys
import os
import time

# Ensure imports work from bot and Starter_Kit
workspace_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(workspace_root, "bot"))
sys.path.insert(0, os.path.join(workspace_root, "Starter_Kit"))

t0 = time.perf_counter()
import bot
t1 = time.perf_counter()
startup_time = t1 - t0
print(f"1. Startup & table build time: {startup_time:.4f}s")
assert startup_time < 2.0, f"Startup took too long: {startup_time:.4f}s"

import engine
import numpy as np

# 2. Verify Evaluator Parity with engine.best_hand across 10,000 comparisons
print("2. Verifying Split-Radix Evaluator vs engine.best_hand across 10,000 comparisons...")
rng = np.random.default_rng(20261106)
mismatches = 0
n_tests = 10_000

for i in range(n_tests):
    h1_idx = rng.choice(52, 7, replace=False)
    h2_idx = rng.choice(52, 7, replace=False)

    h1_cards = [bot.INT_TO_CARD[c] for c in h1_idx]
    h2_cards = [bot.INT_TO_CARD[c] for c in h2_idx]

    eng_h1 = engine.best_hand(h1_cards[:2], h1_cards[2:])
    eng_h2 = engine.best_hand(h2_cards[:2], h2_cards[2:])
    eng_cmp = 1 if eng_h1 > eng_h2 else (-1 if eng_h1 < eng_h2 else 0)

    c1 = bot.CARD_CODE[h1_idx].sum()
    c2 = bot.CARD_CODE[h2_idx].sum()
    s1 = bot.CARD_SUIT_BIT[:, h1_idx].sum(axis=1)[:, None]
    s2 = bot.CARD_SUIT_BIT[:, h2_idx].sum(axis=1)[:, None]

    sc1 = bot.eval_batch_codes(np.array([c1], dtype=np.int64), s1)[0]
    sc2 = bot.eval_batch_codes(np.array([c2], dtype=np.int64), s2)[0]
    bot_cmp = 1 if sc1 > sc2 else (-1 if sc1 < sc2 else 0)

    if eng_cmp != bot_cmp:
        mismatches += 1
        print(f"Mismatch at test {i}: eng={eng_cmp}, bot={bot_cmp}")
        print(f"Hand 1: {h1_cards} -> {eng_h1}, score={sc1:#x}")
        print(f"Hand 2: {h2_cards} -> {eng_h2}, score={sc2:#x}")
        break

print(f"Completed {n_tests} comparisons: {mismatches} mismatches.")
assert mismatches == 0, f"Found {mismatches} evaluator mismatches!"

# 3. Verify Preflop Isomorphism: all C(52, 3) = 22,100 hands map to [0, 1754]
print("3. Verifying all 22,100 3-card preflop hands map onto [0, 1754]...")
all_cids = set()
total_combos = 0
for c0 in range(52):
    for c1 in range(c0 + 1, 52):
        for c2 in range(c1 + 1, 52):
            total_combos += 1
            cid = int(bot.PREFLOP_CANON_ID[c0, c1, c2])
            all_cids.add(cid)
            # Permutation symmetry check
            assert bot.PREFLOP_CANON_ID[c0, c2, c1] == cid
            assert bot.PREFLOP_CANON_ID[c1, c0, c2] == cid
            assert bot.PREFLOP_CANON_ID[c2, c1, c0] == cid

print(f"Checked {total_combos} combinations. Distinct canonical classes: {len(all_cids)}")
assert total_combos == 22100, f"Expected 22,100 combos, got {total_combos}"
assert len(all_cids) == 1755, f"Expected 1,755 canonical classes, got {len(all_cids)}"
assert min(all_cids) == 0 and max(all_cids) == 1754, "Canonical IDs out of range"

print("\n--- ALL PHASE 1 CORE UNIT TESTS PASSED SUCCESSFULLY! ---")

