import sys
import os
import time
import lzma
import base64
import numpy as np

workspace = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(workspace, "bot"))

from bot import (
    CANON_REPRESENTATIVE, evaluate_3way_flop_discard, RNG
)

def build_1755_preflop_table(n_rollouts_per_class=1200):
    t0 = time.perf_counter()
    eq_matrix = np.zeros((1755, 4), dtype=np.float32)

    n_flops = 30
    rollouts_per_flop = max(20, n_rollouts_per_class // n_flops)

    print(f"Starting 1,755 preflop table generation ({n_flops} flops x {rollouts_per_flop} rollouts per class)...")

    for cid in range(1755):
        hero3 = CANON_REPRESENTATIVE[cid]
        rem49 = np.setdiff1d(np.arange(52, dtype=np.int32), hero3)

        eq_rand_sum = 0.0
        eq_top50_sum = 0.0
        eq_top20_sum = 0.0
        nut_count = 0

        for _ in range(n_flops):
            flop3 = RNG.choice(rem49, size=3, replace=False)
            _, eq_r, _ = evaluate_3way_flop_discard(hero3, flop3, n_rollouts=rollouts_per_flop, opp_aggr=0.0)
            _, eq_50, _ = evaluate_3way_flop_discard(hero3, flop3, n_rollouts=rollouts_per_flop, opp_aggr=0.55)
            _, eq_20, _ = evaluate_3way_flop_discard(hero3, flop3, n_rollouts=rollouts_per_flop, opp_aggr=1.10)
            eq_rand_sum += eq_r
            eq_top50_sum += eq_50
            eq_top20_sum += eq_20
            if eq_r >= 0.72:
                nut_count += 1

        eq_matrix[cid, 0] = eq_rand_sum / n_flops
        eq_matrix[cid, 1] = eq_top50_sum / n_flops
        eq_matrix[cid, 2] = eq_top20_sum / n_flops
        eq_matrix[cid, 3] = nut_count / float(n_flops)

        if (cid + 1) % 250 == 0 or (cid + 1) == 1755:
            elapsed = time.perf_counter() - t0
            rate = (cid + 1) / elapsed
            eta = (1755 - (cid + 1)) / max(0.01, rate)
            print(f"[{cid+1:4d}/1755] elapsed={elapsed:5.1f}s | rate={rate:4.1f} classes/s | ETA={eta:5.1f}s")

    quantized = eq_matrix.astype(np.float16).tobytes()
    compressed = lzma.compress(quantized, preset=9)
    b64_str = base64.b64encode(compressed).decode("ascii")

    total_time = time.perf_counter() - t0
    print(f"\nCompleted in {total_time:.2f}s!")
    print(f"Raw array size: {len(quantized)} bytes ({quantized.nbytes if hasattr(quantized, 'nbytes') else len(quantized)})")
    print(f"Compressed size: {len(compressed)} bytes")
    print(f"Base64 length: {len(b64_str)} chars")

    output_path = os.path.join(workspace, "dist", "preflop_1755_b64.txt")
    with open(output_path, "w") as f:
        f.write(b64_str)
    print(f"Saved Base64 table to {output_path}")

    return b64_str

if __name__ == "__main__":
    build_1755_preflop_table(n_rollouts_per_class=1200)

