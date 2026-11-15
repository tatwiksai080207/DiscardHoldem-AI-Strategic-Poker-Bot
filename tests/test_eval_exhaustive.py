import os, sys, itertools, collections
import numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "bot"))
import bot

def check(cond, msg):
    if not cond:
        print("FAIL:", msg)
        sys.exit(1)

def score_batch(cards):
    cards = np.asarray(cards, dtype=np.int64)
    codes = bot.CARD_CODE[cards].sum(axis=1)
    sb = np.zeros((4, cards.shape[0]), dtype=np.uint16)
    for k in range(cards.shape[1]):
        sb += bot.CARD_SUIT_BIT[:, cards[:, k]]
    return bot.eval_batch_codes(codes, sb).astype(np.int64)

# ---------- A. all 2,598,960 five-card hands ----------
n5 = 2598960
all5 = np.fromiter(itertools.chain.from_iterable(itertools.combinations(range(52), 5)),
                   dtype=np.int8, count=n5 * 5).reshape(-1, 5)
scores = np.empty(n5, dtype=np.int64)
for i in range(0, n5, 500_000):
    scores[i:i + 500_000] = score_batch(all5[i:i + 500_000])

check(len(np.unique(scores)) == 7462, f"expected 7462 distinct hand classes, got {len(np.unique(scores))}")
expected = {0: 1302540, 1: 1098240, 2: 123552, 3: 54912, 4: 10200, 5: 5108, 6: 3744, 7: 624, 8: 40}
cats = scores >> 20
got = {c: int((cats == c).sum()) for c in range(9)}
check(got == expected, f"category counts wrong: {got}")
print("A. 5-card exhaustive: 7462 classes and all category counts correct")

# ---------- B. ordering vs an independent pure-python reference ----------
def ref5(cs):
    ranks = sorted((c // 4 for c in cs), reverse=True)
    suits = [c % 4 for c in cs]
    cnt = collections.Counter(ranks)
    groups = sorted(cnt.items(), key=lambda kv: (kv[1], kv[0]), reverse=True)
    shape = [g[1] for g in groups]
    kick = tuple(g[0] for g in groups)
    flush = len(set(suits)) == 1
    uniq = sorted(set(ranks), reverse=True)
    sh = None
    if len(uniq) == 5:
        if uniq[0] - uniq[4] == 4:
            sh = uniq[0]
        elif uniq == [12, 3, 2, 1, 0]:
            sh = 3
    if flush and sh is not None: return (8, sh)
    if shape == [4, 1]: return (7,) + kick
    if shape == [3, 2]: return (6,) + kick
    if flush: return (5,) + tuple(ranks)
    if sh is not None: return (4, sh)
    if shape == [3, 1, 1]: return (3,) + kick
    if shape == [2, 2, 1]: return (2,) + kick
    if shape == [2, 1, 1, 1]: return (1,) + kick
    return (0,) + tuple(ranks)

rng = np.random.default_rng(777)
ia = rng.integers(0, n5, 300_000)
ib = rng.integers(0, n5, 300_000)
bad = 0
for a, b in zip(ia.tolist(), ib.tolist()):
    ra, rb = ref5(all5[a].tolist()), ref5(all5[b].tolist())
    sa, sb_ = int(scores[a]), int(scores[b])
    if (ra > rb) != (sa > sb_) or (ra == rb) != (sa == sb_):
        bad += 1
check(bad == 0, f"{bad} ordering disagreements vs reference evaluator")
print("B. ordering matches reference on 300k random pairs")

# ---------- C. 7-card score == best of its 21 five-card subsets ----------
def best_of_21(hands):
    best = np.zeros(len(hands), dtype=np.int64)
    for idx in itertools.combinations(range(7), 5):
        best = np.maximum(best, score_batch(hands[:, list(idx)]))
    return best

N = 200_000
hands = np.argpartition(rng.random((N, 52)), 7, axis=1)[:, :7].astype(np.int64)
check((score_batch(hands) == best_of_21(hands)).all(), "7-card mismatch on uniformly random hands")
print("C1. 7-card vs best-of-21 on 200k random hands OK")

def biased_hands(rng, n):
    out = []
    for _ in range(n):
        mode = int(rng.integers(0, 4))
        if mode == 0:            # flush / straight-flush heavy
            s = int(rng.integers(0, 4)); k = int(rng.integers(4, 8))
            a = [int(x) for x in rng.choice([r * 4 + s for r in range(13)], size=k, replace=False)]
        elif mode == 1:          # trips / quads heavy
            r = int(rng.integers(0, 13)); k = int(rng.integers(3, 5))
            a = [int(x) for x in rng.choice([r * 4 + i for i in range(4)], size=k, replace=False)]
        elif mode == 2:          # straight heavy, includes the wheel (start -1 => A,2,3,4,5)
            st = int(rng.integers(-1, 9))
            a = [((st + i) % 13) * 4 + int(rng.integers(0, 4)) for i in range(5)]
        else:                    # two-pair / full-house heavy
            r1, r2 = [int(x) for x in rng.choice(13, size=2, replace=False)]
            a = [int(x) for x in rng.choice([r1 * 4 + i for i in range(4)], size=int(rng.integers(2, 4)), replace=False)]
            a += [int(x) for x in rng.choice([r2 * 4 + i for i in range(4)], size=2, replace=False)]
        a = list(dict.fromkeys(a))
        rest = [c for c in range(52) if c not in a]
        b = [int(x) for x in rng.choice(rest, size=7 - len(a), replace=False)] if len(a) < 7 else []
        out.append((a + b)[:7])
    return np.array(out, dtype=np.int64)

bh = biased_hands(rng, 60_000)
check((score_batch(bh) == best_of_21(bh)).all(), "7-card mismatch on biased rare-category hands")
print("C2. 7-card vs best-of-21 on 60k rare-category hands OK")
print("test_eval_exhaustive: ALL PASS")
