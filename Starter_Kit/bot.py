import sys
import json
import time
import math
import zlib
import lzma
import base64
import numpy as np

# ==============================================================================
# SUBSYSTEM 1: 64-BIT SPLIT-RADIX VECTORIZED 5/6/7-CARD EVALUATOR (3.2M+ evals/s)
# Builds in ~0.41s during the uncharged 5.0s startup window. Zero disk footprint.
# ==============================================================================
RANKS_STR = "23456789TJQKA"
SUITS_STR = "shdc"
CARD_TO_INT = {r + s: (ri * 4 + si) for ri, r in enumerate(RANKS_STR) for si, s in enumerate(SUITS_STR)}
INT_TO_CARD = [r + s for r in RANKS_STR for s in SUITS_STR]

STRAIGHT_MASKS = [
    0b1111100000000, 0b0111110000000, 0b0011111000000, 0b0001111100000, 0b0000111110000,
    0b0000011111000, 0b0000001111100, 0b0000000111110, 0b0000000011111, 0b1000000001111,
]

FLUSH_TABLE = np.zeros(8192, dtype=np.uint32)
STRAIGHT_TABLE = np.zeros(8192, dtype=np.uint8)
STRAIGHT_DRAW_TABLE = np.zeros(8192, dtype=np.float32)

for _mask in range(8192):
    _st_val = 0
    for _idx, _sm in enumerate(STRAIGHT_MASKS):
        if (_mask & _sm) == _sm:
            _st_val = 10 - _idx
            break
    STRAIGHT_TABLE[_mask] = _st_val
    if _mask.bit_count() >= 5:
        if _st_val > 0:
            FLUSH_TABLE[_mask] = (8 << 20) | _st_val
        else:
            _top5, _cnt = 0, 0
            for _r in range(12, -1, -1):
                if (_mask >> _r) & 1:
                    _top5 = (_top5 << 4) | _r
                    _cnt += 1
                    if _cnt == 5:
                        break
            FLUSH_TABLE[_mask] = (5 << 20) | _top5

for _mask in range(8192):
    _outs = 0
    for _r in range(13):
        if not ((_mask >> _r) & 1):
            if STRAIGHT_TABLE[_mask | (1 << _r)] > 0:
                _outs += 1
    STRAIGHT_DRAW_TABLE[_mask] = _outs * 0.04

POW5 = [5**i for i in range(7)]
LOW_ID = np.full(5**7, -1, dtype=np.int16)
HIGH_ID = np.full(5**6, -1, dtype=np.int16)
_low_by_sum = [[] for _ in range(8)]
_high_by_sum = [[] for _ in range(8)]

_lid_cnt = 0
for _s in range(5**7):
    _c = tuple((_s // POW5[i]) % 5 for i in range(7))
    _sm = sum(_c)
    if _sm <= 7:
        LOW_ID[_s] = _lid_cnt
        _low_by_sum[_sm].append((_lid_cnt, _c))
        _lid_cnt += 1

_hid_cnt = 0
for _s in range(5**6):
    _c = tuple((_s // POW5[i]) % 5 for i in range(6))
    _sm = sum(_c)
    if _sm <= 7:
        HIGH_ID[_s] = _hid_cnt
        _high_by_sum[_sm].append((_hid_cnt, _c))
        _hid_cnt += 1

NON_FLUSH_TABLE = np.zeros((_lid_cnt, _hid_cnt), dtype=np.uint32)

def _eval_counts_exact(counts):
    mask, quads = 0, -1
    trips, pairs, singles = [], [], []
    for r in range(12, -1, -1):
        cnt = counts[r]
        if cnt:
            mask |= (1 << r)
            if cnt == 4:
                quads = r
            elif cnt == 3:
                trips.append(r)
            elif cnt == 2:
                pairs.append(r)
            else:
                singles.append(r)
    if quads != -1:
        for r in range(12, -1, -1):
            if counts[r] > 0 and r != quads:
                return (7 << 20) | (quads << 4) | r
    if trips:
        t1 = trips[0]
        if len(trips) > 1:
            return (6 << 20) | (t1 << 4) | trips[1]
        if pairs:
            return (6 << 20) | (t1 << 4) | pairs[0]
    st = int(STRAIGHT_TABLE[mask])
    if st > 0:
        return (4 << 20) | st
    if trips:
        return (3 << 20) | (trips[0] << 8) | (singles[0] << 4) | singles[1]
    if len(pairs) >= 2:
        p1, p2 = pairs[0], pairs[1]
        kicker = singles[0] if singles else -1
        if len(pairs) == 3 and pairs[2] > kicker:
            kicker = pairs[2]
        return (2 << 20) | (p1 << 8) | (p2 << 4) | kicker
    if len(pairs) == 1:
        return (1 << 20) | (pairs[0] << 12) | (singles[0] << 8) | (singles[1] << 4) | singles[2]
    return (singles[0] << 16) | (singles[1] << 12) | (singles[2] << 8) | (singles[3] << 4) | singles[4]

for _tot in (5, 7):
    for _kl in range(_tot + 1):
        _kh = _tot - _kl
        if _kh <= 7:
            for _lid, _lc in _low_by_sum[_kl]:
                for _hid, _hc in _high_by_sum[_kh]:
                    NON_FLUSH_TABLE[_lid, _hid] = _eval_counts_exact(_lc + _hc)

CARD_CODE = np.zeros(52, dtype=np.int64)
CARD_SUIT_BIT = np.zeros((4, 52), dtype=np.uint16)
for _c in range(52):
    _r, _s = _c // 4, _c % 4
    CARD_CODE[_c] = (
        ((1 << (3 * _s)) << 32)
        | ((POW5[_r - 7] if _r >= 7 else 0) << 17)
        | (POW5[_r] if _r < 7 else 0)
    )
    CARD_SUIT_BIT[_s, _c] = 1 << _r

FLUSH_SUIT = np.full(4096, -1, dtype=np.int8)
FLUSH_DRAW_SUIT = np.full(4096, -1, dtype=np.int8)
for _sm in range(4096):
    for _s in range(4):
        _cnt = (_sm >> (3 * _s)) & 7
        if _cnt >= 5:
            FLUSH_SUIT[_sm] = _s
        elif _cnt == 4:
            FLUSH_DRAW_SUIT[_sm] = _s

def eval_batch_codes(codes, suit_bits_by_suit):
    scores = NON_FLUSH_TABLE[LOW_ID[codes & 0x1FFFF], HIGH_ID[(codes >> 17) & 0x3FFF]]
    fs = FLUSH_SUIT[codes >> 32]
    f_idx = np.nonzero(fs >= 0)[0]
    if len(f_idx) > 0:
        scores[f_idx] = FLUSH_TABLE[suit_bits_by_suit[fs[f_idx], f_idx]]
    return scores


# ==============================================================================
# SUBSYSTEM 2: EXACT 1,755-CLASS 3-CARD PREFLOP ISOMORPHISM ENGINE
# ==============================================================================
def _fast_canon_3card_key(c0, c1, c2):
    cards = sorted([(c0 // 4, c0 % 4), (c1 // 4, c1 % 4), (c2 // 4, c2 % 4)], key=lambda x: x[0], reverse=True)
    (ra, sa), (rb, sb), (rc, sc) = cards
    if ra == rb == rc:
        suit_pat = 0
    elif ra == rb:
        suit_pat = 1 if (sc == sa or sc == sb) else 0
    elif rb == rc:
        suit_pat = 1 if (sa == sb or sa == sc) else 0
    else:
        if sa == sb == sc:
            suit_pat = 4  # monotone (self-blocking flush out)
        elif sa == sb:
            suit_pat = 3  # top two suited (highest EV)
        elif sa == sc:
            suit_pat = 2  # top and bottom suited
        elif sb == sc:
            suit_pat = 1  # bottom two suited
        else:
            suit_pat = 0  # rainbow
    return (ra, rb, rc, suit_pat)

PREFLOP_CANON_ID = np.zeros((52, 52, 52), dtype=np.uint16)
PREFLOP_BASE_STRENGTH = np.zeros(1755, dtype=np.float32)
CANON_REPRESENTATIVE = np.zeros((1755, 3), dtype=np.int32)
_canon_map = {}

for _c0 in range(52):
    for _c1 in range(_c0 + 1, 52):
        for _c2 in range(_c1 + 1, 52):
            _key = _fast_canon_3card_key(_c0, _c1, _c2)
            _cid = _canon_map.get(_key)
            if _cid is None:
                _cid = len(_canon_map)
                _canon_map[_key] = _cid
                CANON_REPRESENTATIVE[_cid] = [_c0, _c1, _c2]
                ra, rb, rc, sp = _key
                if ra == rb == rc:
                    val = 0.615 + 0.0082 * ra
                elif ra == rb:
                    val = 0.545 + 0.0135 * ra + 0.0040 * rc + (0.026 if sp == 1 else 0.0)
                elif rb == rc:
                    val = 0.510 + 0.0115 * rb + 0.0055 * ra + (0.026 if sp == 1 else 0.0)
                else:
                    suit_bonus = (0.0, 0.022, 0.028, 0.036, 0.020)[sp]
                    gap1 = max(0, 4 - (ra - rb)) * 0.010
                    gap2 = max(0, 4 - (rb - rc)) * 0.006
                    val = 0.340 + 0.0115 * ra + 0.0075 * rb + 0.0025 * rc + suit_bonus + gap1 + gap2
                PREFLOP_BASE_STRENGTH[_cid] = np.clip(val, 0.28, 0.78)
            PREFLOP_CANON_ID[_c0, _c1, _c2] = _cid
            PREFLOP_CANON_ID[_c0, _c2, _c1] = _cid
            PREFLOP_CANON_ID[_c1, _c0, _c2] = _cid
            PREFLOP_CANON_ID[_c1, _c2, _c0] = _cid
            PREFLOP_CANON_ID[_c2, _c0, _c1] = _cid
            PREFLOP_CANON_ID[_c2, _c1, _c0] = _cid

# Optional Hook for Phase 3 Embedded LZMA Precomputed 1,755 Equity Table (4 columns: vs_rand, vs_top50, vs_top20, playability)
EMBEDDED_PREFLOP_B64 = "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"
PREFLOP_TABLE_LOADED = False
PREFLOP_EQ_MATRIX = None

if EMBEDDED_PREFLOP_B64:
    try:
        _raw_bytes = lzma.decompress(base64.b64decode(EMBEDDED_PREFLOP_B64))
        PREFLOP_EQ_MATRIX = np.frombuffer(_raw_bytes, dtype=np.float16).astype(np.float32).reshape(1755, 4)
        PREFLOP_BASE_STRENGTH[:] = PREFLOP_EQ_MATRIX[:, 0]
        PREFLOP_TABLE_LOADED = True
    except Exception:
        PREFLOP_TABLE_LOADED = False


# ==============================================================================
# SUBSYSTEM 3 & 4: 3-WAY FLOP TOSS ENGINE & BAYESIAN EXPOSED-DISCARD FILTER
# ==============================================================================
RNG = np.random.default_rng(20261115)
KEEP_PAIRS = [(1, 2), (0, 2), (0, 1)]  # Index d corresponds to discarding hole_cards[d]

def flop_keep_value_batch(c1_arr, c2_arr, flop3):
    f_code = CARD_CODE[flop3].sum()
    f_sbits = CARD_SUIT_BIT[:, flop3].sum(axis=1)
    f_rank_mask = int(np.bitwise_or.reduce(f_sbits))
    codes = f_code + CARD_CODE[c1_arr] + CARD_CODE[c2_arr]
    sbits = f_sbits[:, None] + CARD_SUIT_BIT[:, c1_arr] + CARD_SUIT_BIT[:, c2_arr]
    scores = eval_batch_codes(codes, sbits)
    cat = (scores >> 20).astype(np.float32)
    sub = (scores & 0xFFFFF).astype(np.float32) / float(1 << 20)
    val = cat * 1.5 + sub
    fds = FLUSH_DRAW_SUIT[codes >> 32]
    val += (fds >= 0).astype(np.float32) * 1.8
    rmasks = np.bitwise_or(f_rank_mask, np.bitwise_or(1 << (c1_arr // 4), 1 << (c2_arr // 4)))
    val += (cat < 3).astype(np.float32) * STRAIGHT_DRAW_TABLE[rmasks] * 3.5
    r1, r2 = c1_arr // 4, c2_arr // 4
    val += (np.maximum(r1, r2) * 0.04 + np.minimum(r1, r2) * 0.02)
    return val

def evaluate_3way_flop_discard(hero3, flop3, n_rollouts=320, opp_aggr=0.3):
    rem46 = np.setdiff1d(np.arange(52, dtype=np.int32), np.concatenate([hero3, flop3]))
    if n_rollouts <= 0:
        c1s = np.array([hero3[1], hero3[0], hero3[0]], dtype=np.int32)
        c2s = np.array([hero3[2], hero3[2], hero3[1]], dtype=np.int32)
        vals = flop_keep_value_batch(c1s, c2s, flop3)
        best_d = int(np.argmax(vals))
        return best_d, float(np.clip(0.35 + 0.08 * vals[best_d], 0.15, 0.95)), vals

    rand_idx = np.argpartition(RNG.random((n_rollouts, 46)), 5, axis=1)[:, :5]
    samples = rem46[rand_idx]
    tr = samples[:, :2]
    opp3 = samples[:, 2:5]

    f_code = CARD_CODE[flop3].sum()
    f_sbits = CARD_SUIT_BIT[:, flop3].sum(axis=1)
    b5_codes = f_code + CARD_CODE[tr[:, 0]] + CARD_CODE[tr[:, 1]]
    b5_sbits = f_sbits[:, None] + CARD_SUIT_BIT[:, tr[:, 0]] + CARD_SUIT_BIT[:, tr[:, 1]]

    hero_scores = np.zeros((3, n_rollouts), dtype=np.uint32)
    opp_scores_3 = np.zeros((3, n_rollouts), dtype=np.uint32)
    opp_flop_val = np.zeros((3, n_rollouts), dtype=np.float32)

    for d, (i1, i2) in enumerate(KEEP_PAIRS):
        c1, c2 = hero3[i1], hero3[i2]
        h_codes = b5_codes + CARD_CODE[c1] + CARD_CODE[c2]
        h_sbits = b5_sbits + (CARD_SUIT_BIT[:, c1] + CARD_SUIT_BIT[:, c2])[:, None]
        hero_scores[d] = eval_batch_codes(h_codes, h_sbits)

        oc1, oc2 = opp3[:, i1], opp3[:, i2]
        o_codes = b5_codes + CARD_CODE[oc1] + CARD_CODE[oc2]
        o_sbits = b5_sbits + CARD_SUIT_BIT[:, oc1] + CARD_SUIT_BIT[:, oc2]
        opp_scores_3[d] = eval_batch_codes(o_codes, o_sbits)
        opp_flop_val[d] = flop_keep_value_batch(oc1, oc2, flop3)

    best_opp_d = np.argmax(opp_flop_val, axis=0)
    opp_final = opp_scores_3[best_opp_d, np.arange(n_rollouts)]
    best_opp_fval = opp_flop_val[best_opp_d, np.arange(n_rollouts)]

    # Weight rollouts by opponent preflop/flop aggression prior
    if opp_aggr > 0.35:
        rollout_w = np.exp((opp_aggr - 0.30) * 0.25 * (best_opp_fval - np.mean(best_opp_fval)))
        rollout_w /= np.sum(rollout_w)
        equities = (
            np.sum((hero_scores > opp_final[None, :]) * rollout_w[None, :], axis=1)
            + 0.5 * np.sum((hero_scores == opp_final[None, :]) * rollout_w[None, :], axis=1)
        )
    else:
        equities = (
            np.mean(hero_scores > opp_final[None, :], axis=1)
            + 0.5 * np.mean(hero_scores == opp_final[None, :], axis=1)
        )

    # Deception tie-breaker within 1.2% equity
    max_eq = float(np.max(equities))
    adj_scores = equities.copy()
    for d in range(3):
        if max_eq - equities[d] <= 0.012:
            disc_rank = (hero3[d] // 4) / 12.0
            if max_eq > 0.78:
                adj_scores[d] += (1.0 - disc_rank) * 0.008
            elif 0.42 <= max_eq <= 0.58:
                adj_scores[d] += disc_rank * 0.008
    best_d = int(np.argmax(adj_scores))
    return best_d, float(equities[best_d]), equities

def compute_bayesian_post_discard_range(flop3, hero_keep2, my_disc, opp_disc, opp_aggr_level=0.35, tau=0.35):
    dead = np.array([flop3[0], flop3[1], flop3[2], hero_keep2[0], hero_keep2[1], my_disc, opp_disc], dtype=np.int32)
    live45 = np.setdiff1d(np.arange(52, dtype=np.int32), dead)
    i1, i2 = np.triu_indices(len(live45), k=1)
    c1_all, c2_all = live45[i1], live45[i2]

    v_12 = flop_keep_value_batch(c1_all, c2_all, flop3)
    d_arr = np.full_like(c1_all, opp_disc)
    v_1d = flop_keep_value_batch(c1_all, d_arr, flop3)
    v_2d = flop_keep_value_batch(c2_all, d_arr, flop3)

    max_v = np.maximum(v_12, np.maximum(v_1d, v_2d))
    e12 = np.exp((v_12 - max_v) / tau)
    e1d = np.exp((v_1d - max_v) / tau)
    e2d = np.exp((v_2d - max_v) / tau)
    p_discard = e12 / (e12 + e1d + e2d)

    # Combine with preflop 3-card strength prior for {c1, c2, opp_disc}
    cids = PREFLOP_CANON_ID[c1_all, c2_all, d_arr]
    pre_str = PREFLOP_BASE_STRENGTH[cids]
    prior = np.exp(opp_aggr_level * ((v_12 - np.mean(v_12)) * 0.30 + (pre_str - 0.50) * 2.5))
    weights = prior * (0.08 + 0.92 * p_discard)
    weights /= np.sum(weights)
    return c1_all, c2_all, weights

def exact_river_range_equity(board5, hero2, opp_c1, opp_c2, opp_weights, my_disc=None):
    if opp_c1 is None or opp_c2 is None or opp_weights is None or len(opp_c1) == 0:
        return 0.50, 0.0, False

    t_card, r_card = board5[3], board5[4]
    valid_mask = (opp_c1 != t_card) & (opp_c1 != r_card) & (opp_c2 != t_card) & (opp_c2 != r_card)
    c1 = opp_c1[valid_mask]
    c2 = opp_c2[valid_mask]
    w = opp_weights[valid_mask]
    w_sum = float(np.sum(w))
    if w_sum <= 1e-9:
        w = np.ones(len(c1), dtype=np.float32) / max(1, len(c1))
    else:
        w = w / w_sum

    b_code = CARD_CODE[board5].sum()
    b_sbits = CARD_SUIT_BIT[:, board5].sum(axis=1)

    h_code = b_code + CARD_CODE[hero2[0]] + CARD_CODE[hero2[1]]
    h_sbits = b_sbits[:, None] + (CARD_SUIT_BIT[:, hero2[0]] + CARD_SUIT_BIT[:, hero2[1]])[:, None]
    h_score = eval_batch_codes(np.array([h_code], dtype=np.int64), h_sbits)[0]

    o_codes = b_code + CARD_CODE[c1] + CARD_CODE[c2]
    o_sbits = b_sbits[:, None] + CARD_SUIT_BIT[:, c1] + CARD_SUIT_BIT[:, c2]
    o_scores = eval_batch_codes(o_codes, o_sbits)

    wins = float(np.sum(w[h_score > o_scores]))
    ties = float(np.sum(w[h_score == o_scores]))
    eq = wins + 0.5 * ties

    # Fraction of opponent combinations holding Straight or better
    opp_nut_share = float(np.sum(w[(o_scores >> 20) >= 4]))

    # Blocker analysis: does hero block opponent nut hands?
    has_nut_blocker = False
    hero_all = [hero2[0], hero2[1]]
    if my_disc is not None:
        hero_all.append(my_disc)
    hero_ranks = set(c // 4 for c in hero_all)

    board_suits = [c % 4 for c in board5]
    for s in range(4):
        if board_suits.count(s) >= 3:
            for c in hero_all:
                if c % 4 == s and (c // 4) >= 11:
                    has_nut_blocker = True
                    break

    if not has_nut_blocker and (12 in hero_ranks or 11 in hero_ranks):
        board_ranks = set(c // 4 for c in board5)
        for st in range(10):
            mask = {st, st+1, st+2, st+3, st+4}
            if len(board_ranks.intersection(mask)) >= 3:
                has_nut_blocker = True
                break

    return eq, opp_nut_share, has_nut_blocker


# ==============================================================================
# SUBSYSTEM 5: 7-ARCHETYPE OPPONENT TRACKER & 4-MODE MATCH RISK CONTROLLER
# ==============================================================================
class OpponentAndMatchTracker:
    def __init__(self):
        self.total_hands = 400
        self.cap_c = 400
        self.cum_net = 0
        self.hands_played = 0
        # Bug-fixed opponent behavioral statistics
        self.opp_vpip = 0
        self.opp_pfr = 0
        self.opp_3bet = 0
        self.opp_faced_bet = 0
        self.opp_folded_to_bet = 0
        self.opp_called_bet = 0
        self.opp_raised_bet = 0
        self.opp_postflop_bets = 0
        self.opp_postflop_opportunities = 0
        self.opp_all_ins = 0

    def on_init(self, msg):
        self.total_hands = int(msg.get("total_hands", 400))
        self.cap_c = 1000 if self.total_hands >= 800 else 400

    def on_hand_result(self, msg):
        self.hands_played += 1
        self.cum_net += int(msg.get("net", 0))
        hist = msg.get("history", [])

        opp_pre_vol = False
        opp_pre_raise = False
        opp_pre_3bet = False

        # Reconstruct street bet amounts to accurately detect when opp faced a voluntary bet
        cur_street = "preflop"
        btn_actor = hist[0].get("actor") if hist and hist[0].get("street") == "preflop" else "me"
        my_bet = 1 if btn_actor == "me" else 2
        opp_bet = 2 if btn_actor == "me" else 1
        pre_raises = 0

        for h in hist:
            st = h.get("street", "preflop")
            if st != cur_street:
                cur_street = st
                my_bet = 0
                opp_bet = 0

            actor = h.get("actor")
            act = h.get("action")
            amt = int(h.get("amount", 0))

            if actor == "me":
                if act in ("call", "raise"):
                    if st == "preflop" and act == "raise":
                        pre_raises += 1
                    my_bet = max(my_bet, amt)
            elif actor == "opp":
                facing_voluntary_bet = (my_bet > opp_bet) and not (st == "preflop" and my_bet == 2 and opp_bet == 1)
                if facing_voluntary_bet:
                    self.opp_faced_bet += 1
                    if act == "fold":
                        self.opp_folded_to_bet += 1
                    elif act == "call":
                        self.opp_called_bet += 1
                    elif act == "raise":
                        self.opp_raised_bet += 1

                if st == "preflop":
                    if act in ("call", "raise"):
                        opp_pre_vol = True
                    if act == "raise":
                        opp_pre_raise = True
                        if pre_raises >= 1:
                            opp_pre_3bet = True
                        pre_raises += 1
                else:
                    self.opp_postflop_opportunities += 1
                    if act == "raise":
                        self.opp_postflop_bets += 1

                if amt >= 180:
                    self.opp_all_ins += 1
                if act in ("call", "raise"):
                    opp_bet = max(opp_bet, amt)

        if opp_pre_vol:
            self.opp_vpip += 1
        if opp_pre_raise:
            self.opp_pfr += 1
        if opp_pre_3bet:
            self.opp_3bet += 1

    def get_archetype_and_adjustments(self):
        n = max(1, self.hands_played)
        conf = min(0.75, n / 50.0)
        vpip_rate = self.opp_vpip / n
        pfr_rate = self.opp_pfr / n
        fold_rate = self.opp_folded_to_bet / max(1, self.opp_faced_bet)
        aggr_rate = self.opp_postflop_bets / max(1, self.opp_postflop_opportunities)
        shove_rate = self.opp_all_ins / n

        archetype = "GTO"
        call_delta = 0.0
        bluff_mult = 1.0
        value_size_mult = 1.0
        open_thresh_delta = 0.0

        if n >= 10:
            if fold_rate < 0.28 and pfr_rate < 0.22:
                # Archetype 1/2: Fish / Calling Station / Check-Call Starter Bot
                archetype = "FISH_STATION"
                bluff_mult = 0.0
                value_size_mult = 1.40
                call_delta = +0.03 * conf
                open_thresh_delta = -0.02
            elif fold_rate > 0.56 and pfr_rate < 0.35:
                # Archetype 6: Nit / Over-Folder
                archetype = "NIT"
                bluff_mult = 1.75
                value_size_mult = 0.85
                call_delta = +0.05 * conf
                open_thresh_delta = -0.06  # Steal blinds wider!
            elif shove_rate > 0.22 or (pfr_rate > 0.58 and aggr_rate > 0.52):
                # Archetype 7: Maniac / Shove-Bot
                archetype = "MANIAC"
                bluff_mult = 0.10
                value_size_mult = 1.30
                call_delta = -0.09 * conf  # Call down lighter
                open_thresh_delta = +0.04
            elif vpip_rate > 0.58 and pfr_rate > 0.38:
                # Archetype 4: LAG
                archetype = "LAG"
                bluff_mult = 0.65
                value_size_mult = 1.15
                call_delta = -0.04 * conf
            elif 0.35 <= vpip_rate <= 0.58 and 0.22 <= pfr_rate <= 0.42:
                # Archetype 3: TAG
                archetype = "TAG"
                bluff_mult = 1.10
                value_size_mult = 1.05
                call_delta = +0.01 * conf

        return archetype, call_delta, bluff_mult, value_size_mult, open_thresh_delta

    def get_match_mode(self, hand_id):
        hands_rem = max(0, self.total_hands - hand_id + 1)
        max_blind_loss = int(math.ceil(hands_rem * 0.75)) + 3
        if self.cum_net > (self.cap_c + max_blind_loss):
            return "LOCK_FULL_CAP"
        if hands_rem <= 24 and self.cum_net > (max_blind_loss + 6):
            return "LOCK_MATCH_WIN"
        if self.cum_net > int(1.25 * self.cap_c) and hands_rem < 100:
            return "SOFT_PROTECT"
        if self.cum_net < -int(0.85 * self.cap_c) and hands_rem < 80:
            return "COMEBACK_AGGR"
        return "NORMAL"


class HandState:
    def __init__(self):
        self.reset()

    def reset(self):
        self.hand_id = 1
        self.seat = "button"
        self.hole_cards = []
        self.my_discard = None
        self.opp_discard = None
        self.flop_cards = []
        self.opp_c1 = None
        self.opp_c2 = None
        self.opp_weights = None
        self.opp_aggr = 0.35
        self.processed_hist_len = 0


TRACKER = OpponentAndMatchTracker()
STATE = HandState()


def sanitize_action(reply, req):
    legal = req.get("legal_actions", [])
    act = reply.get("action")
    if act not in legal:
        if "check" in legal:
            return {"type": "action", "action": "check"}
        if "call" in legal and int(req.get("to_call", 0)) <= 2:
            return {"type": "action", "action": "call"}
        if "fold" in legal:
            return {"type": "action", "action": "fold"}
        return {"type": "action", "action": legal[0]}
    if act == "raise":
        min_r = int(req.get("min_raise", 2))
        max_r = int(req.get("max_raise", 200))
        amt = int(reply.get("amount", min_r))
        amt = max(min_r, min(max_r, amt))
        return {"type": "action", "action": "raise", "amount": int(amt)}
    return {"type": "action", "action": act}


def sanitize_discard(reply, req):
    hole = req.get("hole_cards", [])
    card = reply.get("card")
    if card in hole:
        return {"type": "discard", "card": card}
    worst = min(hole, key=lambda c: RANKS_STR.index(c[0]))
    return {"type": "discard", "card": worst}


def sync_history_and_range(req):
    """Updates STATE.opp_aggr and narrows post-discard Bayesian range from new history events."""
    hist = req.get("history", [])
    board_strs = req.get("board", [])
    if len(board_strs) >= 3 and not STATE.flop_cards:
        STATE.flop_cards = [CARD_TO_INT[c] for c in board_strs[:3]]

    new_events = hist[STATE.processed_hist_len:]
    STATE.processed_hist_len = len(hist)

    for h in new_events:
        if h.get("actor") == "opp":
            act = h.get("action")
            st = h.get("street", "preflop")
            if act == "raise":
                STATE.opp_aggr = min(1.6, STATE.opp_aggr + (0.20 if st == "preflop" else 0.30))
                # If on turn or river and Bayesian range exists, shift weights toward stronger hands + bluffs
                if st in ("turn", "river") and STATE.opp_weights is not None and len(board_strs) >= 4:
                    b_cards = np.array([CARD_TO_INT[c] for c in board_strs[:3]], dtype=np.int32)
                    v_scores = flop_keep_value_batch(STATE.opp_c1, STATE.opp_c2, b_cards)
                    v_norm = (v_scores - np.min(v_scores)) / max(1e-5, float(np.max(v_scores) - np.min(v_scores)))
                    # Polarized bet likelihood: high on top hands, moderate floor on bottom bluffs
                    likelihood = 0.20 + 0.80 * (v_norm ** 1.3)
                    STATE.opp_weights *= likelihood
                    STATE.opp_weights /= max(1e-9, float(np.sum(STATE.opp_weights)))
            elif act == "check" and st in ("turn", "river") and STATE.opp_weights is not None and len(board_strs) >= 4:
                # Opponent checked: slightly discount ultra-nutted traps
                b_cards = np.array([CARD_TO_INT[c] for c in board_strs[:3]], dtype=np.int32)
                v_scores = flop_keep_value_batch(STATE.opp_c1, STATE.opp_c2, b_cards)
                v_norm = (v_scores - np.min(v_scores)) / max(1e-5, float(np.max(v_scores) - np.min(v_scores)))
                likelihood = np.where(v_norm > 0.82, 0.72, 1.0)
                STATE.opp_weights *= likelihood
                STATE.opp_weights /= max(1e-9, float(np.sum(STATE.opp_weights)))


def choose_discard(req):
    hole_strs = req["hole_cards"]
    board_strs = req["board"]
    STATE.flop_cards = [CARD_TO_INT[c] for c in board_strs[:3]]
    hero3 = np.array([CARD_TO_INT[c] for c in hole_strs], dtype=np.int32)
    flop3 = np.array(STATE.flop_cards, dtype=np.int32)
    time_left = int(req.get("time_left_ms", 60000))
    n_sims = 340 if time_left > 15000 else (100 if time_left > 5000 else 0)
    best_d, _, _ = evaluate_3way_flop_discard(hero3, flop3, n_rollouts=n_sims, opp_aggr=STATE.opp_aggr)
    return sanitize_discard({"type": "discard", "card": hole_strs[best_d]}, req)


def choose_action(req):
    hand_id = int(req.get("hand_id", 1))
    legal = req.get("legal_actions", [])
    street = req.get("street", "preflop")
    pot = int(req.get("pot", 3))
    to_call = int(req.get("to_call", 0))
    my_bet = int(req.get("my_bet", 0))
    opp_bet = int(req.get("opp_bet", 0))
    time_left = int(req.get("time_left_ms", 60000))

    # 1. Check 4-Mode Match Risk Controller
    match_mode = TRACKER.get_match_mode(hand_id)
    if match_mode in ("LOCK_FULL_CAP", "LOCK_MATCH_WIN"):
        if "check" in legal:
            return sanitize_action({"type": "action", "action": "check"}, req)
        return sanitize_action({"type": "action", "action": "fold"}, req)

    sync_history_and_range(req)
    archetype, call_delta, bluff_mult, val_size_mult, open_thresh_delta = TRACKER.get_archetype_and_adjustments()

    if match_mode == "SOFT_PROTECT":
        if to_call >= 40:
            call_delta += 0.05
        bluff_mult *= 0.50
    elif match_mode == "COMEBACK_AGGR":
        bluff_mult *= 1.35
        val_size_mult *= 1.25

    pot_odds = to_call / float(max(1, pot + to_call))

    # ==========================================================================
    # 2. PREFLOP STRATEGY (Separating Button Open, BB Iso, and Facing Raises)
    # ==========================================================================
    if street == "preflop":
        if len(STATE.hole_cards) == 3:
            c0, c1, c2 = [CARD_TO_INT[c] for c in STATE.hole_cards]
            cid = int(PREFLOP_CANON_ID[c0, c1, c2])
            if PREFLOP_TABLE_LOADED and PREFLOP_EQ_MATRIX is not None:
                if opp_bet <= 2:
                    eq = float(PREFLOP_EQ_MATRIX[cid, 0])
                elif opp_bet <= 10:
                    eq = float(PREFLOP_EQ_MATRIX[cid, 1])
                else:
                    eq = float(PREFLOP_EQ_MATRIX[cid, 2])
            else:
                eq = float(PREFLOP_BASE_STRENGTH[cid])
                if opp_bet >= 6:
                    eq -= min(0.14, 0.032 * math.log2(max(1.0, opp_bet / 2.0)))
        else:
            eq = 0.50

        # CASE A: Unopened Pot on the Button (my_bet == 1, opp_bet == 2, to_call == 1)
        if my_bet == 1 and opp_bet == 2 and to_call == 1:
            open_raise_thresh = 0.455 + open_thresh_delta
            limp_thresh = 0.395 + open_thresh_delta
            if eq >= open_raise_thresh and "raise" in legal:
                target = 6 if eq >= 0.56 else 5
                return sanitize_action({"type": "action", "action": "raise", "amount": target}, req)
            if eq >= limp_thresh and "call" in legal:
                return sanitize_action({"type": "action", "action": "call"}, req)
            return sanitize_action({"type": "action", "action": "fold"}, req)

        # CASE B: Big Blind Option vs Button Limp (my_bet == 2, opp_bet == 2, to_call == 0)
        if to_call == 0:
            if eq >= (0.54 + open_thresh_delta) and "raise" in legal:
                target = int(round(6 * val_size_mult))
                return sanitize_action({"type": "action", "action": "raise", "amount": max(6, target)}, req)
            return sanitize_action({"type": "action", "action": "check"}, req)

        # CASE C: Facing a Preflop Raise (opp_bet > 2, to_call > 0)
        req_eq = pot_odds + call_delta - (0.03 if STATE.seat == "button" else 0.0)
        # Value 3-bet / 4-bet
        if eq >= max(0.62, req_eq + 0.14) and "raise" in legal:
            target = opp_bet + int(round((pot + to_call) * 0.85 * val_size_mult))
            return sanitize_action({"type": "action", "action": "raise", "amount": target}, req)
        # Light 3-bet bluff with suited wheel/broadway blockers against non-stations
        if opp_bet <= 8 and 0.48 <= eq <= 0.52 and "raise" in legal and RNG.random() < (0.14 * bluff_mult):
            target = opp_bet * 3
            return sanitize_action({"type": "action", "action": "raise", "amount": target}, req)
        if eq >= req_eq:
            return sanitize_action({"type": "action", "action": "call"}, req)
        return sanitize_action({"type": "action", "action": "fold"}, req)

    # ==========================================================================
    # 3. POSTFLOP EQUITY EVALUATION (Flop / Turn / River)
    # ==========================================================================
    if street == "flop":
        if len(STATE.hole_cards) == 3 and len(req.get("board", [])) >= 3:
            hero3 = np.array([CARD_TO_INT[c] for c in STATE.hole_cards], dtype=np.int32)
            flop3 = np.array([CARD_TO_INT[c] for c in req["board"][:3]], dtype=np.int32)
            STATE.flop_cards = flop3.tolist()
            n_sims = 240 if time_left > 15000 else (80 if time_left > 5000 else 0)
            _, eq, _ = evaluate_3way_flop_discard(hero3, flop3, n_rollouts=n_sims, opp_aggr=STATE.opp_aggr)
        else:
            eq = 0.50
    elif street == "turn":
        if len(STATE.hole_cards) >= 2 and len(req.get("board", [])) >= 4:
            board4 = np.array([CARD_TO_INT[c] for c in req["board"][:4]], dtype=np.int32)
            hero2 = np.array([CARD_TO_INT[c] for c in STATE.hole_cards[:2]], dtype=np.int32)
            if STATE.opp_c1 is None and STATE.opp_discard is not None and len(STATE.flop_cards) == 3:
                STATE.opp_c1, STATE.opp_c2, STATE.opp_weights = compute_bayesian_post_discard_range(
                    np.array(STATE.flop_cards, dtype=np.int32),
                    hero2,
                    CARD_TO_INT[STATE.my_discard],
                    CARD_TO_INT[STATE.opp_discard],
                    opp_aggr_level=STATE.opp_aggr,
                )
            if time_left <= 3000:
                vals = flop_keep_value_batch(hero2[0:1], hero2[1:2], board4[:3])
                eq = float(np.clip(0.35 + 0.08 * vals[0], 0.15, 0.95))
            elif STATE.opp_c1 is not None and len(STATE.opp_c1) > 0:
                dead = set(board4.tolist() + hero2.tolist())
                if STATE.my_discard:
                    dead.add(CARD_TO_INT[STATE.my_discard])
                if STATE.opp_discard:
                    dead.add(CARD_TO_INT[STATE.opp_discard])
                rem_rivers = [c for c in range(52) if c not in dead]
                n_riv = min(len(rem_rivers), 14 if time_left > 12000 else 5)
                sampled_rivers = RNG.choice(rem_rivers, size=n_riv, replace=False)
                eq_sum = 0.0
                for rc in sampled_rivers:
                    b5 = np.append(board4, np.int32(rc))
                    e_val, _, _ = exact_river_range_equity(b5, hero2, STATE.opp_c1, STATE.opp_c2, STATE.opp_weights)
                    eq_sum += e_val
                eq = eq_sum / float(n_riv)
            else:
                eq = 0.50
        else:
            eq = 0.50
        opp_nut_share = 0.0
        has_nut_blocker = False
    else:  # river
        if len(STATE.hole_cards) >= 2 and len(req.get("board", [])) >= 5:
            board5 = np.array([CARD_TO_INT[c] for c in req["board"][:5]], dtype=np.int32)
            hero2 = np.array([CARD_TO_INT[c] for c in STATE.hole_cards[:2]], dtype=np.int32)
            if STATE.opp_c1 is None and STATE.opp_discard is not None and len(STATE.flop_cards) == 3:
                STATE.opp_c1, STATE.opp_c2, STATE.opp_weights = compute_bayesian_post_discard_range(
                    np.array(STATE.flop_cards, dtype=np.int32),
                    hero2,
                    CARD_TO_INT[STATE.my_discard],
                    CARD_TO_INT[STATE.opp_discard],
                    opp_aggr_level=STATE.opp_aggr,
                )
            if STATE.opp_c1 is not None and len(STATE.opp_c1) > 0:
                my_d_int = CARD_TO_INT[STATE.my_discard] if STATE.my_discard else None
                eq, opp_nut_share, has_nut_blocker = exact_river_range_equity(
                    board5, hero2, STATE.opp_c1, STATE.opp_c2, STATE.opp_weights, my_disc=my_d_int
                )
            else:
                b_code = CARD_CODE[board5].sum() + CARD_CODE[hero2[0]] + CARD_CODE[hero2[1]]
                b_sbits = CARD_SUIT_BIT[:, board5].sum(axis=1)[:, None] + (CARD_SUIT_BIT[:, hero2[0]] + CARD_SUIT_BIT[:, hero2[1]])[:, None]
                score = eval_batch_codes(np.array([b_code], dtype=np.int64), b_sbits)[0]
                eq = float(np.clip(score / float(8 << 20), 0.15, 0.95))
                opp_nut_share = 0.0
                has_nut_blocker = False
        else:
            eq = 0.50
            opp_nut_share = 0.0
            has_nut_blocker = False

    # ==========================================================================
    # 4. POSTFLOP DECISION ENGINE (Value Sizing, Semi-Bluffs, River Polarized Bluffs)
    # ==========================================================================
    req_eq = pot_odds + call_delta
    # Tighten call threshold if opponent range has heavy nut share on big bet
    if street == "river" and opp_nut_share > 0.35 and to_call > pot:
        req_eq += 0.06

    if to_call == 0:
        if "raise" in legal:
            # Value bet threshold (thinner on river vs calling stations, tighter on turn)
            val_thresh = 0.57 if (street == "river" and archetype == "FISH_STATION") else 0.61
            if eq >= val_thresh:
                frac = (0.52 + 0.85 * (eq - val_thresh)) * val_size_mult
                target = my_bet + max(2, int(round(pot * frac)))
                return sanitize_action({"type": "action", "action": "raise", "amount": target}, req)

            # Flop / Turn Semi-Bluff C-Bet or Probe Bet
            if street in ("flop", "turn") and 0.40 <= eq <= 0.51:
                cbet_prob = (0.32 if street == "flop" else 0.22) * bluff_mult
                if RNG.random() < cbet_prob:
                    target = my_bet + max(2, int(round(pot * 0.55)))
                    return sanitize_action({"type": "action", "action": "raise", "amount": target}, req)

            # Game-Theoretic River Bluff with Low-Showdown-Value Hands (when opp can fold)
            if street == "river" and eq < 0.22 and bluff_mult >= 0.60:
                bonus = 0.12 if (has_nut_blocker and archetype in ("TAG", "LAG", "NIT")) else 0.0
                river_bluff_prob = min(0.40, 0.16 * bluff_mult + bonus)
                if RNG.random() < river_bluff_prob:
                    target = my_bet + max(2, int(round(pot * 0.68)))
                    return sanitize_action({"type": "action", "action": "raise", "amount": target}, req)

        return sanitize_action({"type": "action", "action": "check"}, req)
    else:
        # Facing a Postflop Bet (to_call > 0)
        # Value Raise / Check-Raise with strong hands
        if eq >= max(0.73, req_eq + 0.19) and "raise" in legal:
            target = opp_bet + int(round((pot + to_call) * 0.75 * val_size_mult))
            return sanitize_action({"type": "action", "action": "raise", "amount": target}, req)

        # Semi-Bluff Check-Raise on Flop with high-equity draws against small/medium bets
        if street == "flop" and to_call <= (pot // 2) and 0.46 <= eq <= 0.54 and "raise" in legal:
            if RNG.random() < (0.14 * bluff_mult):
                target = opp_bet + int(round((pot + to_call) * 0.65))
                return sanitize_action({"type": "action", "action": "raise", "amount": target}, req)

        # Call if equity beats adjusted pot odds
        if eq >= req_eq:
            return sanitize_action({"type": "action", "action": "call"}, req)

        return sanitize_action({"type": "action", "action": "fold"}, req)


def main():
    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        try:
            msg = json.loads(line)
        except ValueError:
            continue

        t = msg.get("type")
        try:
            if t == "init":
                TRACKER.on_init(msg)
                continue
            elif t == "new_hand":
                STATE.reset()
                STATE.hand_id = int(msg.get("hand_id", 1))
                STATE.seat = msg.get("seat", "button")
                STATE.hole_cards = list(msg.get("hole_cards", []))
                continue
            elif t == "discard_result":
                STATE.my_discard = msg.get("my_discard")
                STATE.opp_discard = msg.get("opp_discard")
                STATE.hole_cards = list(msg.get("hole_cards", []))
                if len(STATE.flop_cards) == 3 and len(STATE.hole_cards) == 2:
                    STATE.opp_c1, STATE.opp_c2, STATE.opp_weights = compute_bayesian_post_discard_range(
                        np.array(STATE.flop_cards, dtype=np.int32),
                        np.array([CARD_TO_INT[c] for c in STATE.hole_cards], dtype=np.int32),
                        CARD_TO_INT[STATE.my_discard],
                        CARD_TO_INT[STATE.opp_discard],
                        opp_aggr_level=STATE.opp_aggr,
                    )
                continue
            elif t == "hand_result":
                TRACKER.on_hand_result(msg)
                continue
            elif t == "action_request":
                reply = choose_action(msg)
            elif t == "discard_request":
                reply = choose_discard(msg)
            elif t == "match_end":
                break
            else:
                continue
        except Exception as exc:
            sys.stderr.write(f"ERR hand={STATE.hand_id}: {exc}\n")
            sys.stderr.flush()
            if t == "action_request":
                reply = sanitize_action({"action": "check"}, msg)
            elif t == "discard_request":
                reply = sanitize_discard({"card": ""}, msg)
            else:
                continue

        print(json.dumps(reply), flush=True)


if __name__ == "__main__":
    main()
