import os
from pathlib import Path

core_path = Path('scripts/assets/core.py')
preflop_path = Path('scripts/assets/preflop.b85')

core_code = core_path.read_text(encoding='utf-8')
preflop_b85 = preflop_path.read_text(encoding='utf-8').strip()
compiled_core = core_code.replace('__PREFLOP_DATA__', preflop_b85)
compiled_core = compiled_core.replace(
    'fm=np.take_along_axis(masks,np.maximum(fs,0)[...,None],axis=-1)[...,0]',
    'fm=np.take_along_axis(masks,np.maximum(fs,0)[...,None],axis=-1)[...,0]&8191'
)

agent_v9_code = '''
STREETS = ('preflop', 'flop', 'turn', 'river')

def events(history, seat):
    """Reconstruct actual facing-bet opportunities and pot-relative sizing."""
    bets = [1, 2] if seat == 'button' else [2, 1]
    committed = bets.copy()
    street = 'preflop'
    out = []
    for h in history:
        st = h.get('street', 'preflop')
        if st != street:
            bets = [0, 0]
            street = st
        a = 0 if h.get('actor') == 'me' else 1
        call = max(0, bets[1 - a] - bets[a])
        pot = sum(committed)
        act = h.get('action')
        amount = int(h.get('amount', bets[a]) or 0)
        add = max(0, amount - bets[a]) if act in ('raise', 'call') else 0
        ratio = (max(0, amount - bets[1 - a]) if act == 'raise' else call) / max(1, pot + call)
        faced_ratio = call / max(1, pot - call)
        out.append((a, st, act, call > 0, ratio, committed[a] + add >= 200, faced_ratio))
        bets[a] += add
        committed[a] += add
    return out

# 5 Behavioral Archetypes: [opening aggression, fold when facing a bet, re-raise probability]
ARCH = np.array([
    [0.03, 0.05, 0.02], # Calling Station
    [0.20, 0.20, 0.20], # Random / Inconsistent
    [0.24, 0.62, 0.10], # Tight Selective
    [0.43, 0.37, 0.15], # Ordinary Aggressive
    [0.86, 0.06, 0.60], # Maniac
])

class Model:
    def __init__(self):
        self.logp = np.log(np.array([0.20, 0.15, 0.15, 0.40, 0.10]))
        self.disc_log = np.log(np.array([0.65, 0.20, 0.15]))
        self.count = np.zeros((4, 3, 3), dtype=float) # street, size bucket, fold/call/raise
        self.open = np.zeros((4, 2), dtype=float)
        self.hands = 0

    def mix(self):
        p = np.exp(self.logp - self.logp.max())
        return 0.92 * norm(p) + 0.08 * np.array([0.2, 0.15, 0.15, 0.4, 0.1])

    def discmix(self):
        p = np.exp(self.disc_log - self.disc_log.max())
        return 0.90 * norm(p) + 0.10 * np.array([0.65, 0.20, 0.15])

    def foldrate(self, street, ratio):
        ix = STREETS.index(street)
        k = 0 if ratio < 0.45 else (1 if ratio < 1.1 else 2)
        a = self.count[ix, k]
        total = self.count.sum(axis=(0, 1))
        global_rate = (total[0] + 3) / (total.sum() + 8)
        return float((a[0] + 8 * global_rate) / (a.sum() + 8))

    def update(self, msg, state):
        self.hands += 1
        self.logp *= 0.985
        for a, st, act, facing, ratio, allin, faced_ratio in events(msg.get('history', []), state.seat):
            if a != 1:
                continue
            ix = STREETS.index(st)
            k = 0 if faced_ratio < 0.45 else (1 if faced_ratio < 1.1 else 2)
            if facing:
                p = ARCH[:, 1] if act == 'fold' else (ARCH[:, 2] if act == 'raise' else 1 - ARCH[:, 1] - ARCH[:, 2])
                if act in ('fold', 'call', 'raise'):
                    self.count[ix, k, ('fold', 'call', 'raise').index(act)] += 1
            else:
                p = ARCH[:, 0] if act == 'raise' else 1 - ARCH[:, 0]
                self.open[ix, int(act == 'raise')] += 1
            self.logp += 0.24 * np.log(np.maximum(p, 0.015))
        self.logp -= self.logp.max()
        self.logp = np.maximum(self.logp, -9)

        oc = msg.get('opp_hole_cards')
        d = msg.get('opp_discard')
        board = msg.get('board', [])
        if oc and d and len(board) >= 3:
            c = np.array([CID[x] for x in oc] + [CID[d]])
            q = np.array([float(quality(c[k], [CID[x] for x in board[:3]])) for k in KEEP])
            z = np.exp((q - q.max()) / 0.45)
            rational = z[2] / z.sum()
            low = int(np.argmin(c))
            likelihood = np.array([rational, 0.96 if low == 2 else 0.02, 1 / 3])
            self.disc_log = 0.99 * self.disc_log + 0.45 * np.log(np.maximum(likelihood, 0.015))
            self.disc_log -= self.disc_log.max()
            self.disc_log = np.maximum(self.disc_log, -7)

    def likelihood(self, strength, ev):
        _, st, act, facing, ratio, allin, faced_ratio = ev
        if act != 'raise':
            ratio = faced_ratio
        s = np.asarray(strength)
        mix = self.mix()
        p = np.zeros_like(s, dtype=float)
        for k, m in enumerate(mix):
            if k == 0: # Station
                fold = 0.025 + 0.10 * sigmoid((0.12 - s) * 12)
                raise_p = 0.015 + 0.80 * sigmoid((s - 0.96) * 35)
            elif k == 1: # Random
                fold = np.full_like(s, 0.20)
                raise_p = np.full_like(s, 0.20)
            elif k == 2: # Tight
                fold = sigmoid((0.57 + 0.09 * min(ratio, 2) - s) * 14)
                raise_p = 0.015 + 0.72 * sigmoid((s - (0.89 if facing else 0.78)) * 22)
            elif k == 3: # Aggressive
                fold = sigmoid((0.29 + 0.12 * min(ratio, 2) - s) * 11)
                raise_p = 0.055 + 0.72 * sigmoid((s - (0.86 if facing else 0.61)) * 15)
            else: # Maniac
                fold = np.full_like(s, 0.06)
                raise_p = 0.66 + 0.16 * s
            if facing:
                raise_p = np.minimum(raise_p, 1 - fold - 0.01)
                v = fold if act == 'fold' else (raise_p if act == 'raise' else 1 - fold - raise_p)
            else:
                v = raise_p if act == 'raise' else 1 - raise_p
            if act == 'raise' and ratio > 1 and k in (0, 2, 3):
                v = v * (0.15 + 0.85 * sigmoid((s - (0.68 + 0.06 * min(ratio, 3))) * 13))
            p += m * v
        return np.maximum(p, 0.018) ** 0.72

class State:
    def __init__(self):
        self.rng = np.random.default_rng()
        self.model = Model()
        self.total = 400
        self.cap_c = 400
        self.cum_net = 0
        self.match_locked = False
        self.errors = 0
        self.reset({'hand_id': 1, 'hole_cards': ['As', 'Kh', 'Qc'], 'seat': 'button'})

    def reset(self, msg):
        self.id = int(msg['hand_id'])
        self.seat = msg['seat']
        self.hole = np.array([CID[x] for x in msg['hole_cards']], dtype=np.int64)
        self.original = self.hole.copy()
        self.flop = []
        self.discard = None
        self.opp_discard = None
        self.history = []
        self.flop_cache = None
        self.pf_cache = None
        self.range_cache = None
        self.score_cache = {}
        self.last_discard = 0

    def budget(self, req):
        left = max(0, req.get('time_left_ms', 30000)) / 1000
        remain = max(1, self.total - self.id + 1)
        perhand = max(0, left - 0.5) / remain
        return min(0.12, max(0.001, perhand * 0.23)), perhand

    def is_mathematical_lock(self, hand_id, street='preflop', my_bet=1):
        """Rule 8.2 Match Point Guarantee: true worst-case fold bleed bound 1.5 * H_rem."""
        if self.match_locked:
            return True
        hands_rem = self.total - hand_id
        if hands_rem > 28:
            return False
        max_blind_loss = int(math.ceil(1.5 * hands_rem))
        if self.cum_net > max_blind_loss:
            if hands_rem <= 12:
                self.match_locked = True
                return True
            if self.cum_net >= (max_blind_loss + 4):
                self.match_locked = True
                return True
        return False

    def get_risk_cap(self, hand_id):
        """Rule 8.3 Capped Delta Protection in the final 36 hands."""
        hands_rem = self.total - hand_id
        if hands_rem > 36:
            return 200
        max_loss = int(math.ceil(1.5 * hands_rem))
        surplus = self.cum_net - (self.cap_c + max_loss)
        if surplus > 0:
            return max(20, min(80, surplus + 10))
        return 200

S = State()

def weights_for(strengths, history):
    n = len(next(iter(strengths.values())))
    w = np.ones(n)
    for ev in events(history, S.seat):
        if ev[0] == 1 and ev[1] in strengths:
            w *= S.model.likelihood(strengths[ev[1]], ev)
    return norm(w)

def relative(q):
    sq = np.sort(q)
    return (np.searchsorted(sq, q, 'left') + np.searchsorted(sq, q, 'right')) / (2 * len(q))

def policy_discard(opp, flop, rng):
    q = np.stack([quality(opp[:, k], flop) for k in KEEP], axis=1)
    probs = np.exp((q - q.max(axis=1)[:, None]) / 0.40)
    probs /= probs.sum(axis=1)[:, None]
    mix = S.model.discmix()
    low = np.argmin(opp, axis=1)
    probs = mix[0] * probs + mix[2] / 3
    probs[np.arange(len(opp)), low] += mix[1]
    idx = (rng.random(len(opp))[:, None] > np.cumsum(probs, axis=1)).sum(axis=1)
    idx = np.minimum(idx, 2)
    return np.take_along_axis(opp, KEEP[idx], axis=1), q.max(axis=1)

def preflop(req, budget):
    base = float(pf_equity(S.original))
    hist = req.get('history', [])
    if not any(h.get('actor') == 'opp' and h.get('action') == 'raise' for h in hist):
        return base, 0.006, None
    n = 768 if budget > 0.015 else 256
    if S.pf_cache is None:
        live = [c for c in range(52) if c not in S.original]
        x = draws(S.rng, live, n, 8)
        op = x[:, :3]
        board = x[:, 3:]
        flop = board[:, :3]
        oq, _ = policy_discard(op, flop, S.rng)
        hq = np.stack([quality(S.original[k], flop) for k in KEEP], axis=1)
        hp = S.original[KEEP[np.argmax(hq, axis=1)]]
        hs = evaluate(joint(hp, board))
        os = evaluate(joint(oq, board))
        wins = (hs > os) + 0.5 * (hs == os)
        S.pf_cache = (wins, percentile(pf_equity(op)))
    wins, strength = S.pf_cache
    w = weights_for({'preflop': strength}, hist)
    eq = float(w @ wins)
    ess = 1 / float(w @ w)
    eq = (eq * ess + base * 12) / (ess + 12)
    return eq, math.sqrt(max(0.01, eq * (1 - eq)) / (ess + 12)), None

def flop_samples(n):
    live = [c for c in range(52) if c not in set(S.original) | set(S.flop)]
    x = draws(S.rng, live, n, 5)
    opp = x[:, :3]
    board = np.concatenate((np.tile(S.flop, (n, 1)), x[:, 3:]), axis=1)
    op, q = policy_discard(opp, S.flop, S.rng)
    os = evaluate(joint(op, board))
    hs = np.stack([evaluate(joint(S.original[k], board)) for k in KEEP], axis=1)
    wins = (hs > os[:, None]) + 0.5 * (hs == os[:, None])
    return wins, percentile(pf_equity(opp)), q

def flop_equity(req, budget, discard=False):
    S.flop = [CID[x] for x in req['board'][:3]]
    start = time.monotonic()
    if S.flop_cache is None:
        S.flop_cache = flop_samples(768 if budget > 0.012 else 192)
    wins, pf, q = S.flop_cache
    hist = req.get('history', S.history)

    def compute():
        w = weights_for({'preflop': pf, 'flop': relative(q)}, hist)
        eq = w @ wins
        return w, eq, 1 / float(w @ w)

    w, eq, ess = compute()
    gap = np.sort(eq)[-1] - np.sort(eq)[-2]
    threshold = req.get('to_call', 0) / max(1, req.get('pot', 0) + req.get('to_call', 0))
    need = (discard and gap < 0.07) or abs(eq.max() - threshold) < 0.10
    if need and len(wins) < 3072 and time.monotonic() - start < budget * 0.55:
        extra = flop_samples(1536 if budget > 0.03 else 384)
        wins = np.concatenate((wins, extra[0]))
        pf = np.concatenate((pf, extra[1]))
        q = np.concatenate((q, extra[2]))
        S.flop_cache = (wins, pf, q)
        w, eq, ess = compute()

    # Discard choice with quality tie-breaker on very close equities
    top2_idx = np.argsort(eq)[-2:]
    if abs(eq[top2_idx[1]] - eq[top2_idx[0]]) < 0.006:
        q_scores = [float(quality(S.original[k], S.flop)) for k in KEEP]
        idx = top2_idx[1] if q_scores[top2_idx[1]] >= q_scores[top2_idx[0]] else top2_idx[0]
    else:
        idx = int(np.argmax(eq))
    S.last_discard = idx
    value = float(eq[idx])
    err = math.sqrt(max(0.01, value * (1 - value)) / ess)
    return value, err, (wins[:, idx], w, relative(q))

def make_range():
    dead = set(S.original) | set(S.flop) | {S.opp_discard}
    live = [c for c in range(52) if c not in dead]
    pairs = np.array([(a, b) for i, a in enumerate(live) for b in live[i + 1:]], dtype=np.int64)
    d = np.full(len(pairs), S.opp_discard)
    original = np.column_stack((pairs, d))
    qs = np.stack([quality(original[:, k], S.flop) for k in KEEP], axis=1)
    soft = np.exp((qs - qs.max(axis=1)[:, None]) / 0.45)
    prob = soft[:, 2] / soft.sum(axis=1)
    low = np.argmin(original, axis=1) == 2
    mix = S.model.discmix()
    prior = mix[0] * (0.04 + 0.96 * prob) + mix[1] * np.where(low, 0.96, 0.02) + mix[2] / 3
    S.range_cache = (pairs, prior, percentile(pf_equity(original)), qs.max(axis=1))

def postflop(req, budget):
    board = np.array([CID[x] for x in req.get('board', [])])
    if len(S.flop) < 3 and len(board) >= 3:
        S.flop = list(board[:3])
    st = req.get('street', 'turn')
    if S.range_cache is None:
        make_range()
    pairs, prior, pf, fq = S.range_cache
    valid = ~np.any(np.isin(pairs, board), axis=1)
    pairs = pairs[valid]
    prior = prior[valid]
    pf = pf[valid]
    fq = fq[valid]
    strengths = {'preflop': pf, 'flop': relative(fq)}
    for n in (4, 5):
        if len(board) >= n:
            name = STREETS[n - 2]
            strengths[name] = relative(quality(pairs, board[:n]))
    w = norm(prior * weights_for(strengths, req.get('history', [])))
    strength = strengths[st]
    if st == 'river':
        hs = evaluate(joint(S.hole, board))
        os = evaluate(joint(pairs, board))
        wins = (hs > os) + 0.5 * (hs == os)
        return float(w @ wins), 0.0, (wins, w, strength)
    n = 2048 if budget > 0.025 else (768 if budget > 0.01 else 192)
    ix = S.rng.choice(len(pairs), size=n, p=w)
    op = pairs[ix]
    live = np.array([c for c in range(52) if c not in set(S.original) | set(board) | {S.opp_discard}])
    river = S.rng.choice(live, size=n)
    bad = (river == op[:, 0]) | (river == op[:, 1])
    while np.any(bad):
        river[bad] = S.rng.choice(live, size=int(bad.sum()))
        bad = (river == op[:, 0]) | (river == op[:, 1])
    b = np.column_stack((np.tile(board, (n, 1)), river))
    hs = evaluate(joint(S.hole, b))
    os = evaluate(joint(op, b))
    wins = (hs > os) + 0.5 * (hs == os)
    eq = float(wins.mean())
    return eq, math.sqrt(max(0.01, eq * (1 - eq)) / n), (wins, np.full(n, 1 / n), strength[ix])

def action(name, amount=None):
    out = {'type': 'action', 'action': name}
    if amount is not None:
        out['amount'] = int(amount)
    return out

def choose(req):
    budget, perhand = S.budget(req)
    st = req['street']
    legal = req['legal_actions']
    S.history = list(req.get('history', []))

    # Rule 8.2 Sticky Mathematical Lock Check
    if S.is_mathematical_lock(S.id, st, req.get('my_bet', 1)):
        if 'fold' in legal:
            return action('fold')
        if 'check' in legal:
            return action('check')
        return action('call')

    if perhand < 0.006:
        return action('check' if 'check' in legal else 'fold')

    if st == 'preflop':
        eq, err, dist = preflop(req, budget)
    elif st == 'flop':
        eq, err, dist = flop_equity(req, budget)
    else:
        eq, err, dist = postflop(req, budget)

    pot = req.get('pot', 0)
    call = req.get('to_call', 0)
    stack = req.get('my_stack', 200 - req.get('my_bet', 0))
    allin = call >= stack
    mix = S.model.mix()
    station = float(mix[0])
    randomish = float(mix[1] + mix[4])
    nit = float(mix[2])

    # Rule 8.3 Late Match Risk Cap
    hands_rem = S.total - S.id
    risk_cap = S.get_risk_cap(S.id)
    surplus = S.cum_net - (S.cap_c + int(math.ceil(1.5 * hands_rem))) if hands_rem <= 36 else 0

    margin = 0.0 if st == 'river' or allin else {'preflop': 0.035, 'flop': 0.045, 'turn': 0.025}[st]
    if S.seat == 'big_blind' and st != 'river' and not allin:
        margin += 0.015
    if st == 'river' and call > 25 and (nit > 0.25 or station > 0.35):
        margin += 0.030
    margin += min(0.025, err * 0.35)
    eq_call = max(0.0, eq - margin)

    # Protect capped delta from high risk calls when holding comfortable lead late
    if surplus > 0 and call > risk_cap and eq_call < 0.62:
        return action('fold' if 'fold' in legal else 'check')

    call_ev = eq_call * (pot + call) - call
    baseline = call_ev if call else eq_call * pot
    choice = action('call') if call_ev >= 0 and call else action('fold' if call else 'check')
    if 'raise' not in legal:
        return choice

    # Positional opening strategy using 1755 table
    if st == 'preflop' and req['opp_bet'] <= 2:
        strength = float(percentile(float(pf_equity(S.original))))
        bar = 0.30 if S.seat == 'button' else 0.58
        bar -= 0.14 * nit
        bar += 0.12 * station
        if strength > bar:
            raise_size = min(req['max_raise'], max(req['min_raise'], 6 if strength < 0.85 else 8))
            if surplus > 0:
                raise_size = min(raise_size, req['my_bet'] + risk_cap)
            return action('raise', raise_size)
        return choice

    # EV Candidate Raise Sizing across 4 Pot Fractions
    best = baseline if not call else max(0.0, baseline)
    for fraction in (0.33, 0.65, 1.0, 1.5):
        target = req['opp_bet'] + round(fraction * (pot + call))
        target = max(req['min_raise'], min(req['max_raise'], target))
        if surplus > 0:
            target = min(target, req['my_bet'] + risk_cap)
        risk = target - req['my_bet']
        extra = target - req['opp_bet']

        if eq > 0.88 and stack - risk < 0.18 * (pot + call + 2 * extra):
            target = req['max_raise']
            if surplus > 0:
                target = min(target, req['my_bet'] + risk_cap)
            risk = target - req['my_bet']
            extra = target - req['opp_bet']

        ratio = extra / max(1, pot + call)
        fold = S.model.foldrate(st, ratio)
        total = S.model.count.sum()
        conf = total / (total + 35)
        expected = float(mix @ ARCH[:, 1])
        fold = conf * fold + (1 - conf) * expected
        fold = max(0.015, min(0.85, fold + 0.06 * math.log(max(0.2, ratio / 0.65))))
        if call:
            fold *= 0.55

        if dist is not None:
            wins, w, strength = dist
            continuance = (station + 0.65 * randomish) + (1 - station - 0.65 * randomish) * sigmoid(
                (strength - (0.33 + 0.12 * min(ratio, 2))) * 12
            )
            wc = w * continuance
            eqc = float(wc @ wins / max(1e-12, wc.sum()))
        else:
            eqc = max(0.01, eq - (0.045 + 0.025 * ratio) * (1 - station - 0.7 * randomish))
        eqc = max(0, eqc - margin)
        bet_ev = fold * pot + (1 - fold) * (eqc * (pot + extra + risk) - risk)

        uncertainty = (0.012 + err * 0.45) * risk * (1 - station * 0.7)
        if st != 'river':
            uncertainty += risk * 0.025 * (1 - station)
        bet_ev -= uncertainty

        value = eqc > (0.53 if not call else 0.60)
        bluff = (eq < 0.43 and fold > 0.48 and S.model.hands >= 12 and not call)
        if not value and not bluff:
            continue
        if bluff and S.rng.random() > 0.32:
            continue
        if bet_ev > best + max(0.25, 0.02 * pot):
            best = bet_ev
            choice = action('raise', target)
    return choice

def discard(req):
    S.history = S.history or []
    budget, perhand = S.budget(req)
    if req.get('board'):
        S.flop = [CID[x] for x in req['board'][:3]]
    if perhand < 0.006:
        i = int(np.argmax([quality(S.original[k], S.flop) for k in KEEP]))
    else:
        flop_equity(req, budget, True)
        i = S.last_discard
    return {'type': 'discard', 'card': CARDS[int(S.original[i])]}

def sanitize(reply, req):
    legal = req.get('legal_actions', [])
    a = reply.get('action')
    if a not in legal:
        a = 'check' if 'check' in legal else ('fold' if 'fold' in legal else legal[0])
    if a == 'raise':
        x = reply.get('amount', req['min_raise'])
        if isinstance(x, bool) or not isinstance(x, (int, np.integer)):
            x = req['min_raise']
        return action(a, max(req['min_raise'], min(req['max_raise'], int(x))))
    return action(a)

def main():
    for line in sys.stdin:
        try:
            msg = json.loads(line)
        except (ValueError, TypeError):
            continue
        if not isinstance(msg, dict):
            continue
        t = msg.get('type')
        try:
            if t == 'init':
                S.total = max(1, int(msg.get('total_hands', msg.get('hands', 400))))
                S.cap_c = int(msg.get('cap', 1000 if S.total >= 800 else 400))
                continue
            if t == 'new_hand':
                S.reset(msg)
                continue
            if t == 'discard_result':
                S.discard = CID[msg['my_discard']]
                S.opp_discard = CID[msg['opp_discard']]
                S.hole = np.array([CID[x] for x in msg['hole_cards']])
                S.range_cache = None
                continue
            if t == 'hand_result':
                S.model.update(msg, S)
                S.cum_net += int(msg.get('net', 0))
                continue
            if t == 'match_end':
                break
            if t == 'action_request':
                reply = sanitize(choose(msg), msg)
                h = {'street': msg['street'], 'actor': 'me', 'action': reply['action']}
                if reply['action'] == 'raise':
                    h['amount'] = reply['amount']
                elif reply['action'] == 'call':
                    h['amount'] = msg['opp_bet']
                S.history.append(h)
            elif t == 'discard_request':
                reply = discard(msg)
                if reply.get('card') not in msg['hole_cards']:
                    reply = {'type': 'discard', 'card': min(msg['hole_cards'], key=lambda c: CID[c])}
            else:
                continue
        except Exception as exc:
            if S.errors < 12:
                print('strategy fallback: ' + repr(exc), file=sys.stderr, flush=True)
            S.errors += 1
            if t == 'action_request':
                reply = sanitize(action('check'), msg)
            elif t == 'discard_request':
                reply = {'type': 'discard', 'card': min(msg['hole_cards'], key=lambda c: CID[c])}
            else:
                continue
        print(json.dumps(reply, separators=(',', ':')), flush=True)

if __name__ == '__main__':
    main()
'''

bot_v9_full = compiled_core + '\n' + agent_v9_code
out_path = Path('bot/bot_v9.py')
out_path.write_text(bot_v9_full, encoding='utf-8')
print(f'Successfully built {out_path} ({len(bot_v9_full)} bytes)')
