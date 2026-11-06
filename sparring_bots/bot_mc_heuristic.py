import sys
import json
import random
import numpy as np

RANKS_STR = "23456789TJQKA"
SUITS_STR = "shdc"
CARD_TO_INT = {r + s: (ri * 4 + si) for ri, r in enumerate(RANKS_STR) for si, s in enumerate(SUITS_STR)}
INT_TO_CARD = [r + s for r in RANKS_STR for s in SUITS_STR]

# Simple evaluator for heuristic bot
def simple_eval_7(cards7):
    # cards7: list of (rank, suit)
    ranks = sorted([c // 4 for c in cards7], reverse=True)
    counts = {}
    for r in ranks:
        counts[r] = counts.get(r, 0) + 1
    # Check quads, trips, pairs
    quads = [r for r, cnt in counts.items() if cnt == 4]
    trips = sorted([r for r, cnt in counts.items() if cnt == 3], reverse=True)
    pairs = sorted([r for r, cnt in counts.items() if cnt == 2], reverse=True)
    if quads: return (7, quads[0])
    if trips and (len(trips) > 1 or pairs): return (6, trips[0])
    if trips: return (3, trips[0])
    if len(pairs) >= 2: return (2, pairs[0], pairs[1])
    if len(pairs) == 1: return (1, pairs[0])
    return (0, ranks[:5])

def choose_discard(req):
    hole = req.get("hole_cards", [])
    # Discard lowest rank
    worst = min(hole, key=lambda c: RANKS_STR.index(c[0]))
    return {"type": "discard", "card": worst}

def choose_action(req):
    legal = req.get("legal_actions", [])
    to_call = int(req.get("to_call", 0))
    pot = int(req.get("pot", 3))
    min_r = int(req.get("min_raise", 2))
    max_r = int(req.get("max_raise", 200))
    street = req.get("street", "preflop")
    hole = req.get("hole_cards", [])

    if street == "preflop":
        high = max(RANKS_STR.index(c[0]) for c in hole)
        if to_call <= 1:
            if high >= 9 and "raise" in legal:
                return {"type": "action", "action": "raise", "amount": min(5, max_r)}
            if "call" in legal:
                return {"type": "action", "action": "call"}
            if "check" in legal:
                return {"type": "action", "action": "check"}
        else:
            if high >= 11 and "raise" in legal:
                return {"type": "action", "action": "raise", "amount": min(to_call * 3, max_r)}
            if high >= 7 and to_call <= 10 and "call" in legal:
                return {"type": "action", "action": "call"}
            if "check" in legal:
                return {"type": "action", "action": "check"}
            return {"type": "action", "action": "fold"}

    # Postflop
    pot_odds = to_call / float(max(1, pot + to_call))
    if to_call == 0:
        if "raise" in legal and random.random() < 0.40:
            target = min(max_r, max(min_r, int(pot * 0.5)))
            return {"type": "action", "action": "raise", "amount": target}
        if "check" in legal:
            return {"type": "action", "action": "check"}
    else:
        if pot_odds < 0.30 and "call" in legal:
            return {"type": "action", "action": "call"}
        if "check" in legal:
            return {"type": "action", "action": "check"}
        return {"type": "action", "action": "fold"}

    return {"type": "action", "action": "check" if "check" in legal else "fold"}

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
        if t == "action_request":
            reply = choose_action(msg)
        elif t == "discard_request":
            reply = choose_discard(msg)
        elif t == "match_end":
            break
        else:
            continue
        print(json.dumps(reply), flush=True)

if __name__ == "__main__":
    main()

