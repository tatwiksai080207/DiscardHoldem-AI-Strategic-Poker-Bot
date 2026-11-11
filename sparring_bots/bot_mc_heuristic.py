import sys
import json
import random

RANKS_STR = "23456789TJQKA"
SUITS_STR = "shdc"
CURRENT_HOLE = []

def choose_discard(req):
    global CURRENT_HOLE
    hole = req.get("hole_cards", []) or CURRENT_HOLE
    # Discard lowest rank
    worst = min(hole, key=lambda c: RANKS_STR.index(c[0])) if hole else "2c"
    return {"type": "discard", "card": worst}

def choose_action(req):
    global CURRENT_HOLE
    legal = req.get("legal_actions", [])
    to_call = int(req.get("to_call", 0))
    pot = int(req.get("pot", 3))
    min_r = int(req.get("min_raise", 2))
    max_r = int(req.get("max_raise", 200))
    street = req.get("street", "preflop")
    hole = CURRENT_HOLE

    if street == "preflop":
        high = max(RANKS_STR.index(c[0]) for c in hole) if hole else 7
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
    global CURRENT_HOLE
    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        try:
            msg = json.loads(line)
        except ValueError:
            continue
        t = msg.get("type")
        if t == "new_hand":
            CURRENT_HOLE = list(msg.get("hole_cards", []))
        elif t == "discard_request":
            CURRENT_HOLE = list(msg.get("hole_cards", []))
            reply = choose_discard(msg)
            print(json.dumps(reply), flush=True)
        elif t == "discard_result":
            CURRENT_HOLE = list(msg.get("hole_cards", []))
        elif t == "action_request":
            reply = choose_action(msg)
            print(json.dumps(reply), flush=True)
        elif t == "match_end":
            break
        else:
            continue

if __name__ == "__main__":
    main()
