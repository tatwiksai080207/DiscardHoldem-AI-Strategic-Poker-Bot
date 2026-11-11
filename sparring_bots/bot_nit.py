import sys
import json

RANKS = "23456789TJQKA"
CURRENT_HOLE = []

def preflop_rank_score(hole):
    if not hole:
        return 10.0
    ranks = sorted([RANKS.index(c[0]) for c in hole], reverse=True)
    if len(ranks) < 2:
        return ranks[0] if ranks else 5.0
    is_pair = (ranks[0] == ranks[1]) or (len(ranks) > 2 and ranks[1] == ranks[2])
    high = ranks[0]
    return (10 if is_pair else 0) + high + ranks[1] * 0.5

def choose_discard(req):
    global CURRENT_HOLE
    hole = req.get("hole_cards", []) or CURRENT_HOLE
    worst = min(hole, key=lambda c: RANKS.index(c[0])) if hole else "2c"
    return {"type": "discard", "card": worst}

def choose_action(req):
    global CURRENT_HOLE
    legal = req.get("legal_actions", [])
    to_call = int(req.get("to_call", 0))
    street = req.get("street", "preflop")

    if street == "preflop":
        score = preflop_rank_score(CURRENT_HOLE)
        # Only play top hands
        if score >= 17:
            if "raise" in legal:
                min_r = int(req.get("min_raise", 2))
                return {"type": "action", "action": "raise", "amount": min(min_r + 2, int(req.get("max_raise", 200)))}
            if "call" in legal:
                return {"type": "action", "action": "call"}
        elif score >= 13 and to_call <= 2:
            if "call" in legal:
                return {"type": "action", "action": "call"}
            if "check" in legal:
                return {"type": "action", "action": "check"}
        if "check" in legal:
            return {"type": "action", "action": "check"}
        return {"type": "action", "action": "fold"}

    # Postflop: check if free, fold to almost any bet unless holding high pair
    if to_call == 0:
        if "check" in legal:
            return {"type": "action", "action": "check"}
    else:
        # Folds to aggression
        if to_call <= 4 and "call" in legal:
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
