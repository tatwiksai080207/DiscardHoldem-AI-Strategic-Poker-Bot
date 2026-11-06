import sys
import json
import random

RANKS = "23456789TJQKA"

def choose_discard(req):
    hole = req.get("hole_cards", [])
    # Discard randomly or lowest
    return {"type": "discard", "card": random.choice(hole)}

def choose_action(req):
    legal = req.get("legal_actions", [])
    pot = int(req.get("pot", 3))
    min_r = int(req.get("min_raise", 2))
    max_r = int(req.get("max_raise", 200))
    to_call = int(req.get("to_call", 0))

    if "raise" in legal and random.random() < 0.65:
        # Shove or pot-size bet
        if random.random() < 0.35:
            target = max_r
        else:
            target = min(max_r, max(min_r, int(pot * 1.5)))
        return {"type": "action", "action": "raise", "amount": target}

    if "call" in legal and random.random() < 0.85:
        return {"type": "action", "action": "call"}

    if "check" in legal:
        return {"type": "action", "action": "check"}

    return {"type": "action", "action": legal[0]}

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

