import sys
import json
import random

RANKS = "23456789TJQKA"

def choose_discard(req):
    hole = req.get("hole_cards", [])
    worst = min(hole, key=lambda c: RANKS.index(c[0]))
    return {"type": "discard", "card": worst}

def choose_action(req):
    legal = req.get("legal_actions", [])
    to_call = int(req.get("to_call", 0))
    pot = int(req.get("pot", 3))

    if to_call == 0:
        # Occasionally min-raise with random chance, otherwise check
        if "raise" in legal and random.random() < 0.10:
            min_r = int(req.get("min_raise", 2))
            return {"type": "action", "action": "raise", "amount": min_r}
        if "check" in legal:
            return {"type": "action", "action": "check"}

    # Station calls almost anything up to reasonable bet sizes
    if "call" in legal:
        if to_call <= 30 or random.random() < 0.85:
            return {"type": "action", "action": "call"}

    if "check" in legal:
        return {"type": "action", "action": "check"}
    return {"type": "action", "action": "fold"}

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

