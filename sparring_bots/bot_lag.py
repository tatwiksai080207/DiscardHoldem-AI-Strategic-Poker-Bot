import sys
import json
import random

RANKS = "23456789TJQKA"

def choose_discard(req):
    hole = req.get("hole_cards", [])
    # Strategic discard
    board = req.get("board", [])
    board_ranks = [c[0] for c in board]
    def card_val(c):
        return (10 if c[0] in board_ranks else 0) + RANKS.index(c[0])
    worst = min(hole, key=card_val)
    return {"type": "discard", "card": worst}

def choose_action(req):
    legal = req.get("legal_actions", [])
    street = req.get("street", "preflop")
    to_call = int(req.get("to_call", 0))
    pot = int(req.get("pot", 3))
    min_r = int(req.get("min_raise", 2))
    max_r = int(req.get("max_raise", 200))

    if street == "preflop":
        if to_call <= 1:
            # Button open wide (75%)
            if "raise" in legal and random.random() < 0.75:
                return {"type": "action", "action": "raise", "amount": min(5, max_r)}
            if "call" in legal:
                return {"type": "action", "action": "call"}
        else:
            # 3-bet or call wide
            if "raise" in legal and random.random() < 0.25:
                return {"type": "action", "action": "raise", "amount": min(to_call * 3, max_r)}
            if to_call <= 12 and "call" in legal:
                return {"type": "action", "action": "call"}
            if "check" in legal:
                return {"type": "action", "action": "check"}
            return {"type": "action", "action": "fold"}

    # Postflop: aggressive c-betting and turn barreling
    if to_call == 0:
        barrel_prob = 0.70 if street == "flop" else (0.55 if street == "turn" else 0.40)
        if "raise" in legal and random.random() < barrel_prob:
            target = min(max_r, max(min_r, int(pot * 0.65)))
            return {"type": "action", "action": "raise", "amount": target}
        if "check" in legal:
            return {"type": "action", "action": "check"}
    else:
        # Call or raise back
        if "raise" in legal and random.random() < 0.18:
            return {"type": "action", "action": "raise", "amount": min(max_r, int((pot + to_call) * 0.8))}
        if to_call <= (pot * 0.75) and "call" in legal:
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

