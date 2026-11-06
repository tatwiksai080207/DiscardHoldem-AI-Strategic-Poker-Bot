import sys
import json
import random

RANKS = "23456789TJQKA"

def eval_3card(hole):
    ranks = sorted([RANKS.index(c[0]) for c in hole], reverse=True)
    is_pair = (ranks[0] == ranks[1]) or (ranks[1] == ranks[2])
    is_suited = (hole[0][1] == hole[1][1]) or (hole[0][1] == hole[2][1]) or (hole[1][1] == hole[2][1])
    score = (ranks[0] + ranks[1] * 0.7 + ranks[2] * 0.3) / 12.0
    if is_pair: score += 0.22
    if is_suited: score += 0.08
    return min(1.0, score)

def choose_discard(req):
    hole = req.get("hole_cards", [])
    board = req.get("board", [])
    # Keep pairs with board or highest cards
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
        score = eval_3card(req.get("hole_cards", []))
        if to_call <= 1:  # button open or BB check
            if score >= 0.52 and "raise" in legal:
                return {"type": "action", "action": "raise", "amount": min(6, max_r)}
            if score >= 0.42 and "call" in legal:
                return {"type": "action", "action": "call"}
            if "check" in legal:
                return {"type": "action", "action": "check"}
            return {"type": "action", "action": "fold"}
        else:
            # facing raise
            if score >= 0.68 and "raise" in legal:
                return {"type": "action", "action": "raise", "amount": min(to_call * 3, max_r)}
            if score >= 0.50 and "call" in legal and to_call <= 15:
                return {"type": "action", "action": "call"}
            return {"type": "action", "action": "fold"}

    # Postflop
    if to_call == 0:
        if "raise" in legal and random.random() < 0.55:
            target = min(max_r, max(min_r, int(pot * 0.6)))
            return {"type": "action", "action": "raise", "amount": target}
        if "check" in legal:
            return {"type": "action", "action": "check"}
    else:
        # Facing bet: calls medium bets, folds to big shoves unless strong
        pot_odds = to_call / float(pot + to_call)
        if pot_odds < 0.35 and "call" in legal:
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

