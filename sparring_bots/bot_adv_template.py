import sys, json, os
MODE = os.path.basename(__file__).replace("bot_adv_", "").replace(".py", "")
RANKS = "23456789TJQKA"
HANDS = 0

def A(a, amt=None):
    d = {"type": "action", "action": a}
    if amt is not None:
        d["amount"] = int(amt)
    return d

def act(req):
    legal = req["legal_actions"]
    to_call = req.get("to_call", 0)
    if MODE == "shove":
        if "raise" in legal: return A("raise", req["max_raise"])
        return A("call") if "call" in legal else A("check")
    if MODE == "minraise":
        if "raise" in legal: return A("raise", req["min_raise"])
        return A("call") if "call" in legal else A("check")
    if MODE == "limp":
        if "check" in legal: return A("check")
        if "call" in legal and req["street"] == "preflop" and to_call <= 1: return A("call")
        return A("fold")
    if MODE == "alwayscall":
        return A("call") if "call" in legal else A("check")
    if MODE == "alwaysfold":
        return A("check") if "check" in legal else A("fold")
    if MODE == "crash":
        if HANDS > 60:
            os._exit(1)                       # simulate a crash mid-match
        return A("call") if "call" in legal else A("check")
    return A("check") if "check" in legal else A("fold")

def main():
    global HANDS
    for line in sys.stdin:
        try:
            msg = json.loads(line)
        except ValueError:
            continue
        t = msg.get("type")
        if t == "new_hand":
            HANDS += 1
        elif t == "action_request":
            print(json.dumps(act(msg)), flush=True)
        elif t == "discard_request":
            worst = min(msg["hole_cards"], key=lambda c: RANKS.index(c[0]))
            print(json.dumps({"type": "discard", "card": worst}), flush=True)
        elif t == "match_end":
            break

if __name__ == "__main__":
    main()
