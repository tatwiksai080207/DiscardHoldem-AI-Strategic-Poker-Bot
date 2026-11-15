import os, sys, json, random, traceback
if os.environ.get("FUZZ_NO_LZMA") == "1":
    sys.modules["lzma"] = None          # makes "import lzma" raise ImportError -> tests the fallback path
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "bot"))
import bot

R = "23456789TJQKA"; S = "shdc"
DECK = [r + s for r in R for s in S]
N_CASES = int(os.environ.get("FUZZ_N", "20000"))

def build_case(rng):
    street = rng.choice(["preflop", "flop", "turn", "river"])
    nb = {"preflop": 0, "flop": 3, "turn": 4, "river": 5}[street]
    cards = rng.sample(DECK, 9)
    hero3, board5, opp_disc = cards[:3], cards[3:8], cards[8]

    T = bot.TRACKER
    T.reset()
    T.total_hands = rng.choice([400, 1000])
    T.cap_c = 400 if T.total_hands == 400 else 1000
    T.cum_net = rng.choice([-900, -300, 0, 40, 120, 380, 450, 700, 1100])
    T.hands_played = rng.randint(0, 400)
    T.match_locked = rng.random() < 0.10
    started_locked = T.match_locked

    S_ = bot.STATE
    S_.reset()
    hand_id = rng.randint(1, T.total_hands)
    S_.hand_id = hand_id
    S_.seat = rng.choice(["button", "big_blind"])
    if street in ("preflop", "flop"):
        S_.hole_cards = list(hero3)
    else:
        S_.hole_cards = list(hero3[:2])
        S_.my_discard = hero3[2]
        S_.opp_discard = opp_disc
        S_.flop_cards = [bot.CARD_TO_INT[c] for c in board5[:3]]

    prior = 0 if street == "preflop" else rng.choice([4, 12, 30, 80, 150])
    max_raise = 200 - prior
    if street == "preflop":
        my_bet = rng.choice([1, 2])
        opp_bet = max(my_bet, rng.choice([2, 2, 6, 10, 24, 60, 200]))
    else:
        my_bet = rng.choice([0, 0, 0, 6, 20])
        opp_bet = my_bet + rng.choice([0, 0, 2, 6, 20, 60, 400])
    opp_bet = min(opp_bet, max_raise)
    my_bet = min(my_bet, opp_bet)
    to_call = opp_bet - my_bet
    raises_left = rng.choice([0, 1, 2, 3, 4])
    can_raise = raises_left > 0 and opp_bet < max_raise
    legal = ["fold", "call"] if to_call > 0 else ["check"]
    if can_raise:
        legal.append("raise")

    req = {
        "type": "action_request", "hand_id": hand_id, "street": street,
        "board": board5[:nb], "pot": 2 * prior + my_bet + opp_bet,
        "my_bet": my_bet, "opp_bet": opp_bet, "to_call": to_call,
        "my_stack": 200 - prior - my_bet, "opp_stack": 200 - prior - opp_bet,
        "legal_actions": legal, "raises_left": raises_left,
        "history": [], "time_left_ms": rng.choice([60000, 20000, 8000, 2500, 500]),
    }
    if can_raise:
        req["min_raise"] = min(max_raise, opp_bet + max(2, rng.choice([2, 4, 10, 40])))
        req["max_raise"] = max_raise
    if street in ("flop", "turn", "river") and rng.random() < 0.3:
        h = {"street": street, "actor": "opp", "action": rng.choice(["raise", "check"])}
        if h["action"] == "raise":
            h["amount"] = 6
        req["history"] = [h]
    return req, started_locked

def check_reply(req, r, started_locked):
    legal = req["legal_actions"]
    assert isinstance(r, dict) and r.get("type") == "action", f"bad reply shape {r!r}"
    json.dumps(r)                                   # raises TypeError if numpy types leaked in
    assert r["action"] in legal, f"action {r['action']} not in {legal}"
    if r["action"] == "raise":
        a = r["amount"]
        assert type(a) is int, f"raise amount type {type(a)}"
        assert req["min_raise"] <= a <= req["max_raise"], f"raise {a} outside [{req['min_raise']},{req['max_raise']}]"
    if r["action"] == "fold":
        assert req["to_call"] > 0, "fold with nothing to call"
    if started_locked:
        assert r["action"] in ("check", "fold"), f"locked bot played {r['action']}"

rng = random.Random(20261115)
fails = 0
for i in range(N_CASES):
    req, locked = build_case(rng)
    try:
        r = bot.choose_action(req)
        check_reply(req, r, locked)
    except Exception:
        fails += 1
        if fails <= 5:
            print(f"--- FAIL case {i} ---")
            print(json.dumps(req))
            traceback.print_exc()

print(f"lzma_loaded={bot.PREFLOP_TABLE_LOADED}  cases={N_CASES}  failures={fails}")
if os.environ.get("FUZZ_NO_LZMA") == "1" and bot.PREFLOP_TABLE_LOADED:
    print("FAIL: lzma was supposed to be blocked"); sys.exit(1)

# Discard path fuzzing: 2,000 random discard requests
discard_fails = 0
for j in range(2000):
    bot.TRACKER.reset()
    bot.STATE.reset()
    cards = rng.sample(DECK, 6)
    hero3, board3 = cards[:3], cards[3:6]
    req = {
        "type": "discard_request",
        "hand_id": 1,
        "hole_cards": list(hero3),
        "board": list(board3),
        "pot": 24,
        "my_stack": 188,
        "opp_stack": 188,
        "time_left_ms": 60000
    }
    try:
        dr = bot.choose_discard(req)
        json.dumps(dr)
        assert isinstance(dr, dict) and dr.get("type") == "discard", f"bad discard shape {dr!r}"
        assert dr.get("card") in hero3, f"discarded card {dr.get('card')} not in hole cards {hero3}"
    except Exception:
        discard_fails += 1
        if discard_fails <= 5:
            print(f"--- FAIL discard case {j} ---")
            print(json.dumps(req))
            traceback.print_exc()

print(f"discard_cases=2000  discard_failures={discard_fails}")

total_fails = fails + discard_fails
sys.exit(1 if total_fails else 0)
