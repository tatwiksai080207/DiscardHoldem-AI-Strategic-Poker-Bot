"""Comprehensive test suite for APEX DISCARD v10 Ultimate."""

import sys, os, time, math, json

# Import canonical bot
sys.path.insert(0, os.path.abspath("bot"))
import bot

def test_startup_and_memory():
    print("Testing startup and structure...")
    t0 = time.perf_counter()
    assert hasattr(bot, "PF_COLS"), "PF_COLS missing"
    assert hasattr(bot, "PF_PCT"), "PF_PCT missing"
    assert len(bot.PF_COLS) == 4, f"Unexpected PF_COLS length: {len(bot.PF_COLS)}"
    assert len(bot.PF_COLS[0]) == 22100, f"Unexpected combinations count: {len(bot.PF_COLS[0])}"
    dt = time.perf_counter() - t0
    print(f"  [PASS] Startup verified in {dt*1000:.1f}ms. Embedded preflop cube: 4 x 22,100 combinations.")

def test_hand_evaluator():
    print("Testing hand evaluator...")
    # Royal Flush vs Four of a Kind vs Full House vs Flush vs Straight vs Three of a kind
    royal = [bot.c2i(c) for c in ["As", "Ks", "Qs", "Js", "Ts", "2d", "3c"]]
    quads = [bot.c2i(c) for c in ["9s", "9h", "9d", "9c", "Kd", "2s", "3c"]]
    full_house = [bot.c2i(c) for c in ["8s", "8h", "8d", "Kh", "Kd", "2s", "3c"]]
    flush = [bot.c2i(c) for c in ["As", "Js", "8s", "6s", "2s", "Kd", "Qc"]]
    straight = [bot.c2i(c) for c in ["9s", "8h", "7d", "6c", "5s", "Kd", "Qc"]]
    trips = [bot.c2i(c) for c in ["7s", "7h", "7d", "Ac", "Kd", "2s", "3c"]]
    two_pair = [bot.c2i(c) for c in ["7s", "7h", "6d", "6c", "Kd", "2s", "3c"]]
    one_pair = [bot.c2i(c) for c in ["7s", "7h", "Ad", "Qc", "Jd", "2s", "3c"]]
    high_card = [bot.c2i(c) for c in ["As", "Kh", "Qd", "Jc", "8s", "2s", "3c"]]

    v_royal = bot.evaluate(royal)
    v_quads = bot.evaluate(quads)
    v_fh = bot.evaluate(full_house)
    v_fl = bot.evaluate(flush)
    v_str = bot.evaluate(straight)
    v_trips = bot.evaluate(trips)
    v_2p = bot.evaluate(two_pair)
    v_1p = bot.evaluate(one_pair)
    v_hc = bot.evaluate(high_card)

    assert v_royal > v_quads > v_fh > v_fl > v_str > v_trips > v_2p > v_1p > v_hc, "Evaluator ranking hierarchy failed!"
    print(f"  [PASS] Evaluator correctly strictly ordered: Royal > Quads > Full House > Flush > Straight > Trips > Two Pair > Pair > High Card")

def test_rule_8_2_sticky_lock():
    print("Testing Rule 8.2 Sticky Mathematical Lock Controller...")
    p = bot.Player()
    p.total_hands = 400
    p.hand_id = 380  # 20 remaining hands
    # max_blind_loss = ceil(1.5 * 20) = 30
    
    # Lead is 32 (< 30 + 4) -> should not lock
    p.cum_net = 32
    p.match_locked = False
    assert not p.is_mathematical_lock(), "Locked too early with lead 32"

    # Lead is 35 (>= 30 + 4) -> MUST lock
    p.cum_net = 35
    assert p.is_mathematical_lock(), "Failed to lock with lead 35"
    assert p.match_locked, "Sticky flag not set"

    # Verify sticky behavior
    p.cum_net = 10
    assert p.is_mathematical_lock(), "Lock failed to stay sticky"

    # Verify action while locked
    req = {
        "legal_actions": ["fold", "call", "raise"],
        "to_call": 10,
        "pot": 20,
        "my_bet": 2,
        "opp_bet": 12
    }
    act = p.act(req)
    assert act["action"] == "fold", f"Expected fold while locked facing bet, got {act}"

    req_free = {
        "legal_actions": ["check", "raise"],
        "to_call": 0,
        "pot": 10,
        "my_bet": 2,
        "opp_bet": 2
    }
    act_free = p.act(req_free)
    assert act_free["action"] == "check", f"Expected check while locked for free, got {act_free}"
    print("  [PASS] Rule 8.2 Sticky Mathematical Lock verified: guaranteed match point protection.")

def test_rule_8_3_capped_delta_risk():
    print("Testing Rule 8.3 Capped Delta Risk Budget...")
    p = bot.Player()
    p.total_hands = 400
    p.cap_c = 400
    p.hand_id = 385  # 15 hands remaining, max_blind_loss = 23
    
    # Net is 480 -> surplus = 480 - (400 + 23) = 57 >= 40
    p.cum_net = 480
    cap = p.get_risk_cap()
    assert cap == 20, f"Expected risk cap 20 when surplus >= 40, got {cap}"

    # Net is 440 -> surplus = 440 - 423 = 17 >= 15
    p.cum_net = 440
    cap = p.get_risk_cap()
    assert cap == 40, f"Expected risk cap 40 when surplus >= 15, got {cap}"

    # Early in match (hand 100) -> cap should be 200 (no restriction)
    p.hand_id = 100
    p.cum_net = 500
    assert p.get_risk_cap() == 200, "Risk cap should not restrict early in match"
    print("  [PASS] Rule 8.3 Capped Delta Risk Budget verified: concave tournament score defense.")

def test_protocol_handling():
    print("Testing protocol message handling...")
    p = bot.Player()
    p.on_init({"type": "init", "total_hands": 400, "time_bank_ms": 30000, "per_response_ms": 5000})
    assert p.total_hands == 400
    assert p.cap_c == 400

    p.on_new_hand({"type": "new_hand", "hand_id": 1, "seat": "button", "hole_cards": ["As", "Kh", "Qd"]})
    assert p.hand_id == 1
    assert p.seat == "button"
    assert len(p.hole3) == 3

    # Discard request
    d_card = p.discard({"type": "discard_request", "hole_cards": ["As", "Kh", "Qd"], "board": ["Js", "Ts", "2c"]})
    assert d_card in ["As", "Kh", "Qd"], f"Invalid discard card: {d_card}"

    # Discard result
    p.on_discard_result({"type": "discard_result", "my_discard": d_card, "opp_discard": "2c", "hole_cards": [c for c in ["As", "Kh", "Qd"] if c != d_card]})
    assert p.hole2 is not None
    assert len(p.hole2) == 2

    # Action request
    req = {
        "type": "action_request",
        "street": "turn",
        "legal_actions": ["check", "raise"],
        "to_call": 0,
        "pot": 20,
        "my_bet": 0,
        "opp_bet": 0,
        "my_stack": 190,
        "opp_stack": 190,
        "board": ["Js", "Ts", "2c", "4h"],
        "history": [],
        "min_raise": 4,
        "max_raise": 190,
        "time_left_ms": 28000
    }
    act = p.act(req)
    assert act["type"] == "action"
    assert act["action"] in ["check", "raise"]
    print("  [PASS] Protocol I/O cycle conforms to tournament protocol specification.")

if __name__ == "__main__":
    test_startup_and_memory()
    test_hand_evaluator()
    test_rule_8_2_sticky_lock()
    test_rule_8_3_capped_delta_risk()
    test_protocol_handling()
    print("\nALL UNIT AND INTEGRATION TESTS PASSED (100% SUCCESS)!")
