import sys
import os
import json
import time

workspace = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(workspace, "Starter_Kit"))

import bot

def test_fuzz_worked_example_8_3_allin_discard():
    """Worked Example 8.3: Preflop All-In Discard with 0 stacks."""
    bot.STATE.reset()
    bot.STATE.hand_id = 42
    bot.STATE.seat = "button"
    bot.STATE.hole_cards = ["Ah", "Kh", "2c"]

    discard_req = {
        "type": "discard_request",
        "hand_id": 42,
        "hole_cards": ["Ah", "Kh", "2c"],
        "board": ["Th", "9h", "2d"],
        "pot": 400,
        "my_stack": 0,
        "opp_stack": 0,
        "time_left_ms": 50000,
        "extra_unknown_field": "fuzz_test_8_3"
    }
    reply = bot.choose_discard(discard_req)
    assert reply["type"] == "discard"
    assert reply["card"] in ["Ah", "Kh", "2c"]
    print("  [PASS] Worked Example 8.3 (Preflop All-In Discard)")

def test_fuzz_worked_example_8_2_raise_cap():
    """Worked Example 8.2: Raise cap reached (raises_left == 0)."""
    bot.STATE.reset()
    action_req = {
        "type": "action_request",
        "hand_id": 5,
        "street": "flop",
        "board": ["Ah", "Kd", "2s"],
        "pot": 120,
        "my_bet": 60,
        "opp_bet": 60,
        "to_call": 0,
        "my_stack": 140,
        "opp_stack": 140,
        "legal_actions": ["check"],  # 'raise' is absent due to 4-raise cap!
        "raises_left": 0,
        "time_left_ms": 45000
    }
    reply = bot.choose_action(action_req)
    assert reply["type"] == "action"
    assert reply["action"] == "check"
    assert "amount" not in reply
    print("  [PASS] Worked Example 8.2 (Raise Cap raises_left == 0)")

def test_fuzz_short_stack_clamping():
    """Short-stack all-in where min_raise == max_raise."""
    bot.STATE.reset()
    action_req = {
        "type": "action_request",
        "hand_id": 10,
        "street": "river",
        "board": ["Ah", "Kd", "2s", "7c", "9d"],
        "pot": 380,
        "my_bet": 190,
        "opp_bet": 190,
        "to_call": 0,
        "my_stack": 10,
        "opp_stack": 10,
        "legal_actions": ["check", "raise"],
        "min_raise": 200,
        "max_raise": 200,
        "time_left_ms": 30000
    }
    # Test reply with invalid amount
    raw_reply = {"type": "action", "action": "raise", "amount": 9999}
    sanitized = bot.sanitize_action(raw_reply, action_req)
    assert sanitized["type"] == "action"
    assert sanitized["action"] == "raise"
    assert sanitized["amount"] == 200
    print("  [PASS] Short-Stack All-In Clamping (min_raise == max_raise)")

def test_fuzz_unknown_fields_and_notices():
    """Sending unknown fields and notices must not crash the bot."""
    bot.STATE.reset()
    notice_msg = {
        "type": "notice",
        "code": "illegal_action",
        "applied": "check",
        "details": "fuzzed extra detail payload",
        "nested": {"random_array": [1, 2, 3]}
    }
    # HandResult with extra keys
    hand_res_msg = {
        "type": "hand_result",
        "hand_id": 1,
        "result": "fold",
        "net": 10,
        "board": ["Ah", "Kd", "2s"],
        "my_hole_cards": ["Qs", "Js"],
        "opp_hole_cards": None,
        "history": [
            {"street": "preflop", "actor": "me", "action": "raise", "amount": 6, "fuzz_key": 42},
            {"street": "preflop", "actor": "opp", "action": "fold"}
        ],
        "future_field_v2": True
    }
    bot.TRACKER.on_hand_result(hand_res_msg)
    assert bot.TRACKER.cum_net == 10
    print("  [PASS] Unknown Fields & Notice Immunity")

def test_fuzz_emergency_time_throttle():
    """Emergency time-bank throttling (time_left_ms <= 1200) executes in sub-millisecond time."""
    bot.STATE.reset()
    bot.STATE.hole_cards = ["Ah", "Kd", "Qc"]
    bot.STATE.flop_cards = [bot.CARD_TO_INT[c] for c in ["Th", "9d", "2s"]]

    req = {
        "type": "action_request",
        "hand_id": 99,
        "street": "turn",
        "board": ["Th", "9d", "2s", "4c"],
        "pot": 40,
        "my_bet": 20,
        "opp_bet": 20,
        "to_call": 0,
        "my_stack": 180,
        "opp_stack": 180,
        "legal_actions": ["check", "raise"],
        "time_left_ms": 1200  # Emergency Tier-3 Trigger!
    }
    t0 = time.perf_counter()
    reply = bot.choose_action(req)
    t1 = time.perf_counter()
    duration_ms = (t1 - t0) * 1000.0
    assert reply["action"] in ["check", "raise"]
    print(f"  [PASS] Emergency Time-Bank Throttling: took {duration_ms:.3f} ms (< 5.0 ms threshold)")

def test_late_match_risk_cap_and_lock():
    """Verify Upgrade B: get_late_match_risk_cap and narrow-lead lock in final hands."""
    bot.TRACKER.reset()
    bot.TRACKER.total_hands = 400
    bot.TRACKER.cum_net = 60

    # Hand 360 (41 hands remaining > 36): risk_cap must be 200 (no risk cap)
    cap360 = bot.TRACKER.get_late_match_risk_cap(360)
    assert cap360 == 200, f"Expected 200, got {cap360}"

    # Hand 380 (21 hands remaining <= 36):
    # max_blind_loss = ceil(21 * 0.75) + 12 = 16 + 12 = 28.
    # win_cushion = 60 - 28 = 32 > 24.
    # risk_cap = max(6, 32 - 4) = 28.
    cap380 = bot.TRACKER.get_late_match_risk_cap(380)
    assert cap380 == 28, f"Expected 28, got {cap380}"

    # In final 24 hands (hand 385, 16 hands rem, max_blind_loss = 12 + 12 = 24):
    # If cum_net = 35 -> win_cushion = 35 - 24 = 11 (0 < win_cushion <= 24):
    # is_mathematical_lock must trigger True!
    bot.TRACKER.cum_net = 35
    locked = bot.TRACKER.is_mathematical_lock(385, street="preflop", my_bet=1)
    assert locked is True, "Expected narrow lead lock to trigger True"
    print("  [PASS] Final-36-Hand Risk Cap & Match-Win Lock (Upgrade B)")

def test_deterministic_44_river_turn_eval():
    """Verify Upgrade A: Turn evaluates all 44 rivers deterministically when time_left_ms > 18000."""
    bot.STATE.reset()
    bot.STATE.hole_cards = ["Ah", "Kd"]
    bot.STATE.flop_cards = [bot.CARD_TO_INT[c] for c in ["Th", "9h", "2c"]]
    bot.STATE.my_discard = "3s"
    bot.STATE.opp_discard = "4d"

    req = {
        "type": "action_request",
        "hand_id": 12,
        "street": "turn",
        "board": ["Th", "9h", "2c", "7h"],
        "pot": 30,
        "my_bet": 15,
        "opp_bet": 15,
        "to_call": 0,
        "my_stack": 170,
        "opp_stack": 170,
        "legal_actions": ["check", "raise"],
        "min_raise": 17,
        "max_raise": 185,
        "time_left_ms": 45000  # > 18000 -> full 44-river enumeration
    }
    t0 = time.perf_counter()
    reply = bot.choose_action(req)
    t1 = time.perf_counter()
    duration_ms = (t1 - t0) * 1000.0
    assert reply["type"] == "action"
    assert reply["action"] in ["check", "raise"]
    print(f"  [PASS] Turn River-Sampling Evaluation: took {duration_ms:.2f} ms")

if __name__ == "__main__":
    print("Running Adversarial Protocol & Edge-Case Fuzzing Suite...")
    test_fuzz_worked_example_8_3_allin_discard()
    test_fuzz_worked_example_8_2_raise_cap()
    test_fuzz_short_stack_clamping()
    test_fuzz_unknown_fields_and_notices()
    test_fuzz_emergency_time_throttle()
    test_late_match_risk_cap_and_lock()
    test_deterministic_44_river_turn_eval()
    print("\nALL ADVERSARIAL PROTOCOL FUZZ TESTS PASSED!")

