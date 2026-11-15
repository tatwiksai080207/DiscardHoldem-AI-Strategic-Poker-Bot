"""Unit and regression tests for APEX DISCARD v11 Dominator."""

import sys, os
sys.path.insert(0, os.path.abspath("."))

def test_v11():
    print("Running v11 verification suite...")
    import sparring_bots.bot_v11 as bot

    # 1. Verify tables loaded
    assert len(bot.PF_COLS) == 4
    assert len(bot.PF_COLS[0]) == 22100
    print("1. Preflop 4-tier equity cube verified.")

    # 2. Verify evaluator
    from sparring_bots.bot_v11 import evaluate, c2i, best_keep, KEY
    rf = [c2i(c) for c in ["As", "Ks", "Qs", "Js", "Ts"]]
    fh = [c2i(c) for c in ["As", "Ah", "Ad", "Ks", "Kh"]]
    pair = [c2i(c) for c in ["As", "Ah", "7d", "5s", "2h"]]
    assert evaluate(rf) > evaluate(fh) > evaluate(pair)
    print("2. Exact table evaluator verified.")

    # 3. Verify Player initialization
    p = bot.Player()
    p.on_init({"total_hands": 200, "time_bank_ms": 30000})
    assert p.total_hands == 200
    assert not p.is_stage2

    p2 = bot.Player()
    p2.on_init({"total_hands": 500, "time_bank_ms": 75000})
    assert p2.total_hands == 500
    assert p2.is_stage2
    print("3. Stage 1 and Stage 2 detection verified.")

    # 4. Critical Review Case: Royal Flush on river facing bet
    # "Hero holds As Ks, board Qs Js Ts 3c 4d (Royal Flush). Hero committed 190, opp 200, pot 390, calling costs 10."
    p_rf = bot.Player()
    p_rf.seat = "button"
    p_rf.hole3 = [c2i(c) for c in ["As", "Ks", "2h"]]
    p_rf.hole2 = [c2i(c) for c in ["As", "Ks"]]
    p_rf.my_disc = c2i("2h")
    p_rf.opp_disc = c2i("7c")
    p_rf.hand_id = 200 # Final hand!
    p_rf.total_hands = 200
    req_rf = {
        "street": "river",
        "board": ["Qs", "Js", "Ts", "3c", "4d"],
        "pot": 390,
        "to_call": 10,
        "my_bet": 190,
        "opp_bet": 200,
        "my_stack": 10,
        "opp_stack": 0,
        "legal_actions": ["fold", "call"],
        "history": [],
        "time_left_ms": 20000
    }
    action = p_rf.act(req_rf)
    print(f"Action on Royal Flush: {action}")
    assert action.get("action") == "call", f"Expected call, got {action}"
    print("4. Critical review failure case PASSED: Bot CALLS with Royal Flush!")

    # 5. Verify Draw Synergy in Discard
    # Hero holds Ah Kh 9c, board Qh Jh 2d
    # Discarding 9c leaves Ah Kh (nut flush draw + gutshot straight draw)
    # Discarding Kh leaves Ah 9c
    # Discarding Ah leaves Kh 9c
    req_disc = {
        "hole_cards": ["Ah", "Kh", "9c"],
        "board": ["Qh", "Jh", "2d"],
        "time_left_ms": 25000
    }
    p_disc = bot.Player()
    p_disc.seat = "button"
    disc_card = p_disc.discard(req_disc)
    print(f"Discard chosen for [Ah, Kh, 9c] on [Qh, Jh, 2d]: {disc_card}")
    assert disc_card == "9c", f"Expected 9c discarded, got {disc_card}"
    print("5. Draw synergy discard verified.")

    print("\nALL v11 VERIFICATION TESTS PASSED SUCCESSFULLY!")

if __name__ == "__main__":
    test_v11()
