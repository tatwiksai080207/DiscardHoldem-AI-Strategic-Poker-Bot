import os
import sys
import time
import random
import traceback
import numpy as np

# Import canonical bot
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "bot"))
import bot

R = "23456789TJQKA"
S_SUITS = "shdc"
DECK = [r + s for r in R for s in S_SUITS]

def test_startup_and_memory():
    print("Testing startup and memory...")
    assert hasattr(bot, "TABLE"), "TABLE missing"
    assert hasattr(bot, "PF"), "PF missing"
    assert bot.TABLE.shape[0] == 4922640, f"Unexpected TABLE size: {bot.TABLE.shape}"
    assert bot.PF.shape[0] == 10985, f"Unexpected PF size: {bot.PF.shape}"
    print(f"  [PASS] Evaluator structures verified (TABLE: {bot.TABLE.shape}, PF: {bot.PF.shape})")

def test_lock_math():
    print("Testing Rule 8.2 lock math...")
    # Verify lock threshold logic
    # Lead > 1.5 * remaining hands in final 28 hands
    dummy_msg = {'hand_id': 1, 'seat': 'button', 'hole_cards': ['Ah', 'Kh', 'Qh']}
    
    # Hand 380 -> 20 remaining. max_blind_loss = 30.
    # cum_net = 50 > 30 + 4 -> True
    bot.S.reset(dummy_msg)
    bot.S.total = 400
    bot.S.match_locked = False
    bot.S.cum_net = 50
    assert bot.S.is_mathematical_lock(380) == True
    
    # cum_net = 32 < 30 + 4 -> False
    bot.S.reset(dummy_msg)
    bot.S.total = 400
    bot.S.match_locked = False
    bot.S.cum_net = 32
    assert bot.S.is_mathematical_lock(380) == False
    
    # Hand 350 -> 50 remaining > 28 -> False
    bot.S.reset(dummy_msg)
    bot.S.total = 400
    bot.S.match_locked = False
    bot.S.cum_net = 100
    assert bot.S.is_mathematical_lock(350) == False
    
    # Final 12 hands: Hand 390 -> 10 remaining. max_blind_loss = 15.
    # cum_net = 16 > 15 -> True
    bot.S.reset(dummy_msg)
    bot.S.total = 400
    bot.S.match_locked = False
    bot.S.cum_net = 16
    assert bot.S.is_mathematical_lock(390) == True
    
    # cum_net = -10 -> False
    bot.S.reset(dummy_msg)
    bot.S.total = 400
    bot.S.match_locked = False
    bot.S.cum_net = -10
    assert bot.S.is_mathematical_lock(390) == False
    
    print("  [PASS] is_mathematical_lock verified")

def test_fuzz_actions():
    print("Testing 5,000 randomized protocol action states...")
    rng = random.Random(42)
    start_time = time.time()
    
    for i in range(5000):
        # Reset state
        hand_id = rng.randint(1, 400)
        cards = rng.sample(DECK, 9)
        dummy_msg = {'hand_id': hand_id, 'seat': rng.choice(['button', 'big_blind']), 'hole_cards': cards[:3]}
        bot.S.reset(dummy_msg)
        bot.S.total = rng.choice([400, 1000])
        bot.S.cap_c = 400 if bot.S.total == 400 else 1000
        bot.S.cum_net = rng.randint(-1500, 1500)
        bot.S.match_locked = rng.random() < 0.15
        
        street = rng.choice(["preflop", "flop", "turn", "river"])
        bot.S.original = np.array([bot.CID[c] for c in cards[:3]], dtype=np.int64)
        
        if street in ("preflop", "flop"):
            bot.S.hole = np.array([bot.CID[c] for c in cards[:3]], dtype=np.int64)
            board = cards[3:6] if street == "flop" else []
            bot.S.flop = [bot.CID[c] for c in cards[3:6]] if street == "flop" else []
        elif street == "turn":
            bot.S.hole = np.array([bot.CID[c] for c in cards[:2]], dtype=np.int64)
            bot.S.discard = bot.CID[cards[2]]
            bot.S.opp_discard = bot.CID[cards[8]]
            board = cards[3:7]
            bot.S.flop = [bot.CID[c] for c in cards[3:6]]
        else: # river
            bot.S.hole = np.array([bot.CID[c] for c in cards[:2]], dtype=np.int64)
            bot.S.discard = bot.CID[cards[2]]
            bot.S.opp_discard = bot.CID[cards[8]]
            board = cards[3:8]
            bot.S.flop = [bot.CID[c] for c in cards[3:6]]
        
        pot = rng.randint(4, 300)
        to_call = rng.choice([0, 0, 2, 4, 10, 25, 75, 150])
        my_bet = rng.randint(1, 100)
        opp_bet = my_bet + to_call
        raises_left = rng.choice([0, 1, 2, 3, 4])
        min_raise = opp_bet + rng.choice([2, 4, 10])
        max_raise = min(400, min_raise + rng.choice([0, 5, 20, 100]))
        if min_raise > max_raise:
            min_raise = max_raise
            
        legal_actions = []
        if to_call == 0:
            legal_actions.append("check")
        else:
            legal_actions.extend(["fold", "call"])
        if raises_left > 0 and max_raise >= min_raise:
            legal_actions.append("raise")
            
        my_stack = 200 - my_bet
        opp_stack = 200 - opp_bet
        
        req = {
            "type": "action",
            "hand_id": hand_id,
            "street": street,
            "pot": pot,
            "my_bet": my_bet,
            "opp_bet": opp_bet,
            "to_call": to_call,
            "my_stack": my_stack,
            "opp_stack": opp_stack,
            "raises_left": raises_left,
            "min_raise": min_raise,
            "max_raise": max_raise,
            "legal_actions": legal_actions,
            "board": board,
            "time_bank_ms": rng.randint(5000, 60000)
        }
        
        raw_reply = bot.choose(req)
        reply = bot.sanitize(raw_reply, req)
        atype = reply["action"]
        
        # Invariants:
        assert atype in legal_actions, f"Illegal action {atype} not in {legal_actions}"
        if to_call == 0:
            assert atype in ("check", "raise"), f"Folded when checking is free! reply: {reply}"
        if raises_left == 0:
            assert atype in ("fold", "call", "check"), f"Raised when raises_left == 0! reply: {reply}"
        if atype == "raise":
            assert raises_left > 0, f"Raised with no raises left: {reply}"
            assert min_raise <= reply["amount"] <= max_raise, f"Raise amount {reply['amount']} outside [{min_raise}, {max_raise}]"
        if atype == "call":
            assert to_call > 0, f"Called when to_call == 0! reply: {reply}"
        if bot.S.match_locked:
            if to_call == 0:
                assert atype == "check", f"Match locked but did not check on free action: {reply}"
            else:
                assert atype == "fold", f"Match locked but did not fold facing bet: {reply}"
                
    elapsed = time.time() - start_time
    print(f"  [PASS] 5,000 action states passed all invariants in {elapsed:.2f}s ({elapsed/5000*1000:.3f} ms/action)")

def test_fuzz_discards():
    print("Testing 2,000 randomized discard states...")
    rng = random.Random(99)
    start_time = time.time()
    
    for i in range(2000):
        hand_id = i + 1
        cards = rng.sample(DECK, 6)
        dummy_msg = {'hand_id': hand_id, 'seat': 'button', 'hole_cards': cards[:3]}
        bot.S.reset(dummy_msg)
        bot.S.total = 400
        bot.S.flop = [bot.CID[c] for c in cards[3:6]]
        
        req = {
            "type": "discard",
            "hand_id": hand_id,
            "hole_cards": list(cards[:3]),
            "board": list(cards[3:6]),
            "pot": rng.randint(4, 100),
            "time_bank_ms": rng.randint(5000, 60000)
        }
        
        reply = bot.discard(req)
        assert reply["type"] == "discard", f"Non-discard reply: {reply}"
        assert reply["card"] in cards[:3], f"Discarded card {reply['card']} not in hole {cards[:3]}"
        
    elapsed = time.time() - start_time
    print(f"  [PASS] 2,000 discard states passed in {elapsed:.2f}s ({elapsed/2000*1000:.3f} ms/discard)")

if __name__ == "__main__":
    test_startup_and_memory()
    test_lock_math()
    test_fuzz_actions()
    test_fuzz_discards()
    print("\nALL V9 HARDENING TESTS COMPLETED SUCCESSFULLY!")
