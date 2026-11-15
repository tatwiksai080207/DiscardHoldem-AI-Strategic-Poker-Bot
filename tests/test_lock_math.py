import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "bot"))
import bot

def fresh(cum_net, total=400, cap=400):
    t = bot.TRACKER
    t.reset()
    t.total_hands = total
    t.cap_c = cap
    t.cum_net = cum_net
    return t

def check(cond, msg):
    if not cond:
        print("FAIL:", msg)
        sys.exit(1)

# 1. helper dominates the true worst-case fold-everything bleed (H + ceil(H/2))
for H in range(0, 1201):
    check(bot.worst_case_fold_bleed(H) >= H + (H + 1) // 2, f"bleed helper too small at H={H}")

# 2. For every H in 1..24: the smallest cum_net that triggers each lock must still win the match
#    even if we then lose the maximum possible blinds in every remaining hand.
for H in range(1, 25):
    hand_id = 400 - H + 1
    worst = H + (H + 1) // 2
    lo_mode = None
    for cn in range(-50, 200):
        if fresh(cn).get_match_mode(hand_id) == "LOCK_MATCH_WIN":
            lo_mode = cn
            break
    check(lo_mode is not None, f"LOCK_MATCH_WIN never triggers at H={H}")
    check(lo_mode - worst > 0, f"LOCK_MATCH_WIN at H={H} cum_net={lo_mode} can still lose the match")
    lo_lock = None
    for cn in range(-50, 200):
        if fresh(cn).is_mathematical_lock(hand_id, "preflop", 1):
            lo_lock = cn
            break
    check(lo_lock is not None, f"narrow lock never triggers at H={H}")
    check(lo_lock - worst > 0, f"narrow lock at H={H} cum_net={lo_lock} can still lose the match")

# 3. Pinned examples (hand_id 377 => exactly 24 hands remaining of 400)
check(fresh(28).get_match_mode(377) == "NORMAL", "cum_net=28 at H=24 must now be NORMAL")
check(fresh(38).get_match_mode(377) == "NORMAL", "cum_net=38 at H=24 must be NORMAL")
check(fresh(39).get_match_mode(377) == "LOCK_MATCH_WIN", "cum_net=39 at H=24 must lock")
t = fresh(39)
check(t.is_mathematical_lock(377, "preflop", 1) is True and t.match_locked, "narrow lock at +39 must trigger and be sticky")
check(fresh(38).is_mathematical_lock(377, "preflop", 1) is False, "narrow lock at +38 must NOT trigger")

# 4. Unchanged behaviour: full-cap lock, far-from-end play, and the late risk cap
t = fresh(430)
check(t.is_mathematical_lock(377, "preflop", 1) is True and t.match_locked, "full-cap lock must be unchanged")
check(fresh(60).get_match_mode(300) == "NORMAL", "far from the end nothing may lock")
check(fresh(100).get_late_match_risk_cap(377) == 66, "late risk cap must be unchanged")

print("test_lock_math: ALL PASS")
