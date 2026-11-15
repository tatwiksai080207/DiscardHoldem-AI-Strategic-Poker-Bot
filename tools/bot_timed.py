import os, sys, time, atexit
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "bot"))
import bot

_times = []
def _wrap(name, fn):
    def inner(req):
        t0 = time.perf_counter()
        r = fn(req)
        _times.append(((time.perf_counter() - t0) * 1000.0, name, req.get("street", "discard")))
        return r
    return inner

bot.choose_action = _wrap("action", bot.choose_action)
bot.choose_discard = _wrap("discard", bot.choose_discard)

def _dump():
    if not _times:
        return
    ms = sorted(t[0] for t in _times)
    p = lambda q: ms[min(len(ms) - 1, int(q * len(ms)))]
    worst = sorted(_times, reverse=True)[:5]
    with open(os.path.join(HERE, "..", "audit", "latency_report.txt"), "a") as f:
        f.write(f"n={len(ms)} p50={p(0.50):.2f}ms p99={p(0.99):.2f}ms max={ms[-1]:.2f}ms worst5={worst}\n")

atexit.register(_dump)
bot.main()
