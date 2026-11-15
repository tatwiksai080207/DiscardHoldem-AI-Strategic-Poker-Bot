#!/usr/bin/env python3
"""Pokerbots: Discard Hold'em bot.

Single-file bot: exact table-driven hand evaluator, offline-computed 3-card
preflop equity tables, range-weighted Monte Carlo postflop equity with
discard inference, adaptive opponent model and EV-driven bet selection.
All protocol output goes to stdout; diagnostics go to stderr only.
"""
import sys
import json
import math
import random
import time
import zlib
import base64
import array

_T0 = time.perf_counter()  # monotonic

# ============================================================ evaluator
# Cards are ints r*4+s (r: 0=2 .. 12=A, s: 0..3 = s h d c).
RANKS = "23456789TJQKA"
SUITS = "shdc"


def c2i(s):
    return RANKS.index(s[0]) * 4 + SUITS.index(s[1])


I2C = [RANKS[c >> 2] + SUITS[c & 3] for c in range(52)]
KEY = [(1 << (3 * (c >> 2))) | (1 << (39 + 3 * (c & 3))) for c in range(52)]
BIT = [1 << (c >> 2) for c in range(52)]
M39 = (1 << 39) - 1

FSUIT = [-1] * 4096
for _k in range(4096):
    for _s in range(4):
        if (_k >> (3 * _s)) & 7 >= 5:
            FSUIT[_k] = _s

STRH = [-1] * 8192
for _m in range(8192):
    for _hi in range(12, 3, -1):
        _w = 31 << (_hi - 4)
        if _m & _w == _w:
            STRH[_m] = _hi
            break
    else:
        if _m & 0b1000000001111 == 0b1000000001111:
            STRH[_m] = 3

# number of ranks that would complete a straight (only when no straight yet)
SOUTS = [0] * 8192
for _m in range(8192):
    if STRH[_m] < 0:
        n = 0
        for _x in range(13):
            if not (_m >> _x) & 1 and STRH[_m | (1 << _x)] >= 0:
                n += 1
        SOUTS[_m] = n


def _enc(cat, lst):
    v = cat
    for r in lst:
        v = (v << 4) | r
    return v << (4 * (5 - len(lst)))


FV = [0] * 8192
for _m in range(8192):
    if bin(_m).count("1") >= 5:
        _sh = STRH[_m]
        if _sh >= 0:
            FV[_m] = _enc(8, [_sh])
        else:
            FV[_m] = _enc(5, [r for r in range(12, -1, -1) if (_m >> r) & 1][:5])


def _nonflush(k):
    quads = []
    trips = []
    pairs = []
    present = []
    mask = 0
    for r in range(12, -1, -1):
        c = (k >> (3 * r)) & 7
        if c:
            mask |= 1 << r
            present.append(r)
            if c >= 4:
                quads.append(r)
            elif c == 3:
                trips.append(r)
            elif c == 2:
                pairs.append(r)
    if quads:
        q = quads[0]
        return _enc(7, [q] + [r for r in present if r != q][:1])
    if trips and (len(trips) > 1 or pairs):
        t = trips[0]
        return _enc(6, [t, max(trips[1:] + pairs)])
    sh = STRH[mask]
    if sh >= 0:
        return _enc(4, [sh])
    if trips:
        t = trips[0]
        return _enc(3, [t] + [r for r in present if r != t][:2])
    if len(pairs) >= 2:
        p1, p2 = pairs[0], pairs[1]
        return _enc(2, [p1, p2] + [r for r in present if r != p1 and r != p2][:1])
    if pairs:
        p = pairs[0]
        return _enc(1, [p] + [r for r in present if r != p][:3])
    return _enc(0, present[:5])


class _RankTable(dict):
    def __missing__(self, k):
        v = _nonflush(k)
        self[k] = v
        return v


RT = _RankTable()


def evaluate(cards):
    k = 0
    for c in cards:
        k += KEY[c]
    s = FSUIT[k >> 39]
    if s < 0:
        return RT[k & M39]
    m = 0
    for c in cards:
        if c & 3 == s:
            m |= BIT[c]
    return FV[m]


def ev_key(k, cards):
    """Evaluate given precomputed key sum of the same cards."""
    s = FSUIT[k >> 39]
    if s < 0:
        return RT[k & M39]
    m = 0
    for c in cards:
        if c & 3 == s:
            m |= BIT[c]
    return FV[m]


def _fill_rank_table(sizes=(7,), deadline=None):
    import time as _t
    out = []

    def rec(r, left, k):
        if left == 0:
            out.append(k)
            return
        if r < 0:
            return
        for c in range(min(4, left), -1, -1):
            rec(r - 1, left - c, k + c * (1 << (3 * r)))

    for n in sizes:
        out.clear()
        rec(12, n, 0)
        for i, k in enumerate(out):
            if k not in RT:
                RT[k] = _nonflush(k)
            if deadline is not None and (i & 1023) == 0 and _t.perf_counter() > deadline:
                return


# made-hand + draw strength proxy in [0,1]; used for ranges and opponent discards
_CAT_BASE = (0.10, 0.38, 0.66, 0.78, 0.86, 0.90, 0.95, 0.99, 1.0)
_CAT_SPAN = (0.12, 0.25, 0.12, 0.08, 0.04, 0.04, 0.03, 0.0, 0.0)


def hscore(a, b, board, bk, nboard):
    k = bk + KEY[a] + KEY[b]
    sk = k >> 39
    s = FSUIT[sk]
    if s < 0:
        v = RT[k & M39]
    else:
        m = BIT[a] if a & 3 == s else 0
        if b & 3 == s:
            m |= BIT[b]
        for c in board:
            if c & 3 == s:
                m |= BIT[c]
        v = FV[m]
    cat = v >> 20
    sc = _CAT_BASE[cat] + _CAT_SPAN[cat] * ((v >> 16) & 15) / 12.0
    ra = a >> 2
    rb = b >> 2
    sc += 0.02 * (ra if ra > rb else rb) / 12.0
    if nboard < 5 and cat < 5:
        fl = nboard == 3
        sa = a & 3
        sb = b & 3
        if (sk >> (3 * sa)) & 7 == 4 or (sb != sa and (sk >> (3 * sb)) & 7 == 4):
            sc += 0.16 if fl else 0.09
        elif fl and sa == sb and (sk >> (3 * sa)) & 7 == 3:
            sc += 0.03
        if cat < 4:
            mask = BIT[a] | BIT[b]
            for c in board:
                mask |= BIT[c]
            o = SOUTS[mask]
            if o >= 2:
                sc += 0.12 if fl else 0.07
            elif o == 1:
                sc += 0.06 if fl else 0.035
    return sc


def best_keep(h3, flop, fk):
    """Heuristic keep for a 3-card hand on a flop: returns (score, discard_index)."""
    x, y, z = h3
    s0 = hscore(y, z, flop, fk, 3)
    s1 = hscore(x, z, flop, fk, 3)
    s2 = hscore(x, y, flop, fk, 3)
    if s0 >= s1 and s0 >= s2:
        return s0, 0
    if s1 >= s2:
        return s1, 1
    return s2, 2


def combo_index(a, b, c):
    # a,b,c distinct ints -> index into 22100 combos
    if a > b:
        a, b = b, a
    if b > c:
        b, c = c, b
    if a > b:
        a, b = b, a
    return c * (c - 1) * (c - 2) // 6 + b * (b - 1) // 2 + a


# Preflop equity of every 3-card combo vs the top 100/50/25/12% of 3-card hands
# (x10000, uint16 little-endian, zlib+base64). Generated by dev/tools/gen_pf.py
# with 16000 deals per suit-isomorphism class: both players discard on the flop
# with best_keep, showdown with 2 hole + 5 board cards, ties count 1/2.
_PF_BLOB = (
    "eNpcfQucT9X6vgqdMt/v3mvty3cu6DjJoXLLJRHpIro5U/qkiOOMJB0kl3IZlUslKblfmoSfZCS3zsl8MilyJxVDIxElckmTxvyF"
    "qf963veZc+v97GZ/91rr/a691vs+z7PWwryU9pLYBm+D94hcj/zrDlfCT/hnvDPyqaIUT/Splh5MPMKSDSw98686Cf9goqLmwcTB"
    "RF2/rr/Mw7XMa5B4UezlRGaiS1qXtD7mZfcT933MUf9leZaZwP3Rf7XClZnQshcT6qGP6ZKmPo76eo92uD/qv2neNLc7m+5shDN4"
    "0G+v66NFxbe8LK3hrYF4hB/1im/vI17eFD8j5MInfB5Bz7O9ZZ56hen7VLzNy85HxbfAm751A/mWPuzfm+zfdH7Pm/ye6c7z7fy2"
    "Efy22d5s717/XvcW98qb5CXeEFuYGJeYnzY/rb9Z6H7ivr/pIvf4hPsuZnzyDrFGyVFiVZPNky0SLRLDko8ny9PK0zraqu4zng1z"
    "n/Gso+0mz/B0mNR5PIln3WxFD3CNS+j3vJHQ3vQ389O0P12M3qMPuO9imifV3+P0d0dSezWKvepoWyS0X+Vp2q9uVp+hD3iGPuBZ"
    "N7vO2T+cLXC2ytmlwaXBWXvWLuMI4Sd6UTEKC6VH6GGe9BJ9055idPobfCu+53F+D76xqvREe9vNoqfjpbfa347SN+0x3gY9xuh0"
    "lJ6tk76tkgs9w/UP9noVe32WvY6MxhB6rr2uGDf87GIq3gA91RnPkzfQXqMHj7MHOoNVOYPDXH90/qpy/oZJDzUSxjMSGsmbdOR4"
    "ruN4LrA6npcG/+D7rOL7nOX7ROYffKNVfKOzfKPIRGawP9hPkyvN/zXROgnbnbgsWZBWkPai2Z0oSMP9i+YSuccn3F9i+iYPi+1M"
    "vimWk/xrclhiWGJj8qVkbiI38ZTNcZ/xbKP7jGdP2XnyDE83Sp2Xkng2z2Z6PcT+6D0pVpSc48HqezW94cnhybPJO72ZyZnJwV51"
    "r2WyZXJ3UN89R9lZ9xRlg91zlO0OdkoZSlGG0sHSprqHsp1Bxdviuiyp79Q6qW/+oilI03e/xOg93hf3l5i/JrXvL7Hvh5M6Am9y"
    "BJ6ywxI6Brkcg3lWn+F9czkKeDbP1vS0j3eyj9XZxx6ejsSTHIk5HIndwfCkjsVMjkVLjsXOQMvwvjM5Gi05GijbGfzubKGz750d"
    "c9YsbBbWcNbIWW1nGAmd/cE+3rxilvHzEoNRwdi8KOOho4PZf9HgTfFuL/Hd8JY58vY6QvMsRqevjJCO0VMyHjpKuRwlzP5TFm+G"
    "d7iT71Cd74C3qi9vrSPVkiO1M8AoZcpI6VjN4VjtlvHQ0ZrJ0WrJ0cLs75bR+F3G45hcGI1mMh615VrIETvGEUMpymtzxB6waRyt"
    "wRyxijjBz0tMxehhlDSbfk1g9HTE8PYv8e01O3KYHRvdWAzjKOZyFDE6mmV9mWU7ZRR1xPCWd/Itq/MtNSvqezM5li05ljuD4RzN"
    "mRzNlhxNjIpmXyaz74/eHI4nRnM34+d3xs/3gcZPM8ZPo3Ahx/QYx1THsxnHs5EbMW3/O9ujho5qbY7qA/YBm+fn+S3kauHXcREC"
    "m5WslWyTaJNYZGa5n7hfZOrLPT7hvr7Z4cZJ7YLYr260FjoMrutV9TKTmcmT9lf3Gc/qus94dtKukGd4WlfqVPXwbIV93tsh9r53"
    "sQ9b7s0X6+l95r2SfCU52fvE25bc5uJsi7ckuSRZK+zpnqNssnuOsjnuOcpqhW8HPaXNZ95kafOJN0fabPFQ9nZQ04/Ezno9fFgl"
    "P1vsgNfYh5333vTw38/eAS/bWZlXx9+R3JFM8wf5aV6a1zg670rflBrZUqfM25FEHRedHuo0jraG58XHm1IHtVAHtdLExyAfdbaG"
    "FSOPq1ZSx/evSZ2FRaZNQuehvtF7jD3u65vDSR3Hqp6O48bkDs6EzsZJuzCh85GZ1PlYYfUZxj4zqTOCZyvsZ56O1yccry0crx2e"
    "zsrFvs7KfE9npVb4SlLnZRvnZQnn5e1AyzD22zgzSzgzKHs7eNPTcTnAcanj67gM4rhEvs5OD85ONmenccXsRJ6n85PN+dmR1Plx"
    "rWV+toZaB2OfzRlCHRl7T2cIdbaGo6PRURtndZ01dpbtrI+zsc4ecvaIs7XOMDuaHXk+ZqMiC2bJzGCm6shsYY50xpAdiwxGH+ON"
    "0cd4Y+R/lRnRWVthN3LeLnDeTsoc6cxlJnXmkB0nLUYb4/oJx3ULxxUj3VNmQmdvCWfv7QAz97zMns4fZm65zB7mSGdwG2dwCWcQ"
    "2VErxOielznS8cPoYvwGcfwwsudlBnQWMbIol9iXGcAM1pRZ1HnM5jw2rpjHCHOkM5nNmUQUYCYRBZhJZEdjmaHRMkeN5cIMZcsc"
    "PSQXZggXaqFeY85mH87mQ5zNtZzNU0ELzmQeZ7Mir/CzvqmYWcygImEdmVmdTcyMYltdT5Ht16QiW103T4prvyYV1+rKzCk+7iBC"
    "bpQZ1tnEDHzCGdjCGVBE60lEm0xEm+Pm5RXO9DbO9BLONGZMkfN5Iuf7RM7lMtM6mxjpAxxpxbIybxBHWpHsPJHsZyJZmadIlubm"
    "weOMZ3PGFcdQQ2ccM6WIWtPvwTnP5pw3rphzyTDNt9HMt7rMt2zm21jm2yOc89Gc87qc82zO+VjO+SNuNttw5htz5vtw5h/izK/l"
    "zJ8KTgVr/bX+aQ/XacfU08RGJycllyeWJz40o91P3H9oWso9PuG+pQm8a8Q6ekvESpILktc5G+B19uDDBiXu0wJ5Mi2JZzZYbkuk"
    "zgJ5hqd4ttyu9GIfdtCr5cNu9duIFXgbvdjZR16p19ZZZ7+DP9Qb6rUKC9zTjVLSVso6u+coaxUeDwqkzUYpQ2lnadPBR5kzb4oP"
    "u8Z/Rmy8P1zsDX+l2E/eIa+LsxbuuwY7PTLFn+Cv8lZ5ef4ov8Qr8XpEP7nSQ1JjsIc6U1w56uS5ctTpEWVJHdRqIT46+FPExwQ/"
    "T3yM8lEnK/rYPyD2vl/ZwM761Q2slblS7Bf/hFgff5uP/973n/TrOfvaH+3f78yYIt9zdt4v86s5ax33cbW2Sc16UvdrV2u01PT8"
    "IqlZTeq2jnOkLmq/Lz6flLqojbqofV58lkndnLgiSnBNSmosTEtqxHxolic0ZloavUec4L6lWZDUOe/s6Zxf42nkLGHk2OC6pMbO"
    "NMbOcqvPECfTGD14ttxu9HRuSzm3HXyd29jXCKrFCGrDCGoVxp7GUFvG0FDG0PFAyxAnbRlFQxlFKDseHPJ0Djv4OocTOIejOIdT"
    "fI2kZxhJwxlJKxlJPaIunsbSYMbSKsZSCWMpK9I6iJPBjKZVjKYSRhPqZEXbfJ2rJzlXozlXRZyrMs7VAV8jqrLRiKrOiLqSEXXC"
    "14hqHfuMqXqMqfsZUx5jqhpjKifWuoiTeoyq+xlVHqOqGqMKdXPifGeraSucdXJW5OwDZ6872+lso7NiZ1OdIaIUfdb6iKAKlBkt"
    "0YSIGS4RhrjSKAP6fGgQMSUSVxojiJYSiSKNtOUWURZIpGms2QBxpdE2jdEG9LEBIqRA4kpjARGCWEB0FEj0aMQNZcQdDxBtiLla"
    "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"
    "Xy3k/tVi7l+NoN7rQL33BPXeCOq9rdR7E6j3VlLvPU+9N5d6bw713kLqvcXUeyOcIiuk6ptK1bebqm8xVV8RVV8BVd9iqr79VH0F"
    "VH1PU/VNp+rLpuorr1Feo081tU+SnyQfl+vxf93p9UnyWmd69/i/nuhTvd9W7fF/lXzyr5KKa1u1iprbqm2rtiG5IdlUrqbJHmlL"
    "xdISH6bNqTanWiMvLTGnGu4bebfLPT7h/navohWuD9O0bGmaemjkzammPm739B7tcH+718FZSbIkOd4b713iDB702zck0aLiW/Dz"
    "dg/eeohH+FGv+PZG4gUXvODCJ3zGJ3z+Oq0pvW6g54p+4OftXsW3wJu+dQ/5lkbsXwf2bzy/pwO/Z7zzrOUdWI4nX6dFXuQ9mcT1"
    "ZPLSRHOxrWmXJAZVG1TtYW9r2qBquH/Yy5J7fMJ9ljc00VksM5EjVilxVWJE2oi0BxOpxEfVPqp22KvkPuPZg+4znh32+sozPH1Q"
    "6qQSeNb3Xz3AdUlCv6d5QnvzsDeomvYny9N79AH3Wd5VCfWXor/OCe1VDnt12BuRpv36qJr2q6+nz9AHPEMf8AzXYe8dZ9872+Qs"
    "4Sf8Rd4i6ZWOTuShFxWjgJ9ZHnp4qfQSfdOeYnQe9vCt+J4UvwffWEl6or3t66GnQ6W32t/D0jftMd4GPe77r54dlr5tkgs9w/UO"
    "yzax14vY64UJ9Fd7rr2uGDf8zPIq3gA91Rm/VN5Ae40epNgDncFKnMEHXX90/ipx/h6UHmokDGUkZMqbVIznYY7n956OZ8J/h++z"
    "ie+ziO+zMPEO32gT32gR32hhYmHi9+TvyeVyLU9elPgtDVYpcTqtgbMcF1cN0k7LXUO5xyfcN/SeT+SK3ZV4QGxU4onEmrQ1aX1c"
    "f/PT8tPS/VHuM571cZ/xLN3f5Y2SOk8k+kidoQk82+VtT/RJwhon9b9tiR1iLyeqJ1skWiR+S5xOGGe+K1uftj4t23/ZPUXZb+7p"
    "aSlZn4aybP8eKUPpb9LmtJShFGX3+BVvi+t0mr7Tb2n65jleA757Q0/v8b64b+g9kdC+D2XfcxM6Ag9wBNL9NWk6Bvkcg12ePsP7"
    "5nMU8GyXVz2pfTyd0D4mk9rHPkkdiSRHYkdCRyLbb8GxMByL9Wk6Fvf4Wob3NRwNlOF9UXaPn+/sY2f4NyiXOCtz9qazImcHnGEk"
    "dPZ/T+LNK2YZPxt6v3FscmQ8dHQw+zke3hTvNpTvhrccJW+vI7TLw+g8LyOkY5TuYzx0lPI5Spj9dB9v9rKMh74D3gzvgLfCc8OR"
    "wlv58tYYJYxVkmOFUdomI4Xx0NEyHC2MLEYLs58to5Ev47FELoxGmYzHAbk+5ogt4Yi9yRE7wBGbKmOlo6YjVhEn+NnQqxg9jJJm"
    "00UJjJ6OGN5+KN9es2MUs6OPG4s1HMV8jiJGR7PseWbZXTKKOmJ4S82L3xKaF35Ss+JlZsVvCc0K37275sTLzInfEpoTvoyKZt/2"
    "RJLjqdm3TUYzm/GTz/g5xvgpY/wUcTzzOZ7HOJ5lHM8iN2Ifc1SXcFTf5Kge4KhOTU5NYjG5LIlrWfK3tLYJ2IBEo0TNtJppud4A"
    "9xP3uY6PB8izRolcYedBiYlinyZeEstz4zs+bXza54nNiUVpi9JOe3nuM5597j7j2WlvpzzD08+lzuYEnu30GiZbi+1OqF4qTzRP"
    "wjKSjyavSVyTuC05PDk2MdaNY8dkt0S3xA4/wz1H2W3uOcqqu+co2+FPkjKU3iZthierS5uOSZRN8u9LPiXWJvmq2P3Jx8QOJEYl"
    "YfWStydnJWYlKidzk2fSzrg5HJRsnWiduCLZPpmdyE70N/VcOepUduWo47ty1LnClaNOf9NG6qBWZfGRm/TFx6DkFeKjfRJ12piK"
    "kcfVKKHj2zahs5Dr1UzTeWjk6T3GHveNnFbRcdzMcZyY0Nl4ibNx2hufpvOxiPOx09NnGPtFnBE82+k9mtTxGs7x6sjxap3UWXmc"
    "s9Kcs7LDvyah8zKW89KN8zLJ1zKM/VjOTDfODMom+bcndVxyOS6DOC7tOS5PJXV2XuXsPMbZGcXZ6W9mJXR+zqTp/LRO6Pxkc37a"
    "GK2DsUcdjD3qYOyzOUOo08YscbbNGf4l3WJn+C15zZyNc7bQGf49WZQtY4bgJ2ajIgtkPjzMFOYrV+ZIZwzZketh9DHemzneGPk8"
    "mRGdtZ0eZmyQzJrO22mZI525RZw5ZMdpD6ONcR3Oce3IccVIZ8hM6Ox14+xN8jFzDWX2dP6ac/52yBzpDI7lDHbjDCI7dvgYXYxf"
    "LsdvEMevPccPI1tPZkBnESPrywzoLLYxmMH7ZBZ1Hh/jPI7iPPaXOdKZRBRgJhEFmMlsziSyo7/M0BKZo2K5MEPnZY4WyoUZwrWN"
    "s1nM2WzG2VzI2fyJs1ki86gzqrNZkVcyc17FzGIGFQl/S8PM6mxiZjZzZhTZ8ohsn7t5Gs8ZXsQZxswpQg4iQn4qM6yziRkYzhno"
    "yBlQRPv/XH0LnI7V2r6ixDzPWs/hfWfMOLNT/CsqfCpCJOOz9+ecQ9mTSoV/OjBIhHJMUiHjnxSi5KySyHlQKqcMObSdtnEqEpPG"
    "yH9d933Nt3e7+/c07zzrXvf7vGvd93Vday0zk05Ea05Eq+Dm5f9wpl/hTD/MmcaMKXLWNr041/U515hpnU2M9Isc6ec50g9wpBXJ"
    "ahLJShHJAiJZZTcPOZxxxbFSxLGAOFZZZkoRtR0R9V4iagcznHOOGX+a9Taf9XaW9XaZ9TYm1Hq7O9rKed/Lea/HeZ/DeT/LeT9n"
    "t3Lm93Lm63Hm53Dmz3Lmz9lz9hVnfxhcf5hfvJ4+7AY3judSzqVMsje4r3g9yXaS1/gOrzvZlf5QsTF+nliuP8GtNL52c1zBYL1Z"
    "N8h13+PeFv+Ah3t1g2vlHu7iHu7i3rVB0s01rKurPtgbbh5glcwtpqHf0H/dvGmwNnjYTDHr/fX+6aCSu4+21919tD3s7qPttGP2"
    "StLnFvO69HnTPCx9ppjTohvam3pib5uFYngFG2/aijU1WWamP9MfZSabZ/1nXQ58amb7s/0x5lp7xj/jvxY2de3wGeXa4VPTtcNn"
    "jDnjw+e18FHxgdcoiTHZ1JQYnxr4wOs1+X3dT5rNYi+62LCPqOJeNyPF+psLYl1cnCn+FH+T+dhl8BX/KzPWZc+P/ism18z35/sH"
    "zXEz0Z/oT426OD/4bnJ+8P3K+cH3FecH34POD75To3biC+9NEvNj85XEHGtekZi55qDEPG7g2y4qzhJcD/uaCz19zZhJ9lyK5kwn"
    "q6+RJ3jdyU7wdc4rGJ3zob5mTh4zp27wtae5c4C5c22g95AnB5g9uHdtcIvRuX2TczuFc/s3oxnUkxn0ADPodNDQ1xyawxxazxxa"
    "G2gb8mQOs2g9swhta4Mso3M4mXP4KefwWqtzWM9oJi1kJr3NTGrLTHotnOlrLj3LXJrNXDrDXHo0VB/kybPMptnMpjPMJvg8Go4y"
    "Olcfc67Gcq5yOVfHOVebjWbUtVYz6qrRjBrJjLrAjJoaTfE1p64wp35kTs1nTk1kTrWL1Bd5coVZ9SOzaj6zaiKzCr7tonPOvNiL"
    "6zvD30BdFa2KbohviAc5u8UZ/j7q+eh8VCuuFZ+NzkbIKEWfVywyqBhlbpBsQnYhxyZJXmmWAX0mWWQMcgQZgxxBtuRKFmmmXRsg"
    "y1ZKpmmu1ZW80mw7wGwD+tQNkCHIhTeZC1OYC8iOSpI9mnHrmXFrA2RbUjJOc+4B5txpySvNujnMuvXMOqDP6QAZgTmfzDn/lHOO"
    "jMCcIxuaStZo5s1m5p1h5j0aIuvaS+Zp7r3N3GvL3HtN8kqz71lm32xm3xlmH9DntRAZgLn9mHM7lnOby7k9zrnF7HeRbNEM/JEZ"
    "OJ8ZOJEZ2C5C9j0pGag5iOz7SDJQc/ACc3Cq5JVm4RVm4Y/MwvnMwonMQqDPVMkqXMiq+pJXN8iFrBokeXU+woWswgVv+LdhJt7A"
    "TLyFmXg+0kw8y0zMDJGDmo2aicU4hq+dbHFWIvuUJX/xkJWaicgq5b0tZL1cst4W/9rga2bnAWYnsk7ZcyXZc4xkp2YisudNZs8U"
    "Zo+yXSWy3etku4ddTjVkls5hlq5nliLblFWTZNWuZNU3JEs1E5Elk5klnzJLlOfGkOWakuVGkeVqkuXGmEfDmczWZ5mts5mtZ5it"
    "yDJl2/Zk27dp9ci24yVbNRORDR8zG8YyG3KZDceZDcpuXchum8huX5HdXiG7HXQ5M4VZe4VZ+yOzdj6zdiKzFtmlLPwkWfhFsvBH"
    "ZOHXycL9JWunEt/ORYpv9WPFt1WR4tsg4lsl4lutWPP2XNSGuXsDc/cW5q7mbSXmba04M9T46KHZewOz9xZmr8avxPjokRnm2Bzb"
    "Sq5WLj9zxOb4H/nPe897ZYM57itelw0etHPk3kc+Xj9ol/qjxY77aQbW3n9fNP05/7IPbXt/0N59j3vn3Pe4d39wh9zD3XPic9nH"
    "vTuCQpehsBFOw6up+m5u6pimflP/A/OEGeAP8O82681If6RvwubuPto+cPfRdre7jzYTngyaS5865gPp84S5W/qsN0b+6loZc7tY"
    "bzNNbKWZLvaBOSxm3HwO8ge5WvnQjPBHuFw5KftIZ0wF+5v/m78nNK4dPq+7dvhscu3wOWN+8+GzxyllIzGeNK9LjA/NJolx0sAH"
    "XntET+8w+8ReNfvFSth8A/vKFIodM+fEJpr/a2b5s/y33Urrshvfg+YB29TVz3STYcub8mapOSVz8Es00fnB921z2YfvQecF3+nO"
    "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"
    "JVxQS7jmUpFtoSL7korsKyqyIioyREd8REd8REf8B6nIylCR1acia01F1rDSNKqxI1RkxftU+Hrqf9UZVNhK6rNLYnr+doDnbx/y"
    "9O0AT98+DCaGc6jScqnSJoaq0qDTCqnToNJUkUFF6bnbxWAkVZSeuv3EU7eLQRVqtSDKoFq7n2qtCtUaVJee7p0K7qFey6Reg1pT"
    "RQa1NJZqaRnVkp635fO07R6eto3nadtInrblOy2lZ2338KxtPM/aRvKsLV/Ulp76HQ3mU7cNpG5bSd2m5+56zvY9z9mW85xtcdiL"
    "qqgtVZGesn3PU7blPGVbHBZQu12mdquaGEn1dpzqbSvVWwHV22WqN6gsPQ0cz9PAXjwN/Jq67V/6DepNFRnUzyWqHz1fy+T52t95"
    "vlaC52uFPF1rzNO1G3m6lsnTtb/zdK0ET9cKo6xkNaq4VlRxPajiplPFLaOKW0QVB3Wlp4S/hO2p43pQx+2kjltCHZdHHQcVp4oM"
    "KqcNVc59VDmvUuWMocrRc7UMnqs1i/VU7VKop2pteao2kKdqu3mq9lnUlFpuAbXcylQ9U7sU6plaW56pDeSZ2m6eqX0WNaWaW0A1"
    "B1Wlp4dD//enPfdQz31KPVdIPaenhz9EA6jnoOZUkUHNNKOaqUs1s5hqZi7VjJ6n/RhnUs3MoJrR07TbeJr2IE/TRvI07XOepr3O"
    "07Qf4z7UdHqatsQpMj1Lu41naQ/yLG0kz9I+51na6zxL+zHuQ1WnZ2mIoKoOuu516ro11HV6qlg6UZ267ibqurup66ZQ10HVqSKD"
    "avmaqmU3VYueo9XhOVqC52i1eY62I3GaqkXP0V5I6ilaP56ireIp2qw4n9ouj9ruKrWdnqLtSMyntjtIbVc3Q8/Q+vEMbRXP0GbF"
    "+VR3eVR3V6nu9AwNkVTdHaS6g4rS00bvT+oO+u4d6rt11Hcnqe/0tHFNogv13avUd1B3qsigTvT8zOP52e08P3uP52cjeH5WkDhN"
    "daLnZwN4frYrqednU1L19GxuXESNp6dnt/P07D2eno3g6VlBIocaT0/PBvD0bFfyKDXemfJ6djY3LqLK07Oz23l29h7Pzkbw7AwR"
    "VeXp2dkAnp3tSh6lyoN60lPISTyFnMlTyM0JPYUszVPIpjyFfI0nbb15CvkPnkKOTg6mzoPKU0UGFaLnZnt5brY80ZkqZARVyAKq"
    "kBNUIb9Thei52XCem61OnUEVoqdm2Tw128tTs+UJPTWrlHyWWm8btV4htZ6empXkqdlwnpqtTi2k1qtbSc/MsnlmtpdnZssTemaG"
    "yKr2tlHtFVLt6ZlZSZ6ZDeeZ2erUQqo9qCYds1t4Otmfp5P1eTp5gaeTm3g6eSX5CvWenk5u4OnkGp5Obkh9kXoPau8F7l9N5/7V"
    "HO5fHamk+1deZd2/mldJ96+ur6z7V8e4f/VrJd2/Ksf9q7KVdf/qaiXdv7q98lxqvi3UfF9S831FzVdEzad67/rKqveOUe/9WulB"
    "ar4y1Hz1qflaU/M15PNP5/PP4fPjHVT1FVH16fNfX1mf/xifH++gqq8MVV99qr7WVH0NKzWstKmMWn2/vp8qV+r/vsJVwdlyb7mH"
    "74pbcUfvausfZVLZUp+t2q7XH2WKPf8o80eZbD/b3+jh2uhtTvmHWCsv4W0tu7XsMr+V+4rXy/wq8hrf4XWV/+2FK+Fp2z9SNMIy"
    "f2tZjVHF19foh9dV/CnOrnpXvVH+KP8JZ4ig757to0fxu7SS3oi2WSIijkbFuy+TKLgQBRe+w/f4Dt8/WRYxNbpGLn4OfK3iF78L"
    "oumn3izvsozPN4XPN4rvM4XvM8pF1vYpbMedJ8s+4z/j/+7h+t1bmOJA21k9L/A2lt1YNs+v577idZ5fR17jO7yu47f2nhNLevXF"
    "xnpDvNYprVOqeS28iikVUw77Y933uFfNfY97h/1jcg93q4lPCw/3jv3vE+AKPH2fJJ8mz99YVp+njq+v8Qx4Xccf4mm8Foz3nKdP"
    "VZ9PddhvnaLPVZHPdczXe3iGinwy3Dvmz3a20dleZzOc4Xe64R6eSkfnGR9PUTwK9eSJ8IR4zjx5Nn1SjE6ej3fF+7Tg++Adx8qT"
    "6NMe8/GkreVp9XkPy7PpE1fkE2N0DsuTzZZnmyEXngzXRj71DD71bD512RQ8rz65PnXxuOFrHb/4E+BJdcYXpuAT6FPjCVrwCXQG"
    "x3IGq7nnac1PUpGfBE+omdCamZCUT3KY4zmb47nX1/GsZTby88zg55nNz1M2ZSM/0Qx+otn8RGVTyqa09dv6N8p1o78lpb0H25nS"
    "zCsoW1D2XX9nSkFZvH7XHySv8R1eD/L/y/tI7GrKWQ82ysvxBqcMThngDfKeSnkq5St/lPse9wa473HvK/8TuYe7A8RnkId7n/in"
    "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"
    "wdG/G+VoMDw4egA5eh85ejI5Guy+V1RClaoxlaovlWo6lQrs3kFUQpWqkEp1OhUqBa16nlo1kVrVhlr1CLVqMbWqB7XqG2pV1TTo"
    "jKrVMKpVY6pVX6rVdKpVhlW1glJCrQqpVph9VE0Do38rOqNcfC25eBi5uAe5OI9c/Da5eOdlXAw2R54lVKxpVKwSKtaSyxRrLRVr"
    "ABXrNBVrfzrUqrsolmrWemrWdGrWHmrWp9Ssg9SsRdSsadSshaIzqlpLqFrTqFolVK0ll6nWWqrWAKrWaaoWZh8L06Ey94vO/JqB"
    "F1QmU3TmL/KCyqSIzpwSpYHKdBadmZOBF1TmOtGZ30RpkA35kA35kA35/kLluofKhWzIh2zIh2zIh2zI91uGKteB9CZUrQNUrrJ5"
    "EL6ONGUqBrXSVbV2omKqXFAhXSc7x1WykVwlO+eUS9fIRnKNDEdUzaBnF6lnslBmdX1sSajrY9dxfayF0dWxJaGujl3H1bEWToN0"
    "bWxJqGtj13FtrIWok67CzeYq3Inwr9Q1qJoqF1RlKVVF18ViXBfrbXRVrICrYs+HuioW46pYb6c5uiZWwDWx50NdE4txTay3qJKu"
    "znXl6txJrs59F+rqXH1RN1UuqIeuh7XmethRroetD8dQPXQ1rCVXw1pzNewoV8PWh7oa1stpjK6FteRaWGuuhR3lWtj6UNfCeoka"
    "6ardk1y1q8RVu/1ctfudP4v8D1E5VS6oxDCqxHaqhK6D9eA62GCugy01ugq2KNRVMMNVsOZcBevBVbDBXAVz7eO6BrYo1DUwwzWw"
    "5lwD68E1sMFcA5MecV3N+5yreev4c9P/w9W8dK7m5XE1b7KonSoX1OAWqsFrVIPpVIODVANd/4rbDlQDXf16jqtfL3D1axBXv7Zw"
    "9etTo6tfcaurX6VOuXTt6zmufb3Ata9BXPvawrWvT42ufcWtrn2hp6oedG8uda8VdW8bdW8nde88dW8PdQ+qp8oF1j9O1td1rye4"
    "7vUy170Gc92rA9e9lnDda6PVVa+9XPXaahpT+/pS+6ZT+3TVqwNXvZZw1Wuj04zbqX7DqH6NqX59qX7TqX665tWBa15LuOa1UVRH"
    "V/9Kufp3yEyk/rWh/j1C/VtM/etB/fuG+gf1U+UCu3cku19Ldh9Gdu9Bds8ju79Ndt95Gbvrate3Zgk1cBo1sIQauOQyDVxLDRxA"
    "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"
    "1Ok1aZoDGltApZ5Hpf6QSn2YSu1ZVerrqdQ1qdTIsSbtNas6lEsdepg6tJg6tJI6NI86dEukOvQWdWgNdSgjUrX+w6paZ0Wq1iHV"
    "uhXV+mWq9QKq9Sqq9QGqdf+MzUb1+kqrep1Gva5BvR5FvR5OvV5Bvd5GvU6LVK+dwkouaOyVVhU7jYpdg4o9ioo9nIq9goq9jYqd"
    "FqliI9fTGaet6k3VSPUm+zK9eYF604d6s4l6k4ir3iSpNy2pNzfGVbUPR6ra98VVtS9GqtqHqdoN4qraN1O1+1C1F1G1p1O1R1Zc"
    "blW3d1G3S6nbV0Sq2/dSt/9C3W5C3X6Zul1I3Y7iqtsLKmpOaOwuKncplfuKSJX7Xir3X6jcTajcL1O5C6ncUVyVGzkXVPwi64us"
    "Qy6OuLCVbKVlWcuyUiulVtqctTnrtIs3XfziYoOLt1yYSqZSiosDWQey9rgoV6lcpTQXF7MuZjWv1LzS6qzVkutVozNkfIVKl82C"
    "V4tig9nWiYpDu1XJMTvOt1Bl6DBUGToMRS4RpVY172d/pZ5DyXNEzaHdquj1QlV0zI4nW6gw9BYqDL09Rb2FAuM4FPgFUWhV9VEW"
    "ig5d70Bd30tdXyvarcreKVRl30JlByrXWqgudBWqe1K0W3UVqgtdheJuEmVWdR9NdV9CdW8abaO+Q9lvEnVXfV9Afb9LtFsVvkao"
    "Cj+aCr+ECo/Z8V0RVBb6eZz6CZWFfq6lfq6hfkJh0e4sVR4KmxRFVpWfTJXfFkHBofNQ+C2i8qrz/6LOt6POV45Du1Xpz1Lp80JV"
    "+glU+slUesyOK8ehqtDJEdTJUuokVBU6uZU6CVWFTkJR0T6i2ren2nen2i+k2h+l2t+cUKXHS/UeSg+9/yf1voB6v4F631m0WxU/"
    "ouK3p+J3p+IvpOIfpeJjdtw5ARWFHuZTD/tTDx+jHp6jHh6nHkJFoYdQ0BxRYFX9HVR9KOg+UWBV/fFU/aFU/ZeSK6j766j7X1H3"
    "D1L3ofjQ/TbU/dbU/Umi3ar8W6n8O6j8xqjy96byj6fyD6XyY3Y8KQnVhO6dpO5tpO5BNaF7Hal7q6l7UM2vRbtV96CY6F9A9Z9H"
    "9f+Q6n+Y6g/FLBDlVfWvSfVfk9aH+l9E/V9P/R9H/Z9J/T9C/d9M/Yca3izqD+1WB1BABzCPDuBDOoDDdACeVQdwPR1ATToAzI4f"
    "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"
    "B6Ye7D/il9SFHYurCztCFwaH2Ez8krqwtnRhH9OFnaELg0McIH5JXRhWf/6TBefRUvySeoTW9AhT6BF60SMsoUdYSY8A51FZ/JJ6"
    "hPtS1SOco0eA62gprkadWHc6sTZ0YqV0YgfpxOA6kLcXndhEOjG4DuTtRSe2rwpcGLxYNXqx2+nF4ML2ihNTL/Y4vdgOejG4MHix"
    "XvRiRfRip+jFvhS/pG4MzhFurDvdWBu6sVK6sYN0Y3COcGO96MYm0o3BOWaJX1I3htWfL6vALWWLXyovL7gldUy58oJbSopf6iQv"
    "uKWe4pfGygtuqYP4pa7yglvqIH4pX17Ijvzl6cjS6chy6cg8OrJOdGSN6MjG0pH1pCPrSkfWgo4sn45s6DXwYurK1JGVrVPha3Nb"
    "5s7gwnS3rqW4M3VkcFcT6a50922mKaVH625H06WV0qXBfekuXnvu4q0Sl6aODC5qAl3USroo3XWbw123T4zuuj3knJXuuc3hntsn"
    "RvfccEbdGvxaffq1Z+jX4NbUkcEtzaBb+oFuSffb9nK3rRN32+7ibls37rbtdc5A99o6ca/tLu61deNe215xELrrd7WtRt/Wg75t"
    "PH0bXJs6MriirXRFNbl+NYCuKJuuSHfZmnGX7VXusqGVerciereW8Zvo3u6meyuge/uZ7q2I7g0uS3cDm3E3sCl3A1/ibuAM7gbO"
    "F/emjgzu51u6H3U9D9q36X6m0f28Rfeju2uzubs2jetK/+vhjtLD6e7ajVGXhO6tzebe2jTurf2viztKF6d7a+ihLg4+7h36uJ30"
    "cbpL+Dl3Cd/nLuE+Cxenjgwu51a6nCvpcv5FlzOWLmc9XY7uqy2JdFdtO3fVPuCu2o82j16uOr1cK3q5Enq5Q0ndU9vOPbUPuKf2"
    "o82jm6tON9eKbq6Ebg6uSncPt3L38BB3D0u4e3iMu4fn7GD6ua/o5+Dm1JH9/25G99OmcT/tGPfTRkRt6GYu0M3oftr6uO6mFXI3"
    "7TPupk3jbtox7qaNiHQ3rSF3097lbtp654V0/ayQe2mfcS9tGvfSjnEvbUSke2kNuZf2LvfS1oub0l3Fn7mr+B13FfdHTejr6tPX"
    "daOvm01fV0JfB1enjgyu5Ue6lt/pWnQfrRb30fpwH+1d7qMtjus+Wir30b5K6C7aT5HuohnuovXnLlot7qL14S7au9xFWxzXXbRU"
    "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"
    "xxEPbdDqotRYDb+q5LjKj0t1lT15vHry+bbkKHzvvtNx6M73ePbd5ekfuuw5ruNo/MLR2OTO6nis5ngU8thMGQkdkUJ5+m34vJ7k"
    "83qRz+s2X0cll6NyDUelnXvqOi4ZHJeKHJeQ5/DsMzgyFTkyoTz9I54+l7t8fS41+Fyu4nN5gaNTnqNTh6NTnqMT94tjOj4dPR2f"
    "/8R0fL7n+CTZBs++o6cjhDZ49t9zhJLy9D/zPvNmuhjiApr/qYtnXeBYgYsUP8Uf7g2X0VF0TJHRKENBKxkZjFQXGa3v5ThGrLuM"
    "0qHLnjfeXZQR0VErlBE7IKOm47ZJxkhHbjVHrlBGqQ2f65N8ri/yuWZz9DI4ehU5eqGM3P/I6On4XcPxaydjpCOYwRGsyBEMZZTw"
    "dPH88HTPyxjp87uKzw9P9oSMgI4inizOf89RTMoI9pRR1HGsw3Esz3GMyxjpSHb0dCRRBRjJ7zmSSRkljNBnMkZF8sIIfSpjVCAv"
    "jBBeaIV2RRzNZzmaBRzN4RzN+SnVOJJTOJpluMLX7v8dWYxgS47t9xKHLhsZZbaLZLaf3SjN5Aiv5ghj5NZxjH/hGG+SaMMReJIj"
    "8CJHIJvjnMFxrshxDv1sjnQGR7oiRxojdhvHOpdjfQ3Hup2EctkJctl5ctlF7yo+aWWyE2Sy82Syi973HO+kX8wRVx47Tx5DCx1x"
    "jNQLHPPyHPM6HPPyHPO4jLni7TPibQjx9inxNtNTvKX4OuafccyHcMw/5ZjP9IZz3OenzOTIF3Hkn+XIF3Dkh3Pk56fMTzntnfZy"
    "5JXjXePVkbjN6+uFsTC2zL0LY3i/zBst7/Ed3o92c4nPJHZ630iM9FZ4H8Q+iH3k1fd3xXY5VRvpvsexj7xdMRzb5PWRYziKYziK"
    "Y33cmMqvPbgn1FLiJo7az57vZ3qZ7vxSv4PXwWvv9/UHeYO8a92ZTDlX1x3HufbuOM5d6548zuFsXemz1G8vffr6OFfVP+X9JvFn"
    "uorn/VoSFfz7JUrdTCThJbxbnCfp6nX1GvmdfDybBq6anvKe8nq4Fglpc4s7jzaN3Hm0aeDOo00P/1sPbdDqFsnR1G8kOTr5DSTH"
    "bT7afOtVct8javttJG7lPVzr2iMOc6b1nRufG1z86P3F3x/b757bWe9uF795z8q9/uLq92DsYGyK/51rdVpa7o+hbX3/bu+stExI"
    "21+8gzG0neKezneS87S0RWu0Revf5N6f9dEWrdG2rl9WJXj1ZS3UYcUsc99pzYzme9TJaKmUFRzz+r6O+WesnG9YOZvcWa2dXayd"
    "PjyGOtnF6sGxPm5MdWyXcmz7cmzTfa2glqygXFbQta5CtIY6sIYGsYaq8hzqpAOraBCrqKpUSjlfx7Apx7ATx/A2juFvnlbSnb5W"
    "Ui1W0v2spB7uaWstdWUt5bCWnmItfetpG9RJV1ZTDqvpKVbTt1JNpz0dq7/4OlZnPR2rZ30dqxocqwa+VlQbVtT9rKhGrKhiTytq"
    "in8Da2p/TGvqbk9rKsGaOhjTmqrra1vUCdqiTu72tKoSrCq0RZ3UlUoZ56KFi1dc9HMxxkU1FwUucC8TXPR20dcFzqOilH1OSwWV"
    "scxtUk2ormukwpbJcVTZaKmsFawRVAxqBNUyUqpIK62PVNlEqTSttU1SV1ptu1htYJ9NUiE/S11pLfRlLWSy4jqw4gax4qpKtdWV"
    "itOay2XNXSt1pVXXgVU3iFVXVSqrHMe8Kce8E8f8No45qgHnu7Lyclh5T7HyvpWqOyWVp7VXi7V3P2uvh9SeVl9XVl8Oq+8pVt+3"
    "Un2ogO+krnRsUQEYW1TAb1JXOrY3sAIx+miH0Ue7BCsQo492daX6KkkFag3ezxpsxBpE9R2WCkRdaRWiolGFd3tahQlWISoaVVhX"
    "KgtVNU7qqp+8UFVjpK7ulxeqaoLUVT95tWAl9mMlVmMl3s9K7M1K7MdKvK9CDqvwNCuxjMfwdfR/qxLVV4d1uUxiBatKde8jqt5I"
    "qt5HrsI+YHXuYnWi6j5jfX7D+lQP5LN6lrJ6+rJ6MlmjHVijg1ijVf1MVmkHVukgVimqLZ112pJ1mss6vVaiHKukKaukE6vkNlaJ"
    "qlyp15W1msNafYq1+i01Di20WnNYrU+xWr9ltaJeVW3/TLV9nkpXQapVKxHVoPr2o6f6Vt9XffuN+vYL1e07qtuPnqpbfT/BmlV1"
    "+8U9IdW276htP3qqbfX9BKtWtQ0t61KFK1GFa/PebqUKX+urCh+Wqp1CfhtHfnuF/DaG/FZAfptAfuvLuh3Hun2FdTuGdVvAup3A"
    "uu3rKrEFq7cfq7caq/d+Vm9vVm8/Vu99FVz1+jn+GA+vMV5972mJz721Xm4sN1bN/9x9xftq/jrvczm21qsmK2GPeM9IrHKfDzHC"
    "ecMZsRmx7V4zf0tsS6yOP8J9j2PbvS0xHKvjf+2NkDbDPRzDURz72j2jEonr/L9LTPVHSRzxGvqHY4dj5fxH/Pru7vr5ffyp3lRv"
    "mztzOIZz5dxxnOvnjuPcNn+/h3M4W076POL3kz59fJzb733lNeC4PS1xr99coof/ksQlN/bYB2nov+mne+lerv+g395r7z3jv+Gv"
    "9FZ6z7sWtaVNQ3cebXLdebR5xp1Hm+f9ttIGrRpKjjf9XMnxoP+M5HjDf15WqC56LaQmsvxWEhlu9oi4258lUVZJn7rnVcHFea+t"
    "39hr7Kq6np/n5XkPu2c1wj3TJu4rxu/frmUFaXvetULbGq4d2j7s2qFtE9cObf/t7uhTybld2qJ1DclZz39Ycv7dbyI5/+7/W9Df"
    "imPyiLsrxJ28x778fhBZYpGbcSEy3Rhd76Klw2o3r5vDSyf/Oe85d537fKyf5bj5zaXYpZh144Q5UbMg07/eOyI9unnoU9e1R5+2"
    "rj365Lj26GNde/RpFqRIH/RqKdeo4NeVa3Ty28o17vNz5Bo9fSvXKOejT0pQVvF4rWVdP83qr+bnxrT+1/E9an6dVP1w1m8zX+v3"
    "GaLgMFFQx58RUxxsIQ6+5jHU/BYiAce+dpWhdfoI67QP67SEaPi7r2gYRTRsc9WueKjvKR6mEg/7PT2Hmsc51PxUImK/IOI7T+vx"
    "TV/r8UHW4xusxwa+ouJpoqI5UfESUfG8X9tTXKQTF+2Ji5XERVu2Qc2nExntiYyVREZbqfrtntZdW1/rrh7r7u+X1V0LoqMV0fEk"
    "0TGL6GhEdPzbr+ApPhoTH3nExwjiYwzxkcu2qPnGREgeETKCCBlDhORK1R/xtL4q+FpfnVhf97G+erK+yrG+RhEltxIls4iSW4mS"
    "XKLkPU9R0iy4njjpRpw8R5zkEyeXYoqTYuIkJdA+qPluRMpzREo+kYI+qPliIgV9UoImYZPwUnApwL+V9Gvwa+CFXvhV8FVwb3hv"
    "GHfRzUW6i1QXfwR/BD+5mOkCKFF1yPGBijIV+FwQAsTUF9QAK4qcdYIWoAB1DxSg7oGAEYIMRc/XgpxHBD2KH6xxAF1A0BYiCOpQ"
    "x2/I+n6E9d2H9Y2Kx3FUfDlBhKJovyCoWFCkOBpFHG0THCmSgDogaSqRtF+QhCq/JFjROn6QdfwG67g20ZRONLUnmlYSTW1FW4Cn"
    "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"
    "zMH2nubg7/gasu4GkxCvIet+J2n3e+bQfObQbebQKKM5lONpGmYzDe83mobDmIapZmBI8/AA87CEedjM0zys9PQaZN0BJmIJE7GZ"
    "p4lYKYn4e0/z5u/Mm1+YNy8ZzZs/M29GMRXfYypGG03Fo56m4kWm4hTTL6S5uIG5WN3TXGzPXMxgLtY0ei2ybgOTsbqnydieyZjB"
    "ZKwpaZfuaa6sYK4cYK5cZa4MNZormcyVHzxNx++ZjtFG0zGV6fga07HS03Q8Z3aENB9XMx/TPM3HDOZjKvOxGfNxu9E1yLrVTMg0"
    "TxMygwmZyoRsxoTcLmnX0mh+1GB+hJkfA5kfO5kfecyPHsyPuUzJJUzJLKbkOKbkB0zJfkzJ9kzJ6f5TnubkCOZkPeZkDnNyKnOy"
    "DXOyN3Oym69rkXUjmJT1mJQ5TMqpTMo2TMreTEqs7ebjJBY58ZPRnHjSak6UGc2JalZzIoE5cdZoTlxmTvxsNC2fs5qWaUzLPzEt"
    "k5iWKUzLBKblVqNpmRhpxrzsw7xszrwcyLyMZ152ZV7GMS+3e5qX1SLaA1nXh4nZnIk5kIkZz8TsysSMY2KiB7IOPapFKqMqo1rV"
    "alUryVU1V4+6uhp1NeqWq0JXtWvVrnWvK/xXSS65OuRqp6vBrla42uPqM1dIS7XDX73+zEW1QCQlkrNA0rOMxthf7LBM0hD5hzRE"
    "/iEJ50tCaooulwT9RFJUc7SD5Kgm6W4mKeywg0H6IeeQfsi5d5hzSL5sSUZNUyRfF0nGjczTg8xTJOnrkqbITE3UG0xUpA8S9XeS"
    "mkg75Nl85tlt5hnSDnmGpOspiaipWsJURdL1lkTMYa5mM1eRqMjVYczVVMlVTdYDTNYSJmszT5O1UpIV6Ybc+jtz6xfmFtINufVn"
    "5haSbbwkoaYrku2gJKGmawbTtaYZxXx9j/mKlEK+IlmRrxeZr1MkMzVhNzBhq3uasO2ZsBlM2JqSmkgz5NMK5tMB5tNV5hPSDPmU"
    "yXxCks2VBNSURZKtlATUlE1lyjZjym43PzBnv2fO4t7vk5TVnH2NOYuvHXL2nGSmJu1qJm2ap0mbwaRNZdI2Y9Jul9REeiGHajCH"
    "wsyhgcyhncyhPOZQD+YQkquKJJ+mbT2mbQ7TdirTtg3TtjfTtpuPpH1Q0lbzNot5O455+wHzth/ztj3zdrpkpibuCCZuPSZuDhN3"
    "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"
    "0L2qjpNk1HHKjT4Hn0kyajnlYjT7jDpIHasOUkoH6WnVdCbSdD6k6YxwV6vrZBh1naV0nRf4Gnwmw6jtLKXtvCBGM9+oayRadY2J"
    "dI2TdI27vpNj1Xj60XgKaDypbrU6T55R58mn8xTRedryGvhMnlHryaf1FNF62orRzOb3Y/xMp+hr1Sn+QqfIo1P8gebTkOYzg+Zz"
    "mOYziuZTbDHrgc+0pPssoPuU033G0H3q8Vr4TEvazwLaTzntZwztp54YTa5RdyiiO7Sx6g55dIciusMmusMoGtAzNKDJNKDtNKAf"
    "aUCXaUCr/S5GHSibDpRJB4pYdaAhRh2oplUHusI18JlsWlAmLShi1YKwBj5T06oFXRGjuceqIywz6ggNrTpCfzpCHzrCPjrCOTpC"
    "GU0ojyZURhOK+GpCd+yK33jQXprQ4WCEURdaQhfaTRcqoQvtoAttoAs9adWFygJdC59ZQhvaTRsqoQ3toA1toA09adWGsLYsgAPA"
    "BR721QWeoQsk0AUepguk0AX+Shf4ii4QQyPa5KsRfUwjWkwjeptGVDNQI9rtqxF9QyPaFwWTgc+U0olu0Iku0Ym+oBOdohPtohO1"
    "9tWJ8qO0B3ymlFZ0g1Z0iVb0Ba3oFK1oF62ota9WhB75UX/zlfnryPyLZP4PZP6bZP4uMj8UKPN7kfmHyPxhgZrRFJrRDppRP5rR"
    "VJpRc5rRaJrRVzSj7mE1oykxrXx1o750o850o8F0o7p0oyy6UQrdaBrdqGOgbjQzRnvBZ/rSjjrTjgbTjurSjrJoRym0o2m0o46B"
    "2hF6zYzpFCjb+5Ht+WT702R7NNnehWwvItvjw8r2YWR7bETZ3jyshtSChlSVhpRGQ2pJQxpDQ+pPQ/qWhnSChtQ3ooZ0ps4bgTpS"
    "azpSZzrSK3SkN+lIbehInehIm+lIcWF1pFV0pG11tCd8pjUtqTMt6RVa0pu0pDa0pE60pM20pLiwWtIqWhJ6Op8JK8MXkOFlZPhU"
    "MjyNDH+NDH+fDI+PKMMnkOGVZPj6KGV4YURNaQZNKYemNI+mNJumFEtT2kRT6hulpjSdplSrlppSDk3pRN3eYXWlj+hKa+lKoYi6"
    "0r6wulIlXelzulLviLqSiVJXakRXKqUrfVRXe8NnPqItraUtoTd8Zl9YbamStvQ5bal3RG0JveEzjWhLpbQl9P6o7sEEraOuqtav"
    "Wv9cwrmEW662uDrpCn+RxO2E2/J6Qv2E+hcSLiQ0qN+gfoqrZ101d9XN1X+5muSqMqEyIbF+Yv0kV4+7ginp9GeXmNHdKQ/+uUqs"
    "Ce7URNxJ7WmVGBNMCO5TSveBBeHnsCD8vFzsqVQMSh2qoTiUWlSSUYsqF2OC+cBxYD5wnFI6DqwHz8N6hooVqUm9IBaVICalLvUh"
    "XWqE+JLaVIZRm1pKm3pBjAmmA5eB6cBlJtJlTtJlYDl4HZYzTWxIjaqIRtXW/suo1Kn60akK6FSp4ktqVXlGrSqfVlVEq2orxjSb"
    "3wPzM50FZgNn+QudJY/OAqvBdS1pVgtoVuU0qzE0q3oyTYJbNaRbzaBbHaZbjaJbFYsvqV21pF0toF2V067G0K7qiTHBZOAmRXQT"
    "mAzcJI9uUkQ32UQ3gcXg+mwaViYNCxZTIfajhgWLyRX7GUXHeoaONZmOtZ2O9SMd6zIda7UPX1LLyqZlZdKyIlYtCwYHy6pp1bKu"
    "iDHBXOAgy4w6CMwFDtKfDtKHDrKPDnKODgJrGSjWo6a1m6ZVQtPaQdPaQNOCtWBdWVBG18qja5XRtWBZcC1Y1orfmJa61mHxJbWt"
    "JbSt3bStEtrWDtrWBtrWk1ZtC9Ofw4FOXhbJ5AWu8QxdI4Gu8TBdI4Wu8Ve6xld0DZ26LJKpyyyxHTWuSzSuL2hcp2hcu2hcsBSs"
    "z4+CbcG5YFv5YlzqXIvpXG/TuWBbcC7Y1noxLnWufeJLal2ltK4btK5LtK4vaF2naF27aF2tfbUuTH/2RcFM4BTr6BQX6RQ/0Cne"
    "pFPsolPATOAUvegUh+gUsBL06Uvz6kzzGkzzqkvzyqJ5pdC8ptG8YCXoMzNmGN1rCt1rB92rH91rKt2rOd1rNN3rK7oXrKuDmBd8"
    "Se2rL+2rM+1rMO2rLu0ri/aVQvuaRvvqGKh9YfozJQYmskR8Sd0hn+7wNN0hmu7Qhe5QRHeAicAdhtEdYCJwB1jIErEbNbDONLBX"
    "aGBv0sDa0MA60cA208BgIQfEbtTAttVpTgdrQQerSgdLo4O1pIONoYP1p4N9Swc7QQeDfSWLgcGX1MJa08I608JeoYW9SQtrQwvr"
    "RAvbTAuLC6uFraKFYfpzpg7MY4z4kjpCGR1hKh0hjY7wGh3hfToCzAOOMIGOUElHWB+ljgDrGCNWoya2liYG61gqVqMmVkkT+5wm"
    "ButAX1jHj2I1amKlNLGP6sLCOoiJqYvl0MXm0cVm08Vi6WKb6GKwMLjYdLoYLAwulkMXOyG+pDb2EW1sLW0M5ggb2xdWG6ukjX1O"
    "G+sdURuDOcLGGtHGSmljmP6c+DdfqlofD9jSOfGlk/KALeEBW8IDtoQHbClFfKmbPGBL/yW+lCgP2BIed/ujO/qjO/qfpJHdppEl"
    "1Fcja0Aje5ZG1o1GNolGlkgje5xGNjdBv3sCCbaL53B3T+Fq2FXmrp2top3Bz5qIn+n5Wwuevz3B07cWPH17wnFdz95a8OztCSH9"
    "CXqanuLdy5MoPXdL5LnbUJ67reSpWyJP3Yby1G2lMys9c0vkmdtQnrmtFOvqSV+bSF/7kL42QkrP2xryvG0az9tyed52hadtDXna"
    "No2nbbk8bbviTErP2hryrG0az9pyedZ2RWzr360N3taP3lZAb0uVms3vP/qZVqTnbAt5zlbIc7blPGWrylO2JTxlW8hTtkKesi13"
    "5hRLe2tJe1tAeyunvY2hvcGy/kB/a0h/m0F/O0x/G0V/K5bS87Uknq8V8nztRZ6vVfB87X2er+XydC2Jp2uFPF17kadrFTxde5+n"
    "a7nu66Zna0k8Wyvk2dqLPFur4Nna+zxbyxW7GkWPe4YeN5ket50e9yM97jI9DhanRgbL0XO1fJ6rrbH9aTl9aDn7aDnnaDl6qjaQ"
    "p2r5PFVbY0vocjvochvocnqqVssvC/RMbSDP1PJ5prbGltDmdtDmNtDm9EwNK9Xm4HN59Lky+pyeHn7N08MVv7E5+BxsTo3sX+dp"
    "s3ietpDnadk8T1vC87Q1PE/72v+KNvPb07RZPE1byNO0bJ6mLeFp2hqepn3t62laYpAf9duztFk8S1vIs7RsnqUt4VnaGp6lfe3r"
    "WRo6qNXB6/RUMZ+nivN5qjibp4ov8lTxrzxVXM9TxWtidWpksJZ1tJaLtJYfaC1v0lp20Vr0HC3gOdpLwSFai56ihYK+dLvOdLvB"
    "dLu6dLssul0K3W4a3U5P0aqFZ8boGVoo6Eu760y7G0y7q0u7y6LdpdDuptHu9AwNndTu4HdT6Hc76Hf96HdT6XfN6Xej6Xdf0e/0"
    "tLGD2J0aGeykH+0kn3byNO0kmnbShXZSRDvR87NGPD87ENbzs8yInp4t4enZmaAzHe8VOt6bdLw2dLxOdLzNdDw9PTsQXkXH21ZH"
    "z86W8OzsTNCZlvcKLe9NWl4bWl4nWt5mWp6enR0Ir6LlwZ70FLItTyGr8xQylqeQSeGW9Lwx9Lz+9Lxv6Xkn6Hl6CpkslqdGBgtZ"
    "QAspo4VMpYWk0UJeo4W8TwvRc7NL4Qm0kEpaiJ6bNamlp2ZjeGr2Hk/N1vDUbClPzYbw1KyEp2a7eGp2KaynZj9GGtH1Sul6H9XV"
    "M7MxPDN7j2dma3hmtpRnZkN4ZlbCM7NdPDNDZ7W9RrS9UtoerElPJztEZtD3cuh78+h7s+l7sfS9TfQ9PZ08GplO39PTyWNROfQ9"
    "2N5vjQzWpPOrc5xfbUnQ+ZVXX+dXRzm/upCg86sUzq+ac371X/V1flWZoPOrpN8YGR7qe+foe1sSbtP51PeO0vcuJDxL5+tG55tE"
    "50uk8z1O55ub8K93UOvT+z/H+8c7qPXp/R/l/eMd1Pq60fom0foSaX2P0/rmJsxN+H/98Z7f"
)

# ===================================================================== params
# d0/d1/dstreet/r_ip/r_oop: promoted 13 Nov 2026 after a holdout head-to-head win
# (TEST_RESULTS.md, experiment 3); previous values 0.012/0.012/0.01/1.03/0.92.
# allin_eq stays 0.80: 0.70 lost Stage 1 matches to tight-passive play (experiments 6-7).
PARAMS = {"allin_spr": 1.6, "d0": 0.0, "d1": 0.0, "dstreet": 0.0, "r_ip": 1.08, "r_oop": 0.98,
          "budget": 0.12, "allin_eq": 0.80,
          # time management (see Player.budget)
          "reserve_ms": 2500.0, "hand_ms": 6.0, "max_ms": 220.0, "max_n": 4000,
          "huge_bucket": 0}

# ===================================================================== tables

def _load_pf():
    raw = zlib.decompress(base64.b64decode(_PF_BLOB))
    arr = array.array("H")
    arr.frombytes(raw)
    if sys.byteorder != "little":
        arr.byteswap()
    n = 22100
    cols = [[arr[j * n + i] / 10000.0 for i in range(n)] for j in range(4)]
    order = sorted(range(n), key=cols[0].__getitem__)
    pct = [0.0] * n
    for r, i in enumerate(order):
        pct[i] = r / (n - 1.0)
    return cols, pct


PF_COLS, PF_PCT = _load_pf()
PF_GRID = (1.0, 0.5, 0.25, 0.12)
_LG = [math.log(c) for c in PF_GRID]


def pf_eq(idx, c):
    """Preflop equity of combo idx against the top-c fraction of 3-card hands."""
    if c >= 1.0:
        return PF_COLS[0][idx]
    lc = math.log(max(c, 0.04))
    for j in range(3):
        if lc >= _LG[j + 1]:
            t = (_LG[j] - lc) / (_LG[j] - _LG[j + 1])
            return PF_COLS[j][idx] * (1 - t) + PF_COLS[j + 1][idx] * t
    # extrapolate below 12%
    t = (_LG[3] - lc) / (_LG[2] - _LG[3])
    v = PF_COLS[3][idx] - t * (PF_COLS[2][idx] - PF_COLS[3][idx])
    return max(0.02, v)


try:
    _fill_rank_table((7, 6, 5), deadline=_T0 + 3.0)
except Exception:
    pass


def _sig(x):
    if x > 30:
        return 1.0
    if x < -30:
        return 0.0
    return 1.0 / (1.0 + math.exp(-x))


def log(*a):
    try:
        print(*a, file=sys.stderr)
    except Exception:
        pass


# ============================================================= opponent model

class Freq:
    __slots__ = ("c", "n")

    def __init__(self, prior, strength):
        self.c = {k: v * strength for k, v in prior.items()}
        self.n = 0

    def add(self, k):
        if k in self.c:
            self.c[k] += 1
            self.n += 1

    def p(self, k):
        return self.c[k] / sum(self.c.values())


class MeanStat:
    __slots__ = ("s", "w", "n")

    def __init__(self, prior, strength):
        self.s = prior * strength
        self.w = strength
        self.n = 0

    def add(self, x):
        self.s += x
        self.w += 1
        self.n += 1

    def mean(self):
        return self.s / self.w


FOLD_ADJ = (0.85, 1.0, 1.12, 1.25, 1.40)


def size_bucket(frac):
    if frac < 0.45:
        return 0
    if frac < 0.8:
        return 1
    if frac < 1.2:
        return 2
    if frac >= 2.5 and PARAMS["huge_bucket"]:
        return 4  # large overbets / shoves into small pots are learned separately
    return 3


class OppModel:
    def __init__(self):
        self.pf_open = Freq({"raise": 0.45, "call": 0.35, "fold": 0.20}, 6)
        self.pf_vs_raise = Freq({"fold": 0.35, "call": 0.45, "raise": 0.20}, 6)
        self.pf_vs_limp = Freq({"check": 0.6, "raise": 0.4}, 5)
        self.post_open = Freq({"check": 0.6, "raise": 0.4}, 6)
        self.post_face = Freq({"fold": 0.33, "call": 0.52, "raise": 0.15}, 6)
        self.post_face_rr = Freq({"fold": 0.22, "call": 0.53, "raise": 0.25}, 6)
        self.pf_vs_3bet = Freq({"fold": 0.25, "call": 0.50, "raise": 0.25}, 6)
        self.fold_b = [[0.0, 0] for _ in range(5)]  # folds, n per size bucket
        self.open_sz = [0, 0, 0, 0, 0]   # opp bets by size bucket (when not facing a bet)
        self.face_sz = [0, 0, 0, 0, 0]   # opp raises by size bucket (facing our bet)
        self.pf_honest = MeanStat(0.74, 3)
        self.post_honest = MeanStat(0.80, 3)
        self.disc_match = MeanStat(0.75, 4)
        self.sd_bet_river = []   # opp hand pct at showdown when they bet/raised the river
        self.sd_call_river = []  # opp hand pct at showdown when they called our river bet
        self.hands = 0
        self.vpip = 0
        self.pfr = 0
        self.showdowns = 0

    # -- derived
    def alpha_pf(self):
        return min(1.0, max(0.05, (self.pf_honest.mean() - 0.5) / 0.24))

    def alpha_post(self):
        return min(1.0, max(0.05, (self.post_honest.mean() - 0.52) / 0.28))

    def rational(self):
        return min(1.0, max(0.0, (self.disc_match.mean() - 1 / 3.0) / (2 / 3.0)))

    def fold_prob(self, frac):
        base = self.post_face.p("fold")
        b = size_bucket(frac)
        f, n = self.fold_b[b]
        prior = min(0.95, base * FOLD_ADJ[b])
        m = 4.0
        return min(0.95, max(0.0, (f + m * prior) / (n + m)))

    def sized_freq(self, open_ctx, rsize):
        """Frequency with which the opponent makes a bet/raise of this size bucket."""
        b = size_bucket(rsize)
        if open_ctx:
            st, cnt, prior = self.post_open, self.open_sz, (0.30, 0.40, 0.22, 0.08, 0.03)
        else:
            st, cnt, prior = self.post_face, self.face_sz, (0.20, 0.40, 0.25, 0.15, 0.06)
        tot = st.p("raise")
        nr = sum(cnt)
        share = (cnt[b] + 4.0 * prior[b]) / (nr + 4.0)
        return max(0.02, min(tot, tot * share))

    def raise_prob(self):
        return self.post_face.p("raise")

    def opp_raised_street(self, ev, street):
        for e in ev:
            if e["actor"] == "opp" and e["street"] == street and e["action"] == "raise":
                return True
        return False


def replay(history, my_seat):
    """Annotate history entries with the betting context the actor faced."""
    me_btn = my_seat == "button"
    bets = {"me": 1 if me_btn else 2, "opp": 2 if me_btn else 1}
    committed = dict(bets)
    street = "preflop"
    out = []
    first = {"me": True, "opp": True}
    raised = {"me": False, "opp": False}
    for h in history:
        st = h.get("street")
        if st != street:
            street = st
            bets = {"me": 0, "opp": 0}
            raised = {"me": False, "opp": False}
        a = h.get("actor")
        if a not in ("me", "opp"):
            continue
        o = "opp" if a == "me" else "me"
        to_call = bets[o] - bets[a]
        pot = committed["me"] + committed["opp"]
        act = h.get("action")
        e = {"street": st, "actor": a, "action": act, "to_call": to_call, "pot": pot,
             "reraise": raised[a]}
        if st == "preflop":
            if first[a] and to_call == 1 and bets[o] == 2:
                e["ctx"] = "open"
            elif to_call == 0:
                e["ctx"] = "vs_limp"
            else:
                e["ctx"] = "vs_raise"
            first[a] = False
        else:
            e["ctx"] = "open" if to_call == 0 else "face"
            if to_call > 0:
                e["frac"] = to_call / max(1.0, pot - to_call)
        if act == "call":
            add = to_call
        elif act == "raise":
            amt = h.get("amount") or bets[a]
            add = amt - bets[a]
            e["rsize"] = (amt - bets[o]) / max(1.0, pot + to_call)
        else:
            add = 0
        bets[a] += add
        committed[a] += add
        if act == "raise":
            raised[a] = True
        out.append(e)
    return out


# ======================================================================= bot

SIG_T = 0.10


class Player:
    def __init__(self):
        self.model = OppModel()
        self.total_hands = 200
        self.hand_id = 0
        self.seat = "button"
        self.hole3 = []
        self.hole2 = None
        self.my_disc = None
        self.opp_disc = None
        self.rng = random.Random()
        self.net = 0
        self.mc_time = 0.0
        self.n_dec = 0
        self.n_mc = 0
        self.time_bank_ms = 30000.0
        self.per_response_ms = 5000.0
        # last action request of this hand and our reply to it (for discard-time history)
        self._last_req = None
        self._last_reply = None
        # diagnostics (stderr only)
        self.lat_max = 0.0
        self.lat_tot = 0.0
        self.n_err = 0

    # ---------------------------------------------------------------- messages
    def on_init(self, msg):
        self.total_hands = int(msg.get("total_hands", 200) or 200)
        try:
            self.time_bank_ms = float(msg.get("time_bank_ms") or self.time_bank_ms)
            self.per_response_ms = float(msg.get("per_response_ms") or self.per_response_ms)
        except (TypeError, ValueError):
            pass

    def on_new_hand(self, msg):
        self.hand_id = int(msg.get("hand_id", self.hand_id + 1))
        self.seat = msg.get("seat", "button")
        self.hole3 = [c2i(c) for c in msg.get("hole_cards", [])]
        self.hole2 = None
        self.my_disc = None
        self.opp_disc = None
        self._last_req = None
        self._last_reply = None

    def note_reply(self, req, reply):
        self._last_req = req
        self._last_reply = reply

    def discard_history(self):
        """Full voluntary history at discard time.

        The last action_request's history lacks our own final action and the
        opponent's closing response. Both are implied: betting closed without a
        fold, so a raise of ours was called, and a check of ours as first actor
        on the flop (big blind) was checked behind.
        """
        req = self._last_req
        if not req:
            return []
        hist = list(req.get("history") or [])
        rep = self._last_reply or {}
        st = req.get("street", "preflop")
        act = rep.get("action")
        if act == "raise":
            amt = rep.get("amount")
            hist.append({"street": st, "actor": "me", "action": "raise", "amount": amt})
            hist.append({"street": st, "actor": "opp", "action": "call", "amount": amt})
        elif act == "call":
            hist.append({"street": st, "actor": "me", "action": "call",
                         "amount": req.get("opp_bet", 0)})
        elif act == "check":
            hist.append({"street": st, "actor": "me", "action": "check"})
            if st == "flop" and self.seat == "big_blind":
                hist.append({"street": st, "actor": "opp", "action": "check"})
        return hist

    def on_discard_result(self, msg):
        try:
            self.my_disc = c2i(msg["my_discard"])
            self.opp_disc = c2i(msg["opp_discard"])
            self.hole2 = [c2i(c) for c in msg["hole_cards"]]
        except Exception:
            self.n_err += 1
            _log_exc("discard_result")

    def on_hand_result(self, msg):
        try:
            self.net += int(msg.get("net", 0))
        except Exception:
            pass
        try:
            self._learn(msg)
        except Exception:
            self.n_err += 1
            _log_exc("learn")

    def _learn(self, msg):
        m = self.model
        m.hands += 1
        ev = replay(msg.get("history") or [], self.seat)
        pf_raised = False
        pf_vol = False
        post_aggr = False
        for e in ev:
            if e["actor"] != "opp":
                continue
            act = e["action"]
            if e["street"] == "preflop":
                ctx = e["ctx"]
                if ctx == "open":
                    m.pf_open.add(act)
                elif ctx == "vs_limp":
                    m.pf_vs_limp.add(act)
                elif e["reraise"]:
                    m.pf_vs_3bet.add(act)
                else:
                    m.pf_vs_raise.add(act)
                if act in ("call", "raise"):
                    pf_vol = True
                if act == "raise":
                    pf_raised = True
            else:
                if act == "raise" and not e["reraise"]:
                    sb = size_bucket(e.get("rsize", 0.66))
                    if e["ctx"] == "open":
                        m.open_sz[sb] += 1
                    else:
                        m.face_sz[sb] += 1
                if e["ctx"] == "open":
                    m.post_open.add(act)
                elif e["reraise"]:
                    m.post_face_rr.add(act)
                else:
                    m.post_face.add(act)
                    b = size_bucket(e.get("frac", 0.66))
                    m.fold_b[b][1] += 1
                    if act == "fold":
                        m.fold_b[b][0] += 1
                if act == "raise":
                    post_aggr = True
        if pf_vol:
            m.vpip += 1
        if pf_raised:
            m.pfr += 1
        oh = msg.get("opp_hole_cards")
        od = msg.get("opp_discard")
        board = [c2i(c) for c in (msg.get("board") or [])]
        if oh and od and len(board) == 5:
            m.showdowns += 1
            o2 = [c2i(c) for c in oh]
            odc = c2i(od)
            o3 = o2 + [odc]
            p = PF_PCT[combo_index(*o3)]
            if pf_raised:
                m.pf_honest.add(p)
            flop = board[:3]
            fk = KEY[flop[0]] + KEY[flop[1]] + KEY[flop[2]]
            _, d = best_keep(o3, flop, fk)
            m.disc_match.add(1.0 if d == 2 else 0.0)
            riv = [e for e in ev if e["street"] == "river"]
            r_bet = any(e["actor"] == "opp" and e["action"] == "raise" for e in riv)
            r_call = (not r_bet) and any(e["actor"] == "opp" and e["action"] == "call" for e in riv)
            if post_aggr or r_bet or r_call:
                known = set(o3 + board + self.hole3)
                pv = self.hand_pct(o2, board, known, 200)
                if post_aggr:
                    m.post_honest.add(pv)
                if r_bet:
                    m.sd_bet_river.append(pv)
                if r_call:
                    m.sd_call_river.append(pv)

    def hand_pct(self, h2, board, known, n):
        """Strength of a 2-card hand on a full board vs random 2-card hands."""
        rest = [c for c in range(52) if c not in known]
        bk = 0
        for c in board:
            bk += KEY[c]
        cards = list(board)
        ov = ev_key(bk + KEY[h2[0]] + KEY[h2[1]], cards + list(h2))
        win = 0.0
        rs = self.rng.sample
        for _ in range(n):
            a, b = rs(rest, 2)
            v = ev_key(bk + KEY[a] + KEY[b], cards + [a, b])
            if ov > v:
                win += 1
            elif ov == v:
                win += 0.5
        return win / n

    def emp_eq(self, lst, my_pct):
        """(equity, weight) of my_pct against an empirical list of opponent pcts."""
        if not lst:
            return 0.5, 0.0
        recent = lst[-40:]
        t = 0.0
        for s in recent:
            t += _sig((my_pct - s) / 0.015)
        n = len(recent)
        return t / n, n / (n + 6.0)

    # ------------------------------------------------------------- budgeting
    def budget(self, time_left_ms):
        """Monte Carlo time for one decision, in seconds.

        A reserve (fixed safety margin + a per-remaining-hand allowance for the
        non-Monte-Carlo work: parsing, preflop, learning, I/O) is set aside
        first; a fraction of what is left is spread over the expected number
        of Monte Carlo decisions in the remaining hands of this leg.
        """
        hands_left = max(1, self.total_hands - self.hand_id + 1)
        reserve = PARAMS["reserve_ms"] + PARAMS["hand_ms"] * hands_left
        usable = max(0.0, time_left_ms - reserve)
        per_hand = usable / hands_left
        # expected Monte Carlo decisions per hand, learned online (prior 4)
        dph = (self.n_mc + 20.0) / (max(0, self.hand_id - 1) + 5.0)
        b = per_hand * PARAMS["budget"] / max(1.0, dph)
        if PARAMS.get("debug"):
            log("BUD tl=%d hl=%d dph=%.2f b=%.1f" % (time_left_ms, hands_left, dph, b))
        if time_left_ms < 8000:
            b = min(b, per_hand * 0.15, 6.0)
        if time_left_ms < 2500:
            b = min(b, 1.5)
        cap = min(PARAMS["max_ms"], 0.05 * self.per_response_ms)
        return max(0.5, min(b, cap)) / 1000.0

    # ------------------------------------------------------- range factors
    def _pf_factor_fn(self, ev):
        """Return f(pct)->weight for opponent preflop actions this hand."""
        m = self.model
        alpha = m.alpha_pf()
        steps = []
        rem = 1.0
        for e in ev:
            if e["actor"] != "opp" or e["street"] != "preflop":
                continue
            ctx, act = e["ctx"], e["action"]
            if ctx == "open":
                fr = m.pf_open.p("raise")
                fc = fr + m.pf_open.p("call")
                if act == "raise":
                    rem2 = rem * fr
                    steps.append(("aggr", 1 - rem2, 0))
                    rem = rem2
                elif act == "call":
                    steps.append(("cont", 1 - rem * fc, 1 - rem * fr))
                    rem = rem * fc
            elif ctx == "vs_limp":
                fr = m.pf_vs_limp.p("raise")
                if act == "raise":
                    rem2 = rem * fr
                    steps.append(("aggr", 1 - rem2, 0))
                    rem = rem2
                else:
                    steps.append(("check", 1 - rem * fr, 0))
            else:
                st = m.pf_vs_3bet if e["reraise"] else m.pf_vs_raise
                fr = st.p("raise")
                fc = fr + st.p("call")
                if act == "raise":
                    rem2 = rem * max(fr, 0.03)
                    steps.append(("aggr", 1 - rem2, 0))
                    rem = rem2
                elif act == "call":
                    steps.append(("cont", 1 - rem * fc, 1 - rem * max(fr, 0.03)))
                    rem = rem * fc
        rem = max(rem, 0.04)
        means = [_gmean(kind, t1, t2) for kind, t1, t2 in steps]
        tab = []
        for i in range(101):
            p = i / 100.0
            f = 1.0
            for (kind, t1, t2), mu in zip(steps, means):
                f *= (1 - alpha) * mu + alpha * max(_gfun(kind, p, t1, t2), 0.02)
            tab.append(f)
        return tab, rem, alpha

    def _post_steps(self, ev):
        m = self.model
        fb = m.post_open.p("raise")
        steps = []
        rem = 1.0
        for e in ev:
            if e["actor"] != "opp" or e["street"] == "preflop":
                continue
            act = e["action"]
            if e["ctx"] == "open":
                if act == "raise":
                    rem2 = rem * m.sized_freq(True, e.get("rsize", 0.66))
                    steps.append(("aggr", 1 - rem2, 0))
                    rem = rem2
                elif act == "check":
                    steps.append(("check", 1 - rem * fb, 0))
            else:
                st = m.post_face_rr if e["reraise"] else m.post_face
                ff = st.p("fold")
                fr = max(st.p("raise"), 0.03)
                if not e["reraise"]:
                    fr = max(0.02, m.sized_freq(False, e.get("rsize", 0.66)))
                if act == "raise":
                    rem2 = rem * fr
                    steps.append(("aggr", 1 - rem2, 0))
                    rem = rem2
                elif act == "call":
                    fc = 1 - ff
                    steps.append(("cont", 1 - rem * fc, 1 - rem * fr))
                    rem = rem * fc
            rem = max(rem, 0.06)
        return steps

    # ------------------------------------------------------------ Monte Carlo
    def monte_carlo(self, board, ev, budget_s):
        t_start = time.perf_counter()
        deadline = t_start + budget_s
        self.n_mc += 1
        pft, _, _ = self._pf_factor_fn(ev)
        r = self.model.rational()
        nb = len(board)
        flop = board[:3]
        fk = KEY[flop[0]] + KEY[flop[1]] + KEY[flop[2]]
        bk = 0
        for c in board:
            bk += KEY[c]
        shuf = self.rng.shuffle
        rnd = self.rng.random
        maxn = int(PARAMS["max_n"])
        scores = []
        pw = []
        res = []
        if self.hole2 is None:
            # pre-discard: opponent holds 3 cards, both will discard
            h = self.hole3
            keeps = ((h[1], h[2]), (h[0], h[2]), (h[0], h[1]))
            kk = [KEY[x] + KEY[y] for x, y in keeps]
            dead = set(h + board)
            deck = [c for c in range(52) if c not in dead]
            res3 = ([], [], [])
            need = 5 - nb
            kdraw = 3 + need
            nd = len(deck)
            pos = nd
            i = 0
            while True:
                if (i & 15) == 0 and i >= 64 and time.perf_counter() > deadline:
                    break
                if i >= maxn:
                    break
                i += 1
                if pos + kdraw > nd:
                    shuf(deck)
                    pos = 0
                s = deck[pos:pos + kdraw]
                pos += kdraw
                o = s[:3]
                sc, d = best_keep(o, flop, fk)
                if r < 1.0 and rnd() > r:
                    d = int(rnd() * 3) % 3
                oa = o[1] if d == 0 else o[0]
                ob = o[1] if d == 2 else o[2]
                b5 = board + s[3:]
                k5 = bk
                for c in s[3:]:
                    k5 += KEY[c]
                ov = ev_key(k5 + KEY[oa] + KEY[ob], b5 + [oa, ob])
                for j in range(3):
                    x, y = keeps[j]
                    mv = ev_key(k5 + kk[j], b5 + [x, y])
                    res3[j].append(1.0 if mv > ov else (0.5 if mv == ov else 0.0))
                scores.append(sc)
                pw.append(pft[int(PF_PCT[combo_index(o[0], o[1], o[2])] * 100)])
            result = EqResult(scores, pw, None, self._post_steps(ev), self.model.alpha_post())
            eqs = [result.eq_with(res3[j]) for j in range(3)]
            best = max(range(3), key=lambda j: (eqs[j] + (0.0005 * (12 - (h[j] >> 2)) / 12.0)))
            result.set_res(res3[best])
            result.keep_eqs = eqs
            result.best_discard = best
        else:
            h2 = self.hole2
            od = self.opp_disc
            dead = set(self.hole3 + board)
            if self.my_disc is not None:
                dead.add(self.my_disc)
            if od is not None:
                dead.add(od)
            deck = [c for c in range(52) if c not in dead]
            need = 5 - nb
            mk = KEY[h2[0]] + KEY[h2[1]]
            myv = ev_key(bk + mk, board + h2) if need == 0 else 0
            match_w = r + (1 - r) / 3.0
            miss_w = (1 - r) / 3.0 + 0.02
            kdraw = 2 + need
            nd = len(deck)
            pos = nd
            i = 0
            while True:
                if (i & 15) == 0 and i >= 64 and time.perf_counter() > deadline:
                    break
                if i >= maxn:
                    break
                i += 1
                if pos + kdraw > nd:
                    shuf(deck)
                    pos = 0
                s = deck[pos:pos + kdraw]
                pos += kdraw
                a = s[0]
                b = s[1]
                w = 1.0
                if od is not None:
                    _, d = best_keep((a, b, od), flop, fk)
                    w = match_w if d == 2 else miss_w
                    w *= pft[int(PF_PCT[combo_index(a, b, od)] * 100)]
                sc = hscore(a, b, board, bk, nb)
                if need:
                    b5 = board + s[2:]
                    k5 = bk
                    for c in s[2:]:
                        k5 += KEY[c]
                    mv = ev_key(k5 + mk, b5 + h2)
                    ov = ev_key(k5 + KEY[a] + KEY[b], b5 + [a, b])
                else:
                    mv = myv
                    ov = ev_key(bk + KEY[a] + KEY[b], board + [a, b])
                res.append(1.0 if mv > ov else (0.5 if mv == ov else 0.0))
                scores.append(sc)
                pw.append(w)
            result = EqResult(scores, pw, res, self._post_steps(ev), self.model.alpha_post())
        el = time.perf_counter() - t_start
        self.mc_time += el
        if PARAMS.get("debug"):
            log("MC budget %.1fms took %.1fms n=%d" % (budget_s * 1000, el * 1000, len(scores)))
        return result

    # --------------------------------------------------------------- actions
    def act(self, req):
        legal = req.get("legal_actions") or []
        street = req.get("street", "preflop")
        ev = replay(req.get("history") or [], self.seat)
        if street == "preflop":
            return self.act_preflop(req, legal, ev)
        board = [c2i(c) for c in req.get("board") or []]
        tl = float(req.get("time_left_ms", 30000) or 0)
        if tl < 600:
            return self.panic_action(req, legal, board)
        budget = self.budget(tl)
        eqr = self.monte_carlo(board, ev, budget)
        if street == "river" and self.hole2 is not None and len(board) == 5:
            try:
                known = set(self.hole3 + board)
                if self.opp_disc is not None:
                    known.add(self.opp_disc)
                my_pct = self.hand_pct(self.hole2, board, known, 200)
                m = self.model
                if req.get("to_call", 0) > 0 and m.opp_raised_street(ev, "river"):
                    e, w = self.emp_eq(m.sd_bet_river, my_pct)
                    if w > 0:
                        eqr.eq = (1 - w) * eqr.eq + w * e
                        eqr.emp_call = None
                else:
                    e, w = self.emp_eq(m.sd_call_river, my_pct)
                    if w > 0:
                        eqr.emp_call = (e, w)
            except Exception:
                self.n_err += 1
                _log_exc("emp")
        return self.act_post(req, legal, street, eqr, ev)

    def _sizes_post(self, req, fracs, allow_allin=True):
        pot = req["pot"]
        to_call = req["to_call"]
        mn = req["min_raise"]
        mx = req["max_raise"]
        out = set()
        for f in fracs:
            x = int(round(req["opp_bet"] + f * (pot + to_call)))
            x = max(mn, min(mx, x))
            if mx - x < 0.35 * (pot + (x - req["my_bet"]) + (x - req["opp_bet"])):
                x = mx
            out.add(x)
        if allow_allin or mx - req["my_bet"] <= PARAMS["allin_spr"] * (pot + to_call):
            out.add(mx)
        return sorted(out)

    def act_post(self, req, legal, street, E, ev):
        m = self.model
        pot = req["pot"]
        to_call = req["to_call"]
        my_bet = req["my_bet"]
        opp_bet = req["opp_bet"]
        my_stack = req["my_stack"]
        opp_stack = req["opp_stack"]
        ip = self.seat == "button"
        eq = E.eq
        alpha = E.alpha
        if street == "river":
            R = 1.0
        elif street == "turn":
            R = 0.99 if ip else 0.95
        else:
            R = 0.98 if ip else 0.93
        cands = []
        if to_call > 0:
            if "fold" in legal:
                cands.append((0.0, "fold", None))
            if "call" in legal:
                cands.append((eq * R * (pot + to_call) - to_call, "call", None))
        else:
            if "check" in legal:
                cands.append((eq * R * pot, "check", None))
        rr = m.opp_raised_street(ev, street)
        if "raise" in legal and "min_raise" in req:
            fracs = (0.33, 0.5, 0.75, 1.0, 1.5) if to_call == 0 else (0.6, 0.85, 1.2)
            for x in self._sizes_post(req, fracs, E.eq >= PARAMS["allin_eq"]):
                add = x - my_bet
                oadd = x - opp_bet
                if add > my_stack:
                    continue
                frac = oadd / max(1.0, pot + to_call)
                allin = (x == req["max_raise"]) or oadd >= opp_stack
                if rr:
                    pf = m.post_face_rr.p("fold") * (0.85 + 0.15 * min(frac, 2.0))
                    pr = m.post_face_rr.p("raise")
                else:
                    pf = m.fold_prob(frac)
                    pr = m.raise_prob()
                if allin:
                    pr = 0.0
                pc = max(0.0, 1.0 - pf - pr)
                eq_c = alpha * E.top(1.0 - pf) + (1 - alpha) * eq
                if E.emp_call is not None and not rr:
                    ee, ww = E.emp_call
                    eq_c = (1 - ww) * eq_c + ww * ee
                delta = PARAMS["d0"] + PARAMS["d1"] * min(frac, 3.0)
                if street != "river":
                    delta += PARAMS["dstreet"]
                fpot = pot + add + oadd
                ev_c = (eq_c - delta) * (R if not ip else 1.0) * fpot - add
                evv = pf * pot + pc * ev_c
                if pr > 0:
                    eq_r = alpha * E.top(max(0.04, pr)) + (1 - alpha) * eq
                    e = min(fpot, my_stack - add, opp_stack - oadd)
                    e = max(0, e)
                    ev_r = max(-add, (eq_r - delta) * (fpot + 2 * e) - add - e)
                    evv += pr * ev_r
                cands.append((evv, "raise", x))
        if not cands:
            return safe_default(legal)
        cands.sort(key=lambda t: -t[0])
        best = cands[0]
        if PARAMS.get("debug"):
            log("H%d %s %s eq=%.3f t50=%.3f t25=%.3f n=%d pot=%d tc=%d a=%.2f | %s" % (
                self.hand_id, street, " ".join(I2C[c] for c in (self.hole2 or self.hole3)), E.eq,
                E.top(0.5), E.top(0.25), E.n, pot, to_call, alpha,
                " ".join("%s%s:%.1f" % (c[1][0], c[2] or "", c[0]) for c in cands[:5])))
        # mild randomisation between near-equal options to stay unpredictable
        if len(cands) > 1 and cands[1][1] != best[1]:
            gap = best[0] - cands[1][0]
            if gap < 0.015 * pot and self.rng.random() < 0.25:
                best = cands[1]
        self.n_dec += 1
        return make_action(best[1], best[2], req)

    def act_preflop(self, req, legal, ev):
        m = self.model
        pot = req["pot"]
        to_call = req["to_call"]
        my_bet = req["my_bet"]
        opp_bet = req["opp_bet"]
        my_stack = req["my_stack"]
        opp_stack = req["opp_stack"]
        ip = self.seat == "button"
        idx = combo_index(*self.hole3)
        _, rem, alpha = self._pf_factor_fn(ev)
        def eqv(c):
            return alpha * pf_eq(idx, c) + (1 - alpha) * PF_COLS[0][idx]
        eq = eqv(rem)
        R = PARAMS["r_ip"] if ip else PARAMS["r_oop"]
        cands = []
        if to_call > 0:
            if "fold" in legal:
                cands.append((0.0, "fold", None))
            if "call" in legal:
                cands.append((eq * R * (pot + to_call) - to_call, "call", None))
        elif "check" in legal:
            cands.append((eq * R * pot, "check", None))
        if "raise" in legal and "min_raise" in req:
            mn, mx = req["min_raise"], req["max_raise"]
            if opp_bet <= 2:
                sizes = (5, 6, 8) if to_call > 0 else (6, 8)
            else:
                sizes = (int(opp_bet * 2.6), int(opp_bet * 3.2), mx)
            xs = set()
            for x in sizes:
                x = max(mn, min(mx, x))
                if x > 0.42 * (my_bet + my_stack):
                    x = mx
                xs.add(x)
            if m.opp_raised_street(ev, "preflop"):
                fold_base = m.pf_vs_3bet.p("fold")
                pr = m.pf_vs_3bet.p("raise")
            else:
                fold_base = m.pf_vs_raise.p("fold")
                pr = m.pf_vs_raise.p("raise")
            for x in sorted(xs):
                add = x - my_bet
                oadd = x - opp_bet
                if add > my_stack:
                    continue
                allin = x == mx
                frac = oadd / max(1.0, pot + to_call)
                pf = min(0.95, fold_base * (0.85 + 0.15 * min(frac, 3.0)))
                p_r = 0.0 if allin else pr
                pc = max(0.0, 1.0 - pf - p_r)
                eq_c = eqv(rem * (1.0 - pf))
                fpot = pot + add + oadd
                delta = 0.0 if fpot < 60 else 0.02 + 0.03 * min(1.0, fpot / 400.0)
                rr = R if fpot < 100 else 1.0
                ev_c = (eq_c - delta) * rr * fpot - add
                evv = pf * pot + pc * ev_c
                if p_r > 0:
                    eq_r = eqv(rem * max(p_r, 0.04))
                    e = min(int(2.5 * oadd) + 2, my_stack - add, opp_stack - oadd)
                    e = max(0, e)
                    ev_r = max(-add, (eq_r - 0.02) * rr * (fpot + 2 * e) - add - e)
                    evv += p_r * ev_r
                cands.append((evv, "raise", x))
        if not cands:
            return safe_default(legal)
        cands.sort(key=lambda t: -t[0])
        best = cands[0]
        if len(cands) > 1 and cands[1][1] != best[1] and best[0] - cands[1][0] < 0.04 * pot \
                and self.rng.random() < 0.2:
            best = cands[1]
        return make_action(best[1], best[2], req)

    def panic_action(self, req, legal, board):
        """Near-zero-cost decision used only when the time bank is almost empty."""
        h = self.hole2 or self.hole3
        bk = 0
        for c in board:
            bk += KEY[c]
        if self.hole2:
            sc = hscore(h[0], h[1], board, bk, len(board))
        else:
            sc, _ = best_keep(h, board[:3], KEY[board[0]] + KEY[board[1]] + KEY[board[2]])
        to_call = req.get("to_call", 0)
        pot = req.get("pot", 1)
        if sc > 0.8 and "raise" in legal:
            return make_action("raise", req.get("opp_bet", 0) + 0.7 * (pot + to_call), req)
        if to_call == 0:
            return safe_default(legal)
        if sc > 0.45 + 0.6 * to_call / max(1.0, pot + to_call) and "call" in legal:
            return {"type": "action", "action": "call"}
        return safe_default(legal)

    def discard(self, req):
        hole = [c2i(c) for c in req.get("hole_cards") or []]
        if len(hole) != 3:
            return None
        self.hole3 = hole
        self.hole2 = None
        board = [c2i(c) for c in req.get("board") or []][:3]
        if float(req.get("time_left_ms", 30000) or 0) < 600:
            _, d = best_keep(hole, board, KEY[board[0]] + KEY[board[1]] + KEY[board[2]])
            return I2C[hole[d]]
        ev = replay(self.discard_history(), self.seat)
        budget = self.budget(float(req.get("time_left_ms", 30000))) * 1.3
        E = self.monte_carlo(board, ev, budget)
        return I2C[hole[E.best_discard]]


def _gmean(kind, t1, t2):
    t = 0.0
    for i in range(41):
        t += _gfun(kind, i / 40.0, t1, t2)
    return t / 41.0


def _gfun(kind, p, t1, t2):
    if kind == "aggr":
        return _sig((p - t1) / SIG_T)
    if kind == "cont":
        return _sig((p - t1) / SIG_T) * (1.0 - 0.45 * _sig((p - t2) / SIG_T))
    # check: strongest hands would usually have bet
    return 1.0 - 0.55 * _sig((p - t1) / SIG_T)


class EqResult:
    """Weighted Monte Carlo samples with action-conditioned opponent range."""

    def __init__(self, scores, pw, res, steps, alpha):
        n = len(scores)
        self.alpha = alpha
        order = sorted(range(n), key=scores.__getitem__)
        tot = 0.0
        for w in pw:
            tot += w
        tot = tot or 1.0
        # weighted mid-rank percentile; tied scores share one percentile
        pct = [0.0] * n
        acc = 0.0
        k = 0
        while k < n:
            s0 = scores[order[k]]
            j = k
            gw = 0.0
            while j < n and scores[order[j]] == s0:
                gw += pw[order[j]]
                j += 1
            p = (acc + 0.5 * gw) / tot
            for q in range(k, j):
                pct[order[q]] = p
            acc += gw
            k = j
        wt = list(pw)
        if steps:
            a = alpha
            # combined action factor tabulated on a 200-point percentile grid
            tab = [1.0] * 201
            for kind, t1, t2 in steps:
                mu = (1 - a) * _gmean(kind, t1, t2)
                for j in range(201):
                    g = _gfun(kind, j / 200.0, t1, t2)
                    tab[j] *= mu + a * (g if g > 0.02 else 0.02)
            for i in range(n):
                wt[i] *= tab[int(pct[i] * 200.0 + 0.5)]
        self.w = wt
        self.pct = pct
        self.order_desc = order[::-1]
        self.n = n
        self.res = None
        self.eq = 0.5
        self.emp_call = None
        if res is not None:
            self.set_res(res)

    def eq_with(self, res):
        s = 0.0
        t = 0.0
        w = self.w
        for i in range(self.n):
            s += w[i] * res[i]
            t += w[i]
        return s / t if t > 0 else 0.5

    def set_res(self, res):
        self.res = res
        self.eq = self.eq_with(res)

    def top(self, c):
        """Equity against the strongest c fraction (by weight) of the range."""
        if c >= 0.999 or self.res is None:
            return self.eq
        w = self.w
        res = self.res
        tot = 0.0
        for x in w:
            tot += x
        lim = c * tot
        s = 0.0
        t = 0.0
        for i in self.order_desc:
            s += w[i] * res[i]
            t += w[i]
            if t >= lim:
                break
        return s / t if t > 0 else self.eq


# ================================================================ protocol

def safe_default(legal):
    if "check" in legal:
        return {"type": "action", "action": "check"}
    if "fold" in legal:
        return {"type": "action", "action": "fold"}
    if "call" in legal:
        return {"type": "action", "action": "call"}
    return {"type": "action", "action": legal[0] if legal else "check"}


def make_action(action, amount, req):
    legal = req.get("legal_actions") or []
    if action not in legal:
        return safe_default(legal)
    if action == "raise":
        try:
            mn = int(req["min_raise"])
            mx = int(req["max_raise"])
            amt = int(amount)
            amt = max(mn, min(mx, amt))
            return {"type": "action", "action": "raise", "amount": amt}
        except Exception:
            if "call" in legal:
                return {"type": "action", "action": "call"}
            return safe_default(legal)
    return {"type": "action", "action": action}


def _log_exc(where):
    """Errors are never silent: count them and log a short traceback to stderr."""
    try:
        import traceback
        log(where, "error:", traceback.format_exc(limit=4)[-900:])
    except Exception:
        pass


def main():
    player = Player()
    out = sys.stdout
    clock = time.perf_counter
    for line in sys.stdin:
        t_in = clock()
        line = line.strip()
        if not line:
            continue
        try:
            msg = json.loads(line)
        except Exception:
            continue
        if not isinstance(msg, dict):
            continue
        t = msg.get("type")
        reply = None
        if t == "action_request":
            try:
                reply = player.act(msg)
            except Exception:
                player.n_err += 1
                _log_exc("act")
                reply = safe_default(msg.get("legal_actions") or [])
            player.note_reply(msg, reply)
        elif t == "discard_request":
            hole = msg.get("hole_cards") or []
            card = None
            try:
                card = player.discard(msg)
            except Exception:
                player.n_err += 1
                _log_exc("discard")
            if card not in hole:
                card = hole[0] if hole else "2c"
                try:
                    card = min(hole, key=lambda c: RANKS.index(c[0]))
                except Exception:
                    pass
            reply = {"type": "discard", "card": card}
        elif t == "match_end":
            break
        else:
            try:
                if t == "new_hand":
                    player.on_new_hand(msg)
                elif t == "discard_result":
                    player.on_discard_result(msg)
                elif t == "hand_result":
                    player.on_hand_result(msg)
                elif t == "init":
                    player.on_init(msg)
            except Exception:
                player.n_err += 1
                _log_exc("msg " + str(t))
            continue
        try:
            out.write(json.dumps(reply) + "\n")
            out.flush()
        except Exception:
            break
        dt = clock() - t_in
        player.lat_tot += dt
        if dt > player.lat_max:
            player.lat_max = dt
    m = player.model
    log("hands %d net %d mc %.2fs reply_tot %.2fs reply_max %.1fms errs %d | opp vpip %.2f pfr %.2f "
        "fold %.2f aggr %.2f a_pf %.2f a_post %.2f rat %.2f sd %d" % (
            m.hands, player.net, player.mc_time, player.lat_tot, player.lat_max * 1000, player.n_err,
            m.vpip / max(1, m.hands), m.pfr / max(1, m.hands),
            m.post_face.p("fold"), m.post_open.p("raise"), m.alpha_pf(), m.alpha_post(),
            m.rational(), m.showdowns))


if __name__ == "__main__":
    main()
