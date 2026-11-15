# Discard Holdem Strategic AI Poker Bot

An autonomous game theoretic and Bayesian exploitative poker bot built for the Discard Holdem heads up tournament.

## Overview

Discard Holdem is a heads up variant of No Limit Texas Holdem featuring two core modifications:
1. Each player is dealt three hole cards preflop instead of two.
2. Following the flop betting round, both players simultaneously select one card from their hand to discard face up.

The tournament runs duplicate deals across four hundred hands in Stage 1 and one thousand hands in Stage 2, evaluated under a strict sandbox with one virtual processor, one gigabyte of memory, zero network access, and cumulative match time banks.

## Architecture

The bot combines multiple game theoretic and statistical subsystems:

1. **Base 5 Multiset Vectorized Hand Evaluator:** Exact best five of five, six, or seven card evaluation using a 73,775 pattern lookup table compressed to a 7.8 kilobyte payload, decompressing in 0.33 seconds and utilizing 59.6 megabytes of memory.
2. **Preflop Waugh Lookup Cube:** A 1,755 class suit isomorphic lookup table trained on 14,376,960 deals providing exact equity estimates and variance in microseconds.
3. **5 Archetype Bayesian Dirichlet Opponent Model:** Continuous classification across five behavioral archetypes (Calling Station, Random/Inconsistent, Tight Selective, Ordinary Aggressive, and Maniac) tracking three pot relative wager size buckets across all four streets.
4. **Dynamic EV Candidate Bet Sizing Engine:** Rigorous evaluation of four candidate wager sizes (0.33, 0.65, 1.0, and 1.5 pot) based on conditional continuing equity and empirical fold probabilities.
5. **Rule 8.2 Sticky Mathematical Lock Controller:** Automatically enforces the worst case fold bleed bound ($1.5 \times H_{\text{rem}}$) in the final 28 hands to secure 1.0 Match Point victories without variance risk.
6. **Rule 8.3 Late Match Capped Delta Risk Budget:** Strict risk caps in the final 36 hands to protect tournament tie breaker scores when holding a surplus lead.
7. **Quality Tie Breaker Discard Heuristic and River Bluff Catching Discipline:** Sharpens keep selections on close flop equities and prevents calling off big bets against passive value ranges.

## Project Structure

* `bot/bot.py`: Primary canonical tournament submission bot (v9 Supreme).
* `dist/`: Packaged submission archives ready for deployment (`bot_upload_v9_supreme.zip`).
* `docs/`: Official competition specifications, protocol rulebooks, and research archive.
* `scripts/`: Modular bot builder, packaging script, and multi seed gauntlet benchmarks.
* `sparring_bots/`: Diverse collection of protocol compliant sparring bots for local benchmarking.
* `Starter_Kit/`: Reference match engine and communication protocol utilities.
* `tests/`: Adversarial protocol fuzzing and comprehensive 7,000 case hardening tests.
* `tools/`: Multi threaded duplicate deal tournament runner and statistics aggregator.
* `audit/`: Verification records, SHA 256 checksum logs, and sandbox neutrality outputs.
* `pokerbots_complete_system_architecture_report.md`: Definitive technical reference and complete architecture report.

## Verification and Testing

Run the comprehensive v9 hardening and fuzzing suite:
```powershell
py -3.11 tests/test_v9_hardening.py
```

Run the core unit test suite:
```powershell
py -3.11 -m unittest tests/verify_phase1.py
```

Run the adversarial protocol fuzzing suite:
```powershell
py -3.11 tests/test_fuzz_protocol.py
```

Run duplicate deal match benchmarks:
```powershell
py -3.11 Starter_Kit/engine.py bot/bot.py sparring_bots/bot_tag.py --hands 400 --stage 1 --seed 202611
```

