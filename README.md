# Discard Holdem Strategic AI Poker Bot

An autonomous game theoretic and Bayesian exploitative poker bot built for the Discard Holdem heads up tournament.

## Overview

Discard Holdem is a heads up variant of No Limit Texas Holdem featuring two core modifications:
1. Each player is dealt three hole cards preflop instead of two.
2. Following the flop betting round, both players simultaneously select one card from their hand to discard face up.

The tournament runs duplicate deals across four hundred hands in Stage 1 and one thousand hands in Stage 2, evaluated under a strict sandbox with one virtual processor, one gigabyte of memory, zero network access, and cumulative match time banks.

## Architecture

The bot combines multiple game theoretic and statistical subsystems:

1. **Split Radix Hand Evaluator:** A sixty four bit base five and base eight vectorized hand evaluator in pure Python and NumPy, evaluating over three million hands per second with zero disk footprint.
2. **Preflop Isomorphism Engine:** Exact mapping of all twenty two thousand one hundred three card combinations into one thousand seven hundred fifty five canonical suit isomorphic classes, backed by an embedded compressed rollout matrix.
3. **Flop Toss Engine:** Evaluates all three keep options simultaneously while applying a one point two percent deception rank signaling threshold to mask hand strength.
4. **Bayesian Exposed Discard Filter:** A rational softmax counterfactual filter over all nine hundred ninety live combinations, updating opponent range distributions following the revealed discard.
5. **River Subgame Evaluator:** Exact range versus range equity calculations incorporating nut advantage detection and nut blocker analysis to balance value bets and bluffs.
6. **Opponent Classifier:** Real time classification into seven archetypes (Calling Station, Starter, Tight Aggressive, Loose Aggressive, Equilibrium, Nit, and Maniac) with dynamic exploit adjustments.
7. **Match Risk Controller:** Implements a mathematical lock threshold based on net chip lead and remaining blind costs to secure match points and tie breaker caps without variance exposure.

## Project Structure

* `bot/bot.py`: Primary submission bot containing all self contained subsystems.
* `dist/`: Packaged submission archives ready for deployment.
* `sparring_bots/`: Diverse collection of protocol compliant sparring bots for local benchmarking.
* `tools/arena.py`: Multi threaded duplicate deal tournament runner and statistics aggregator.
* `tools/generate_preflop_table.py`: Offline simulation script for preflop rollout matrix generation.
* `tests/verify_phase1.py`: Unit tests validating table build time, preflop mapping, and evaluator parity.
* `tests/test_fuzz_protocol.py`: Adversarial protocol and edge case fuzzing suite.
* `Starter_Kit/`: Reference match engine and communication protocol utilities.

## Verification and Testing

Run the core unit test suite:
```powershell
python -m unittest tests/verify_phase1.py
```

Run the adversarial fuzzing suite:
```powershell
python tests/test_fuzz_protocol.py
```

Run tournament benchmarks against the sparring pool:
```powershell
python tools/arena.py --matches 5 --jobs 4
```
