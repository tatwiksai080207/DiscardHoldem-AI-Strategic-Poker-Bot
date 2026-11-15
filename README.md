# Discard Holdem Strategic AI Poker Bot

An autonomous game theoretic and Bayesian exploitative poker bot built for the Discard Holdem heads up tournament.

## Overview

Discard Holdem is a heads up variant of No Limit Texas Holdem featuring two core modifications:
1. Each player is dealt three hole cards preflop instead of two.
2. Following the flop betting round, both players simultaneously select one card from their hand to discard face up.

The tournament runs duplicate deals across four hundred hands in Stage 1 (two legs of two hundred hands) and one thousand hands in Stage 2 (two legs of five hundred hands), evaluated under a strict sandbox with one virtual processor, one gigabyte of memory, zero network access, and cumulative match time banks (30 seconds per leg in Stage 1; 75 seconds per leg in Stage 2).

## Architecture

The bot combines game theoretic, statistical, and simulation subsystems:

1. **Exact Table Driven Hand Evaluator:** Instantaneous seven card hand evaluation using exact bitwise rank keys and suit masks with zero external dependencies (pure Python standard library).
2. **Four Tier Precomputed Preflop Cube:** Embedded 4 by 22,100 combination equity cube providing exact preflop equities against top 100%, 50%, 25%, and 12% opponent ranges in microseconds.
3. **Positional Equity Realization:** Scales pot values by position realization multipliers ($R_{\text{ip}} = 1.08$ on button; $R_{\text{oop}} = 0.98$ out of position).
4. **Three Way Action EV Branching:** Models opponent fold ($p_f$), call ($p_c$), and re raise ($p_r$) probabilities across all candidate wager sizes, preventing over betting into strong raising ranges.
5. **Draw Synergy Discard Refinement:** Evaluates candidate discard holdings using Monte Carlo equity combined with draw potential (flush outs, straight connectivity, and pair/set outs) to resolve statistical ties in favor of maximum implied odds.
6. **Online Adaptive Opponent Modeling:** Learns opponent tendencies dynamically during match play (preflop VPIP, PFR, 3 bet frequency, fold to 3 bet, continuation betting, and check raise frequency).
7. **Unconstrained Expected Value Maximization:** Eliminates artificial fold locks and call suppression, ensuring the bot never surrenders committed chips or folds winning holdings.

## Project Structure

* [bot/bot.py](file:///c:/Projects/PokerBots/bot/bot.py): Primary canonical tournament submission bot (v11 Dominator).
* [Starter_Kit/bot.py](file:///c:/Projects/PokerBots/Starter_Kit/bot.py): Official starter kit target submission bot.
* [Starter_Kit/engine.py](file:///c:/Projects/PokerBots/Starter_Kit/engine.py): Official two leg tournament match engine.
* [dist/bot.py](file:///c:/Projects/PokerBots/dist/bot.py): Release standalone bot script.
* [dist/bot_upload_v11_dominator.zip](file:///c:/Projects/PokerBots/dist/bot_upload_v11_dominator.zip): Official packaged submission archive.
* [docs/](file:///c:/Projects/PokerBots/docs/): Official competition specifications, protocol rulebooks, and research archive.
* [scripts/](file:///c:/Projects/PokerBots/scripts/): Modular bot builder, promoter, and multi match gauntlet runners.
* [sparring_bots/](file:///c:/Projects/PokerBots/sparring_bots/): Diverse collection of protocol compliant sparring bots for local benchmarking.
* [tests/](file:///c:/Projects/PokerBots/tests/): Unit tests, regression suites, and protocol validation suites.
* [tools/](file:///c:/Projects/PokerBots/tools/): Static analysis and verification tools.
* [pokerbots_complete_system_architecture_report.md](file:///c:/Projects/PokerBots/pokerbots_complete_system_architecture_report.md): Definitive technical reference and complete architecture report.

## Verification and Testing

Run the comprehensive v11 test suite:
```powershell
py -3.11 tests/test_v11.py
```

Run head to head matches against Challenger:
```powershell
py -3.11 scripts/run_v11_vs_challenger.py
```

Run the sparring suite validation sweep:
```powershell
py -3.11 scripts/test_v11_sparring_suite.py
```


