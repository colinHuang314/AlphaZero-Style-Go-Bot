# Training notes: what to watch and how to adjust

This is the playbook for reading `runs/<name>/metrics.csv` (or TensorBoard:
`tensorboard --logdir runs`) between overnight runs and deciding what to change.
Change **one thing at a time** and write down why, so the effect can be measured.

## Findings about the original models (measured, Sept 2026)

| Check | AZNET9_epoch_300 | What it means |
|---|---|---|
| Value std across the 8 board symmetries (own-play positions) | 0.30–0.40 | The same position gets values up to ±0.4 apart depending on orientation, so the value head memorized games instead of learning Go. |
| Policy top-move agreement across symmetries | 59–68% | The policy is somewhat better than the value head, but still inconsistent. |
| Tactics: capture-in-1 | raw policy finds it; the choice after 32–200 visits does not | More search makes it *worse*, so the value head misleads the search. |

Root causes, from the code review:
1. Policy targets were stored after temperature, so they were one-hot for most of the game.
2. Dirichlet noise never fired (`sim == 0` was never true).
3. At the search root the net saw no history, unlike in training.
4. The search ignored an opponent pass before the root.
5. `MIN_PRIOR` flattened the priors.
6. About 30 games per generation were reused about 40 times each (8 stored augmentations × about 5 passes).

## Metric → action table

| Symptom | Likely cause | Adjustment |
|---|---|---|
| `val.value` rising while `train.value` falls | value overfitting (too much reuse) | lower `loop.train_ratio` (4 → 2), raise `min_window` |
| `val.policy` ≫ `train.policy` and growing | policy overfitting | same as above; or more `full_visits` for better targets |
| `train.policy_entropy` ≪ `train.target_entropy` | net overconfident vs search | lower lr, check reuse |
| `sp.black_winrate` stuck near 0% or 100% | value collapse / komi exploitation | check `sp.score_abs_mean`; more noise (`dirichlet_eps`), higher `temp_end` |
| `sp.capped_frac` > 5% | games not ending (bots don't pass) | normal very early; if it persists, inspect SGFs in `runs/*/sgf` |
| `sp.len_mean` very long (>150 on 9x9) | filling own eyes / not passing | usually fixes itself by ~cycle 20–40; inspect SGFs |
| `eval.prev.winrate` ≈ 50% for many evals | plateau | lower lr ×0.3–0.5; if already low, raise `full_visits` or net size |
| `eval.prev.winrate` < 45% repeatedly | regression / instability | lower lr; check grad_norm spikes |
| `train.grad_norm` spikes / NaNs | lr too high | lower lr, keep grad_clip |
| `sp.resign_false_pos` > 5% (once resign enabled) | resigning winnable games | lower `resign_threshold` (e.g. -0.95 → -0.98) |

## Learning-rate schedule (manual, between runs)

AlphaZero dropped the lr in steps. Here the rule is: drop lr by ~3× when
`eval.prev.winrate` has been within 50±5% for ~3 evals **and** val losses are
flat. Typical path: 0.02 → 0.006 → 0.002. Edit `configs/9x9.yaml` and resume;
the new value takes effect immediately (the lr is set every step from the config).

## When to change search settings

`c_puct`, `fpu_reduction` and visits can be tuned *without retraining* by
matches: `python tools/match.py latest.pt latest.pt --b-c-puct 2.0 --pairs 100`.
Do this once the net is reasonably strong (tuning on a weak net doesn't transfer).

## Final evaluation protocol (the 95% target)

```
python tools/match.py runs/9x9_a/latest.pt anchors/AZNET9_epoch_300.pt --pairs 200 --visits 400
python tools/match.py runs/9x9_a/latest.pt anchors/AZNET9_epoch_100.pt --pairs 200 --visits 400
```
400 games per anchor, equal visits, same MCTS settings, colors swapped on every
random 2-move opening. **Pass criterion: win rate ≥ 95% with the lower end of
the 95% Wilson interval ≥ 92.5%** (at 400 games that means ≥ 380 wins).
Also run the anchors with their best c_puct (`--b-c-puct 2 / 3`) so the old
models aren't handicapped by settings chosen for the new net.

## External strength (GTP)

`python -m gozero.gtp --model M --visits 400` speaks GTP. `tools/gtp_match.py`
referees any two GTP programs under the training rules (paired openings, colors
swapped, Tromp-Taylor scoring, optional CGOS-style clock with `--time`).
GNU Go runs in WSL: `wsl gnugo --mode gtp --chinese-rules --capture-all-dead --level 10`.

| Date | Match | Result |
|---|---|---|
| Sep 29 | c392 (400 visits) vs GNU Go 3.8 level 10, komi 7.5, 100 games | **100–0** (95% CI 96.3–100%, so at least +566 Elo). Every game was played out to two passes, with no clock. Winning margins 1.5–15.5, average 7.2: the bot maximizes win probability, not points. |

GNU Go 3.7.10 is the fixed 1800 anchor on CGOS 9x9, so this puts c392 at roughly 2,350 or more on that scale. That's only a floor: a 100–0 result can't show how much stronger it is. The real number needs opponents near its level, which CGOS has (Aya ~2150, Gurencho ~2430, KataGo nets ~2700–3200). Plan: CGOS for a public Elo, then an OGS bot account for a human 9x9 kyu/dan rank.

**Time management (`--time-manage`, added Oct 1).** On CGOS the 3,200-visit cap binds long before the clock (about 100 of 300 s used per game), so the GTP engine got an optional time manager, off by default, so plain `--visits 3200` behaves exactly as before:
- **Early stop:** the search ends once the leading move can't be overtaken in the visits or time left (by count, or by time at the current speed). The move is the same; the time stays on the clock.
- **Hard moves:** at the normal limit, if the top two moves are close (second ≥ half the first's visits), the best-Q move isn't the most visited (by more than 0.02), or the root Q swung by more than 0.15 since our previous move, keep searching up to 3× the visits and time (capped at 1/6 of the remaining clock). Stop once the move is no longer hard.
- **Decided games:** with |Q| > 0.94 (win estimate above 97% or below 3%), use half the visits and never extend.
- **Symmetry:** moves that are the same under a symmetry of the position (and of the previous position, so ko can't differ) are pooled into one candidate for all of the above, and the move is picked by pooled visits.

Local test (Oct 1 afternoon): `runs/gtp/tm_vs_v3200_partial_oct1/`, c439 with time management vs c439 at plain 3,200 visits, komi 7, 5 min sudden death, 4 matches × 50 games in parallel (a 3,200-visit move takes about 3.1 s that way vs a 5.3 s budget, so the cap still binds). Stopped by Colin after 10 games (about 15 min) to free the machine: time management 7.5–2.5 (one draw), which is far too few games to conclude anything (95% CI roughly 44–92%). No time losses or illegal moves. Rerun before trusting it. Time use in those games: tm averaged 1.77 s and 2,663 visits per move (18% of moves extended, max 10.7 s; 83% stopped early), v3200 2.13 s and 3,200 visits (max 4.1 s), so tm used *less* of the clock overall. Where the time went (rough, vs 2.13 s per plain move): early stops (54% of moves, 0.85 s each) saved ~335 s over the 10 games, the half cap in decided games (28%, 1.36 s) ~105 s, and extensions (18%, 5.17 s) spent ~270 s. Both versions used only about a third of the 300 s clock.

**Revised overnight plan (Oct 1, 5 pm; replaces the cap-tuning plan right below):** fixed caps don't carry over from self-play to CGOS. c439 vs c439 reuses about a third of each move's visits (fresh 3,200 ≈ 3.1 s, but plain v3200 averaged 2.13 s), while unfamiliar CGOS opponents allow less reuse, so clock use there would come out 1.3–1.5× higher. Colin's point: value grows with the log of *total* visits, so reuse should still count. That makes "every move aims at the same total" the best split of the clock. So the target is set from the clock: `--tm-use u` aims each move at the visits a fresh search reaches in u × its budget (measured speed; reused visits count; `--visits` is just an upper limit). A reused tree reaches the target sooner and banks the time, which raises later budgets and targets. Worst-case use depends only on u, not the opponent.

A worst-case simulation (no reuse, no early stops, 18% of moves extended 3×) also showed the old clock formula (110-move horizon, min 10 moves, 15 s reserve) **can lose on time in 160–220-move games**, even at full-budget use with 0.6 s lag. New formula: 120-move horizon, never fewer than 20 more own moves, 30 s reserve. Worst case for a 100-move game: u 0.5 → 63%, 0.6 → ~65%, 0.85 → 75%, 1.0 → 79–81%; no flag at u ≤ 0.85 even for 220 moves at 0.6 s lag. Every move's log line now shows its reused visits.

Tonight (`python runs/gtp/tm_tune/run.py`): stage 1 (1.5 h) tm60 and tm85 each play 2 matches vs plain v3200 (sanity, real clock use, reuse rate); stage 2 (until 11:30) tm85 vs tm60. Final: tm85 if it scores > 50%, never lost on time and its worst game ≤ 90%, else tm60. CGOS names: `gz-c439-4060-tm85` / `gz-c439-4060-tm60`.

**Result (Thu 10:05 pm → Fri 9:51 am, stopped early by Colin once the answer was clear; `runs/gtp/tm_tune/recommendation.txt`). No losses on time in 632 games.**

| Match | Games | Score | Elo (95% CI) |
|---|---|---|---|
| tm85 vs plain v3200 | 40 | 32–8 (80%) | **+241** (+109 to +372) |
| tm60 vs plain v3200 | 40 | 26.5–13.5 (66%) | +117 (+5 to +229) |
| tm85 vs tm60 | 552 | 316–236 (57.2%) | **+51** (+21 to +80) |

| Clock use per game (of 300 s) | mean | sd | p95 | worst | visits reused |
|---|---|---|---|---|---|
| tm85 (stage 2) | 148 s (49%) | 39 s | 206 s | 245 s (82%) | 34% |
| tm60 (stage 2) | 125 s (42%) | 34 s | 178 s | 207 s (69%) | 32% |
| plain v3200 (stage 1) | ~77 s (26%) | ~19 s | ~107 s | 120 s (40%) | ~30% |

Using more of the clock pays: tm85 beats tm60 clearly, and time management as a whole is worth roughly +240 Elo over plain 3,200 visits at the same 5-minute clock. Self-play reuses about a third of each move's visits; CGOS opponents are less predictable, so expect clock use there above 49% on average, with long games approaching the simulated worst case (≈ 90% for a 160-move game, no flag up to 220 moves at 0.6 s lag). **Deployed to CGOS on Fri Oct 2 as `gz-c439-4060-tm85`** (`runs/cgos/cgos.cfg` updated). Check the first CGOS games for clock use, network lag and reuse rate. If clock use there is well under ~65%, a higher `--tm-use` (e.g. 1.0) is the next thing to try, under a new account name.

**Overnight tuning plan (Thu Oct 1, 11 pm → Fri 11:30 am, review and deploy by noon; superseded by the revision above):** `python runs/gtp/tm_tune/run.py`. Also added a 15 s clock reserve to the GTP engine: per-move budgets come from the remaining main time minus 15 s, so the end of a long game never runs down to where network lag can flag it.
- Stage 1 (~2 h), clock calibration: time-managed caps 4,800 / 6,400 / 9,600 / 12,800 (hard moves up to 3×), 20 games each vs plain v3200, four matches in parallel. Records clock used per game (mean, sd, p95, worst) and losses on time.
- Choice: A = largest cap with mean use ≤ 75%, worst game ≤ 90% and no time losses; B = next cap down.
- Stage 2 (until 11:30): A vs B head to head, about 330–380 games (±35 Elo). Tests whether spending more of the clock pays off.
- Final: A if it scores > 50% with no time losses and worst game ≤ 90%, else B. Written to `runs/gtp/tm_tune/recommendation.txt` with the CGOS command line and account name (`gz439-4060-v<cap>tm`). Not tuned: the 3× extension and the hard-move thresholds, whose effects are likely below what 12 hours can resolve.
- Result: _to fill in_

**Earlier plan (Oct 1, superseded):** raise the time-managed base cap to 6,400 visits (Colin's call) so it uses more of the clock: about 3.5 s per move, ~170 s per game, expected. Above ~6,000 the per-move time budget (~5.2 s early on) binds before the cap, so 6,400 is near the useful maximum. The full comparison, tm6400 vs plain v3200, runs overnight Thu from ~11 pm (`python runs/gtp/tm_compare/run.py`, 400 games, ~7.5 h, `summary.txt` at the end). It answers which version to deploy, not how much each change contributes. Friday morning, deploy the winner on CGOS until Monday morning: plain as `gz-c439-4060-v3200`, or time-managed as `gz439-4060-v6400tm`.

**CGOS games so far** (`python tools/cgos_report.py`, which reads the client log and `runs/cgos/log/moves.log`; no GPU). First 4 games, Fri Oct 2 morning: won W+2 as white vs GNU Go 3.7.10 (1800); lost B+2 as white vs khd_N97-C (3044); lost W+88 as black vs hs_j13b6a1-10_234 (2876); lost W+88 as black vs khd_N97-C (3043). Provisional rating ~2700. Findings:
- **The game turns on sudden Q collapses, i.e. misjudged fights.** Game 2: +0.41 at our move 22 → −0.21 at 24 → −0.59 at 26. Game 4: −0.53 → −0.91 between moves 37 and 39. The bot thought it was ahead, then one or two opponent moves showed it wasn't. Same picture as the puzzle benchmark: the value head misreads sharp fights.
- **W+88 = white owns the whole board.** Once Q hits −1.0, every move looks equally lost, so the bot passes (12 times in game 3) while the opponent keeps playing and captures everything. No rating effect (a loss is a loss), but it looks bad, and it would against humans on OGS.
- **Network lag is ~0** (server clock drops match engine move times within ~0.1 s), so the 0.3 s margin and 30 s reserve are ample. Reuse is 19–35% (opponents are less predictable than self-play). Clock use 57–82% per game.
- **Wasted time while lost:** game 3 extended moves 57–81 to 11–15 s at Q −0.8 to −0.92 ("close" fires when everything is losing), about 60 s on hopeless moves. **Speed collapse in decided endgames:** 2,500 → ~110 visits/s by move 147. In a very narrow tree most of the 16 leaves per GPU batch collide, so the GPU gets 1–2 positions per call.

**After 15 games (Fri noon):** 6/6 vs GNU Go (1800); 0 wins, 1 draw, 8 losses vs the 2876–3044 bots (khd_N97-C, hs_j13b6a1-10_234, kata1_b28s1231v1, kata1_b28s679v1). Rating ~2600 and still drifting down; 0.5/9 vs ~2950 suggests ~2500–2600. **Blind spots:** `python tools/cgos_report.py --blindspots runs/cgos/blindspots` writes, for every Q drop ≥ 0.25 between two of our moves, the game cut just before our move and just after the opponent's reply, as SGFs for the UI's Load SGF, with `index.txt`. 12 so far, mostly moves 22–40 (middlegame fights). **The draw vs kata1_b28s1231v1 looks like a komi effect:** as white the bot was at Q +0.92 (move 86), then +0.67, +0.30, +0.08, and the game ended in a draw. The net always assumes komi 7.5, so a W+0.5 it plays for is a draw under CGOS's 7.0. Only terminal positions in the search score with the real komi, which would explain Q collapsing late. To confirm in the UI (which scores with 7.5, so the position will look won there). If confirmed, komi awareness (plan item 7) moves up.

**Goal (Colin, Oct 2): CGOS ≈ 3100**, above the bots it lost to (khd_N97-C ~3044, kata1_b28s1231v1 ~2923, hs_j13b6a1-10_234 ~2876–2966). After 9 games it's 4/4 vs GNU Go and 0/5 vs those, so its true level is somewhere around 2400–2800 and the gap is probably 300–700 Elo. The weekend's rating will narrow that. No single change below closes it: it needs several engine gains plus substantially more and better training, and possibly a bigger net. A calibrated local opponent near the bot's level (e.g. KataGo with a public net at a fixed low visit count) would make each step measurable without waiting for CGOS.

**Parallel self-play (Oct 2, built while CGOS runs; not yet measured on the GPU).** Self-play is one Python process at ~3,500 evals/s, an estimated 10–20% of what the 4060 can evaluate for this net, and it slowed by a third when GNU Go (CPU only) ran alongside, so it's CPU-bound. New `loop.selfplay_workers: K` (default 1 = unchanged): K spawned processes (`gozero/selfplay/workers.py`) each play `num_parallel / K` games with their own copy of the net on the GPU and send finished games to the training process. New weights go to them after each training phase (atomic file + version counter). Workers keep playing while the main process trains, so a cycle's samples can be up to one cycle stale (as in KataGo); `sp.evals_per_s` then measures throughput over the whole cycle. CPU smoke test with a tiny net: 2 workers, 186 cycles in 75 s, weights propagated, clean shutdown; all 48 tests pass. **Next (after CGOS, before the next training night):** `python tools/sp_bench.py --workers 1 2 4 6` (self-play only, ~10 min) to pick K, watching GPU memory (8 GB: each worker adds a CUDA context). Then set `selfplay_workers` in `configs/9x9.yaml`. This is also the prerequisite for renting a cloud GPU (a < $10 Vast.ai RTX 4090 weekend is the plan if the laptop speedup is real).

**Value target from search Q (Oct 2, ready).** Self-play now stores each recorded move's root value q (the 1000-visit search's Q, side to move's view) next to the game result z. New `train.value_q_weight` w: value target = (1 − w)·z + w·q (default 0 = unchanged). Buffers saved before this have no q and load with q = z, so old samples keep the old target. With a 400k window, the blend reaches about 37% of training data after a 30-cycle laptop night and all of it after ~80 cycles. The logged `train.value` / `val.value` still measure against z, so they stay comparable with every earlier night; `train.value_target` is the loss actually optimized. Tested: unit tests (exact BCE values, buffer round trip incl. old-format load) and a CPU run through 2 workers (all stored q in [−1, 1] and different from z). 49 tests pass.

**Weekend experiments (Oct 2–4), both from runs/9x9_a at cycle 471, one change each:**
- Run A, cloud: `configs/cloud_a.yaml` = 9x9.yaml + `value_q_weight: 0.5`.
- Run B, laptop (Sat after CGOS): `configs/laptop_b.yaml` = 9x9.yaml + `lr: 0.0006`, run dir `runs/9x9_b` (a copy of runs/9x9_a).
Sunday morning: A vs c439, B vs c439 (200 games each), A vs B (100), 3-seed puzzles. Then run C combines what worked.

**Cloud run (rented GPU, Vast.ai).** Scripts in `tools/cloud/`:
1. Laptop: `bash tools/cloud/pack.sh cloud_a` → `dist/cloud_a_bundle.tar.gz` (102 MB: code as it is now, the run state at c471, anchor c439; `VERSION` = git describe).
2. Instance: RTX 4090 ×1, on-demand (not interruptible), ≥16 vCPUs, ≥32 GB RAM, ≥40 GB disk, a PyTorch image, SSH access.
3. Upload: `scp -P PORT dist/cloud_a_bundle.tar.gz tools/cloud/setup.sh root@HOST:/workspace/`, then on the instance `bash /workspace/setup.sh cloud_a` (installs numba/pyyaml, prints GPU/CPUs, runs the tests).
4. `bash tools/cloud/bench.sh cloud_a` (~12 min) → pick K (laptop single process ≈ 3,500 evals/s).
5. `bash tools/cloud/train.sh cloud_a K HOURS`, with HOURS set from the remaining rental budget.
6. Laptop backups any time: `bash tools/cloud/pull.sh HOST PORT cloud_a`; at the end `... --full` (buffers + snapshots).

**Improvement plan (Oct 2; revise as more CGOS games come in).** Don't change the CGOS engine during the weekend run, so the rating stays clean.
1. *Training: value target from search Q (biggest lever).* The value head has been flat for 5 nights, and the CGOS losses are value misjudgments. Train value on a blend, e.g. 0.5·z + 0.5·(root Q of the 1000-visit search), for recorded moves. This needs root Q stored in self-play records. One night, then 200 games vs c439.
2. *Training: lr 0.002 → 0.0006.* Config only, a one-time boost (earlier drops gave +190 and +81). Run on a night not used for item 1, so each effect is measured on its own.
3. *Training: start some self-play games from CGOS turning points* (a few moves before each big Q drop), e.g. 10% of games. Gives proper search targets on positions self-play rarely reaches, against non-self-play styles (KataGo does similar).
4. *Engine: pondering* (keep searching on the opponent's time). Reuse on CGOS is only 19–35%, so searching the likely replies in advance should raise both reuse and effective search per move a lot. Probably the largest engine-only gain. Needs a background search thread in `gozero/gtp.py`, stopped by the next command.
5. *Engine: play on sensibly when decided.* Add a small score term to the search utility (we have a score head) so a lost or won position still prefers moves that keep stones alive. Also never pass while the opponent is still placing stones in our area, and optionally resign at Q < −0.97 for several moves (normal on CGOS and OGS).
6. *Engine: cheaper fixes.* Treat |Q| > 0.8 as decided for extensions (0.94 now). Fix narrow-tree batching (adaptive batch size, or skip collided leaves without shrinking the batch). Try leaf batch 32 for throughput.
7. *Later:* komi as a net input with randomized komi in training (CGOS uses 7.0, OGS varies); a bigger net if training plateaus again; faster inference (CUDA graphs).
Each change gets a local 200-game match under CGOS conditions before it's deployed, and each deployed version gets a new CGOS account name. Turning-point positions from losses also go to deep analysis (≥100k visits) and become new puzzles.

**CGOS deployment: moved up to Thu Oct 1, ~11 pm, as `gz-c439-4060-v3200` with `runs/eval_night6/candidate_c439.pt` at 3,200 visits.** `runs/cgos/cgos.cfg` is updated, the engine command was smoke-tested (loads, plays, answers `name`), and yss-aya.com:6809 is reachable. Earlier plan, kept for the naming rule: Before the first login, rename the account in `runs/cgos/cgos.cfg` (`ServerUser`) so the name records the model's training cycle (iteration) and the hardware, and point `--model` at the best frozen candidate at that point (currently c439 at `runs/eval_night6/candidate_c439.pt`, which beats c392 61.5%, so the name would be `gz-c439-4060-v3200`). CGOS names are at most 18 characters, for example `gz-c450-4060-v3200` (exactly 18): go-zero, cycle 450, RTX 4060 Laptop GPU, 3,200 visits per move. Every later model gets a new name in the same format, since a CGOS rating belongs to the account.

## Run log

| Night | Change | Why | Result |
|---|---|---|---|
| 1 (cycles 1–79, 9.7 h) | baseline config, lr 0.02 | — | 395–5 vs AZNET9_epoch_300 at 400 visits (98.8%); vs-prev gains slowing to about +100 Elo per 10 cycles; policy-target gap flat at 0.41 |
| 2 (from cycle 79, 13 h planned) | lr 0.02 → 0.006; progress matches vs frozen `runs/eval_night1/candidate_c79.pt` | the net stopped closing its gap to the search targets (flat for 50 cycles) at a fixed lr | cycles 80–130: one-time jump to ~75% vs c79 (+190 Elo) by cycle 90, then flat (73–75%); policy-target gap 0.42 → 0.36, then flat |
| 2b (from cycle 130, 1:41 → ~8:30) | `full_visits` 300 → 600 (lr stays 0.006) | lr gains plateaued within an hour; life & death needs deeper reading for good targets | 200-game matches at 400 visits: c162 vs c79 **74.5% (+186 Elo)**; c162 vs c130 **61.5% (+81)**; c130 vs c120 50.5% (+3, so lr phase had fully plateaued). Square four now judged dead; vital points still 0/4 |
| 3 (from cycle 162, Sat 1:45 pm → Sun ~8:30 am) | no training change (lr 0.006, full_visits 600); progress matches vs frozen c162 | 600-visit phase still gaining (+81 Elo in 6.7 h) | 82 cycles, 18.7 h. 200-game match at 400 visits: c244 vs c162 **75.5% (+196 Elo, 95% CI +140 to +251)**, about 2.4 Elo per cycle, the same rate as night 2b, so no plateau. The policy-target gap stayed flat near 0.40 all night while strength kept rising, so a flat gap alone does not mean a plateau. Self-play black win rate fell from 50% to about 35% after cycle 190 (komi 7.5 favors white under strong play), and train value loss fell 0.356 → 0.321 with it. Puzzles (16): raw policy 3 → 5, 800 visits 13 → 14 |
| 4 (from cycle 244, Sun 10:26 am → Mon ~8:20 am) | no training change; progress matches vs frozen c244 | still gaining ~2.4 Elo per cycle, so change nothing | 97 cycles, 22 h. 200-game match at 400 visits: c341 vs c244 **59.5% (+67 Elo, 95% CI +18 to +116)**, about 0.7 Elo per cycle, down from 2.4 (the two nights' intervals don't overlap). In-loop checks vs c244 pooled 343/640 (53.6%). Search-target entropy flat at ~1.69 and policy loss flat at ~2.08 from cycle 200 on, which in hindsight was the plateau signal. The 600-visit phase is mostly used up. Puzzles (20): 200 visits 11 → 16, 800 visits 18 → 18, raw policy 5 → 3 |
| 5 (from cycle 341, Mon 2:43 pm → Tue ~8:20 am) | `full_visits` 600 → 1000 (lr stays 0.006); progress matches vs frozen c341 | 600-visit phase slowed to ~0.7 Elo per cycle; hard puzzles need 1,600+ visits, so deeper targets should teach more reading | 51 cycles, 17.7 h (about 18 min per cycle once the PC was idle, 25–28 min while the PC was in use). 200-game match at 400 visits: c392 vs c341 **53.5% (+24 Elo, 95% CI −24 to +72)**. 100-game match: c392 vs c244 **66.0% (+115, CI +44 to +187)**. Going through c244 (c341 was +67 over it) implies about +48 over c341. Either way that's about 0.5–0.9 Elo per cycle, the same as night 4, so 1000 visits did not restart progress. Train policy loss 2.085 → 2.05 and val policy loss 2.12 → about 2.03; val value loss noisy (0.31–0.36), no trend. Puzzles (29): raw policy 5 → 8, but 800 visits 22 → 19 (see below) |
| 6 (from cycle 392, Tue 5:34 pm → Wed ~11:40 am, `--hours 18.1`) | lr 0.006 → 0.002 (`full_visits` stays 1000); progress matches vs frozen c392; morning test adds 100 games vs c341 | nights 4–5 both ~0.7 Elo per cycle and checks against the previous model at 41–59%, which meets the lr-drop rule; the night-2 drop gave +190 Elo within ~10 cycles; value head noisy (search moves away from correct policy moves in med7, tactics8) | Restarted at 7:43 pm from cycle 396 (the app restart for a `/compact` killed the first launch; cycle 397 was lost). Ran to cycle 439, 47 cycles, about 21 min per cycle. 200-game match at 400 visits: c439 vs c392 **61.5% (+81 Elo, 95% CI +32 to +131)**. 100-game match: c439 vs c341 **73.0% (+173, CI +97 to +249)**. About 1.7 Elo per cycle, against 0.5–0.9 on nights 4–5, so the lr drop worked. In-loop checks vs c392: 56% (c400), 62.5% (c410), 62.5% (c420), 59.4% (c430), so as on night 2, most of the gain came in the first 10–20 cycles. Train policy loss 2.03 → 1.99. Train value loss flat at 0.30–0.31, val value noisy (0.30–0.33): the value head still didn't improve. Puzzles (29, old single-orientation run): 800 visits 19 → 22, but raw policy 8 → 6. With the fixed 3-seed benchmark the searched results are flat vs c392 and raw policy is 6.8 → 7.5 (see the puzzle section). The old run showed tactics8 and blindspot2 unsolved at 12,800, but that turned out to be a benchmark artifact (fixed orientation; see the correction below) |
| 7 (from cycle 439, Wed 9:45 pm → Thu ~8:05 am, `--hours 10.33`; morning tests done by ~9:30) | no training change (lr 0.002, `full_visits` 1000); progress matches vs frozen c439; morning test: 200 games vs c439, 100 vs c392, 3-seed puzzles | night 6 gained +81 over c392, but most of it in the first 10–20 cycles, so check whether lr 0.002 keeps gaining before changing anything. If the checks vs c439 sit near 50% by ~cycle 460, the lr gain is used up and the value head (e.g. blending search Q into the value target) is next | 32 cycles (440–471), 10.6 h, about 18–19 min per cycle with the PC idle. 200-game match at 400 visits: c471 vs c439 **50.0% (100–100, Elo 0, 95% CI −48 to +48)**. 100-game match: c471 vs c392 **67.0% (+123, CI +51 to +195)**, in line with c439's +81 over c392. In-loop checks vs c439: 56%, 37.5%, 47%, 55% (pooled 125/256 = 48.8%). **So the lr-0.002 gain was used up during night 6: plateau.** Train policy loss still drifting down (1.99 → 1.97, val 1.97 → 1.94), value loss flat at 0.31 for the fifth night. Puzzles (3-seed): policy 7.5 → 8.1, searched rows the same as c439 within noise. c439 stays the anchor and the CGOS candidate |

**Night 2 follow-ups (resolved).** Cycle 130's 19–45 loss to cycle 120 was noise: 101–99 over 200 games at 400 visits. In-loop checks (64 games, 100 visits) have a ±12% interval, so don't act on a single one. One slow cycle (~1,200 evals/s at 1:15–1:30) did not recur.

**What to check after night 2.** Does `train.policy − train.target_entropy` drop
below 0.41, and does `eval.candidate_c79.winrate` go clearly above 50%? Also
rerun `python tools/diagnose.py runs/9x9_a/latest.pt` and compare the
life-and-death section with the c79 baseline below.

**Life-and-death baseline (c79).** 0/4 vital points found (straight three and
bent three, kill and live, at 1 to 400 visits). Square four is judged "unsure"
(ownership +0.11 / +0.27, where +1 means dead); two real eyes are judged alive
(−0.70). If a lower lr improves general play but not this, the next lever is
more `full_visits`. Reading dead shapes needs deeper search to produce good
value and ownership targets for those positions.

## Puzzle benchmark (`problems/`, 11 whole-board life-and-death puzzles by Colin)

`python tools/diagnose.py MODEL --problems problems/` grows one search per puzzle
to 12,800 visits. "needs" = the visits from which the top move stays correct.
Track two numbers per model: **solved at 800 visits** (play strength budget) and
**solved by the raw policy** (whether the net has internalized the shapes).

| Model | Raw policy | 400v | 800v | 1600v | 6400v | Hardest puzzle needs |
|---|---|---|---|---|---|---|
| c79 (night 1) | 2/11 | 7/11 | 8/11 | 9/11 | 11/11 | 6,400 (simple, black to live) |
| c162 (night 2) | 2/11 | 9/11 | 10/11 | 10/11 | 11/11 | 6,400 (hard life or death) |

The set grew to 16 puzzles on Sep 26 (five more added). Rows below use all 16.

| Model | Raw policy | 400v | 800v | 1600v | 3200v | 6400v | Hardest puzzle needs |
|---|---|---|---|---|---|---|---|
| c162 (night 2) | 3/16 | 11/16 | 13/16 | 13/16 | 14/16 | 16/16 | 6,400 (hard life or death; med5) |
| c244 (night 3) | 5/16 | 13/16 | 14/16 | 14/16 | 15/16 | 16/16 | 6,400 (med5) |

c162 → c244: "hard life or death" dropped from 6,400 to 800 visits, and the raw policy now solves bulky 5 and med7. Med7 is odd for c244: the raw policy picks J4 (correct), but searches from 100 to 1,600 visits move away from it until 3,200. That points at the value head misjudging the follow-up, not at the policy.

The set grew to 20 puzzles on Sep 27. Rows below use all 20.

| Model | Raw policy | 100v | 200v | 400v | 800v | 1600v | 12800v | Hardest puzzle needs |
|---|---|---|---|---|---|---|---|---|
| c244 (night 3) | 5/20 | 10/20 | 11/20 | 17/20 | 18/20 | 18/20 | 19/20 | >12,800 (hard2) |
| c341 (night 4) | 3/20 | 10/20 | 16/20 | 17/20 | 18/20 | 19/20 | 19/20 | >12,800 (bulky 5) |

c244 → c341: many puzzles are now found at 200 visits instead of 400 (med5, med7, throw-in, nose hit, simple, capture race); hard2 went from unsolved to 1,600. Bulky 5 flipped from solved at 100 visits to unsolved, but both models rate the puzzle's answer E1 and the alternative H7 almost equally (search Q 0.93 vs 0.94). After W H7, both models think black must answer (ignoring it with E1 loses the top right, white Q 0.99), and white then plays E1. If H7 really works too, the puzzle has two answers and c341 isn't wrong.

Colin then rebuilt bulky 5 as an endgame position with no big points elsewhere, so E1 is the only answer; rerun it before comparing with the rows above. Also seen in c341: the opening move shifted from E5 (86% in earlier models) to D4 (~50%), which matches what modern 9x9 engines prefer.

The set grew to 29 puzzles on Sep 28 (blindspot 2/4/5/6/11, blind16, lots of groups, tactics8, very hard fight). A tenth, "black to cut", was deleted on Sep 29 because its answer could not be proven. The rows below use all 29, with the rebuilt bulky 5, and leave black to cut out of both runs. c341 was rerun on this set.

| Model | Raw policy | 100v | 200v | 400v | 800v | 1600v | 3200v | 6400v | 12800v |
|---|---|---|---|---|---|---|---|---|---|
| c341 (night 4) | 5/29 | 14/29 | 20/29 | 21/29 | 22/29 | 22/29 | 24/29 | 26/29 | 27/29 |
| c392 (night 5) | 8/29 | 15/29 | 19/29 | 19/29 | 19/29 | 21/29 | 25/29 | 26/29 | 27/29 |
| c439 (night 6) | 6/29 | 14/29 | 17/29 | 17/29 | 22/29 | 23/29 | 24/29 | 25/29 | 26/29 |

c341 → c392: the raw policy now finds med7, simple and simple13. But with 400–1,600 visits c392 is worse on med7 (200 → 1,600), hard life or death (800 → 1,600), hard2 (1,600 → 3,200), tactics8 (solved at 100–800 → lost at 400–1,600) and blindspot11 (3,200 → 12,800). Only blindspot2 improved (12,800 → 3,200). Med7 has the same pattern as c244: the raw policy is right and the search moves away from it. So the policy learned from the deeper targets, but the value head didn't improve, and it now misleads the search in some sharp fights. Files: `runs/eval_night5/c392_problems.txt`, `c341_problems30.txt` (both still include black to cut).

Neither model solves blind16 or very hard fight at 12,800 visits. Colin checked c392 on them in the UI with deeper searches: blind16 (E9) is found at about 45k visits (52.1% of visits at 80k), and very hard fight (C5) at about 80k (50% at 160k). Both need 45–80× the training search depth (`full_visits` 1000), so they measure long-term progress: watch whether the visits needed come down.

c392 → c439: c439 reads much deeper. Blind16 is now solved at 12,800 visits (69% on E9), down from about 45k, and the visits needed fell for blindspot5 (3,200 → 800), lots of groups (6,400 → 1,600), blindspot11 (12,800 → 6,400), hard life or death and med7 (1,600 → 800). Two puzzles regressed badly: tactics8 (3,200 → unsolved, 7.8% on G8 at 12,800) and blindspot2 (3,200 → unsolved, 0%). Easy puzzles got a little slower (nose hit 200 → 800, move order 400 → 800, med5 and simple 100 → 200), and the raw policy lost med7, simple, simple13 and blindspot4 while gaining the throw-in and blindspot5. Very hard fight is still unsolved. So the gains are uneven, with some puzzles far worse. The matches (+81 over c392) are the main signal, and the puzzles don't contradict them. File: `runs/eval_night6/c439_problems.txt`.

**Correction (Sep 30): the tactics8 and blindspot2 "regressions" are an artifact of the benchmark.** Colin found c439 solves blindspot2 in about 6k visits and tactics8 in under 2k in the UI. `tools/diagnose.py` sets `random_symmetry = False`, so every evaluation sees the puzzle in its original orientation (symmetry 0), while the UI, GTP engine and matches use a random symmetry per batch. Rerunning both puzzles on c439 under each fixed symmetry and under random symmetry (scratch test):

| Puzzle | Symmetry 0 (the benchmark) | Other 7 fixed symmetries | Random symmetry, 4 seeds |
|---|---|---|---|
| tactics8 | unsolved at 12,800 (7.8%) | 100–3,200 | 100–800 |
| blindspot2 | unsolved at 12,800 (0%) | 1,600–6,400; symmetry 2 also unsolved | 1,600–3,200 |

So symmetry 0 happens to be the worst orientation for both puzzles, and every row in the tables above measures one orientation, not what the bot does in play. It also shows the net isn't close to symmetry-invariant on sharp fights. Fixed on Sep 30: `tools/diagnose.py --problems` now runs `--seeds 3` searches per puzzle (default), each with seeded random symmetry as in play. The raw-policy column counts how many of the 8 symmetries pick the answer. `--seeds 0` reproduces the old single-orientation run. All rows above this point are the old single-orientation runs.

**29 puzzles, random symmetry, average of 3 searches** (policy = average over the 8 symmetries). Files: `runs/eval_night6/c<cycle>_problems_sym3.txt`, about 9 minutes per model.

| Model | Raw policy | 100v | 200v | 400v | 800v | 1600v | 3200v | 6400v | 12800v |
|---|---|---|---|---|---|---|---|---|---|
| c341 (night 4) | 5.4 | 12.0 | 15.7 | 18.7 | 20.3 | 24.0 | 24.3 | 26.3 | 26.7 |
| c392 (night 5) | 6.8 | 12.3 | 14.3 | 17.7 | 22.3 | 23.0 | 24.7 | 26.3 | 26.7 |
| c439 (night 6) | 7.5 | 11.7 | 14.7 | 17.3 | 19.3 | 23.0 | 25.0 | 26.0 | 27.3 |
| c471 (night 7) | 8.1 | 12.3 | 17.3 | 18.7 | 19.7 | 22.7 | 26.0 | 26.3 | 27.0 |

- The searched rows are flat across the three models, within about ±1.5 puzzles, while the matches show c439 about +173 Elo over c341. At this size the benchmark can't separate models whose playing strength is clearly different. Only the raw policy trends up (5.4 → 6.8 → 7.5 of 29).
- Run-to-run noise is large: the visits a puzzle needs often vary 2–8× between seeds of the same model (c439 on hard life or death: 6,400 / 800 / 12,800). A single-run "needs" number, and so every per-puzzle comparison in the old tables, is mostly noise unless the gap is large.
- Changes that hold across all 3 seeds: blind16 slowly coming into reach (c341 0/3, c392 1/3, c439 2/3 at 12,800) and blindspot5 (c439 800–1,600, older models 1,600–6,400). Tactics8 is really weaker in c392 (1/3 at 12,800) than in c341 (3/3), with c439 back to 3/3.
- Use the puzzles for specific reading questions (blind16, very hard fight, tactics8), not as a strength measure. Strength comes from the matches.

Search, not the policy, solves these: the raw policy gets 2/11 for both. Puzzles
needing more than `full_visits` in training won't get correct policy targets
from self-play, so this benchmark is the signal for when to raise `full_visits`.
