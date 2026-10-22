# go-zero plan: Fri Oct 2 → Fri Oct 9, 2026

> **Project finished Thu Oct 22, 2026.** Final net E_c2600 (20 blocks, run E) rated **≈2854 on CGOS** after
> 161 games as `gz-c2600-4060-tm85` (Oct 21–22), up from 2630 for c439. The 3000 goal was not reached.
> Summary, data and analysis: [docs/RESULTS.md](docs/RESULTS.md). The sections below are the plan as it was
> written along the way, kept for the record.

Goal: CGOS 9x9 rating of about **3100** (above khd_N97-C ~3044, kata1_b28 ~2923, hs_j13b6a1 ~2876–3064),
staying **pure zero**: self-play only, no learning from other engines. Details and history are in NOTES.md.
This file is the schedule; update it as steps finish.

## Where things stand (Fri Oct 2, 2 pm)

- Model: c439 (night 6). c471 (night 7) tied it 100–100, so training has plateaued at lr 0.002.
- On CGOS since Fri 10 am as `gz-c439-4060-tm85` (time management, `--tm-use 0.85`). After 29 games:
  11–0 vs GNU Go / Aya, wins vs kata181b40 (2735) and rn630_945_cpu_p100 (2934), draws vs khd_N97-C
  and kata1_b28s1231v1, provisional rating ~2734.
  `python tools/cgos_report.py --blindspots runs/cgos/blindspots` gives the game report and blind-spot SGFs.
- Ready and committed: value target from search Q (`train.value_q_weight`), parallel self-play
  (`loop.selfplay_workers`), cloud scripts (`tools/cloud/`), run configs `configs/cloud_a.yaml` and
  `configs/laptop_b.yaml`, the bundle `dist/cloud_a_bundle.tar.gz` (version 4956f28), and the SSH key
  `~/.ssh/vast_gozero`.
- `tools/cgos_report.py` counts draws as half a win (commit 0618a43, with this file).

## Update Mon Oct 5

- Run C (value blend, 10 blocks) plateaued: C_c611 vs c439 103–97 (+10), C vs A_c508 100–100. The
  10×128 net is at its limit (~c439 + 25 Elo).
- Komi fix tested (C_c611 self-match, komi 7, 400 visits): `--komi-fix` alone +38 Elo (95–73, 32 draws;
  White's draws 27 → 5, Black results unchanged). `--score-utility 0.05` hurt (+3, longer games): don't use.
- **No CGOS deployment until the bot has a real chance at ~3000** (Colin, Oct 5). Deployment engine flags
  when it happens: `--time-manage --tm-use 0.85 --komi-fix --ponder --resign -0.97` (pondering tested Oct 7: 1.65x visits; resign after 3 of our moves in a row below -0.97, added Oct 7).
- **Run D** (started Mon 10:48 am): 15 blocks × 128, grown from C_c611 by `tools/deepen.py` (new blocks
  start as identities; outputs verified identical), C's buffers, `configs/laptop_d.yaml`, runs/exp_d,
  until Tue ~6 am. In-run evals vs c439 and C_c611. Tue ~6:15 am: D vs C_c611, 200 games at 400 visits
  (`runs/eval_d/after_run.py`, results in runs/eval_d/).
- **Pondering implemented** (GTP `--ponder`, `--ponder-max`): searches the tree below our move on the
  opponent's time; the opponent's reply's subtree becomes the new root. Unit + end-to-end tests pass.
  Still to do: a timed local test (needs GPU time between runs).

## Faster search (Wed Oct 7, night)

Profile (15-block D_c689, laptop_d self-play): the GPU can evaluate ~19,000 positions/s at batch 128,
self-play got 2,700/s. Two causes, both fixed:
1. **Kernel launch overhead**: an eager forward of this small net costs ~8 ms of CPU time launching ~100
   kernels. `TorchEvaluator` now replays CUDA graphs captured per padded batch size (8–256); outputs equal
   eager up to fp16 rounding, and in-place weight updates (training) are seen (checked).
2. **Lockstep self-play**: every game waited for the slowest search, so with playout cap randomization
   (75% of searches stop at 60 visits, the rest go to 1000) batches averaged ~20. `SelfPlay.step()` now
   lets each game play its move as soon as its own search is done (same targets, noise and records).

| | before | after |
|---|---|---|
| self-play, evals/s | 2,720 | **8,550** (3.1×; games: len 104, Black 40%, 25.5 samples/game, as before) |
| GTP search (one game, leaf batch 16), visits/s | 1,580 | **5,120** (3.2×; 7,190 at leaf batch 64) |

Run D resumed with it Wed 10:54 pm: **59 cycles overnight** (689 → 748, ~10 min each, ~7,200 evals/s;
the night before: 33 cycles), and the 200-game morning match ran in 15 min instead of 28. **D_c748 vs C_c611:
126–74 (63.0%, Elo +92 [+43, +142])**, the first clearly significant gain since the 10-block plateau
(D_c656 +24, D_c689 +42). In-run checks at 7:40 am: +89 vs c439, +137 vs C_c611 (64 games each).
Yardstick (KataGo@1, same openings): c439 20–80 (≈2630), D_c689 24–76 (≈2670), **D_c748 33–67 (≈2750)**.
Next: one long run Thu Oct 8 11 pm → Tue Oct 13 8:50 am (no school Fri), match vs C_c611 after, done by 10 am.
During it: yardstick on the newest snapshot daily at 2 pm Fri–Mon (`runs/yardstick/daily.py`). Sat 6 pm
plateau check (`runs/eval_d/saturday.py`, rule agreed with Colin): if both the yardstick (Fri vs Sat snapshot)
and the pooled in-run evals vs C_c611 (Fri vs Sat) gained < 25 Elo, drop lr 0.002 → 0.0006
(`configs/laptop_d_lr.yaml`; night.py restarts the loop from its save). Otherwise no change. To try next: a bigger GTP `--leaf-batch` (32–64) under CGOS time.

## Speed profile Thu Oct 8 (D_c748) and ideas

GPU per batch (forward + transfers): 8: 1.4 ms, 16: 1.8, 32: 2.6, 64: 3.6, 128: 6.7 (19k evals/s), 256: 13.7.
Self-play was 6,200 evals/s with 31% of CPU time waiting on the GPU: one `.cpu()` synchronized the whole
stream, so the two halves never overlapped. **Fixed (commit 1069264)**: pinned inputs/outputs + per-batch
CUDA events → 7,160 evals/s, now CPU-bound (~107 µs of Python tree work per eval: select_child, GoState.play
copying the 245-entry history, backprop, features, batch_priors, _submit grouping, legal masks).
Single-game (CGOS) search 5,110 → 5,490 visits/s; serial (descend 16 leaves, wait, backprop).
**Workers** (`tools/sp_bench.py`, fresh games, 60 s): 1: 10,080, **2: 13,350 (1.32×)**, 3: 8,890, 4: 9,210
(more processes fight over the GPU); 512 games in flight didn't help. Trial of the full loop with
`loop.selfplay_workers: 2` on a copy of run D (runs/_wtest) on Thu evening decides whether run D uses it.
Remaining ideas, best value first:
3. Two leaf batches in flight in `search_single` (virtual loss): CGOS ~1.3-1.5×.
4. Fold BatchNorm into conv weights for evaluation (~1/3 fewer kernels; latency-bound at batch 8-16):
   ~1.2-1.3× GPU side; self-play would re-fold after each training cycle.
5. Bigger GTP `--leaf-batch` (32: 6,150, 64: 7,190 visits/s): needs a strength test, not just speed.
6. Cheaper states: share move history via the parent chain instead of copying; batch features/legal
   masks in one numba call: ~1.2× CPU.
7. Array-based tree in numba (descent + backup compiled): 2-4× CPU; the long-term fix.

## Weekend run D results and plan to 3000 (written Tue Oct 13)

**Run D, Thu Oct 8 → Tue Oct 13** (cycles 748 → 1736, 2 self-play workers, ~12k evals/s, ~10 cycles/h):
- Plateau Thu–Sat at lr 0.002: in-run evals vs C_c611 flat at +92..+111; yardstick flat.
- Sat 18:50 lr 0.002 → 0.0006 (c1166; the automatic Saturday rule wrongly said "still climbing" because it
  compared a 200-game yardstick on new openings with a 100-game baseline; Colin switched by hand).
- Sun 16:27 `fast_visits` 60 → 150 (c1380, `configs/laptop_d_sims.yaml`).
- D_c1736 vs C_c611 (200 games, 400 visits): **160–40, +241** (D_c748: +92).
- Yardstick vs KataGo@1 (800 visits, 200 games): c1120 36.0%, c1350 36.5%, c1570 33.5% (plateau ≈ 2766),
  **c1736 41.5% ≈ 2811** [2763, 2860]. A 100-game yardstick is ±80 Elo: the same net scored 33 and 23.
  From Sat on it is 200 games; compare games 1–100 (the old openings) across nets.

**Why we lose to KataGo** (377 losses of c1350/c1570/c1736, KataGo 64 visits as referee,
`runs/yardstick/analysis/`): 97 were lost from the random 2-move opening. The other 280 turn at move ~20
(quartiles 12–31) on a median swing of only 3.5 points (46 are ≥ 10-point blunders); the 70-point final
margins come after, once every move looks equally lost. At the turning point our net thinks ~55% after the
losing move (KataGo 11%) and rates it ≥ KataGo's move in 177/280: **positional value judgment** is the main
weakness. KataGo's move has a median prior of 5.7% in our net (< 2% in 85). More search fixes most of it:
800 visits repeats the losing move 178/280, 6400 visits avoids it in 198/280.

**Step 1 result (Wed Oct 14 night, 200 games each; old openings = games 1–100, comparable with c439@800):**

| vs KataGo@1 | 800 visits | 3200 visits | gain from 4× visits |
|---|---|---|---|
| c439 (anchor) | 20/100 (−241) | old 36/100 (−100), all 80/200 (−70) | +141 |
| D_c1736 | old 42/100 (−56), all 83/200 (−60) | old 55/100 (+35), all 114/200 (+49) | +91 |

c439 gains *more* from search than D (normal: the weaker net leaves more for search to fix). D's lead over c439
shrinks from ~+185 at 800 visits to ~+120–135 at 3200, so at CGOS time D is ≈ **2750–2765**, not 2811: no hidden
bonus from search; the net itself has to get better. Engine speed is worth less than hoped (~+45 Elo per doubling
of visits for D, so 1.4× ≈ +20). Use 3200-visit yardsticks for the deploy decision.

**Plan** (Colin agreed Tue Oct 13):
1. **Strength at CGOS-like search** (Wed Oct 14 night): yardstick at 3200 visits for D_c1736 *and* c439 (the
   scale's anchor), 200 games each. If D gains more from search than c439, D is closer to 3000 at CGOS time
   (~6800 visits/move) than the 800-visit number says. Most important unknown, so first.
2. **Run D on school nights** (laptop_d_sims.yaml); drop lr 0.0006 → 0.0002 when the in-run evals and the
   yardstick (old openings, like-for-like) are both flat. Expected +30–60.
3. **Engine speed** (code by day, test at night): speed ideas #3 (two leaf batches in flight) and #4
   (BatchNorm folding) → more visits per CGOS move. Expected +30–50 at CGOS time.
4. **Run E, 20 blocks** (Fri Oct 16 → Mon): `tools/deepen.py` from D's final net (identity blocks), train all
   weekend, daily yardsticks. Capacity is the classic fix for weak value judgment; C → D (10 → 15 blocks)
   gave +241 vs C. Expected +50–100 over 1–2 weekends. Biggest single lever.
5. **Optional cloud weekend** (4090 + many CPU cores, 4–8 workers, ~3–5× throughput, < $10; Colin pays) once
   run E shows that the bigger net is the right path.
6. **Deploy** when the yardstick at CGOS-like visits says ≥ ~2950, with `--time-manage --tm-use 0.85
   --komi-fix --ponder --resign -0.97`, under a new CGOS name (cycle, GPU, search setting).

Odds: 3000 is plausible in 2–4 weeks, not certain; it depends on step 1 and on whether 20 blocks pays off.
The CGOS scale rests on c439's 71 CGOS games (±50).

## Road to 3000 (written Wed Oct 7)

**Run D so far** (15 blocks, 200 games at 400 visits vs C_c611, seed 111):
D_c656 (Tue, 45 cycles in) 107–93, +24 [−24, +72]; **D_c689 (Wed, 78 cycles in) 112–88, +42 [−6, +90]**.
Rising, but self-play Elo vs our own ancestors usually overstates gains vs other engines. Policy entropy has
crept up 1.88 → 1.98 since deepening (flatter targets, not sharper); watch it.

**Where the CGOS points went** (c439, 71 games, final ~2630; `runs/cgos/report_final.txt`): 4/34 vs bots
rated 2900+, 4/18 vs 2600–2900. 8 draws, all as White (komi bug, now fixed). 19 of 41 losses ended W+88/B+88
(passing while the opponent captured everything); most were already lost, so this only matters where it
passed early.

**Elo budget (CGOS scale, guesses):**

| Step | Gain | Cost |
|---|---|---|
| Komi fix (`--komi-fix`, done) | +20 | none |
| Pondering (`--ponder`, built, needs a timed test) | +40 to +70 | 1–2 GPU hours |
| Engine fixes 1, 2, 4 below (decided threshold, early passes, endgame batching) | +10 to +30 | CPU-time coding |
| Faster search: profile first, then CUDA graphs / TensorRT and/or the tree out of Python | +40 to +80 at play time, plus more self-play games per hour | biggest coding job |
| Bigger net + ~900 more GPU hours over 8 weeks (~190 h used so far) | +100 to +250 | nights and weekends |

Total ≈ 2850–3100: possible, but only if training keeps climbing (the 10-block net already stalled once).

**Order of work**
1. *This week:* run D continues; decide 15 blocks on Thu morning (keep it if D ≥ 55% vs C_c611 over 200
   games; D_c689 is already at 56%). Daytime: engine fixes and a search profile (no GPU). Weekend: 1–2 GPU
   hours for the pondering and speed tests under CGOS-like time (komi 7, 5 min sudden death), rest training.
2. *Weeks 2–8:* continuous training on the winning net, one 200-game check a week. Change the recipe only
   on a plateau, one change at a time: first KataGo-style forced playouts + policy target pruning, then
   20 blocks if 15 clearly helped. Any search speed-up also goes into self-play.
3. *Deploy* when the yardstick below estimates ≥ ~2950 with all engine flags.

**Measurement gap:** self-play only says "better than our old nets". Proposed yardstick (needs Colin's OK
and a download, file/source/size stated first): KataGo with a small net at a fixed visit count, used only
as an opponent to measure against, never as training data (stays pure zero). Calibrate it once against c439
(known ~2630 on CGOS), then measure each new net against it.

**Yardstick set up Wed Oct 7** (`tools/yardstick.py`, `configs/katago_yardstick.cfg`, KataGo v1.18.1 OpenCL +
kata1-b18c384nbt-s9996604416, in gitignored `tools/katago/`; results in `runs/yardstick/results.txt`).
Our bot at 800 visits, komi 7.5, same 50 opening pairs for every net. KataGo at 16 and 4 visits won 19–1
and 18–2, so the yardstick is KataGo at **1 visit** (policy only): c439 20–80 (−241) → KataGo@1 ≈ 2870 on
the c439-anchored scale. **D_c689 24–76 (−200) → ≈ 2670**, i.e. +41 over c439, matching the self-play
results. 3000 needs ~70% vs KataGo@1. The scale is anchored to c439's CGOS rating but measured at 800
visits (CGOS games averaged ~6,800), so absolute numbers are rough; differences between nets are the point.

**Pondering test Wed Oct 7** (`runs/ponder_test/`): D_c689 with the deployment flags, 60 s sudden death,
komi 7, vs KataGo@1 held for as long as our last move (new `gtp_match.py --b-delay -1`, so a "thinking"
opponent without a second engine on the GPU), same 25 opening pairs. With `--ponder`: 9.5–40.5, mean
**565 visits/move, 62% reused**. Without: 9–41, **342 visits/move, 17% reused**. So pondering gives
**~1.65× visits per move** for the same clock; the Elo effect is too small to see in 50 games (±120).
Keep `--ponder` for deployment. 39 of the 100 games ended W+88 (all as Black, our stones all captured
after passing), but in every one the bot's first pass came when it was already lost (Q −0.76 to −1.00,
nearly all ≤ −0.99): the passing costs margin, not wins, so engine fix 2 is worth ~0 rating points.
Resigning (fix 3) is the cheap way to stop the +88 games.

## Schedule (revised Fri 8 pm: no cloud, everything on the laptop)

Colin's availability: computer free all day Fri–Mon; Tue–Thu only at night (school during the day).
The Vast.ai run was dropped (payment options didn't fit), so runs A and B go one after the other on the
laptop, each from c471 with one change. CGOS was stopped Fri ~8 pm after 69 games (rating ~2634).

| When | Laptop | Claude |
|---|---|---|
| **Fri ~8:30 pm** | `python tools/sp_bench.py --config configs/laptop_a.yaml --model runs/exp_a/latest.pt --workers 1 2 4 6` (~10 min) → set `selfplay_workers` in laptop_a.yaml and laptop_b.yaml | Final CGOS report + blind spots |
| **Fri 8:21 pm → Sat ~8:40 am** | **Run A**: `python -m gozero.loop --config configs/laptop_a.yaml --run runs/exp_a --hours 12` | |
| **Sat ~8:40 am → ~8:40 pm** | **Run B**: copy runs/9x9_a (c471: latest.pt, buffer.npz, val_buffer.npz, metrics.csv) → runs/exp_b; `--config configs/laptop_b.yaml --run runs/exp_b --hours 12` | Starts automatically when A ends (`runs/exp_b_waiter.py`, log `runs/exp_b_waiter.log`; skipped with `runs/exp_b_NOT_STARTED.txt` if A crashed) |
| **Sat ~8:30 pm → ~10 pm** | Matches at 400 visits: A vs c439 and B vs c439 (200 games each), A vs B (100); 3-seed puzzles. Start automatically when B ends (`runs/eval_ab/after_run.py`; results in runs/eval_ab, DONE at the end) | Review, decide run C |
| **Sat ~9:40 pm → Mon ~8 am** | **Run C**, started automatically by `runs/eval_ab/next_steps.py`: decision in runs/eval_ab/decision.txt, config written to configs/laptop_c.yaml, folder runs/exp_c (copy of the base run), until Mon 7:40 am + the last cycle | Review the decision Sunday |
| **Mon ~8 → 9 am** | Automatic tests: C vs c439 (200 games), C vs the base run's final model (200 games), puzzles → runs/eval_c/ (DONE at the end) | Review |
| **Mon ~9 → 10 am** | Komi-fix test, chained automatically (`runs/komi_fix/after_monday.py`): C with `--komi-fix` (+ score utility 0.05, and without) vs plain C, komi 7, 400 visits, 200 games each → runs/komi_fix/ | Decide the CGOS engine flags |
| **Tue / Wed / Thu nights** | Run C continues (~10 h a night); morning tests each morning | Daytime, CPU only: engine fixes (komi first), pondering; review each morning |
| **Fri Oct 9** | Test engine fixes locally (CGOS conditions), deploy the best model to CGOS under a new name (`gz-c<cycle>-4060-tm85`, plus a suffix if the engine changed) | Config + analysis |
| **Weekend Oct 10–11 (optional)** | OGS bot (bot account flagged by a moderator, API key, gtp2ogs). 9x9 only, komi 7.5, no handicap | gtp2ogs setup (ask before downloading) |

## Experiments (both start from runs/9x9_a at cycle 471, one change each)

| Run | Folder | Config | The one change | Expected |
|---|---|---|---|---|
| A | runs/exp_a | `configs/laptop_a.yaml` | `value_q_weight: 0.5`: value target = 0.5·z + 0.5·root q (old buffer samples keep z until they age out) | +50–150 Elo (value head flat for 5 nights; CGOS losses are value misjudgments) |
| B | runs/exp_b | `configs/laptop_b.yaml` | `lr: 0.0006` | +30–80 Elo (earlier drops: +190, +81) |
| C | to be decided | written Saturday night | the winners combined | |

Decision rule (automated in runs/eval_ab/next_steps.py): a run gains if it scores >= 53% vs c439 at 400
visits. Both gain → keep both changes, from the run with more evidence ((A−B vs c439) + (A vs B − 0.5)).
Only one gains → keep its change, from its weights. Neither → continue the better run anyway and flag a
plateau (then the question is a bigger net).

Unchanged everywhere: full/fast visits 1000/60, 5000 samples per cycle, 256 games in flight, anchor c439.
`configs/cloud_a.yaml` and `tools/cloud/` stay in the repo unused (in case cloud is tried later).

## Engine work (no retraining; CGOS engine unchanged until Oct 9)

0. **Komi (confirmed bug; implemented Sun Oct 4 as `ScoreUtility`, GTP `--komi-fix` / `--score-utility`, test Mon):** the net assumes 7.5, CGOS uses 7.0, so as White it settles for a
   half-point margin that is a draw on CGOS (7 White draws, 2 of them vs GNU Go, all with Q ≈ +0.99 until
   the end). Engine-side fix: at integer komi, use the score head to require a margin of at least 1 point.
1. Count |Q| > 0.8 as decided for extensions (now 0.94): stops spending 10–15 s per move in lost positions.
2. Don't pass while the opponent is still playing in our area, and add a small score term to the search
   utility in decided positions: fixes the W+88 collapses (passing while the opponent captures everything).
3. Optional resign at Q < −0.97 for several moves (normal on CGOS/OGS).
4. Narrow-tree batching: speed drops from ~2,500 to ~110 visits/s in decided endgames because the
   16-leaf batches collide.
5. **Pondering** (search on the opponent's time): probably the biggest engine-only gain, since CGOS
   reuse is only ~20–35%.
6. Komi awareness (komi as a net input, randomized komi in self-play), if the draw vs kata1_b28s1231v1
   turns out to be the 7.0-vs-7.5 komi mismatch (check positions `1002-1203_*` in the UI).

Each engine change gets a local match under CGOS conditions (komi 7, 5 min sudden death,
`tools/gtp_match.py` or a `runs/gtp/tm_tune`-style script) before it's deployed. Each deployed version
gets a new CGOS account name: cycle, GPU and search setting.

## Standing rules

- Training, the UI, matches and CGOS all share the one GPU: never run two at once. Stop the UI before training.
- Don't start or stop runs without Colin saying so. Don't change system/power settings (suggest them instead).
- Colin enters passwords and starts the CGOS client personally, and creates the accounts and pays on Vast/OGS.
- Re-signing in to or restarting the Claude app kills processes it launched: check runs afterwards.
- Commit only when asked.
