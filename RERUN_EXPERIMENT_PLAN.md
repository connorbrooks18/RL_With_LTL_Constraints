# Corrected Section One experiment plan

## Research question

Can a deep recurrent Q network learn motion-planning policies for temporal
tasks when the grid position is partially observable and the automaton state
is available to the controller?

The existing Section One runs do not answer this corrected question. Their
POMDP GRU and MLP controllers hid the automaton state `q`. Junchao Li's
Chapter 4, Section 4.5.1, Task 1 explicitly uses an automaton-state sequence
alongside observation history. The old results remain useful historical
baselines, but they will not be cited as evidence for the corrected input
setting. Reference: Li, J. (2024), *Reinforcement learning-based motion
planning in partially observable environments for complex tasks*, Chapter 4,
Section 4.5.1, pp. 81-83 ([University of Iowa thesis record](https://doi.org/10.25820/etd.007512)).
The thesis is a protocol reference; the user's requested GRU, four-step
history, and visible `q` define this study. In the code history, the switch
happened when the POMDP observation was introduced in commit `4f91bb7`
(`good pomdp run`, 2026-10-09): its controller observation became
`(pair, noisy row, noisy column)` and explicitly omitted the LDBA state. The
earlier fully observed product-state experiment did include `q`. The later
one-seed q-MLP diagnostic added `q` to that MLP only; it did not change the
history-4 GRU input.

## Shared environment and inputs

- One agent in the repository's 5 x 4 safe-absorbing-states task, with
  specification `(F G a | F G b) & G !c`.
- Grid observation error remains 0.25. The agent receives the noisy row and
  column and the true current automaton state `q` at every step. The Rabin-pair
  index is also included; it is not a substitute for `q`.
- The GRU-DQN receives the last four `(pair, q, noisy row, noisy column)`
  observations with the preceding action on each token. It does not receive a
  persistent epsilon-target feature; visible `q` makes that inherited
  q-hidden-policy memory redundant. The MLP
  receives only the current `(pair, q, noisy row, noisy column)` observation.
  Both use the same q-derived legal-action mask. No true grid coordinate,
  label, reward state, or product state is passed in place of the noisy grid
  observation.
- Thesis-inspired training scale: 15,000 episodes x 600 steps, batch 32,
  (9 million environment steps per run), target-network copy every 50
  environment steps. Existing project settings
  remain: learning rate 1e-4, replay capacity 100,000, replay update every 4
  steps, epsilon floor 0.1 reached at episode 3,750, and automaton-action
  probability 0.2. The project keeps intended-move probability 0.8 and
  nonaccepting/accepting discounts 0.99999/0.99 rather than attempting to
  reproduce the thesis's LSTM, transition model, or 10 x 10 grid. The thesis
  uses observation-history length 5 and automaton-state-history length 3;
  this study keeps the requested four-step history and includes `q` at each
  of those four steps.
- All runs execute serially. Training seed is the unit of replication;
  rollout samples are paired for policy comparisons and are not counted as
  independent training replications.

## Set 1: test the recurrent architecture

### Comparison

Train history-4 GRU-DQN and a feedforward DQN that sees only the most recent
observation. Match seeds 17-21, episode and step budgets, exploration,
optimizer, replay, target updates, and evaluation scenarios. This tests the
value of observation history and recurrence when both policies see the same
current `q` and noisy grid reading.

As a non-trained reference, evaluate product-MDP value iteration using the
same visible `q` and noisy position reading. The controller indexes its
table at that observation; it is a useful classical reference, not a
belief-state POMDP solver. The full-state value-iteration result may be shown
only as an oracle context line, not as a like-for-like policy.

### Evaluation and success criterion

Evaluate greedily on all traversable starts with 100 common rollouts per
start and a 1,000-step horizon. Report discounted task return, per-seed paired
GRU-minus-MLP difference, the 95% interval across training seeds, and the
fraction of seeds favoring the GRU. Use the same rollout bank for MLP and
value-iteration comparisons. The claim that recurrence helps requires a
positive across-seed paired interval and a positive GRU-minus-MLP difference
on all five training seeds. Also report the paired GRU-minus-value-iteration
interval and seed count; claim that the GRU beats that policy only if this
interval is positive. Otherwise report the VI comparison as not established,
even if the GRU beats the MLP.

## Set 2: compare training speed and convergence

Use the same five training seeds, task, observations, 15,000-episode budget,
epsilon schedule, replay sample count, and DQN update cadence for:

1. Full GRU-DQN, reusing each Set 1 run and its 10k, 12.5k, and 15k
   checkpoints.
2. GRU-DQN trained jointly through episode 10,000, then with the GRU frozen
   and only its Q head trained through episode 15,000.
3. A compact RSSM-inspired encoder trained jointly with the same DQN. Use a
   16-unit deterministic state and two 4-class categorical variables. Train
   its observation and reward prediction losses using only the observations
   and rewards available to the agent; do not use hidden grid coordinates or
   hidden labels as inputs or targets.

Unlike the earlier RSSM pilot, the rerun uses the same replay update interval
(every 4 environment steps) and target-copy interval (every 50 steps) as the
GRU. The earlier pilot's one-eighth update frequency and 5,000-episode budget
make its timing and return unsuitable for the primary architecture
comparison.

Save and evaluate all arms at 10k, 12.5k, and 15k episodes. Plot policy
return against both episodes and measured training-loop wall time. Report
per-seed timings, mean and 95% across-seed intervals, return-versus-time
curves, and same-grid per-start return visualizations. Include total process
wall time and evaluation time separately. Use 20 common rollouts per start
with a 500-step horizon for the convergence curve; evaluate every final 15k
checkpoint on the Set 1 protocol (100 rollouts per start, 1,000-step
horizon) for the final quality comparison. Use a non-inferiority margin of
0.05 discounted-return points at 15k: a faster method preserves final quality
only if the lower bound of its paired 95% interval (method minus full GRU)
is above -0.05. Claim a speed improvement at comparable quality only when
that quality criterion passes and the paired 95% interval for training-time
savings is above zero. Also report the earliest saved checkpoint reaching
90% of that seed's full-GRU 15k return; this checkpoint comparison is
descriptive because it is evaluated at only three training budgets. A shorter
run with lower final return is a time-quality tradeoff, not a speedup at equal
quality.

## Explicit scope

The office experiment, training-length sweep, automaton-input ablation, and
the earlier freeze and RSSM pilots remain archived as prior work. They are not
rerun as part of these two claims. The office experiment in particular is a
separate transfer question, with a different task and observation model.

### Prior experimental inventory

| Prior study | Runs and result | Why it is historical for the corrected question |
|:--|:--|:--|
| Core history-4 GRU vs feedforward DQN | Ten matched seeds, 15k x 600; mean return 0.513 vs 0.086, GRU won 10/10. | Both learned policies hid `q`. |
| GRU vs noisy-input value iteration | Same ten GRU seeds; mean paired gain +0.114, GRU won 9/10. | VI inferred `q` from noisy position; the corrected comparison gives the controller true `q`. |
| Training-length sweep | Four seeds at 15k, 30k, 45k, and 90k; mean returns 0.482, 0.541, 0.585, and 0.552. | q was hidden; 45k-to-90k difference was inconclusive. |
| Office transfer | Five diagnostic seeds and seven main seeds, 10k x 300, analytical wall-signature observation; safe two-cycle probability 0.114 (main cohort). | Different transfer task; q was hidden. It does not answer the core architecture or training-speed claim. |
| GRU freeze | Seed-17 7.5k pilot; nine paired 15k runs with freeze at 10k; seed-27 stop and 25k extensions. The nine-seed study saved about 9% wall time, with quality preservation inconclusive. | q was hidden. |
| RSSM-inspired encoder | One seed, 5k episodes; best return 0.359 at 3.75k. The run used one eighth the GRU optimizer-update frequency. | q was hidden; budget and update cadence differed, so this was a feasibility pilot. |
| Automaton-input ablation | One-seed MLP q ablation: 0.105 without q and 0.104 with q under the q-hidden-style global action mask; q-visible VI was 0.533 and GRU was 0.572. An earlier MLP using true-state action legality scored 0.212 and was confounded. | Diagnostic only, single seed, and not a matched q-visible GRU-vs-MLP cohort. |

The main report also retains per-start grid values, policy maps, training
curves, attempted/aborted RSSM variants, and run-level manifests. Those remain
valuable for comparison and reproducibility, but the table above identifies
which claims the new cohorts can support.

## Run artifacts

The serial runner is `run_q_observed_section_one.py`. It writes checkpoints to
`checkpoints_q_observed_section_one/` and a resumable manifest and logs under
`artifacts/q_observed_section_one/`. Once training finishes,
`run_q_observed_evaluations.py` runs the matched evaluations, convergence
curves, standard notebook-style value grids, and policy-map gallery, all
serially. Old results and visualizations are kept; runtime source snapshots
and hashes are saved alongside the training manifest.

## Completed results (2026-11-11)

The corrected runs and evaluations are complete. Full results, input
definition, grid tables, timing data, figures, and limitations are in
[SECTION_ONE_Q_OBSERVED_RESULTS.md](SECTION_ONE_Q_OBSERVED_RESULTS.md).

- **Recurrence criterion passed:** history-4 GRU mean return 0.525 versus
  current-observation MLP 0.324; paired difference +0.201 (95% interval
  +0.076 to +0.326) across five training seeds; GRU won 5/5.
- **Value-iteration superiority not established:** q-aware noisy-input MDP
  value iteration returned 0.533; GRU minus VI was −0.008 (95% interval
  −0.134 to +0.118).
- **Frozen-GRU time was lower in these runs:** paired mean saving 1,503
  training-loop seconds (95% interval +225 to +2,781) after freezing at 10k.
  The final frozen-versus-full quality interval (−0.192 to +0.088) did not
  pass the −0.05 non-inferiority threshold.
- **RSSM speed and quality were inconclusive:** paired mean time saving −326
  seconds (95% interval −1,646 to +994); final return difference −0.022
  (95% interval −0.169 to +0.124), which also did not pass the quality
  threshold.
- Training and policy comparisons included true current `q`; only grid
  position was noisy. All arms trained serially for 15k × 600 episodes with
  five matched seeds.

The full-GRU runs preceded all frozen/RSSM runs, and full-GRU throughput
changed sharply during the campaign. Therefore, measured time savings are
descriptive; a counterbalanced timing replication is needed before making a
strong causal speed claim. See the report for exact UTC timestamps,
per-episode wall times, plots, rollout grids, and the complete manifests.
