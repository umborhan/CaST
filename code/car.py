#!/usr/bin/env python3
"""CaST: Causal-Support Transfer for safety-aware end-to-end driving (CARLA 0.9.15).

Driving interface (2026-11-03b), fresh source run required:
  --lateral-control controller (default): steering by route tracking (the route-guidance controller the
  teacher already steers with); the policy decides speed: braking for cars, pedestrians and lights,
  yielding and starting. --lateral-control policy restores learned steering.
  --light-source map_camera (default): the light token belongs to the light that governs the ego lane and
  its distance is the stop-line distance from the map; the colour comes from the camera light cluster at
  that light's pole (unknown if not seen). The earlier token was the largest light cluster any front
  camera saw, at the distance of the light head; in Town01/02 that is often another approach's light.
  v2i (map + broadcast state) is a separate, assisted setting; camera is the earlier behaviour.
  --profile fast: 150k source steps (40k teacher, 80k hand-over, 30k policy alone).
  car_eval.py must be run with the same --lateral-control and --light-source; loading refuses otherwise.

Late-braking demonstrations (2026-11-03), fresh source run required:
  --late-brake-prob 0.5 (default): in the teacher phase, half of the teacher's braking onsets for a car,
  pedestrian or red line are executed 0.4-1.4 s late (cruise throttle kept, never when the teacher could
  not still stop with 4 m/s^2 and 3 m to spare); the stored label is always the clean teacher action.
  Frozen evaluation of rgbd10 showed the policy driving on toward visible stopped cars and crossing red
  lines in 40% of its own training episodes; the early-braking teacher never produced those states.

Source-stage value fixes (2026-10-31), new defaults (fresh source run required):
  - --critic-target lcb --critic-disagreement-penalty 0.3: with the mean target the critic overrated
    the executed action by +11 over the observed entropy-adjusted return in late hand-over (rgbd9:
    Q 19.7 vs 7.7); the min-of-two target underrated it by 20-40. A small disagreement penalty sits between.
  - --source-rl-max 0.25 caps the actor's RL weight in the source stage. With the full weight, brake
    take-overs rose to 28-38% of steps while imitation batches showed no braking misses.
  - --takeover-loss-weight 1 is now the default. Learning lines print critic_spread and rl_weight.
  Earlier runs resume only with their original options (source_rl_max=1.0, their critic target).

Alignment-scale fix (2026-10-30b):
  The policy std is capped at 0.05 and gaussian_kl clamps log-std at -3, so the KL used by CaST
  alignment (Eq. 16), invariance (Eq. 25) and the uniform-transfer baseline reduced to 200 x the
  squared mean gap. On the rgbd7 checkpoints KL(source || adapted) was 226 x the squared action gap
  (mean 13.6) against a normalised RL term of about 2.5 and an imitation term of about 0.1, so these
  terms dominated every actor update. --align-metric action (default) measures the squared gap in the
  squashed action space (the units of imitation); --align-metric kl restores the old definition.
  Source checkpoints resume unchanged (no alignment in the source stage); ADAPT lines now print
  L_A, L_I and the actor's Q.

Value-learning experiment v9 (2026-10-30):
  Fresh policy/replay directory required; reuse the frozen perception/segnet.pt.
  --critic-target mean (default), lcb, or min2; the existing upper target cap remains.
  --critic-disagreement-penalty 0.1 selects the LCB coefficient (lcb mode only).
  --takeover-loss-weight 0 (default) or 1 enables synthetic terminal -5 labels
  for newly rejected handover actions; these are separate from observed transitions.
  --target-state-bc-weight 0 (default) or 2 enables privileged expert imitation
  on target states selected by --target-state-bc-threshold (default 0.35).
  Value audits log matched executed-action Q and discounted behavior returns,
  including an entropy-adjusted diagnostic. They are not frozen-policy calibration.
  Truncated/aborted trajectories are censored, not zero-bootstrap Monte Carlo targets.
  First run the mean-only source experiment, validate frozen driving, then adapt.
  LiDAR/radar are not included in this change. No driving success is guaranteed.

Preserved server recovery v8 (2026-10-29):
  Readiness and map-name RPCs run in disposable subprocesses. A native timeout,
  SIGABRT, or hanging client destructor cannot kill/block the training supervisor.
  Probe exit codes and stderr are saved in logs/carla_rpc_probe.log during startup.
  Server startup timeouts during town switches are retried up to three times.
  Dead launcher processes are detected early; logs include per-attempt diagnostics.
  Pending town requests take priority when resuming. Within v9, rerun the exact
  command and output directory to resume; v8 policy/replay cannot resume into v9.

Control and measurement corrections (2026-10-28):
  - Teacher longitudinal control respects route-guidance braking/turn slowdowns.
  - Red-light events require a forward front-bumper crossing of the finite stop line;
    stopping/restarting beyond a line and green-to-red changes after crossing do not count.
  - Robust adaptation balances town blocks and initially visits each weather twice per town;
    CaST template priorities still apply within the eligible town/weather.
  - Robust training penalizes each verified red crossing by -5 (configurable), once per event.
  - Old policy weights can be evaluated, but training requires a fresh output directory:
    old replay contains obsolete teacher labels and red-light rewards.
  - These are experimental training changes, not a promise of higher driving success.

Optional short-route protocol (2026-10-26):
  --route-length 200 selects road arc-length 200 +/- 5 m in training and car_eval.py.
  The base episode limit scales to 2,400 steps; existing traffic/near-goal extensions
  still apply. Omitting the option preserves each profile's original route protocol.
  Weights and replay remain compatible. Resuming changes only future routes and retains
  old replay; completed training stages are still skipped. This is a mixed-length run
  if earlier episodes used 500 m. Use --eval-tag town20_200m for separate evaluation.
  Source/perception budgets count steps, so shorter episodes do not reduce those budgets.
  Shorter routes change the task difficulty; higher success is not proof of better driving.

Runtime recovery patch (2026-10-26), checkpoint compatible with the learning revision below:
  --repair-town03-tm-cache --carla-root ~/carla backs up ONLY loose Town03_Opt.bin
  files under Carla/Maps/TM in the installation/client caches, then restarts CARLA.
  This addresses the reported CARLA 0.9.15 Town03_Opt Traffic Manager cache defect:
  https://github.com/carla-simulator/carla/issues/7210 . It does not change the policy,
  learning budget, rewards, routes, traffic settings, or checkpoint format.
  Persistent TM startup crashes stop after two attempts with runtime_startup.json diagnostics.

Experimental learning revision (2026-10-23), to be evaluated separately from prior paper runs:
  - One smoothly bounded Gaussian for target acting, actor updates and Bellman targets;
    stable tanh log probabilities and a cap-aware entropy target. Fresh checkpoints required.
  - Separate, stratified source expert minibatches keep rare braking/start/hold examples
    in imitation updates without changing the critic's replay sampling distribution.
  - --profile robust: 300,000 source steps (60k teacher, 120k handover, 120k policy),
    50 target adaptation episodes, 20 evaluation routes. Defaults are hypotheses, not guarantees.
  - Multiple --source-towns are visited in rotation by source training. The non-resumable
    perception collection is pinned to the FIRST source town to avoid losing its frame buffer
    on every map switch; its detector still receives source photometric augmentation.
  - Episode-averaged TD diagnostics and actual actor-update counts are logged.
  - Group-balanced source imitation (moving brake/drive, stopped start/hold), weight 2.
    The actor's RL objective is disabled during the teacher phase and ramps up during handover.
  - Source expert replay imitation during adaptation (weight 2), target actor LR 3e-5,
    policy pre-tanh standard deviation capped at 0.05. No new target expert labels are used.
  - Mean fusion of active uncertainty channels; --uncertainty-fusion sum restores the original.
    Mean fusion changes the method and is not a formal uncertainty calibration guarantee.
  - Main paper-profile budget 10. --stages perception,source,pilot runs one suite-A adaptation
    at --seed; the full sweep still includes separate proxy and Fig. 5 budgets.
Use a fresh output directory. The old source weights do not acquire these changes by resuming.
References: https://arxiv.org/abs/2106.06860 (BC regularization),
https://proceedings.mlr.press/v15/ross11a.html (DAgger), and
https://spinningup.openai.com/en/latest/algorithms/sac.html (SAC objective).
The grouping, uncertainty fusion and numerical defaults here are engineering hypotheses to test.

Implementation of "Transfer What Is Caused and Covered: Causal-Support-Aware Policy
Transfer for Safety-Aware End-to-End Driving" (main.tex). Training and evaluation are two programs:

  car.py       training only: perception pre-training, the source stage, and every target-stage
               adaptation run of the paper; checkpoints go to <out-dir>/source and <out-dir>/runs.
      python car.py --port 2200 --out-dir ./cast_results --profile paper --carla-cmd "<CarlaUE4.sh ...>"
  car_eval.py  evaluation only, one CARLA town per process, frozen checkpoints, deterministic
               actions, no gradient updates; writes the CSV files of the paper to <out-dir>/results.
      python car_eval.py --town all --port 2200 --carla-cmd "<CarlaUE4.sh ...>"

car.py runs a supervisor that runs the training in a worker process. If the worker dies (for example
a CARLA TimeoutException inside a C++ thread), it is restarted and every stage resumes from its last
checkpoint. With --carla-cmd the supervisor also restarts CARLA. The spectator follows the ego car.

Maps: client.load_world() is never used on a running server. When training needs another town it
saves its progress and exits with code 75; the supervisor starts a fresh CARLA server with that map
on its command line, and the run continues. The source stage uses one town (Town10HD).

Source stage (paper profile): 1.2x10^5 environment steps. The privileged teacher drives the first
4x10^4 steps; during the next 6x10^4 steps the policy drives and the teacher takes over only when
the policy is about to fail (HG-DAgger); the policy drives alone for the last 2x10^4 steps.
Target stage: B = 10 episodes per main adaptation run (budgets 5, 10, 20, 40 for Fig. 5).
Source training uses labels on visited source states (DAgger); target training uses SAC and
an imitation regularizer on saved source expert labels, plus the selected transfer objective.
Evaluation: 20 routes per suite, spread evenly over the towns of the suite.

Code-to-paper map
  Sec. 3.1  six surround cameras and the frozen camera-based BEV detector (entity tokens with the
            predicted detection variance, pooled BEV feature b(s)), pre-trained on source data
            CameraRig, BEVDetector, run_perception_stage, CaSTCarlaEnv._collect_entity_features
  Sec. 3.1  MDP, SAC, relational state, motion head, uncertainty
            CaSTCarlaEnv, RelationalEncoder (Eq. 4), CaSTActor.motion_head (Eq. 5),
            CaSTAgent.ubar (Eq. 6), CaSTAgent.update (Eqs. 2-3)
  Sec. 3.2  causal support map and the three sets (Eq. 7)           RunSpec, CaSTAgent.tau_c / tau_rho
  Sec. 3.3  coverage in a frozen space (Eq. 14), scene coverage (Eq. 15), gate and floor (Eq. 16),
            alignment (Eq. 17), two-channel uncertainty (Eq. 18), cautious value (Eq. 19)
            CoverageModel, SourceArtifacts.banks, gate_and_floor, CaSTAgent.update
  Sec. 3.4  margin (Eq. 20), raw effect with the privileged expert (Eq. 21), distance-matched null
            calibration (Eq. 22), reliance (Eq. 23), causal head (Eq. 24), invariance (Eq. 25)
            rollout_margin, expert_action, entity_effects, null_probe_effects, NullBank,
            CaSTAgent.reliance, CausalHead, CaSTAgent.invariance_loss
  Sec. 3.5  unresolved causal mass (Eq. 26), template sampler (Eq. 27), local counterfactual
            expansion, validated coverage growth (Eq. 28), full objective (Eq. 29)
            TemplateSampler, run_adaptation, CaSTAgent.update
  Sec. 4    suites, metrics, statistics, CSV aggregation      build_suites, evaluate_suite, aggregate

Output files (results/)
  perception_eval.csv     frozen detector on held-out source frames (recall, precision, position error)
  source_policy_eval.csv  source policy on 20 routes of each suite (source, A, L, C)
  table1_transfer.csv     Table 1 (DS with 95% bootstrap CI, SR, collisions/km, Wilcoxon vs uniform, Holm)
  table1_reference.csv    Table 1 reference rows (privileged expert, detection recall)
  fig3_reliability.csv    Fig. 3 (embedding gap for Assumption 1, failure AUROC and collisions per coverage bin)
  coverage_shift.csv      Sec. 4.2 share of causal interactions with rho < q per suite
  missed_detection.csv    Sec. 4.2 detection recall and decision uncertainty with and without the scene term
  table2_shortcut.csv     Table 2 (barrel and queue suites, identification precision and recall)
  causal_head.csv         Sec. 4.3 causal-head accuracy, matched versus far null precision, cost
  fig5_budget.csv, fig5_curve.csv   Fig. 5
  table5_ablation.csv     Table 5 (components and oracle rows, motion NLL of the penalty variant)
  fig6_sensitivity.csv    Fig. 6
  cost.csv, suites_used.csv
  Table 3 (Bench2Drive) and Table 4 (NAVSIM) need external harnesses and are not produced here.

Deviation register (the manuscript text should state these choices)
  1  Perception front end (matches Secs. 3.1 and 4.1). Six surround RGB cameras (256x128, FOV 75)
     feed a camera-based BEV detector: a shared CNN backbone, geometric lifting of image features to
     an 80 m x 64 m BEV grid at two heights, fusion with the previous frame, and centre-point heads for
     vehicles, pedestrians, and static objects (position, relative velocity, and the log-variance of
     the position, which is the detection variance sigma_i^2 of the token). A front-camera head
     predicts the state and distance of the traffic light of the ego lane. b(s) is the pooled BEV
     feature (32-d). The detector is pre-trained on source frames (Town10HD, clear and cloudy) while
     the teacher drives (perception stage) and is frozen afterwards. Camera noise, motion blur, and
     frame dropout of the combined suite are applied to the images. Lane curvature and heading
     offset in the tokens come from the HD map. Ground-truth actors are associated with detections
     only to label interventions, roles, and detection recall; they never enter the policy.
  2  Interventions. CARLA 0.9.15 cannot save and restore a world, so Eqs. (20)-(22) are evaluated on
     a privileged kinematic copy of the scene (constant-velocity agents, kinematic bicycle ego model
     replaying the recorded ego actions). In the reliance of Eq. (23) the occlusion operation paints an
     occluder over the projected 3D box of the entity in all six images and runs the frozen detector
     again; freezing, velocity shift, and signal switch, which cannot be shown by editing a single
     frame, edit the detected token directly.
  3  Expert. The privileged expert is an IDM expert in the style of PDM-Lite (IDM target speed from
     the leading vehicle, pedestrian, static object, or red light). Steering follows the route and is
     identical under every intervention, so only throttle and brake enter Eq. (21). For the reference
     row of Table 1 the expert drives with the environment's route-following steering.
  4  Attention prior. Eq. (4) uses -lambda_d ||p_i||^2 - lambda_sigma sigma_i^2 so that near and
     confidently detected entities receive more weight. The ratio -||p_i||^2/(sigma_i^2+eps) printed
     in the manuscript increases the weight of uncertain entities and contradicts its own text.
  5  alpha_0 is tuned by the standard SAC dual objective toward the target entropy -dim(A), starting
     from 0.2; Eq. (6) keeps its state-dependent factor (1 - u_bar). A fixed alpha_0 made the entropy
     term several times larger than the progress reward.
  6  Seeds. One source policy is trained; the reported seeds vary the target stage (template
     sampling, interventions, updates, and evaluation traffic).
  7  Motion target. Eq. (5) predicts the change of the relative position of entity i over 1 s
     (motion_horizon_steps = 20) in units of 2 m (motion_scale_m), so that u_i = 1 - exp(-sigma_hat^2)
     is small for predictable entities and large only for genuinely uncertain ones. The map g of
     Eq. (6) is g(v) = clip((v - median)/(q95 - median), 0, 1) on held-out source states.
  8  Baselines are adapted to online RL with the same backbone: open-set entropy (known/unknown by the
     k-NN distance of the decision feature to source decision features), contrastive filtering
     (frozen source feature distance instead of a learned contrastive encoder), optimal-transport
     filtering (Sinkhorn mass between source and target batches), ACE (per-dimension entropy weights
     from |corr(a_d, r)| instead of causal discovery), latent de-confounding (prototype backdoor
     adjustment inserted into the relational encoder).
  9  Source shortcut. Barrels precede every forced stop in 30% of source episodes
     (--barrel-source-frac); target templates contain decorrelated barrels and adjacent-lane queues.
 10  Statistics. Bootstrap over (seed, route) pairs; Wilcoxon signed-rank over paired routes.
 11  Local counterfactual expansion perturbs weather, traffic density, walker density, and
     degradation of a high-U template (CARLA scenario templates here have no occluder or timing
     parameters). Validated growth uses the finished adaptation episode itself as the validation
     episode: it must end without collision and with the margin proxy above validation_margin.
 13  Teacher warm start. The first 2x10^4 of 5x10^4 source steps are driven by the privileged teacher
     (expert of item 3 with small action noise). The actor receives a behaviour-cloning term
     ||mu_theta(s) - a_teacher||^2 on teacher transitions, weight 1 during the teacher phase and
     linearly decayed to 0 at the end of the source stage. During the source stage the actor's RL
     term is scaled by 2.5 / max(mean|Q|, 2.5) (TD3+BC style) and the entropy bonus by
     (1 - 0.9 bc_weight), so the critic, trained mostly on teacher actions, cannot pull the policy
     away from the teacher at the hand-over. All transfer methods start from this same source
     policy, so the comparison is unaffected.
 17  Imitation fixes (source stage). (a) The previous action, scalars 1-3 of the ego token, is hidden
     from the actor (the critic keeps it); otherwise behaviour cloning learns to copy it. (b) The
     teacher labels every source state with its clean action, including states the policy reaches
     itself (DAgger); behaviour cloning uses these labels. (c) While the teacher drives, temporally
     correlated steering and throttle disturbances (DART) produce recovery data. (d) After the
     teacher phase comes a hand-over phase of 4x10^4 steps (--handover-steps) in which the policy
     drives and the teacher takes over for 1.5 s whenever the policy leaves the lane, fails to
     brake where the teacher brakes hard, or stands still where the teacher would drive
     (HG-DAgger, class InterventionGuard); the last 10^4 steps are driven by the policy alone.
     Linear blending of teacher and policy actions under-steered in turns and hit kerbs and poles.
     (e) The behaviour-cloning weight stays at 1 (--bc-final), i.e. TD3+BC weighting of the RL term.
 19  Reward (the CaST manuscript only bounds it; it follows the earlier work with three changes).
     (a) Safety and uncertainty are penalties: w_s (r_s - 1) and -w_u u_bar replace w_s r_s and
     w_u (1 - u_bar); the original form paid a standing car about 80% of a driving car's reward.
     (b) Terminal outcomes: -10 for a collision, -5 for leaving the route, +5 for reaching the goal.
     (c) Progress is credited only up to the road's speed limit (superseded by item 23).
 20  Action. The policy outputs a longitudinal command in [-1, 1] (throttle if positive, brake if
     negative) and the steering command; the manuscript's three-dimensional throttle-brake-steer
     action is the vehicle command these map to. Brake can rise by 0.25 per step (full braking in
     0.2 s); the earlier 0.08 needed 0.6 s.
 21  Critic target. y uses the minimum of two randomly chosen target critics (REDQ) instead of the
     ensemble mean of Eq. (2), which overestimates Q. The actor still maximises the ensemble mean.
 22  The source stage runs interventions on 10% of visited states (2% in the target stage) so the
     causal head has enough labels.
 23  Hazard-aware progress and proximity. r_p = 1 - |v_route - v_des| / v_0, where v_des is the
     teacher's IDM following speed for the nearest obstacle, light, or predicted conflict (0 at a
     red line or behind a stopped car) and v_0 = 5 m/s is the cruise speed;
     psi_P is the violated fraction of the IDM safe following distance. With the arc-length
     progress and exp(-d/tau_p) a car waiting at a light in traffic lost about 0.25 per step, a
     2000-step route that reached the goal returned about -700 while an early crash returned
     about -30, so the policy learned to crash. Now correct waiting earns as much as driving.
 24  Teacher hazard forecast. The IDM teacher and the intervention expert also yield to moving
     vehicles and pedestrians whose 5 s constant-velocity forecast enters the ego corridor where
     the ego will be (crossing traffic, oncoming traffic during turns, cut-ins), as in PDM-Lite, and
     use the oriented footprint of each car, so a car standing at an angle in a junction blocks the
     corridor. The DART disturbances of item 17(c) are small (steer 0.04-0.12, throttle 0.1-0.25,
     0.3-1.0 s) and are never applied while the teacher brakes or an obstacle is within 25 m.
 31  "Stuck" means the ego stands still for 12.5 s while the hazard-aware reference speed of item 23
     is above 1 m/s (the teacher would drive). Waiting for a pedestrian, crossing traffic, a stopped
     car or a scripted light is not stuck. Standing obstacles block the corridor only within the
     ego's half-width plus 0.4 m (moving ones within half a lane), so cars parked at the kerb do not
     stop the teacher. Episodes that end stuck or blocked print what the teacher waited for.
 32  Teacher geometry. Obstacles use their real bounding boxes; a car standing beside the ego without
     touching it (separating-axis test) does not block the way ahead; cars of the map's parked-vehicle
     layer block only when their centre lies inside the ego lane; the scripted cones stand 1.0 m
     beyond the outer edge of the rightmost driving lane with physics off, so neither the ego nor
     traffic can be in their way. The hand-over also takes over when a standing object lies on the
     policy's own heading line less than 2 s ahead, and behaviour cloning gives braking samples the
     same total weight as cruising samples while moving. Collisions with the ground or kerb in the
     first 10 steps after spawning (the car settling) are ignored. In the source
     stage the policy acts with its mean action; the take-over thresholds are 0.8 m and 15 deg.
 33  Standstill behaviour of the teacher. The IDM standstill gap is 3 m, and the teacher holds (brake
     0.4) while it stands within 6 m of a standing obstacle or a red line instead of creeping up to
     it; the reference speed v_des of item 23 is 0 in that zone. Behaviour cloning weights "go" at
     standstill at most 3x. The source log reports lead_seen, the share of steps in which the car or
     pedestrian the teacher reacts to (within 20 m) is among the detector's tokens.
 34  Teacher transitions in every batch. After the teacher phase, each source-stage batch draws a fixed
     share from the teacher-driven transitions (teacher phase and take-overs): 0.5 at the start of the
     hand-over, decaying linearly to 0.2 at the end of the source stage (--demo-frac,
     --demo-frac-final), as in DDPGfD and RLPD; the rest comes from the policy's own driving.
 35  Target stage objective. The RL term of the actor keeps the source stage's 2.5 / mean|Q| scale and
     the entropy bonus keeps its 0.1 factor in the target stage, for every method. Before, the switch
     to the target stage multiplied the entropy bonus by 10 and removed the Q scaling, and the source
     policy degenerated within the first target episodes (stuck or colliding from episode 1 on).
 36  Detector training. Labels contain only actors within 40 m that a ray from the camera rig
     reaches (CARLA ray casting), not actors hidden behind buildings or other cars. Training frame
     pairs are taken every 5 steps over the whole perception collection (4000 pairs on disk) and the
     300 validation pairs are spread over it; 2x10^4 extra iterations (batch 16, learning rate x0.1
     for the last 30%) follow the collection. The heat-map threshold is the one with the best vehicle
     F1 on the validation pairs. perception_eval.csv adds vehicle recall at 0-15, 15-25, 25-40 m.
 37  Perception mode (--perception). camera (default): six cameras and the frozen BEV detector of
     Sec. 3.1. gt: the observation model of the CVPR 2026 code: ground-truth actors and the traffic
     light within range, with distance-, fog- and degradation-dependent dropout and Gaussian
     position/velocity noise whose level is the token variance; the scene descriptor b(s) is the
     weather, light and traffic density; occlusion reliance removes the actor's token. No perception
     stage is run in gt mode.
 30  Orphan actors. At start-up the environment removes every vehicle, pedestrian, walker controller,
     sensor and spawned prop left on the server by earlier runs that died (their actors were never
     cleaned because actor roles carry the process id), and at every reset it removes actors of
     other CaST processes. The teacher's privileged scene contains every vehicle on the server.
 25  One CARLA server run per town. A town switch restarts the server with the map on its command
     line (--carla-cmd); client.load_world() on a long-running server leaked memory and hung it.
     Without --carla-cmd one load is tried in a child process, otherwise the run waits until CARLA
     is restarted in the printed town.
 26  Temperature. alpha_0 is tuned only after the teacher phase and is capped at 0.5, with a target
     entropy of -1.5 per action dimension; behaviour cloning labels are clipped to +-0.95 so the
     squashed mean does not saturate.
 27  Comfort term. The jerk of r_c is computed from an exponentially smoothed derivative of the
     speed instead of CARLA's frame-to-frame acceleration (which jitters by several m/s^2 at 20 Hz
     and pinned r_c at its floor whenever the car moved), and r_c is bounded at -1.
 28  Behaviour cloning at standstill. Standstill samples labelled "go" and "stay" get equal total
     weight (at most 5x per sample), so the cloned policy does not learn "speed 0 -> brake"
     (the inertia problem of behaviour cloning).
 29  Training (car.py) and evaluation (car_eval.py, one CARLA town per process) are separate programs.
 38  RGB-D perception (--perception rgbd). The camera BEV detector (no depth, 8-pixel feature cells)
     reached 35% vehicle and 0.25% pedestrian recall, and the policy crept into cars it lost at 2 m.
     The RGB-D front end adds a depth camera at each RGB camera pose; a U-Net segments every RGB-D
     image (vehicle, pedestrian, prop, light state of the ego-lane light, other lights), labelled
     pixels are lifted to 3D with the depth, clustered on the ground plane into object tokens, and
     tracked for 0.2 s (velocity from a 0.5 s line fit). Labels are exact and visible-only: a pixel
     belongs to an object when its depth point lies inside the object's 3D box. Weather and night
     still degrade the RGB channels; depth is unaffected. Clusters are kept only inside the BEV range
     of the other modes, with at least 6/4/3 points (car/pedestrian/prop), mean class probability
     0.6, and at least 25% of all depth points in their ground cells; split parts of one vehicle are
     joined; a pedestrian inside a car's footprint is dropped. The light token uses the three front
     cameras at full resolution, and light-state pixels are over-sampled (a quarter of each batch).
 39  Stopping is not free. r_p falls 1.5 times faster below the safe reference speed than above it
     (standing on a free road: r_p = -0.5 instead of 0), and "stuck" (standing for 12.5 s while the
     privileged reference speed says the road is free) terminates the episode with -5 instead of
     truncating it. Without this, target-stage fine-tuning learned to stand still: zero reward, no
     collision risk, and the value of standing was bootstrapped at truncation.
 40  Critic targets are clipped at 1.05 ((w_p + alpha_max H_max) / (1 - gamma) + r_goal), with
     alpha_max = 0.5 the global temperature bound and H_max the action entropy at the log-std bound: a
     heuristic bound on the soft return (the clip acts on sampled targets; the clipped share is logged). The source stage logged Q = 130-170 against a reward-only bound of about 35.
 42  Calibrated aleatoric channel. The motion head's variance (m^2) entered u = 1 - exp(-sigma^2)
     unscaled, so the median source token gave about 0.6, u_bar was about 0.7 in the source domain
     and pinned at 1.0 in suite A, and every u_bar-dependent term (temperature, cautious value,
     reward term) was constant. sigma^2 is now divided by sig_ref, fitted on source states so that
     the median source token gives 0.1 (fitted from the source snapshot for older source agents).
 43  A scripted green light holds the other lights of its junction red for at most 20 s and never the
     light that controls the ego.
 44  Target-stage updates (every method): the critic is fitted alone for the first 2000 updates, then
     the actor is updated on every second step at a learning rate of 1e-4 (3e-4 in the source
     stage). Plain fine-tuning at the source settings left the road within one or two episodes.
 45  Bounded exploration and uncertainty-robust imitation. (a) Target-stage actions are sampled with the
     pre-tanh standard deviation capped at 0.15; the actor's log-std bound of +1 (std 2.7) gave random
     full throttle, full brake and full steering lock in rarely visited states (pre-crash traces: the
     command flipped between +1 and -1 at a stop line; fine-tuning left the road at up to 14 m/s).
     (b) In the source stage half of the training samples have every token's detection variance
     multiplied by U(1, 5) with the teacher label unchanged, so the policy brakes for a car or
     pedestrian whose detection is uncertain (rain, fog, glare and night raise exactly that variance;
     the traces showed the lead visible for 3 s, the teacher braking fully, the policy holding throttle).
 46  Reproducibility and accounting on recovery. Python, NumPy and torch (CPU and CUDA) random states
     are checkpointed with the source and adaptation stages and restored on resume; adaptation
     checkpoints after every episode. An adaptation episode aborted by a non-fatal step failure is
     repeated with the same condition (at most twice) and does not consume the budget or update the
     template sampler; its transitions and updates are kept. A third failure in a row stops the worker
     as a simulator failure (CARLA restarted, run resumed); it is never counted as an episode. The simulator's own state is not restored, so a resumed run
     is not bit-identical to an uninterrupted one. Simulator-down recoveries count toward
     --max-restarts.
 48  Token encoder: h_i = phi(x_i) is a two-layer MLP (Z units, ReLU) instead of a linear map, and the
     relational state also carries the max over valid tokens of h_i. With a linear map the pooled state
     is a weighted mean of token features, which mixes the in-path car with parked cars beside it.
     Offline brake-decision benchmark (in-path car 4-40 m among 0-8 distractors, IDM labels, 4000
     updates): decision accuracy 0.955 -> 0.986, action MAE 0.068 -> 0.029, brake recall with 4+
     other cars 0.960 -> 0.991. Saved agents load with the encoder they were trained with.
 47  Oracle-causal labels for the next state are those computed at the next step (scene and token
     slots after the transition), not a copy of the current state's labels. The scene-free decision
     uncertainty is computed without the scene term before clipping. The teacher's current-obstacle
     query (predict=False) uses only current lateral offsets.
 41  Tokens carry route-relative coordinates: the two per-token slots that repeated the ego lane
     curvature and heading offset (already scalars) hold the token's lateral offset from the route
     (in 3.5 m lane widths, clipped at 3) and its arc length along the route. The teacher decides
     whether a car blocks the ego by this offset; without it the policy confused parked or adjacent
     cars with a stopped lead car (stopping for nothing, or creeping into the lead at 0.2-1 m/s).
 18  Traffic. NPC vehicles are cars and vans only, drive 10% above to 25% below the speed limit,
     and do not change lanes. A watchdog removes an NPC vehicle that has not moved for 20 s inside a
     junction, or for 60 s elsewhere, unless it waits at a red light, and a pedestrian standing still
     on a driving lane for 20 s. An episode in which the ego vehicle cannot move for 90 s ends with
     reason "blocked". When a scripted light is green, the other lights of its junction are held red.
 16  Scripted traffic lights are held red or green only while the ego vehicle approaches; a held red
     light turns green after 3 s of waiting (or 20 s at most). TrafficLight.freeze() is not used
     because it stops every light in the map.
 14  Towns. The source domain is Town10HD only (spawn points, weather, and traffic change between
     episodes). The appearance suite A uses Town10HD under adverse weather; the layout suite L and
     the combined suite C use the unseen Town01-Town05 (Town12/13 can be added). Target-stage
     episodes run in blocks of min(5, B/4) per town (--town-block), so every budget visits several
     towns; Eq. (27) over all templates selects the town of each block and then the template inside it.
 15  Evaluation uses 20 routes per suite (Sec. 4.1 of the manuscript states 50), balanced over towns.
 12  Clearance scale d_0 = 2 m and TTC scale T_0 = 4 s in phi(s) of Eq. (20); vehicles use rectangle
     extents, so objects on the sidewalk and parallel traffic in adjacent lanes do not reduce phi.
"""


from __future__ import annotations

import argparse
import csv
import json
import glob
import math
import os
import random
import sys
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np

try:
    import gymnasium as gym
    from gymnasium import spaces
except ImportError as exc:
    raise ImportError(
        "Gymnasium is required. Install it in this environment with: "
        "python -m pip install gymnasium"
    ) from exc

import torch
import torch.nn as nn
import torch.nn.functional as F


# ======================================================
# CARLA import (0.9.15 robust)
# ======================================================
def _candidate_carla_roots() -> List[str]:
    env_root = os.environ.get("CARLA_ROOT", "").strip()
    candidates = [
        env_root,
        os.path.expanduser("~/carla"),
        os.path.expanduser("~/CARLA_0.9.15"),
        os.path.expanduser("~/CARLA_0.9.15/LinuxNoEditor"),
        os.path.expanduser("~/CARLA_0.9.14"),
        os.path.expanduser("~/CARLA_0.9.14/LinuxNoEditor"),
        os.path.expanduser("~/CARLA_0.9.13"),
        os.path.expanduser("~/CARLA_0.9.13/LinuxNoEditor"),
    ]
    return [p for p in candidates if p and os.path.exists(p)]

def _setup_carla_pythonapi() -> None:
    """
    Import CARLA without letting helper-source folders shadow a working install.

    Order:
    1) If `import carla` already works, keep it.
    2) Otherwise, try CARLA wheel/egg from PythonAPI/carla/dist.
    3) Do NOT add helper folders here; add them later after carla imports.
    """
    try:
        import carla  # noqa: F401
        return
    except Exception:
        pass

    pymaj, pymin = sys.version_info.major, sys.version_info.minor
    searched_roots: List[str] = []
    searched_dist_dirs: List[str] = []
    tried_pkgs: List[str] = []

    for root in _candidate_carla_roots():
        searched_roots.append(root)

        pythonapi = os.path.join(root, "PythonAPI")
        carla_pkg_dir = os.path.join(pythonapi, "carla")
        dist_dir = os.path.join(carla_pkg_dir, "dist")

        if not os.path.isdir(dist_dir):
            continue

        searched_dist_dirs.append(dist_dir)

        pkg_patterns = [
            os.path.join(dist_dir, f"carla-*-cp{pymaj}{pymin}-*.whl"),
            os.path.join(dist_dir, f"carla-*-py{pymaj}.{pymin}-linux-x86_64.egg"),
            os.path.join(dist_dir, "carla-*.whl"),
            os.path.join(dist_dir, "carla-*.egg"),
        ]

        matches: List[str] = []
        for pat in pkg_patterns:
            matches.extend(glob.glob(pat))

        for pkg in sorted(set(matches), reverse=True):
            tried_pkgs.append(pkg)
            if pkg not in sys.path:
                sys.path.insert(0, pkg)
            try:
                import carla  # noqa: F401
                return
            except Exception:
                try:
                    sys.path.remove(pkg)
                except ValueError:
                    pass

    raise ImportError(
        "Could not import CARLA PythonAPI.\n"
        f"Python version: {pymaj}.{pymin}\n"
        f"Searched CARLA roots: {searched_roots}\n"
        f"Searched dist dirs: {searched_dist_dirs}\n"
        f"Tried packages: {tried_pkgs}\n\n"
        "Your previous run indicates CARLA was importable before this patch, so the usual cause here is\n"
        "that helper-source paths were added too early and shadowed the working CARLA package.\n\n"
        "Fix options:\n"
        "1) Keep this function exactly as shown\n"
        "2) Make sure your working environment can do: python3 -c 'import carla; print(carla)'\n"
        "3) If needed, export CARLA_ROOT to the real CARLA install root"
    )

def _setup_carla_helper_paths() -> None:
    """
    Add helper-module paths only AFTER `import carla` succeeds.

    Needed for:
        from agents.navigation.global_route_planner import GlobalRoutePlanner
    """
    for root in _candidate_carla_roots():
        pythonapi = os.path.join(root, "PythonAPI")
        carla_pkg_dir = os.path.join(pythonapi, "carla")

        # `agents` lives under PythonAPI/carla/agents
        if os.path.isdir(carla_pkg_dir) and carla_pkg_dir not in sys.path:
            sys.path.insert(0, carla_pkg_dir)

        # Optional helper path; safe after `carla` is already imported
        if os.path.isdir(pythonapi) and pythonapi not in sys.path:
            sys.path.append(pythonapi)

_setup_carla_pythonapi()

import carla  # noqa: E402

_setup_carla_helper_paths()

try:
    from agents.navigation.global_route_planner import GlobalRoutePlanner
except Exception as e1:
    try:
        from carla.agents.navigation.global_route_planner import GlobalRoutePlanner  # type: ignore
    except Exception as e2:
        print(f"[WARN] GlobalRoutePlanner import failed: {e1}")
        print(f"[WARN] Fallback GlobalRoutePlanner import failed: {e2}")
        GlobalRoutePlanner = None
try:
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cudnn.benchmark = True
except Exception:
    pass


# ======================================================
# Config
# ======================================================
@dataclass
class Config:
    fps: int = 20
    client_timeout: float = 300.0
    tm_port: int = 8000
    seed: int = 42
    debug_mode: bool = True
    debug_step_freq: int = 20
    follow_ego_view: bool = True
    no_rendering_mode: bool = False

    car_name: str = "vehicle.tesla.model3"
    fixed_color: str = "255,0,0"

    fixed_weather: str = "night_rain_fog"
    target_weather: str = "mixed"
    # Route assistance is opt-in during learning. The default training path
    # always uses the policy command directly.
    use_safety_shield: bool = False
    mode_name: str = "train"

    # D3: practical off-policy bootstrap that is NOT described in the paper.
    # Defaults to OFF so a default run matches the manuscript's method exactly.
    # Enable with --enable-training-curriculum, and if enabled, disclose it in
    # Sec. IV-A and apply the same bootstrap to the baseline arms.
    use_training_curriculum: bool = False
    curriculum_steps: int = 60000
    curriculum_initial_blend: float = 0.95
    curriculum_final_blend: float = 0.0
    curriculum_action_noise: float = 0.06
    curriculum_traffic_steps: int = 100000

    npc_min: int = 0
    npc_max: int = 2
    # Pedestrian NPCs (paper Sec. III-C / Table I: pedestrians are first-class
    # relational entities; Fig. 2 / Fig. 12 pedestrian-crossing scenarios).
    walker_min: int = 2
    walker_max: int = 6
    train_walker_min: int = 8
    train_walker_max: int = 12
    walker_cross_factor: float = 0.6      # fraction of walkers allowed to cross roads
    walker_spawn_radius_m: float = 80.0   # bias walker spawns near the planned route
    npc_spawn_radius_m: float = 120.0     # bias NPC vehicle spawns near the planned route
    # Keep the loaded CARLA world alive between env instances and episodes:
    # resets then only destroy/respawn the ego, NPC vehicles, walkers, and
    # sensors instead of triggering a full client.load_world() map reload.
    reuse_loaded_world: bool = True
    world_settle_wait_s: float = 5.0
    max_entity_obs: int = 10
    entity_max_dist: float = 60.0

    # A route endpoint is selected automatically unless a fixed spawn-point
    # goal is explicitly supplied. Every accepted route is trimmed to about 500 m.
    use_fixed_destination: bool = False
    fixed_goal_index: int = -1
    target_goal_index: int = -1
    # Paper Sec. III-B: reference route R = {r_j} along lane centerlines.
    # This implementation uses the requested ~500 m experimental protocol.
    min_route_length_m: float = 480.0          # hard floor on arc-length
    # Euclidean prefilter only: routes wind, so the straight-line spawn-to-goal
    # distance must stay well below the 500 m arc-length target or long routes
    # become unreachable in small towns. The arc window below governs length.
    candidate_goal_min_dist_m: float = 50.0
    candidate_goal_max_tries: int = 160
    route_success_pct: float = 98.0
    strict_goal_route: bool = True
    allow_fallback_route: bool = True
    route_target_length_m: float = 500.0       # target arc-length
    route_target_tolerance_m: float = 10.0     # accepted 490--510 m
    route_soft_min_length_m: float = 490.0
    route_soft_max_length_m: float = 510.0
    prefer_auto_length_route: bool = True
    enforce_route_length_for_fixed_goal: bool = True   # fixed-length route protocol
    max_reset_start_progress_pct: float = 10.0
    max_reset_start_dL_m: float = 3.5

    # D1 (see module docstring): "reliability" implements the behavior stated
    # in the text and Table I; "paper_ratio" reproduces Eq. (6) as printed.
    attention_score_form: str = "reliability"
    attention_distance_weight: float = 4.0
    attention_uncertainty_weight: float = 2.0
    attention_ratio_eps: float = 1e-6
    # D2: the MAML meta-gradient (Eq. 14) is a first-order approximation.
    # Kept as an explicit constant so the deviation is visible in the config.
    maml_first_order: bool = True

    # Semantic classes of Sec. 3.1: vehicle, pedestrian, traffic signal, static object.
    sem_dim: int = 4
    eps_min: float = 1.5
    eps_max: float = 4.0

    tau_d: float = 1.6
    tau_p: float = 12.0
    # Paper Sec. III-D: rp = tanh(Δs/τs). Smaller τs = more signal at small steps.
    # At 18 km/h and 20 Hz, Δs≈0.25 m. τs=0.5 gives a strong immediate
    # progress signal while retaining the exact tanh form in Sec. III-D.
    tau_s: float = 0.5
    tau_v: float = 5.0   # paper Sec. III-D velocity-surrogate scale (v̂_t / τv)
    k_l: float = 0.8
    # Eq. (4) constraint penalties. phi_L is the corridor excess beyond
    # epsilon(mu_A); lambda_phi_L scales it, and the term is capped so a single
    # bad step cannot dominate the return.
    lambda_phi_L: float = 0.35
    phi_L_cap: float = 4.0
    # Explicit wrong-lane / opposite-lane components of phi_L. Being in the
    # neighbouring lane is a lane-discipline failure, not merely a large d_L.
    lambda_wrong_lane: float = 0.25
    lambda_opposite_lane: float = 0.60
    k_p: float = 0.6
    k_r: float = 0.7
    # Coefficients operate on SI jerk and steering-rate values. Values are
    # kept small enough that the comfort term does not swamp route progress.
    k_j: float = 0.002
    k_delta: float = 0.02

    # Paper Eq. (8): r_t = w_s r_s + w_p r_p + w_c r_c + w_u r_u, Σw = 1
    w_s: float = 0.45
    w_p: float = 0.30
    w_c: float = 0.15
    w_u: float = 0.10
    # Paper Sec. III-D uses either arc progress or a velocity surrogate.
    # The default is the stated arc-length form, not a mixture.
    rp_arc_weight: float = 1.0
    tl_near_dist: float = 22.0
    tl_stop_dist: float = 14.0
    # Scan range for lights the ego is approaching but has not yet entered the
    # trigger volume of. vehicle.get_traffic_light() alone gives roughly 10-15 m
    # of warning, which is shorter than the braking distance at 25 km/h.
    tl_detect_dist: float = 45.0
    # Comfortable deceleration used to derive the distance at which the red-light
    # penalty must already be active: d_brake = v^2 / (2a) + tl_stop_margin.
    tl_brake_decel: float = 3.0
    tl_stop_margin: float = 2.5
    # Perception/actuation lag folded into the onset distance so the penalty and
    # the assist both appear before braking becomes impossible.
    tl_reaction_s: float = 0.6
    # How far past the stop line a red light is still tracked, so that crossing
    # on red is detected and scored as full noncompliance (rho_t = 1).
    tl_cross_track_m: float = 8.0
    # Bounded red-light braking in the default (no-shield) path. Off by default:
    # enabling it raises the reported intervention rate, which Sec. IV-F requires
    # to be read alongside DS.
    red_light_assist: bool = False
    min_ttc_s: float = 1.8
    terminate_on_collision: bool = True
    terminate_on_offroad: bool = True

    n_critics: int = 3                   # Sec. 4.1: K = 3 critics
    critic_target: str = "lcb"
    align_metric: str = "action"          # L_A / L_I / uniform transfer: "action" (squared squashed-action gap) or legacy "kl"
    critic_disagreement_penalty: float = 0.3
    source_rl_max: float = 0.25          # cap on the actor's RL weight in the source stage (imitation decides braking)
    target_rl_max: float = 0.25          # actor RL weight in the adaptation stage (rgbd14 used 1.0: froze at spawn in rain)
    hazard_aug_prob: float = 0.25        # share of free-road imitation states given an inserted stopped/slow car or pedestrian
    hazard_aug_version: int = 2          # 1 (rgbd14d): free road only, 6-30 m; 2: also in traffic, half of them 4.8-12 m
    late_brake_prob: float = 0.5         # teacher phase: share of braking onsets demonstrated late (0 disables)
    lateral_control: str = "controller"  # "controller": route-tracking steering, the policy sets speed; "policy": learned steering
    light_source: str = "map_camera"     # light token: "map_camera" (map stop line + camera colour), "v2i", or "camera"
    light_token_version: int = 2         # map_camera token: 2 = light-head association, colour memory, flagged unknown; 1 = rgbd12
    tl_colour_hold_s: float = 2.0        # map_camera v2: a light's last seen colour is kept this long while it is not seen
    eval_light_override: bool = False    # car_eval.py diagnostic only: evaluate with a light source the policy was not trained with
    takeover_loss_weight: float = 1.0
    takeover_penalty: float = -5.0
    target_state_bc_weight: float = 0.0
    target_state_bc_threshold: float = 0.35
    value_audit_stride: int = 20
    calib_temperature: float = 1.0
    action_dim: int = 3
    gamma: float = 0.99
    target_tau: float = 5e-3
    actor_lr: float = 3e-4
    # Target stage (every method): the critic is fitted alone on the first target transitions, then the
    # actor moves at a tenth of the source learning rate and on every second update.
    balanced_adaptation: bool = False
    adapt_min_weather_episodes: int = 0
    target_actor_lr: float = 3e-5
    target_bc_weight: float = 2.0       # retain source expert behavior during target updates
    uncertainty_fusion: str = "mean"   # calibrated experiment variant; "sum" restores original fusion
    target_critic_warmup: int = 2000
    target_policy_delay: int = 2
    # Same pre-tanh standard deviation bound for acting AND all SAC policy samples.
    # Retain the CLI name for compatibility. This is a std bound, not a bound on each draw.
    target_explore_std: float = 0.05
    # Source stage: share of training samples whose token detection variances are inflated (x1-x5).
    sigma_aug_prob: float = 0.5
    critic_lr: float = 3e-4
    alpha_lr: float = 3e-4
    # Learned SAC temperature is combined with the paper's fixed
    # uncertainty-gated entropy term.
    init_alpha: float = 0.2
    # Paper Fig. 9 traces a decaying SAC temperature alongside sigma_bar, so
    # the learned temperature must be active. With it off, the effective
    # entropy coefficient is pinned at beta0 (1 - sigma_bar) and never anneals,
    # which keeps the actor near-random and produces steady lateral drift.
    use_learned_alpha: bool = True
    beta0: float = 1.0
    # lambda_ent scales L_ent in Eq. (15); the manuscript does not fix its
    # value. The effective SAC temperature is
    # lambda_ent * beta0 * (1 - sigma_bar) + alpha. At sigma_bar ~ 0.37 and
    # beta0 = 1.0, lambda_ent = 1.0 gives
    # ~0.63, several times a workable temperature for a 3-D action space, so
    # exploration noise dominates steering. 0.2 keeps the gate's dynamics while
    # putting the temperature in the usual SAC range.
    lambda_ent: float = 0.2
    lambda_transfer: float = 1.0
    lambda_alpha_align: float = 1.0
    lambda_u_align: float = 1.0
    replay_size: int = 1000000           # Sec. 4.1: replay buffer of 10^6 transitions
    batch_size: int = 512
    maml_inner_lr: float = 1e-4
    maml_inner_steps: int = 1
    maml_meta_step_size: float = 1.0
    # Keep the uncertainty mapping fixed while replay is populated; changing
    # it would silently change the reward definition for newer transitions.
    # Dense-traffic training protocol: ~20 NPC vehicles and ~10 pedestrians
    # around the route (CLI flags override these defaults).
    train_npc_min: int = 14
    train_npc_max: int = 22
    adapt_episodes: int = 100

    obs_miss_base: float = 0.08
    obs_pos_noise_m: float = 0.35
    obs_vel_noise_ms: float = 0.45
    tl_obs_noise_m: float = 1.25

    route_point_spacing_m: float = 2.0
    lookahead_base_m: float = 6.0
    lookahead_speed_gain: float = 0.35
    route_steer_kp: float = 2.6
    route_cte_kp: float = 1.95
    route_speed_kp: float = 0.13
    target_speed_kmh: float = 18.0
    min_target_speed_kmh: float = 5.5
    curve_speed_penalty_kmh: float = 11.0
    max_safe_steer_at_speed: float = 0.24
    hard_speed_cap_kmh: float = 21.0
    route_override_dL_m: float = 0.95
    route_hard_dL_m: float = 1.80
    route_override_heading_rad: float = 0.22
    route_hard_heading_rad: float = 0.38

    min_throttle_deadzone: float = 0.05
    min_brake_deadzone: float = 0.05
    low_speed_steer_limit: float = 0.18
    max_throttle_step: float = 0.06
    max_brake_step: float = 0.25          # full braking within 0.2 s (0.08 needed 0.6 s, too slow to avoid collisions)
    max_steer_step: float = 0.08

    stuck_speed_kmh: float = 1.0
    stuck_steps_threshold: int = 18
    release_duration_steps: int = 25
    release_throttle: float = 0.42
    stuck_terminate_steps: int = 250

    front_vehicle_max_dist: float = 25.0
    front_vehicle_block_dist: float = 7.5
    front_vehicle_soft_block_dist: float = 12.0
    front_vehicle_soft_block_speed_kmh: float = 1.0

    goal_reach_dist_m: float = 10.0
    goal_reach_remaining_s_m: float = 8.0

    max_route_deviation_m: float = 5.5
    offroute_grace_steps: int = 22
    offroute_heading_gate_rad: float = 0.35
    strong_offroute_factor: float = 2.0

    route_search_back: int = 2
    route_search_ahead: int = 20

    # Paper-faithful dense reward uses no terminal bonus/penalty terms.
    # These are kept as zeroed compatibility fields for older checkpoints/configs.
    goal_bonus: float = 0.0
    collision_penalty: float = 0.0
    offroute_penalty: float = 0.0
    timeout_penalty: float = 0.0

    blocked_timeout_extension_steps: int = 600
    blocked_timeout_speed_kmh: float = 1.0
    near_goal_timeout_extension_steps: int = 1000   # extra steps when within near_goal_remaining_s_m
    near_goal_remaining_s_m: float = 30.0            # "near goal" zone starts at 30 m remaining
    coast_speed_band_kmh: float = 1.5
    soft_brake_speed_excess_kmh: float = 4.0
    brake_release_threshold: float = 0.12
    center_deadband_m: float = 0.06
    center_push_gain: float = 0.55
    center_push_hard_gain: float = 0.85
    center_push_start_m: float = 0.45
    route_soft_dL_m: float = 0.85
    route_hard_dL_m_tight: float = 1.45
    # Optional bounded stabilization (Table I). Nominal policy actions
    # are untouched; assistance activates only near route/safety constraints.
    steer_guidance_blend: float = 0.35
    policy_route_blend_base: float = 0.0
    policy_route_blend_bad: float = 0.35
    policy_route_blend_hard: float = 0.85
    open_road_brake_soft_limit: float = 0.06
    open_road_brake_curve_limit: float = 0.14
    caution_ttc_s: float = 2.8
    hard_ttc_s: float = 1.8

    spectator_distance_m: float = 8.0
    spectator_height_m: float = 4.0
    spectator_pitch_deg: float = -15.0

    enable_collision_sensor: bool = True
    enable_lane_invasion_sensor: bool = False
    cleanup_sleep_s: float = 0.10
    tick_retry_count: int = 3
    tick_retry_sleep_s: float = 0.50
    post_spawn_settle_ticks: int = 2
    warmup_reset_ticks: int = 3
    rebuild_env_on_reset_failure: bool = True
    destroy_stale_owned_actors_on_reset: bool = True
    destroy_stale_spawn_blockers: bool = True
    spawn_blocker_radius_m: float = 2.5
    spawn_retry_lift_m: float = 0.35

    # 500 m at the nominal 18 km/h requires about 2000 steps at 20 Hz.
    # A 6000-step budget permits traffic-light waits and recovery without
    # allowing stalled episodes to dominate the replay buffer.
    max_episode_steps: int = 6000
    out_dir: str = "./cast_carla_0915"

    # ------------------------------------------------------------------
    # CaST (Causal-Support Transfer), Secs. 3.1-3.5 and 4.1 of main.tex
    # ------------------------------------------------------------------
    cast_q: float = 0.05                 # single level q: tau_c = 1 - q, tau_rho = q
    knn_k: int = 10                      # k of the k-NN coverage, Eq. (14)
    bank_size: int = 50000               # memory bank Z_S, Eq. (14)
    n_reference: int = 10000             # held-out reference distances, Eq. (14)
    null_total: int = 5000               # distance-matched null effects, Eq. (22)
    null_bins: int = 10                  # 10 distance bins of 5 m
    null_bin_width_m: float = 5.0
    intervention_horizon: int = 40       # H, Eq. (20)
    intervention_fraction: float = 0.02  # interventions on 2% of visited target states
    # The source stage labels the causal head; at 2% it produced under 1,000 labels in 5x10^4 steps.
    source_intervention_fraction: float = 0.10
    kappa_pess: float = 1.0              # kappa, Eq. (19)
    alpha0: float = 0.2                  # alpha_0, Eq. (6); see deviation register item 5
    eps_ordinary: float = 0.2            # epsilon, Eq. (27)
    sampler_temp: float = 0.1            # T, Eq. (27)
    lambda_A: float = 1.0                # Eq. (29)
    lambda_I: float = 0.5
    lambda_E: float = 1.0
    motion_horizon_steps: int = 20       # displacement target of Eq. (5): 1 s ahead
    motion_scale_m: float = 2.0          # displacement of Eq. (5) in units of 2 m (deviation register, item 7)
    z_dim: int = 64
    attention_relevance_hidden: int = 64
    # Token encoder: "mlp" = per-token two-layer embedding h_i = phi(x_i) and an extra max-pooled channel;
    # "linear" = h_i = W x_i with attention pooling only (Eq. 4 as written).
    token_encoder: str = "mlp"
    validation_margin: float = 0.25      # m_H safety level for validated growth, Eq. (28)
    ttc_norm_s: float = 4.0              # T_0 in phi(s), Eq. (20)
    clearance_norm_m: float = 2.0        # d_0 in phi(s), Eq. (20)
    expert_target_speed_ms: float = 5.0  # IDM desired speed (18 km/h, matches target_speed_kmh)
    frame_drop_prob_max: float = 0.15    # sensor degradation of the combined suite
    coverage_bins: Tuple[float, ...] = (0.0, 0.05, 0.10, 0.25, 0.50, 1.0001)
    # terminal rewards (a terminal transition has no future reward, so these set the outcome cost)
    # Policy action: a longitudinal command in [-1, 1] (throttle if positive, brake if negative) and the
    # steering command. Separate squashed throttle and brake outputs sit at the edge of their range
    # whenever the car is not braking, where the squashed-Gaussian log-density explodes; the entropy
    # term then pushes brake and throttle toward the middle and the temperature tuning runs away.
    policy_action_dim: int = 2
    r_collision: float = -10.0
    r_offroute: float = -5.0
    r_goal: float = 5.0
    r_stuck: float = -5.0                # standing still for stuck_terminate_steps while the road is free
    r_red_light: float = 0.0             # robust profile: -5 per verified stop-line crossing
    rp_slow_factor: float = 1.5          # r_p falls 1.5x faster below the safe reference speed than above it
    # perception front end (Secs. 3.1, 4.1)
    use_cameras: bool = True
    perception_mode: str = "camera"      # camera: frozen BEV detector | rgbd: RGB-D segmentation | gt: noisy ground truth
    use_depth: bool = False              # depth cameras at the RGB camera poses (--perception rgbd)
    rgbd_max_depth: float = 45.0         # RGB-D points farther than this are ignored
    rgbd_stride: int = 2                 # every 2nd pixel row and column is lifted to 3D at run time
    rgbd_cluster_res: float = 0.4        # ground-plane cell size of the object clustering (m)
    track_coast: int = 4                 # frames a track survives without a detection (0.2 s)
    ego_half_length: float = 2.45        # set from the ego's bounding box at spawn
    ego_half_width: float = 1.05
    cam_width: int = 256
    cam_height: int = 128
    cam_fov: float = 75.0
    bev_x_min: float = -20.0
    bev_x_max: float = 60.0
    bev_y_min: float = -32.0
    bev_y_max: float = 32.0
    bev_res: float = 1.0
    det_threshold: float = 0.35          # replaced by the F1-optimal threshold of the validation frames
    label_max_dist: float = 40.0         # detector labels: visible actors within this range
    det_lr: float = 1e-3

    @property
    def dt(self) -> float:
        return 1.0 / float(self.fps)

    @property
    def edge_dim(self) -> int:
        # [p_x, p_y, v_x, v_y, class one-hot (4), signal state, route lateral offset, route arc length, sigma^2]
        return 2 + 2 + self.sem_dim + 1 + 2 + 1

    @property
    def scene_dim(self) -> int:
        return 32          # pooled BEV feature b(s) of the detector

    @property
    def max_entities(self) -> int:
        return self.max_entity_obs

    @property
    def scalar_dim(self) -> int:
        return 13

    @property
    def model_dir(self) -> str:
        return os.path.join(self.out_dir, "models")

CFG = Config()


# ======================================================
# Utilities
# ======================================================
def ensure_dirs(cfg: Config) -> None:
    os.makedirs(cfg.out_dir, exist_ok=True)
    os.makedirs(cfg.model_dir, exist_ok=True)


def set_global_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def vec3_length(v: carla.Vector3D) -> float:
    return float(math.sqrt(v.x * v.x + v.y * v.y + v.z * v.z))


def distance2d(a: carla.Location, b: carla.Location) -> float:
    dx = float(a.x - b.x)
    dy = float(a.y - b.y)
    return float(math.sqrt(dx * dx + dy * dy))


def wrap_pi(a: float) -> float:
    return (a + math.pi) % (2.0 * math.pi) - math.pi


def point_segment_projection(p: np.ndarray, a: np.ndarray, b: np.ndarray) -> Tuple[np.ndarray, float, float]:
    ab = b - a
    ab2 = float(np.dot(ab, ab) + 1e-9)
    t = float(np.clip(np.dot(p - a, ab) / ab2, 0.0, 1.0))
    proj = a + t * ab
    dist = float(np.linalg.norm(p - proj))
    seg_len = float(np.linalg.norm(ab))
    return proj, dist, seg_len * t


# Weather presets of Sec. 4.1: source = clear and cloudy daytime; appearance
# shift = dense fog, heavy rain, night, low sun glare.
CAST_WEATHERS: Dict[str, Dict[str, float]] = {
    "clear_day": dict(cloudiness=10.0, precipitation=0.0, precipitation_deposits=0.0, wind_intensity=5.0,
                      sun_altitude_angle=60.0, sun_azimuth_angle=45.0, fog_density=0.0, fog_distance=100.0, wetness=0.0),
    "cloudy_day": dict(cloudiness=80.0, precipitation=0.0, precipitation_deposits=0.0, wind_intensity=15.0,
                       sun_altitude_angle=45.0, sun_azimuth_angle=90.0, fog_density=0.0, fog_distance=100.0, wetness=0.0),
    "fog_dense": dict(cloudiness=60.0, precipitation=0.0, precipitation_deposits=0.0, wind_intensity=5.0,
                      sun_altitude_angle=35.0, sun_azimuth_angle=90.0, fog_density=70.0, fog_distance=5.0, wetness=20.0),
    "rain_heavy": dict(cloudiness=95.0, precipitation=95.0, precipitation_deposits=90.0, wind_intensity=60.0,
                       sun_altitude_angle=30.0, sun_azimuth_angle=90.0, fog_density=10.0, fog_distance=40.0, wetness=100.0),
    "night": dict(cloudiness=30.0, precipitation=0.0, precipitation_deposits=0.0, wind_intensity=5.0,
                  sun_altitude_angle=-60.0, sun_azimuth_angle=90.0, fog_density=5.0, fog_distance=60.0, wetness=0.0),
    "glare": dict(cloudiness=0.0, precipitation=0.0, precipitation_deposits=0.0, wind_intensity=5.0,
                  sun_altitude_angle=4.0, sun_azimuth_angle=0.0, fog_density=8.0, fog_distance=60.0, wetness=0.0),
}
SOURCE_WEATHERS: Tuple[str, ...] = ("clear_day", "cloudy_day")
ADVERSE_WEATHERS: Tuple[str, ...] = ("fog_dense", "rain_heavy", "night", "glare")


def apply_weather(world: carla.World, mode: str) -> None:
    if mode == "night_rain_fog":
        weather = carla.WeatherParameters(
            cloudiness=90.0,
            precipitation=90.0,
            precipitation_deposits=85.0,
            wind_intensity=30.0,
            sun_altitude_angle=-25.0,
            fog_density=40.0,
            fog_distance=8.0,
            wetness=100.0,
        )
    elif mode == "mixed":
        weather = random.choice(
            [
                carla.WeatherParameters(
                    cloudiness=80.0,
                    precipitation=60.0,
                    precipitation_deposits=55.0,
                    wind_intensity=20.0,
                    sun_altitude_angle=-20.0,
                    fog_density=20.0,
                    fog_distance=20.0,
                    wetness=70.0,
                ),
                carla.WeatherParameters(
                    cloudiness=50.0,
                    precipitation=20.0,
                    precipitation_deposits=20.0,
                    wind_intensity=10.0,
                    sun_altitude_angle=10.0,
                    fog_density=5.0,
                    fog_distance=60.0,
                    wetness=20.0,
                ),
                carla.WeatherParameters(
                    cloudiness=95.0,
                    precipitation=80.0,
                    precipitation_deposits=75.0,
                    wind_intensity=35.0,
                    sun_altitude_angle=-10.0,
                    fog_density=25.0,
                    fog_distance=15.0,
                    wetness=90.0,
                ),
            ]
        )
    elif mode in CAST_WEATHERS:
        weather = carla.WeatherParameters(**CAST_WEATHERS[mode])
    else:
        weather = carla.WeatherParameters.Default
    world.set_weather(weather)


def get_fog_norm(world: carla.World) -> float:
    try:
        return float(np.clip(float(world.get_weather().fog_density) / 100.0, 0.0, 1.0))
    except Exception:
        return 0.0


def rng_state() -> Dict[str, object]:
    return {"py": random.getstate(), "np": np.random.get_state(), "torch": torch.get_rng_state(),
            "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None}


def load_rng_state(path: str) -> bool:
    """Restore the Python, NumPy and PyTorch (CPU and CUDA) random streams saved with a checkpoint.
    The simulator's own randomness (traffic, pedestrians) cannot be restored."""
    if not os.path.exists(path):
        return False
    try:
        st = torch.load(path, weights_only=False)
        random.setstate(st["py"])
        np.random.set_state(st["np"])
        torch.set_rng_state(st["torch"])
        if st.get("cuda") is not None and torch.cuda.is_available():
            torch.cuda.set_rng_state_all(st["cuda"])
        return True
    except BaseException as e:
        print(f"[WARN] could not restore the random state ({e}); reseeding")
        return False


def ckpt_recover(ck: str) -> None:
    """Finish an interrupted checkpoint swap (see ckpt_commit)."""
    import shutil
    new, old = ck + ".new", ck + ".old"
    if not os.path.exists(ck):
        if os.path.exists(os.path.join(new, "state.json")):
            os.rename(new, ck)
        elif os.path.exists(old):
            os.rename(old, ck)
    shutil.rmtree(new, ignore_errors=True)
    shutil.rmtree(old, ignore_errors=True)


def ckpt_begin(ck: str) -> str:
    """Checkpoint files are written into <ck>.new and swapped in only when complete, so a crash while
    writing never leaves an agent, replay buffer and state.json from different episodes."""
    import shutil
    tmp = ck + ".new"
    shutil.rmtree(tmp, ignore_errors=True)
    os.makedirs(tmp, exist_ok=True)
    return tmp


def ckpt_commit(ck: str) -> None:
    import shutil
    old = ck + ".old"
    shutil.rmtree(old, ignore_errors=True)
    if os.path.exists(ck):
        os.rename(ck, old)
    os.rename(ck + ".new", ck)
    shutil.rmtree(old, ignore_errors=True)


def _pid_alive(role: str) -> bool:
    """Role names are culrt_<pid>_<id>_<kind>; is that process still running?"""
    try:
        pid = int(str(role).split("_")[1])
        if pid == os.getpid():
            return True
        os.kill(pid, 0)
        return True
    except (ValueError, IndexError):
        return False
    except PermissionError:
        return True
    except BaseException:
        return False


def safe_float(x: object, default: float = 0.0) -> float:
    try:
        return float(x)
    except Exception:
        return default


def load_module_state_compat(module: nn.Module, state_dict: Dict[str, torch.Tensor]) -> None:
    """Load weights while tolerating scalar-state expansion."""
    model_state = module.state_dict()
    patched: Dict[str, torch.Tensor] = {}
    for key, value in state_dict.items():
        if key not in model_state:
            continue
        target = model_state[key]
        if tuple(value.shape) == tuple(target.shape):
            patched[key] = value
            continue
        if key.endswith('mlp_scal.0.weight') and value.ndim == 2 and target.ndim == 2 and value.shape[0] == target.shape[0]:
            new_weight = target.clone()
            cols = min(value.shape[1], target.shape[1])
            new_weight[:, :cols] = value[:, :cols]
            patched[key] = new_weight
    module.load_state_dict(patched, strict=False)


def resolve_existing_path(path_text: str) -> str:
    path_text = str(path_text).strip()
    if not path_text:
        return path_text
    if os.path.isabs(path_text):
        return path_text

    script_dir = os.path.dirname(os.path.abspath(__file__))
    candidates = [
        os.path.abspath(path_text),
        os.path.abspath(os.path.join(script_dir, path_text)),
        os.path.abspath(os.path.join(os.path.dirname(script_dir), path_text)),
    ]
    for cand in candidates:
        if os.path.exists(cand):
            return cand
    return candidates[0]


def resolve_output_dir(path_text: str) -> str:
    path_text = str(path_text).strip()
    if not path_text:
        script_dir = os.path.dirname(os.path.abspath(__file__))
        return os.path.abspath(os.path.join(script_dir, "culrt_carla_0915_aligned"))

    if os.path.isabs(path_text):
        return path_text

    script_dir = os.path.dirname(os.path.abspath(__file__))

    # Prefer output relative to the script location.
    script_relative = os.path.abspath(os.path.join(script_dir, path_text))
    cwd_relative = os.path.abspath(path_text)

    if os.path.exists(script_relative):
        return script_relative
    if os.path.exists(cwd_relative):
        return cwd_relative

    return script_relative



# ======================================================
# Environment
# ======================================================
class StopLineCrossingTracker:
    """Track finite stop-line crossings, not occupancy beyond a red line.

    Positions are the ego front bumper in world coordinates. Each line contains
    an actor/lane key, stop-line centre, direction, lane width and signal state
    sampled immediately before the physics step. A light is rearmed only after
    the vehicle approaches it again from at least 5 m upstream.
    """

    def __init__(self, position):
        self.previous = np.asarray(position, dtype=np.float64)
        self.crossed = set()

    def update(self, position, lines):
        current = np.asarray(position, dtype=np.float64)
        previous, self.previous = self.previous, current
        if not (np.isfinite(previous).all() and np.isfinite(current).all()):
            raise ValueError("Non-finite position in stop-line tracking")
        # A reset/teleport is not a driven crossing.
        if float(np.linalg.norm(current - previous)) > 10.0:
            self.crossed.clear()
            return []
        events, seen_lights = [], set()
        for line in lines:
            key = tuple(line['key'])
            centre = np.asarray(line['centre'], dtype=np.float64)
            direction = np.asarray(line['direction'], dtype=np.float64)
            before = float(np.dot(centre[:2] - previous[:2], direction))
            after = float(np.dot(centre[:2] - current[:2], direction))
            if after > 5.0:
                self.crossed.discard(key)
            travel = before - after
            if key in self.crossed or not (before >= 0.0 and after < 0.0 and travel > 1e-6):
                continue
            crossing = previous + (before / travel) * (current - previous)
            lateral = abs(float(np.dot(crossing[:2] - centre[:2], [-direction[1], direction[0]])))
            if lateral > 0.5 * float(line['width']) or abs(float(crossing[2] - centre[2])) > 2.0:
                continue
            self.crossed.add(key)  # remember green crossings as well
            light_id = int(line['light_id'])
            if line['state'] == 'Red' and light_id not in seen_lights:
                seen_lights.add(light_id)
                events.append({'light_id': light_id, 'stop_line_key': list(key),
                               'state': 'Red', 'position': crossing.tolist(),
                               'distance_before_m': before, 'distance_after_m': after})
        return events


class CarlaReliableTransferEnv(gym.Env):
    metadata = {"render_modes": []}

    def __init__(
        self,
        host: str = "localhost",
        port: int = 2200,
        town_name: str = "Town02",
        fixed_spawn_index: int = 0,
        fixed_goal_index: Optional[int] = None,
        weather_mode: str = "night_rain_fog",
        cfg: Config = CFG,
    ):
        super().__init__()
        self.cfg = cfg
        self.host = host
        self.port = port
        self.requested_town_name = town_name
        self.fixed_spawn_index = fixed_spawn_index
        self.fixed_goal_index = fixed_goal_index if fixed_goal_index is not None else cfg.fixed_goal_index
        self.weather_mode = weather_mode
        # Actor ownership must be unique per environment instance.  Generic
        # roles such as "hero" and "autopilot" can cause one client to delete
        # another client's actors during cleanup.
        self.owner_tag = f"culrt_{os.getpid()}_{id(self):x}".lower()
        self.ego_role = f"{self.owner_tag}_ego"
        self.npc_role = f"{self.owner_tag}_npc"
        self.walker_role = f"{self.owner_tag}_walker"

        self.client = carla.Client(host, port)
        self.client.set_timeout(cfg.client_timeout)
        cache_dir = os.path.join(os.path.abspath(cfg.out_dir), "carla_cache")
        os.makedirs(cache_dir, exist_ok=True)
        self.client.set_files_base_folder(cache_dir)
        client_version = str(self.client.get_client_version())
        server_version = str(self.client.get_server_version())
        print(f"[CARLA API] client={client_version} server={server_version} "
              f"module={getattr(carla, '__file__', '?')} requested={town_name} tm_port={cfg.tm_port}", flush=True)
        if client_version.split('-')[0] != server_version.split('-')[0]:
            raise RuntimeError(f"CARLA client/server version mismatch: {client_version} vs {server_version}. "
                               "Install the Python API matching this server before resuming.")
        self.world: carla.World = self._load_world_with_fallback(town_name)
        self.map = self.world.get_map()
        self.bp_lib = self.world.get_blueprint_library()
        self.original_settings = self.world.get_settings()
        self._apply_sync_settings()

        write_runtime_startup(cfg.out_dir, "traffic_manager_start", self.map.name, cfg.tm_port,
                              client_version, server_version)
        self.tm = self.client.get_trafficmanager(cfg.tm_port)
        self.tm.set_synchronous_mode(True)
        self.tm.set_random_device_seed(cfg.seed)
        write_runtime_startup(cfg.out_dir, "traffic_manager_ready", self.map.name, cfg.tm_port,
                              client_version, server_version)
        try:
            self.tm.set_global_distance_to_leading_vehicle(2.5)
            self.tm.set_hybrid_physics_mode(False)
        except BaseException:
            pass

        self.spawn_points = self.map.get_spawn_points()
        if not self.spawn_points:
            raise RuntimeError("No spawn points available in the loaded CARLA map.")

        self.vehicle: Optional[carla.Vehicle] = None
        self.npcs: List[carla.Vehicle] = []
        self.walkers: List[carla.Actor] = []
        self.walker_controllers: List[carla.Actor] = []
        self.sensor_list: List[carla.Actor] = []
        self.collision_events: List[object] = []
        self.lane_invasion_events: List[object] = []
        
        self._teardown_in_progress = False
        self._episode_live = False
        # Remove what earlier (crashed or stopped) runs left on this server before anything is spawned.
        self._sweep_foreign_actors(startup=True)
        self._terminal_reason = ""

        self.episode_steps = 0
        self.prev_action = np.zeros(3, dtype=np.float32)
        self.prev_steer = 0.0
        self.prev_acc = np.zeros(2, dtype=np.float32)
        self.prev_loc: Optional[carla.Location] = None
        self.distance_driven_m = 0.0

        self.stuck_steps = 0
        self.release_steps_left = 0
        self.offroute_steps = 0
        self.prev_route_s: Optional[float] = None
        self.safety_interventions = 0
        self.blocked_steps_credit = 0

        self.route_xy: List[Tuple[float, float]] = []
        self.route_wps: List[carla.Waypoint] = []
        self.route_cumdist: List[float] = []
        self.goal_transform: Optional[carla.Transform] = None
        self.goal_spawn_transform: Optional[carla.Transform] = None
        self.current_goal_index: Optional[int] = None
        self.current_spawn_index: Optional[int] = None
        self.grp = None
        self.route_progress_idx = 0
        self.route_total_len_m = 0.0

        self.action_space = spaces.Box(
            low=np.array([0.0, 0.0, -1.0], dtype=np.float32),
            high=np.array([1.0, 1.0, 1.0], dtype=np.float32),
            dtype=np.float32,
        )
        self.observation_space = spaces.Dict(
            {
                "scalars": spaces.Box(low=-10.0, high=10.0, shape=(cfg.scalar_dim,), dtype=np.float32),
                "edges": spaces.Box(low=-10.0, high=10.0, shape=(cfg.max_entities, cfg.edge_dim), dtype=np.float32),
                "mask": spaces.Box(low=0.0, high=1.0, shape=(cfg.max_entities,), dtype=np.float32),
            }
        )

        self._safe_tick(label="env_init")
        if self.cfg.debug_mode:
            print(
                f"[OK] Connected to CARLA | host={self.host} port={self.port} "
                f"town={self.town_name} sync=True dt={self.cfg.dt:.3f}s weather={self.weather_mode} "
                f"fixed_goal_index={self.fixed_goal_index}"
            )

    def _available_map_names(self) -> List[str]:
        try:
            return list(self.client.get_available_maps())
        except BaseException:
            return []

    @staticmethod
    def _basename_map(map_path: str) -> str:
        if not map_path:
            return map_path
        return map_path.split("/")[-1].split(".")[0]

    def _resolve_town_name(self, requested: str) -> str:
        available = self._available_map_names()
        available_basenames = {self._basename_map(x): x for x in available}
        preferred: List[str] = []
        if requested:
            preferred.append(requested)
            if requested.endswith("_Opt"):
                preferred.append(requested.replace("_Opt", ""))
            else:
                preferred.append(f"{requested}_Opt")
        preferred.extend(["Town02", "Town03", "Town10HD_Opt", "Town10HD"])
        for name in preferred:
            if name in available_basenames:
                return available_basenames[name]
        return available[0] if available else requested

    def _load_world_with_fallback(self, requested_town: str) -> carla.World:
        resolved = self._resolve_town_name(requested_town)
        # Reuse the already-loaded world when it matches the requested town.
        # client.load_world() performs a full map reload (a "total reset" of
        # the simulator); reusing the world means episode resets only
        # destroy/respawn the ego, NPC vehicles, pedestrians, and sensors.
        if bool(getattr(self.cfg, "reuse_loaded_world", True)):
            try:
                current = self.client.get_world()
                current_name = self._basename_map(current.get_map().name)
                if current_name == self._basename_map(resolved):
                    self.town_name = current_name
                    if self.cfg.debug_mode:
                        print(f"[OK] Reusing loaded CARLA world: {current_name} (no map reload)")
                    return current
            except BaseException:
                pass
        try:
            world = self.client.load_world(resolved, reset_settings=False)
            self.town_name = self._basename_map(resolved)
            return world
        except Exception as e:
            print(f"[WARN] Failed to load {resolved}: {e}")
            fallback = self._resolve_town_name("Town02")
            world = self.client.load_world(fallback, reset_settings=False)
            self.town_name = self._basename_map(fallback)
            return world

    def _apply_sync_settings(self) -> None:
        settings = self.world.get_settings()
        settings.synchronous_mode = True
        settings.fixed_delta_seconds = self.cfg.dt
        settings.no_rendering_mode = bool(self.cfg.no_rendering_mode)
        self.world.apply_settings(settings)

    def _restore_world_settings(self) -> None:
        try:
            self.tm.set_synchronous_mode(bool(self.original_settings.synchronous_mode))
        except BaseException:
            pass
        try:
            # Restore the exact settings that existed before this environment
            # changed them; forcing asynchronous mode corrupts nested/restarted
            # training environments that share a server.
            self.world.apply_settings(self.original_settings)
        except BaseException:
            pass

    def _make_spectator_transform(self, vehicle_tf: carla.Transform) -> carla.Transform:
        yaw = math.radians(vehicle_tf.rotation.yaw)
        cam_loc = carla.Location(
            x=vehicle_tf.location.x - self.cfg.spectator_distance_m * math.cos(yaw),
            y=vehicle_tf.location.y - self.cfg.spectator_distance_m * math.sin(yaw),
            z=vehicle_tf.location.z + self.cfg.spectator_height_m,
        )
        cam_rot = carla.Rotation(
            pitch=self.cfg.spectator_pitch_deg,
            yaw=vehicle_tf.rotation.yaw,
            roll=0.0,
        )
        return carla.Transform(cam_loc, cam_rot)

    def _snap_spectator_to_ego(self) -> None:
        if self.vehicle is None:
            return
        try:
            spectator = self.world.get_spectator()
            spectator.set_transform(self._make_spectator_transform(self.vehicle.get_transform()))
        except BaseException:
            pass

    def _update_spectator(self) -> None:
        if self.vehicle is None or not self.cfg.follow_ego_view:
            return
        self._snap_spectator_to_ego()

    def _safe_tick(self, label: str = "tick", raise_on_fail: bool = True) -> bool:
        last_err: Optional[Exception] = None
        for attempt in range(1, self.cfg.tick_retry_count + 1):
            try:
                self.world.tick()
                return True
            except BaseException as e:
                last_err = Exception(str(e))
                if "time-out" in str(e).lower():
                    # The server did not answer for the whole client time-out: retrying waits again.
                    self._server_down = True
                    raise SimulatorDown(f"CARLA world tick failed during {label}: {e}")
                if self.cfg.debug_mode:
                    print(f"[WARN] {label} failed attempt {attempt}/{self.cfg.tick_retry_count}: {e}")
                time.sleep(self.cfg.tick_retry_sleep_s)
        if raise_on_fail and last_err is not None:
            raise RuntimeError(f"CARLA world tick failed during {label}: {last_err}")
        return False

    def _destroy_actor(self, actor: carla.Actor) -> None:
        if actor is None:
            return
        try:
            if getattr(actor, "is_alive", False):
                actor.destroy()
        except BaseException:
            pass

    def _stop_sensor_only(self, actor: carla.Actor) -> None:
        if actor is None:
            return
        try:
            if getattr(actor, "is_alive", False) and bool(getattr(actor, "is_listening", True)):
                actor.stop()
        except BaseException:
            pass

    def _batch_destroy_actors(self, actors: List[carla.Actor]) -> None:
        """
        Safe destroy for synchronous mode:
        - detach TM first for vehicles
        - destroy one-by-one
        - tick between small groups
        """
        alive: List[carla.Actor] = [a for a in actors if self._actor_is_alive(a)]
        if not alive:
            return

        for actor in alive:
            try:
                if "vehicle." in getattr(actor, "type_id", ""):
                    self._safe_set_autopilot(actor, False)
            except BaseException:
                pass

        self._drain_world_ticks(1, "batch_detach_tm", sleep_s=0.02)
        # Destroy through server-side commands, filtered against a live actor
        # snapshot. Per-actor destroy() on a cached handle is what produced the
        # "unable to destroy actor: not found" lines during cleanup.
        self._batch_destroy_ids(alive, label="residual_actors")


    def describe_wait(self) -> str:
        """What the teacher is waiting for (printed when an episode ends stuck or blocked)."""
        try:
            scene = self.privileged_scene()
            gap, v_lead, eid = expert_lead(scene, scene.entities, v_cruise=self.cfg.expert_target_speed_ms)
            if eid == -1 or gap >= 1e8:
                return f"wait=nothing,v_des>0,ego={scene.ego_v:.1f}m/s"
            e = next(x for x in scene.entities if x.eid == eid)
            _, d, _ = route_frame(scene, np.array([e.x]), np.array([e.y]))
            static_eid = expert_lead(scene, scene.entities, v_cruise=self.cfg.expert_target_speed_ms, predict=False)[2]
            if e.kind == KIND_SIGNAL:
                what = f"light_{e.tl_state}"
                if e.tl_state == "Red":
                    what += f"(red_for={getattr(self, '_red_still', 0) * float(self.cfg.dt):.0f}s)"
            elif eid <= -10000:
                what = "map_parked_car"
            elif e.kind == KIND_WALKER:
                what = "pedestrian"
            elif e.kind == KIND_STATIC:
                what = "prop"
            else:
                what = "car" if any(int(n.id) == eid for n in self.npcs if self._actor_is_alive(n)) else "foreign_car"
            if static_eid != eid:
                what += "_predicted_crossing"
            return (f"wait={what},gap={gap:.1f}m,lat={float(d[0]):+.1f}m,"
                    f"its_speed={math.hypot(e.vx, e.vy):.1f}m/s")
        except BaseException as ex:
            return f"wait=unknown({type(ex).__name__})"

    def _map_parked_vehicles(self) -> List[Tuple[float, float, float, float, float]]:
        """Parked cars of the map layers (environment objects, not actors): (x, y, yaw, half-length,
        half-width), read once."""
        cached = getattr(self, "_parked_cache", None)
        if cached is not None:
            return cached
        out: List[Tuple[float, float, float]] = []
        for name in ("Car", "Truck", "Bus", "Motorcycle", "Bicycle", "Vehicles"):
            label = getattr(getattr(carla, "CityObjectLabel", None), name, None)
            if label is None or (name == "Vehicles" and out):
                continue                     # "Vehicles" is the pre-0.9.14 label covering all of them
            try:
                for o in self.world.get_environment_objects(label):
                    bb = o.bounding_box
                    out.append((float(bb.location.x), float(bb.location.y), math.radians(float(bb.rotation.yaw)),
                                float(bb.extent.x), float(bb.extent.y)))
            except BaseException:
                continue
        self._parked_cache = out
        return out

    def _sweep_foreign_actors(self, startup: bool) -> int:
        """Destroy actors that belong to no live CaST process.

        Actor roles carry the process id, so the per-episode cleanup only removes this process's own
        actors. When a training process dies (segfault -11, time-out, Ctrl+C) its cars, pedestrians,
        cameras and barrels stay on the server: the cars stand still in their lanes without a
        Traffic Manager, and the next process neither knew about them nor removed them, so the
        teacher drove into them ("collision(vehicle,front,...,other=0.0m/s)"). At start-up every
        vehicle, pedestrian, walker controller, sensor and spawned prop is removed (this process owns
        none yet); at every reset, actors of other CaST processes are removed."""
        try:
            actors = list(self.world.get_actors())
        except BaseException:
            return 0
        own = set()
        for group in (getattr(self, "npcs", []), getattr(self, "walkers", []),
                      getattr(self, "walker_controllers", []), getattr(self, "sensor_list", []),
                      getattr(self, "props", []), [getattr(self, "vehicle", None)]):
            for a in group or []:
                try:
                    own.add(int(a.id))
                except BaseException:
                    pass
        kill_ctrl, kill = [], []
        for a in actors:
            try:
                tid = str(a.type_id)
                if not tid.startswith(("vehicle.", "walker.", "sensor.", "controller.ai.walker", "static.prop.")):
                    continue
                if int(a.id) in own:
                    continue
                role = self._role_name_of_actor(a)
                if role.startswith(self.owner_tag):
                    continue
                if startup or (role.startswith("culrt_") and not _pid_alive(role)):
                    (kill_ctrl if tid.startswith("controller.") else kill).append(a)
            except BaseException:
                continue
        if not kill and not kill_ctrl:
            return 0
        for c in kill_ctrl:
            try:
                c.stop()
            except BaseException:
                pass
        try:
            self.client.apply_batch_sync([carla.command.DestroyActor(int(a.id)) for a in kill_ctrl + kill], False)
        except BaseException:
            for a in kill_ctrl + kill:
                try:
                    a.destroy()
                except BaseException:
                    pass
        self._safe_tick(label="sweep_foreign_actors", raise_on_fail=False)
        n = len(kill) + len(kill_ctrl)
        print(f"[ENV] removed {n} actors left on the server by earlier runs "
              f"({sum(1 for a in kill if str(a.type_id).startswith('vehicle.'))} vehicles)")
        return n

    def _role_name_of_actor(self, actor: carla.Actor) -> str:
        try:
            return str(actor.attributes.get("role_name", "")).strip().lower()
        except BaseException:
            return ""

    def _spawn_transform_variants(self, tf: carla.Transform) -> List[carla.Transform]:
        base = carla.Transform(
            carla.Location(tf.location.x, tf.location.y, tf.location.z),
            carla.Rotation(tf.rotation.pitch, tf.rotation.yaw, tf.rotation.roll),
        )
        dz = float(max(self.cfg.spawn_retry_lift_m, 0.05))
        variants = [base]
        for mul in (1.0, 2.0):
            variants.append(
                carla.Transform(
                    carla.Location(base.location.x, base.location.y, base.location.z + dz * mul),
                    carla.Rotation(base.rotation.pitch, base.rotation.yaw, base.rotation.roll),
                )
            )
        return variants

    def _destroy_residual_owned_actors(self) -> int:
        # Flush pending destroys so the actor snapshot below is not stale;
        # on the reused world a stale snapshot after _cleanup_episode_actors
        # produces "already dead" / "not found" server warnings.
        self._safe_tick(label='pre_residual_sweep', raise_on_fail=False)
        actors_to_destroy: List[carla.Actor] = []
        try:
            actors = self._world_actors()
            if hasattr(actors, 'filter'):
                iterable = list(actors.filter('vehicle.*'))
            else:
                iterable = [a for a in actors if 'vehicle.' in getattr(a, 'type_id', '')]
            for actor in iterable:
                if actor is None:
                    continue
                try:
                    if self.vehicle is not None and actor.id == self.vehicle.id:
                        continue
                except BaseException:
                    pass
                role = self._role_name_of_actor(actor)
                if role in {self.ego_role, self.npc_role}:
                    actors_to_destroy.append(actor)
            # Residual pedestrians and their AI controllers left on the reused
            # world by a previous episode or env instance.
            if hasattr(actors, 'filter'):
                controller_iter = list(actors.filter('controller.ai.walker'))
                walker_iter = list(actors.filter('walker.pedestrian.*'))
            else:
                controller_iter = [a for a in actors if 'controller.ai.walker' in getattr(a, 'type_id', '')]
                walker_iter = [a for a in actors if 'walker.pedestrian' in getattr(a, 'type_id', '')]
            for ctrl in controller_iter:
                if ctrl is None or not self._actor_is_alive(ctrl):
                    continue
                try:
                    parent = ctrl.parent
                except BaseException:
                    parent = None
                if parent is None or self._role_name_of_actor(parent) != self.walker_role:
                    continue
                try:
                    ctrl.stop()
                except BaseException:
                    pass
                actors_to_destroy.append(ctrl)
            for wk in walker_iter:
                if wk is None or not self._actor_is_alive(wk):
                    continue
                if self._role_name_of_actor(wk) == self.walker_role:
                    actors_to_destroy.append(wk)
        except BaseException:
            return 0
        if not actors_to_destroy:
            return 0
        self._batch_destroy_actors(actors_to_destroy)
        time.sleep(self.cfg.cleanup_sleep_s)
        self._safe_tick(label='destroy_residual_owned_actors', raise_on_fail=False)
        return len(actors_to_destroy)

    def _clear_spawn_blockers(self, tf: carla.Transform, radius: Optional[float] = None) -> int:
        radius = float(self.cfg.spawn_blocker_radius_m if radius is None else radius)
        actors_to_destroy: List[carla.Actor] = []
        try:
            actors = self._world_actors()
            if hasattr(actors, 'filter'):
                iterable = list(actors.filter('vehicle.*'))
            else:
                iterable = [a for a in actors if 'vehicle.' in getattr(a, 'type_id', '')]
            for actor in iterable:
                if actor is None:
                    continue
                try:
                    if self.vehicle is not None and actor.id == self.vehicle.id:
                        continue
                except BaseException:
                    pass
                try:
                    if distance2d(actor.get_location(), tf.location) <= radius:
                        actors_to_destroy.append(actor)
                except BaseException:
                    continue
        except BaseException:
            return 0
        if not actors_to_destroy:
            return 0
        self._batch_destroy_actors(actors_to_destroy)
        time.sleep(self.cfg.cleanup_sleep_s)
        self._safe_tick(label='clear_spawn_blockers', raise_on_fail=False)
        return len(actors_to_destroy)

    def _cleanup_episode_actors(self) -> None:
        """
        Robust synchronous cleanup.

        Order:
        1. Mark teardown active, stop sensors, detach TM.
        2. Drain queued callbacks.
        3. Destroy sensors.
        4. Destroy ego.
        5. Destroy NPCs after TM detach.
        6. Final settle ticks.
        """
        self._begin_episode_teardown(reason="cleanup")

        sensors = list(self.sensor_list)
        self.sensor_list.clear()

        ego = self.vehicle
        self.vehicle = None

        npcs = list(self.npcs)
        self.npcs.clear()

        walker_controllers = list(self.walker_controllers)
        self.walker_controllers.clear()
        walkers = list(self.walkers)
        self.walkers.clear()

        # Extra drain after stop() because goal often arrives at nonzero speed.
        self._drain_world_ticks(3, "cleanup_drain_pre", sleep_s=0.03)

        # Destroy sensors directly. The collision/lane-invasion callbacks are
        # already gated by _teardown_in_progress/_episode_live, and destroy()
        # unsubscribes the active stream itself; calling stop() first makes
        # the later destroy emit CARLA's "attempting to unsubscribe ...
        # wasn't listening" warning, so the explicit stop is skipped.
        self._batch_destroy_ids(sensors, label="sensors")

        self.collision_events.clear()
        self.lane_invasion_events.clear()
        self._drain_world_ticks(2, "cleanup_post_sensor_destroy", sleep_s=0.03)

        # Destroy ego next.
        if self._actor_is_alive(ego):
            self._safe_set_autopilot(ego, False)
            try:
                ego.apply_control(
                    carla.VehicleControl(
                        throttle=0.0,
                        steer=0.0,
                        brake=1.0,
                        hand_brake=False,
                        reverse=False,
                        manual_gear_shift=False,
                    )
                )
            except BaseException:
                pass

        self._batch_destroy_ids([ego], label="ego_vehicle")
        self._drain_world_ticks(2, "cleanup_post_ego", sleep_s=0.03)

        # Detach TM from NPCs before destroying them.
        alive_npcs = [npc for npc in npcs if self._actor_is_alive(npc)]
        for npc in alive_npcs:
            self._safe_set_autopilot(npc, False)

        self._drain_world_ticks(2, "cleanup_post_tm_detach", sleep_s=0.03)

        # Destroy NPC vehicles through the batch-command path so ids the
        # server has already collected are skipped without error output.
        self._batch_destroy_ids(alive_npcs, label="npc_vehicles")

        # Stop walker AI controllers, then destroy controllers and walkers
        # through the batch-command path. Ids the server has already collected
        # (e.g. a walker killed mid-episode, which auto-removes its attached
        # controller) are filtered out against a live snapshot first.
        live_ids = self._live_actor_ids()
        for ctrl in walker_controllers:
            try:
                if int(ctrl.id) not in live_ids:
                    continue
                ctrl.stop()
            except BaseException:
                pass
        self._drain_world_ticks(1, "cleanup_post_walker_stop", sleep_s=0.02)
        self._batch_destroy_ids(walker_controllers, label="walker_controllers")
        self._batch_destroy_ids(walkers, label="walkers")

        self._drain_world_ticks(3, "cleanup_final", sleep_s=0.03)

        try:
            time.sleep(max(self.cfg.cleanup_sleep_s, 0.05))
        except BaseException:
            pass

        self._terminal_reason = ""
        self._teardown_in_progress = False

    def _get_global_planner(self):
        if GlobalRoutePlanner is None:
            return None
        if self.grp is None:
            self.grp = GlobalRoutePlanner(self.map, self.cfg.route_point_spacing_m)
        return self.grp
        
    def _rebuild_local_route_from_current_pose(self) -> bool:
        if self.vehicle is None:
            return False
        cur_wp = self.map.get_waypoint(
            self.vehicle.get_location(), project_to_road=True,
            lane_type=carla.LaneType.Driving,
        )
        if cur_wp is None:
            return False
        self.route_progress_idx = 0
        target_with_margin = self.cfg.route_target_length_m + max(
            2.0 * self.cfg.route_point_spacing_m,
            self.cfg.route_target_tolerance_m,
        )
        route = self._extend_waypoint_chain(
            cur_wp, target_with_margin, ds=self.cfg.route_point_spacing_m
        )
        try:
            self._install_route(route, goal_index=None)
        except RuntimeError:
            return False
        return True

    def _route_length_from_wps(self, wps: Sequence[carla.Waypoint]) -> float:
        if len(wps) < 2:
            return 0.0
        s = 0.0
        for i in range(len(wps) - 1):
            s += distance2d(wps[i].transform.location, wps[i + 1].transform.location)
        return float(s)

    def _route_length_ok(self, route_len: float, strict: bool = False) -> bool:
        route_len = float(route_len)
        if strict:
            return abs(route_len - self.cfg.route_target_length_m) <= self.cfg.route_target_tolerance_m
        return self.cfg.route_soft_min_length_m <= route_len <= self.cfg.route_soft_max_length_m

    def _route_length_score(self, route_len: float) -> float:
        return -abs(float(route_len) - float(self.cfg.route_target_length_m))

    def _dedupe_waypoints(self, wps: Sequence[carla.Waypoint], min_sep: float = 0.75) -> List[carla.Waypoint]:
        out: List[carla.Waypoint] = []
        prev: Optional[carla.Waypoint] = None
        for wp in wps:
            if prev is None or distance2d(prev.transform.location, wp.transform.location) > min_sep:
                out.append(wp)
                prev = wp
        return out

    def _compute_route_cumdist(self) -> None:
        self.route_cumdist = []
        s = 0.0
        for i, wp in enumerate(self.route_wps):
            if i == 0:
                self.route_cumdist.append(0.0)
            else:
                s += distance2d(self.route_wps[i - 1].transform.location, wp.transform.location)
                self.route_cumdist.append(float(s))
        self.route_total_len_m = float(self.route_cumdist[-1]) if self.route_cumdist else 0.0

    def _route_projection_global(self, loc: carla.Location) -> Tuple[float, float, int]:
        if len(self.route_xy) < 2:
            return 0.0, 0.0, 0
        p = np.array([loc.x, loc.y], dtype=np.float32)
        best_d = 1e9
        best_s = 0.0
        best_idx = 0
        for i in range(len(self.route_xy) - 1):
            a = np.array(self.route_xy[i], dtype=np.float32)
            b = np.array(self.route_xy[i + 1], dtype=np.float32)
            _, d, local_s = point_segment_projection(p, a, b)
            if d < best_d:
                best_d = d
                best_s = self.route_cumdist[i] + local_s
                best_idx = i
        return float(best_s), float(best_d), int(best_idx)


    def _route_projection_start_window(self, loc: carla.Location, window_frac: float = 0.15) -> Tuple[float, float, int]:
        """Project onto the first window_frac of the route only.

        At episode start the ego is at the route origin by construction. A
        lane-follow long route can loop back near its own start (city blocks),
        so an unrestricted global projection may match a late segment and
        report a large spurious start progress. Restricting the search to the
        opening window makes the reset alignment check reflect the ego's
        actual position on the route.
        """
        if len(self.route_xy) < 2:
            return 0.0, 0.0, 0
        n_seg = len(self.route_xy) - 1
        hi = int(np.clip(int(math.ceil(window_frac * n_seg)), 1, n_seg))
        p = np.array([loc.x, loc.y], dtype=np.float32)
        best_d = 1e9
        best_s = 0.0
        best_idx = 0
        for i in range(hi):
            a = np.array(self.route_xy[i], dtype=np.float32)
            b = np.array(self.route_xy[i + 1], dtype=np.float32)
            _, d, local_s = point_segment_projection(p, a, b)
            if d < best_d:
                best_d = d
                best_s = self.route_cumdist[i] + local_s
                best_idx = i
        return float(best_s), float(best_d), int(best_idx)

    def _route_projection_monotonic(self, loc: carla.Location) -> Tuple[float, float, int, np.ndarray, np.ndarray]:
        """Project the ego onto the route.

        Returns (arc_s, cross_track, segment_index, projection_point, tangent).

        cross_track is pure geometry: the distance to the nearest admissible
        segment in the local window. It is never rate limited, because it is a
        distance rather than a rate; limiting it was what froze the projection
        and produced phantom off-route terminations.

        arc_s is monotonic and rate limited to what the vehicle can physically
        cover in one control step, so a route that loops back near itself cannot
        teleport progress. When the geometric match lies beyond the budget, arc_s
        slews toward it rather than rejecting it, so it always catches up.
        """
        if len(self.route_xy) < 2:
            return 0.0, 0.0, 0, np.zeros(2, dtype=np.float32), np.array([1.0, 0.0], dtype=np.float32)
        p = np.array([loc.x, loc.y], dtype=np.float32)
        start_idx = max(0, self.route_progress_idx - self.cfg.route_search_back)
        end_idx = min(len(self.route_xy) - 2, self.route_progress_idx + self.cfg.route_search_ahead)
        ego_forward = np.array([1.0, 0.0], dtype=np.float32)
        speed_ms = 0.0
        if self.vehicle is not None:
            tf = self.vehicle.get_transform()
            yaw = math.radians(tf.rotation.yaw)
            ego_forward = np.array([math.cos(yaw), math.sin(yaw)], dtype=np.float32)
            speed_ms = vec3_length(self.vehicle.get_velocity())
        prev_s = float(self.prev_route_s if self.prev_route_s is not None else 0.0)

        best_score = 1e18
        best_d = 1e9
        cand_s_best = prev_s
        best_idx = start_idx
        best_proj = np.zeros(2, dtype=np.float32)
        best_dir = np.array([1.0, 0.0], dtype=np.float32)
        fallback_d = 1e9
        fallback = None

        for i in range(start_idx, end_idx + 1):
            a = np.array(self.route_xy[i], dtype=np.float32)
            b = np.array(self.route_xy[i + 1], dtype=np.float32)
            proj, d, local_s = point_segment_projection(p, a, b)
            seg = b - a
            seg_norm = float(np.linalg.norm(seg))
            if seg_norm < 1e-6:
                continue
            seg_dir = seg / seg_norm
            cand_s = float(self.route_cumdist[i] + local_s)
            # Nearest segment regardless of heading, used if the alignment gate
            # rejects everything (e.g. mid-recovery while pointing off-route).
            if d < fallback_d:
                fallback_d = d
                fallback = (d, cand_s, i, proj, seg_dir)
            align = float(np.dot(ego_forward, seg_dir))
            if align < -0.35 and d > 0.75:
                continue
            # Prefer close, forward-aligned segments. A mild penalty on large
            # forward offsets discourages latching onto a later pass of the same
            # street, without ever rejecting a candidate outright.
            score = float(d + 1.5 * max(0.0, 1.0 - align) + 0.02 * max(cand_s - prev_s, 0.0))
            if score < best_score:
                best_score = score
                best_d = d
                cand_s_best = cand_s
                best_idx = i
                best_proj = proj
                best_dir = seg_dir

        if best_score >= 1e17:
            if fallback is None:
                return prev_s, 0.0, self.route_progress_idx, p, ego_forward
            best_d, cand_s_best, best_idx, best_proj, best_dir = fallback

        # Rate limit the arc coordinate only. Forward budget is twice the
        # distance covered in one step plus slack; a small backward allowance
        # absorbs projection noise without letting progress unwind.
        fwd_budget = max(0.35, 2.0 * float(speed_ms) * float(self.cfg.dt) + 0.30)
        back_budget = 0.50
        best_s = float(np.clip(cand_s_best, prev_s - back_budget, prev_s + fwd_budget))
        best_s = float(np.clip(best_s, 0.0, max(self.route_total_len_m, 1e-6)))

        # Anchor the search window to the accepted arc position, not to the raw
        # geometric index, so a distant false match cannot drag the window along.
        idx_for_s = int(np.clip(np.searchsorted(self.route_cumdist, best_s, side="right") - 1,
                                0, len(self.route_xy) - 2))
        self.route_progress_idx = max(self.route_progress_idx, idx_for_s)
        return float(best_s), float(best_d), int(best_idx), best_proj, best_dir

    def _route_projection(self, loc: carla.Location) -> Tuple[float, float]:
        s_arc, dL, _, _, _ = self._route_projection_monotonic(loc)
        return float(s_arc), float(dL)

    def _route_reference_state(self, loc: carla.Location, lookahead_m: float) -> Dict[str, object]:
        s_arc, dL, _, _, _ = self._route_projection_monotonic(loc)
        remaining_s = max(0.0, self.route_total_len_m - s_arc)

        if len(self.route_wps) == 0:
            return {
                "s_arc": 0.0,
                "dL": 0.0,
                "remaining_s": 0.0,
                "ref_idx": 0,
                "ref_wp": None,
                "signed_lane_err": 0.0,
                "heading_err": 0.0,
                "lane_width": 3.5,
                "wrong_lane": False,
                "opposite_lane": False,
            }

        ref_s = min(s_arc + max(2.0, 0.5 * lookahead_m), self.route_total_len_m)
        ref_idx = int(np.searchsorted(np.asarray(self.route_cumdist, dtype=np.float32), ref_s, side="left"))
        ref_idx = int(np.clip(ref_idx, 0, len(self.route_wps) - 1))
        ref_wp = self.route_wps[ref_idx]
        ref_tf = ref_wp.transform
        ref_loc = ref_tf.location

        route_yaw = math.radians(ref_tf.rotation.yaw)
        ego_yaw = math.radians(self.vehicle.get_transform().rotation.yaw)
        route_right = np.array([-math.sin(route_yaw), math.cos(route_yaw)], dtype=np.float32)
        rel = np.array([loc.x - ref_loc.x, loc.y - ref_loc.y], dtype=np.float32)
        signed_lane_err = float(np.dot(rel, route_right))
        heading_err = wrap_pi(route_yaw - ego_yaw)
        lane_width = max(float(getattr(ref_wp, "lane_width", 3.5)), 3.5)

        ego_wp = self.map.get_waypoint(loc, project_to_road=True, lane_type=carla.LaneType.Driving)
        wrong_lane = False
        opposite_lane = False
        if ego_wp is not None and not ref_wp.is_junction:
            same_road = ego_wp.road_id == ref_wp.road_id and ego_wp.section_id == ref_wp.section_id
            same_lane = same_road and (ego_wp.lane_id == ref_wp.lane_id)
            if same_road and ego_wp.lane_id != 0 and ref_wp.lane_id != 0:
                opposite_lane = np.sign(ego_wp.lane_id) != np.sign(ref_wp.lane_id)
            wrong_lane = opposite_lane or ((not same_lane) and (abs(signed_lane_err) > 0.55 * lane_width))

        return {
            "s_arc": float(s_arc),
            "dL": float(dL),
            "remaining_s": float(remaining_s),
            "ref_idx": int(ref_idx),
            "ref_wp": ref_wp,
            "signed_lane_err": float(signed_lane_err),
            "heading_err": float(heading_err),
            "lane_width": float(lane_width),
            "wrong_lane": bool(wrong_lane),
            "opposite_lane": bool(opposite_lane),
        }

    def _waypoint_visit_key(self, wp: carla.Waypoint) -> Tuple[int, int, int, int]:
        return (
            int(getattr(wp, "road_id", 0)),
            int(getattr(wp, "section_id", 0)),
            int(getattr(wp, "lane_id", 0)),
            int(round(float(getattr(wp, "s", 0.0)) / max(self.cfg.route_point_spacing_m, 1.0))),
        )

    def _select_route_successor(
        self,
        cur: carla.Waypoint,
        candidates: Sequence[carla.Waypoint],
        visit_count: Dict[Tuple[int, int, int, int], int],
        recent: Sequence[carla.Waypoint],
    ) -> Optional[carla.Waypoint]:
        """Choose a continuous, mostly forward, non-repeating lane successor.

        CARLA returns multiple successors at junctions. Selecting only the same
        road/lane can create short loops and ambiguous route projection. This
        score prefers forward continuity while strongly discouraging revisits.
        """
        if not candidates:
            return None
        cur_yaw = math.radians(cur.transform.rotation.yaw)
        cur_dir = np.array([math.cos(cur_yaw), math.sin(cur_yaw)], dtype=np.float32)
        recent_locs = [wp.transform.location for wp in recent[-30:]]
        scored: List[Tuple[float, carla.Waypoint]] = []
        for cand in candidates:
            loc0 = cur.transform.location
            loc1 = cand.transform.location
            delta = np.array([loc1.x - loc0.x, loc1.y - loc0.y], dtype=np.float32)
            d = float(np.linalg.norm(delta))
            if d < 0.20:
                continue
            tangent = delta / max(d, 1e-6)
            forward = float(np.dot(cur_dir, tangent))
            key = self._waypoint_visit_key(cand)
            revisits = int(visit_count.get(key, 0))
            near_recent = 0
            for rloc in recent_locs[:-3]:
                if distance2d(loc1, rloc) < 3.0:
                    near_recent += 1
            score = 8.0 * forward
            score += 1.5 if cand.lane_id == cur.lane_id else 0.0
            score += 0.5 if cand.road_id == cur.road_id else 0.0
            score -= 12.0 * revisits
            score -= 4.0 * near_recent
            # Seeded jitter prevents every training episode from taking the
            # identical junction branch while preserving deterministic seeds.
            score += random.uniform(-0.15, 0.15)
            scored.append((score, cand))
        if not scored:
            return None
        scored.sort(key=lambda item: item[0], reverse=True)
        return scored[0][1]

    def _extend_waypoint_chain(
        self,
        start_wp: carla.Waypoint,
        target_length_m: float,
        ds: Optional[float] = None,
        prefix: Optional[Sequence[carla.Waypoint]] = None,
    ) -> List[carla.Waypoint]:
        ds = float(self.cfg.route_point_spacing_m if ds is None else ds)
        chain = list(prefix or [])
        if not chain:
            chain = [start_wp]
        cur = chain[-1]
        visit_count: Dict[Tuple[int, int, int, int], int] = {}
        for wp in chain:
            key = self._waypoint_visit_key(wp)
            visit_count[key] = visit_count.get(key, 0) + 1
        length = self._route_length_from_wps(chain)
        max_points = int(max(100, math.ceil((target_length_m + 80.0) / max(ds, 0.5))))
        while length < target_length_m and len(chain) < max_points:
            try:
                candidates = list(cur.next(ds))
            except BaseException:
                candidates = []
            nxt = self._select_route_successor(cur, candidates, visit_count, chain)
            if nxt is None:
                break
            seg = distance2d(cur.transform.location, nxt.transform.location)
            if seg < 0.20:
                break
            chain.append(nxt)
            cur = nxt
            length += seg
            key = self._waypoint_visit_key(nxt)
            visit_count[key] = visit_count.get(key, 0) + 1
        return self._dedupe_waypoints(chain, min_sep=0.50)

    def _trim_route_to_target(
        self,
        wps: Sequence[carla.Waypoint],
        target_m: Optional[float] = None,
    ) -> List[carla.Waypoint]:
        """Trim a continuous waypoint route to the waypoint nearest target_m."""
        route = self._dedupe_waypoints(wps, min_sep=0.50)
        if len(route) < 2:
            return route
        target = float(self.cfg.route_target_length_m if target_m is None else target_m)
        cum = [0.0]
        for i in range(1, len(route)):
            cum.append(cum[-1] + distance2d(route[i - 1].transform.location, route[i].transform.location))
        idx = int(np.argmin(np.abs(np.asarray(cum, dtype=np.float32) - target)))
        idx = max(1, idx)
        return route[: idx + 1]

    def _install_route(self, wps: Sequence[carla.Waypoint], goal_index: Optional[int] = None) -> None:
        route = self._trim_route_to_target(wps)
        if len(route) < 2:
            raise RuntimeError("Route contains fewer than two waypoints.")
        route_len = self._route_length_from_wps(route)
        if not self._route_length_ok(route_len, strict=True):
            raise RuntimeError(
                f"Could not construct a {self.cfg.route_target_length_m:.0f} m route: "
                f"obtained {route_len:.1f} m (allowed ±{self.cfg.route_target_tolerance_m:.1f} m)."
            )
        self.route_wps = list(route)
        self.route_xy = [(float(wp.transform.location.x), float(wp.transform.location.y)) for wp in route]
        self._compute_route_cumdist()
        goal_wp = route[-1]
        goal_loc = goal_wp.transform.location
        self.goal_transform = carla.Transform(
            carla.Location(goal_loc.x, goal_loc.y, goal_loc.z + 0.5),
            goal_wp.transform.rotation,
        )
        self.current_goal_index = goal_index
        self.goal_spawn_transform = self.spawn_points[goal_index] if goal_index is not None else None

    def _build_route_polyline_local(self, start_wp: carla.Waypoint, ds: float = 2.0) -> carla.Waypoint:
        target_with_margin = self.cfg.route_target_length_m + max(2.0 * ds, self.cfg.route_target_tolerance_m)
        route = self._extend_waypoint_chain(start_wp, target_with_margin, ds=ds)
        self.route_wps = self._trim_route_to_target(route)
        self.route_xy = [(float(wp.transform.location.x), float(wp.transform.location.y)) for wp in self.route_wps]
        return self.route_wps[-1] if self.route_wps else start_wp

    def _lane_follow_extension(self, start_wp: carla.Waypoint, extra_m: float, ds: float = 2.0) -> List[carla.Waypoint]:
        if extra_m <= 0.0:
            return []
        route = self._extend_waypoint_chain(start_wp, extra_m, ds=ds)
        return route[1:] if len(route) > 1 else []

    def _project_drive_wp(self, loc: carla.Location) -> Optional[carla.Waypoint]:
        try:
            return self.map.get_waypoint(loc, project_to_road=True, lane_type=carla.LaneType.Driving)
        except BaseException:
            return None

    def _snap_spawn_to_driving_lane(self, tf: carla.Transform) -> carla.Transform:
        wp = self._project_drive_wp(tf.location)
        if wp is None:
            return carla.Transform(
                carla.Location(tf.location.x, tf.location.y, tf.location.z),
                carla.Rotation(tf.rotation.pitch, tf.rotation.yaw, tf.rotation.roll),
            )
        lane_tf = wp.transform
        z = max(float(tf.location.z), float(lane_tf.location.z)) + max(0.15, float(self.cfg.spawn_retry_lift_m))
        return carla.Transform(
            carla.Location(float(lane_tf.location.x), float(lane_tf.location.y), z),
            carla.Rotation(float(lane_tf.rotation.pitch), float(lane_tf.rotation.yaw), float(lane_tf.rotation.roll)),
        )

    def _build_route_from_goal_index(self, start_tf: carla.Transform, start_wp: carla.Waypoint, goal_index: int) -> bool:
        grp = self._get_global_planner()
        if grp is None or goal_index < 0 or goal_index >= len(self.spawn_points):
            return False
        start_wp_proj = self._project_drive_wp(start_tf.location) or start_wp
        goal_wp_proj = self._project_drive_wp(self.spawn_points[goal_index].location)
        if goal_wp_proj is None:
            return False
        try:
            trace = grp.trace_route(
                start_wp_proj.transform.location,
                goal_wp_proj.transform.location,
            )
        except BaseException:
            return False
        route = self._dedupe_waypoints([wp for wp, _ in trace if wp is not None], min_sep=0.50)
        route_len = self._route_length_from_wps(route)
        # A fixed destination remains a fixed destination: do not truncate or
        # extend its route. It is accepted only when already close to 500 m.
        if len(route) < 2 or not self._route_length_ok(route_len, strict=True):
            return False
        self._install_route(route, goal_index=goal_index)
        return True

    def _find_best_auto_route_from_spawn(
        self,
        start_tf: carla.Transform,
        start_wp: carla.Waypoint,
    ) -> Tuple[List[carla.Waypoint], Optional[int], float]:
        start_loc = start_wp.transform.location
        start_yaw = math.radians(start_tf.rotation.yaw)
        start_dir = np.array([math.cos(start_yaw), math.sin(start_yaw)], dtype=np.float32)
        grp = self._get_global_planner()
        if grp is None:
            return [], None, 0.0

        candidate_ids = list(range(len(self.spawn_points)))
        random.shuffle(candidate_ids)
        max_tries = min(len(candidate_ids), max(self.cfg.candidate_goal_max_tries, len(candidate_ids)))

        best_soft_route: List[carla.Waypoint] = []
        best_soft_goal_idx: Optional[int] = None
        best_soft_score = -1e18

        best_relaxed_route: List[carla.Waypoint] = []
        best_relaxed_goal_idx: Optional[int] = None
        best_relaxed_score = -1e18

        best_any_route: List[carla.Waypoint] = []
        best_any_goal_idx: Optional[int] = None
        best_any_score = -1e18

        # Why candidates get rejected — reported once per reset in debug mode
        # so a persistent lane-follow fallback is diagnosable instead of silent.
        rej = {"no_wp": 0, "euclid": 0, "trace_fail": 0, "trace_short": 0,
               "too_short": 0, "detour": 0, "align": 0, "kept": 0}

        for idx in candidate_ids[:max_tries]:
            if self.current_spawn_index is not None and idx == int(self.current_spawn_index):
                continue
            goal_sp = self.spawn_points[idx]
            goal_wp_proj = self._project_drive_wp(goal_sp.location)
            if goal_wp_proj is None:
                rej["no_wp"] += 1
                continue
            goal_loc_req = goal_wp_proj.transform.location
            euclid = distance2d(start_loc, goal_loc_req)
            # Euclidean prefilter only. Town10HD spans roughly 200 x 210 m, so
            # no pair of spawn points is 500 m apart in a straight line and the
            # planner's shortest paths top out near the map diagonal. The floor
            # is therefore a small fraction of the arc target; the route is
            # brought up to length by lane-follow extension in _build_route.
            if euclid < min(self.cfg.candidate_goal_min_dist_m, 0.12 * self.cfg.route_target_length_m):
                rej["euclid"] += 1
                continue
            try:
                trace = grp.trace_route(start_loc, goal_loc_req)
            except BaseException:
                rej["trace_fail"] += 1
                continue
            if len(trace) < 2:
                rej["trace_short"] += 1
                continue
            route = [wp for wp, _ in trace if wp is not None]
            route = self._dedupe_waypoints(route, min_sep=0.75)
            if len(route) < 2:
                rej["trace_short"] += 1
                continue
            route_len = self._route_length_from_wps(route)
            # Planner shortest paths in small towns are often shorter than the
            # 500 m target. Keep them as "any"-tier candidates; _build_route
            # extends the winner with lane-follow continuation to target length.
            if route_len < 0.55 * self.cfg.min_route_length_m:
                rej["too_short"] += 1
                continue

            detour_ratio = route_len / max(euclid, 1e-3)
            # Long fixed-length routes legitimately wind (arc >> Euclid). In a
            # compact town a 500 m route over a 100 m displacement is ratio 5,
            # so this cap must stay generous.
            if detour_ratio > 7.0:
                rej["detour"] += 1
                continue

            k = min(3, len(route) - 1)
            p0 = np.array([route[0].transform.location.x, route[0].transform.location.y], dtype=np.float32)
            pk = np.array([route[k].transform.location.x, route[k].transform.location.y], dtype=np.float32)
            tangent = pk - p0
            tangent_norm = float(np.linalg.norm(tangent))
            if tangent_norm < 1e-6:
                continue
            tangent = tangent / tangent_norm
            align = float(np.dot(start_dir, tangent))
            if align < -0.35:
                rej["align"] += 1
                continue
            rej["kept"] += 1

            # Planner routes are extended to the target length afterwards, so
            # score for shape (forward alignment, useful length, sane detour)
            # and only mildly penalise falling short of the target.
            len_error = max(0.0, self.cfg.route_target_length_m - route_len)
            score = (
                16.0 * align
                + 0.10 * route_len
                - 0.02 * len_error
                - 3.0 * max(detour_ratio - 3.0, 0.0)
            )
            any_score = 12.0 * align + 0.15 * route_len + 1.20 * euclid - 4.0 * max(detour_ratio - 2.8, 0.0)

            if self._route_length_ok(route_len, strict=False):
                if score > best_soft_score:
                    best_soft_score = score
                    best_soft_route = route
                    best_soft_goal_idx = idx
            elif route_len >= self.cfg.min_route_length_m and route_len <= max(
                self.cfg.route_target_length_m + max(180.0, 0.35 * self.cfg.route_target_length_m),
                self.cfg.route_soft_max_length_m + 120.0,
            ):
                relaxed_score = score - 0.010 * max(0.0, len_error - self.cfg.route_target_tolerance_m)
                if relaxed_score > best_relaxed_score:
                    best_relaxed_score = relaxed_score
                    best_relaxed_route = route
                    best_relaxed_goal_idx = idx

            if any_score > best_any_score:
                best_any_score = any_score
                best_any_route = route
                best_any_goal_idx = idx

        if self.cfg.debug_mode and rej["kept"] == 0:
            print(
                f"[INFO] auto-route: no candidate kept from spawn {self.current_spawn_index} "
                f"(tried={min(max_tries, len(candidate_ids))} no_wp={rej['no_wp']} euclid={rej['euclid']} "
                f"trace_fail={rej['trace_fail']} trace_short={rej['trace_short']} "
                f"too_short={rej['too_short']} detour={rej['detour']} align={rej['align']})"
            )

        if len(best_soft_route) >= 2:
            return best_soft_route, best_soft_goal_idx, self._route_length_from_wps(best_soft_route)
        if len(best_relaxed_route) >= 2:
            return best_relaxed_route, best_relaxed_goal_idx, self._route_length_from_wps(best_relaxed_route)
        if len(best_any_route) >= 2:
            return best_any_route, best_any_goal_idx, self._route_length_from_wps(best_any_route)
        return [], None, 0.0

    def _build_route(self, start_tf: carla.Transform, start_wp: carla.Waypoint, goal_index: Optional[int] = None) -> None:
        self.route_xy = []
        self.route_wps = []
        self.route_cumdist = []
        self.goal_transform = None
        self.goal_spawn_transform = None
        self.current_goal_index = None
        self.route_progress_idx = 0
        self.route_total_len_m = 0.0

        chosen_goal = goal_index
        if chosen_goal is None and self.cfg.use_fixed_destination:
            chosen_goal = self.fixed_goal_index
        if chosen_goal is not None and int(chosen_goal) < 0:
            chosen_goal = None

        if chosen_goal is not None:
            if self._build_route_from_goal_index(start_tf, start_wp, int(chosen_goal)):
                return
            if self.cfg.strict_goal_route and not self.cfg.allow_fallback_route:
                raise RuntimeError(
                    f"Goal index {chosen_goal} does not define a route within "
                    f"{self.cfg.route_target_length_m:.0f}±{self.cfg.route_target_tolerance_m:.0f} m. "
                    "Use goal index -1 for automatic 500 m route generation."
                )

        # Prefer a planner route with useful global structure, then continue
        # along connected lane-center waypoints until the route exceeds 500 m.
        best_route, _, best_len = self._find_best_auto_route_from_spawn(start_tf, start_wp)
        target_with_margin = self.cfg.route_target_length_m + max(
            2.0 * self.cfg.route_point_spacing_m,
            self.cfg.route_target_tolerance_m,
        )
        if len(best_route) >= 2:
            route = list(best_route)
            if best_len < target_with_margin:
                route = self._extend_waypoint_chain(
                    route[-1], target_with_margin, ds=self.cfg.route_point_spacing_m,
                    prefix=route,
                )
            try:
                self._install_route(route, goal_index=None)
                return
            except RuntimeError as exc:
                if self.cfg.debug_mode:
                    print(f"[WARN] planner-based 500 m route rejected: {exc}")

        # Guaranteed fallback: construct a continuous lane-follow route from
        # the actual ego lane and trim it to the waypoint nearest 500 m.
        local_route = self._extend_waypoint_chain(
            start_wp, target_with_margin, ds=self.cfg.route_point_spacing_m
        )
        self._install_route(local_route, goal_index=None)
        if self.cfg.debug_mode:
            print(
                f"[INFO] installed automatic route: {self.route_total_len_m:.1f} m "
                f"from spawn {self.current_spawn_index} in {self.town_name}"
            )

    def _route_completion_pct(self, s_arc: Optional[float] = None) -> float:
        if self.route_total_len_m <= 1e-6:
            return 0.0
        if s_arc is None and self.vehicle is not None:
            s_arc, _ = self._route_projection(self.vehicle.get_location())
        s_arc = 0.0 if s_arc is None else float(s_arc)
        return float(np.clip(100.0 * s_arc / max(self.route_total_len_m, 1e-6), 0.0, 100.0))

    def _curvature_ahead(self, wp: Optional[carla.Waypoint], ds: float = 4.0) -> float:
        if wp is None:
            return 0.0
        nxt = wp.next(ds)
        if not nxt:
            return 0.0
        nxt2 = nxt[0].next(ds)
        if not nxt2:
            return 0.0
        p0 = np.array([wp.transform.location.x, wp.transform.location.y], dtype=np.float32)
        p1 = np.array([nxt[0].transform.location.x, nxt[0].transform.location.y], dtype=np.float32)
        p2 = np.array([nxt2[0].transform.location.x, nxt2[0].transform.location.y], dtype=np.float32)
        a = np.linalg.norm(p1 - p0)
        b = np.linalg.norm(p2 - p1)
        c = np.linalg.norm(p2 - p0)
        if float(a * b * c) < 1e-6:
            return 0.0
        area2 = abs(np.cross(p1 - p0, p2 - p0))
        return float(2.0 * area2 / (a * b * c))

    def _lane_heading_offset(self, wp_onlane: Optional[carla.Waypoint]) -> float:
        if wp_onlane is None or self.vehicle is None:
            return 0.0
        yaw_lane = math.radians(wp_onlane.transform.rotation.yaw)
        yaw_veh = math.radians(self.vehicle.get_transform().rotation.yaw)
        return float(np.clip(wrap_pi(yaw_veh - yaw_lane) / (math.pi / 2.0), -1.0, 1.0))

    @staticmethod
    def _tl_state_name(state: object) -> str:
        try:
            return str(state).split(".")[-1]
        except BaseException:
            return "Unknown"

    def _front_bumper_position(self):
        tf = self.vehicle.get_transform()
        box = self.vehicle.bounding_box
        point = tf.transform(carla.Location(x=float(box.location.x + box.extent.x),
                                           y=float(box.location.y), z=float(box.location.z)))
        return (float(point.x), float(point.y), float(point.z))

    def _reset_stop_line_tracker(self):
        self._stop_line_tracker = StopLineCrossingTracker(self._front_bumper_position())
        self._stop_lines = []
        self.red_light_events = []
        self._step_red_light_events = []
        for light in self.world.get_actors().filter("traffic.traffic_light*"):
            for wp in light.get_stop_waypoints() or []:
                loc = wp.transform.location
                yaw = math.radians(wp.transform.rotation.yaw)
                self._stop_lines.append((light, {
                    'key': [int(light.id), int(wp.road_id), int(wp.lane_id), round(float(wp.s), 2)],
                    'light_id': int(light.id), 'centre': [float(loc.x), float(loc.y), float(loc.z)],
                    'direction': [math.cos(yaw), math.sin(yaw)], 'width': float(wp.lane_width)}))

    def _stop_line_snapshot(self):
        position = np.asarray(self._front_bumper_position())
        lines, states = [], {}
        for light, geometry in self._stop_lines:
            if np.linalg.norm(np.asarray(geometry['centre'])[:2] - position[:2]) > 15.0:
                continue
            lid = geometry['light_id']
            if lid not in states:
                states[lid] = self._tl_state_name(light.get_state())
            lines.append(dict(geometry, state=states[lid]))
        return lines

    def _stop_line_longitudinal(self, light: carla.Actor, loc: carla.Location,
                                lane_fwd: np.ndarray) -> float:
        """Signed along-lane distance from the ego to a light's stop line.

        Positive means the stop line is ahead, negative means the ego has already
        crossed it. Returns 1e9 when no stop waypoint of this light belongs to
        the ego's travel direction.
        """
        best = 1e9
        try:
            stop_wps = light.get_stop_waypoints() or []
        except BaseException:
            return 1e9
        for swp in stop_wps:
            try:
                vec = swp.transform.location - loc
                vec2 = np.array([vec.x, vec.y], dtype=np.float32)
                forward = float(np.dot(vec2, lane_fwd))
                lateral = abs(float(np.linalg.norm(vec2 - forward * lane_fwd)))
                # Same travel direction as the ego lane, and laterally on our lane.
                yaw_s = math.radians(swp.transform.rotation.yaw)
                s_fwd = np.array([math.cos(yaw_s), math.sin(yaw_s)], dtype=np.float32)
                if float(np.dot(s_fwd, lane_fwd)) < 0.60:
                    continue
                if lateral > 3.0:
                    continue
                if forward < -float(self.cfg.tl_cross_track_m):
                    continue
                if abs(forward) < abs(best):
                    best = forward
            except BaseException:
                continue
        return float(best)

    def _get_active_traffic_light_info(self) -> Tuple[str, float]:
        """Return (state, signed along-lane distance to the stop line).

        The distance is negative for a few metres after the ego crosses the line
        so that running a red light remains observable to the reward. 1e9 means
        no relevant light.
        """
        if self.vehicle is None:
            return "None", 1e9
        loc = self.vehicle.get_location()
        wp_onlane = self.map.get_waypoint(loc, project_to_road=True, lane_type=carla.LaneType.Driving)
        if wp_onlane is None:
            return "None", 1e9
        yaw_lane = math.radians(wp_onlane.transform.rotation.yaw)
        lane_fwd = np.array([math.cos(yaw_lane), math.sin(yaw_lane)], dtype=np.float32)

        candidates: List[carla.Actor] = []
        ego_light = None
        try:
            ego_light = self.vehicle.get_traffic_light()
        except BaseException:
            ego_light = None
        if ego_light is not None:
            candidates.append(ego_light)
        # Look further ahead than the trigger volume so braking can begin in time.
        try:
            actors = self.world.get_actors()
            iterable = actors.filter("traffic.traffic_light*") if hasattr(actors, "filter") else                 [a for a in actors if "traffic_light" in getattr(a, "type_id", "")]
            for tl in iterable:
                if tl is None or (ego_light is not None and int(tl.id) == int(ego_light.id)):
                    continue
                if distance2d(tl.get_location(), loc) > float(self.cfg.tl_detect_dist) + 25.0:
                    continue
                candidates.append(tl)
        except BaseException:
            pass

        best_state = "None"
        best_dist = 1e9
        for tl in candidates:
            d = self._stop_line_longitudinal(tl, loc, lane_fwd)
            if d > 1e8 or d > float(self.cfg.tl_detect_dist):
                continue
            if abs(d) < abs(best_dist):
                try:
                    best_state = self._tl_state_name(tl.get_state())
                except BaseException:
                    best_state = "Unknown"
                best_dist = d
        if best_dist > 1e8:
            return "None", 1e9
        return best_state, float(best_dist)

    def _red_brake_distance(self) -> float:
        """Distance at which a red light must already be influencing control."""
        v = 0.0
        if self.vehicle is not None:
            try:
                v = float(vec3_length(self.vehicle.get_velocity()))
            except BaseException:
                v = 0.0
        a = max(float(self.cfg.tl_brake_decel), 0.5)
        return float(v * v / (2.0 * a) + v * float(self.cfg.tl_reaction_s) + float(self.cfg.tl_stop_margin))

    def _world_actors(self) -> Iterable[carla.Actor]:
        try:
            return self.world.get_actors()
        except BaseException:
            return []

    def _nearby_walkers(self, ego_loc: carla.Location) -> List[carla.Actor]:
        walkers: List[carla.Actor] = []
        try:
            actors = self._world_actors()
            if hasattr(actors, "filter"):
                iterable = actors.filter("walker.pedestrian.*")
            else:
                iterable = [a for a in actors if "walker.pedestrian" in getattr(a, "type_id", "")]
            for actor in iterable:
                if distance2d(actor.get_location(), ego_loc) <= self.cfg.entity_max_dist:
                    walkers.append(actor)
        except Exception:
            pass
        return walkers

    def _get_front_vehicle_info(self, max_dist: Optional[float] = None) -> Tuple[Optional[float], float, Optional[carla.Vehicle]]:
        if self.vehicle is None:
            return None, 0.0, None

        if max_dist is None:
            max_dist = self.cfg.front_vehicle_max_dist

        ego_loc = self.vehicle.get_location()
        ego_speed_kmh = 3.6 * vec3_length(self.vehicle.get_velocity())
        lookahead_m = max(self.cfg.lookahead_base_m + 0.25 * max(ego_speed_kmh, 0.0), 5.0)

        rs = self._route_reference_state(ego_loc, lookahead_m=lookahead_m)
        ref_wp = rs.get("ref_wp", None)
        if ref_wp is None:
            return None, 0.0, None

        ref_loc = ref_wp.transform.location
        route_yaw = math.radians(ref_wp.transform.rotation.yaw)
        route_fwd = np.array([math.cos(route_yaw), math.sin(route_yaw)], dtype=np.float32)
        route_right = np.array([-math.sin(route_yaw), math.cos(route_yaw)], dtype=np.float32)
        lane_width = max(float(getattr(ref_wp, "lane_width", 3.5)), 3.5)

        best_metric = 1e9
        best_dist: Optional[float] = None
        best_speed_kmh = 0.0
        best_actor: Optional[carla.Vehicle] = None

        for npc in self.npcs:
            if npc is None or not npc.is_alive:
                continue
            try:
                nloc = npc.get_location()
                rel_ego = np.array([nloc.x - ego_loc.x, nloc.y - ego_loc.y], dtype=np.float32)
                rel_ref = np.array([nloc.x - ref_loc.x, nloc.y - ref_loc.y], dtype=np.float32)
                forward_dist = float(np.dot(rel_ego, route_fwd))
                lateral_dist = abs(float(np.dot(rel_ref, route_right)))

                if forward_dist <= 0.5 or forward_dist > float(max_dist):
                    continue
                if lateral_dist > max(1.40, 0.45 * lane_width):
                    continue

                nwp = self.map.get_waypoint(nloc, project_to_road=True, lane_type=carla.LaneType.Driving)
                if nwp is None:
                    continue
                npc_yaw = math.radians(nwp.transform.rotation.yaw)
                yaw_err = abs(wrap_pi(npc_yaw - route_yaw))
                same_road_family = (nwp.road_id == ref_wp.road_id) or bool(ref_wp.is_junction) or bool(nwp.is_junction)
                same_direction = yaw_err < 0.60
                if not same_road_family or not same_direction:
                    continue

                metric = forward_dist + 1.75 * lateral_dist
                if metric < best_metric:
                    best_metric = metric
                    best_dist = forward_dist
                    best_actor = npc
                    best_speed_kmh = 3.6 * vec3_length(npc.get_velocity())
            except Exception:
                continue

        # Pedestrians ahead in the ego corridor block progress exactly like a
        # lead vehicle. They are matched on corridor geometry only (no lane /
        # heading test) because walkers cross rather than follow the lane.
        for walker in self._nearby_walkers(ego_loc):
            if walker is None:
                continue
            try:
                wloc = walker.get_location()
                rel_ego = np.array([wloc.x - ego_loc.x, wloc.y - ego_loc.y], dtype=np.float32)
                rel_ref = np.array([wloc.x - ref_loc.x, wloc.y - ref_loc.y], dtype=np.float32)
                forward_dist = float(np.dot(rel_ego, route_fwd))
                lateral_dist = abs(float(np.dot(rel_ref, route_right)))
                if forward_dist <= 0.5 or forward_dist > float(max_dist):
                    continue
                if lateral_dist > max(1.20, 0.40 * lane_width):
                    continue
                metric = forward_dist + 1.75 * lateral_dist
                if metric < best_metric:
                    best_metric = metric
                    best_dist = forward_dist
                    best_actor = walker
                    best_speed_kmh = 3.6 * vec3_length(walker.get_velocity())
            except Exception:
                continue

        return best_dist, float(best_speed_kmh), best_actor

    def _get_min_vehicle_ttc(self, max_dist: float = 25.0) -> Tuple[float, float]:
        if self.vehicle is None:
            return 999.0, 1e9
        ego_loc = self.vehicle.get_location()
        ego_vel = self.vehicle.get_velocity()
        ego_v = np.array([ego_vel.x, ego_vel.y], dtype=np.float32)
        min_ttc = 999.0
        min_dist = 1e9
        for npc in self.npcs:
            if npc is None or not npc.is_alive:
                continue
            try:
                nloc = npc.get_location()
                dist = distance2d(ego_loc, nloc)
                if dist < 0.5 or dist > max_dist:
                    continue
                min_dist = min(min_dist, dist)
                nvel = npc.get_velocity()
                rel_p = np.array([nloc.x - ego_loc.x, nloc.y - ego_loc.y], dtype=np.float32)
                rel_v = np.array([nvel.x, nvel.y], dtype=np.float32) - ego_v
                closing = max(0.0, -float(np.dot(rel_p, rel_v)) / max(float(np.linalg.norm(rel_p)), 1e-6))
                if closing > 1e-3:
                    min_ttc = min(min_ttc, float(dist / closing))
            except Exception:
                continue
        return float(min_ttc), float(min_dist)

    def _observation_dropout(self, dist: float, fog_norm: float, kind: str = "vehicle") -> bool:
        dist_n = float(np.clip(dist / max(self.cfg.entity_max_dist, 1e-3), 0.0, 1.0))
        miss = self.cfg.obs_miss_base + 0.22 * dist_n + 0.22 * fog_norm
        if kind == "walker":
            miss += 0.08
        if kind == "traffic_light":
            miss += 0.05
        miss = float(np.clip(miss, 0.02, 0.65))
        return random.random() < miss

    def _noisy_relative_measurement(
        self,
        delta_world: np.ndarray,
        rel_v_world: np.ndarray,
        dist: float,
        fog_norm: float,
        ego_forward: np.ndarray,
        ego_left: np.ndarray,
    ) -> Tuple[float, float, float, float, float]:
        dist_n = float(np.clip(dist / max(self.cfg.entity_max_dist, 1e-3), 0.0, 1.0))
        pos_std = self.cfg.obs_pos_noise_m * (0.45 + 0.80 * dist_n + 0.90 * fog_norm)
        vel_std = self.cfg.obs_vel_noise_ms * (0.35 + 0.75 * dist_n + 0.80 * fog_norm)
        noisy_delta = delta_world + np.random.normal(0.0, pos_std, size=2).astype(np.float32)
        noisy_rel_v = rel_v_world + np.random.normal(0.0, vel_std, size=2).astype(np.float32)
        rel_x = float(np.dot(noisy_delta, ego_forward) / self.cfg.entity_max_dist)
        rel_y = float(np.dot(noisy_delta, ego_left) / self.cfg.entity_max_dist)
        rel_vx = float(np.dot(noisy_rel_v, ego_forward) / 15.0)
        rel_vy = float(np.dot(noisy_rel_v, ego_left) / 10.0)
        sigma2 = float(np.clip(0.05 + 0.22 * dist_n + 0.28 * fog_norm + 0.08 * pos_std + 0.04 * vel_std, 0.05, 0.95))
        return rel_x, rel_y, rel_vx, rel_vy, sigma2

    def _collect_entity_features(self) -> Tuple[np.ndarray, np.ndarray, float]:
        assert self.vehicle is not None
        loc = self.vehicle.get_location()
        vel = self.vehicle.get_velocity()
        wp_onlane = self.map.get_waypoint(loc, project_to_road=True, lane_type=carla.LaneType.Driving)
        curv = self._curvature_ahead(wp_onlane)
        lane_curv = float(np.tanh(10.0 * curv))
        lane_head = float(self._lane_heading_offset(wp_onlane))
        ego_tf = self.vehicle.get_transform()
        ego_yaw = math.radians(ego_tf.rotation.yaw)
        ego_forward = np.array([math.cos(ego_yaw), math.sin(ego_yaw)], dtype=np.float32)
        ego_left = np.array([-math.sin(ego_yaw), math.cos(ego_yaw)], dtype=np.float32)
        ego_vel = np.array([vel.x, vel.y], dtype=np.float32)
        fog_norm = get_fog_norm(self.world)

        feats: List[Tuple[float, np.ndarray]] = []
        for npc in self.npcs:
            if npc is None or not npc.is_alive:
                continue
            nloc = npc.get_location()
            delta = np.array([nloc.x - loc.x, nloc.y - loc.y], dtype=np.float32)
            dist = float(np.linalg.norm(delta))
            if dist < 1.5 or dist > self.cfg.entity_max_dist:
                continue
            if self._observation_dropout(dist, fog_norm, kind="vehicle"):
                continue
            nvel = npc.get_velocity()
            rel_v = np.array([nvel.x, nvel.y], dtype=np.float32) - ego_vel
            rel_x, rel_y, rel_vx, rel_vy, sigma2 = self._noisy_relative_measurement(delta, rel_v, dist, fog_norm, ego_forward, ego_left)
            c = np.array([1.0, 0.0, 0.0], dtype=np.float32)
            kappa = np.array([lane_curv, lane_head], dtype=np.float32)
            e = np.concatenate([np.array([rel_x, rel_y, rel_vx, rel_vy], dtype=np.float32), c, kappa, np.array([sigma2], dtype=np.float32)])
            feats.append((dist, e))

        for walker in self._nearby_walkers(loc):
            try:
                wloc = walker.get_location()
                delta = np.array([wloc.x - loc.x, wloc.y - loc.y], dtype=np.float32)
                dist = float(np.linalg.norm(delta))
                if dist < 1.5 or dist > self.cfg.entity_max_dist:
                    continue
                if self._observation_dropout(dist, fog_norm, kind="walker"):
                    continue
                wvel = walker.get_velocity()
                rel_v = np.array([wvel.x, wvel.y], dtype=np.float32) - ego_vel
                rel_x, rel_y, rel_vx, rel_vy, sigma2 = self._noisy_relative_measurement(delta, rel_v, dist, fog_norm, ego_forward, ego_left)
                sigma2 = float(np.clip(sigma2 + 0.06, 0.05, 0.98))
                c = np.array([0.0, 1.0, 0.0], dtype=np.float32)
                kappa = np.array([lane_curv, lane_head], dtype=np.float32)
                e = np.concatenate([np.array([rel_x, rel_y, rel_vx, rel_vy], dtype=np.float32), c, kappa, np.array([sigma2], dtype=np.float32)])
                feats.append((dist, e))
            except Exception:
                continue

        tl_state, tl_dist = self._get_active_traffic_light_info()
        # Include a light once it is within braking range, not just the fixed
        # near distance, so the relational state can see it early enough to act.
        tl_edge_range = max(float(self.cfg.tl_near_dist), self._red_brake_distance() + 6.0)
        if 0.0 <= tl_dist < tl_edge_range and (not self._observation_dropout(tl_dist, fog_norm, kind="traffic_light")):
            noisy_tl_dist = float(max(0.0, tl_dist + np.random.normal(0.0, self.cfg.tl_obs_noise_m * (0.35 + fog_norm))))
            rel_x = float(np.clip(noisy_tl_dist / self.cfg.entity_max_dist, 0.0, 1.0))
            sigma2 = float(np.clip(0.08 + 0.24 * (noisy_tl_dist / max(self.cfg.tl_near_dist, 1e-3)) + 0.25 * fog_norm, 0.05, 0.90))
            c = np.array([0.0, 0.0, 1.0], dtype=np.float32)
            kappa = np.array([lane_curv, lane_head], dtype=np.float32)
            e = np.concatenate([np.array([rel_x, 0.0, 0.0, 0.0], dtype=np.float32), c, kappa, np.array([sigma2], dtype=np.float32)])
            feats.append((noisy_tl_dist, e))

        # Top-K relational aggregation (Sec. III / Table I): keep the K = M
        # nearest entities within 60 m for the compact single-actor state.
        feats.sort(key=lambda x: x[0])
        feats = feats[: self.cfg.max_entities]

        edges = np.zeros((self.cfg.max_entities, self.cfg.edge_dim), dtype=np.float32)
        mask = np.zeros((self.cfg.max_entities,), dtype=np.float32)
        if len(feats) == 0:
            mask[0] = 1.0
            edges[0, -1] = 0.8
            nearest = self.cfg.entity_max_dist
        else:
            for i, (_, e) in enumerate(feats):
                edges[i, :] = e
                mask[i] = 1.0
            nearest = float(feats[0][0])
        return edges, mask, nearest

    def _route_observation_features(self) -> Dict[str, float]:
        assert self.vehicle is not None
        loc = self.vehicle.get_location()
        vel = self.vehicle.get_velocity()
        ego_speed_ms = vec3_length(vel)
        ego_speed_kmh = 3.6 * ego_speed_ms
        wp_onlane = self.map.get_waypoint(loc, project_to_road=True, lane_type=carla.LaneType.Driving)
        lane_curv = float(np.tanh(10.0 * self._curvature_ahead(wp_onlane)))
        lane_head = float(self._lane_heading_offset(wp_onlane))

        lookahead_m = max(self.cfg.lookahead_base_m + self.cfg.lookahead_speed_gain * max(ego_speed_kmh, 0.0), 4.0)
        rs = self._route_reference_state(loc, lookahead_m=lookahead_m)
        lane_width = max(float(rs.get('lane_width', 3.5)), 3.5)
        signed_lane_err = float(rs.get('signed_lane_err', 0.0))
        heading_err = float(rs.get('heading_err', 0.0))
        ref_wp = rs.get('ref_wp', None)

        ego_tf = self.vehicle.get_transform()
        ego_yaw = math.radians(ego_tf.rotation.yaw)
        ego_forward = np.array([math.cos(ego_yaw), math.sin(ego_yaw)], dtype=np.float32)
        ego_left = np.array([-math.sin(ego_yaw), math.cos(ego_yaw)], dtype=np.float32)
        ego_vel = np.array([vel.x, vel.y], dtype=np.float32)

        look_x = 0.0
        look_y = 0.0
        v_route = 0.0
        if ref_wp is not None:
            ref_loc = ref_wp.transform.location
            delta = np.array([ref_loc.x - loc.x, ref_loc.y - loc.y], dtype=np.float32)
            look_x = float(np.clip(np.dot(delta, ego_forward) / max(lookahead_m, 1.0), -2.0, 2.0))
            look_y = float(np.clip(np.dot(delta, ego_left) / max(1.5 * lane_width, 1.0), -2.0, 2.0))
            route_yaw = math.radians(ref_wp.transform.rotation.yaw)
            route_fwd = np.array([math.cos(route_yaw), math.sin(route_yaw)], dtype=np.float32)
            v_route = float(np.clip(np.dot(ego_vel, route_fwd) / 10.0, -1.5, 2.0))

        fog_norm = get_fog_norm(self.world)
        density_den = self.cfg.train_npc_max if str(self.cfg.mode_name).lower() in ("train", "adapt", "policy") else self.cfg.npc_max
        density_norm = float(np.clip(len(self.npcs) / max(density_den, 1), 0.0, 1.0))
        curv_norm = float(np.clip(abs(lane_curv), 0.0, 1.0))
        # μ_A ∈ [0,1] (Sec. III-B): membership of fragile conditions (rain/fog,
        # dense traffic, sharp curvature). High μ_A means a fragile scene.
        muA = float(np.clip(0.45 * density_norm + 0.35 * fog_norm + 0.20 * curv_norm, 0.0, 1.0))

        return {
            'lane_curv': float(lane_curv),
            'lane_head': float(lane_head),
            'route_cte': float(np.clip(signed_lane_err / max(lane_width, 1.0), -2.0, 2.0)),
            'route_heading': float(np.clip(heading_err / (math.pi / 2.0), -1.0, 1.0)),
            'lookahead_x': float(look_x),
            'lookahead_y': float(look_y),
            'v_route': float(v_route),
            'wrong_lane': float(1.0 if bool(rs.get('wrong_lane', False)) else 0.0),
            'opposite_lane': float(1.0 if bool(rs.get('opposite_lane', False)) else 0.0),
            'muA': float(muA),
            'remaining_s': float(rs.get('remaining_s', 0.0)),
        }

    def _get_obs(self) -> Dict[str, np.ndarray]:
        assert self.vehicle is not None
        loc = self.vehicle.get_location()
        vel = self.vehicle.get_velocity()
        speed = vec3_length(vel)
        v_ego = float(np.clip(speed / 10.0, 0.0, 2.0))

        s_arc, _ = self._route_projection(loc)
        remaining_s = max(0.0, self.route_total_len_m - s_arc)
        total_route_len = max(self.route_total_len_m, 1.0)
        d_goal = float(np.clip(remaining_s / total_route_len, 0.0, 1.5))

        route_obs = self._route_observation_features()
        scalars = np.array(
            [
                v_ego,
                float(self.prev_action[0]),
                float(self.prev_action[1]),
                float(self.prev_action[2]),
                d_goal,
                float(route_obs['lane_curv']),
                float(route_obs['lane_head']),
                float(route_obs['route_cte']),
                float(route_obs['route_heading']),
                float(route_obs['lookahead_x']),
                float(route_obs['lookahead_y']),
                float(route_obs['v_route']),
                float(route_obs['muA']),
            ],
            dtype=np.float32,
        )
        edges, mask, _ = self._collect_entity_features()
        return {"scalars": scalars, "edges": edges, "mask": mask}

    def _compute_route_guidance_action(self, tl_state: str, tl_dist: float, front_vehicle_dist: Optional[float], front_vehicle_speed_kmh: float) -> Tuple[np.ndarray, Dict[str, float]]:
        if self.vehicle is None or len(self.route_wps) < 2:
            return np.array([0.0, 0.0, 0.0], dtype=np.float32), {
                "desired_speed_kmh": 0.0,
                "route_idx": 0.0,
                "route_target_idx": 0.0,
                "route_cte": 0.0,
                "route_heading_err": 0.0,
                "route_remaining_s": 0.0,
                "route_dL": 0.0,
                "wrong_lane": 0.0,
                "opposite_lane": 0.0,
            }

        loc = self.vehicle.get_location()
        vel = self.vehicle.get_velocity()
        speed_kmh = 3.6 * vec3_length(vel)
        lookahead_m = max(self.cfg.lookahead_base_m + self.cfg.lookahead_speed_gain * max(speed_kmh, 0.0), 4.0)
        rs = self._route_reference_state(loc, lookahead_m=lookahead_m)
        dL = float(rs["dL"])
        remaining_s = float(rs["remaining_s"])
        ref_idx = int(rs["ref_idx"])
        ref_wp = rs["ref_wp"]
        signed_lane_err = float(rs["signed_lane_err"])
        heading_err = float(rs["heading_err"])
        lane_width = float(rs["lane_width"])
        wrong_lane = bool(rs["wrong_lane"])
        opposite_lane = bool(rs["opposite_lane"])

        curv = self._curvature_ahead(ref_wp)
        curv_n = abs(float(np.tanh(10.0 * curv)))
        desired_speed_kmh = self.cfg.target_speed_kmh - self.cfg.curve_speed_penalty_kmh * curv_n
        desired_speed_kmh = max(self.cfg.min_target_speed_kmh, desired_speed_kmh)
        desired_speed_kmh = min(desired_speed_kmh, self.cfg.hard_speed_cap_kmh)

        if remaining_s < 30.0:
            desired_speed_kmh = min(desired_speed_kmh, 12.0)
        if remaining_s < 15.0:
            desired_speed_kmh = min(desired_speed_kmh, 8.0)
        if remaining_s < 8.0:
            desired_speed_kmh = min(desired_speed_kmh, 4.0)

        abs_lane_err = abs(signed_lane_err)
        if abs_lane_err > 0.22 * lane_width:
            desired_speed_kmh = min(desired_speed_kmh, 13.0)
        if abs_lane_err > 0.35 * lane_width:
            desired_speed_kmh = min(desired_speed_kmh, 11.0)
        if abs_lane_err > 0.50 * lane_width:
            desired_speed_kmh = min(desired_speed_kmh, 9.0)
        if abs_lane_err > 0.70 * lane_width:
            desired_speed_kmh = min(desired_speed_kmh, 7.0)
        if abs_lane_err > 0.95 * lane_width:
            desired_speed_kmh = min(desired_speed_kmh, 5.5)

        if abs(heading_err) > 0.25:
            desired_speed_kmh = min(desired_speed_kmh, 11.0)
        if abs(heading_err) > 0.40:
            desired_speed_kmh = min(desired_speed_kmh, 8.5)
        if abs(heading_err) > 0.60:
            desired_speed_kmh = min(desired_speed_kmh, 6.5)

        if wrong_lane:
            desired_speed_kmh = min(desired_speed_kmh, 4.5)
        if opposite_lane:
            desired_speed_kmh = min(desired_speed_kmh, 4.0)

        if tl_state == "Red" and tl_dist >= 0.0:
            if tl_dist <= self.cfg.tl_stop_dist:
                desired_speed_kmh = 0.0
            elif tl_dist < self.cfg.tl_near_dist:
                ratio = (tl_dist - self.cfg.tl_stop_dist) / max(self.cfg.tl_near_dist - self.cfg.tl_stop_dist, 1e-3)
                desired_speed_kmh = min(desired_speed_kmh, max(0.0, ratio * self.cfg.min_target_speed_kmh))
        elif tl_state == "Yellow" and 0.0 <= tl_dist < self.cfg.tl_stop_dist:
            desired_speed_kmh = min(desired_speed_kmh, 4.0)

        if front_vehicle_dist is not None:
            if front_vehicle_dist < self.cfg.front_vehicle_block_dist:
                desired_speed_kmh = 0.0
            elif front_vehicle_dist < self.cfg.front_vehicle_soft_block_dist:
                ratio = (front_vehicle_dist - self.cfg.front_vehicle_block_dist) / max(self.cfg.front_vehicle_soft_block_dist - self.cfg.front_vehicle_block_dist, 1e-3)
                desired_speed_kmh = min(desired_speed_kmh, max(0.0, ratio * max(front_vehicle_speed_kmh, 5.0)))
            elif front_vehicle_dist < self.cfg.front_vehicle_max_dist:
                desired_speed_kmh = min(desired_speed_kmh, max(front_vehicle_speed_kmh + 3.0, 7.0))

        cte_term = math.atan2(1.45 * signed_lane_err, max(0.45 * lookahead_m, 1.0))
        steer = (self.cfg.route_steer_kp + 0.55) * heading_err - (self.cfg.route_cte_kp + 0.55) * cte_term

        center_push = 0.0
        if abs_lane_err > self.cfg.center_deadband_m:
            lane_push_mag = max(0.0, abs_lane_err - self.cfg.center_push_start_m)
            center_push = self.cfg.center_push_gain * math.tanh(lane_push_mag / 0.75) * float(np.sign(signed_lane_err))
            steer -= center_push
            if abs_lane_err > self.cfg.route_hard_dL_m_tight:
                hard_push = self.cfg.center_push_hard_gain * math.tanh((abs_lane_err - self.cfg.route_hard_dL_m_tight) / 0.60)
                steer -= hard_push * float(np.sign(signed_lane_err))

        if wrong_lane:
            steer *= 1.20
        if opposite_lane:
            steer *= 1.35
        steer = float(np.clip(steer, -1.0, 1.0))
        if speed_kmh > 10.0:
            steer = float(np.clip(steer, -self.cfg.max_safe_steer_at_speed, self.cfg.max_safe_steer_at_speed))

        speed_err = desired_speed_kmh - speed_kmh
        if desired_speed_kmh <= 0.1:
            throttle = 0.0
            brake = float(np.clip(0.35 + (max(0.0, self.cfg.tl_stop_dist - tl_dist) / max(self.cfg.tl_stop_dist, 1e-3)), 0.35, 1.0))
        elif speed_err >= 0.0:
            throttle = float(np.clip(0.12 + self.cfg.route_speed_kp * speed_err, 0.0, 0.75))
            brake = 0.0
        elif speed_err > -self.cfg.coast_speed_band_kmh:
            throttle = 0.0
            brake = 0.0
        elif speed_err > -self.cfg.soft_brake_speed_excess_kmh:
            throttle = 0.0
            brake = float(np.clip(((-speed_err) - self.cfg.coast_speed_band_kmh) / 24.0, 0.0, 0.10))
        else:
            throttle = 0.0
            brake = float(np.clip(0.10 + (((-speed_err) - self.cfg.soft_brake_speed_excess_kmh) / 10.0), 0.0, 0.42))

        meta = {
            "desired_speed_kmh": float(desired_speed_kmh),
            "route_idx": float(ref_idx),
            "route_target_idx": float(min(ref_idx + 1, len(self.route_wps) - 1)),
            "route_cte": float(signed_lane_err),
            "route_heading_err": float(heading_err),
            "route_remaining_s": float(remaining_s),
            "route_dL": float(dL),
            "wrong_lane": float(1.0 if wrong_lane else 0.0),
            "opposite_lane": float(1.0 if opposite_lane else 0.0),
            "center_push": float(center_push),
        }
        return np.array([throttle, brake, steer], dtype=np.float32), meta

    def route_guidance_action(self) -> np.ndarray:
        """Return the deterministic lane/traffic controller action.

        It is used only to bootstrap the replay buffer during the optional
        training curriculum and by the explicitly enabled safety shield.
        """
        if self.vehicle is None:
            return np.zeros(3, dtype=np.float32)
        tl_state, tl_dist = self._get_active_traffic_light_info()
        front_dist, front_speed, _ = self._get_front_vehicle_info()
        action, _ = self._compute_route_guidance_action(
            tl_state=tl_state,
            tl_dist=tl_dist,
            front_vehicle_dist=front_dist,
            front_vehicle_speed_kmh=front_speed,
        )
        return action.astype(np.float32)

    def _red_light_assist(self, a: np.ndarray) -> Tuple[np.ndarray, bool, str]:
        """Bounded braking when a red stop line is inside braking distance.

        Disabled unless cfg.red_light_assist is set, because every activation is
        counted in the reported intervention rate (Sec. IV-F).
        """
        if not bool(getattr(self.cfg, "red_light_assist", False)):
            return a, False, ""
        try:
            tl_state, tl_dist = self._get_active_traffic_light_info()
        except BaseException:
            return a, False, ""
        if tl_state != "Red" or tl_dist > 1e8:
            return a, False, ""
        onset = max(float(self.cfg.tl_stop_dist), self._red_brake_distance())
        if tl_dist < 0.0 or tl_dist > onset:
            return a, False, ""
        a = np.asarray(a, dtype=np.float32).copy()
        urgency = float(np.clip((onset - tl_dist) / max(onset, 1e-3), 0.0, 1.0))
        a[0] = 0.0
        a[1] = max(float(a[1]), float(np.clip(0.35 + 0.55 * urgency, 0.35, 1.0)))
        return a, True, "red_light"

    def _policy_passthrough_filter(self, raw_action: np.ndarray) -> Tuple[np.ndarray, Dict[str, float], bool]:
        """Apply actuator bounds only; no hidden route-controller blending."""
        assert self.vehicle is not None
        speed_kmh = 3.6 * vec3_length(self.vehicle.get_velocity())
        tl_state, tl_dist = self._get_active_traffic_light_info()
        front_dist, front_speed, _ = self._get_front_vehicle_info()
        _, route_meta = self._compute_route_guidance_action(
            tl_state=tl_state,
            tl_dist=tl_dist,
            front_vehicle_dist=front_dist,
            front_vehicle_speed_kmh=front_speed,
        )
        a = np.asarray(raw_action, dtype=np.float32).copy()
        a[0] = float(np.clip(a[0], 0.0, 1.0))
        a[1] = float(np.clip(a[1], 0.0, 1.0))
        a[2] = float(np.clip(a[2], -1.0, 1.0))
        if a[0] > 0.0 and a[1] > 0.0:
            if a[0] >= a[1]:
                a[1] = 0.0
            else:
                a[0] = 0.0
        # Actuator slew limits are modeled as vehicle-interface dynamics, not
        # route assistance. The replay buffer stores this applied action.
        a[0] = float(np.clip(a[0], self.prev_action[0] - self.cfg.max_throttle_step, self.prev_action[0] + self.cfg.max_throttle_step))
        a[1] = float(np.clip(a[1], self.prev_action[1] - self.cfg.max_brake_step, self.prev_action[1] + self.cfg.max_brake_step))
        a[2] = float(np.clip(a[2], self.prev_action[2] - self.cfg.max_steer_step, self.prev_action[2] + self.cfg.max_steer_step))
        route_meta.update({
            "safety_tl_state": tl_state,
            "safety_tl_dist": float(tl_dist),
            "safety_front_vehicle_dist": -1.0 if front_dist is None else float(front_dist),
            "safety_front_vehicle_speed_kmh": float(front_speed),
            "safety_front_ttc": 999.0,
            "safety_min_vehicle_ttc": 999.0,
            "safety_reason": "none",
            "policy_route_blend": 0.0,
            "policy_speed_kmh": float(speed_kmh),
        })
        # Optional bounded red-light braking (cfg.red_light_assist, off by
        # default). Counted as an intervention so it stays visible in the
        # reported intervention rate.
        a, red_assist, red_reason = self._red_light_assist(a)
        if red_assist:
            a[1] = float(np.clip(a[1], self.prev_action[1] - self.cfg.max_brake_step, self.prev_action[1] + self.cfg.max_brake_step))
            route_meta["safety_reason"] = red_reason
            return a.astype(np.float32), route_meta, True
        return a.astype(np.float32), route_meta, False

    def _safety_filter(self, raw_action: np.ndarray) -> Tuple[np.ndarray, Dict[str, float], bool]:
        """Optional bounded control stabilization from Table I.

        Nominal actions pass through unchanged. Interventions are activated only
        by route-constraint, TTC, red-light, stopped-lead-vehicle, overspeed, or
        prolonged-stall conditions, and every activation is reported.
        """
        assert self.vehicle is not None
        base, route_meta, _ = self._policy_passthrough_filter(raw_action)
        tl_state = str(route_meta.get("safety_tl_state", "None"))
        tl_dist = safe_float(route_meta.get("safety_tl_dist", 1e9), 1e9)
        front_dist_raw = safe_float(route_meta.get("safety_front_vehicle_dist", -1.0), -1.0)
        front_dist = None if front_dist_raw < 0.0 else front_dist_raw
        front_speed = safe_float(route_meta.get("safety_front_vehicle_speed_kmh", 0.0))
        guidance, guidance_meta = self._compute_route_guidance_action(
            tl_state=tl_state,
            tl_dist=tl_dist,
            front_vehicle_dist=front_dist,
            front_vehicle_speed_kmh=front_speed,
        )
        route_meta.update(guidance_meta)
        speed_kmh = 3.6 * vec3_length(self.vehicle.get_velocity())
        cte = abs(safe_float(route_meta.get("route_cte", 0.0)))
        hdg = abs(safe_float(route_meta.get("route_heading_err", 0.0)))
        wrong_lane = bool(safe_float(route_meta.get("wrong_lane", 0.0)) > 0.5)
        opposite_lane = bool(safe_float(route_meta.get("opposite_lane", 0.0)) > 0.5)
        route_bad = wrong_lane or opposite_lane or cte > self.cfg.route_override_dL_m or hdg > self.cfg.route_override_heading_rad
        route_hard = wrong_lane or opposite_lane or cte > self.cfg.route_hard_dL_m or hdg > self.cfg.route_hard_heading_rad

        ego_speed_ms = vec3_length(self.vehicle.get_velocity())
        front_ttc = 999.0
        if front_dist is not None:
            closing = max(0.0, ego_speed_ms - front_speed / 3.6)
            if closing > 1e-3:
                front_ttc = front_dist / closing
        min_ttc, min_ttc_dist = self._get_min_vehicle_ttc(max_dist=max(self.cfg.front_vehicle_max_dist, 25.0))
        red_stop = (
            tl_state == "Red"
            and tl_dist < 1e8
            and tl_dist <= max(self.cfg.tl_stop_dist + 0.5, self._red_brake_distance())
        )
        front_blocked = front_dist is not None and (
            front_dist < self.cfg.front_vehicle_block_dist
            or (front_ttc < self.cfg.hard_ttc_s and front_dist < self.cfg.front_vehicle_soft_block_dist)
        )
        open_road = (not red_stop) and (front_dist is None or front_dist > self.cfg.front_vehicle_soft_block_dist)

        a = base.copy()
        intervention = False
        reason = "none"
        blend = 0.0
        if route_hard:
            blend = float(self.cfg.policy_route_blend_hard)
            reason = "route_recovery"
        elif route_bad:
            blend = float(self.cfg.policy_route_blend_bad)
            reason = "route_stabilization"
        if blend > 0.0:
            a = (1.0 - blend) * a + blend * guidance
            intervention = True

        if min_ttc < self.cfg.caution_ttc_s and min_ttc_dist < 18.0 and not red_stop:
            a[0] = min(float(a[0]), 0.20)
            if min_ttc < self.cfg.hard_ttc_s:
                a[1] = max(float(a[1]), 0.35)
            intervention = True
            reason = "ttc"

        if speed_kmh > self.cfg.hard_speed_cap_kmh:
            a[0] = 0.0
            a[1] = max(float(a[1]), float(np.clip(0.20 + 0.08 * (speed_kmh - self.cfg.hard_speed_cap_kmh), 0.20, 1.0)))
            intervention = True
            reason = "speed_cap"

        if red_stop:
            a[0] = 0.0
            a[1] = max(float(a[1]), 0.80)
            a[2] = float(np.clip(guidance[2], -0.30, 0.30))
            intervention = True
            reason = "red_light"
        elif front_blocked:
            a[0] = 0.0
            a[1] = max(float(a[1]), 0.70)
            intervention = True
            reason = "front_vehicle"

        if speed_kmh < self.cfg.stuck_speed_kmh and open_road and not route_hard:
            self.stuck_steps += 1
        else:
            self.stuck_steps = 0
        if self.stuck_steps >= self.cfg.stuck_steps_threshold and open_road:
            self.release_steps_left = max(self.release_steps_left, self.cfg.release_duration_steps)
            self.stuck_steps = 0
        if self.release_steps_left > 0 and open_road:
            a = guidance.copy()
            a[0] = max(float(a[0]), self.cfg.release_throttle)
            a[1] = 0.0
            a[2] = float(np.clip(a[2], -self.cfg.low_speed_steer_limit, self.cfg.low_speed_steer_limit))
            self.release_steps_left -= 1
            intervention = True
            reason = "unstuck"

        a[0] = float(np.clip(a[0], 0.0, 1.0))
        a[1] = float(np.clip(a[1], 0.0, 1.0))
        a[2] = float(np.clip(a[2], -1.0, 1.0))
        if a[0] > 0.0 and a[1] > 0.0:
            if a[0] >= a[1]:
                a[1] = 0.0
            else:
                a[0] = 0.0
        route_meta.update({
            "safety_front_ttc": float(front_ttc),
            "safety_min_vehicle_ttc": float(min_ttc),
            "safety_reason": reason,
            "policy_route_blend": float(blend),
        })
        return a.astype(np.float32), route_meta, bool(intervention)

    def _actor_is_alive(self, actor: Optional[carla.Actor]) -> bool:
        if actor is None:
            return False
        try:
            return bool(actor.is_alive)
        except BaseException:
            return False

    def _safe_set_autopilot(self, actor: Optional[carla.Actor], enabled: bool) -> None:
        if not self._actor_is_alive(actor):
            return
        try:
            actor.set_autopilot(bool(enabled), self.tm.get_port())
            return
        except TypeError:
            pass
        except BaseException:
            pass
        try:
            actor.set_autopilot(bool(enabled))
        except BaseException:
            pass

    def _live_actor_ids(self) -> set:
        """Ids the server currently knows, from one world snapshot.

        Cheaper and fresher than per-actor get_actor() calls, and unlike the
        cached handle's is_alive it reflects server-side collection (e.g. a
        walker removed mid-episode, which also removes its attached
        controller).
        """
        try:
            return {int(a.id) for a in self.world.get_actors()}
        except BaseException:
            return set()

    def _batch_destroy_ids(self, actors: Sequence[Optional[carla.Actor]], label: str = "batch") -> int:
        """Destroy actors through client.apply_batch_sync(DestroyActor(...)).

        The batch command path is the CARLA-recommended teardown: ids absent
        from the server are reported in the response instead of raising, which
        avoids the "unable to destroy actor: not found" server error lines that
        per-actor destroy() emits from the C++ layer.
        """
        live_ids = self._live_actor_ids()
        ids: List[int] = []
        for actor in actors:
            if actor is None:
                continue
            try:
                aid = int(actor.id)
            except BaseException:
                continue
            if aid in live_ids and aid not in ids:
                ids.append(aid)
        if not ids:
            return 0
        try:
            self.client.apply_batch_sync([carla.command.DestroyActor(x) for x in ids], True)
        except BaseException as e:
            if self.cfg.debug_mode:
                print(f"[WARN] {label} batch destroy failed ({e}); falling back to per-actor destroy")
            for actor in actors:
                live = self._refetch(actor)
                if live is not None:
                    self._safe_destroy_one(live, sleep_s=0.02)
        self._drain_world_ticks(1, f"{label}_post_batch_destroy", sleep_s=0.02)
        return len(ids)

    def _refetch(self, actor: Optional[carla.Actor]) -> Optional[carla.Actor]:
        """Re-fetch an actor by id from the current world snapshot.

        Returns None when the server no longer knows the actor. The cached
        client-side handle's is_alive can be stale in synchronous mode, so
        destroying through a fresh handle avoids CARLA's "already dead" /
        "unable to destroy actor: not found" teardown warnings.
        """
        if actor is None:
            return None
        try:
            return self.world.get_actor(int(actor.id))
        except BaseException:
            return None

    def _safe_destroy_one(self, actor: Optional[carla.Actor], sleep_s: float = 0.0) -> None:
        if not self._actor_is_alive(actor):
            return
        try:
            actor.destroy()
        except RuntimeError as e:
            # Ignore the exact error we are fixing.
            if "destroyed actor" not in str(e).lower() and self.cfg.debug_mode:
                print(f"[WARN] destroy actor runtime error: {e}")
        except BaseException as e:
            if self.cfg.debug_mode:
                print(f"[WARN] destroy actor failed: {e}")
        if sleep_s > 0.0:
            try:
                time.sleep(sleep_s)
            except BaseException:
                pass

    def _drain_world_ticks(self, count: int, label: str, sleep_s: float = 0.03) -> None:
        for i in range(max(0, int(count))):
            self._safe_tick(label=f"{label}_{i+1}", raise_on_fail=False)
            if sleep_s > 0.0:
                try:
                    time.sleep(sleep_s)
                except BaseException:
                    pass

    def _on_collision_event(self, event: object) -> None:
        if self._teardown_in_progress or (not self._episode_live):
            return
        if not self._actor_is_alive(self.vehicle):
            return
        try:
            self.collision_events.append(event)
        except BaseException:
            pass

    def _on_lane_invasion_event(self, event: object) -> None:
        if self._teardown_in_progress or (not self._episode_live):
            return
        if not self._actor_is_alive(self.vehicle):
            return
        try:
            self.lane_invasion_events.append(event)
        except BaseException:
            pass

    def _begin_episode_teardown(self, reason: str = "") -> None:
        """
        Stop new callbacks and detach Traffic Manager ownership before reset().
        This is intentionally lightweight and idempotent.
        """
        if self._teardown_in_progress:
            return

        self._teardown_in_progress = True
        self._episode_live = False
        self._terminal_reason = str(reason or self._terminal_reason or "")

        # Stop sensors first so no new callbacks are queued.
        for sensor in list(self.sensor_list):
            self._stop_sensor_only(sensor)

        # Detach TM from NPCs before any future destroy.
        for npc in list(self.npcs):
            self._safe_set_autopilot(npc, False)

        # Freeze ego control.
        if self._actor_is_alive(self.vehicle):
            self._safe_set_autopilot(self.vehicle, False)
            try:
                self.vehicle.apply_control(
                    carla.VehicleControl(
                        throttle=0.0,
                        steer=0.0,
                        brake=1.0,
                        hand_brake=False,
                        reverse=False,
                        manual_gear_shift=False,
                    )
                )
            except BaseException:
                pass

        # One drain tick here helps clear queued callbacks before reset() starts.
       # self._drain_world_ticks(1, "terminal_teardown", sleep_s=0.02)    

    def _setup_collision(self) -> None:
        if not self.cfg.enable_collision_sensor or not self._actor_is_alive(self.vehicle):
            return
        try:
            bp = self.bp_lib.find("sensor.other.collision")
            sensor = self.world.spawn_actor(bp, carla.Transform(), attach_to=self.vehicle)
            sensor.listen(self._on_collision_event)
            self.sensor_list.append(sensor)
        except Exception as e:
            if self.cfg.debug_mode:
                print(f"[WARN] collision sensor setup failed: {e}")

    def _setup_lane_invasion(self) -> None:
        if not self.cfg.enable_lane_invasion_sensor or not self._actor_is_alive(self.vehicle):
            return
        try:
            bp = self.bp_lib.find("sensor.other.lane_invasion")
            sensor = self.world.spawn_actor(bp, carla.Transform(), attach_to=self.vehicle)
            sensor.listen(self._on_lane_invasion_event)
            self.sensor_list.append(sensor)
        except Exception as e:
            if self.cfg.debug_mode:
                print(f"[WARN] lane invasion sensor setup failed: {e}")

    def _tick(self) -> None:
        self._safe_tick(label="step")
        self._update_spectator()

    def _ego_spawn_candidate_indices(self) -> List[int]:
        n = len(self.spawn_points)
        if n == 0:
            return [0]
        # fixed_spawn_index < 0  → fully random spawn each episode (prevents
        # overfitting to a single scenario; confirmed necessary from training log
        # where 97% collisions all happened at the same map location).
        if self.fixed_spawn_index < 0:
            indices = list(range(n))
            random.shuffle(indices)
            return indices
        # fixed_spawn_index >= 0 → deterministic base + nearby fallbacks
        offsets = [0, 1, 2, 3, 5, 8, 13]
        out: List[int] = []
        seen: set = set()
        for off in offsets:
            idx = (self.fixed_spawn_index + off) % n
            if idx not in seen:
                out.append(idx)
                seen.add(idx)
        return out

    def _route_exists_between(self, start_tf: carla.Transform, goal_index: int) -> bool:
        grp = self._get_global_planner()
        if grp is None:
            return False
        if goal_index < 0 or goal_index >= len(self.spawn_points):
            return False
        try:
            start_wp = self._project_drive_wp(start_tf.location)
            goal_wp = self._project_drive_wp(self.spawn_points[goal_index].location)
            if start_wp is None or goal_wp is None:
                return False
            start_loc = start_wp.transform.location
            goal_loc = goal_wp.transform.location
            if distance2d(start_loc, goal_loc) < self.cfg.candidate_goal_min_dist_m:
                return False
            trace = grp.trace_route(start_loc, goal_loc)
            if len(trace) < 2:
                return False
            route = [wp for wp, _ in trace if wp is not None]
            route = self._dedupe_waypoints(route, min_sep=0.75)
            if len(route) < 2:
                return False
            route_len = self._route_length_from_wps(route)
            return self._route_length_ok(route_len, strict=True)
        except BaseException:
            return False

    def _spawn_ego(self, preferred_goal_index: Optional[int] = None) -> carla.Transform:
        bp = self.bp_lib.find(self.cfg.car_name)
        if bp.has_attribute("color"):
            bp.set_attribute("color", self.cfg.fixed_color)
        if bp.has_attribute("role_name"):
            bp.set_attribute("role_name", self.ego_role)

        base_candidates = self._ego_spawn_candidate_indices()
        ordered_indices: List[int] = []
        seen: set[int] = set()

        def _append_idx(idx: int) -> None:
            if idx not in seen:
                ordered_indices.append(idx)
                seen.add(idx)

        for idx in base_candidates:
            _append_idx(idx)
        for idx in range(len(self.spawn_points)):
            _append_idx(idx)

        last_spawn_error: Optional[Exception] = None
        best_route_issue: Optional[str] = None

        for idx in ordered_indices:
            raw_tf = self.spawn_points[idx]
            snapped_tf = self._snap_spawn_to_driving_lane(raw_tf)
            spawn_bases: List[carla.Transform] = [snapped_tf]

            raw_offset = distance2d(raw_tf.location, snapped_tf.location)
            if raw_offset > 0.75:
                spawn_bases.append(raw_tf)
                if self.cfg.debug_mode:
                    print(
                        f"[SPAWN] spawn {idx} snapped to lane center by {raw_offset:.2f}m "
                        f"for map {self.town_name}"
                    )

            if self.cfg.destroy_stale_spawn_blockers:
                cleared = 0
                for base_tf in spawn_bases:
                    cleared += self._clear_spawn_blockers(base_tf)
                if cleared > 0 and self.cfg.debug_mode:
                    print(f"[SPAWN] cleared {cleared} blocking actor(s) near spawn {idx}")

            # Cheap pre-check: skip fixed-goal spawns that definitely cannot route.
            if preferred_goal_index is not None and int(preferred_goal_index) >= 0 and self.cfg.strict_goal_route:
                try:
                    start_wp_hint = self._project_drive_wp(snapped_tf.location)
                    if start_wp_hint is None or (not self._route_exists_between(snapped_tf, int(preferred_goal_index))):
                        best_route_issue = f"No valid fixed-goal route from spawn {idx} to goal {int(preferred_goal_index)}"
                        continue
                except Exception as e:
                    best_route_issue = f"Route precheck failed at spawn {idx}: {e}"
                    continue

            for base_tf in spawn_bases:
                for tf in self._spawn_transform_variants(base_tf):
                    vehicle = None

                    try:
                        vehicle = self.world.try_spawn_actor(bp, tf)
                    except Exception as e:
                        last_spawn_error = e
                        vehicle = None

                    if vehicle is None and self.cfg.destroy_stale_spawn_blockers:
                        cleared = self._clear_spawn_blockers(tf, radius=max(self.cfg.spawn_blocker_radius_m, 3.0))
                        if cleared > 0 and self.cfg.debug_mode:
                            print(f"[SPAWN] retry after clearing {cleared} actor(s) near spawn {idx}")
                        try:
                            vehicle = self.world.try_spawn_actor(bp, tf)
                        except Exception as e:
                            last_spawn_error = e
                            vehicle = None

                    if vehicle is None:
                        continue

                    self.vehicle = vehicle
                    try:
                        self.vehicle.set_autopilot(False)
                    except BaseException:
                        pass
                    self.current_spawn_index = idx
                    self._snap_spectator_to_ego()
                    return self.vehicle.get_transform()

            time.sleep(0.05)

        if preferred_goal_index is not None and int(preferred_goal_index) >= 0 and self.cfg.strict_goal_route and best_route_issue is not None:
            raise RuntimeError(
                f"Failed to find a route-valid spawn for goal {int(preferred_goal_index)} in map {self.town_name}. "
                f"Last issue: {best_route_issue}."
            )
        if last_spawn_error is not None:
            raise RuntimeError(f"Failed to spawn ego vehicle after trying multiple spawn points: {last_spawn_error}")
        raise RuntimeError(
            f"Failed to spawn ego vehicle after trying multiple spawn points in map {self.town_name}. "
            "Spawn locations appear occupied or blocked by residual actors."
        )


    def _npc_blueprints(self) -> List[carla.ActorBlueprint]:
        """Cars and vans only. Bicycles, motorcycles, buses and trucks are the usual cause of Traffic
        Manager jams at junctions (slow two-wheelers, long vehicles that cannot finish a turn)."""
        out = []
        for bp in self.bp_lib.filter("vehicle.*"):
            try:
                if bp.has_attribute("base_type"):
                    if bp.get_attribute("base_type").as_str().lower() not in ("car", "van"):
                        continue
                elif bp.has_attribute("number_of_wheels") and bp.get_attribute("number_of_wheels").as_int() != 4:
                    continue
            except BaseException:
                continue
            out.append(bp)
        return out or list(self.bp_lib.filter("vehicle.*"))

    def _spawn_npcs(self, npc_count: int) -> None:
        npc_bps = self._npc_blueprints()
        if not npc_bps or self.vehicle is None or npc_count <= 0:
            return
        ego_loc = self.vehicle.get_location()
        goal_loc = self.goal_transform.location if self.goal_transform is not None else None
        # Prefer spawn points near the planned route so NPC traffic is dense
        # where the ego actually drives (Sec. IV-A dense-traffic protocol)
        # rather than scattered across the whole town; remaining slots fall
        # back to town-wide spawn points.
        anchors: List[carla.Location] = []
        if self.route_wps:
            step = max(1, len(self.route_wps) // 16)
            anchors = [self.route_wps[i].transform.location for i in range(0, len(self.route_wps), step)]
        near_route: List[carla.Transform] = []
        far_route: List[carla.Transform] = []
        for sp in self.spawn_points:
            if distance2d(sp.location, ego_loc) <= 25.0:
                continue
            if goal_loc is not None and distance2d(sp.location, goal_loc) <= 20.0:
                continue
            if anchors and min(distance2d(sp.location, a) for a in anchors) <= self.cfg.npc_spawn_radius_m:
                near_route.append(sp)
            else:
                far_route.append(sp)
        random.shuffle(near_route)
        random.shuffle(far_route)
        candidates = near_route + far_route
        for sp in candidates[: max(0, npc_count * 5)]:
            if len(self.npcs) >= npc_count:
                break
            npc_bp = random.choice(npc_bps)
            if npc_bp.has_attribute("role_name"):
                npc_bp.set_attribute("role_name", self.npc_role)
            npc = self.world.try_spawn_actor(npc_bp, sp)
            if npc is None:
                continue
            try:
                npc.set_autopilot(True, self.tm.get_port())
                # Speeds between 10% above and 25% below the limit, no lane changes: speeding and
                # lane changes inside junctions are the main sources of Traffic Manager gridlock.
                self.tm.vehicle_percentage_speed_difference(npc, random.uniform(-10.0, 25.0))
                self.tm.distance_to_leading_vehicle(npc, random.uniform(2.5, 5.0))
                self.tm.auto_lane_change(npc, False)
                self.npcs.append(npc)
            except BaseException:
                self._destroy_actor(npc)

    def _spawn_walkers(self, walker_count: int) -> None:
        """Spawn pedestrian NPCs with AI controllers (Sec. III-C / Table I).

        Walkers are biased toward the planned route so they are
        control-relevant entities for the ego-relational state and for the
        pedestrian-crossing interaction scenarios (Fig. 2 / Fig. 12).
        """
        if walker_count <= 0 or self.vehicle is None:
            return
        try:
            self.world.set_pedestrians_cross_factor(
                float(np.clip(self.cfg.walker_cross_factor, 0.0, 1.0))
            )
        except BaseException:
            pass
        walker_bps = list(self.bp_lib.filter("walker.pedestrian.*"))
        try:
            controller_bp = self.bp_lib.find("controller.ai.walker")
        except BaseException:
            controller_bp = None
        if not walker_bps or controller_bp is None:
            return

        anchors: List[carla.Location] = []
        if self.route_wps:
            step = max(1, len(self.route_wps) // 12)
            anchors = [self.route_wps[i].transform.location for i in range(0, len(self.route_wps), step)]

        ego_loc = self.vehicle.get_location()
        spawned = 0
        attempts = 0
        max_attempts = max(walker_count * 8, 16)
        while spawned < walker_count and attempts < max_attempts:
            attempts += 1
            try:
                loc = self.world.get_random_location_from_navigation()
            except BaseException:
                loc = None
            if loc is None:
                continue
            if distance2d(loc, ego_loc) < 8.0:
                continue
            if anchors:
                near = min(distance2d(loc, a) for a in anchors)
                # Mostly keep walkers near the route; allow a few far ones.
                if near > self.cfg.walker_spawn_radius_m and random.random() < 0.8:
                    continue
            bp = random.choice(walker_bps)
            if bp.has_attribute("is_invincible"):
                bp.set_attribute("is_invincible", "false")
            if bp.has_attribute("role_name"):
                bp.set_attribute("role_name", self.walker_role)
            tf = carla.Transform(carla.Location(loc.x, loc.y, loc.z + 0.3))
            walker = self.world.try_spawn_actor(bp, tf)
            if walker is None:
                continue
            try:
                controller = self.world.spawn_actor(controller_bp, carla.Transform(), attach_to=walker)
            except BaseException:
                controller = None
            if controller is None:
                self._destroy_actor(walker)
                continue
            self.walkers.append(walker)
            self.walker_controllers.append(controller)
            spawned += 1

        if spawned <= 0:
            return
        # One tick so controllers register before start()/go_to_location().
        self._safe_tick(label="walker_spawn_settle", raise_on_fail=False)
        for controller in self.walker_controllers:
            try:
                controller.start()
                target = self.world.get_random_location_from_navigation()
                if target is not None:
                    controller.go_to_location(target)
                controller.set_max_speed(float(random.uniform(0.8, 2.0)))
            except BaseException:
                continue

    def reset(self, *, seed: Optional[int] = None, options: Optional[Dict] = None):
        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)
        self._episode_live = False
        self._terminal_reason = ""

        self._cleanup_episode_actors()
        self._sweep_foreign_actors(startup=False)
        if self.cfg.destroy_stale_owned_actors_on_reset:
            cleared = self._destroy_residual_owned_actors()
            if cleared > 0 and self.cfg.debug_mode:
                print(f"[RESET] destroyed {cleared} residual owned actor(s) before spawning")
        apply_weather(self.world, self.weather_mode)

        opt = options or {}
        npc_requested = int(opt.get("npc_count", random.randint(self.cfg.npc_min, self.cfg.npc_max)))
        npc_cap = max(int(self.cfg.npc_max), int(self.cfg.train_npc_max))
        npc_count = int(np.clip(npc_requested, 0, npc_cap))
        mode_l = str(self.cfg.mode_name).strip().lower()
        if mode_l in ("train", "adapt", "policy"):
            walker_lo, walker_hi = int(self.cfg.train_walker_min), int(self.cfg.train_walker_max)
        else:
            walker_lo, walker_hi = int(self.cfg.walker_min), int(self.cfg.walker_max)
        walker_lo, walker_hi = min(walker_lo, walker_hi), max(walker_lo, walker_hi)
        walker_requested = int(opt.get("walker_count", random.randint(walker_lo, walker_hi)))
        walker_count = int(np.clip(walker_requested, 0, 40))
        goal_index_raw = opt.get("goal_index", self.fixed_goal_index if self.cfg.use_fixed_destination else None)
        goal_index = None if goal_index_raw is None or int(goal_index_raw) < 0 else int(goal_index_raw)

        self.episode_steps = 0
        self.prev_action[:] = 0.0
        self.prev_steer = 0.0
        self.prev_acc[:] = 0.0
        self.prev_loc = None
        self.distance_driven_m = 0.0
        self.stuck_steps = 0
        self.release_steps_left = 0
        self.offroute_steps = 0
        self.prev_route_s = None
        self.safety_interventions = 0
        self.blocked_steps_credit = 0
        self._free_stuck_steps = 0
        self.route_progress_idx = 0
        self.route_total_len_m = 0.0

        requested_goal_index = goal_index
        spawn_tf = self._spawn_ego(preferred_goal_index=requested_goal_index)
        start_wp = self.map.get_waypoint(spawn_tf.location, project_to_road=True, lane_type=carla.LaneType.Driving)
        if start_wp is None:
            raise RuntimeError("Could not obtain a driving waypoint for the ego spawn location.")

        self._build_route(spawn_tf, start_wp, goal_index=requested_goal_index)

        self._snap_spectator_to_ego()
        for _ in range(max(0, self.cfg.post_spawn_settle_ticks)):
            self.vehicle.apply_control(carla.VehicleControl(throttle=0.0, steer=0.0, brake=0.0))
            self._safe_tick(label="reset_settle")
            self._snap_spectator_to_ego()

        for _ in range(max(6, self.cfg.warmup_reset_ticks)):
            self.vehicle.apply_control(carla.VehicleControl(throttle=0.0, steer=0.0, brake=0.0))
            self._tick()

                # If we are using local fallback routing (no planner goal selected),
        # rebuild the local route after the ego has settled on the road.
        if requested_goal_index is None and self.current_goal_index is None:
            # Anchor the lane-follow route to the pose the ego actually settled
            # in. The pre-settle route starts from the spawn waypoint, which
            # can sit metres away once physics settles the vehicle onto the
            # lane, so rebuilding here guarantees start_progress ~ 0 and
            # start_dL ~ 0 without a conditional check.
            rebuilt = self._rebuild_local_route_from_current_pose()
            if not rebuilt and self.cfg.debug_mode:
                print("[WARN] lane-follow route rebuild failed; keeping pre-settle route")

        # Episode start: project within the opening window so a route that
        # loops back near its origin cannot report spurious start progress.
        s0_global, d0_global, idx0_global = self._route_projection_start_window(self.vehicle.get_location())
        progress0_pct = 100.0 * s0_global / max(self.route_total_len_m, 1e-6)

        start_to_route0 = d0_global
        if self.route_wps:
            try:
                start_to_route0 = distance2d(self.vehicle.get_location(), self.route_wps[0].transform.location)
            except BaseException:
                pass

        if progress0_pct > self.cfg.max_reset_start_progress_pct or d0_global > self.cfg.max_reset_start_dL_m:
            # For local fallback routes, do not hard-fail reset after rebuild.
            if requested_goal_index is None and self.current_goal_index is None:
                if self.cfg.debug_mode:
                    print(
                        f"[WARN] relaxing reset alignment for local fallback route: "
                        f"progress={progress0_pct:.2f}% dL={d0_global:.2f}m"
                    )
                idx0_global = 0
                s0_global = 0.0
                progress0_pct = 0.0
                d0_global = 0.0
            elif start_to_route0 <= max(self.cfg.max_reset_start_dL_m, 4.0):
                if self.cfg.debug_mode:
                    print(
                        f"[WARN] reset alignment relaxed: progress={progress0_pct:.2f}% dL={d0_global:.2f}m; "
                        "clamping start progress to route origin"
                    )
                idx0_global = 0
                s0_global = 0.0
                progress0_pct = 0.0
            else:
                raise RuntimeError(
                    f"Invalid reset alignment: progress={progress0_pct:.2f}% dL={d0_global:.2f}m "
                    f"route_len={self.route_total_len_m:.2f}m goal_idx={self.current_goal_index}"
                )


        self.route_progress_idx = idx0_global
        self.prev_route_s = s0_global

        self._spawn_npcs(npc_count)
        self._spawn_walkers(walker_count)

        self._setup_collision()
        self._setup_lane_invasion()
        self._safe_tick(label="post_sensor_attach", raise_on_fail=False)
        self._snap_spectator_to_ego()
        self.prev_loc = self.vehicle.get_location()
        self._reset_stop_line_tracker()
        self._teardown_in_progress = False
        self._episode_live = True

        obs = self._get_obs()
        info = {
            "town": self.town_name,
            "npc_count": len(self.npcs),
            "walker_count": len(self.walkers),
            "weather": self.weather_mode,
            "spawn_index": self.current_spawn_index,
            "goal_index": self.current_goal_index,
            "requested_goal_index": requested_goal_index,
            "route_total_len_m": float(self.route_total_len_m),
            "carla_server_version": getattr(self.client, "get_server_version", lambda: "unknown")(),
            "carla_client_version": getattr(self.client, "get_client_version", lambda: "unknown")(),
        }
        if self.cfg.debug_mode:
            tf = self.vehicle.get_transform()
            print(
                f"[RESET] town={info['town']} npc={info['npc_count']} walkers={info['walker_count']} "
                f"weather={info['weather']} "
                f"spawn_idx={info['spawn_index']} goal_idx={info['goal_index']} route_len={info['route_total_len_m']:.1f}m "
                f"spawn=({tf.location.x:.1f},{tf.location.y:.1f},{tf.location.z:.1f}) yaw={tf.rotation.yaw:.1f} "
                f"start_progress={progress0_pct:.2f}% start_dL={d0_global:.2f}"
            )
        return obs, info

    def _compute_reward_components(self, applied_action: np.ndarray) -> Tuple[float, float, float, Dict[str, float]]:
        """Dense multi-objective reward shaping (paper Sec. III-D, Eqs. 8–9).

        r_t = w_s r_s + w_p r_p + w_c r_c + w_u r_u    (Eq. 8)

        r_s: safety  – lane adherence + proximity + red-light   (Sec. III-D)
        r_p: progress – tanh(Δs / τ_s) or tanh(v̂_t / τ_v)      (Sec. III-D)
        r_c: comfort  – quadratic jerk + steer-rate penalty     (Sec. III-D)
        r_u: uncertainty – 1 − σ̄  (computed externally, Eq. 9)

        No terminal bonus / penalty (goal_bonus = collision_penalty = 0).
        """
        assert self.vehicle is not None
        loc = self.vehicle.get_location()
        vel = self.vehicle.get_velocity()
        speed = vec3_length(vel)
        wp_onlane = self.map.get_waypoint(loc, project_to_road=True, lane_type=carla.LaneType.Driving)
        lane_curv = self._curvature_ahead(wp_onlane)
        lane_curv_s = float(np.tanh(10.0 * lane_curv))
        lane_head = self._lane_heading_offset(wp_onlane)
        s_arc, dL = self._route_projection(loc)
        route_obs = self._route_observation_features()
        route_heading_err = float(route_obs.get('route_heading', 0.0)) * (math.pi / 2.0)
        route_cte_norm = float(route_obs.get('route_cte', 0.0))
        delta_s = 0.0 if self.prev_route_s is None else float(np.clip(s_arc - self.prev_route_s, -2.0, 2.0))

        fog_norm = get_fog_norm(self.world)
        density_den = self.cfg.train_npc_max if str(self.cfg.mode_name).lower() in ("train", "adapt", "policy") else self.cfg.npc_max
        density_norm = float(np.clip(len(self.npcs) / max(density_den, 1), 0.0, 1.0))
        curv_norm = float(np.clip(abs(lane_curv_s), 0.0, 1.0))
        # μ_A ∈ [0,1] (Sec. III-B): membership of fragile conditions (rain/fog,
        # dense traffic, sharp curvature). ε(μ_A) = ε_min + (ε_max − ε_min)·μ_A,
        # so the admissible corridor tolerance adapts with the scene context.
        muA = float(np.clip(0.45 * density_norm + 0.35 * fog_norm + 0.20 * curv_norm, 0.0, 1.0))
        eps_mu = self.cfg.eps_min + (self.cfg.eps_max - self.cfg.eps_min) * muA
        # Paper Eq. (8): ψ_L = tanh(d_L / τ_d). The adaptive corridor
        # epsilon(mu_A) is used for constraint/recovery logic, not inserted into
        # the printed reward equation.
        psiL = math.tanh(dL / max(self.cfg.tau_d, 1e-3))

        # Eq. (4) phi_L(d_L, mu_A): corridor excess. Zero inside the admissible
        # corridor, quadratic outside it, so the gradient survives where psi_L
        # has saturated. eps_mu was computed above from mu_A.
        corridor_excess = max(0.0, float(dL) - float(eps_mu))
        phi_L = float(np.clip((corridor_excess / max(eps_mu, 1e-3)) ** 2, 0.0, self.cfg.phi_L_cap))
        # route_obs already exposes these as 0/1 floats (see _route_observation_features).
        wrong_lane_f = float(route_obs.get("wrong_lane", 0.0))
        opposite_lane_f = float(route_obs.get("opposite_lane", 0.0))
        phi_lane_total = float(
            self.cfg.lambda_phi_L * phi_L
            + self.cfg.lambda_wrong_lane * wrong_lane_f
            + self.cfg.lambda_opposite_lane * opposite_lane_f
        )

        # Proximity and time-to-conflict τ_t (Sec. III-B)
        nearest = self.cfg.entity_max_dist
        tau_t = 999.0
        ego_vel = np.array([vel.x, vel.y], dtype=np.float32)
        for npc in self.npcs:
            if npc is None or not npc.is_alive:
                continue
            nloc = npc.get_location()
            dist = distance2d(loc, nloc)
            if dist < nearest:
                nearest = dist
            nvel = npc.get_velocity()
            rel_p = np.array([nloc.x - loc.x, nloc.y - loc.y], dtype=np.float32)
            rel_v = np.array([nvel.x, nvel.y], dtype=np.float32) - ego_vel
            closing = max(0.0, -float(np.dot(rel_p, rel_v)) / max(float(np.linalg.norm(rel_p)), 1e-6))
            if dist > 0.5:
                tau = dist / max(closing, 1e-3) if closing > 1e-6 else 999.0
                tau_t = min(tau_t, tau)
        for walker in self._nearby_walkers(loc):
            try:
                wloc = walker.get_location()
                dist = distance2d(loc, wloc)
                nearest = min(nearest, dist)
                wvel = walker.get_velocity()
                rel_p = np.array([wloc.x - loc.x, wloc.y - loc.y], dtype=np.float32)
                rel_v = np.array([wvel.x, wvel.y], dtype=np.float32) - ego_vel
                closing = max(0.0, -float(np.dot(rel_p, rel_v)) / max(float(np.linalg.norm(rel_p)), 1e-6))
                if dist > 0.5:
                    tau = dist / max(closing, 1e-3) if closing > 1e-6 else 999.0
                    tau_t = min(tau_t, tau)
            except BaseException:
                continue
        # ψ_P: proximity surrogate ψ_P = exp(-dist/τ_p)  (Sec. III-D)
        psiP = math.exp(-nearest / max(self.cfg.tau_p, 1e-3))

        # Red-light noncompliance ρ_t ∈ [0,1]  (Sec. III-D).
        # The onset distance follows the braking requirement, so the penalty
        # appears while stopping is still physically possible; crossing the line
        # on red saturates it at 1.
        rho = 0.0
        red_violation = bool(self._step_red_light_events)
        tl_state, tl_dist = self._get_active_traffic_light_info()
        if tl_state == "Red" and tl_dist < 1e8:
            onset = max(float(self.cfg.tl_stop_dist), self._red_brake_distance())
            if 0.0 <= tl_dist < onset:
                z1 = float(np.clip((onset - tl_dist) / max(onset, 1e-3), 0.0, 1.0))
                z2 = float(np.clip(speed / 3.0, 0.0, 1.0))
                rho = float(np.clip(z1 * z2, 0.0, 1.0))

        if red_violation:
            rho = 1.0

        # Sec. III-D: r_s = 1 − κ_L ψ_L − κ_P ψ_P − κ_R ρ_t
        rs = float(np.clip(1.0 - self.cfg.k_l * psiL - self.cfg.k_p * psiP - self.cfg.k_r * rho, -2.0, 1.0))

        # Sec. III-D: use the incremental arc-length form by default.
        # rp_arc_weight remains a compatibility option for the stated velocity
        # surrogate, but the two alternatives are not mixed in the default run.
        # Progress is credited only up to the road's speed limit, so faster-than-legal driving earns nothing extra.
        try:
            v_lim = float(self.vehicle.get_speed_limit()) / 3.6
        except BaseException:
            v_lim = 0.0
        if not (v_lim > 1.0):
            v_lim = 30.0 / 3.6
        ds_credit = min(delta_s, v_lim * float(self.cfg.dt))
        rp_arc = float(math.tanh(ds_credit / max(self.cfg.tau_s, 1e-3)))
        # Velocity projection onto route tangent v̂_t (Sec. III-D surrogate)
        v_hat = float(route_obs.get('v_route', 0.0)) * 10.0  # de-normalise
        rp_vel = float(math.tanh(v_hat / max(self.cfg.tau_v, 1e-3)))
        w_arc = float(np.clip(self.cfg.rp_arc_weight, 0.0, 1.0))
        rp = float(w_arc * rp_arc + (1.0 - w_arc) * rp_vel)

        # Sec. III-D: r_c = −κ_j j_t² − κ_δ δ̇_t²
        acc = self.vehicle.get_acceleration()
        acc2 = np.array([acc.x, acc.y], dtype=np.float32)
        jerk = float(np.linalg.norm(acc2 - self.prev_acc) / max(self.cfg.dt, 1e-3))
        steer_rate = float((applied_action[2] - self.prev_steer) / max(self.cfg.dt, 1e-3))
        lane_cross = 1.0 if len(self.lane_invasion_events) > 0 else 0.0

        rc_raw = -(self.cfg.k_j * (jerk ** 2)) - (self.cfg.k_delta * (steer_rate ** 2))
        rc = float(np.clip(rc_raw, -5.0, 0.0))

        return rs, rp, rc, {
            "dL": float(dL),
            "route_s": float(s_arc),
            "delta_s": float(delta_s),
            "muA": float(muA),
            "eps_mu": float(eps_mu),
            "psiL": float(psiL),
            "nearest_entity_dist": float(nearest),
            "psiP": float(psiP),
            "rho_red": float(rho),
            "phi_lane": float(phi_lane_total),
            "phi_L": float(phi_L),
            "wrong_lane": float(wrong_lane_f),
            "opposite_lane": float(opposite_lane_f),
            "red_light_violation": bool(red_violation),
            "red_light_events": list(self._step_red_light_events),
            "speed_ms": float(speed),
            "jerk": float(jerk),
            "steer_rate": float(steer_rate),
            "lane_curv": float(lane_curv_s),
            "lane_head": float(lane_head),
            "route_heading_err": float(route_heading_err),
            "route_cte_norm": float(route_cte_norm),
            "lane_cross": float(lane_cross),
            "rc_raw": float(rc_raw),
            "rp_arc": float(rp_arc),
            "rp_vel": float(rp_vel),
            "time_to_conflict": float(tau_t),
            "tl_state": tl_state,
            "tl_dist": float(tl_dist) if tl_dist < 1e8 else -1.0,
        }

    def _apply_lateral_control(self, raw: np.ndarray) -> np.ndarray:
        """With --lateral-control controller the steering comes from route tracking (heading and cross-track
        terms of the route-guidance controller, the same law the teacher steers with) and the policy decides
        the longitudinal command only. The stored command carries the executed steering, so the critic sees
        what was actually done."""
        if str(getattr(self.cfg, "lateral_control", "policy")) != "controller":
            return raw
        try:
            raw[2] = float(np.clip(self._compute_route_guidance_action("None", 1e9, None, 0.0)[0][2], -1.0, 1.0))
        except Exception:
            pass
        return raw

    def step(self, action: np.ndarray):
        if self._teardown_in_progress:
            raise RuntimeError("step() called while teardown is in progress")
        if not self._actor_is_alive(self.vehicle):
            raise RuntimeError("step() called with missing/destroyed ego actor")

        self.episode_steps += 1
        self.lane_invasion_events.clear()

        raw = np.array(action, dtype=np.float32).copy()
        raw[0] = float(np.clip(raw[0], 0.0, 1.0))
        raw[1] = float(np.clip(raw[1], 0.0, 1.0))
        raw[2] = float(np.clip(raw[2], -1.0, 1.0))
        raw = self._apply_lateral_control(raw)
        # The replay action is the command selected by the policy.  Deadzones,
        # mutual exclusion, slew limiting, and an optional safety filter are
        # environment dynamics; storing the post-filter action would train Q
        # on a different action distribution than the actor optimizes.
        command_action = raw.copy()
        if raw[0] < self.cfg.min_throttle_deadzone:
            raw[0] = 0.0
        if raw[1] < self.cfg.min_brake_deadzone:
            raw[1] = 0.0
        if raw[0] > 0.0 and raw[1] > 0.0:
            if raw[0] >= raw[1]:
                raw[1] = 0.0
            else:
                raw[0] = 0.0

        if self.cfg.use_safety_shield:
            applied, route_meta, shield_active = self._safety_filter(raw)
        else:
            applied, route_meta, shield_active = self._policy_passthrough_filter(raw)
        applied[0] = float(np.clip(applied[0], 0.0, 1.0))
        applied[1] = float(np.clip(applied[1], 0.0, 1.0))
        applied[2] = float(np.clip(applied[2], -1.0, 1.0))
        if applied[0] > 0.0 and applied[1] > 0.0:
            if applied[0] >= applied[1]:
                applied[1] = 0.0
            else:
                applied[0] = 0.0

        stop_lines_before_tick = self._stop_line_snapshot()
        self.vehicle.apply_control(
            carla.VehicleControl(
                throttle=float(applied[0]),
                steer=float(applied[2]),
                brake=float(applied[1]),
                hand_brake=False,
                reverse=False,
                manual_gear_shift=False,
            )
        )
        self._tick()

        cur_loc = self.vehicle.get_location()
        if self.prev_loc is not None:
            self.distance_driven_m += distance2d(cur_loc, self.prev_loc)
        self.prev_loc = cur_loc

        self._step_red_light_events = self._stop_line_tracker.update(
            self._front_bumper_position(), stop_lines_before_tick)
        for event in self._step_red_light_events:
            event.update(step=int(self.episode_steps), distance_m=float(self.distance_driven_m))
        self.red_light_events.extend(self._step_red_light_events)
        rs, rp, rc, comp = self._compute_reward_components(applied)
        terminated = False
        truncated = False
        reason: Optional[str] = None
        if self.cfg.terminate_on_collision and len(self.collision_events) > 0:
            terminated = True
            reason = "collision"

        dL_after = float(comp.get("dL", 0.0))
        eps_mu_now = float(comp.get("eps_mu", self.cfg.eps_max))
        route_heading_err_now = abs(float(route_meta.get("route_heading_err", 0.0)))
        offroute_thresh = max(self.cfg.max_route_deviation_m, eps_mu_now + 0.75)
        strong_offroute = dL_after > (self.cfg.strong_offroute_factor * offroute_thresh)
        offroute_bad = strong_offroute or ((dL_after > offroute_thresh) and (route_heading_err_now > self.cfg.offroute_heading_gate_rad))
        if offroute_bad:
            self.offroute_steps += 1
        else:
            self.offroute_steps = max(0, self.offroute_steps - 1)
        if (not terminated) and self.cfg.terminate_on_offroad and self.offroute_steps >= self.cfg.offroute_grace_steps:
            terminated = True
            reason = "off_route"

        s_arc_now = float(comp.get("route_s", 0.0))
        remaining_s = max(0.0, self.route_total_len_m - s_arc_now)
        goal_euclid = 1e9
        if self.goal_transform is not None:
            goal_euclid = float(cur_loc.distance(self.goal_transform.location))
        route_completion_pct = self._route_completion_pct(s_arc_now)
        success = False
        completion_gate = route_completion_pct >= self.cfg.route_success_pct
        physical_goal_gate = (
            remaining_s <= self.cfg.goal_reach_remaining_s_m
            and goal_euclid <= self.cfg.goal_reach_dist_m
        )
        final_projection_gate = (
            route_completion_pct >= 99.5
            and goal_euclid <= 1.5 * self.cfg.goal_reach_dist_m
        )
        # Completion alone is not sufficient: the vehicle must physically reach
        # the final route endpoint, preventing premature success on looped routes.
        if (not terminated) and ((completion_gate and physical_goal_gate) or final_projection_gate):
            terminated = True
            reason = "goal"
            success = True

        blocked_wait = (
            (str(comp.get("tl_state", "None")) == "Red" and (3.6 * vec3_length(self.vehicle.get_velocity())) <= self.cfg.blocked_timeout_speed_kmh)
            or (
                safe_float(route_meta.get("safety_front_vehicle_dist", -1.0), -1.0) > 0.0
                and safe_float(route_meta.get("safety_front_vehicle_dist", -1.0), -1.0) < self.cfg.front_vehicle_soft_block_dist
                and (3.6 * vec3_length(self.vehicle.get_velocity())) <= self.cfg.blocked_timeout_speed_kmh
            )
        )
        if blocked_wait:
            self.blocked_steps_credit = min(self.blocked_steps_credit + 1, self.cfg.blocked_timeout_extension_steps)
        else:
            self.blocked_steps_credit = max(0, self.blocked_steps_credit - 1)

        tl_now = str(comp.get("tl_state", "None"))
        front_dist_now = safe_float(route_meta.get("safety_front_vehicle_dist", -1.0), -1.0)
        genuinely_open = (
            tl_now not in ("Red", "Yellow")
            and (front_dist_now < 0.0 or front_dist_now > self.cfg.front_vehicle_soft_block_dist)
            # CaST: the hazard-aware reference speed (item 23) says the road ahead is free; waiting for a
            # pedestrian, a crossing car, a stopped car, or a scripted light is not "stuck".
            and safe_float(comp.get("v_des", 5.0), 5.0) > 1.0
        )
        if (3.6 * vec3_length(self.vehicle.get_velocity())) < self.cfg.stuck_speed_kmh and genuinely_open:
            self._free_stuck_steps = getattr(self, "_free_stuck_steps", 0) + 1
        else:
            self._free_stuck_steps = 0

        if (not terminated) and self._free_stuck_steps >= self.cfg.stuck_terminate_steps:
            # A failure of the policy (the privileged reference speed says the road is free), so the episode
            # terminates with a penalty instead of being truncated: truncation bootstrapped the value of
            # standing still, and fine-tuning learned that stopping is safe and free.
            terminated = True
            reason = "stuck"
            self._free_stuck_steps = 0

        near_goal_extra = self.cfg.near_goal_timeout_extension_steps if remaining_s < self.cfg.near_goal_remaining_s_m else 0
        hard_timeout_steps = self.cfg.max_episode_steps + self.blocked_steps_credit + near_goal_extra
        if (not terminated) and (self.episode_steps >= hard_timeout_steps):
            truncated = True
            reason = "timeout"

        self.prev_route_s = s_arc_now
        if shield_active:
            self.safety_interventions += 1
        self.prev_action = applied.copy()
        self.prev_steer = float(applied[2])
        acc = self.vehicle.get_acceleration()
        self.prev_acc = np.array([acc.x, acc.y], dtype=np.float32)

        obs = self._get_obs()
        info = {
            "rs": float(rs),
            "rp": float(rp),
            "rc": float(rc),
            "term_reason": reason or "",
            "goal_reached": bool(reason == "goal"),
            "collision": bool(reason == "collision"),
            "off_road": bool(reason == "off_route"),
            "off_route": bool(reason == "off_route"),
            "timeout": bool(reason in ("timeout", "stuck")),
            "stuck": bool(reason == "stuck"),
            "success": bool(success),
            "steps": int(self.episode_steps),
            "goal_dist": float(remaining_s),
            "goal_euclid": float(goal_euclid),
            "route_completion_pct": float(route_completion_pct),
            "route_total_len_m": float(self.route_total_len_m),
            "stuck_steps": int(self.stuck_steps),
            "offroute_steps": int(self.offroute_steps),
            "blocked_steps_credit": int(self.blocked_steps_credit),
            "shield_active": int(bool(shield_active)),
            "safety_intervention": int(bool(shield_active)),
            "intervention_rate": float(self.safety_interventions / max(self.episode_steps, 1)),
            "desired_speed_kmh": float(route_meta.get("desired_speed_kmh", 0.0)),
            "route_idx": float(route_meta.get("route_idx", 0.0)),
            "route_target_idx": float(route_meta.get("route_target_idx", 0.0)),
            "route_cte": float(route_meta.get("route_cte", 0.0)),
            "route_heading_err": float(route_meta.get("route_heading_err", 0.0)),
            "route_remaining_s": float(route_meta.get("route_remaining_s", remaining_s)),
            "route_policy_blend": float(route_meta.get("policy_route_blend", 0.0)),
            "spawn_index": self.current_spawn_index,
            "goal_index": self.current_goal_index,
            "command_action": command_action,
            "applied_action": applied.copy(),
            "distance_driven_m": float(self.distance_driven_m),
        }
        info.update(comp)
        # ------------------------------------------------------------------
        # σ̄ proxy for env-time reward (Eq. 9: r_u = 1 − σ̄).
        # During rollout the critic ensemble is not called every step for
        # efficiency; instead we compute a lightweight proxy.
        # Proxy: σ̄ ≈ μ_A (fragile contexts → higher uncertainty; Sec. III-B).
        # The training loop overwrites this with the real ensemble-based σ̄
        # via recompute_agent_reward() so the stored replay reward is correct.
        # ------------------------------------------------------------------
        muA_val = float(np.clip(safe_float(info.get("muA", 0.5)), 0.0, 1.0))
        # Also incorporate nearest-entity distance for a richer proxy:
        # closer entities → higher uncertainty even in clear weather.
        near_norm = float(np.clip(safe_float(info.get("nearest_entity_dist", self.cfg.entity_max_dist))
                                  / max(self.cfg.entity_max_dist, 1.0), 0.0, 1.0))
        # sigma_proxy ↑ when μ_A ↑ (fog/dense/curved) or entities are very close
        sigma_proxy = float(np.clip(muA_val * 0.70 + (1.0 - near_norm) * 0.30, 0.0, 1.0))
        reward = build_reward_from_info(info, sigma_bar=sigma_proxy, cfg=self.cfg)
        info["sigma_proxy"] = float(sigma_proxy)
        info["env_reward_proxy"] = float(reward)

        if self.cfg.debug_mode and (self.episode_steps == 1 or self.episode_steps % self.cfg.debug_step_freq == 0 or terminated or truncated):
            vel = self.vehicle.get_velocity()
            speed_kmh = 3.6 * vec3_length(vel)
            print(
                f"[STEP] step={self.episode_steps:04d} speed={speed_kmh:6.2f}km/h "
                f"thr={applied[0]:.2f} brk={applied[1]:.2f} str={applied[2]:+.2f} safe={int(bool(shield_active))} "
                f"progress={route_completion_pct:.1f}% goal={remaining_s:.2f}m dL={dL_after:.2f} "
                f"ttc={safe_float(info.get('time_to_conflict', 999.0)):.2f}s tl={info.get('tl_state', 'None')} "
                f"credit={int(self.blocked_steps_credit)} reason={reason or ''}"
            )
        if terminated or truncated:
            self._begin_episode_teardown(reason or ("timeout" if truncated else "terminated"))    
        return obs, reward, terminated, truncated, info

    def close(self) -> None:
        if getattr(self, "_server_down", False):
            return      # the actors go with the server; every destroy() would wait the full time-out
        try:
            self._cleanup_episode_actors()
        except BaseException:
            pass
        try:
            self._restore_world_settings()
        except BaseException:
            pass




# ======================================================
# Shared helpers from the previous code base
# ======================================================

class UncertaintyCalibrator:
    def __init__(self, temperature: float = 1.0):
        self.center = 0.5
        self.scale = 0.25
        self.min_val = 0.0
        self.max_val = 1.0
        self.temperature = float(max(temperature, 1e-3))
        self.fitted = False

    def fit_from_values(self, values: np.ndarray, q_lo: float = 0.05, q_hi: float = 0.95) -> Dict[str, float]:
        arr = np.asarray(values, dtype=np.float32).reshape(-1)
        arr = arr[np.isfinite(arr)]
        if arr.size == 0:
            return {
                "fitted": 0.0,
                "center": float(self.center),
                "scale": float(self.scale),
                "min_val": float(self.min_val),
                "max_val": float(self.max_val),
            }
        q_lo = float(np.clip(q_lo, 0.0, 1.0))
        q_hi = float(np.clip(q_hi, q_lo + 1e-6, 1.0))
        lo = float(np.quantile(arr, q_lo))
        hi = float(np.quantile(arr, q_hi))
        if not np.isfinite(lo):
            lo = float(np.min(arr))
        if not np.isfinite(hi):
            hi = float(np.max(arr))
        if hi <= lo:
            hi = lo + 1e-3
        arr_mm = np.clip((arr - lo) / max(hi - lo, 1e-6), 0.0, 1.0)
        center = float(np.median(arr_mm))
        mad = float(np.median(np.abs(arr_mm - center)))
        scale = max(1.4826 * mad, 0.05)
        self.center = center
        self.scale = scale
        self.min_val = lo
        self.max_val = hi
        self.fitted = True
        return {
            "fitted": 1.0,
            "center": self.center,
            "scale": self.scale,
            "min_val": self.min_val,
            "max_val": self.max_val,
        }

    def transform_tensor(self, x: torch.Tensor) -> torch.Tensor:
        if self.fitted:
            x = (x - self.min_val) / max(self.max_val - self.min_val, 1e-6)
            x = x.clamp(0.0, 1.0)
        z = (x - self.center) / max(self.scale * self.temperature, 1e-6)
        return torch.sigmoid(z)

    def load_state_dict(self, state: Dict[str, object]) -> None:
        if not state:
            return
        self.center = float(state.get("center", self.center))
        self.scale = float(max(float(state.get("scale", self.scale)), 1e-6))
        self.min_val = float(state.get("min_val", self.min_val))
        self.max_val = float(max(float(state.get("max_val", self.max_val)), self.min_val + 1e-6))
        self.temperature = float(max(float(state.get("temperature", self.temperature)), 1e-6))
        self.fitted = bool(state.get("fitted", self.fitted))

    def state_dict(self) -> Dict[str, object]:
        return {
            "center": float(self.center),
            "scale": float(self.scale),
            "min_val": float(self.min_val),
            "max_val": float(self.max_val),
            "temperature": float(self.temperature),
            "fitted": bool(self.fitted),
        }



def build_reward_from_info(info: Dict[str, object], sigma_bar: float, cfg: Config = CFG) -> float:
    """Dense reward (deviation register, items 19 and 23):
    r_t = w_s (r_s - 1) + w_p r_p + w_c r_c - w_u u_bar - phi_lane, plus a terminal reward for a
    collision (-10), leaving the route (-5), or reaching the goal (+5)."""
    rs = safe_float(info.get("rs", 0.0))
    rp = safe_float(info.get("rp", 0.0))
    rc = safe_float(info.get("rc", 0.0))
    # Safety and uncertainty enter as penalties (r_s - 1 <= 0, -sigma_bar <= 0). In the original form
    # (r_s and 1 - sigma_bar) a car standing still earned about 80% of the reward of a car driving,
    # with no collision risk, so RL learned to stop. Now only progress is positive.
    r_dense = float(cfg.w_s * (rs - 1.0) + cfg.w_p * rp + cfg.w_c * rc
                    - cfg.w_u * float(np.clip(sigma_bar, 0.0, 1.0)))
    reason = str(info.get("term_reason", ""))
    if reason == "collision":
        r_dense += float(cfg.r_collision)
    elif reason == "off_route":
        r_dense += float(cfg.r_offroute)
    elif reason == "goal":
        r_dense += float(cfg.r_goal)
    elif reason == "stuck":
        r_dense += float(cfg.r_stuck)
    # Crossing events are pulses. Do not penalize occupancy beyond the stop line,
    # which could teach the agent to freeze after entering an intersection.
    r_dense += float(cfg.r_red_light) * len(info.get("red_light_events", []))
    # Eq. (4): the constrained objective subtracts the smooth penalty vector
    # lambda^T phi_t from the dense reward. phi_lane carries phi_L(d_L, mu_A)
    # together with the wrong-lane / opposite-lane components. Eq. (8) and its
    # weight normalisation are untouched.
    phi_lane = safe_float(info.get("phi_lane", 0.0))
    return float(r_dense - phi_lane)



def robust_reset(
    env: CarlaReliableTransferEnv,
    env_builder: Callable[[], CarlaReliableTransferEnv],
    npc_count: int,
    goal_index: int,
    walker_count: Optional[int] = None,
    max_tries: int = 5,
) -> Tuple[CarlaReliableTransferEnv, Dict[str, np.ndarray], Dict[str, object]]:
    last_err: Optional[Exception] = None
    cur_env = env
    sleep_secs = [2.0, 4.0, 8.0, 12.0, 20.0]
    for attempt in range(1, max_tries + 1):
        try:
            reset_options: Dict[str, object] = {"npc_count": int(npc_count), "goal_index": int(goal_index)}
            if walker_count is not None:
                reset_options["walker_count"] = int(walker_count)
            obs, info = cur_env.reset(options=reset_options)
            return cur_env, obs, info
        except BaseException as e:
            last_err = Exception(str(e))
            print(f"[WARN] reset attempt {attempt}/{max_tries} failed: {e}")
        try:
            cur_env.close()
        except BaseException:
            pass
        if attempt >= max_tries:
            break
        wait = sleep_secs[min(attempt - 1, len(sleep_secs) - 1)]
        print(f"[WARN] waiting {wait:.0f}s before rebuild (attempt {attempt+1}/{max_tries})")
        time.sleep(wait)
        try:
            cur_env = env_builder()
        except BaseException as e2:
            print(f"[WARN] env_builder() failed: {e2}")
            last_err = Exception(str(e2))
    raise RuntimeError(f"robust_reset failed after {max_tries} attempts. Last: {last_err}")



def _carla_probe_child_code() -> str:
    """Small standalone program: do not import this training module or initialize CUDA."""
    return r'''
import json, os, sys, time
try:
    import resource
    resource.setrlimit(resource.RLIMIT_CORE, (0, 0))
except (ImportError, OSError, ValueError):
    pass
deadline = time.monotonic() + float(sys.argv[4])
try:
    import carla
    client = carla.Client(sys.argv[1], int(sys.argv[2]))
    def rpc_budget():
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            raise TimeoutError("probe RPC deadline expired")
        client.set_timeout(min(30.0 if sys.argv[3] == "map" else 5.0, remaining))
    rpc_budget()
    version = str(client.get_server_version())
    rpc_budget()
    world = client.get_world()
    result = {"server_version": version, "world_id": int(world.id)}
    if sys.argv[3] == "map":
        rpc_budget()
        result["map"] = str(world.get_map().name)
    print("_CAST_RPC_RESULT_=" + json.dumps(result), flush=True)
except Exception as exc:
    print("CARLA probe: " + type(exc).__name__ + ": " + str(exc), file=sys.stderr, flush=True)
    sys.exit(2)
'''


def _carla_probe_in_child(host: str, port: int, timeout_s: float = 15.0,
                           include_map: bool = False) -> Dict[str, object]:
    """No CARLA Client may live in the supervisor across a server restart.

    Even get_world() can start C++ background threads whose exceptions bypass
    Python handlers. Require both a valid reply AND a clean child exit. The wall
    deadline also covers import and client teardown; never accept a reply from a
    child that subsequently aborts or hangs. Child core dumps are disabled.
    """
    import subprocess
    timeout = float(timeout_s)
    if not math.isfinite(timeout) or timeout <= 0.0:
        raise ValueError("CARLA probe timeout must be finite and positive")
    env = os.environ.copy()
    env["PYTHONPATH"] = os.pathsep.join([str(p) for p in sys.path if p] +
                                      ([env["PYTHONPATH"]] if env.get("PYTHONPATH") else []))
    cmd = [sys.executable, "-c", _carla_probe_child_code(), str(host), str(int(port)),
           "map" if include_map else "ready", str(timeout)]
    child = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                             text=True, errors="replace", env=env, start_new_session=True)
    try:
        stdout, stderr = child.communicate(timeout=timeout)
    except subprocess.TimeoutExpired:
        _stop_carla_launch(child)
        stdout, stderr = child.communicate()
        detail = (stderr.strip() or stdout.strip())[-2000:]
        raise TimeoutError(f"isolated RPC probe exceeded {timeout:g}s; {detail}") from None
    except BaseException:
        # KeyboardInterrupt must propagate, after reaping only this owned child.
        _stop_carla_launch(child)
        child.communicate()
        raise
    if child.returncode != 0:
        detail = (stderr.strip() or stdout.strip())[-2000:]
        status = f"exit code {child.returncode}"
        if child.returncode < 0:
            import signal
            try:
                status += f" ({signal.Signals(-child.returncode).name})"
            except ValueError:
                pass
        raise RuntimeError(f"isolated RPC probe {status}; {detail}")
    prefix = "_CAST_RPC_RESULT_="
    for line in reversed(stdout.splitlines()):
        if line.startswith(prefix):
            try:
                result = json.loads(line[len(prefix):])
                if (not isinstance(result, dict) or not result.get("server_version")
                        or not isinstance(result.get("world_id"), int)
                        or (include_map and not isinstance(result.get("map"), str))
                        or (include_map and not result.get("map"))):
                    raise ValueError("missing readiness/map fields")
                return result
            except (ValueError, TypeError) as exc:
                raise RuntimeError(f"invalid isolated RPC probe reply: {exc}") from exc
    raise RuntimeError("isolated RPC probe exited without a readiness reply; " +
                       (stderr.strip() or stdout.strip())[-2000:])


def wait_for_carla_ready(host: str, port: int, timeout_s: float = 60.0,
                         poll_interval: float = 2.0, process: Optional[object] = None,
                         probe_log_path: Optional[str] = None) -> None:
    """Bounded readiness polling in children, including native-crash containment.

    A live process or open socket is insufficient. Check version and world RPCs,
    reap the probe before continuing, and notice a dead launcher. The optional
    process must be the launcher owned by this caller. Ctrl+C is not retried.
    """
    timeout = float(timeout_s)
    if not math.isfinite(timeout) or timeout <= 0.0:
        raise ValueError("CARLA readiness timeout must be finite and positive")
    interval = float(poll_interval)
    if not math.isfinite(interval) or interval < 0.0:
        raise ValueError("CARLA readiness poll interval must be finite and nonnegative")
    deadline = time.monotonic() + timeout
    next_status = time.monotonic() + 30.0
    last_error = None
    probe_attempt = 0
    while time.monotonic() < deadline:
        if process is not None and process.poll() is not None:
            raise RuntimeError(f"CARLA launcher pid={process.pid} exited with code {process.returncode}")
        remaining = deadline - time.monotonic()
        if remaining <= 0.0:
            break
        probe_attempt += 1
        try:
            _carla_probe_in_child(host, int(port), timeout_s=min(15.0, remaining))
        except Exception as exc:
            last_error = exc
            if probe_log_path:
                try:
                    with open(probe_log_path, "a") as f:
                        f.write(f"probe={probe_attempt} endpoint={host}:{port}: {exc}\n")
                except OSError as log_exc:
                    print(f"[CARLA] could not write probe log: {log_exc}", flush=True)
        else:
            if process is not None and process.poll() is not None:
                raise RuntimeError(f"CARLA launcher pid={process.pid} exited with code {process.returncode}")
            return
        if process is not None and process.poll() is not None:
            raise RuntimeError(f"CARLA launcher pid={process.pid} exited with code {process.returncode}")
        if probe_attempt == 1 or time.monotonic() >= next_status:
            print(f"[CARLA] waiting for isolated RPC at {host}:{port}; "
                  f"{max(0.0, deadline - time.monotonic()):.0f}s remaining; last error: {last_error}", flush=True)
            next_status = time.monotonic() + 30.0
        remaining = deadline - time.monotonic()
        if remaining > 0.0:
            time.sleep(min(max(interval, 0.01), remaining))
    if process is not None and process.poll() is not None:
        raise RuntimeError(f"CARLA launcher pid={process.pid} exited with code {process.returncode}")
    raise TimeoutError(f"CARLA at {host}:{port} did not answer RPC within {timeout:.1f}s; "
                       f"last error: {last_error}")




# ======================================================
# CaST environment layer (Secs. 3.1, 3.4, 4.1, 4.3)
# ======================================================
KIND_VEHICLE, KIND_WALKER, KIND_SIGNAL, KIND_STATIC = 0, 1, 2, 3
KIND_NAMES = ("vehicle", "walker", "signal", "static")
EGO_RADIUS_M = 2.2                    # ego half-length used for longitudinal gaps
EGO_HALF_WIDTH_M = 1.0
KIND_RADIUS_M = (2.3, 0.3, 0.0, 0.4)  # half-length per kind: vehicle, walker, signal, static
KIND_HALF_WIDTH_M = (1.0, 0.3, 0.0, 0.4)
PROP_BLUEPRINTS = (
    "static.prop.constructioncone",
    "static.prop.trafficcone01",
    "static.prop.trafficcone02",
    "static.prop.barrel",
)


# ======================================================
# Perception front end: six surround cameras and a camera-based BEV detector (Secs. 3.1, 4.1)
# ======================================================
# Camera layout (ego frame: x forward, y right, z up); yaw in degrees, positive to the right.
CAMERA_LAYOUT: Tuple[Tuple[str, float, float, float, float], ...] = (
    ("front", 1.0, 0.0, 1.8, 0.0),
    ("front_left", 0.5, -0.4, 1.8, -60.0),
    ("front_right", 0.5, 0.4, 1.8, 60.0),
    ("back", -1.0, 0.0, 1.8, 180.0),
    ("back_left", -0.5, -0.4, 1.8, -120.0),
    ("back_right", -0.5, 0.4, 1.8, 120.0),
)
DET_CLASSES = (KIND_VEHICLE, KIND_WALKER, KIND_STATIC)     # heat-map channels
TL_STATES = ("none", "Red", "Yellow", "Green")


class CameraRig:
    """Six synchronised RGB cameras attached to the ego vehicle. The callbacks only copy pixels
    (no RPC in the sensor thread)."""

    def __init__(self, world: carla.World, bp_lib: carla.BlueprintLibrary, vehicle: carla.Actor, cfg: Config,
                 depth: bool = False):
        import threading
        self.cfg = cfg
        self.lock = threading.Lock()
        self.latest: Dict[int, Tuple[int, np.ndarray]] = {}
        self.actors: List[carla.Actor] = []
        self.depth = bool(depth)
        self.n_cam = len(CAMERA_LAYOUT)
        kinds = [("sensor.camera.rgb", False)] + ([("sensor.camera.depth", True)] if self.depth else [])
        for k, (bp_name, is_depth) in enumerate(kinds):
            bp = bp_lib.find(bp_name)
            bp.set_attribute("image_size_x", str(cfg.cam_width))
            bp.set_attribute("image_size_y", str(cfg.cam_height))
            bp.set_attribute("fov", str(cfg.cam_fov))
            for i, (_, x, y, z, yaw) in enumerate(CAMERA_LAYOUT):
                tf = carla.Transform(carla.Location(x=x, y=y, z=z), carla.Rotation(yaw=yaw))
                cam = world.spawn_actor(bp, tf, attach_to=vehicle)
                cam.listen(self._callback(k * self.n_cam + i, is_depth))
                self.actors.append(cam)

    def _callback(self, i: int, is_depth: bool = False):
        H, W = int(self.cfg.cam_height), int(self.cfg.cam_width)

        def cb(image) -> None:
            try:
                raw = np.frombuffer(image.raw_data, dtype=np.uint8).reshape(H, W, 4)
                arr = decode_depth(raw) if is_depth else raw[:, :, 2::-1].copy()
                with self.lock:
                    self.latest[i] = (int(image.frame), arr)
            except BaseException:
                pass
        return cb

    def get(self, frame: int, timeout_s: float = 2.0) -> Optional[Tuple[np.ndarray, Optional[np.ndarray]]]:
        """RGB images (and depth images when the rig has depth cameras) of the given simulator frame, or
        the newest ones after the timeout."""
        t0 = time.time()
        n = self.n_cam * (2 if self.depth else 1)
        while True:
            with self.lock:
                ok = len(self.latest) == n and all(self.latest[i][0] >= frame for i in range(n))
                if ok or time.time() - t0 > timeout_s:
                    if len(self.latest) < n:
                        return None
                    rgb = np.stack([self.latest[i][1] for i in range(self.n_cam)], 0)
                    dep = np.stack([self.latest[self.n_cam + i][1] for i in range(self.n_cam)], 0) if self.depth else None
                    return rgb, dep
            time.sleep(0.002)


def camera_intrinsics(cfg: Config) -> float:
    return float(cfg.cam_width) / (2.0 * math.tan(math.radians(float(cfg.cam_fov)) / 2.0))


def project_points(cfg: Config, pts: np.ndarray, cam: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Project ego-frame points (n, 3) into camera `cam`: pixel u, v and a validity mask."""
    _, cx, cy, cz, yaw = CAMERA_LAYOUT[cam]
    f = camera_intrinsics(cfg)
    ps = math.radians(yaw)
    dx, dy, dz = pts[:, 0] - cx, pts[:, 1] - cy, pts[:, 2] - cz
    xc = math.cos(ps) * dx + math.sin(ps) * dy
    yc = -math.sin(ps) * dx + math.cos(ps) * dy
    valid = xc > 0.3
    xs = np.where(valid, xc, 1.0)
    u = cfg.cam_width / 2.0 + f * yc / xs
    v = cfg.cam_height / 2.0 - f * dz / xs
    valid &= (u >= 0) & (u < cfg.cam_width) & (v >= 0) & (v < cfg.cam_height)
    return u, v, valid


def _conv(cin: int, cout: int, stride: int = 1) -> nn.Sequential:
    return nn.Sequential(nn.Conv2d(cin, cout, 3, stride, 1, bias=False), nn.BatchNorm2d(cout), nn.ReLU(inplace=True))


class BEVDetector(nn.Module):
    """Camera-based BEV detector: a shared image backbone, geometric lifting of image features to a
    bird's-eye-view grid at two heights (inverse perspective sampling), temporal fusion with the
    previous frame, and centre-point heads for class heat maps, sub-cell offsets, relative velocity,
    and the log-variance of the position (the detection variance sigma_i^2 of Sec. 3.1). A head on
    the front camera predicts the state and distance of the traffic light that controls the ego
    lane. b(s) is the pooled BEV feature."""

    HEIGHTS = (0.2, 1.0)

    def __init__(self, cfg: Config):
        super().__init__()
        self.cfg = cfg
        C = 64
        self.backbone = nn.Sequential(_conv(3, 32, 2), _conv(32, 48, 2), _conv(48, C, 2), _conv(C, C, 1))
        self.feat_c = C
        self.nx = int(round((cfg.bev_x_max - cfg.bev_x_min) / cfg.bev_res))
        self.ny = int(round((cfg.bev_y_max - cfg.bev_y_min) / cfg.bev_res))
        grids, masks = self._geometry()
        self.register_buffer("grids", grids)    # (heights, cams, nx, ny, 2)
        self.register_buffer("masks", masks)    # (heights, cams, nx, ny)
        lifted = C * len(self.HEIGHTS)
        self.bev = nn.Sequential(_conv(2 * lifted, 128), _conv(128, 128), _conv(128, 128))
        self.hm = nn.Conv2d(128, len(DET_CLASSES), 1)
        nn.init.constant_(self.hm.bias, -2.19)
        self.off = nn.Conv2d(128, 2, 1)
        self.vel = nn.Conv2d(128, 2, 1)
        self.lv = nn.Conv2d(128, 1, 1)
        self.scene = nn.Conv2d(128, cfg.scene_dim, 1)
        self.tl = nn.Sequential(nn.Linear(2 * C, 64), nn.ReLU(), nn.Linear(64, len(TL_STATES) + 2))
        self.register_buffer("mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 1, 3, 1, 1))
        self.register_buffer("std", torch.tensor([0.229, 0.224, 0.225]).view(1, 1, 3, 1, 1))

    def _geometry(self) -> Tuple[torch.Tensor, torch.Tensor]:
        cfg = self.cfg
        xs = cfg.bev_x_min + (np.arange(self.nx) + 0.5) * cfg.bev_res
        ys = cfg.bev_y_min + (np.arange(self.ny) + 0.5) * cfg.bev_res
        X, Y = np.meshgrid(xs, ys, indexing="ij")
        grids, masks = [], []
        for z in self.HEIGHTS:
            pts = np.stack([X.ravel(), Y.ravel(), np.full(X.size, z)], 1)
            g_h, m_h = [], []
            for c in range(len(CAMERA_LAYOUT)):
                u, v, ok = project_points(cfg, pts, c)
                gu = (u + 0.5) / cfg.cam_width * 2.0 - 1.0
                gv = (v + 0.5) / cfg.cam_height * 2.0 - 1.0
                g_h.append(np.stack([gu, gv], -1).reshape(self.nx, self.ny, 2))
                m_h.append(ok.reshape(self.nx, self.ny).astype(np.float32))
            grids.append(np.stack(g_h, 0))
            masks.append(np.stack(m_h, 0))
        return torch.from_numpy(np.stack(grids, 0)).float(), torch.from_numpy(np.stack(masks, 0)).float()

    def preprocess(self, imgs: torch.Tensor, degradation: float = 0.0, drop: bool = False) -> torch.Tensor:
        """uint8 images (B, 6, H, W, 3) to normalised float; sensor degradation of the combined suite:
        Gaussian camera noise, horizontal motion blur, and frame dropout."""
        x = imgs.float().permute(0, 1, 4, 2, 3) / 255.0
        if drop:
            return torch.zeros_like(x) - self.mean / self.std
        if degradation > 0.0:
            k = 1 + 2 * int(round(3.0 * degradation))
            if k > 1:
                B, N, C, H, W = x.shape
                w = torch.ones(C, 1, 1, k, device=x.device) / k
                x = F.conv2d(x.reshape(B * N, C, H, W), w, padding=(0, k // 2), groups=C).reshape(B, N, C, H, W)
            x = (x + torch.randn_like(x) * 0.08 * degradation).clamp(0.0, 1.0)
        return (x - self.mean) / self.std

    def lift(self, feats: torch.Tensor) -> torch.Tensor:
        B, N, C, h, w = feats.shape
        out = []
        flat = feats.reshape(B * N, C, h, w)
        for hi in range(len(self.HEIGHTS)):
            g = self.grids[hi].unsqueeze(0).expand(B, -1, -1, -1, -1).reshape(B * N, self.nx, self.ny, 2)
            s = F.grid_sample(flat, g, mode="bilinear", padding_mode="zeros", align_corners=False)
            s = s.reshape(B, N, C, self.nx, self.ny)
            m = self.masks[hi].unsqueeze(0).unsqueeze(2)
            out.append((s * m).sum(1) / m.sum(1).clamp(min=1.0))
        return torch.cat(out, 1)

    def forward(self, x: torch.Tensor, prev_lifted: Optional[torch.Tensor]) -> Dict[str, torch.Tensor]:
        B, N = x.shape[:2]
        feats = self.backbone(x.reshape(B * N, *x.shape[2:]))
        feats = feats.reshape(B, N, *feats.shape[1:])
        lifted = self.lift(feats)
        prev = lifted.detach() if prev_lifted is None else prev_lifted
        f = self.bev(torch.cat([lifted, prev], 1))
        front = feats[:, 0]
        tl = self.tl(torch.cat([front.mean((2, 3)), front.amax((2, 3))], 1))
        return {"hm": self.hm(f), "off": self.off(f), "vel": self.vel(f), "lv": self.lv(f).clamp(-6.0, 4.0),
                "scene": self.scene(f).mean((2, 3)), "tl_logits": tl[:, :len(TL_STATES)],
                "tl_dist": tl[:, len(TL_STATES)], "tl_lv": tl[:, len(TL_STATES) + 1].clamp(-6.0, 4.0),
                "lifted": lifted.detach()}

    @torch.no_grad()
    def decode(self, out: Dict[str, torch.Tensor], thr: float = 0.35, k: int = 16) -> List[Dict[str, float]]:
        """Centre-point decoding for batch element 0: peaks of the class heat maps after 3x3 NMS."""
        cfg = self.cfg
        hm = torch.sigmoid(out["hm"][0])
        peak = (hm == F.max_pool2d(hm.unsqueeze(0), 3, 1, 1)[0]) & (hm > thr)
        score = (hm * peak.float()).reshape(-1)
        vals, idx = torch.topk(score, min(k, score.numel()))
        dets = []
        nxy = self.nx * self.ny
        for v_, i in zip(vals.tolist(), idx.tolist()):
            if v_ <= 0.0:
                break
            c, rem = divmod(i, nxy)
            ix, iy = divmod(rem, self.ny)
            ox, oy = out["off"][0, :, ix, iy].tolist()
            x = cfg.bev_x_min + (ix + 0.5 + ox) * cfg.bev_res
            y = cfg.bev_y_min + (iy + 0.5 + oy) * cfg.bev_res
            vx, vy = out["vel"][0, :, ix, iy].tolist()
            var = float(math.exp(float(out["lv"][0, 0, ix, iy])))
            dets.append({"kind": DET_CLASSES[c], "x": x, "y": y, "vx": vx, "vy": vy, "var": var, "score": v_})
        return dets

    def cell(self, x: float, y: float) -> Optional[Tuple[int, int, float, float]]:
        cfg = self.cfg
        fx = (x - cfg.bev_x_min) / cfg.bev_res
        fy = (y - cfg.bev_y_min) / cfg.bev_res
        ix, iy = int(math.floor(fx)), int(math.floor(fy))
        if 0 <= ix < self.nx and 0 <= iy < self.ny:
            return ix, iy, fx - ix - 0.5, fy - iy - 0.5
        return None


def detector_targets(det: BEVDetector, labels: List[np.ndarray], tl: np.ndarray, device: torch.device) -> Dict[str, torch.Tensor]:
    """CenterNet-style targets from privileged labels (n, 5): kind, x, y, vx, vy in the ego frame."""
    B = len(labels)
    nc = len(DET_CLASSES)
    hm = np.zeros((B, nc, det.nx, det.ny), np.float32)
    reg = np.zeros((B, 4, det.nx, det.ny), np.float32)
    msk = np.zeros((B, det.nx, det.ny), np.float32)
    radius = {KIND_VEHICLE: 2.0, KIND_WALKER: 1.0, KIND_STATIC: 1.0}
    gx = np.arange(det.nx)[:, None]
    gy = np.arange(det.ny)[None, :]
    for b, lab in enumerate(labels):
        for kind, x, y, vx, vy in lab:
            kind = int(kind)
            if kind not in DET_CLASSES:
                continue
            c = det.cell(float(x), float(y))
            if c is None:
                continue
            ix, iy, ox, oy = c
            ci = DET_CLASSES.index(kind)
            r = radius[kind]
            g = np.exp(-((gx - ix) ** 2 + (gy - iy) ** 2) / (2.0 * r * r))
            hm[b, ci] = np.maximum(hm[b, ci], g)
            reg[b, :, ix, iy] = (ox, oy, vx, vy)
            msk[b, ix, iy] = 1.0
    return {"hm": torch.from_numpy(hm).to(device), "reg": torch.from_numpy(reg).to(device),
            "msk": torch.from_numpy(msk).to(device), "tl": torch.from_numpy(tl.astype(np.float32)).to(device)}


def detector_loss(out: Dict[str, torch.Tensor], tg: Dict[str, torch.Tensor]) -> Tuple[torch.Tensor, Dict[str, float]]:
    p = torch.sigmoid(out["hm"]).clamp(1e-4, 1 - 1e-4)
    gt = tg["hm"]
    pos = gt.eq(1.0).float()
    neg = 1.0 - pos
    pos_l = (torch.log(p) * (1 - p) ** 2 * pos).sum()
    neg_l = (torch.log(1 - p) * p ** 2 * (1 - gt) ** 4 * neg).sum()
    npos = pos.sum().clamp(min=1.0)
    l_hm = -(pos_l + neg_l) / npos
    m = tg["msk"].unsqueeze(1)
    nm = m.sum().clamp(min=1.0)
    l_off = ((out["off"] - tg["reg"][:, 0:2]).abs() * m).sum() / nm
    l_vel = ((out["vel"] - tg["reg"][:, 2:4]).abs() * m).sum() / nm
    err2 = ((out["off"].detach() - tg["reg"][:, 0:2]) ** 2).sum(1, keepdim=True)
    lv = out["lv"]
    l_var = ((err2 / (2.0 * lv.exp()) + 0.5 * lv) * m).sum() / nm
    st = tg["tl"][:, 0].long()
    l_tl = F.cross_entropy(out["tl_logits"], st)
    has = (st > 0).float()
    dt = tg["tl"][:, 1]
    l_tld = (((out["tl_dist"] - dt).abs()) * has).sum() / has.sum().clamp(min=1.0)
    l_tlv = ((((out["tl_dist"].detach() - dt) ** 2) / (2.0 * out["tl_lv"].exp()) + 0.5 * out["tl_lv"]) * has).sum() / has.sum().clamp(min=1.0)
    loss = l_hm + l_off + 0.5 * l_vel + 0.1 * l_var + l_tl + 0.5 * l_tld + 0.1 * l_tlv
    return loss, {"hm": float(l_hm.detach()), "off": float(l_off.detach()), "vel": float(l_vel.detach()), "tl": float(l_tl.detach())}


def match_detections(dets: List[Dict[str, float]], truth: List[Tuple[int, int, float, float]]) -> List[int]:
    """Greedy association of detections with ground-truth actors (id per detection, -1 if none).
    Used only to label interventions and roles, never as policy input."""
    thr = {KIND_VEHICLE: 3.0, KIND_WALKER: 1.5, KIND_STATIC: 1.5}
    pairs = []
    for i, d in enumerate(dets):
        for j, (eid, kind, x, y) in enumerate(truth):
            if kind != d["kind"]:
                continue
            dist = math.hypot(d["x"] - x, d["y"] - y)
            if dist <= thr[kind]:
                pairs.append((dist, i, j))
    pairs.sort()
    out = [-1] * len(dets)
    used = set()
    for _, i, j in pairs:
        if out[i] < 0 and j not in used:
            out[i] = truth[j][0]
            used.add(j)
    return out


# ======================================================
# RGB-D perception front end (--perception rgbd)
# ======================================================
# The camera BEV detector above lifts image features onto two fixed heights without depth: with
# 8-pixel feature cells, cars 20 m, 30 m and 40 m ahead fall into the same feature row, and it reached
# 35% vehicle recall. The RGB-D front end separates the two problems. A small U-Net labels every
# pixel of the six RGB images (vehicle, pedestrian, static prop, the light state of the traffic light
# that controls the ego lane, other traffic lights); the depth camera at the same pose places each
# labelled pixel in 3D; labelled points are clustered in the ground plane into object tokens; a
# short-term tracker gives velocities and bridges single missed frames. Training labels are exact and
# visible-only by construction: a pixel is labelled with an object only if its depth point lies inside
# that object's 3D box. Fog, rain and night still degrade the RGB input, so suite A remains a shift.
SEG_CLASSES = ("background", "vehicle", "walker", "static", "tl_red", "tl_yellow", "tl_green", "tl_other")
KIND_OF_SEG = {1: KIND_VEHICLE, 2: KIND_WALKER, 3: KIND_STATIC}
SEG_OF_KIND = {KIND_VEHICLE: 1, KIND_WALKER: 2, KIND_STATIC: 3}
TL_SEG_OF_STATE = {"Red": 4, "Yellow": 5, "Green": 6}
TL_CODE_OF_SEG = {4: 1.0, 5: 0.5, 6: 0.0}           # token code of the light state (red 1, yellow 0.5, green 0)
SEG_DEFAULT_WEIGHTS = (1.0, 2.0, 6.0, 6.0, 10.0, 10.0, 10.0, 4.0)
# Visible points lie on the near faces of an object; its centre is about this far behind them.
RGBD_CENTER_SHIFT = {KIND_VEHICLE: 1.2, KIND_WALKER: 0.15, KIND_STATIC: 0.15}
RGBD_MIN_POINTS = {KIND_VEHICLE: 6, KIND_WALKER: 4, KIND_STATIC: 3}
RGBD_MIN_CONF = 0.6                                  # mean class probability of a kept cluster
RGBD_MIN_PURITY = 0.25                               # share of all depth points in the cluster's cells
# Points of one object are linked when they are at most this many cluster cells apart (0.4 m cells), and
# only points up to this height are used (a car roof seen at a grazing angle gives points metres apart).
RGBD_LINK_CELLS = {KIND_VEHICLE: 2, KIND_WALKER: 1, KIND_STATIC: 1}
RGBD_MAX_Z = {KIND_VEHICLE: 1.3, KIND_WALKER: 2.2, KIND_STATIC: 1.5}
EGO_BOX_HEIGHT = 2.2                                 # ego exclusion box: real half-extents + 0.15 m, this high


def decode_depth(raw: np.ndarray) -> np.ndarray:
    """CARLA depth image (H, W, 4 BGRA uint8) to metres along the camera axis."""
    a = raw.astype(np.float32)
    return (a[..., 2] + a[..., 1] * 256.0 + a[..., 0] * 65536.0) * (1000.0 / 16777215.0)


_RAY_CACHE: Dict[Tuple[int, int, float, int], Tuple[np.ndarray, np.ndarray]] = {}


def camera_rays(cfg: Config, stride: int) -> Tuple[np.ndarray, np.ndarray]:
    """Per camera: ego-frame direction of every `stride`-th pixel, scaled so that the camera-axis
    component is 1 (CARLA depth is the distance along the camera axis), and the camera offset."""
    key = (int(cfg.cam_width), int(cfg.cam_height), float(cfg.cam_fov), int(stride))
    if key in _RAY_CACHE:
        return _RAY_CACHE[key]
    f = camera_intrinsics(cfg)
    W, H = int(cfg.cam_width), int(cfg.cam_height)
    a = (np.arange(0, W, stride) + 0.5 - W / 2.0) / f
    b = (H / 2.0 - (np.arange(0, H, stride) + 0.5)) / f
    A, B = np.meshgrid(a, b)                     # (h, w): row = image row
    dirs, offs = [], []
    for _, cx, cy, cz, yaw in CAMERA_LAYOUT:
        ps = math.radians(yaw)
        dx = math.cos(ps) - math.sin(ps) * A
        dy = math.sin(ps) + math.cos(ps) * A
        dirs.append(np.stack([dx, dy, B], -1).reshape(-1, 3))
        offs.append([cx, cy, cz])
    out = (np.stack(dirs, 0).astype(np.float32), np.asarray(offs, np.float32))
    _RAY_CACHE[key] = out
    return out


def backproject(depth: np.ndarray, cfg: Config, stride: int) -> Tuple[np.ndarray, np.ndarray]:
    """Depth images (N, H, W) -> ego-frame points (N, h*w, 3) (x forward, y right, z up from the ego
    origin) and the depth of each point (N, h*w)."""
    dirs, offs = camera_rays(cfg, stride)
    d = np.asarray(depth[:, ::stride, ::stride], np.float32).reshape(depth.shape[0], -1)
    return offs[:, None, :] + d[:, :, None] * dirs, d


def ego_box_mask(pts: np.ndarray, cfg: Optional[Config] = None) -> np.ndarray:
    hl = float(getattr(cfg, "ego_half_length", 2.45)) + 0.15
    hw = float(getattr(cfg, "ego_half_width", 1.05)) + 0.15
    return (np.abs(pts[..., 0]) < hl) & (np.abs(pts[..., 1]) < hw) & (pts[..., 2] < EGO_BOX_HEIGHT)


def _oriented_extent(xy: np.ndarray) -> Tuple[np.ndarray, np.ndarray, float, float]:
    """Centre, major axis, and the lengths along the major and minor axes of a ground-plane point set."""
    c = xy.mean(0)
    X = xy - c
    if xy.shape[0] < 3:
        return c, np.array([1.0, 0.0]), float(np.ptp(X[:, 0]) if len(X) else 0.0), float(np.ptp(X[:, 1]) if len(X) else 0.0)
    w, v = np.linalg.eigh(X.T @ X / X.shape[0])
    major, minor = v[:, 1], v[:, 0]
    return c, major, float(np.ptp(X @ major)), float(np.ptp(X @ minor))


def _min_gap(a: np.ndarray, b: np.ndarray) -> float:
    if a.shape[0] > 150:
        a = a[np.linspace(0, a.shape[0] - 1, 150).astype(int)]
    if b.shape[0] > 150:
        b = b[np.linspace(0, b.shape[0] - 1, 150).astype(int)]
    return float(np.sqrt(((a[:, None, :] - b[None, :, :]) ** 2).sum(-1)).min())


def merge_vehicle_parts(X: np.ndarray, groups: List[np.ndarray]) -> List[np.ndarray]:
    """Join clusters of one vehicle that were split (rear and side faces seen separately, long vans and
    buses): two clusters merge when they are less than 1.2 m apart and together still fit a vehicle
    footprint (at most 6.5 m by 2.8 m). Cars in the next lane (about 5 m wide together) or queued
    bumper to bumper (more than 9 m long together) stay separate."""
    groups = list(groups)
    changed = True
    while changed:
        changed = False
        for i in range(len(groups)):
            for j in range(i + 1, len(groups)):
                gi, gj = groups[i], groups[j]
                if np.hypot(*(X[gi, :2].mean(0) - X[gj, :2].mean(0))) > 7.0:
                    continue
                u = np.concatenate([gi, gj])
                _, _, L, Wd = _oriented_extent(X[u, :2])
                if L <= 6.5 and Wd <= 2.8 and _min_gap(X[gi, :2], X[gj, :2]) < 1.2:
                    groups[i] = u
                    del groups[j]
                    changed = True
                    break
            if changed:
                break
    return groups


def label_points_with_boxes(pts: np.ndarray, valid: np.ndarray, boxes: Sequence[Tuple]) -> np.ndarray:
    """Segmentation label of each point (M, 3): the class of the oriented 3D box that contains it.
    boxes: (seg_class, cx, cy, cz, yaw_rel_rad, ext_x, ext_y, ext_z) in the ego frame. Later boxes win,
    so traffic-light boxes (listed last) are not overwritten by the pole behind them."""
    lab = np.zeros(pts.shape[0], np.uint8)
    idx_all = np.nonzero(valid)[0]
    if idx_all.size == 0:
        return lab
    P = pts[idx_all]
    for cls, cx, cy, cz, yaw, ex, ey, ez in boxes:
        m = 0.1 if cls >= 4 else 0.15
        dx, dy = P[:, 0] - cx, P[:, 1] - cy
        near = dx * dx + dy * dy <= (ex + ey + 2.0 * m) ** 2
        if not near.any():
            continue
        c, s = math.cos(yaw), math.sin(yaw)
        sel = np.nonzero(near)[0]
        lx = dx[sel] * c + dy[sel] * s
        ly = -dx[sel] * s + dy[sel] * c
        lz = P[sel, 2] - cz
        inside = (np.abs(lx) <= ex + m) & (np.abs(ly) <= ey + m) & (np.abs(lz) <= ez + m)
        lab[idx_all[sel[inside]]] = int(cls)
    return lab


def cluster_xy(xy: np.ndarray, res: float, min_pts: int, link: int = 1) -> List[np.ndarray]:
    """Connected components of the occupied ground-plane cells (cells at most `link` apart are joined);
    returns the point indices of every component with at least `min_pts` points."""
    if xy.shape[0] == 0:
        return []
    cells = np.floor(xy / res).astype(np.int64)
    keys, inv = np.unique(cells, axis=0, return_inverse=True)
    inv = np.asarray(inv).reshape(-1)
    n = keys.shape[0]
    parent = list(range(n))
    index = {(int(k[0]), int(k[1])): i for i, k in enumerate(keys)}

    def find(a: int) -> int:
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a

    for i in range(n):
        kx, ky = int(keys[i, 0]), int(keys[i, 1])
        for ddx in range(-link, link + 1):
            for ddy in range(-link, link + 1):
                j = index.get((kx + ddx, ky + ddy))
                if j is not None:
                    ri, rj = find(i), find(j)
                    if ri != rj:
                        parent[ri] = rj
    roots = np.array([find(i) for i in range(n)])
    lab = roots[inv]
    groups = []
    for r in np.unique(roots):
        g = np.nonzero(lab == r)[0]
        if g.size >= min_pts:
            groups.append(g)
    return groups


def rgbd_detect(probs: np.ndarray, depth: np.ndarray, cfg: Config, lights_out: Optional[list] = None
                ) -> Tuple[List[Dict[str, float]], Optional[Tuple[float, float, float]]]:
    """Object detections (ego frame, no velocity yet) and the traffic-light token from the per-pixel class
    probabilities (N, C, H, W) and depth images (N, H, W) of one frame. Objects use every
    cfg.rgbd_stride-th pixel; the traffic light uses the full resolution of the three front cameras
    (a light head 30 m away is only about 3 by 6 pixels)."""
    N, C, H, W = probs.shape
    s = int(cfg.rgbd_stride)
    pts, d = backproject(depth, cfg, s)
    P = probs[:, :, ::s, ::s].reshape(N, C, -1)
    cls = P.argmax(1)
    conf = P.max(1)
    valid = (d > 0.5) & (d < float(cfg.rgbd_max_depth)) & ~ego_box_mask(pts, cfg) \
        & (pts[..., 0] >= cfg.bev_x_min) & (pts[..., 0] < cfg.bev_x_max) \
        & (pts[..., 1] >= cfg.bev_y_min) & (pts[..., 1] < cfg.bev_y_max)
    # depth points of any class per ground cell: a real object fills its cells (its surfaces hide what
    # is behind it), scattered mislabelled ground or wall pixels make up a small share of theirs
    res = float(cfg.rgbd_cluster_res)
    allc = np.floor(pts[valid][:, :2] / res).astype(np.int64)
    all_keys, all_cnt = np.unique(allc[:, 0] * 1000003 + allc[:, 1], return_counts=True)

    def purity(Xg: np.ndarray) -> float:
        c = np.floor(Xg[:, :2] / res).astype(np.int64)
        k = c[:, 0] * 1000003 + c[:, 1]
        uk = np.unique(k)
        idx = np.clip(np.searchsorted(all_keys, uk), 0, max(len(all_keys) - 1, 0))
        tot = float(all_cnt[idx][all_keys[idx] == uk].sum()) if len(all_keys) else 0.0
        return Xg.shape[0] / max(tot, 1.0)

    found: Dict[int, List[Tuple[np.ndarray, np.ndarray]]] = {}
    for seg_c, kind in KIND_OF_SEG.items():
        m = valid & (cls == seg_c) & (conf > 0.5) & (pts[..., 2] > -1.0) & (pts[..., 2] < RGBD_MAX_Z[kind])
        if int(m.sum()) < RGBD_MIN_POINTS[kind]:
            continue
        X, S = pts[m], conf[m]
        groups = cluster_xy(X[:, :2], float(cfg.rgbd_cluster_res), RGBD_MIN_POINTS[kind], RGBD_LINK_CELLS[kind])
        if kind == KIND_VEHICLE:
            groups = merge_vehicle_parts(X, groups)
        found[kind] = [(X[g], S[g]) for g in groups
                       if float(S[g].mean()) >= RGBD_MIN_CONF and purity(X[g]) >= RGBD_MIN_PURITY]
    # A pedestrian or prop inside a detected vehicle's footprint is a mislabelled part of that vehicle.
    rects = []
    for Xv, _ in found.get(KIND_VEHICLE, []):
        c, ax, L, Wd = _oriented_extent(Xv[:, :2])
        rects.append((c, ax, L / 2.0 + 0.3, Wd / 2.0 + 0.3))

    def inside_vehicle(p: np.ndarray) -> bool:
        for c, ax, hl, hw in rects:
            q = p - c
            if abs(float(q @ ax)) <= hl and abs(float(q[0] * -ax[1] + q[1] * ax[0])) <= hw:
                return True
        return False

    dets: List[Dict[str, float]] = []
    for kind, clusters in found.items():
        for Xg, Sg in clusters:
            cx, cy = float(Xg[:, 0].mean()), float(Xg[:, 1].mean())
            if kind != KIND_VEHICLE and inside_vehicle(np.array([cx, cy])):
                continue
            r = max(math.hypot(cx, cy), 1e-3)
            sh = RGBD_CENTER_SHIFT[kind]
            cx, cy = cx + sh * cx / r, cy + sh * cy / r
            sc = float(Sg.mean())
            dn = min(r / 40.0, 1.0)
            s2 = float(np.clip(0.05 + 0.25 * dn + 0.6 * (1.0 - sc) + (0.15 if Xg.shape[0] < 3 * RGBD_MIN_POINTS[kind] else 0.0),
                               0.05, 0.95))
            dets.append({"kind": kind, "x": cx, "y": cy, "vx": 0.0, "vy": 0.0, "var": 2.0 * s2, "score": sc,
                         "n": int(Xg.shape[0])})
    # the traffic light that controls the ego lane: light-state pixels of the three front cameras
    tl = None
    dirs, offs = camera_rays(cfg, 1)
    df = np.asarray(depth[:3], np.float32).reshape(3, -1)
    pf = (offs[:3, None, :] + df[:, :, None] * dirs[:3]).reshape(-1, 3)
    ptl = np.transpose(probs[:3, 4:7].reshape(3, 3, -1), (0, 2, 1)).reshape(-1, 3)
    psum = ptl.sum(1)
    m = (psum > 0.4) & (df.reshape(-1) > 0.5) & (df.reshape(-1) < 70.0) & (pf[:, 0] > 1.0)
    if int(m.sum()) >= 4:
        Pt, pr, cs = pf[m], ptl[m], psum[m]
        groups = [g for g in cluster_xy(Pt[:, :2], 1.0, 4) if math.hypot(*Pt[g, :2].mean(0)) < 60.0]
        if lights_out is not None:
            for q in groups:
                kq = int(np.argmax(pr[q].sum(0)))
                cx_, cy_ = (float(c) for c in Pt[q, :2].mean(0))
                dq = math.hypot(cx_, cy_)
                scq = float(cs[q].mean())
                s2q = float(np.clip(0.08 + 0.2 * min(dq / 40.0, 1.0) + 0.5 * (1.0 - scq), 0.05, 0.9))
                lights_out.append((cx_, cy_, TL_CODE_OF_SEG[4 + kq], s2q / 4.0))
        if groups:
            g = max(groups, key=lambda q: q.size)
            k = int(np.argmax(pr[g].sum(0)))
            dist = float(math.hypot(*Pt[g, :2].mean(0)))
            sc = float(cs[g].mean())
            s2 = float(np.clip(0.08 + 0.2 * min(dist / 40.0, 1.0) + 0.5 * (1.0 - sc), 0.05, 0.9))
            tl = (dist, TL_CODE_OF_SEG[4 + k], s2 / 4.0)
    return dets, tl


class SegNet(nn.Module):
    """RGB-D U-Net for per-pixel classes (SEG_CLASSES) and the scene descriptor b(s) (weather, light,
    traffic density), regressed from the pooled bottleneck of the six images."""

    def __init__(self, cfg: Config):
        super().__init__()
        self.cfg = cfg

        def dil(c: int) -> nn.Sequential:
            return nn.Sequential(nn.Conv2d(c, c, 3, 1, 2, dilation=2, bias=False), nn.BatchNorm2d(c), nn.ReLU(inplace=True))

        self.e1 = nn.Sequential(_conv(4, 32, 2), _conv(32, 32))
        self.e2 = nn.Sequential(_conv(32, 64, 2), _conv(64, 64))
        self.e3 = nn.Sequential(_conv(64, 96, 2), _conv(96, 96))
        self.e4 = nn.Sequential(_conv(96, 128, 2), _conv(128, 128), dil(128))
        self.d3 = _conv(128 + 96, 96)
        self.d2 = _conv(96 + 64, 64)
        self.d1 = _conv(64 + 32, 48)
        self.d0 = _conv(48 + 4, 32)
        self.head = nn.Conv2d(32, len(SEG_CLASSES), 1)
        self.scene = nn.Sequential(nn.Linear(256, 64), nn.ReLU(), nn.Linear(64, cfg.scene_dim))
        self.register_buffer("mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 1, 3, 1, 1))
        self.register_buffer("std", torch.tensor([0.229, 0.224, 0.225]).view(1, 1, 3, 1, 1))

    def normalize(self, x01: torch.Tensor, depth: torch.Tensor) -> torch.Tensor:
        """x01 (B, N, 3, H, W) in [0, 1], depth (B, N, H, W) in metres -> (B, N, 4, H, W)."""
        d = torch.log1p(depth.float().clamp(0.0, 200.0)) / math.log1p(100.0) * 2.0 - 1.0
        return torch.cat([(x01 - self.mean) / self.std, d.unsqueeze(2)], 2)

    def preprocess(self, imgs: torch.Tensor, depth: torch.Tensor, degradation: float = 0.0) -> torch.Tensor:
        """uint8 images (B, N, H, W, 3) and depth (B, N, H, W); sensor degradation of the combined suite
        (Gaussian noise and horizontal motion blur) as for the camera detector."""
        x = imgs.float().permute(0, 1, 4, 2, 3) / 255.0
        if degradation > 0.0:
            k = 1 + 2 * int(round(3.0 * degradation))
            if k > 1:
                B, N, C, H, W = x.shape
                w = torch.ones(C, 1, 1, k, device=x.device) / k
                x = F.conv2d(x.reshape(B * N, C, H, W), w, padding=(0, k // 2), groups=C).reshape(B, N, C, H, W)
            x = (x + torch.randn_like(x) * 0.08 * degradation).clamp(0.0, 1.0)
        return self.normalize(x, depth)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """x (M, 4, H, W) -> class logits (M, C, H, W) and per-image scene vectors (M, scene_dim)."""
        e1 = self.e1(x)
        e2 = self.e2(e1)
        e3 = self.e3(e2)
        e4 = self.e4(e3)

        def up(t: torch.Tensor, ref: torch.Tensor) -> torch.Tensor:
            return F.interpolate(t, size=ref.shape[-2:], mode="bilinear", align_corners=False)

        d3 = self.d3(torch.cat([up(e4, e3), e3], 1))
        d2 = self.d2(torch.cat([up(d3, e2), e2], 1))
        d1 = self.d1(torch.cat([up(d2, e1), e1], 1))
        d0 = self.d0(torch.cat([up(d1, x), x], 1))
        return self.head(d0), self.scene(torch.cat([e4.mean((2, 3)), e4.amax((2, 3))], 1))

    @torch.no_grad()
    def infer(self, imgs: np.ndarray, depth: np.ndarray, device: torch.device, degradation: float = 0.0
              ) -> Tuple[np.ndarray, np.ndarray]:
        """One frame: imgs (N, H, W, 3) uint8, depth (N, H, W) -> probabilities (N, C, H, W), scene vector."""
        x = self.preprocess(torch.from_numpy(np.ascontiguousarray(imgs)).to(device).unsqueeze(0),
                            torch.from_numpy(np.asarray(depth, np.float32)).to(device).unsqueeze(0), degradation)
        logits, sc = self(x[0])
        return torch.softmax(logits, 1).cpu().numpy(), sc.mean(0).cpu().numpy()


class TokenTracker:
    """Short-term tracking of RGB-D detections in world coordinates: velocities from a line fit over
    the last 10 positions (0.5 s), and a track survives `coast` frames without a detection so that one
    missed frame does not remove a car 2 m ahead from the policy's input."""
    GATE = {KIND_VEHICLE: 2.5, KIND_WALKER: 1.2, KIND_STATIC: 1.2}
    HIST = 10

    def __init__(self, dt: float, coast: int = 4):
        self.dt, self.coast = float(dt), int(coast)
        self.reset()

    def reset(self) -> None:
        self.tracks: List[Dict[str, object]] = []
        self.pose = None
        self.t = 0
        self.next_id = 0

    def _assoc(self, pts: List[Tuple[int, float, float]], tracks: List[Dict[str, object]],
               key: Tuple[str, str]) -> List[Tuple[int, int]]:
        pairs = []
        for di, (kind, wx, wy) in enumerate(pts):
            for ti, tr in enumerate(tracks):
                if tr["kind"] != kind:
                    continue
                d = math.hypot(wx - float(tr[key[0]]), wy - float(tr[key[1]]))
                if d <= self.GATE.get(kind, 1.5):
                    pairs.append((d, di, ti))
        pairs.sort()
        used_d, used_t, out = set(), set(), []
        for _, di, ti in pairs:
            if di in used_d or ti in used_t:
                continue
            used_d.add(di)
            used_t.add(ti)
            out.append((di, ti))
        return out

    def _to_world(self, dets: List[Dict[str, float]]) -> List[Tuple[int, float, float]]:
        o, fwd, right, _ = self.pose
        return [(int(d["kind"]), float(o[0] + d["x"] * fwd[0] + d["y"] * right[0]),
                 float(o[1] + d["x"] * fwd[1] + d["y"] * right[1])) for d in dets]

    def update(self, dets: List[Dict[str, float]], o: np.ndarray, fwd: np.ndarray, right: np.ndarray,
               ev: np.ndarray) -> List[Dict[str, float]]:
        self.pose = (np.asarray(o, np.float64), np.asarray(fwd, np.float64), np.asarray(right, np.float64),
                     np.asarray(ev, np.float64))
        self.t += 1
        for tr in self.tracks:
            tr["px"] = float(tr["wx"]) + float(tr["vx"]) * self.dt
            tr["py"] = float(tr["wy"]) + float(tr["vy"]) * self.dt
        wpts = self._to_world(dets)
        pairs = self._assoc(wpts, self.tracks, ("px", "py"))
        matched_d = {di for di, _ in pairs}
        matched_t = {ti for _, ti in pairs}
        for di, ti in pairs:
            tr, d = self.tracks[ti], dets[di]
            _, wx, wy = wpts[di]
            hist = tr["hist"]
            hist.append((self.t, wx, wy))
            del hist[:-self.HIST]
            if int(tr["kind"]) != KIND_STATIC and len(hist) >= 3:
                ts = np.array([h[0] for h in hist], np.float64) * self.dt
                tc = ts - ts.mean()
                den = float((tc * tc).sum())
                vx = float((tc * (np.array([h[1] for h in hist]) - np.mean([h[1] for h in hist]))).sum() / den)
                vy = float((tc * (np.array([h[2] for h in hist]) - np.mean([h[2] for h in hist]))).sum() / den)
                sp = math.hypot(vx, vy)
                if sp > 25.0:
                    vx, vy = vx * 25.0 / sp, vy * 25.0 / sp
                tr["vx"], tr["vy"] = vx, vy
            tr.update(wx=wx, wy=wy, var=float(d["var"]), score=float(d.get("score", 1.0)), miss=0,
                      hits=int(tr["hits"]) + 1)
        for ti, tr in enumerate(self.tracks):
            if ti not in matched_t:
                tr["miss"] = int(tr["miss"]) + 1
                tr["wx"], tr["wy"] = tr["px"], tr["py"]
        self.tracks = [tr for tr in self.tracks if int(tr["miss"]) <= self.coast]
        for di, d in enumerate(dets):
            if di in matched_d:
                continue
            _, wx, wy = wpts[di]
            self.tracks.append({"id": self.next_id, "kind": int(d["kind"]), "wx": wx, "wy": wy, "vx": 0.0, "vy": 0.0,
                                "var": float(d["var"]), "score": float(d.get("score", 1.0)), "miss": 0, "hits": 1,
                                "hist": [(self.t, wx, wy)]})
            self.next_id += 1
        return self.outputs()

    def outputs(self, drop: frozenset = frozenset()) -> List[Dict[str, float]]:
        if self.pose is None:
            return []
        o, fwd, right, ev = self.pose
        out = []
        for tr in self.tracks:
            if int(tr["id"]) in drop:
                continue
            dx, dy = float(tr["wx"]) - o[0], float(tr["wy"]) - o[1]
            rvx, rvy = float(tr["vx"]) - ev[0], float(tr["vy"]) - ev[1]
            out.append({"kind": int(tr["kind"]), "x": dx * fwd[0] + dy * fwd[1], "y": dx * right[0] + dy * right[1],
                        "vx": rvx * fwd[0] + rvy * fwd[1], "vy": rvx * right[0] + rvy * right[1],
                        "var": min(float(tr["var"]) * (1.0 + float(tr["miss"])), 1.9), "score": float(tr["score"]),
                        "tid": int(tr["id"])})
        return out

    def occluded_outputs(self, dets_painted: List[Dict[str, float]]) -> List[Dict[str, float]]:
        """Tokens of this step if the painted images had been observed: tracks that were detected this
        step but are not detected in the painted images disappear; everything else is unchanged."""
        if self.pose is None:
            return []
        live = [tr for tr in self.tracks if int(tr["miss"]) == 0]
        pairs = self._assoc(self._to_world(dets_painted), live, ("wx", "wy"))
        seen = {int(live[ti]["id"]) for _, ti in pairs}
        lost = frozenset(int(tr["id"]) for tr in live if int(tr["id"]) not in seen)
        return self.outputs(drop=lost)


def segnet_batch_loss(net: SegNet, rgb: np.ndarray, depth: np.ndarray, seg: np.ndarray, scene: np.ndarray,
                      w_cls: torch.Tensor, device: torch.device, augment: bool) -> Tuple[torch.Tensor, Dict[str, float]]:
    """rgb (b, H, W, 3) uint8, depth (b, H, W), seg (b, H, W) class ids, scene (b, scene_dim)."""
    x = torch.from_numpy(np.ascontiguousarray(rgb)).to(device).float().permute(0, 3, 1, 2) / 255.0
    d = torch.from_numpy(np.asarray(depth, np.float32)).to(device)
    y = torch.from_numpy(np.asarray(seg, np.int64)).to(device)
    if augment:
        b = x.shape[0]
        # photometric jitter (brightness, contrast, colour gain, sensor noise) and horizontal flips
        x = x * torch.empty(b, 1, 1, 1, device=device).uniform_(0.5, 1.3)
        mu = x.mean((1, 2, 3), keepdim=True)
        x = (x - mu) * torch.empty(b, 1, 1, 1, device=device).uniform_(0.7, 1.3) + mu
        x = x * torch.empty(b, 3, 1, 1, device=device).uniform_(0.9, 1.1)
        x = (x + torch.randn_like(x) * torch.empty(b, 1, 1, 1, device=device).uniform_(0.0, 0.04)).clamp(0.0, 1.0)
        flip = torch.rand(b, device=device) < 0.5
        x = torch.where(flip[:, None, None, None], x.flip(-1), x)
        d = torch.where(flip[:, None, None], d.flip(-1), d)
        y = torch.where(flip[:, None, None], y.flip(-1), y)
    inp = net.normalize(x.unsqueeze(1), d.unsqueeze(1))[:, 0]
    logits, sc = net(inp)
    l_seg = F.cross_entropy(logits, y, weight=w_cls)
    l_sc = F.mse_loss(sc, torch.from_numpy(np.asarray(scene, np.float32)).to(device))
    return l_seg + 0.5 * l_sc, {"seg": float(l_seg.detach()), "scene": float(l_sc.detach())}


def match_with_ignore(dets: List[Dict[str, float]], visible: Sequence[Tuple[int, float, float]],
                      everything: Sequence[Tuple[int, float, float]]) -> Tuple[List[int], List[bool]]:
    """Detections matched to visible ground truth (index or -1), and whether an unmatched detection
    lies on a real object that is only hidden from the label's line-of-sight test or is a parked car
    of the map (ignored, neither true nor false positive)."""
    ids = match_detections(dets, [(j, k, x, y) for j, (k, x, y) in enumerate(visible)])
    thr = {KIND_VEHICLE: 3.0, KIND_WALKER: 1.5, KIND_STATIC: 1.5}
    ign = []
    for d, i in zip(dets, ids):
        ign.append(i < 0 and any(k == d["kind"] and math.hypot(d["x"] - x, d["y"] - y) <= thr[k] for k, x, y in everything))
    return ids, ign


@dataclass
class Conditions:
    """One driving condition: a scenario template of Sec. 3.5 or an evaluation route of Sec. 4.1."""
    town: str
    weather: str
    npc: int
    walkers: int
    degradation: float = 0.0          # 0 = clean sensors, 1 = full degradation of the combined suite
    barrel_mode: str = "none"         # none | coupled (source: barrels precede every forced stop) | random
    queue_mode: str = "none"          # none | green_queue (adjacent-lane queue while the ego light is green)
    in_source_support: bool = True    # used only by the oracle-coverage ablation
    route_seed: Optional[int] = None  # fixed seed for evaluation routes

    def key(self) -> str:
        return (f"{self.town}|{self.weather}|n{self.npc}|w{self.walkers}|d{self.degradation:.2f}|"
                f"b:{self.barrel_mode}|q:{self.queue_mode}")


@dataclass
class PrivEntity:
    eid: int
    kind: int
    x: float
    y: float
    vx: float
    vy: float
    radius: float
    tl_state: str = ""      # traffic signals only
    role: str = "other"     # cause | distractor | other (ground truth of the scripted suites)
    yaw: float = float("nan")   # heading in rad (vehicles); gives the lateral extent of a turned car
    half_w: float = float("nan")  # half-width from the actor's bounding box (radius is the half-length)


@dataclass
class PrivScene:
    """Privileged copy of the scene used for interventions (Sec. 3.4).

    CARLA 0.9.15 has no world snapshot/restore, so Eqs. (20)-(22) are evaluated
    on this privileged kinematic copy of the scene (deviation register, item 2).
    """
    ego_x: float
    ego_y: float
    ego_yaw_deg: float
    ego_v: float
    route: np.ndarray          # (M, 2) route points ahead of the ego
    route_s: np.ndarray        # (M,) arc length along these points, 0 at the ego projection
    lane_width: float
    entities: List[PrivEntity]


class CaSTCarlaEnv(CarlaReliableTransferEnv):
    """Environment with the frozen perception front end of Sec. 3.1 and the suites of Sec. 4.

    The perception front end is a simulated detector: privileged actor states with
    distance-, weather-, and degradation-dependent misses, position/velocity noise,
    and a predicted detection variance (deviation register, item 1).
    """

    def __init__(self, host: str, port: int, town_name: str, cfg: Config):
        self.cond: Conditions = Conditions(town=town_name, weather="clear_day", npc=0, walkers=0)
        self.props: List[carla.Actor] = []
        self.queue_vehicles: List[carla.Actor] = []
        self.forced_lights: List[carla.Actor] = []
        self.managed: List[Dict[str, object]] = []
        self._jam_track: Dict[int, Tuple[Optional[Tuple[float, float]], int]] = {}
        self.jam_removed = 0
        self._ego_still = 0
        self._red_still = 0
        self.scenario_events: List[Dict[str, object]] = []
        self.role_by_id: Dict[int, str] = {}
        self.slot_ids = np.full((cfg.max_entities,), -1, dtype=np.int64)
        self.slot_kinds = np.full((cfg.max_entities,), -1, dtype=np.int64)
        self.scene_vec = np.zeros((cfg.scene_dim,), dtype=np.float32)
        self.det_true = 0
        self.det_found = 0
        self.frame_dropped = False
        self.detector: Optional["BEVDetector"] = None
        self.det_device = torch.device("cpu")
        self.rig: Optional["CameraRig"] = None
        self._prev_lifted: Optional[torch.Tensor] = None
        self._prev_used: Optional[torch.Tensor] = None
        self.last_imgs: Optional[np.ndarray] = None
        self.last_depth: Optional[np.ndarray] = None
        self.tracker = TokenTracker(cfg.dt, int(cfg.track_coast))
        self._tl_hold: Tuple[Optional[Tuple[float, float, float]], int] = (None, 99)
        self._tl_lights: Optional[List[carla.Actor]] = None
        self._tl_boxes: Dict[int, List[Tuple[float, float, float, float, float, float, float]]] = {}
        self.route_lights: List[Tuple[carla.Actor, float]] = []
        self.infractions: List[Tuple[str, float]] = []
        super().__init__(host=host, port=port, town_name=town_name, fixed_spawn_index=-1,
                         fixed_goal_index=-1, weather_mode="clear_day", cfg=cfg)

    # ---------------- sensor callback (no RPC in the sensor thread) ----------------
    def _on_collision_event(self, event: object) -> None:
        if self._teardown_in_progress or (not self._episode_live):
            return
        try:
            # The car settling onto the road right after spawning can touch the kerb (seen as a
            # "static.sidewalk" hit at 17.8 m/s on step 2); that is not a driving collision.
            if int(self.episode_steps) <= 10 and str(event.other_actor.type_id).startswith("static.") \
                    and not str(event.other_actor.type_id).startswith("static.prop"):
                return
        except BaseException:
            pass
        self.collision_events.append(event)
        if len(self.collision_events) == 1:
            self._collision_where_first = self._collision_where()   # speeds at the moment of contact

    # ---------------- conditions ----------------
    def set_conditions(self, cond: Conditions) -> None:
        self.cond = cond
        self.weather_mode = cond.weather

    # ---------------- perception front end (Sec. 3.1) ----------------
    # ---------------- perception front end (Sec. 3.1) ----------------
    def set_detector(self, detector: Optional["BEVDetector"], device: torch.device) -> None:
        self.detector, self.det_device = detector, device

    def _setup_collision(self) -> None:
        super()._setup_collision()
        self.rig = None
        self._prev_lifted = None
        self.last_imgs = None
        self.last_depth = None
        self._route_ego_key = None
        self.tracker.reset()
        try:
            ext = self.vehicle.bounding_box.extent
            self.cfg.ego_half_length, self.cfg.ego_half_width = float(ext.x), float(ext.y)
        except BaseException:
            pass
        self._tl_hold = (None, 99)
        self._map_tl_hold = (None, -1, 99)
        if not self.cfg.use_cameras or not self._actor_is_alive(self.vehicle):
            return
        try:
            self.rig = CameraRig(self.world, self.bp_lib, self.vehicle, self.cfg, depth=bool(self.cfg.use_depth))
            self.sensor_list.extend(self.rig.actors)
        except BaseException as e:
            print(f"[WARN] camera rig setup failed: {e}")
            self.rig = None

    def _ego_frame(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        tf = self.vehicle.get_transform()
        yaw = math.radians(tf.rotation.yaw)
        v = self.vehicle.get_velocity()
        return (np.array([tf.location.x, tf.location.y]), np.array([math.cos(yaw), math.sin(yaw)]),
                np.array([-math.sin(yaw), math.cos(yaw)]), np.array([v.x, v.y]))

    def _true_entities(self) -> List[Tuple[int, int, float, float, float, float]]:
        """Ground-truth actors in the ego frame (id, kind, x, y, vx_rel, vy_rel): detector labels and
        detection recall only, never a policy input."""
        o, fwd, right, ev = self._ego_frame()
        out = []
        loc = self.vehicle.get_location()

        def add(actor: carla.Actor, kind: int) -> None:
            try:
                l = actor.get_location()
                d = np.array([l.x, l.y]) - o
                x, y = float(np.dot(d, fwd)), float(np.dot(d, right))
                if not (self.cfg.bev_x_min <= x < self.cfg.bev_x_max and self.cfg.bev_y_min <= y < self.cfg.bev_y_max):
                    return
                if math.hypot(x, y) > self.cfg.entity_max_dist:
                    return
                vv = actor.get_velocity() if kind != KIND_STATIC else None
                rv = (np.array([vv.x, vv.y]) if vv is not None else np.zeros(2)) - ev
                out.append((int(actor.id), kind, x, y, float(np.dot(rv, fwd)), float(np.dot(rv, right))))
            except BaseException:
                return

        for npc in self.npcs:
            if self._actor_is_alive(npc):
                add(npc, KIND_VEHICLE)
        for w in self._nearby_walkers(loc):
            add(w, KIND_WALKER)
        for p in self.props:
            if self._actor_is_alive(p):
                add(p, KIND_STATIC)
        return out

    def _line_of_sight(self, wx: float, wy: float, wz: float, radius: float) -> bool:
        """Is the actor at (wx, wy, wz) visible from the camera rig (one ray from 1.8 m above the ego)?
        The first non-ground surface the ray hits must lie on the actor. If ray casting is unavailable
        the actor counts as visible."""
        try:
            ego = self.vehicle.get_location()
            start = carla.Location(ego.x, ego.y, ego.z + 1.8)
            end = carla.Location(wx, wy, wz)
            pts = self.world.cast_ray(start, end)
        except BaseException:
            return True
        best = None
        for pt in pts or []:
            name = str(getattr(pt, "label", "")).lower()
            if any(g in name for g in ("road", "sidewalk", "ground", "terrain", "water", "sky", "none", "unlabeled")):
                continue
            d = start.distance(pt.location)
            if best is None or d < best[0]:
                best = (d, pt)
        if best is None:
            return True
        return best[1].location.distance(end) <= radius + 1.5

    def perception_labels(self, visible_only: bool = True) -> Tuple[np.ndarray, np.ndarray]:
        """Detector labels: actors within label_max_dist that the cameras can actually see. Labels for
        actors hidden behind buildings or other cars (or 60 m away) taught the detector to guess, and
        it ended with 35% recall and 38% precision for vehicles."""
        rows = []
        o, fwd, right, _ = self._ego_frame()
        ego_z = float(self.vehicle.get_location().z)
        zoff = {KIND_VEHICLE: 0.8, KIND_WALKER: 0.9, KIND_STATIC: 0.4}
        for _, k, x, y, vx, vy in self._true_entities():
            if math.hypot(x, y) > float(self.cfg.label_max_dist):
                continue
            if visible_only:
                w = o + x * fwd + y * right
                if not self._line_of_sight(float(w[0]), float(w[1]), ego_z + zoff.get(k, 0.5), KIND_RADIUS_M[k]):
                    continue
            rows.append([k, x, y, vx, vy])
        lab = np.array(rows, np.float32).reshape(-1, 5)
        st, dist = self._get_active_traffic_light_info()
        tl_rng = max(float(self.cfg.tl_near_dist), self._red_brake_distance() + 6.0)
        idx = TL_STATES.index(st) if (st in TL_STATES and 0.0 <= dist < tl_rng) else 0
        return lab, np.array([idx, float(dist) / 40.0 if idx > 0 else 0.0], np.float32)

    # ---------------- RGB-D perception (--perception rgbd) ----------------
    def _traffic_lights(self) -> List[carla.Actor]:
        if self._tl_lights is None:
            try:
                self._tl_lights = list(self.world.get_actors().filter("traffic.traffic_light*"))
            except BaseException:
                self._tl_lights = []
        return self._tl_lights

    def _light_boxes(self, tl: carla.Actor) -> List[Tuple[float, float, float, float, float, float, float]]:
        """World boxes (x, y, z, yaw_rad, ext_x, ext_y, ext_z) of the light heads of a traffic light."""
        lid = int(tl.id)
        if lid not in self._tl_boxes:
            out = []
            try:
                for bb in tl.get_light_boxes():
                    out.append((float(bb.location.x), float(bb.location.y), float(bb.location.z),
                                math.radians(float(bb.rotation.yaw)), float(bb.extent.x), float(bb.extent.y),
                                float(bb.extent.z)))
            except BaseException:
                out = []
            self._tl_boxes[lid] = out
        return self._tl_boxes[lid]

    def rgbd_boxes(self, max_dist: float = 70.0) -> List[Tuple]:
        """Ego-frame 3D boxes (seg_class, cx, cy, cz, yaw_rel, ext_x, ext_y, ext_z) of the cars, map-parked
        cars, pedestrians, props, and traffic-light heads around the ego vehicle (training labels)."""
        o, fwd, right, _ = self._ego_frame()
        etf = self.vehicle.get_transform()
        ez, eyaw = float(etf.location.z), math.radians(float(etf.rotation.yaw))
        boxes: List[Tuple] = []

        def to_ego(wx: float, wy: float) -> Tuple[float, float]:
            d = np.array([wx, wy]) - o
            return float(np.dot(d, fwd)), float(np.dot(d, right))

        def add_actor(a: carla.Actor, cls: int) -> None:
            try:
                tf, bb = a.get_transform(), a.bounding_box
                ps = math.radians(float(tf.rotation.yaw))
                wx = tf.location.x + bb.location.x * math.cos(ps) - bb.location.y * math.sin(ps)
                wy = tf.location.y + bb.location.x * math.sin(ps) + bb.location.y * math.cos(ps)
                x, y = to_ego(wx, wy)
                if math.hypot(x, y) > max_dist:
                    return
                boxes.append((cls, x, y, float(tf.location.z + bb.location.z) - ez, ps - eyaw,
                              float(bb.extent.x), float(bb.extent.y), float(bb.extent.z)))
            except BaseException:
                return

        for px, py, pyaw, hl, hw in self._map_parked_vehicles():
            x, y = to_ego(px, py)
            if math.hypot(x, y) <= max_dist:
                boxes.append((1, x, y, 0.75, pyaw - eyaw, hl, hw, 1.5))
        for a in self.npcs:
            if self._actor_is_alive(a):
                add_actor(a, 1)
        for a in self.props:
            if self._actor_is_alive(a):
                add_actor(a, 3)
        for a in self._nearby_walkers(self.vehicle.get_location()):
            add_actor(a, 2)
        active = self._active_light_id()
        for tl in self._traffic_lights():
            try:
                l = tl.get_location()
                if math.hypot(*to_ego(l.x, l.y)) > max_dist:
                    continue
                st = str(tl.get_state()).split(".")[-1]
            except BaseException:
                continue
            cls = TL_SEG_OF_STATE.get(st, 7) if int(tl.id) == int(active) else 7
            for bx, by, bz, byaw, ex_, ey_, ez_ in self._light_boxes(tl):
                x, y = to_ego(bx, by)
                boxes.append((cls, x, y, bz - ez, byaw - eyaw, ex_, ey_, ez_))
        return boxes

    def rgbd_seg_labels(self) -> Optional[np.ndarray]:
        """Per-pixel training labels (6, H, W) of the last depth images: the class of the 3D box that
        contains the pixel's depth point (visible surfaces only, by construction)."""
        if self.last_depth is None:
            return None
        pts, d = backproject(self.last_depth, self.cfg, 1)
        N = pts.shape[0]
        P = pts.reshape(-1, 3)
        valid = (d.reshape(-1) < float(self.cfg.rgbd_max_depth) + 15.0) & ~ego_box_mask(P, self.cfg)
        lab = label_points_with_boxes(P, valid, self.rgbd_boxes())
        return lab.reshape(N, int(self.cfg.cam_height), int(self.cfg.cam_width))

    def all_objects_ego(self) -> List[Tuple[int, float, float]]:
        """Every actor and map-parked car within range (kind, x, y), visible or not (evaluation only)."""
        out = [(int(k), float(x), float(y)) for _, k, x, y, _, _ in self._true_entities()]
        o, fwd, right, _ = self._ego_frame()
        for px, py, _, _, _ in self._map_parked_vehicles():
            d = np.array([px, py]) - o
            x, y = float(np.dot(d, fwd)), float(np.dot(d, right))
            if math.hypot(x, y) <= self.cfg.entity_max_dist:
                out.append((KIND_VEHICLE, x, y))
        return out

    def _rgbd_run(self, imgs: np.ndarray, depth: np.ndarray
                  ) -> Tuple[List[Dict[str, float]], Optional[Tuple[float, float, float]], np.ndarray]:
        probs, scene = self.detector.infer(imgs, depth, self.det_device, float(self.cond.degradation))
        self._camera_lights = []
        dets, tl = rgbd_detect(probs, depth, self.cfg, self._camera_lights)
        return dets, tl, scene.astype(np.float32)

    def _rgbd_entity_features(self) -> Tuple[np.ndarray, np.ndarray, float]:
        cfg = self.cfg
        deg = float(self.cond.degradation)
        self.frame_dropped = bool(deg > 0.0 and random.random() < cfg.frame_drop_prob_max * deg)
        imgs = self._grab_images()
        self.last_imgs = imgs
        self.slot_ids[:] = -1
        self.slot_kinds[:] = -1
        empty = (np.zeros((cfg.max_entities, cfg.edge_dim), np.float32), np.zeros((cfg.max_entities,), np.float32),
                 cfg.entity_max_dist)
        if self.detector is None or imgs is None or self.last_depth is None:
            self.scene_vec = np.zeros((cfg.scene_dim,), np.float32)
            return empty
        lane_curv, lane_head = self._lane_features()
        o, fwd, right, ev = self._ego_frame()
        if self.frame_dropped:
            dets, tl = [], None                      # no new frame: the tracks coast
        else:
            dets, tl, self.scene_vec = self._rgbd_run(imgs, self.last_depth)
        held, age = self._tl_hold
        if tl is not None:
            self._tl_hold = (tl, 0)
        elif held is not None and age < int(cfg.track_coast):
            tl, self._tl_hold = held, (held, age + 1)
        else:
            self._tl_hold = (None, 99)
        tl = self._map_light_token(tl, o, fwd, right)
        self._rgbd_tl = tl
        tracks = self.tracker.update(dets, o, fwd, right, ev)
        truth = self._true_entities()
        ids = match_detections(tracks, [(t[0], t[1], t[2], t[3]) for t in truth])
        tl_true_state, tl_true_dist = self._get_active_traffic_light_info()
        tl_id = self._active_light_id() if (tl is not None and tl_true_dist >= 0.0) else -1
        self._rgbd_tl_id = tl_id
        edges, mask, sid, skind, nearest = self._build_tokens(tracks, tl, ids, lane_curv, lane_head, tl_id)
        self.slot_ids[:], self.slot_kinds[:] = sid, skind
        self.det_true += len(truth)
        self.det_found += sum(1 for i in ids if i >= 0)
        near = {int(t[0]) for t in truth if math.hypot(t[2], t[3]) <= 25.0}
        self.det_true_near += len(near)
        self.det_found_near += sum(1 for i in ids if int(i) in near)
        return edges, mask, nearest

    def _grab_images(self) -> Optional[np.ndarray]:
        if self.rig is None:
            return None
        try:
            frame = int(self.world.get_snapshot().frame)
        except BaseException:
            return None
        got = self.rig.get(frame)
        if got is None:
            self.last_depth = None
            return None
        rgb, dep = got
        self.last_depth = dep
        return rgb

    @torch.no_grad()
    def _run_detector(self, imgs: np.ndarray, prev: Optional[torch.Tensor]) -> Dict[str, torch.Tensor]:
        x = torch.from_numpy(imgs).to(self.det_device).unsqueeze(0)
        x = self.detector.preprocess(x, float(self.cond.degradation), self.frame_dropped)
        return self.detector(x, prev)

    def _tokens_from(self, out: Dict[str, torch.Tensor], lane_curv: float, lane_head: float
                     ) -> Tuple[np.ndarray, np.ndarray, List[Dict[str, float]], Optional[Tuple[float, float, float]]]:
        cfg = self.cfg
        dets = self.detector.decode(out, thr=float(cfg.det_threshold), k=2 * cfg.max_entities)
        dets = [d for d in dets if math.hypot(d["x"], d["y"]) <= cfg.entity_max_dist]
        probs = torch.softmax(out["tl_logits"][0], -1)
        st = int(torch.argmax(probs))
        tl = None
        if st > 0 and float(probs[st]) > 0.5:
            tl = (max(0.0, float(out["tl_dist"][0]) * 40.0), {1: 1.0, 2: 0.5, 3: 0.0}[st], float(math.exp(float(out["tl_lv"][0]))))
        return dets, probs.cpu().numpy(), tl, None

    def _route_relative(self, xy: np.ndarray) -> np.ndarray:
        """Route-frame coordinates (lateral offset d, + right of the route; arc length s from the ego)
        of ego-frame points (n, 2). The route is the navigation input the policy already follows; the
        teacher decides whether a car blocks the ego by exactly this lateral offset."""
        out = np.stack([xy[:, 1], xy[:, 0]], 1).astype(np.float32) if xy.shape[0] else np.zeros((0, 2), np.float32)
        try:
            key = int(self.episode_steps)
            if getattr(self, "_route_ego_key", None) != key:
                o, fwd, right, _ = self._ego_frame()
                s_ego = self._s_ego(self.vehicle.get_location())
                cum = np.asarray(self.route_cumdist, np.float64)
                pts = np.asarray(self.route_xy, np.float64)
                i0 = max(0, int(np.searchsorted(cum, s_ego - 25.0)) - 1)
                i1 = int(np.searchsorted(cum, s_ego + float(self.cfg.entity_max_dist) + 10.0, side="right")) + 1
                P, S = pts[i0:i1], cum[i0:i1] - s_ego
                d = P - o
                self._route_ego = (np.stack([d @ fwd, d @ right], 1), S) if P.shape[0] >= 2 else None
                self._route_ego_key = key
            if self._route_ego is None or xy.shape[0] == 0:
                return out
            P, S = self._route_ego
            A, B = P[:-1], P[1:]
            AB = B - A
            L = np.maximum(np.linalg.norm(AB, axis=1), 1e-6)
            u = AB / L[:, None]
            q = xy[:, None, :] - A[None, :, :]                               # (n, m, 2)
            t = np.clip((q * u[None]).sum(-1), 0.0, L[None, :])
            proj = A[None] + t[..., None] * u[None]
            dist = np.linalg.norm(xy[:, None, :] - proj, axis=-1)
            j = dist.argmin(1)
            r = np.arange(xy.shape[0])
            nrm = np.stack([-u[j, 1], u[j, 0]], 1)
            lat = ((xy - proj[r, j]) * nrm).sum(1)
            arc = S[j] + t[r, j]
            return np.stack([lat, arc], 1).astype(np.float32)
        except BaseException:
            return out

    def _map_light_token(self, tl_camera, o, fwd, right):
        """Light token by --light-source. "camera": the largest light cluster the front cameras see, at the
        distance of the light head (earlier behaviour). "map_camera": the light that governs the ego lane and
        its stop-line distance come from the map (as with an HD map and localisation); the colour comes from
        the camera light cluster on one of that light's heads. "v2i": map association and the broadcast light
        state. In Town01/02 the camera often saw another approach's light, and its head stands metres away from
        the stop line, so the token gave the wrong light and the wrong place to stop.

        Token v2 (default) fixes three faults of v1 (rgbd12), which froze the car at junctions and at spawn:
        an unseen colour was encoded exactly like a low-confidence yellow (code 0.5, the same variance); a seen
        colour was kept for only 0.2 s; and clusters were matched to the pole base, metres from an overhanging
        head. v2 matches clusters to the light-head boxes (pole fallback), keeps the last seen colour of the
        same light for tl_colour_hold_s with growing variance, and marks a colour never seen (a 0.3 m head
        beyond about 25 m covers 1-2 pixels) with a flag in the signal token's otherwise unused slot."""
        src = str(getattr(self.cfg, "light_source", "camera"))
        if src == "camera":
            return tl_camera
        try:
            state, sd = self._get_active_traffic_light_info()
        except Exception:
            return None
        if state not in ("Red", "Yellow", "Green") or not (-2.0 <= sd <= float(self.cfg.tl_detect_dist)):
            self._map_tl_hold = (None, -1, 99)
            return None
        dist = max(float(sd), 0.0)
        if src == "v2i":
            return (dist, {"Red": 1.0, "Yellow": 0.5, "Green": 0.0}[state], 0.005, 0.0)
        lid = int(self._active_light_id())
        if int(getattr(self.cfg, "light_token_version", 2)) < 2:
            return self._map_light_token_v1(dist, lid, o, fwd, right)
        seen = self._camera_colour_of_light(lid, o, fwd, right)
        held_code, held_id, age = getattr(self, "_map_tl_hold", (None, -1, 99))
        hold_n = max(1, int(round(float(self.cfg.tl_colour_hold_s) / float(self.cfg.dt))))
        if seen is not None:
            code, var = seen
            self._map_tl_hold = (code, lid, 0)
            return (dist, code, min(var, 0.15), 0.0)
        if held_code is not None and held_id == lid and age < hold_n:
            age += 1
            self._map_tl_hold = (held_code, lid, age)
            return (dist, held_code, 0.05 + 0.10 * age / hold_n, 0.0)
        self._map_tl_hold = (None, lid, 99)
        return (dist, 0.5, 0.245, 1.0)               # light ahead, colour not seen: flagged unknown

    def _camera_colour_of_light(self, lid: int, o, fwd, right) -> Optional[Tuple[float, float]]:
        """(code, var) of the camera light cluster on a head of light lid (within 3.5 m of a head box centre;
        within 10 m of the pole when the boxes are unavailable), or None."""
        cams = getattr(self, "_camera_lights", None)
        if lid < 0 or not cams:
            return None
        try:
            actor = self.world.get_actor(int(lid))
            if actor is None:
                return None
            heads = [(bx, by) for bx, by, *_ in self._light_boxes(actor)]
            if heads:
                refs, radius = heads, 3.5
            else:
                loc = actor.get_location()
                refs, radius = [(float(loc.x), float(loc.y))], 10.0
            pts = []
            for wx, wy in refs:
                d = np.array([wx, wy]) - o
                pts.append((float(d @ fwd), float(d @ right)))
            best, bd = None, radius
            for c in cams:
                dd = min(math.hypot(c[0] - px, c[1] - py) for px, py in pts)
                if dd < bd:
                    best, bd = c, dd
            return None if best is None else (float(best[2]), float(best[3]))
        except Exception:
            return None

    def _map_light_token_v1(self, dist: float, lid: int, o, fwd, right):
        """rgbd12 behaviour (--light-token-version 1), kept to evaluate those checkpoints unchanged."""
        code, var = None, None
        try:
            actor = self.world.get_actor(int(lid)) if lid >= 0 else None
            if actor is not None and getattr(self, "_camera_lights", None):
                loc = actor.get_location()
                d = np.array([loc.x, loc.y]) - o
                px, py = float(d @ fwd), float(d @ right)
                best = min(self._camera_lights, key=lambda c: math.hypot(c[0] - px, c[1] - py))
                if math.hypot(best[0] - px, best[1] - py) < 10.0:
                    code, var = float(best[2]), float(best[3])
        except Exception:
            code = None
        held_code, held_id, age = getattr(self, "_map_tl_hold", (None, -1, 99))
        if code is not None:
            self._map_tl_hold = (code, lid, 0)
        elif held_code is not None and held_id == lid and age < int(self.cfg.track_coast):
            code, var = held_code, 0.15
            self._map_tl_hold = (held_code, lid, age + 1)
        else:
            code, var = 0.5, 0.225                  # light ahead, colour not seen: unknown (same as yellow)
        return (dist, code, var)

    def _build_tokens(self, dets: List[Dict[str, float]], tl: Optional[Tuple[float, float, float]],
                      ids: List[int], lane_curv: float, lane_head: float, tl_id: int):
        cfg = self.cfg
        cands = []
        # Per token: route-relative lateral offset (in lane widths of 3.5 m, clipped at 3) and arc length.
        # These two slots held the ego lane curvature and heading offset, which are already scalars.
        rr = self._route_relative(np.array([[d["x"], d["y"]] for d in dets], np.float32).reshape(-1, 2))
        for k, (d, eid) in enumerate(zip(dets, ids)):
            onehot = np.zeros(cfg.sem_dim, np.float32)
            onehot[d["kind"]] = 1.0
            s2 = float(np.clip(d["var"] / 2.0, 0.02, 0.98))
            lat = float(np.clip(rr[k, 0] / 3.5, -3.0, 3.0) / 3.0)
            arc = float(np.clip(rr[k, 1] / cfg.entity_max_dist, -1.0, 1.0))
            e = np.concatenate([np.array([d["x"] / cfg.entity_max_dist, d["y"] / cfg.entity_max_dist,
                                          d["vx"] / 15.0, d["vy"] / 10.0], np.float32), onehot,
                                np.array([0.0, lat, arc, s2], np.float32)])
            cands.append((math.hypot(d["x"], d["y"]), eid, d["kind"], e))
        if tl is not None:
            dist, code, var = tl[0], tl[1], tl[2]
            unknown = float(tl[3]) if len(tl) > 3 else 0.0   # map_camera v2: colour not seen (slot unused for lights)
            onehot = np.zeros(cfg.sem_dim, np.float32)
            onehot[KIND_SIGNAL] = 1.0
            s2 = float(np.clip(var * 4.0, 0.02, 0.98))
            e = np.concatenate([np.array([float(np.clip(dist / cfg.entity_max_dist, 0.0, 1.0)), 0.0, 0.0, 0.0], np.float32),
                                onehot, np.array([code, unknown, float(np.clip(dist / cfg.entity_max_dist, 0.0, 1.0)), s2],
                                                 np.float32)])
            cands.append((dist, tl_id, KIND_SIGNAL, e))
        cands.sort(key=lambda c: c[0])
        cands = cands[: cfg.max_entities]
        edges = np.zeros((cfg.max_entities, cfg.edge_dim), np.float32)
        mask = np.zeros((cfg.max_entities,), np.float32)
        sid = np.full((cfg.max_entities,), -1, np.int64)
        skind = np.full((cfg.max_entities,), -1, np.int64)
        for i, (_, eid, kind, e) in enumerate(cands):
            edges[i], mask[i], sid[i], skind[i] = e, 1.0, eid, kind
        nearest = float(cands[0][0]) if cands else cfg.entity_max_dist
        return edges, mask, sid, skind, nearest

    def _lane_features(self) -> Tuple[float, float]:
        wp = self.map.get_waypoint(self.vehicle.get_location(), project_to_road=True, lane_type=carla.LaneType.Driving)
        return float(np.tanh(10.0 * self._curvature_ahead(wp))), float(self._lane_heading_offset(wp))

    def _weather_scene_vec(self) -> np.ndarray:
        """Scene descriptor b(s) in ground-truth perception mode: weather, light and traffic density."""
        v = np.zeros((self.cfg.scene_dim,), np.float32)
        try:
            w = self.world.get_weather()
            vals = [w.cloudiness / 100.0, w.precipitation / 100.0, w.precipitation_deposits / 100.0,
                    w.wetness / 100.0, w.fog_density / 100.0, (w.sun_altitude_angle + 90.0) / 180.0,
                    1.0 if w.sun_altitude_angle < 0.0 else 0.0, float(self.cond.degradation),
                    min(len(self.npcs) / 60.0, 1.0)]
            v[:len(vals)] = np.asarray(vals, np.float32)
        except BaseException:
            pass
        return v

    def _gt_entity_features(self, drop_id: Optional[int] = None) -> Tuple[np.ndarray, np.ndarray, float]:
        """Ground-truth perception mode (the observation model of the CVPR 2026 code): actors within
        range with distance-, fog- and degradation-dependent dropout and position/velocity noise; the
        token variance is the noise level. Used with --perception gt."""
        cfg = self.cfg
        lane_curv, lane_head = self._lane_features()
        if drop_id is not None and getattr(self, "_gt_last", None) is not None:
            # Occlusion intervention: the same noisy observation as this step, without that actor.
            dets0, ids0, tl0, tl_id0 = self._gt_last
            keep = [k for k, i in enumerate(ids0) if int(i) != int(drop_id)]
            edges, mask, _, _, nearest = self._build_tokens([dets0[k] for k in keep], tl0, [ids0[k] for k in keep],
                                                            lane_curv, lane_head, tl_id0)
            return edges, mask, nearest
        fog = min(1.0, get_fog_norm(self.world) + 0.5 * float(self.cond.degradation))
        dets, ids = [], []
        truth = self._true_entities()
        for eid, k, x, y, vx, vy in truth:
            dist = math.hypot(x, y)
            if dist < 1.0 or (drop_id is not None and int(eid) == int(drop_id)):
                continue
            if self._observation_dropout(dist, fog, "walker" if k == KIND_WALKER else "vehicle"):
                continue
            dn = float(np.clip(dist / max(cfg.entity_max_dist, 1e-3), 0.0, 1.0))
            ps = cfg.obs_pos_noise_m * (0.45 + 0.80 * dn + 0.90 * fog)
            vs = cfg.obs_vel_noise_ms * (0.35 + 0.75 * dn + 0.80 * fog)
            s2 = float(np.clip(0.05 + 0.22 * dn + 0.28 * fog + 0.08 * ps + 0.04 * vs + (0.06 if k == KIND_WALKER else 0.0),
                               0.05, 0.95))
            dets.append({"kind": int(k), "x": x + float(np.random.normal(0.0, ps)), "y": y + float(np.random.normal(0.0, ps)),
                         "vx": vx + float(np.random.normal(0.0, vs)), "vy": vy + float(np.random.normal(0.0, vs)),
                         "var": 2.0 * s2, "score": 1.0})
            ids.append(int(eid))
        tl = None
        st, tdist = self._get_active_traffic_light_info()
        tl_rng = max(float(cfg.tl_near_dist), self._red_brake_distance() + 6.0)
        if st in ("Red", "Yellow", "Green") and 0.0 <= tdist < tl_rng and not self._observation_dropout(tdist, fog, "traffic_light"):
            nd = float(max(0.0, tdist + np.random.normal(0.0, cfg.tl_obs_noise_m * (0.35 + fog))))
            s2 = float(np.clip(0.08 + 0.24 * nd / max(cfg.tl_near_dist, 1e-3) + 0.25 * fog, 0.05, 0.90))
            tl = (nd, {"Red": 1.0, "Yellow": 0.5, "Green": 0.0}[st], s2 / 4.0)
        tl_id = self._active_light_id() if tl is not None else -1
        edges, mask, sid, skind, nearest = self._build_tokens(dets, tl, ids, lane_curv, lane_head, tl_id)
        self._gt_last = (dets, ids, tl, tl_id)
        if drop_id is None:
            self.slot_ids[:], self.slot_kinds[:] = sid, skind
            self.det_true += len(truth)
            self.det_found += len(ids)
            near = {int(t[0]) for t in truth if math.hypot(t[2], t[3]) <= 25.0}
            self.det_true_near += len(near)
            self.det_found_near += sum(1 for i in ids if int(i) in near)
            self.scene_vec = self._weather_scene_vec()
        return edges, mask, nearest

    def _collect_entity_features(self) -> Tuple[np.ndarray, np.ndarray, float]:
        """Entity tokens from the frozen camera-based BEV detector (Sec. 3.1), or from noisy ground
        truth with --perception gt."""
        assert self.vehicle is not None
        cfg = self.cfg
        if str(cfg.perception_mode) == "gt":
            self.slot_ids[:] = -1
            self.slot_kinds[:] = -1
            self.frame_dropped = False
            self.last_imgs = None
            return self._gt_entity_features()
        if str(cfg.perception_mode) == "rgbd":
            return self._rgbd_entity_features()
        deg = float(self.cond.degradation)
        self.frame_dropped = bool(deg > 0.0 and random.random() < cfg.frame_drop_prob_max * deg)
        imgs = self._grab_images()
        self.last_imgs = imgs
        self.slot_ids[:] = -1
        self.slot_kinds[:] = -1
        if self.detector is None or imgs is None:
            # perception data collection (no detector yet) or missing images: no tokens
            self.scene_vec = np.zeros((cfg.scene_dim,), np.float32)
            return np.zeros((cfg.max_entities, cfg.edge_dim), np.float32), np.zeros((cfg.max_entities,), np.float32), cfg.entity_max_dist
        lane_curv, lane_head = self._lane_features()
        prev = self._prev_lifted
        out = self._run_detector(imgs, prev)
        self._prev_used = prev
        self._prev_lifted = out["lifted"]
        dets, _, tl, _ = self._tokens_from(out, lane_curv, lane_head)
        truth = self._true_entities()
        ids = match_detections(dets, [(t[0], t[1], t[2], t[3]) for t in truth])
        tl_true_state, tl_true_dist = self._get_active_traffic_light_info()
        tl_id = self._active_light_id() if (tl is not None and tl_true_dist >= 0.0) else -1
        edges, mask, sid, skind, nearest = self._build_tokens(dets, tl, ids, lane_curv, lane_head, tl_id)
        self.slot_ids[:], self.slot_kinds[:] = sid, skind
        self.det_true += len(truth)
        self.det_found += sum(1 for i in ids if i >= 0)
        near = {int(t[0]) for t in truth if math.hypot(t[2], t[3]) <= 25.0}
        self.det_true_near += len(near)
        self.det_found_near += sum(1 for i in ids if int(i) in near)
        self.scene_vec = out["scene"][0].detach().cpu().numpy().astype(np.float32)
        return edges, mask, nearest

    def tokens_with_occlusion(self, eid: int) -> Optional[Tuple[np.ndarray, np.ndarray]]:
        """Occlusion intervention of Eq. (23): paint an occluder over the projected 3D box of actor
        `eid` in every camera image and run the frozen detector again."""
        if str(self.cfg.perception_mode) == "gt":
            edges, mask, _ = self._gt_entity_features(drop_id=int(eid))   # the actor is not observed
            return edges, mask
        if self.detector is None or self.last_imgs is None:
            return None
        actor = None
        for a in list(self.npcs) + list(self.props) + list(self._nearby_walkers(self.vehicle.get_location())):
            if int(a.id) == int(eid):
                actor = a
                break
        if actor is None:
            return None
        try:
            verts = actor.bounding_box.get_world_vertices(actor.get_transform())
        except BaseException:
            return None
        o, fwd, right, _ = self._ego_frame()
        ego_z = float(self.vehicle.get_location().z)
        pts = np.array([[float(np.dot(np.array([v.x, v.y]) - o, fwd)), float(np.dot(np.array([v.x, v.y]) - o, right)),
                         float(v.z) - ego_z] for v in verts], np.float64)
        imgs = self.last_imgs.copy()
        for c in range(len(CAMERA_LAYOUT)):
            u, v, ok = project_points(self.cfg, pts, c)
            if ok.sum() == 0:
                continue
            u0, u1 = int(max(0, u[ok].min() - 2)), int(min(self.cfg.cam_width, u[ok].max() + 3))
            v0, v1 = int(max(0, v[ok].min() - 2)), int(min(self.cfg.cam_height, v[ok].max() + 3))
            imgs[c, v0:v1, u0:u1] = 128
        lane_curv, lane_head = self._lane_features()
        if str(self.cfg.perception_mode) == "rgbd":
            if self.last_depth is None or self.frame_dropped:
                return None
            dets_p, _, _ = self._rgbd_run(imgs, self.last_depth)
            tracks = self.tracker.occluded_outputs(dets_p)
            edges, mask, _, _, _ = self._build_tokens(tracks, getattr(self, "_rgbd_tl", None), [-1] * len(tracks),
                                                      lane_curv, lane_head, -1)
            return edges, mask
        out = self._run_detector(imgs, self._prev_used)
        dets, _, tl, _ = self._tokens_from(out, lane_curv, lane_head)
        edges, mask, _, _, _ = self._build_tokens(dets, tl, [-1] * len(dets), lane_curv, lane_head, -1)
        return edges, mask

    def _active_light_id(self) -> int:
        try:
            light = self.vehicle.get_traffic_light() if self.vehicle is not None else None
            if light is not None:
                return int(light.id)
        except BaseException:
            pass
        loc = self.vehicle.get_location()
        s_ego = self._s_ego(loc)
        best = (-1, 1e9)
        for tl, s_stop in self.route_lights:
            if 0.0 <= s_stop - s_ego < best[1]:
                best = (int(tl.id), s_stop - s_ego)
        return int(best[0]) if best[0] >= 0 else -2

    def _get_obs(self) -> Dict[str, np.ndarray]:
        obs = super()._get_obs()
        obs["scene"] = self.scene_vec.copy()
        return obs

    # ---------------- scripted suites (Sec. 4.3) ----------------
    def _find_route_lights(self) -> List[Tuple[carla.Actor, float]]:
        out: List[Tuple[carla.Actor, float]] = []
        try:
            lights = list(self.world.get_actors().filter("traffic.traffic_light*"))
        except BaseException:
            return out
        for tl in lights:
            try:
                stops = tl.get_stop_waypoints() or []
            except BaseException:
                continue
            best: Optional[float] = None
            for swp in stops:
                s_arc, d, idx = self._route_projection_global(swp.transform.location)
                if d > 2.0 or s_arc < 1.0 or s_arc > self.route_total_len_m - 5.0:
                    continue
                i2 = min(idx + 1, len(self.route_xy) - 1)
                seg = np.array(self.route_xy[i2]) - np.array(self.route_xy[idx])
                yaw_s = math.radians(swp.transform.rotation.yaw)
                if np.linalg.norm(seg) > 1e-3:
                    seg = seg / np.linalg.norm(seg)
                    if float(np.dot(seg, [math.cos(yaw_s), math.sin(yaw_s)])) < 0.6:
                        continue
                best = s_arc if best is None else min(best, s_arc)
            if best is not None:
                out.append((tl, float(best)))
        out.sort(key=lambda t: t[1])
        return out

    def _route_point(self, s: float) -> Tuple[np.ndarray, float]:
        cum = np.asarray(self.route_cumdist, dtype=np.float32)
        i = int(np.clip(np.searchsorted(cum, s) - 1, 0, len(self.route_xy) - 2))
        a, b = np.array(self.route_xy[i]), np.array(self.route_xy[i + 1])
        seg = max(float(cum[i + 1] - cum[i]), 1e-3)
        t = float(np.clip((s - cum[i]) / seg, 0.0, 1.0))
        p = a + t * (b - a)
        yaw = math.atan2(float(b[1] - a[1]), float(b[0] - a[0]))
        return p, yaw

    def _spawn_prop_cluster(self, s: float) -> None:
        bp = None
        for name in PROP_BLUEPRINTS:
            try:
                bp = self.bp_lib.find(name)
                break
            except BaseException:
                continue
        if bp is None:
            return
        lane_w = 3.5
        try:
            p0, _ = self._route_point(s)
            wp = self.map.get_waypoint(carla.Location(float(p0[0]), float(p0[1]), 0.0))
            lane_w = max(float(getattr(wp, "lane_width", 3.5)), 3.0)
        except BaseException:
            pass
        for k in range(3):
            p, yaw = self._route_point(s + 1.5 * k)
            right = np.array([-math.sin(yaw), math.cos(yaw)])
            # Beyond the outer edge of the rightmost driving lane (not in a neighbouring lane), +1.0 m.
            off = 0.5 * lane_w
            try:
                wp = self.map.get_waypoint(carla.Location(float(p[0]), float(p[1]), 0.0))
                off = 0.5 * float(wp.lane_width)
                r = wp.get_right_lane()
                for _ in range(4):
                    if r is None or r.lane_type != carla.LaneType.Driving or np.sign(r.lane_id) != np.sign(wp.lane_id):
                        break
                    off += float(r.lane_width)
                    r = r.get_right_lane()
            except BaseException:
                pass
            q = p + right * (off + 1.0)
            z = 0.2
            try:
                z = self.map.get_waypoint(carla.Location(float(p[0]), float(p[1]), 0.0)).transform.location.z + 0.1
            except BaseException:
                pass
            if bp.has_attribute("role_name"):
                bp.set_attribute("role_name", f"{self.owner_tag}_prop")
            actor = self.world.try_spawn_actor(bp, carla.Transform(carla.Location(float(q[0]), float(q[1]), float(z)),
                                                                   carla.Rotation(yaw=math.degrees(yaw))))
            if actor is not None:
                try:
                    actor.set_simulate_physics(False)   # traffic cannot knock the cones into the lane
                except BaseException:
                    pass
                self.props.append(actor)
                self.role_by_id[int(actor.id)] = "distractor"

    def _force_light(self, tl: carla.Actor, red: bool, s_stop: float = 0.0) -> None:
        """Register a scripted light. TrafficLight.freeze() stops every light in the map, so scripted
        lights are instead held at their state while the ego vehicle approaches (see _manage_lights).
        A scripted red light turns green after the ego vehicle has waited, so no episode can block."""
        self.forced_lights.append(tl)
        self.managed.append({"tl": tl, "s": float(s_stop), "red": bool(red), "phase": "idle", "t0": 0, "wait": 0})
        self.role_by_id[int(tl.id)] = "cause" if red else "other"

    def _manage_lights(self) -> None:
        if not self.managed or self.vehicle is None:
            return
        try:
            s_ego = self._s_ego(self.vehicle.get_location())
            v = self.vehicle.get_velocity()
            speed = math.hypot(v.x, v.y)
        except BaseException:
            return
        for m in self.managed:
            if m["phase"] == "released":
                continue
            ahead = m["s"] - s_ego
            if m["phase"] == "idle" and ahead <= 40.0:
                m["phase"], m["t0"] = "active", self.episode_steps
            if m["phase"] != "active":
                continue
            if ahead < -3.0:
                m["phase"] = "released"
                continue
            want_red = m["red"]
            if want_red:
                if speed < 0.5 and ahead < 15.0:
                    m["wait"] += 1
                held = self.episode_steps - m["t0"]
                if m["wait"] >= int(round(3.0 / self.cfg.dt)) or held >= int(round(20.0 / self.cfg.dt)):
                    want_red = False
                    m["red"] = False
            try:
                st = carla.TrafficLightState.Red if want_red else carla.TrafficLightState.Green
                if m["tl"].get_state() != st:
                    m["tl"].set_state(st)
                if not want_red:
                    # Forcing one light green while the junction's controller turns a crossing
                    # direction green lets both streams enter and lock the junction. Every other
                    # light of the same junction is held red while the scripted light is green, but
                    # never the light that controls the ego and never longer than 20 s: before, a
                    # held light could stay red until the ego passed the scripted light, which it
                    # never did while waiting at the held one (episodes "blocked" at a red light).
                    m.setdefault("green_t0", self.episode_steps)
                    if self.episode_steps - int(m["green_t0"]) <= int(round(20.0 / self.cfg.dt)):
                        if "group" not in m:
                            m["group"] = [g for g in m["tl"].get_group_traffic_lights() if int(g.id) != int(m["tl"].id)]
                        ego_tl = int(self._active_light_id())
                        for g in m["group"]:
                            if int(g.id) == ego_tl:
                                continue
                            if g.get_state() != carla.TrafficLightState.Red:
                                g.set_state(carla.TrafficLightState.Red)
            except BaseException:
                m["phase"] = "released"

    def _spawn_green_queue(self, tl: carla.Actor) -> None:
        try:
            stops = tl.get_stop_waypoints() or []
        except BaseException:
            return
        vehicle_bps = [b for b in self.bp_lib.filter("vehicle.*")
                        if b.has_attribute("number_of_wheels") and b.get_attribute("number_of_wheels").as_int() == 4]
        if not vehicle_bps:
            return
        for swp in stops:
            s_arc, d, _ = self._route_projection_global(swp.transform.location)
            if d > 2.0:
                continue
            for side in ("left", "right"):
                try:
                    adj = swp.get_left_lane() if side == "left" else swp.get_right_lane()
                except BaseException:
                    adj = None
                if adj is None or adj.lane_type != carla.LaneType.Driving or np.sign(adj.lane_id) != np.sign(swp.lane_id):
                    continue
                for back in (7.0, 14.0):
                    prev = adj.previous(back)
                    if not prev:
                        continue
                    tf = prev[0].transform
                    tf = carla.Transform(carla.Location(tf.location.x, tf.location.y, tf.location.z + 0.3), tf.rotation)
                    bp = random.choice(vehicle_bps)
                    if bp.has_attribute("role_name"):
                        bp.set_attribute("role_name", self.npc_role)
                    veh = self.world.try_spawn_actor(bp, tf)
                    if veh is None:
                        continue
                    try:
                        veh.apply_control(carla.VehicleControl(throttle=0.0, brake=1.0, hand_brake=True))
                    except BaseException:
                        pass
                    self.queue_vehicles.append(veh)
                    self.npcs.append(veh)
                    self.role_by_id[int(veh.id)] = "distractor"
                return

    def _setup_scenario(self) -> None:
        self.route_lights = self._find_route_lights()
        # Every light on the route is seen by the teacher; only lights at least 25 m along the route are
        # scripted (held red/green), so the barrels 20 m before a forced stop fit on the route.
        scripted = [(tl, st) for tl, st in self.route_lights if st >= 25.0]
        mode_b, mode_q = self.cond.barrel_mode, self.cond.queue_mode
        if mode_b == "coupled":
            # Source domain of the barrel suite: barrels about 20 m before every forced stop.
            for tl, s_stop in scripted:
                stop = random.random() < 0.5
                self._force_light(tl, red=stop, s_stop=s_stop)
                if stop:
                    self._spawn_prop_cluster(max(5.0, s_stop - 20.0))
                    self.scenario_events.append({"type": "red_stop", "s": s_stop, "id": int(tl.id), "barrels": True})
                else:
                    self.scenario_events.append({"type": "green", "s": s_stop, "id": int(tl.id), "barrels": False})
        elif mode_b == "random":
            # Target domain of the barrel suite: barrels unrelated to stops, stops without barrels.
            stops = [s for _, s in scripted]
            for tl, s_stop in scripted:
                red = random.random() < 0.5
                self._force_light(tl, red=red, s_stop=s_stop)
                self.scenario_events.append({"type": "red_stop" if red else "green", "s": s_stop,
                                             "id": int(tl.id), "barrels": False})
            n_clusters = max(2, len(scripted))
            tries = 0
            while n_clusters > 0 and tries < 50:
                tries += 1
                s = random.uniform(30.0, max(31.0, self.route_total_len_m - 30.0))
                if any(-5.0 < st - s < 35.0 for st in stops):
                    continue
                self._spawn_prop_cluster(s)
                self.scenario_events.append({"type": "barrel", "s": s, "id": -1, "barrels": True})
                n_clusters -= 1
        if mode_q == "green_queue":
            for tl, s_stop in scripted:
                if any(ev.get("id") == int(tl.id) for ev in self.scenario_events):
                    continue
                self._force_light(tl, red=False, s_stop=s_stop)
                self._spawn_green_queue(tl)
                self.scenario_events.append({"type": "green_queue", "s": s_stop, "id": int(tl.id), "barrels": False})
        if self.props or self.queue_vehicles:
            self._safe_tick(label="scenario_settle", raise_on_fail=False)

    def reset(self, *, seed: Optional[int] = None, options: Optional[Dict] = None):
        if self.props:
            # The lists are cleared below before super().reset() runs the cleanup, so the previous
            # episode's barrels must be destroyed here (they used to pile up on the server).
            try:
                self._batch_destroy_ids(list(self.props), label="props")
            except BaseException:
                pass
        self.props, self.queue_vehicles, self.forced_lights, self.managed = [], [], [], []
        self._jam_track, self.jam_removed, self._ego_still, self._red_still = {}, 0, 0, 0
        self.scenario_events, self.role_by_id, self.infractions = [], {}, []
        self.det_true = self.det_found = 0
        self.det_true_near = self.det_found_near = 0
        opts = dict(options or {})
        opts.setdefault("npc_count", int(self.cond.npc))
        opts.setdefault("walker_count", int(self.cond.walkers))
        opts.setdefault("goal_index", -1)
        try:
            self.world.freeze_all_traffic_lights(False)   # undo any freeze left by an earlier crashed run
        except BaseException:
            pass
        obs, info = super().reset(seed=seed, options=opts)
        self._setup_scenario()
        obs = self._get_obs()
        info["conditions"] = self.cond.key()
        info["events"] = list(self.scenario_events)
        return obs, info

    # ---------------- traffic jam watchdog ----------------
    def _remove_npcs(self, actors: List[carla.Actor]) -> None:
        cmds = []
        for a in actors:
            try:
                cmds.append(carla.command.SetAutopilot(int(a.id), False, int(self.tm.get_port())))
                cmds.append(carla.command.DestroyActor(int(a.id)))
            except BaseException:
                continue
        if cmds:
            try:
                self.client.apply_batch(cmds)       # applied at the next tick, no extra tick
            except BaseException:
                pass

    def _jam_watchdog(self) -> None:
        """Remove traffic that can no longer move, so neither the NPCs behind it nor the ego wait forever.

        CARLA's Traffic Manager has no deadlock resolution: two vehicles that both yield inside a
        junction, a vehicle stuck after a collision, or a pedestrian that stopped on the road stay
        there for the rest of the episode. Every second, an NPC vehicle that has not moved 1 m for
        20 s inside a junction, or for 60 s elsewhere while not waiting at a red light, is removed;
        a pedestrian standing still on a driving lane for 20 s is removed as well."""
        period = int(round(1.0 / self.cfg.dt))
        if self.episode_steps % period != 0:
            return
        t = self.episode_steps
        jam = int(round(20.0 / self.cfg.dt))
        long_stop = int(round(60.0 / self.cfg.dt))
        drop: List[carla.Actor] = []
        scripted = {int(q.id) for q in self.queue_vehicles}
        for npc in list(self.npcs):
            try:
                if int(npc.id) in scripted or not self._actor_is_alive(npc):
                    continue
                l = npc.get_location()
                pos, since = self._jam_track.get(int(npc.id), (None, t))
                if pos is None or math.hypot(l.x - pos[0], l.y - pos[1]) > 1.0:
                    self._jam_track[int(npc.id)] = ((l.x, l.y), t)
                    continue
                stopped = t - since
                if stopped < jam:
                    continue
                wp = self.map.get_waypoint(l, project_to_road=True, lane_type=carla.LaneType.Driving)
                in_junction = bool(wp is not None and wp.is_junction)
                at_red = False
                try:
                    at_red = bool(npc.is_at_traffic_light()) and npc.get_traffic_light_state() == carla.TrafficLightState.Red
                except BaseException:
                    pass
                if (in_junction and not at_red) or (stopped >= long_stop and not at_red):
                    drop.append(npc)
            except BaseException:
                continue
        if drop:
            ids = {int(a.id) for a in drop}
            self.npcs = [n for n in self.npcs if int(n.id) not in ids]
            for i in ids:
                self._jam_track.pop(i, None)
            self._remove_npcs(drop)
            self.jam_removed += len(drop)
        # pedestrians standing still on a driving lane
        for w, c in list(zip(self.walkers, self.walker_controllers)):
            try:
                if not self._actor_is_alive(w):
                    continue
                l = w.get_location()
                key = -int(w.id)
                pos, since = self._jam_track.get(key, (None, t))
                if pos is None or math.hypot(l.x - pos[0], l.y - pos[1]) > 0.5:
                    self._jam_track[key] = ((l.x, l.y), t)
                    continue
                if t - since < jam:
                    continue
                on_road = self.map.get_waypoint(l, project_to_road=False, lane_type=carla.LaneType.Driving) is not None
                if on_road:
                    try:
                        c.stop()
                    except BaseException:
                        pass
                    self.client.apply_batch([carla.command.DestroyActor(int(c.id)), carla.command.DestroyActor(int(w.id))])
                    self.walkers = [x for x in self.walkers if int(x.id) != int(w.id)]
                    self.walker_controllers = [x for x in self.walker_controllers if int(x.id) != int(c.id)]
                    self._jam_track.pop(key, None)
                    self.jam_removed += 1
            except BaseException:
                continue

    def step(self, action: np.ndarray):
        self._manage_lights()
        self._jam_watchdog()
        obs, reward, terminated, truncated, info = super().step(action)
        # The ego cannot move for 90 s (whatever the reason): end the episode instead of waiting for
        # the 6000-step timeout. Red-light waits are far shorter than 90 s.
        try:
            v = self.vehicle.get_velocity() if self.vehicle is not None else None
            slow = v is not None and 3.6 * math.hypot(v.x, v.y) < 1.0
        except BaseException:
            slow = False
        self._ego_still = self._ego_still + 1 if slow else 0
        if slow and self._ego_still % 20 == 0:
            try:
                self._red_still = (self._red_still + 20) if self._get_active_traffic_light_info()[0] == "Red" else 0
            except BaseException:
                pass
        elif not slow:
            self._red_still = 0
        if not (terminated or truncated) and self._ego_still >= int(round(90.0 / self.cfg.dt)):
            truncated = True
            if safe_float(info.get("goal_dist", 1e9), 1e9) <= 15.0 and safe_float(info.get("route_completion_pct", 0.0)) >= 95.0:
                # Blocked within 15 m of the end of the route (the goal spot is a kerb-side spawn point,
                # often next to parked cars): the route is driven, count it as reached. It is a terminal
                # transition like any goal, so the critic does not bootstrap past it.
                info["term_reason"] = "goal"
                info["success"] = True
                info["goal_reached"] = True
                terminated, truncated = True, False
                reward = build_reward_from_info(info, safe_float(info.get("sigma_proxy", 0.0)), cfg=self.cfg)
                info["env_reward_proxy"] = float(reward)
            else:
                info["term_reason"] = "blocked"
                info["timeout"] = True
            self._begin_episode_teardown(str(info["term_reason"]))
        info["jam_removed"] = int(self.jam_removed)
        # Leaderboard-style infractions for the infraction score (Sec. 4.1).
        if bool(info.get("collision", False)) and self.collision_events:
            other = ""
            try:
                other = str(self.collision_events[-1].other_actor.type_id)
            except BaseException:
                pass
            kind = "collision_walker" if "walker" in other else ("collision_vehicle" if "vehicle" in other else "collision_static")
            self.infractions.append((kind, float(self.distance_driven_m)))
            info["collision_kind"] = kind
            info["collision_where"] = getattr(self, "_collision_where_first", "") or self._collision_where()
        for event in info.get("red_light_events", []):
            self.infractions.append(("red_light", float(event["distance_m"])))
        info["frame_dropped"] = bool(self.frame_dropped)
        return obs, reward, terminated, truncated, info

    def _collision_where(self) -> str:
        """Where the other actor touched the ego (front: the ego drove into it; rear: it drove into the
        ego), with both speeds, so a collision can be attributed to the driver or to the traffic."""
        try:
            ev = self.collision_events[-1]
            other = ev.other_actor
            tf = self.vehicle.get_transform()
            yaw = math.radians(tf.rotation.yaw)
            ol = other.get_location()
            rx, ry = ol.x - tf.location.x, ol.y - tf.location.y
            lon = rx * math.cos(yaw) + ry * math.sin(yaw)
            lat = -rx * math.sin(yaw) + ry * math.cos(yaw)
            side = "front" if lon > 1.5 else ("rear" if lon < -1.5 else ("right" if lat > 0 else "left"))
            ov = other.get_velocity()
            off = ""
            try:
                _, d_ego, _ = self._route_projection_global(tf.location)
                _, d_obj, _ = self._route_projection_global(ol)
                off = f",ego_off_route={d_ego:.1f}m,obj_off_route={d_obj:.1f}m"
            except BaseException:
                pass
            return (f"{side},{str(other.type_id)},ego={vec3_length(self.vehicle.get_velocity()):.1f}m/s,"
                    f"other={math.hypot(ov.x, ov.y):.1f}m/s{off}")
        except BaseException:
            return "unknown"

    def _cleanup_episode_actors(self) -> None:
        self._collision_where_first = ""
        self.forced_lights, self.managed = [], []
        if self.props:
            try:
                self._batch_destroy_ids(list(self.props), label="props")
            except BaseException:
                pass
        self.props = []
        self.queue_vehicles = []
        super()._cleanup_episode_actors()

    # ---------------- privileged scene (Sec. 3.4) ----------------
    def slot_info(self) -> Tuple[np.ndarray, np.ndarray]:
        return self.slot_ids.copy(), self.slot_kinds.copy()

    def true_positions(self) -> Dict[int, Tuple[float, float]]:
        out: Dict[int, Tuple[float, float]] = {}
        for actor in list(self.npcs) + list(self.props) + list(self.walkers):
            if self._actor_is_alive(actor):
                try:
                    l = actor.get_location()
                    out[int(actor.id)] = (float(l.x), float(l.y))
                except BaseException:
                    continue
        return out

    def ego_pose(self) -> Tuple[float, float, float]:
        tf = self.vehicle.get_transform()
        return float(tf.location.x), float(tf.location.y), float(tf.rotation.yaw)

    def privileged_scene(self, ahead_m: float = 120.0) -> PrivScene:
        assert self.vehicle is not None
        # One scene per simulation step: the reward, the teacher label, and the labeler all use it.
        key = (int(self.vehicle.id), int(self.episode_steps), float(ahead_m))
        cached = getattr(self, "_scene_cache", None)
        if cached is not None and cached[0] == key:
            return cached[1]
        scene = self._build_privileged_scene(ahead_m)
        self._scene_cache = (key, scene)
        return scene

    def _compute_reward_components(self, applied_action: np.ndarray) -> Tuple[float, float, float, Dict[str, float]]:
        """Hazard-aware progress and proximity terms (deviation register, item 23).

        r_p = 1 - |v_route - v_des| / v_0 in [-1, 1]: full reward for driving at the safe reference
        speed, which is 0 at a red light or behind a stopped car, so waiting where waiting is
        required earns the same as driving, and stopping on a free road earns nothing.
        psi_P = violated fraction of the IDM safe following distance (replaces exp(-d/tau_p), which
        penalised every car within about 12 m, including a correct queue and parked cars).
        The paper weights w_s, w_p, w_c, w_u and the terms psi_L, rho, r_c are unchanged."""
        rs, rp, rc, comp = super()._compute_reward_components(applied_action)
        self._s_now, self._s_now_step = safe_float(comp.get("route_s", 0.0)), int(self.episode_steps)
        # Comfort: CARLA's get_acceleration() jitters by several m/s^2 between 20 Hz frames, so the jerk
        # of the base term reached hundreds of m/s^3 and r_c sat at its -5 floor (-0.75 per step after
        # w_c) whenever the car moved. Jerk now comes from an exponentially smoothed derivative of the
        # speed, and r_c is bounded at -1.
        try:
            v_now = float(vec3_length(self.vehicle.get_velocity()))
            dt = max(float(self.cfg.dt), 1e-3)
            if int(self.episode_steps) <= 1 or getattr(self, "_rc_v", None) is None:
                self._rc_v, self._rc_a = v_now, 0.0
            a_now = 0.7 * float(self._rc_a) + 0.3 * (v_now - float(self._rc_v)) / dt
            jerk = (a_now - float(self._rc_a)) / dt
            self._rc_v, self._rc_a = v_now, a_now
            steer_rate = safe_float(comp.get("steer_rate", 0.0))
            rc = float(np.clip(-(self.cfg.k_j * jerk ** 2 + self.cfg.k_delta * steer_rate ** 2), -1.0, 0.0))
            comp.update({"jerk": float(jerk), "rc_raw": float(rc)})
        except BaseException:
            rc = float(np.clip(rc, -1.0, 0.0))
        try:
            scene = self.privileged_scene()
            hz = expert_hazard(scene, self.cfg)
            vel = self.vehicle.get_velocity()
            _, _, head = route_frame(scene, np.array([scene.ego_x]), np.array([scene.ego_y]))
            v_route = float(vel.x * math.cos(head[0]) + vel.y * math.sin(head[0]))
        except BaseException:
            return rs, rp, rc, comp
        v0 = max(float(self.cfg.expert_target_speed_ms), 0.1)
        # Too slow costs 1.5x more than too fast (speeding is already penalised by psi_P and collisions):
        # standing still on a free road gives r_p = -0.5, not 0, so stopping is never free.
        err = v_route - hz["v_des"]
        rp = float(np.clip(1.0 - (abs(err) if err >= 0.0 else float(self.cfg.rp_slow_factor) * (-err)) / v0, -1.0, 1.0))
        rs = float(np.clip(1.0 - self.cfg.k_l * safe_float(comp.get("psiL", 0.0)) - self.cfg.k_p * hz["psi_p"]
                           - self.cfg.k_r * safe_float(comp.get("rho_red", 0.0)), -2.0, 1.0))
        comp.update({"psiP": hz["psi_p"], "v_des": hz["v_des"], "v_route": v_route,
                     "hazard_gap": hz["hazard_gap"], "rp_speed": rp})
        return rs, rp, rc, comp

    def _s_ego(self, loc: carla.Location) -> float:
        """Ego arc length along the route: the monotonic projection of this step when the reward has
        computed it, else the nearest-segment projection (only before the first step)."""
        s_now = getattr(self, "_s_now", None)
        if s_now is not None and getattr(self, "_s_now_step", -1) == int(self.episode_steps):
            return float(s_now)
        return float(self._route_projection_global(loc)[0])

    def _build_privileged_scene(self, ahead_m: float = 120.0) -> PrivScene:
        tf = self.vehicle.get_transform()
        v = self.vehicle.get_velocity()
        ego_v = float(math.hypot(v.x, v.y))
        loc = tf.location
        s_ego = self._s_ego(loc)
        cum = np.asarray(self.route_cumdist, dtype=np.float32)
        xy = np.asarray(self.route_xy, dtype=np.float32)
        # The window starts at least one route point behind the ego, so a car behind the ego never
        # projects onto the first segment as if it were ahead.
        i0 = max(0, int(np.searchsorted(cum, s_ego)) - 2)
        i1 = int(np.searchsorted(cum, s_ego + ahead_m, side="right"))
        if i1 - i0 >= 2:
            route, route_s = xy[i0:i1], cum[i0:i1] - s_ego
        else:
            route, route_s = xy[-2:], np.array([0.0, 1.0], dtype=np.float32)
        lane_w = 3.5
        try:
            lane_w = max(float(self.map.get_waypoint(loc).lane_width), 3.0)
        except BaseException:
            pass
        ents: List[PrivEntity] = []

        def add(actor: carla.Actor, kind: int) -> None:
            try:
                l = actor.get_location()
                if math.hypot(l.x - loc.x, l.y - loc.y) > self.cfg.entity_max_dist + 10.0:
                    return
                vv = actor.get_velocity() if kind != KIND_STATIC else None
                yaw = math.radians(float(actor.get_transform().rotation.yaw)) if kind == KIND_VEHICLE else float("nan")
                hl, hw = KIND_RADIUS_M[kind], float("nan")
                if kind == KIND_VEHICLE:
                    try:
                        ex = actor.bounding_box.extent
                        hl, hw = max(float(ex.x), 0.5), max(float(ex.y), 0.3)
                    except BaseException:
                        pass
                ents.append(PrivEntity(int(actor.id), kind, float(l.x), float(l.y),
                                       float(vv.x) if vv else 0.0, float(vv.y) if vv else 0.0,
                                       hl, role=self.role_by_id.get(int(actor.id), "other"), yaw=yaw, half_w=hw))
            except BaseException:
                return

        # Every vehicle on the server, not only the ones this episode spawned: the teacher must see any
        # car that is physically there.
        seen = set()
        for npc in self.npcs:
            if self._actor_is_alive(npc):
                add(npc, KIND_VEHICLE)
                seen.add(int(npc.id))
        try:
            ego_id = int(self.vehicle.id)
            for veh in self._world_actors().filter("vehicle.*"):
                if int(veh.id) != ego_id and int(veh.id) not in seen:
                    add(veh, KIND_VEHICLE)
        except BaseException:
            pass
        for w in self._nearby_walkers(loc):
            add(w, KIND_WALKER)
        # Cars that are part of the map (the parked-vehicle layer) are not actors, so they were invisible.
        for k, (ox, oy, oyaw, ohl, ohw) in enumerate(self._map_parked_vehicles()):
            if math.hypot(ox - loc.x, oy - loc.y) <= self.cfg.entity_max_dist:
                ents.append(PrivEntity(-10000 - k, KIND_VEHICLE, ox, oy, 0.0, 0.0, max(ohl, 0.5),
                                       role="map_parked", yaw=oyaw, half_w=max(ohw, 0.3)))
        for p in self.props:
            if self._actor_is_alive(p):
                add(p, KIND_STATIC)
        for tl, s_stop in self.route_lights:
            rel = s_stop - s_ego
            if -8.0 <= rel <= self.cfg.entity_max_dist:
                p, _ = self._route_point(s_stop)
                try:
                    st = self._tl_state_name(tl.get_state())
                except BaseException:
                    st = "Unknown"
                role = self.role_by_id.get(int(tl.id), "cause" if st == "Red" else "other")
                ents.append(PrivEntity(int(tl.id), KIND_SIGNAL, float(p[0]), float(p[1]), 0.0, 0.0, 0.0,
                                       tl_state=st, role=role))
        return PrivScene(float(loc.x), float(loc.y), float(tf.rotation.yaw), ego_v, route.astype(np.float32),
                         route_s.astype(np.float32), lane_w, ents)

    def near_recall_counts(self) -> Tuple[int, int]:
        return int(getattr(self, "det_true_near", 0)), int(getattr(self, "det_found_near", 0))

    def detection_recall_counts(self) -> Tuple[int, int]:
        return int(self.det_true), int(self.det_found)



# ======================================================
# Interventions, privileged expert, and null calibration (Sec. 3.4)
# ======================================================
OP_REMOVE, OP_FREEZE, OP_SHIFT, OP_SWITCH = 0, 1, 2, 3
PHYSICAL_OPS = {KIND_VEHICLE: (OP_REMOVE, OP_FREEZE, OP_SHIFT), KIND_WALKER: (OP_REMOVE, OP_FREEZE, OP_SHIFT),
                KIND_STATIC: (OP_REMOVE,), KIND_SIGNAL: (OP_SWITCH,)}


def route_frame(scene: PrivScene, px: np.ndarray, py: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Arc length s, signed lateral offset d (right positive), and segment heading of points on the route."""
    R = scene.route.astype(np.float64)
    if len(R) < 2:
        return np.zeros_like(px), np.zeros_like(px), np.zeros_like(px)
    a, b = R[:-1], R[1:]
    ab = b - a
    L2 = np.clip((ab ** 2).sum(1), 1e-6, None)
    p = np.stack([px, py], -1).astype(np.float64)
    t = np.clip(((p[:, None, :] - a[None]) * ab[None]).sum(-1) / L2[None], 0.0, 1.0)
    proj = a[None] + t[..., None] * ab[None]
    d2 = ((p[:, None, :] - proj) ** 2).sum(-1)
    j = np.argmin(d2, axis=1)
    n = np.arange(len(px))
    seg = np.sqrt(L2)
    s = scene.route_s[:-1][j] + t[n, j] * seg[j]
    head = np.arctan2(ab[j, 1], ab[j, 0])
    right = np.stack([-np.sin(head), np.cos(head)], -1)
    d = ((p - proj[n, j]) * right).sum(-1)
    return s, d, head


def lateral_extent(e: PrivEntity, route_head: float) -> float:
    """Half-width of an actor across the route. A car standing at an angle (turning, stuck in a
    junction) reaches much further sideways than its half-width."""
    yaw = e.yaw
    if not np.isfinite(yaw) and math.hypot(e.vx, e.vy) > 0.5:
        yaw = math.atan2(e.vy, e.vx)
    if not np.isfinite(yaw):
        return KIND_HALF_WIDTH_M[e.kind]
    rel = yaw - route_head
    hl = float(e.radius) if e.radius > 0.0 else KIND_RADIUS_M[e.kind]
    hw = float(e.half_w) if np.isfinite(e.half_w) else KIND_HALF_WIDTH_M[e.kind]
    return abs(hl * math.sin(rel)) + abs(hw * math.cos(rel))


def boxes_overlap(e: PrivEntity, s_e: float, d_e: float, route_head: float, margin: float = 0.2) -> bool:
    """Separating-axis test between the ego box (route frame origin, along the route) and the oriented
    box of an actor at (s_e, d_e); used to tell a car standing beside the ego from one in its way."""
    yaw = e.yaw if np.isfinite(e.yaw) else (math.atan2(e.vy, e.vx) if math.hypot(e.vx, e.vy) > 0.5 else route_head)
    rel = yaw - route_head
    hl = (float(e.radius) if e.radius > 0.0 else KIND_RADIUS_M[e.kind]) + margin
    hw = (float(e.half_w) if np.isfinite(e.half_w) else KIND_HALF_WIDTH_M[e.kind]) + margin
    c, sn = math.cos(rel), math.sin(rel)
    a_axes = [(1.0, 0.0), (0.0, 1.0)]
    b_axes = [(c, sn), (-sn, c)]
    for ax in a_axes + b_axes:
        pa = EGO_RADIUS_M * abs(ax[0]) + EGO_HALF_WIDTH_M * abs(ax[1])
        pb = hl * abs(ax[0] * c + ax[1] * sn) + hw * abs(-ax[0] * sn + ax[1] * c)
        if abs(s_e * ax[0] + d_e * ax[1]) > pa + pb:
            return False
    return True


def expert_lead(scene: PrivScene, ents: Sequence[PrivEntity], include_signals: bool = True,
                predict: bool = True, v_cruise: float = 5.0) -> Tuple[float, float, int]:
    """Closest obstacle in the ego corridor along the route: (gap, lead speed, entity id).

    The lateral test uses the oriented footprint of each car, so a car turning or standing at an
    angle in a junction counts as an obstacle. With predict=True, moving vehicles and pedestrians
    outside the corridor are forecast for 5 s at constant velocity (as in PDM-Lite), and the ego
    is forecast along the route accelerating at 1.5 m/s^2 up to the cruise speed (at least 3 m/s).
    An actor that would be inside the corridor where the ego is at the same time (crossing traffic,
    oncoming traffic during a turn, a pedestrian stepping onto the road, a car cutting in) becomes
    an obstacle with speed 0, so the IDM yields to it."""
    corridor = 0.5 * scene.lane_width
    best = (1e9, 0.0, -1)
    if not ents:
        return best
    xs = np.array([e.x for e in ents])
    ys = np.array([e.y for e in ents])
    s, d, head = route_frame(scene, xs, ys)
    movers: List[int] = []
    for k, e in enumerate(ents):
        if e.kind == KIND_SIGNAL:
            if include_signals and e.tl_state in ("Red", "Yellow") and 0.5 < s[k] < 50.0:
                gap = float(s[k]) - 1.0
                if gap < best[0]:
                    best = (gap, 0.0, e.eid)
            continue
        if not (0.0 < s[k] < 50.0):
            continue
        vs = e.vx * math.cos(head[k]) + e.vy * math.sin(head[k])
        vd = -e.vx * math.sin(head[k]) + e.vy * math.cos(head[k])
        d_now = float(d[k])
        # the 1.5 s lateral look-ahead is a forecast: "current obstacle" queries (predict=False) use d_now only
        d_fut = d_now + 1.5 * float(vd) if predict else d_now
        lat = 0.0 if d_now * d_fut < 0.0 else min(abs(d_now), abs(d_fut))
        moving = math.hypot(e.vx, e.vy) > 0.5
        ext = lateral_extent(e, float(head[k]))
        # Moving actors: half a lane plus their extent. Standing actors (parked cars, pedestrians at the
        # kerb, barrels): the ego's own half-width plus their extent plus 0.4 m, so the teacher passes a
        # car parked at the kerb instead of waiting behind it forever.
        half = corridor if moving else min(corridor, EGO_HALF_WIDTH_M + 0.4)
        if e.role == "map_parked" and abs(d_now) >= corridor:
            continue                       # parked map cars block only when they stand inside the lane
        if float(s[k]) - e.radius - EGO_RADIUS_M < 0.0 and not boxes_overlap(e, float(s[k]), d_now, float(head[k])):
            continue                       # beside the ego without overlap: it cannot block the way ahead
        if lat < half + ext:
            gap = float(s[k]) - e.radius - EGO_RADIUS_M
            if gap < best[0]:
                best = (gap, max(float(vs), 0.0), e.eid)
        elif predict and e.kind in (KIND_VEHICLE, KIND_WALKER) and s[k] > 0.5 and math.hypot(e.vx, e.vy) > 0.5:
            movers.append(k)
    if movers:
        T = np.arange(1, 26, dtype=np.float64) * 0.2                      # 0.2 ... 5.0 s
        v_top = max(float(v_cruise), 3.0)
        v_t = np.minimum(float(scene.ego_v) + 1.5 * T, v_top)
        s_ego = np.cumsum(np.maximum(v_t, 3.0 if scene.ego_v > 0.5 else v_t) * 0.2)
        mx = np.array([[ents[k].x + ents[k].vx * t for t in T] for k in movers])
        my = np.array([[ents[k].y + ents[k].vy * t for t in T] for k in movers])
        sp_, dp_, hp_ = route_frame(scene, mx.ravel(), my.ravel())
        sp_, dp_, hp_ = sp_.reshape(mx.shape), dp_.reshape(mx.shape), hp_.reshape(mx.shape)
        for r, k in enumerate(movers):
            e = ents[k]
            ext = np.array([lateral_extent(e, float(h)) for h in hp_[r]])
            inside = np.abs(dp_[r]) < corridor + ext
            meet = np.abs(sp_[r] - s_ego) < EGO_RADIUS_M + KIND_RADIUS_M[e.kind] + 2.0
            hit = np.nonzero(inside & meet & (sp_[r] > 0.5))[0]
            if hit.size:
                gap = float(sp_[r, hit[0]]) - KIND_RADIUS_M[e.kind] - EGO_RADIUS_M - 1.0
                if gap < best[0]:
                    best = (gap, 0.0, e.eid)
    return best


def expert_hazard(scene: PrivScene, cfg: Config) -> Dict[str, float]:
    """Hazard-aware reference quantities used by the reward (deviation register, item 23).

    v_des: the teacher's IDM following speed for the nearest obstacle (vehicle, pedestrian,
      predicted conflict, red light), i.e. the speed v with s*(v, v_lead) = gap and a 2.5 m
      standstill distance, capped at the cruise speed: 0 at a red stop line or behind a stopped car.
    psi_p: the fraction of the IDM safe following distance s* = s0 + v T + v (v - v_l)/(2 sqrt(ab))
      that is violated by the nearest physical obstacle (0 when keeping a safe distance or when
      standing in a queue, 1 when touching)."""
    v = float(scene.ego_v)
    v0 = float(cfg.expert_target_speed_ms)
    gap, v_lead, _ = expert_lead(scene, scene.entities, include_signals=True, v_cruise=v0)
    if gap >= 1e8:
        v_des = v0
    else:
        # The speed at which the IDM gap s*(v) of the teacher equals the current gap (3 m standstill
        # distance), i.e. the teacher's own following speed; 0 once the gap is 3 m or less.
        c = 1.0 / (2.0 * math.sqrt(1.5 * 3.0))
        b_lin = 1.2 - c * v_lead
        disc = b_lin * b_lin + 4.0 * c * max(gap - 3.0, 0.0)
        v_des = float(np.clip((-b_lin + math.sqrt(max(disc, 0.0))) / (2.0 * c), 0.0, v0))
        if v_lead < 0.5 and gap < 6.0:
            v_des = 0.0          # the teacher holds 3-6 m behind a standing obstacle
    gap_o, v_lead_o, _ = expert_lead(scene, scene.entities, include_signals=False, v_cruise=v0)
    psi_p = 0.0
    if gap_o < 1e8:
        s_star = max(2.0, 2.0 + v * 1.2 + v * (v - v_lead_o) / (2.0 * math.sqrt(1.5 * 3.0)))
        psi_p = float(np.clip((s_star - gap_o) / s_star, 0.0, 1.0))
    return {"v_des": float(v_des), "hazard_gap": float(min(gap, 1e3)), "psi_p": psi_p}


def expert_action(scene: PrivScene, ents: Sequence[PrivEntity], cfg: Config) -> np.ndarray:
    """Privileged IDM expert in the style of PDM-Lite, which obtains its target speed with the
    Intelligent Driver Model from the leading vehicle, pedestrian, or traffic light. Steering
    follows the route and is identical under every intervention, so only throttle and brake
    enter Eq. (21). The executed driving teacher additionally respects route-guidance speed
    limits in expert_drive_action; this analytic counterfactual reference is unchanged."""
    v = scene.ego_v
    v0 = float(cfg.expert_target_speed_ms)
    a_max, b_comf, T_head = 1.5, 3.0, 1.2
    gap, v_lead, _ = expert_lead(scene, ents, v_cruise=v0)
    free = a_max * (1.0 - (v / max(v0, 0.1)) ** 4)
    s0 = 3.0
    if gap < 1e8:
        if v < 0.3 and v_lead < 0.5 and gap < s0 + 3.0:
            # Standing behind a standing obstacle (or at a red line): hold, do not creep up to it. A
            # creeping teacher taught the policy to creep, and with imperfect perception it crept into
            # the car in front (collisions at 0-1 m/s).
            return np.array([0.0, 0.4, 0.0], dtype=np.float32)
        s_star = s0 + v * T_head + v * (v - v_lead) / (2.0 * math.sqrt(a_max * b_comf))
        acc = free - a_max * (max(s_star, 0.0) / max(gap, 0.1)) ** 2
    else:
        acc = free
    thr = float(np.clip(acc / 2.5, 0.0, 1.0)) if acc > 0.0 else 0.0
    brk = float(np.clip(-acc / 6.0, 0.0, 1.0)) if acc < 0.0 else 0.0
    return np.array([thr, brk, 0.0], dtype=np.float32)


def rollout_margin(scene: PrivScene, ents: Sequence[PrivEntity], actions: np.ndarray, cfg: Config) -> float:
    """m_H of Eq. (20): minimum of phi(s) over H steps with the ego actions held fixed."""
    dt = float(cfg.dt)
    x, y, yaw, v = scene.ego_x, scene.ego_y, math.radians(scene.ego_yaw_deg), scene.ego_v
    dyn = [e for e in ents if e.kind != KIND_SIGNAL]
    sig = [e for e in ents if e.kind == KIND_SIGNAL and e.tl_state == "Red"]
    sig_s = route_frame(scene, np.array([e.x for e in sig]), np.array([e.y for e in sig]))[0] if sig else np.zeros(0)
    P0 = np.array([[e.x, e.y] for e in dyn], dtype=np.float64).reshape(-1, 2)
    V = np.array([[e.vx, e.vy] for e in dyn], dtype=np.float64).reshape(-1, 2)
    Lh = np.array([KIND_RADIUS_M[e.kind] for e in dyn], dtype=np.float64) + EGO_RADIUS_M
    Wh = np.array([KIND_HALF_WIDTH_M[e.kind] for e in dyn], dtype=np.float64) + EGO_HALF_WIDTH_M
    s_trav = 0.0
    red_cross = False
    phi_min = 1.0
    T0, d0 = float(cfg.ttc_norm_s), float(cfg.clearance_norm_m)
    for j in range(int(actions.shape[0]) + 1):
        if j > 0:
            thr, brk, steer = [float(u) for u in actions[j - 1]]
            acc = 2.5 * thr - 6.0 * brk - 0.05 * v
            v = max(0.0, v + acc * dt)
            yaw += v / 2.9 * math.tan(float(np.clip(steer, -1.0, 1.0)) * 0.7) * dt
            x += v * math.cos(yaw) * dt
            y += v * math.sin(yaw) * dt
            s_prev = s_trav
            s_trav += v * dt
            if sig_s.size and np.any((sig_s > s_prev) & (sig_s <= s_trav)):
                red_cross = True
        if red_cross:
            return 0.0
        if P0.shape[0] == 0:
            continue
        pos = P0 + V * (j * dt)
        rel = pos - np.array([x, y])
        dist = np.linalg.norm(rel, axis=1)
        # rectangle clearance in the ego frame (longitudinal and lateral extents)
        lon = np.abs(rel[:, 0] * math.cos(yaw) + rel[:, 1] * math.sin(yaw))
        lat = np.abs(-rel[:, 0] * math.sin(yaw) + rel[:, 1] * math.cos(yaw))
        clear = np.hypot(np.clip(lon - Lh, 0.0, None), np.clip(lat - Wh, 0.0, None))
        ev = np.array([v * math.cos(yaw), v * math.sin(yaw)])
        closing = -((rel * (V - ev[None])).sum(1)) / np.clip(dist, 1e-3, None)
        ttc = np.where(closing > 1e-3, clear / np.clip(closing, 1e-3, None), np.inf)
        phi = 0.5 * float(np.min(np.minimum(ttc / T0, 1.0))) + 0.5 * float(np.min(np.minimum(clear / d0, 1.0)))
        phi_min = min(phi_min, phi)
    return float(phi_min)


def apply_op(ents: Sequence[PrivEntity], idx: int, op: int) -> List[PrivEntity]:
    out: List[PrivEntity] = []
    for k, e in enumerate(ents):
        if k != idx:
            out.append(e)
            continue
        if op == OP_REMOVE:
            continue
        e2 = PrivEntity(**e.__dict__)
        if op == OP_FREEZE:
            e2.vx, e2.vy = 0.0, 0.0
        elif op == OP_SHIFT:
            sp = math.hypot(e.vx, e.vy)
            ux, uy = (e.vx / sp, e.vy / sp) if sp > 0.1 else (1.0, 0.0)
            e2.vx, e2.vy = e.vx + 2.0 * ux, e.vy + 2.0 * uy
        elif op == OP_SWITCH:
            e2.tl_state = "Green" if e.tl_state in ("Red", "Yellow") else "Red"
        out.append(e2)
    return out


def entity_effects(scene: PrivScene, actions: np.ndarray, cfg: Config,
                   indices: Optional[Sequence[int]] = None) -> Dict[int, Tuple[float, float]]:
    """Raw effect e_i of Eq. (21) and its margin-only part, keyed by entity id."""
    ents = scene.entities
    base_ex = expert_action(scene, ents, cfg)
    base_m = rollout_margin(scene, ents, actions, cfg)
    out: Dict[int, Tuple[float, float]] = {}
    for k in (range(len(ents)) if indices is None else indices):
        e = ents[k]
        full, marg = [], []
        for op in PHYSICAL_OPS[e.kind]:
            ents2 = apply_op(ents, k, op)
            m = rollout_margin(scene, ents2, actions, cfg)
            ex = expert_action(scene, ents2, cfg)
            dm = abs(base_m - m)
            marg.append(dm)
            full.append(float(np.abs(base_ex - ex).sum()) + dm)
        out[e.eid] = (float(np.mean(full)), float(np.mean(marg)))
    return out


def null_probe_effects(scene: PrivScene, actions: np.ndarray, cfg: Config, n_probes: int,
                       far: bool = False) -> List[Tuple[float, float, float]]:
    """Null effects of Eq. (22): probes off the drivable surface at distances matched to real
    entities (or far away for the ablation). Returns (distance, full effect, margin-only effect)."""
    out: List[Tuple[float, float, float]] = []
    base_ex = expert_action(scene, scene.entities, cfg)
    base_m = rollout_margin(scene, scene.entities, actions, cfg)
    max_s = float(scene.route_s[-1]) if len(scene.route_s) else 0.0
    for _ in range(int(n_probes)):
        if far:
            ang = random.uniform(0.0, 2.0 * math.pi)
            r = random.uniform(150.0, 200.0)
            px, py = scene.ego_x + r * math.cos(ang), scene.ego_y + r * math.sin(ang)
        else:
            dist = random.uniform(1.0, cfg.null_bins * cfg.null_bin_width_m)
            s_target = float(np.clip(dist, 0.0, max(max_s - 1.0, 0.0)))
            j = int(np.clip(np.searchsorted(scene.route_s, s_target) - 1, 0, max(len(scene.route) - 2, 0)))
            a, b = scene.route[j], scene.route[min(j + 1, len(scene.route) - 1)]
            head = math.atan2(float(b[1] - a[1]), float(b[0] - a[0]))
            side = 1.0 if random.random() < 0.5 else -1.0
            off = side * (0.5 * scene.lane_width + 2.5)
            px = float(a[0]) - math.sin(head) * off
            py = float(a[1]) + math.cos(head) * off
        probe = PrivEntity(-999, KIND_STATIC, px, py, 0.0, 0.0, 0.5)
        ents2 = list(scene.entities) + [probe]
        m = rollout_margin(scene, ents2, actions, cfg)
        ex = expert_action(scene, ents2, cfg)
        dm = abs(base_m - m)
        out.append((math.hypot(px - scene.ego_x, py - scene.ego_y), float(np.abs(base_ex - ex).sum()) + dm, dm))
    return out


class NullBank:
    """Distance-binned null effects, Eq. (22)."""

    def __init__(self, cfg: Config):
        self.cfg = cfg
        nb = int(cfg.null_bins)
        self.full: List[List[float]] = [[] for _ in range(nb)]
        self.marg: List[List[float]] = [[] for _ in range(nb)]
        self.far_full: List[float] = []
        self.far_marg: List[float] = []
        self.cap = max(20, int(cfg.null_total) // nb)

    def bin_of(self, dist: float) -> int:
        return int(np.clip(int(dist / max(self.cfg.null_bin_width_m, 1e-3)), 0, int(self.cfg.null_bins) - 1))

    def add(self, dist: float, full: float, marg: float, far: bool = False) -> None:
        if far:
            if len(self.far_full) < self.cap * 2:
                self.far_full.append(full)
                self.far_marg.append(marg)
            return
        b = self.bin_of(dist)
        if len(self.full[b]) < self.cap:
            self.full[b].append(full)
            self.marg[b].append(marg)
        else:
            j = random.randrange(self.cap)
            self.full[b][j], self.marg[b][j] = full, marg

    def count(self) -> int:
        return int(sum(len(x) for x in self.full))

    def rank(self, e: float, dist: float, margin_only: bool = False, far: bool = False) -> float:
        """Calibrated causal effect c_i of Eq. (22): fraction of null effects below e_i."""
        if far:
            pool = self.far_marg if margin_only else self.far_full
        else:
            b = self.bin_of(dist)
            pool = (self.marg if margin_only else self.full)[b]
            if len(pool) < 20:
                pool = [u for lst in (self.marg if margin_only else self.full) for u in lst]
        if not pool:
            return 0.0
        arr = np.asarray(pool, dtype=np.float64)
        return float((arr < e).sum() / (len(arr) + 1.0))

    def state_dict(self) -> Dict[str, object]:
        return {"full": self.full, "marg": self.marg, "far_full": self.far_full, "far_marg": self.far_marg}

    def load_state_dict(self, d: Dict[str, object]) -> None:
        self.full = [list(x) for x in d["full"]]
        self.marg = [list(x) for x in d["marg"]]
        self.far_full = list(d["far_full"])
        self.far_marg = list(d["far_marg"])


class CausalDataset:
    """Labelled interactions for the causal head, Eq. (24). Labels are produced per relevance mode."""

    FIELDS = ("x0", "xi", "e_full", "e_marg", "dist", "reliance", "attn", "domain")

    def __init__(self, cfg: Config, capacity: int = 400000):
        self.cfg = cfg
        self.cap = int(capacity)
        self.n = 0
        self.ptr = 0
        self.x0 = np.zeros((self.cap, cfg.scalar_dim), np.float32)
        self.xi = np.zeros((self.cap, cfg.edge_dim), np.float32)
        self.e_full = np.zeros((self.cap,), np.float32)
        self.e_marg = np.zeros((self.cap,), np.float32)
        self.dist = np.zeros((self.cap,), np.float32)
        self.reliance = np.zeros((self.cap,), np.float32)
        self.attn = np.zeros((self.cap,), np.float32)
        self.domain = np.zeros((self.cap,), np.int8)   # 0 source, 1 target

    def add(self, x0, xi, e_full, e_marg, dist, reliance, attn, domain: int) -> None:
        i = self.ptr
        self.x0[i], self.xi[i] = x0, xi
        self.e_full[i], self.e_marg[i], self.dist[i] = e_full, e_marg, dist
        self.reliance[i], self.attn[i], self.domain[i] = reliance, attn, domain
        self.ptr = (self.ptr + 1) % self.cap
        self.n = min(self.n + 1, self.cap)

    def labels(self, idx: np.ndarray, nulls: NullBank, relevance: str, null_mode: str) -> np.ndarray:
        far = null_mode == "far"
        if relevance == "effect":
            return np.array([nulls.rank(float(self.e_full[i]), float(self.dist[i]), False, far) for i in idx], np.float32)
        if relevance == "margin":
            return np.array([nulls.rank(float(self.e_marg[i]), float(self.dist[i]), True, far) for i in idx], np.float32)
        if relevance == "reliance":
            return (1.0 - np.exp(-self.reliance[idx] / 0.05)).astype(np.float32)
        if relevance == "attention":
            return np.clip(self.attn[idx], 0.0, 1.0).astype(np.float32)
        raise ValueError(relevance)

    def save(self, path: str) -> None:
        n = self.n
        np.savez_compressed(path, **{f: getattr(self, f)[:n] for f in self.FIELDS})

    def load(self, path: str) -> None:
        d = np.load(path)
        n = int(d["x0"].shape[0])
        for f in self.FIELDS:
            getattr(self, f)[:n] = d[f]
        self.n, self.ptr = n, n % self.cap



class EpistemicMap:
    """Monotone map g of Eq. (6), fitted on held-out source states.

    g(v) = clip((v - median) / (q95 - median), 0, 1): ordinary source states map to about 0 and
    only states with unusually high critic disagreement add to the decision uncertainty."""

    def __init__(self):
        self.med, self.hi, self.fitted = 0.0, 1.0, False

    def fit_from_values(self, values: np.ndarray) -> None:
        v = np.asarray(values, np.float64).reshape(-1)
        v = v[np.isfinite(v)]
        if v.size < 10:
            return
        self.med = float(np.median(v))
        self.hi = max(float(np.quantile(v, 0.95)), self.med + 1e-6)
        self.fitted = True

    def transform_tensor(self, v: torch.Tensor) -> torch.Tensor:
        if not self.fitted:
            return torch.zeros_like(v)
        return ((v - self.med) / (self.hi - self.med)).clamp(0.0, 1.0)

    def state_dict(self) -> Dict[str, object]:
        return {"med": self.med, "hi": self.hi, "fitted": self.fitted}

    def load_state_dict(self, d: Dict[str, object]) -> None:
        if d and "med" in d:
            self.med, self.hi, self.fitted = float(d["med"]), float(d["hi"]), bool(d["fitted"])


# ======================================================
# Networks (Secs. 3.1, 3.3, 3.4)
# ======================================================
class PrototypeIntervention(nn.Module):
    """Latent de-confounding baseline, adapted from the prototype backdoor adjustment of CausalVAD.

    Prototypes are k-means centres of source interaction embeddings; the adjustment
    weights each prototype by its source prior P(k) and adds the result to z."""

    def __init__(self, z_dim: int, n_proto: int = 16):
        super().__init__()
        self.register_buffer("protos", torch.zeros(n_proto, z_dim))
        self.register_buffer("prior", torch.full((n_proto,), 1.0 / n_proto))
        self.q = nn.Linear(z_dim, z_dim)
        self.k = nn.Linear(z_dim, z_dim)
        self.v = nn.Linear(z_dim, z_dim)
        self.out = nn.Linear(z_dim, z_dim)
        nn.init.zeros_(self.out.weight)
        nn.init.zeros_(self.out.bias)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        att = torch.softmax(self.q(z) @ self.k(self.protos).t() / math.sqrt(z.shape[-1]), dim=-1)
        wts = att * self.prior.unsqueeze(0)
        wts = wts / wts.sum(-1, keepdim=True).clamp(min=1e-6)
        return self.out(wts @ self.v(self.protos))


def match_token_encoder(cfg: "Config", path: str) -> None:
    """Set cfg.token_encoder to the form a saved agent was trained with (linear W or MLP phi), so source
    agents trained before the switch still load; a fresh source stage uses the Config default."""
    if not os.path.exists(path):
        return
    try:
        ck = torch.load(path, map_location="cpu", weights_only=False)
    except Exception:
        return
    keys = ck.get("actor", {}).keys() if isinstance(ck, dict) else []
    found = "mlp" if any(k.startswith("enc.W.0.") for k in keys) else \
        ("linear" if any(k == "enc.W.weight" for k in keys) else None)
    if found and found != str(cfg.token_encoder):
        print(f"[INFO] {path} was trained with token_encoder={found}; using it (Config default "
              f"{cfg.token_encoder} applies to a new source stage)")
        cfg.token_encoder = found


ALPHA_MAX = 0.5      # global bound on the SAC temperature alpha_0 (and so on alpha); used by the critic cap


class RelationalEncoder(nn.Module):
    """Ego-centric relational state, Eq. (4).

    w_i = softmax_i(a_psi(x_0, x_i) - lambda_d ||p_i||^2 - lambda_sigma sigma_i^2) and
    s = [sum_i w_i h_i ; x_0] with h_i = W x_i. The reliability form of the distance prior
    favours near and confidently detected entities (deviation register, item 4)."""

    def __init__(self, cfg: Config, use_prototypes: bool = False):
        super().__init__()
        E, S, Z = cfg.edge_dim, cfg.scalar_dim, cfg.z_dim
        self.token_mlp = str(getattr(cfg, "token_encoder", "linear")) == "mlp"
        # h_i = phi(x_i): with a linear W the pooled state is W times the attention-weighted MEAN of the
        # token features, so the distance of the car in the ego's path is averaged with the parked cars
        # beside it, and "is anything blocking me, and how close" cannot be read out reliably (the
        # source policy held throttle toward a clearly detected stopped car, sigma^2 0.1, attention 2x).
        self.W = nn.Sequential(nn.Linear(E, Z), nn.ReLU(), nn.Linear(Z, Z)) if self.token_mlp else nn.Linear(E, Z)
        self.a_psi = nn.Sequential(nn.Linear(S + E, cfg.attention_relevance_hidden), nn.ReLU(),
                                   nn.Linear(cfg.attention_relevance_hidden, 1))
        self.mlp_scal = nn.Sequential(nn.Linear(S + 1, 128), nn.ReLU(), nn.Linear(128, 128), nn.ReLU())
        self.fuse = nn.Sequential(nn.Linear(Z + 128 + (Z if self.token_mlp else 0), 256), nn.ReLU(),
                                  nn.Linear(256, 256), nn.ReLU())
        self.lam_d = float(cfg.attention_distance_weight)
        self.lam_s = float(cfg.attention_uncertainty_weight)
        self.proto = PrototypeIntervention(Z) if use_prototypes else None

    def attention(self, scalars: torch.Tensor, edges: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        n = edges.shape[1]
        sig2 = edges[..., -1].clamp(1e-3, 2.0)
        dp2 = edges[..., 0:2].pow(2).sum(-1)
        ctx = torch.cat([scalars.unsqueeze(1).expand(-1, n, -1), edges], dim=-1)
        logits = self.a_psi(ctx).squeeze(-1) - self.lam_d * dp2 - self.lam_s * sig2
        valid = mask > 0.5
        logits = logits.masked_fill(~valid, -1e9)
        w = torch.softmax(logits, dim=1) * valid.float()
        return w / w.sum(1, keepdim=True).clamp(min=1e-6)

    def pooled(self, scalars, edges, mask) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        h = self.W(edges)
        w = self.attention(scalars, edges, mask)
        z = (w.unsqueeze(-1) * h).sum(1)
        return z, w, h

    def forward(self, scalars, edges, mask):
        z, w, h = self.pooled(scalars, edges, mask)
        if self.proto is not None:
            z = z + self.proto(z)
        sig_ale = (w * edges[..., -1]).sum(1)
        parts = [z]
        if self.token_mlp:
            # max over tokens of phi(x_i): the strongest "blocking" feature of any token survives pooling
            valid = (mask > 0.5).unsqueeze(-1)
            hm = h.masked_fill(~valid, -1e4).max(1).values
            parts.append(torch.where(valid.any(1), hm, torch.zeros_like(hm)))
        feat = self.fuse(torch.cat(parts + [self.mlp_scal(torch.cat([scalars, sig_ale.unsqueeze(-1)], -1))], -1))
        return feat, w, h


LEARNING_REVISION = "cast-bounded-policy-stratified-bc-2026-10-23"
TRAINING_REVISION = "cast-mean-target-behavior-audit-2026-10-30"
EVALUATION_METRIC_REVISION = "front-bumper-stop-line-crossing-v2"


def validate_training_checkpoint(checkpoint, evaluation_only=False):
    if checkpoint.get("learning_revision") != LEARNING_REVISION:
        raise ValueError("Checkpoint architecture/learning revision is incompatible with this code")
    if checkpoint.get("training_revision") != TRAINING_REVISION and not evaluation_only:
        raise ValueError("Old value-learning/teacher/reward replay cannot resume with this training revision. "
                         "Use a fresh --out-dir, reuse the perception model, and retrain the source. "
                         "Existing weights remain usable for frozen car_eval.py evaluation.")


def value_training_options(cfg):
    """Options that may not silently change when resuming weights/replay."""
    names = ("critic_target", "align_metric", "lateral_control", "light_source", "light_token_version",
             "critic_disagreement_penalty", "source_rl_max", "target_rl_max", "hazard_aug_prob", "hazard_aug_version",
             "late_brake_prob",
             "takeover_loss_weight",
             "takeover_penalty", "target_state_bc_weight", "target_state_bc_threshold")
    return {key: getattr(cfg, key) for key in names}


def validate_value_options(saved, cfg, source_only=False):
    current = value_training_options(cfg)
    if isinstance(saved, dict) and "align_metric" not in saved:
        saved = dict(saved, align_metric="kl")      # checkpoints written before the option existed
    if isinstance(saved, dict) and "source_rl_max" not in saved:
        saved = dict(saved, source_rl_max=1.0)      # earlier runs ramped the RL weight to 1
    if isinstance(saved, dict) and "hazard_aug_version" not in saved:
        saved = dict(saved, hazard_aug_version=1)   # rgbd14d: free-road insertion only
    if isinstance(saved, dict) and "hazard_aug_prob" not in saved:
        saved = dict(saved, hazard_aug_prob=0.0)    # rgbd14 and earlier: no hazard insertion
    if isinstance(saved, dict) and "target_rl_max" not in saved:
        saved = dict(saved, target_rl_max=1.0)      # rgbd14 and earlier: full RL weight in adaptation
    if isinstance(saved, dict) and "late_brake_prob" not in saved:
        saved = dict(saved, late_brake_prob=0.0)    # earlier runs had no late-braking demonstrations
    if isinstance(saved, dict) and "lateral_control" not in saved:
        saved = dict(saved, lateral_control="policy", light_source="camera")   # earlier runs
    if isinstance(saved, dict) and "light_token_version" not in saved:
        saved = dict(saved, light_token_version=1)  # rgbd12 and earlier: unknown colour encoded like yellow
    if source_only and isinstance(saved, dict):
        # The source stage has no alignment/invariance term, so the metric may be chosen afterwards.
        # The target RL weight is unused by the source stage either.
        # hazard_aug_prob may be switched on for the adaptation of an earlier source policy.
        skip = ("align_metric", "target_rl_max", "hazard_aug_prob", "hazard_aug_version")
        saved = {k: v for k, v in saved.items() if k not in skip}
        current = {k: v for k, v in current.items() if k not in skip}
        # Target-state imitation is unused by source training. It may be chosen
        # after the source validation gate, before the first adaptation checkpoint.
        saved = {k: v for k, v in saved.items() if not k.startswith("target_state_")}
        current = {k: v for k, v in current.items() if not k.startswith("target_state_")}
    if saved != current:
        raise ValueError(f"Value-learning options changed: saved={saved}, requested={current}. "
                         "Resume with identical options, or use a fresh experiment directory.")


def validate_training_directory(out, cfg=None):
    from pathlib import Path
    root = Path(out)
    has_target = any((root / "runs").rglob("*.pt"))
    has_policy = any((root / "source").rglob("*.pt")) or has_target
    if not has_policy:
        return
    path = root / "run_config.json"
    if not path.is_file():
        raise ValueError("Existing policy artifacts have no training-protocol record. Use a fresh --out-dir.")
    with path.open() as f:
        previous = json.load(f)
    if previous.get("training_revision") != TRAINING_REVISION:
        raise ValueError("This directory contains an earlier training protocol. Use a fresh --out-dir; "
                         "old checkpoints and replay are preserved for evaluation.")
    if cfg is not None:
        validate_value_options(previous.get("value_training_options"), cfg, source_only=not has_target)


def reduce_critic_target(q_values, mode, penalty):
    """Experimental target reductions; inputs have shape [critics, batch]."""
    if mode == "mean":
        return q_values.mean(0)
    if mode == "lcb":
        return q_values.mean(0) - float(penalty) * q_values.std(0, unbiased=False)
    if mode == "min2":
        pick = torch.randperm(q_values.shape[0], device=q_values.device)[:min(2, q_values.shape[0])]
        return q_values[pick].min(0).values
    raise ValueError(f"Unknown critic target: {mode}")


def target_expert_eligible(domain, uncertainty, cfg):
    return (domain == 1 and cfg.target_state_bc_weight > 0.0
            and math.isfinite(float(uncertainty)) and float(uncertainty) >= cfg.target_state_bc_threshold)


def takeover_label_enabled(reason, weight):
    # The hold interval is a continuation of the SAME intervention, not a new rejection.
    return float(weight) > 0.0 and reason in ("lane", "brake", "obstacle", "start")


def discounted_behavior_returns(rewards, bonuses, gamma):
    """Finite observed sums; Q's entropy starts at t+1, never at the current action."""
    if len(rewards) != len(bonuses):
        raise ValueError("Reward and entropy timelines must match")
    raw, soft = np.zeros(len(rewards)), np.zeros(len(rewards))
    for t in range(len(rewards) - 1, -1, -1):
        raw[t] = rewards[t] + (gamma * raw[t + 1] if t + 1 < len(raw) else 0.0)
        soft[t] = rewards[t] + (gamma * (soft[t + 1] + bonuses[t + 1]) if t + 1 < len(soft) else 0.0)
    return raw, soft


def build_value_audit(rewards, bonuses, samples, gamma, terminal, metadata):
    """Behavior-path audit, not an unbiased estimate of the changing actor's Q^pi."""
    raw, soft = discounted_behavior_returns(rewards, bonuses, gamma)
    points = []
    for sample in samples:
        t = int(sample["t"])
        p = dict(sample, observed_return=float(raw[t]), entropy_adjusted_return=float(soft[t]))
        p["q_minus_soft_return"] = float(p["q_executed"] - soft[t]) if terminal else None
        points.append(p)
    result = dict(metadata, complete_terminal=bool(terminal), censored=not bool(terminal),
                  convention="behavior-path; changing policy; expected entropy at visited states; not Qpi calibration",
                  samples=points, gamma=float(gamma))
    for key in ("q_executed", "observed_return", "entropy_adjusted_return"):
        result["mean_" + key] = float(np.mean([p[key] for p in points])) if points else None
    result["mean_q_minus_soft_return"] = (float(np.mean([p["q_minus_soft_return"] for p in points]))
                                           if terminal and points else None)
    return result


def squashed_gaussian_entropy(mu, logstd, action_scale):
    """Deterministic 5-point Gauss-Hermite approximation; does not consume training RNG."""
    nodes = mu.new_tensor([-2.02018287045609, -0.958572464613819, 0.0,
                          0.958572464613819, 2.02018287045609])
    weights = mu.new_tensor([0.0112574113277207, 0.222075922005613, 0.533333333333333,
                            0.222075922005613, 0.0112574113277207])
    u = mu.unsqueeze(-1) + math.sqrt(2.0) * logstd.exp().unsqueeze(-1) * nodes
    jac = 2.0 * (math.log(2.0) - u - F.softplus(-2.0 * u))
    return (logstd + 0.5 * math.log(2.0 * math.pi * math.e)
            + (jac * weights).sum(-1) + action_scale.log()).sum(-1)



def bounded_policy_logstd(raw: torch.Tensor, cap: float) -> torch.Tensor:
    """Smooth log-std interval [log(cap)-3, log(cap)], avoiding a dead clamped head."""
    if not math.isfinite(float(cap)) or not 1e-4 <= float(cap) <= 1.0:
        raise ValueError("--target-explore-std must be finite and in [0.0001, 1.0]")
    return math.log(float(cap)) - 3.0 + 3.0 * torch.sigmoid(raw)


def policy_entropy_target(cap: float, dimensions: int) -> float:
    # The Gaussian's maximum entropy is an upper bound after tanh. Leave room for
    # squashing at braking/steering actions; this is a tuning choice, not a guarantee.
    h_upper = dimensions * (math.log(float(cap)) + 0.5 * math.log(2.0 * math.pi * math.e))
    return min(-1.5 * dimensions, h_upper - float(dimensions))


class CaSTActor(nn.Module):
    def __init__(self, cfg: Config, use_prototypes: bool = False):
        super().__init__()
        self.cfg = cfg
        self.enc = RelationalEncoder(cfg, use_prototypes)
        self.mu = nn.Sequential(nn.Linear(256, 256), nn.ReLU(), nn.Linear(256, cfg.policy_action_dim))
        self.logstd = nn.Sequential(nn.Linear(256, 256), nn.ReLU(), nn.Linear(256, cfg.policy_action_dim))
        # Motion head g_nu of Eq. (5): next relative displacement, isotropic Gaussian.
        self.motion = nn.Sequential(nn.Linear(cfg.z_dim + 1, 64), nn.ReLU(), nn.Linear(64, 64), nn.ReLU(), nn.Linear(64, 3))
        with torch.no_grad():
            self.mu[-1].bias.copy_(torch.tensor([0.2, 0.0]))       # gentle throttle, straight
            self.logstd[-1].weight.zero_()
            self.logstd[-1].bias.fill_(math.log(4.0))  # sigmoid=.8: initial std=cap*exp(-.6)
            self.motion[-1].bias[2].fill_(-3.0)   # start with a small aleatoric variance
        keep = torch.ones(cfg.scalar_dim)
        keep[1:4] = 0.0
        self.register_buffer("actor_scalar_keep", keep)
        self.register_buffer("action_scale", torch.ones(cfg.policy_action_dim))
        self.register_buffer("action_bias", torch.zeros(cfg.policy_action_dim))

    def dist_params(self, scalars, edges, mask):
        # The previous action (scalars 1-3) is hidden from the actor. At 20 Hz it almost equals the
        # current teacher action, so imitation learns to copy it ("copycat" causal confusion) and the
        # policy then repeats its own errors once it drives. The critic still sees it.
        feat, w, h = self.enc(scalars * self.actor_scalar_keep, edges, mask)
        return self.mu(feat), bounded_policy_logstd(self.logstd(feat), self.cfg.target_explore_std), w, h

    def motion_head(self, h: torch.Tensor, edges: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        out = self.motion(torch.cat([h, edges[..., -1:]], -1))
        return out[..., :2], out[..., 2].clamp(-6.0, 4.0)

    def squash(self, u: torch.Tensor) -> torch.Tensor:
        return torch.tanh(u) * self.action_scale + self.action_bias

    def sample(self, scalars, edges, mask, deterministic: bool = False):
        mu, logstd, w, h = self.dist_params(scalars, edges, mask)
        std = logstd.exp()
        normal = torch.distributions.Normal(mu, std)
        u = mu if deterministic else normal.rsample()
        t = torch.tanh(u)
        action = t * self.action_scale + self.action_bias
        # log(1-tanh(u)^2), evaluated without cancellation near saturated controls.
        log_jacobian = 2.0 * (math.log(2.0) - u - F.softplus(-2.0 * u))
        logp_dims = normal.log_prob(u) - log_jacobian - torch.log(self.action_scale)
        return action, logp_dims.sum(-1), mu, logstd, w, h, logp_dims


class CaSTCritic(nn.Module):
    def __init__(self, cfg: Config):
        super().__init__()
        self.enc = RelationalEncoder(cfg)
        self.q = nn.Sequential(nn.Linear(256 + cfg.policy_action_dim, 256), nn.ReLU(), nn.Linear(256, 256), nn.ReLU(), nn.Linear(256, 1))

    def forward(self, scalars, edges, mask, action):
        feat, _, _ = self.enc(scalars, edges, mask)
        return self.q(torch.cat([feat, action], -1)).squeeze(-1)


class CausalHead(nn.Module):
    """Amortized causal head g_vartheta of Eq. (24)."""

    def __init__(self, cfg: Config):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(cfg.scalar_dim + cfg.edge_dim, 128), nn.ReLU(),
                                 nn.Linear(128, 128), nn.ReLU(), nn.Linear(128, 1))

    def forward(self, x0: torch.Tensor, xi: torch.Tensor) -> torch.Tensor:
        if xi.dim() == 3:
            x0 = x0.unsqueeze(1).expand(-1, xi.shape[1], -1)
        return torch.sigmoid(self.net(torch.cat([x0, xi], -1))).squeeze(-1)


def gaussian_kl(mu_p, logstd_p, mu_q, logstd_q) -> torch.Tensor:
    """KL(N_p || N_q) of the pre-squash Gaussians, summed over action dimensions."""
    # log-std clamped at -3 (std 0.05) inside the KL: with the entropy bonus at 0.1 the policy std can
    # approach exp(-5), and a mean gap of 0.1 would then give a KL of about 100 and dominate the actor loss.
    logstd_p, logstd_q = logstd_p.clamp(min=-3.0), logstd_q.clamp(min=-3.0)
    var_p, var_q = (2 * logstd_p).exp(), (2 * logstd_q).exp()
    return (logstd_q - logstd_p + (var_p + (mu_p - mu_q).pow(2)) / (2 * var_q) - 0.5).sum(-1)


def mmd_rbf(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    xx, yy, xy = torch.cdist(x, x).pow(2), torch.cdist(y, y).pow(2), torch.cdist(x, y).pow(2)
    out = torch.zeros((), device=x.device)
    bw = 1.0
    for _ in range(5):
        out = out + torch.exp(-xx / bw).mean() + torch.exp(-yy / bw).mean() - 2 * torch.exp(-xy / bw).mean()
        bw *= 2.0
    return out.clamp(min=0.0)


# ======================================================
# Coverage (Eqs. 14, 15, 27)
# ======================================================
class KNNBank:
    def __init__(self, data: torch.Tensor, k: int):
        self.data = data
        self.k = int(k)

    def kdist(self, q: torch.Tensor, chunk: int = 4096) -> torch.Tensor:
        """Sorted k smallest Euclidean distances of each query row to the bank."""
        if self.data.shape[0] == 0:
            return torch.full((q.shape[0], self.k), 1e6, device=q.device)
        k = min(self.k, self.data.shape[0])
        outs = []
        for i in range(0, q.shape[0], chunk):
            d = torch.cdist(q[i:i + chunk], self.data)
            outs.append(torch.topk(d, k, dim=1, largest=False).values)
        out = torch.cat(outs, 0)
        if k < self.k:
            out = torch.cat([out, torch.full((out.shape[0], self.k - k), 1e6, device=q.device)], 1)
        return out


class CoverageModel:
    """Calibrated rank coverage of Eq. (14) with validated growth of Eq. (28)."""

    def __init__(self, bank: torch.Tensor, ref_sorted: torch.Tensor, k: int, add_cap: int = 50000):
        self.bank = KNNBank(bank, k)
        self.ref = ref_sorted
        self.k = int(k)
        self.add: List[torch.Tensor] = []
        self.add_cap = int(add_cap)
        self._add_cat: Optional[torch.Tensor] = None

    def rank(self, d: torch.Tensor) -> torch.Tensor:
        n = self.ref.shape[0]
        cnt_ge = n - torch.searchsorted(self.ref, d.contiguous(), right=False)
        return (1.0 + cnt_ge.float()) / (n + 1.0)

    def kdist_source(self, q: torch.Tensor) -> torch.Tensor:
        return self.bank.kdist(q)

    def clone(self) -> "CoverageModel":
        """Own copy for one adaptation run: shares the frozen source bank and reference distances, keeps
        its own growth (runs used to share, and accumulate, each other's added points)."""
        m = CoverageModel.__new__(CoverageModel)
        m.bank, m.ref, m.k, m.add_cap = self.bank, self.ref, self.k, self.add_cap
        m.add = [t.clone() for t in self.add]
        m._add_cat = None
        return m

    def add_points(self, pts: torch.Tensor) -> None:
        if pts.numel() == 0:
            return
        self.add.append(pts.detach())
        cat = torch.cat(self.add, 0)
        if cat.shape[0] > self.add_cap:
            idx = torch.randperm(cat.shape[0], device=cat.device)[: self.add_cap]
            cat = cat[idx]
        self.add = [cat]
        self._add_cat = cat

    def kth_merged(self, q: torch.Tensor, dk_source: torch.Tensor) -> torch.Tensor:
        """k-th neighbour distance to Z_S union the validated additions."""
        if self._add_cat is None or self._add_cat.shape[0] == 0:
            return dk_source[..., -1]
        da = KNNBank(self._add_cat, self.k).kdist(q)
        merged = torch.cat([dk_source, da], -1)
        return torch.topk(merged, self.k, dim=-1, largest=False).values[..., -1]


def gate_and_floor(rho: torch.Tensor, tau_rho: float) -> Tuple[torch.Tensor, torch.Tensor]:
    """Alignment weight beta_i and uncertainty floor ell_i of Eq. (16)."""
    beta = (rho / tau_rho - 1.0).clamp(0.0, 1.0)
    ell = (1.0 - rho / tau_rho).clamp(0.0, 1.0)
    return beta, ell


# ======================================================
# Replay buffer
# ======================================================
class CaSTReplay:
    def __init__(self, capacity: int, cfg: Config, with_coverage: bool):
        N, E, S, B, k = cfg.max_entities, cfg.edge_dim, cfg.scalar_dim, cfg.scene_dim, cfg.knn_k
        self.cap, self.ptr, self.n = int(capacity), 0, 0
        spec = {"scalars": (S,), "edges": (N, E), "mask": (N,), "scene": (B,), "actions": (cfg.policy_action_dim,), "rewards": (),
                "terminals": (), "next_scalars": (S,), "next_edges": (N, E), "next_mask": (N,), "next_scene": (B,),
                "motion_target": (N, 2), "motion_mask": (N,), "support": (), "next_support": (), "teacher": (), "expert_action": (cfg.policy_action_dim,), "has_expert": (), "slow": (),
                "rejected_action": (cfg.policy_action_dim,), "has_rejected": (),
                "episode_end": (), "truncated": (), "domain": (), "label_uncertainty": ()}
        if with_coverage:
            spec.update({"dk": (N, k), "next_dk": (N, k), "dk0": (k,), "next_dk0": (k,),
                         "c_or": (N,), "next_c_or": (N,)})
        self.spec = spec
        self.data = {key: np.zeros((self.cap,) + shp, np.float32) for key, shp in spec.items()}
        # O(1) membership maintenance, including ring-buffer overwrites. These
        # indexes are derived from saved replay rows, not extra checkpoint state.
        self._expert_groups: List[List[int]] = [[], [], [], []]
        self._expert_group = np.full(self.cap, -1, np.int8)
        self._expert_pos = np.full(self.cap, -1, np.int64)

    def __len__(self) -> int:
        return self.n

    def add(self, row: Dict[str, np.ndarray]) -> int:
        i = self.ptr
        for key in self.spec:
            if key in row:
                self.data[key][i] = row[key]
            else:
                self.data[key][i] = 0.0
        self.ptr = (self.ptr + 1) % self.cap
        self.n = min(self.n + 1, self.cap)
        self._index_expert(i)
        return i

    def _index_expert(self, i: int) -> None:
        old = int(self._expert_group[i])
        if old >= 0:
            group = self._expert_groups[old]
            pos = int(self._expert_pos[i])
            last = group[-1]
            group[pos] = last
            self._expert_pos[last] = pos
            group.pop()
        self._expert_group[i] = self._expert_pos[i] = -1
        if self.data["has_expert"][i] <= 0.5:
            return
        slow = self.data["slow"][i] > 0.5
        longitudinal = self.data["expert_action"][i, 0]
        g = (2 if longitudinal > 0.05 else 3) if slow else (0 if longitudinal < -0.1 else 1)
        self._expert_group[i] = g
        self._expert_pos[i] = len(self._expert_groups[g])
        self._expert_groups[g].append(i)

    def _rebuild_expert_index(self) -> None:
        self._expert_groups = [[], [], [], []]
        self._expert_group.fill(-1)
        self._expert_pos.fill(-1)
        for i in range(self.n):
            self._index_expert(i)

    def expert_indices(self, bs: int) -> Optional[np.ndarray]:
        """Equal counts from each available driving group in this replay's labeled states."""
        groups = [g for g in self._expert_groups if g]
        if not groups or int(bs) <= 0:
            return None
        order = np.random.permutation(len(groups))
        q, rem = divmod(int(bs), len(groups))
        out = []
        for j, k in enumerate(order):
            group = groups[int(k)]
            count = q + int(j < rem)
            positions = np.random.randint(0, len(group), size=count)
            out.extend(group[int(p)] for p in positions)
        idx = np.asarray(out, np.int64)
        np.random.shuffle(idx)
        return idx

    def sample_expert(self, bs: int, device: torch.device) -> Optional[Dict[str, torch.Tensor]]:
        idx = self.expert_indices(bs)
        if idx is None:
            return None
        keys = ("scalars", "edges", "mask", "expert_action", "has_expert", "slow")
        return {k: torch.from_numpy(self.data[k][idx]).to(device) for k in keys}

    def set_motion(self, i: int, target: np.ndarray, mask: np.ndarray) -> None:
        self.data["motion_target"][i] = target
        self.data["motion_mask"][i] = mask

    def sample_mixed(self, bs: int, device: torch.device, demo_frac: float) -> Dict[str, torch.Tensor]:
        """Batch with a fixed share of teacher-driven transitions (teacher phase and take-overs), the rest
        from the policy's own driving: the demonstrations stay a large part of every batch even after the
        policy has filled most of the buffer (as in DDPGfD / RLPD)."""
        n = self.n
        demo = np.nonzero(self.data["teacher"][:n] > 0.5)[0]
        own = np.nonzero(self.data["teacher"][:n] <= 0.5)[0]
        k = int(round(float(np.clip(demo_frac, 0.0, 1.0)) * bs))
        if demo.size == 0 or own.size == 0 or k <= 0:
            return self.sample(bs, device)
        idx = np.concatenate([demo[np.random.randint(0, demo.size, size=k)],
                              own[np.random.randint(0, own.size, size=bs - k)]])
        return self.sample(bs, device, idx)

    def sample(self, bs: int, device: torch.device, idx: Optional[np.ndarray] = None) -> Dict[str, torch.Tensor]:
        if idx is None:
            idx = np.random.randint(0, self.n, size=int(bs))
        return {k: torch.from_numpy(v[idx]).to(device) for k, v in self.data.items()}

    def save(self, path: str, max_rows: int) -> None:
        n = min(self.n, int(max_rows))
        if n <= 0:
            raise ValueError("Cannot save an empty source replay snapshot")
        if self.n > n:
            # Protect a small set of each expert group when compressing the
            # source replay; a uniform snapshot can discard every rare brake.
            protected = []
            groups = [g for g in self._expert_groups if g]
            per_group = min(64, n // max(len(groups), 1))
            for group in groups:
                positions = np.random.choice(len(group), size=min(per_group, len(group)), replace=False)
                protected.extend(group[int(p)] for p in positions)
            keep = np.asarray(protected, np.int64)
            available = np.ones(self.n, bool)
            available[keep] = False
            rest = np.random.choice(np.flatnonzero(available), size=n - len(keep), replace=False)
            idx = np.concatenate([keep, rest])
            np.random.shuffle(idx)
        else:
            idx = np.arange(self.n)
        np.savez_compressed(path, **{k: v[idx] for k, v in self.data.items()})

    def save_all(self, path: str) -> None:
        np.savez(path, ptr=self.ptr, n=self.n, **{k: v[:self.n] for k, v in self.data.items()})

    def load_all(self, path: str) -> None:
        d = np.load(path)
        n = min(int(d["n"]), self.cap)
        for k in self.spec:
            if k in d:
                self.data[k][:n] = d[k][:n]
        self.n, self.ptr = n, int(d["ptr"]) % self.cap
        self._rebuild_expert_index()

    @staticmethod
    def load(path: str, cfg: Config) -> "CaSTReplay":
        d = np.load(path)
        n = int(d["scalars"].shape[0])
        buf = CaSTReplay(max(n, 1), cfg, with_coverage=False)
        for k in buf.spec:
            if k in d:
                buf.data[k][:n] = d[k]
        buf.n = n
        buf.ptr = n % buf.cap
        buf._rebuild_expert_index()
        return buf


def cat_batches(a: Dict[str, torch.Tensor], b: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    return {k: torch.cat([a[k], b[k]], 0) for k in a if k in b}


# ======================================================
# Run specification
# ======================================================
TRANSFER_METHODS = ("source_only", "finetune", "uniform", "openset_entropy", "contrastive_filter",
                    "ot_filter", "ace", "deconf", "cast")


@dataclass
class RunSpec:
    name: str
    suite: str
    method: str
    seed: int
    budget: int
    sampler: str = "uniform"          # uniform | cast | uncertainty | safety_critical
    gate: bool = True                 # Eq. (16) coverage gate (False -> uniform alignment of all causal interactions)
    epistemic: bool = True            # epistemic channel of Eq. (18)
    floor_penalty: bool = False       # enforce the floor as a penalty on sigma_hat^2 instead of Eq. (18)
    scene_term: bool = True           # scene-level coverage of Eq. (15)
    cautious: bool = True             # pessimistic value of Eq. (19)
    coverage_space: str = "source"    # source (h^S) | theta (trainable h)
    causal_mask: bool = True          # bank restricted to causal interactions
    relevance: str = "effect"         # effect | margin | reliance | attention
    null_mode: str = "matched"        # matched | far
    invariance: bool = True           # Eq. (25)
    validated_growth: bool = True     # Eq. (28)
    oracle_causal: bool = False
    oracle_coverage: bool = False
    q: float = 0.05
    k: int = 10
    kappa: float = 1.0
    eps: float = 0.2
    int_frac: float = 0.02
    eval_suites: Tuple[str, ...] = ()
    tag: str = ""

    def is_cast(self) -> bool:
        return self.method == "cast"

    def run_id(self) -> str:
        flags = []
        defaults = RunSpec(name="", suite="", method="", seed=0, budget=0)
        for f in ("sampler", "gate", "epistemic", "floor_penalty", "scene_term", "cautious", "coverage_space",
                  "causal_mask", "relevance", "null_mode", "invariance", "validated_growth", "oracle_causal",
                  "oracle_coverage", "q", "k", "kappa", "eps", "int_frac"):
            if getattr(self, f) != getattr(defaults, f):
                flags.append(f"{f}={getattr(self, f)}")
        core = f"{self.suite}__{self.method}__B{self.budget}__s{self.seed}"
        return core + ("__" + "_".join(flags).replace(".", "p").replace("=", "-") if flags else "")


# ======================================================
# Agent: CaST and the transfer baselines
# ======================================================
def balanced_imitation_loss(pred: torch.Tensor, batch: Dict[str, torch.Tensor]) -> Tuple[torch.Tensor, Dict[str, float]]:
    """Equal weight per present driving group, rather than per ordinary cruising sample.

    Source-only expert labels are used. Target states are never given privileged labels.
    Braking/starting diagnostics refer to this training minibatch, not evaluation recall.
    """
    valid = batch["has_expert"] > 0.5
    expert = batch["expert_action"]
    slow = batch["slow"] > 0.5
    brake = valid & ~slow & (expert[:, 0] < -0.1)
    start = valid & slow & (expert[:, 0] > 0.05)
    groups = (brake, valid & ~slow & ~brake, start, valid & slow & ~start)
    error = (pred - expert).pow(2).sum(-1)
    losses = [error[g].mean() for g in groups if bool(g.any())]
    loss = torch.stack(losses).mean() if losses else pred.sum() * 0.0
    stats = {"brake_n": float(brake.sum()), "start_n": float(start.sum())}
    for name, group in (("brake", brake), ("start", start)):
        if bool(group.any()):
            stats[name + "_mae"] = float((pred[group, 0] - expert[group, 0]).abs().mean().detach())
            wrong = pred[group, 0] >= 0.0 if name == "brake" else pred[group, 0] <= 0.05
            stats[name + "_miss"] = float(wrong.float().mean())
            if name == "brake":
                shortfall = (pred[group, 0] - expert[group, 0]).clamp(min=0.0)
                stats["brake_shortfall"] = float(shortfall.mean().detach())
                stats["brake_weak"] = float((shortfall > 0.1).float().mean())
    return loss, stats


class CaSTAgent:
    def __init__(self, cfg: Config, device: torch.device, spec: Optional[RunSpec] = None):
        self.cfg, self.device = cfg, device
        self.spec = spec or RunSpec(name="source", suite="source", method="finetune", seed=0, budget=0)
        use_proto = self.spec.method == "deconf"
        self.actor = CaSTActor(cfg, use_prototypes=use_proto).to(device)
        self.critics = nn.ModuleList([CaSTCritic(cfg).to(device) for _ in range(cfg.n_critics)])
        self.target_critics = nn.ModuleList([CaSTCritic(cfg).to(device) for _ in range(cfg.n_critics)])
        for c, t in zip(self.critics, self.target_critics):
            t.load_state_dict(c.state_dict())
        self.actor_opt = torch.optim.Adam(self.actor.parameters(), lr=cfg.actor_lr)
        self.critic_opt = torch.optim.Adam(self.critics.parameters(), lr=cfg.critic_lr)
        self.causal_head = CausalHead(cfg).to(device)
        self.causal_opt = torch.optim.Adam(self.causal_head.parameters(), lr=1e-3)
        self.v_cal = EpistemicMap()   # monotone map g of Eq. (6)
        # Scale of the predicted motion variance (m^2) in the aleatoric channel 1 - exp(-sigma^2 / sig_ref),
        # fitted on source states so that the median source token gives 0.1 (see fit_uncertainty_scales).
        self.sig_ref, self.sig_ref_fitted = 1.0, False
        # Frozen source objects (target stage only).
        self.W_S: Optional[nn.Linear] = None
        self.src_actor: Optional[CaSTActor] = None
        self.cov: Optional[CoverageModel] = None
        self.scene_cov: Optional[CoverageModel] = None
        self.src_feat_bank: Optional[KNNBank] = None
        self.openset_thr: float = 1e9
        self.source_buffer: Optional[CaSTReplay] = None
        self.ace_w = torch.ones(cfg.policy_action_dim, device=device) / cfg.policy_action_dim
        self.updates = 0
        self.target_updates = 0
        self.bc_weight = 0.0   # teacher warm start of the source stage (deviation register, item 13)
        self.actor_rl_weight = 1.0
        # alpha_0 of Eq. (6) is tuned by the SAC dual objective toward a target entropy. A fixed 0.2 made
        # the entropy term about six times larger than the progress reward and kept the policy noisy.
        self.log_alpha0 = torch.tensor(math.log(float(cfg.alpha0)), device=device, requires_grad=True)
        self.alpha_opt = torch.optim.Adam([self.log_alpha0], lr=3e-4)
        # A target of -3 can exceed even the unsquashed entropy ceiling with
        # std <= .05. Match the target to the bounded distribution.
        self.target_entropy = policy_entropy_target(cfg.target_explore_std, cfg.policy_action_dim)
        self.last_stats: Dict[str, float] = {}
        self.normalize_q = False  # source stage: scale the actor's RL term by 2.5 / max(mean|Q|, 2.5) (TD3+BC)

    # ---------------- persistence ----------------
    def state(self) -> Dict[str, object]:
        return {"learning_revision": LEARNING_REVISION, "training_revision": TRAINING_REVISION,
                "value_training_options": value_training_options(self.cfg),
                "training_role": "source" if self.spec.suite == "source" else "target",
                "policy_std_cap": float(self.cfg.target_explore_std),
                "actor": self.actor.state_dict(), "critics": self.critics.state_dict(),
                "target_critics": self.target_critics.state_dict(), "causal_head": self.causal_head.state_dict(),
                "v_cal": self.v_cal.state_dict(), "sig_ref": float(self.sig_ref), "updates": self.updates,
                "target_updates": int(getattr(self, "target_updates", 0)),
                "clip_frac": float(getattr(self, "clip_frac", 0.0)),
                "actor_opt": self.actor_opt.state_dict(), "critic_opt": self.critic_opt.state_dict(),
                "causal_opt": self.causal_opt.state_dict(), "log_alpha0": float(self.log_alpha0.detach()),
                "alpha_opt": self.alpha_opt.state_dict(),
                "ace_w": self.ace_w.detach().cpu() if torch.is_tensor(getattr(self, "ace_w", None)) else None}

    def save(self, path: str) -> None:
        torch.save(self.state(), path)

    def load(self, path: str, strict_actor: bool = False, evaluation_only: bool = False) -> None:
        try:
            ck = torch.load(path, map_location=self.device, weights_only=False)
        except TypeError:
            ck = torch.load(path, map_location=self.device)
        validate_training_checkpoint(ck, evaluation_only=evaluation_only)
        if not evaluation_only:
            validate_value_options(ck.get("value_training_options"), self.cfg,
                                   source_only=ck.get("training_role") == "source")
        self.loaded_training_revision = ck.get("training_revision", "legacy-2026-10-23")
        self.loaded_value_options = ck.get("value_training_options")
        if evaluation_only and self.loaded_value_options is not None:
            print(f"[EVAL TRAINING OPTIONS] {json.dumps(self.loaded_value_options, sort_keys=True)}")
        if evaluation_only:
            # The driving interface must match training: steering source and light token change what the
            # policy controls and observes, so a mismatch would evaluate a different system.
            saved = dict(self.loaded_value_options or {})
            override = bool(getattr(self.cfg, "eval_light_override", False))
            checks = [("lateral_control", "policy")]
            if not override:
                checks.append(("light_source", "camera"))
                if str(saved.get("light_source", "camera")) == "map_camera":
                    checks.append(("light_token_version", 1))
            for key, legacy in checks:
                want = str(saved.get(key, legacy))
                if str(getattr(self.cfg, key, legacy)) != want:
                    raise ValueError(f"Checkpoint was trained with --{key.replace('_', '-')} {want}; "
                                     f"evaluate with the same option (got {getattr(self.cfg, key, legacy)}).")
            if override:
                print(f"[EVAL DIAGNOSTIC] light source {self.cfg.light_source} (token v{self.cfg.light_token_version}); "
                      f"the checkpoint was trained with {saved.get('light_source', 'camera')} "
                      f"(v{saved.get('light_token_version', 1)}). Diagnostic only, not a paper result.")
        if not math.isclose(float(ck.get("policy_std_cap", -1.0)), float(self.cfg.target_explore_std),
                            rel_tol=1e-7, abs_tol=1e-9):
            raise ValueError("Checkpoint policy std cap differs: pass the training --target-explore-std "
                             "value for both resume and evaluation.")
        self.actor.load_state_dict(ck["actor"], strict=strict_actor)
        self.critics.load_state_dict(ck["critics"])
        self.target_critics.load_state_dict(ck["target_critics"])
        if "causal_head" in ck:
            self.causal_head.load_state_dict(ck["causal_head"])
        self.v_cal.load_state_dict(ck.get("v_cal", {}))
        if "sig_ref" in ck:
            self.sig_ref, self.sig_ref_fitted = float(ck["sig_ref"]), True
        self.updates = int(ck.get("updates", 0))
        self.target_updates = int(ck.get("target_updates", 0))
        self.clip_frac = float(ck.get("clip_frac", 0.0))
        if "log_alpha0" in ck:
            with torch.no_grad():
                self.log_alpha0.fill_(float(ck["log_alpha0"]))
        if ck.get("ace_w") is not None and torch.is_tensor(getattr(self, "ace_w", None)):
            with torch.no_grad():
                self.ace_w.copy_(ck["ace_w"].to(self.ace_w.device))
        if evaluation_only:
            return  # no optimizer state is needed to evaluate a frozen checkpoint
        for name, opt in (("actor_opt", self.actor_opt), ("critic_opt", self.critic_opt), ("causal_opt", self.causal_opt),
                          ("alpha_opt", self.alpha_opt)):
            if name in ck:
                try:
                    opt.load_state_dict(ck[name])
                except BaseException:
                    pass

    def freeze_source_copies(self) -> None:
        self.W_S = copy_module(self.actor.enc.W)
        self.src_actor = copy_module(self.actor)

    # ---------------- helpers ----------------
    @staticmethod
    def obs_tensors(obs: Dict[str, np.ndarray], device: torch.device) -> Tuple[torch.Tensor, ...]:
        return tuple(torch.from_numpy(np.asarray(obs[k], np.float32)[None]).to(device)
                     for k in ("scalars", "edges", "mask", "scene"))

    def h_source(self, edges: torch.Tensor) -> torch.Tensor:
        W = self.W_S if (self.W_S is not None and self.spec.coverage_space == "source") else self.actor.enc.W
        return W(edges)

    @torch.no_grad()
    def causal_scores(self, scalars: torch.Tensor, edges: torch.Tensor, mask: torch.Tensor,
                      c_or: Optional[torch.Tensor] = None) -> torch.Tensor:
        if self.spec.oracle_causal and c_or is not None:
            return c_or * mask
        return self.causal_head(scalars, edges) * mask

    def tau_c(self) -> float:
        return 1.0 - float(self.spec.q)

    def tau_rho(self) -> float:
        return float(self.spec.q)

    @torch.no_grad()
    def coverage_terms(self, edges, mask, scene, dk, dk0, support) -> Dict[str, torch.Tensor]:
        """rho_i (source), rho_tilde_i and floors, and the scene floor, for a batch."""
        B, N = mask.shape
        if self.spec.oracle_coverage:
            rho = support.view(B, 1).expand(B, N).clone()
            rho_t = rho.clone()
            rho0 = support.view(B).clone()
            rho0_t = rho0.clone()
        else:
            hs = self.h_source(edges).reshape(B * N, -1)
            kth_src = dk[..., -1]
            rho = self.cov.rank(kth_src)
            rho_t = self.cov.rank(self.cov.kth_merged(hs, dk.reshape(B * N, -1)).view(B, N))
            rho0 = self.scene_cov.rank(dk0[..., -1])
            rho0_t = self.scene_cov.rank(self.scene_cov.kth_merged(scene, dk0))
        beta, _ = gate_and_floor(rho, self.tau_rho())
        _, ell_t = gate_and_floor(rho_t, self.tau_rho())
        _, ell0_t = gate_and_floor(rho0_t, self.tau_rho())
        return {"rho": rho, "rho_t": rho_t, "beta": beta, "ell_t": ell_t * mask, "rho0": rho0, "ell0_t": ell0_t}

    def ubar(self, w, sig_hat2, ell_t, ell0_t, v, include_scene: bool = True) -> torch.Tensor:
        """Mean fusion avoids the additive saturation seen in the driving log.

        This is an experimental change to Eqs. (6)/(18), not a calibrated probability.
        --uncertainty-fusion sum restores the original additive definition.
        """
        two_channel = self.spec.is_cast() and self.spec.epistemic and not self.spec.floor_penalty and self.cov is not None
        s2 = sig_hat2 / max(float(self.sig_ref), 1e-6)
        if two_channel:
            u = 1.0 - (1.0 - ell_t) * torch.exp(-s2)
        else:
            u = 1.0 - torch.exp(-s2)
        out = (w * u).sum(1) + self.v_cal.transform_tensor(v)
        channels = 2 if self.v_cal.fitted else 1
        if include_scene and self.spec.is_cast() and self.spec.scene_term and self.cov is not None:
            out = out + ell0_t
            channels += 1
        if self.cfg.uncertainty_fusion == "mean":
            out = out / float(channels)
        return out.clamp(0.0, 1.0)

    def critic_stack(self, scalars, edges, mask, action, target: bool = False) -> torch.Tensor:
        cs = self.target_critics if target else self.critics
        return torch.stack([c(scalars, edges, mask, action) for c in cs], 0)

    @torch.no_grad()
    def epistemic_v(self, scalars, edges, mask, mu) -> torch.Tensor:
        a = self.actor.squash(mu)
        return self.critic_stack(scalars, edges, mask, a).var(0, unbiased=False)

    def is_unknown(self, scalars, edges, mask) -> torch.Tensor:
        with torch.no_grad():
            feat, _, _ = self.src_actor.enc(scalars * self.src_actor.actor_scalar_keep, edges, mask)
            d = self.src_feat_bank.kdist(feat)[:, -1]
        return d > self.openset_thr

    def alpha0(self) -> float:
        return float(self.log_alpha0.detach().exp())

    def entropy_scale(self) -> float:
        """Factor on the entropy bonus. Source stage: 1 - 0.9 bc_weight (0.1 with full behaviour cloning).
        Target stage: the same 0.1, so the objective does not change at the source-to-target switch.
        Before, the target stage used the full alpha (10x larger) and an unnormalised Q term, and the
        source policy degenerated within the first target episodes."""
        if self.bc_weight > 0.0:
            return 1.0 - 0.9 * min(self.bc_weight, 1.0)
        return float(getattr(self, "target_entropy_scale", 0.1))

    def alpha_of(self, ubar: torch.Tensor, scalars=None, edges=None, mask=None) -> torch.Tensor:
        a0 = self.alpha0()
        alpha = a0 * (1.0 - ubar)
        if self.spec.method == "openset_entropy" and self.src_feat_bank is not None and scalars is not None:
            unknown = self.is_unknown(scalars, edges, mask)
            alpha = torch.where(unknown, torch.full_like(alpha, a0), -0.1 * a0 * torch.ones_like(alpha))
        return alpha

    def batch_context(self, b: Dict[str, torch.Tensor], nxt: bool) -> Dict[str, torch.Tensor]:
        p = "next_" if nxt else ""
        sc, ed, ms = b[p + "scalars"], b[p + "edges"], b[p + "mask"]
        ctx: Dict[str, torch.Tensor] = {}
        ctx["c_hat"] = self.causal_scores(sc, ed, ms, b.get(p + "c_or"))
        if self.spec.is_cast() and self.cov is not None:
            ctx.update(self.coverage_terms(ed, ms, b[p + "scene"], b[p + "dk"], b[p + "dk0"], b[p + "support"]))
        else:
            B, N = ms.shape
            ctx["ell_t"] = torch.zeros(B, N, device=self.device)
            ctx["ell0_t"] = torch.zeros(B, device=self.device)
            ctx["beta"] = torch.zeros(B, N, device=self.device)
        return ctx

    # ---------------- acting ----------------
    @torch.no_grad()
    def act(self, obs: Dict[str, np.ndarray], deterministic: bool, std_cap: Optional[float] = None) -> np.ndarray:
        """Use the same bounded distribution as SAC updates; evaluation uses its mean."""
        sc, ed, ms, _ = self.obs_tensors(obs, self.device)
        if std_cap is not None and not math.isclose(float(std_cap), float(self.cfg.target_explore_std)):
            raise ValueError("Acting and learning must use the same configured policy std cap")
        a, _, _, _, _, _, _ = self.actor.sample(sc, ed, ms, deterministic=deterministic)
        return policy_to_env(a[0].cpu().numpy())

    @torch.no_grad()
    def step_quantities(self, obs: Dict[str, np.ndarray], dk: Optional[np.ndarray], dk0: Optional[np.ndarray],
                        support: float, c_or: Optional[np.ndarray] = None) -> Dict[str, np.ndarray]:
        """Per-step decision uncertainty, attention, causal scores and floors for one observation."""
        sc, ed, ms, scene = self.obs_tensors(obs, self.device)
        mu, logstd, w, h = self.actor.dist_params(sc, ed, ms)
        _, logvar = self.actor.motion_head(h, ed)
        v = self.epistemic_v(sc, ed, ms, mu)
        b = {"scalars": sc, "edges": ed, "mask": ms, "scene": scene,
             "support": torch.tensor([support], device=self.device, dtype=torch.float32)}
        if dk is not None:
            b["dk"] = torch.from_numpy(dk[None]).to(self.device)
            b["dk0"] = torch.from_numpy(dk0[None]).to(self.device)
        if c_or is not None:
            b["c_or"] = torch.from_numpy(c_or[None].astype(np.float32)).to(self.device)
        ctx = self.batch_context(b, nxt=False)
        ub = self.ubar(w, logvar.exp(), ctx["ell_t"], ctx["ell0_t"], v)
        # the scene-free value is computed separately, not as clip(total) - scene term (clipping is not invertible)
        ub_noscene = self.ubar(w, logvar.exp(), ctx["ell_t"], ctx["ell0_t"], v, include_scene=False) \
            if (self.spec.is_cast() and self.spec.scene_term) else ub
        # unclipped components, logged to diagnose saturation (ubar = 1 drives alpha to zero)
        two_channel = self.spec.is_cast() and self.spec.epistemic and not self.spec.floor_penalty and self.cov is not None
        s2 = logvar.exp() / max(float(self.sig_ref), 1e-6)
        u_m = 1.0 - (1.0 - ctx["ell_t"]) * torch.exp(-s2) if two_channel else 1.0 - torch.exp(-s2)
        scene_on = self.spec.is_cast() and self.spec.scene_term and self.cov is not None
        out = {"ubar": float(ub[0]), "ubar_noscene": float(ub_noscene[0]), "w": w[0].cpu().numpy(),
               "ub_motion": float((w * u_m).sum(1)[0]), "ub_critic": float(self.v_cal.transform_tensor(v)[0]),
               "ub_scene": float(ctx["ell0_t"][0]) if scene_on else 0.0,
               "c_hat": ctx["c_hat"][0].cpu().numpy(), "ell_t": ctx["ell_t"][0].cpu().numpy(),
               "ell0_t": float(ctx["ell0_t"][0])}
        if self.cfg.value_audit_stride > 0:
            out["policy_entropy"] = float(squashed_gaussian_entropy(mu, logstd, self.actor.action_scale)[0])
            out["audit_alpha"] = float(self.alpha_of(ub, sc, ed, ms)[0])
        if "rho" in ctx:
            out["rho"] = ctx["rho"][0].cpu().numpy()
            out["rho0"] = float(ctx["rho0"][0])
        return out

    @torch.no_grad()
    def audit_action_values(self, obs, executed_action):
        """Same state/critic version for executed, policy mean, braking and driving actions."""
        sc, ed, ms, _ = self.obs_tensors(obs, self.device)
        mu, _, _, _ = self.actor.dist_params(sc, ed, ms)
        mean_action = self.actor.squash(mu)[0]
        actions = mean_action.repeat(4, 1)
        actions[0] = torch.as_tensor(executed_action, device=self.device, dtype=sc.dtype)
        actions[2, 0], actions[3, 0] = -0.95, 0.60
        qs = self.critic_stack(sc.expand(4, -1), ed.expand(4, -1, -1), ms.expand(4, -1), actions)
        values = qs.mean(0).cpu().tolist()
        return dict(zip(("q_executed", "q_policy_mean", "q_brake", "q_drive"), values),
                    executed_spread=float(qs[:, 0].std(unbiased=False)))

    @torch.no_grad()
    def reliance(self, obs: Dict[str, np.ndarray], slot: int, kind: int,
                 occluded: Optional[Tuple[np.ndarray, np.ndarray]] = None) -> float:
        """Token-level reliance r_i of Eq. (23): KL change of the policy under edits of token i
        (removal, which also models occlusion, plus freezing, velocity shift, or signal switch)."""
        sc, ed, ms, _ = self.obs_tensors(obs, self.device)
        mu, ls, _, _ = self.actor.dist_params(sc, ed, ms)
        edits = []
        if occluded is not None:
            # occluder painted into the camera images and the frozen detector run again
            edits.append((torch.from_numpy(occluded[0][None]).to(self.device),
                          torch.from_numpy(occluded[1][None]).to(self.device)))
        else:
            m2 = ms.clone()
            m2[0, slot] = 0.0
            edits.append((ed, m2))
        if kind in (KIND_VEHICLE, KIND_WALKER):
            e2 = ed.clone()
            e2[0, slot, 2] = -float(sc[0, 0]) * 10.0 / 15.0
            e2[0, slot, 3] = 0.0
            edits.append((e2, ms))
            e3 = ed.clone()
            e3[0, slot, 2] += 2.0 / 15.0
            edits.append((e3, ms))
        if kind == KIND_SIGNAL:
            e4 = ed.clone()
            sig_col = 4 + self.cfg.sem_dim
            e4[0, slot, sig_col] = 1.0 - e4[0, slot, sig_col]
            edits.append((e4, ms))
        kls = []
        for e_, m_ in edits:
            mu2, ls2, _, _ = self.actor.dist_params(sc, e_, m_)
            kls.append(float(gaussian_kl(mu, ls, mu2, ls2)[0]))
        return float(np.mean(kls))

    def policy_divergence(self, mu_p, logstd_p, mu_q, logstd_q) -> torch.Tensor:
        """Distance between two policies' actions for alignment (Eq. 16), invariance (Eq. 25) and uniform
        transfer. The policy std is capped at 0.05 and gaussian_kl clamps log-std at -3, so the KL reduces to
        200 x (mean gap)^2: measured on the rgbd7 checkpoints, KL(source || adapted) was 226 x the squared
        action gap (mean 13.6), so these terms outweighed the normalised RL term (about 2.5) and the
        imitation term (about 0.1) by one to two orders of magnitude. "action" measures the gap in the
        squashed action space, the units of behaviour cloning; "kl" keeps the earlier definition."""
        if str(getattr(self.cfg, "align_metric", "action")) == "kl":
            return gaussian_kl(mu_p, logstd_p, mu_q, logstd_q)
        return (self.actor.squash(mu_p) - self.actor.squash(mu_q)).pow(2).sum(-1)

    # ---------------- learning ----------------
    def update(self, b: Dict[str, torch.Tensor], source_batch: Optional[Dict[str, torch.Tensor]] = None,
               actor_step: bool = True, bc_batch: Optional[Dict[str, torch.Tensor]] = None,
               target_bc_batch: Optional[Dict[str, torch.Tensor]] = None) -> Dict[str, float]:
        cfg, sp = self.cfg, self.spec
        if sp.method in ("contrastive_filter", "ot_filter") and source_batch is not None:
            b = cat_batches(b, self.filter_source(b, source_batch))
        sc, ed, ms, act = b["scalars"], b["edges"], b["mask"], b["actions"]
        r, d = b["rewards"], b["terminals"]
        # ----- critic, Eq. (2) -----
        with torch.no_grad():
            ctx_n = self.batch_context(b, nxt=True)
            a2, logp2, mu2, _, w2, h2, _ = self.actor.sample(b["next_scalars"], b["next_edges"], b["next_mask"])
            _, logvar2 = self.actor.motion_head(h2, b["next_edges"])
            v2 = self.epistemic_v(b["next_scalars"], b["next_edges"], b["next_mask"], mu2)
            ub2 = self.ubar(w2, logvar2.exp(), ctx_n["ell_t"], ctx_n["ell0_t"], v2)
            alpha2 = self.alpha_of(ub2, b["next_scalars"], b["next_edges"], b["next_mask"])
            alpha2 = alpha2 * self.entropy_scale()
            qn_all = self.critic_stack(b["next_scalars"], b["next_edges"], b["next_mask"], a2, target=True)
            # Mean removes the minimum's downward selection offset. It can still
            # overestimate; retain the cap, log its effect, and keep min2 as an ablation.
            qn = reduce_critic_target(qn_all, cfg.critic_target, cfg.critic_disagreement_penalty)
            selection_offset = qn_all.mean(0) - qn
            y = r + (1.0 - d) * cfg.gamma * (qn - alpha2 * logp2)
            # Upper bound of the soft return: the largest per-step reward (w_p) plus the largest expected
            # entropy bonus, ALPHA_MAX * H_max, forever, then the goal. ALPHA_MAX is a global bound on the
            # temperature (alpha_0 is clamped to <= 0.5, alpha = alpha_0 (1 - ubar) <= alpha_0, and the
            # entropy scale is <= 1), not the current batch's maximum. H_max is the entropy of the action
            # distribution at the configured log-std bound; the tanh only lowers it. The
            # late source stage logged Q = 130-170 against a reward-only bound of 35 (overestimation the actor
            # exploits). The clip acts on sampled targets, so it is a heuristic bound on an expectation; the
            # fraction of clipped targets is logged (clip=) so its effect is visible.
            # Negative entropy contributes no positive bonus. Do not multiply a
            # negative entropy bound by ALPHA_MAX and call that an upper bound.
            h_max = max(0.0, float(cfg.policy_action_dim) *
                        (math.log(float(cfg.target_explore_std)) + 0.5 * math.log(2.0 * math.pi * math.e)))
            q_hi = 1.05 * ((float(cfg.w_p) + ALPHA_MAX * h_max) / (1.0 - float(cfg.gamma))
                           + max(float(cfg.r_goal), 0.0))
            self.clip_frac = 0.99 * float(getattr(self, "clip_frac", 0.0)) + 0.01 * float((y > q_hi).float().mean())
            target_before_cap = float(y.mean())
            y = y.clamp(max=q_hi)
        qs = self.critic_stack(sc, ed, ms, act)
        boot = (torch.rand_like(qs) < 0.8).float()
        critic_loss = (((qs - y.unsqueeze(0)) ** 2) * boot).sum(1).div(boot.sum(1).clamp(min=1.0)).mean()
        rejected = b["has_rejected"] > 0.5
        takeover_loss = qs.new_zeros(())
        if cfg.takeover_loss_weight > 0.0 and bool(rejected.any()):
            rq = self.critic_stack(sc[rejected], ed[rejected], ms[rejected], b["rejected_action"][rejected])
            # Synthetic absorbing transition: target = penalty, zero bootstrap.
            # Do not replace the observed teacher-action transition or fabricate a crash.
            takeover_loss = (rq - float(cfg.takeover_penalty)).pow(2).mean()
            critic_loss = critic_loss + float(cfg.takeover_loss_weight) * takeover_loss
        self.critic_opt.zero_grad(set_to_none=True)
        critic_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.critics.parameters(), 10.0)
        self.critic_opt.step()
        critic_stats = {"td_abs": float((qs.detach() - y.unsqueeze(0)).abs().mean()),
                        "td_target": float(y.mean()), "batch_reward": float(r.mean()),
                        "critic_spread": float(qs.detach().std(0, unbiased=False).mean()),
                        "target_selection_offset": float(selection_offset.mean()),
                        "target_before_cap": target_before_cap,
                        "takeover_loss": float(takeover_loss.detach()),
                        "takeover_batch_n": float(rejected.sum())}
        if not actor_step:
            # critic warm-up / policy delay: only the critic and its target networks move
            with torch.no_grad():
                for c, t in zip(self.critics, self.target_critics):
                    for p_, tp in zip(c.parameters(), t.parameters()):
                        tp.data.mul_(1.0 - cfg.target_tau).add_(cfg.target_tau * p_.data)
            self.updates += 1
            st = dict(getattr(self, "last_stats", {}) or {})
            st["critic_loss"] = float(critic_loss.detach())
            st["critic_q"] = float(qs.detach().mean())
            st["actor_updated"] = 0.0
            st["clip_frac"] = float(getattr(self, "clip_frac", 0.0))
            st.update(critic_stats)
            self.last_stats = st
            return st

        # ----- actor, Eqs. (3), (5), (17)-(19), (25), (29) -----
        ctx = self.batch_context(b, nxt=False)
        action, logp, mu, logstd, w, h, logp_dims = self.actor.sample(sc, ed, ms)
        qa = self.critic_stack(sc, ed, ms, action)
        m_hat, logvar = self.actor.motion_head(h, ed)
        with torch.no_grad():
            v = self.epistemic_v(sc, ed, ms, mu)
            ub = self.ubar(w.detach(), logvar.exp().detach(), ctx["ell_t"], ctx["ell0_t"], v)
            alpha = self.alpha_of(ub, sc, ed, ms)
            # The entropy bonus is scaled the same way in the source and the target stage.
            alpha = alpha * self.entropy_scale()
        q_hat = qa.mean(0)
        if sp.is_cast() and sp.cautious and self.cov is not None:
            q_hat = q_hat - sp.kappa * ub * qa.std(0)
        if sp.method == "ace":
            ent = alpha * (logp_dims * (cfg.policy_action_dim * self.ace_w)).sum(-1)
        else:
            ent = alpha * logp
        loss = (ent - q_hat).mean()
        if self.normalize_q:
            # The critic is trained mostly on teacher actions, so its action gradient is unreliable
            # elsewhere; normalising the RL term keeps it balanced against behaviour cloning.
            loss = loss * (2.5 / q_hat.abs().mean().detach().clamp(min=2.5))
        # Learn imitation first; introduce the critic gradient gradually during handover.
        loss = loss * self.actor_rl_weight
        stats: Dict[str, float] = {}
        if self.bc_weight > 0.0 and "has_expert" in b:
            if bc_batch is not None:
                imitation = augment_hazard(augment_token_uncertainty(bc_batch, float(cfg.sigma_aug_prob)), cfg)
                bc_mu, _, _, _ = self.actor.dist_params(imitation["scalars"], imitation["edges"], imitation["mask"])
                l_bc, bc_stats = balanced_imitation_loss(self.actor.squash(bc_mu), imitation)
            else:
                l_bc, bc_stats = balanced_imitation_loss(self.actor.squash(mu), b)
            loss = loss + self.bc_weight * l_bc
            stats["L_BC"] = float(l_bc.detach())
            stats.update(bc_stats)
        anchor_batch = bc_batch if bc_batch is not None else source_batch
        if self.src_actor is not None and anchor_batch is not None and cfg.target_bc_weight > 0.0:
            # Keep expert behavior on source replay while target RL learns. This uses saved
            # source teacher labels, not the weak source actor as its own teacher.
            anchor = augment_hazard(augment_token_uncertainty(anchor_batch, float(cfg.sigma_aug_prob)), cfg)
            mu_anchor, _, _, _ = self.actor.dist_params(anchor["scalars"], anchor["edges"], anchor["mask"])
            l_anchor, anchor_stats = balanced_imitation_loss(self.actor.squash(mu_anchor), anchor)
            loss = loss + float(cfg.target_bc_weight) * l_anchor
            stats["L_source_BC"] = float(l_anchor.detach())
            stats.update({"source_" + k: value for k, value in anchor_stats.items()})
        if self.src_actor is not None and target_bc_batch is not None and cfg.target_state_bc_weight > 0.0:
            # Privileged expert labels from uncertain target states; opt-in assisted adaptation.
            target_bc_batch = augment_hazard(target_bc_batch, cfg)
            tm, _, _, _ = self.actor.dist_params(target_bc_batch["scalars"], target_bc_batch["edges"],
                                                target_bc_batch["mask"])
            l_target, target_stats = balanced_imitation_loss(self.actor.squash(tm), target_bc_batch)
            loss = loss + float(cfg.target_state_bc_weight) * l_target
            stats["L_target_BC"] = float(l_target.detach())
            stats.update({"target_" + k: value for k, value in target_stats.items()})
        # motion likelihood, Eq. (5)
        mm = b["motion_mask"] * ms
        diff2 = (b["motion_target"] - m_hat).pow(2).sum(-1)
        nll = (diff2 / (2.0 * logvar.exp()) + logvar) * mm
        l_m = nll.sum() / mm.sum().clamp(min=1.0)
        loss = loss + l_m
        stats["motion_nll"] = float(l_m.detach())
        if sp.is_cast() and self.src_actor is not None:
            # Eq. (16): omega_i = c_hat_i beta_i; without the gate every causal interaction is aligned.
            omega = ctx["c_hat"] * (ctx["beta"] if sp.gate else ms)
            hS = self.h_source(ed).detach()
            wd = w.detach()
            l_emb = (wd * omega * (h - hS).pow(2).sum(-1)).sum(1)
            with torch.no_grad():
                mu_s, ls_s, _, _ = self.src_actor.dist_params(sc, ed, ms)
            omega_bar = (wd * omega).sum(1)
            l_a = (l_emb + omega_bar * self.policy_divergence(mu_s, ls_s, mu, logstd)).mean()
            loss = loss + cfg.lambda_A * l_a
            stats["L_A"] = float(l_a.detach())
            if sp.floor_penalty:
                u_ale = 1.0 - torch.exp(-logvar.exp() / max(float(self.sig_ref), 1e-6))
                l_u = (ctx["c_hat"] * (ctx["ell_t"] - u_ale).clamp(min=0.0).pow(2)).sum(1).mean()
                loss = loss + l_u
                stats["L_U"] = float(l_u.detach())
            if sp.invariance:
                l_i = self.invariance_loss(sc, ed, ms, mu, logstd, ctx["c_hat"])
                loss = loss + cfg.lambda_I * l_i
                stats["L_I"] = float(l_i.detach())
        if sp.method == "uniform" and self.src_actor is not None:
            l_t = self.uniform_transfer_loss(b, source_batch, mu, logstd, w, logvar)
            loss = loss + cfg.lambda_transfer * l_t
            stats["L_trans"] = float(l_t.detach())
        self.actor_opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.actor.parameters(), 10.0)
        self.actor_opt.step()
        if sp.method != "openset_entropy" and getattr(self, "tune_alpha", True):
            # SAC temperature update: raise alpha_0 when the policy entropy is below the target, lower it otherwise.
            alpha_loss = -(self.log_alpha0 * (logp.detach() + self.target_entropy)).mean()
            self.alpha_opt.zero_grad(set_to_none=True)
            alpha_loss.backward()
            self.alpha_opt.step()
            with torch.no_grad():
                # The RL term of the actor loss is normalised to |Q| ~ 2.5, so alpha_0 above 0.5 would let
                # the entropy bonus outweigh the value; cap it there.
                self.log_alpha0.clamp_(math.log(1e-3), math.log(ALPHA_MAX))
        with torch.no_grad():
            for c, t in zip(self.critics, self.target_critics):
                for p, tp in zip(c.parameters(), t.parameters()):
                    tp.data.mul_(1.0 - cfg.target_tau).add_(cfg.target_tau * p.data)
        self.updates += 1
        self.motion_nll_ema = 0.99 * getattr(self, "motion_nll_ema", stats["motion_nll"]) + 0.01 * stats["motion_nll"]
        stats.update({"critic_loss": float(critic_loss.detach()), "actor_loss": float(loss.detach()),
                      "ubar": float(ub.mean()), "alpha": float(alpha.mean()), "alpha0": self.alpha0(),
                      "entropy": float(-logp.detach().mean()), "q": float(q_hat.detach().mean()),
                      "clip_frac": float(getattr(self, "clip_frac", 0.0)),
                      "critic_q": float(qs.detach().mean()), "actor_updated": 1.0})
        stats.update(critic_stats)
        stats["policy_std"] = float(logstd.detach().exp().mean())
        self.last_stats = stats
        return stats

    def invariance_loss(self, sc, ed, ms, mu, logstd, c_hat) -> torch.Tensor:
        """Eq. (25) with one random valid token per state, re-weighted by the number of valid tokens."""
        nvalid = ms.sum(1)
        ok = nvalid > 1.5
        if not bool(ok.any()):
            return torch.zeros((), device=self.device)
        probs = ms[ok] / ms[ok].sum(1, keepdim=True)
        j = torch.multinomial(probs, 1).squeeze(1)
        m2 = ms[ok].clone()
        m2[torch.arange(m2.shape[0]), j] = 0.0
        mu2, ls2, _, _ = self.actor.dist_params(sc[ok], ed[ok], m2)
        wgt = (1.0 - c_hat[ok][torch.arange(m2.shape[0]), j]) * nvalid[ok]
        return (wgt * self.policy_divergence(mu[ok], logstd[ok], mu2, ls2)).sum() / ms.shape[0]

    def uniform_transfer_loss(self, b, sb, mu, logstd, w, logvar) -> torch.Tensor:
        """Uniform alignment of Borhan et al.: KL on actions, MMD on attention, uncertainty-moment matching."""
        sc, ed, ms = b["scalars"], b["edges"], b["mask"]
        with torch.no_grad():
            mu_s, ls_s, _, _ = self.src_actor.dist_params(sc, ed, ms)
        kl = self.policy_divergence(mu_s, ls_s, mu, logstd).mean()
        if sb is None:
            return kl
        with torch.no_grad():
            _, _, w_s, h_s = self.src_actor.dist_params(sb["scalars"], sb["edges"], sb["mask"])
            _, lv_s = self.src_actor.motion_head(h_s, sb["edges"])
            u_s = 1.0 - torch.exp(-lv_s.exp() / max(float(self.sig_ref), 1e-6))
            mom_s = moments(u_s, w_s, sb["mask"])
        u_t = 1.0 - torch.exp(-logvar.exp() / max(float(self.sig_ref), 1e-6))
        mom_t = moments(u_t, w, ms)
        mmd = mmd_rbf(w_s, w)
        return kl + self.cfg.lambda_alpha_align * mmd + self.cfg.lambda_u_align * F.mse_loss(mom_t, mom_s)

    @torch.no_grad()
    def filter_source(self, b: Dict[str, torch.Tensor], sb: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """Source-data filtering baselines (adapted): keep the source transitions most relevant to the target."""
        keep = self.src_actor.actor_scalar_keep
        ft, _, _ = self.src_actor.enc(b["scalars"] * keep, b["edges"], b["mask"])
        fs, _, _ = self.src_actor.enc(sb["scalars"] * keep, sb["edges"], sb["mask"])
        if self.spec.method == "contrastive_filter":
            score = -torch.cdist(fs, ft).min(1).values
            keep = torch.topk(score, max(1, fs.shape[0] // 4)).indices
        else:
            C = torch.cdist(fs, ft).pow(2)
            C = C / C.median().clamp(min=1e-6)
            K = torch.exp(-C / 0.05)
            u = torch.ones(fs.shape[0], device=fs.device)
            v = torch.ones(ft.shape[0], device=fs.device)
            a = u / u.sum()
            bb = v / v.sum()
            for _ in range(50):
                u = a / (K @ v).clamp(min=1e-12)
                v = bb / (K.t() @ u).clamp(min=1e-12)
            # Balanced Sinkhorn fixes every row's mass at 1/n_s, so rows are ranked by their mean transport
            # cost to the target batch (low cost = close to the target distribution).
            P = u[:, None] * K * v[None, :]
            cost = (P * C).sum(1) / P.sum(1).clamp(min=1e-12)
            keep = torch.nonzero(cost <= cost.median()).squeeze(1)
        out = {k: v[keep] for k, v in sb.items() if k in b}
        for k in b:
            if k not in out:
                out[k] = torch.zeros((keep.shape[0],) + tuple(b[k].shape[1:]), device=self.device)
        return out

    def train_causal_head(self, data: CausalDataset, nulls: NullBank, steps: int, bs: int = 256) -> float:
        if data.n < bs:
            return float("nan")
        loss_v = 0.0
        for _ in range(int(steps)):
            idx = np.random.randint(0, data.n, size=bs)
            y = torch.from_numpy(data.labels(idx, nulls, self.spec.relevance, self.spec.null_mode)).to(self.device)
            x0 = torch.from_numpy(data.x0[idx]).to(self.device)
            xi = torch.from_numpy(data.xi[idx]).to(self.device)
            pred = self.causal_head(x0, xi)
            loss = F.mse_loss(pred, y)
            self.causal_opt.zero_grad(set_to_none=True)
            loss.backward()
            self.causal_opt.step()
            loss_v = float(loss.detach())
        return loss_v

    def update_ace_weights(self, actions: np.ndarray, rewards: np.ndarray) -> None:
        if actions.shape[0] < 100:
            return
        c = [abs(float(np.corrcoef(actions[:, d], rewards)[0, 1])) if np.std(actions[:, d]) > 1e-6 else 0.0
             for d in range(actions.shape[1])]
        c = np.nan_to_num(np.asarray(c)) + 1e-3
        self.ace_w = torch.from_numpy((c / c.sum()).astype(np.float32)).to(self.device)


def moments(u: torch.Tensor, w: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    m = mask.sum(1).clamp(min=1.0)
    mean = (u * mask).sum(1) / m
    var = (((u - mean.unsqueeze(1)) * mask) ** 2).sum(1) / m
    return torch.stack([(w * u).sum(1).mean(), mean.mean(), var.mean()])


def copy_module(m: nn.Module) -> nn.Module:
    import copy as _copy
    c = _copy.deepcopy(m)
    for p in c.parameters():
        p.requires_grad_(False)
    c.eval()
    return c



# ======================================================
# Suites, templates and evaluation routes (Sec. 4.1)
# ======================================================
@dataclass
class Suite:
    name: str
    templates: List[Conditions]
    eval_routes: List[Conditions]
    distractors: bool = True          # random barrels / green queues inside target templates


def town_base(name: str) -> str:
    return name.split("/")[-1].replace("_Opt", "")


def available_towns(host: str, port: int) -> List[str]:
    wait_for_carla_ready(host, port, timeout_s=120.0)
    client = carla.Client(host, port)
    client.set_timeout(30.0)
    return sorted({town_base(m) for m in client.get_available_maps()})


def resolve_towns(requested: Sequence[str], available: Sequence[str], label: str, log: List[List[str]]) -> List[str]:
    out = []
    for t in requested:
        if town_base(t) in available:
            out.append(t)
            log.append([label, t, "used"])
        else:
            print(f"[WARN] {label}: town {t} is not installed on this CARLA server and is skipped")
            log.append([label, t, "missing"])
    if not out:
        raise RuntimeError(f"No town of the {label} suite is installed: {list(requested)}")
    return out


def build_suites(args: argparse.Namespace, cfg: Config, available: Sequence[str],
                 log: List[List[str]]) -> Dict[str, Suite]:
    src = resolve_towns(args.source_towns.split(","), available, "source", log)
    lay = resolve_towns(args.layout_towns.split(","), available, "layout", log)
    nlo, nhi = int(cfg.train_npc_min), int(cfg.train_npc_max)
    wlo, whi = int(cfg.train_walker_min), int(cfg.train_walker_max)

    def grid(towns, weathers, degs, support):
        return [Conditions(t, w, n, k, d, in_source_support=support and d == 0.0)
                for t in towns for w in weathers for d in degs for n in (nlo, nhi) for k in (wlo, whi)]

    def routes(templates: List[Conditions], n: int, seed0: int, barrel="none", queue="none") -> List[Conditions]:
        """n evaluation routes spread evenly over the towns of the suite, grouped by town so that
        each town is loaded once per evaluation."""
        rng = random.Random(seed0)
        towns = sorted({c.town for c in templates})
        out = []
        for i in range(n):
            town = towns[i % len(towns)]
            c = rng.choice([t for t in templates if t.town == town])
            out.append(Conditions(c.town, c.weather, rng.randint(nlo, nhi), rng.randint(wlo, whi), c.degradation,
                                  barrel, queue, c.in_source_support, route_seed=seed0 + i))
        out.sort(key=lambda c: (towns.index(c.town), c.route_seed))
        return out

    src_t = grid(src, SOURCE_WEATHERS, (0.0,), True)
    a_t = grid(src, ADVERSE_WEATHERS, (0.0,), False)
    l_t = grid(lay, SOURCE_WEATHERS, (0.0,), False)
    c_t = grid(lay, ADVERSE_WEATHERS, (0.5, 1.0), False)
    n_eval, n_scr = int(args.eval_routes), int(args.scripted_routes)
    return {
        "source": Suite("source", src_t, routes(src_t, n_eval, 50000), distractors=False),
        "A": Suite("A", a_t, routes(a_t, n_eval, 10000)),
        "L": Suite("L", l_t, routes(l_t, n_eval, 20000)),
        "C": Suite("C", c_t, routes(c_t, n_eval, 30000)),
        "barrel": Suite("barrel", l_t, routes(l_t, n_scr, 40000, barrel="random"), distractors=False),
        "queue": Suite("queue", l_t, routes(l_t, n_scr, 45000, queue="green_queue"), distractors=False),
    }


def with_distractors(c: Conditions, enabled: bool) -> Conditions:
    if not enabled:
        return c
    u = random.random()
    b = "random" if u < 0.2 else "none"
    q = "green_queue" if 0.2 <= u < 0.4 else "none"
    return Conditions(c.town, c.weather, c.npc, c.walkers, c.degradation, b, q, c.in_source_support)


# ======================================================
# CARLA server control (used only with --carla-cmd)
# ======================================================
def write_runtime_startup(out_dir: str, phase: str, town: str, tm_port: int,
                          client_version: str, server_version: str) -> None:
    """Write before entering native code, where SIGSEGV cannot be caught in Python."""
    path = os.path.join(os.path.abspath(out_dir), "runtime_startup.json")
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        json.dump(dict(phase=phase, town=town, tm_port=int(tm_port), pid=os.getpid(),
                       client_version=client_version, server_version=server_version,
                       api_module=str(getattr(carla, "__file__", "?"))), f, indent=2)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)


def repair_town03_tm_cache(args: argparse.Namespace) -> List[str]:
    """Explicit, reversible workaround for CARLA issue #7210; never touch Nav caches.

    Scope is the supplied local installation, ~/carlaCache and this run's cache.
    The flag authorizes backups, not deletion, broad cleanup or changing the map.
    Packaged .pak content cannot be repaired by this loose-file workaround.
    """
    from pathlib import Path
    import shlex
    import uuid
    if str(args.host).lower() not in ("localhost", "127.0.0.1", "::1"):
        raise ValueError("Cache repair requires a local CARLA server; repair the remote installation there")
    if not args.carla_cmd:
        raise ValueError("--repair-town03-tm-cache requires --carla-cmd to restart CARLA after the backup")
    root_text = str(getattr(args, "carla_root", "")).strip()
    if root_text:
        root = Path(root_text).expanduser().resolve()
    else:
        executable = Path(shlex.split(os.path.expanduser(args.carla_cmd))[0]).expanduser()
        if not executable.is_file():
            raise ValueError("Cannot infer the CARLA installation; pass --carla-root /path/to/carla")
        root = executable.resolve().parent
    if not root.is_dir():
        raise ValueError(f"CARLA installation not found: {root}")
    roots = [root, Path.home() / "carlaCache", Path(args.out_dir).resolve() / "carla_cache"]
    candidates = set()
    for base in roots:
        if not base.is_dir():
            continue
        for path in base.glob("**/Carla/Maps/TM/Town03_Opt.bin"):
            if not path.is_file() or path.is_symlink():
                continue
            # Do not traverse a directory symlink out of the requested root.
            if os.path.commonpath([str(base.resolve()), str(path.resolve())]) == str(base.resolve()):
                candidates.add(path.resolve())
    if not candidates:
        raise RuntimeError("No loose Carla/Maps/TM/Town03_Opt.bin found in " + ", ".join(map(str, roots)) +
                           ". No files changed. Check --carla-root; packaged .pak files need a corrected map package.")
    suffix = ".cast-backup-" + time.strftime("%Y%m%dT%H%M%S", time.gmtime()) + "-" + uuid.uuid4().hex[:8]
    backups = []
    for path in sorted(candidates):
        backup = path.with_name(path.name + suffix)
        path.rename(backup)
        backups.append(str(backup))
        print(f"[TM REPAIR] backed up {path} -> {backup}", flush=True)
    return backups


def carla_command(args: argparse.Namespace, town: Optional[str] = None) -> List[str]:
    """The --carla-cmd command line. With a town, the map is given as the first argument
    (CarlaUE4.sh /Game/Carla/Maps/Town02 ...), so the server starts directly in that map and
    client.load_world() is never called."""
    import shlex
    cmd = os.path.expanduser(str(args.carla_cmd))
    if "carla-rpc-port" not in cmd:
        cmd += f" -carla-rpc-port={int(args.port)}"
    parts = shlex.split(cmd)
    if town:
        parts = parts[:1] + [map_path(town)] + [x for x in parts[1:] if not x.startswith("/Game/")]
    return parts


def map_path(town: str) -> str:
    return town if town.startswith("/Game/") else f"/Game/Carla/Maps/{town}"


def _is_carla_server_process(comm: str, argv: Sequence[str], port: int) -> bool:
    """Match only an actual CARLA binary/launcher and this exact RPC-port argument."""
    if not argv or f"-carla-rpc-port={int(port)}" not in argv:
        return False
    name = str(comm).lower()
    executable = os.path.basename(str(argv[0])).lower()
    if name.startswith("carlaue4") or executable.startswith("carlaue4"):
        return True
    return (executable in ("sh", "bash", "dash") and len(argv) > 1
            and os.path.basename(str(argv[1])).lower() == "carlaue4.sh")


def _carla_server_pids(port: int) -> List[int]:
    """Identify only CARLA processes on this port, excluding Python and shell command strings."""
    import subprocess
    try:
        out = subprocess.run(["pgrep", "-f", f"carla-rpc-port={int(port)}"],
                             capture_output=True, text=True).stdout
    except Exception:
        return []
    excluded = {os.getpid(), os.getppid()}
    pids = []
    for token in out.split():
        try:
            pid = int(token)
            if pid in excluded:
                continue
            with open(f"/proc/{pid}/comm") as f:
                comm = f.read().strip()
            with open(f"/proc/{pid}/cmdline", "rb") as f:
                argv = [x.decode(errors="replace") for x in f.read().split(b"\0") if x]
            if _is_carla_server_process(comm, argv, port):
                pids.append(pid)
        except (OSError, ValueError):
            continue
    return pids


def _carla_startup_log_tail(path: str, offset: int = 0) -> str:
    """Only the current attempt's last 8 KiB/30 lines, never the full append-only log."""
    try:
        with open(path, "rb") as f:
            f.seek(0, os.SEEK_END)
            f.seek(max(int(offset), f.tell() - 8192, 0))
            return "\n".join(f.read().decode(errors="replace").splitlines()[-30:])
    except OSError as exc:
        return f"Could not read server log: {exc}"


def _record_server_startup(out_dir: str, record: Dict[str, object]) -> None:
    path = os.path.join(out_dir, "server_startup.json")
    temporary = path + ".tmp"
    with open(temporary, "w") as f:
        json.dump(record, f, indent=2)
    os.replace(temporary, path)


def _stop_carla_launch(process: object) -> None:
    """Stop only the process group owned by this launch, including its UE4 child."""
    import signal
    import subprocess
    try:
        os.killpg(int(process.pid), signal.SIGKILL)
    except ProcessLookupError:
        pass
    try:
        process.wait(timeout=10.0)
    except subprocess.TimeoutExpired:
        pass


def restart_carla_server(args: argparse.Namespace, town: Optional[str] = None) -> None:
    """Bounded, checkpoint-preserving server startup recovery for every map switch.

    Retain the launcher handle, inspect its exit status, retry failed readiness/map
    loads, and print the current attempt's log tail. Training checkpoints are never
    edited here. Persistent server/driver failures still require diagnosis.
    """
    import signal
    import subprocess
    attempts = int(getattr(args, "carla_startup_attempts", 3))
    timeout = float(getattr(args, "carla_startup_timeout", 600.0))
    if attempts < 1 or not math.isfinite(timeout) or timeout <= 0.0:
        raise SystemExit("--carla-startup-attempts must be positive and --carla-startup-timeout finite and positive")
    out_dir = os.path.abspath(args.out_dir)
    log_dir = os.path.join(out_dir, "logs")
    os.makedirs(log_dir, exist_ok=True)
    log_path = os.path.join(log_dir, "carla_server.log")
    probe_log_path = os.path.join(log_dir, "carla_rpc_probe.log")
    cmd = carla_command(args, town)
    last_error = ""
    for attempt in range(1, attempts + 1):
        process = None
        offset = os.path.getsize(log_path) if os.path.exists(log_path) else 0
        record = {"phase": "starting", "town": town, "host": args.host, "port": int(args.port),
                  "attempt": attempt, "attempts": attempts, "timeout_s": timeout, "log": log_path,
                  "probe_log": probe_log_path, "isolated_rpc": True}
        try:
            # Only actual CARLA processes with an exact port match are selected.
            for pid in _carla_server_pids(int(args.port)):
                try:
                    os.kill(pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
            for _ in range(60):
                if not _carla_server_pids(int(args.port)):
                    break
                time.sleep(1.0)
            if _carla_server_pids(int(args.port)):
                raise RuntimeError("Previous CARLA processes are still alive; refusing an overlapping launch")
            time.sleep(5.0)
            print(f"[CARLA] startup attempt {attempt}/{attempts}: {' '.join(cmd)}", flush=True)
            with open(log_path, "ab", buffering=0) as server_log:
                marker = f"\n=== CaST startup attempt {attempt}/{attempts}, town={town}, port={args.port} ===\n"
                server_log.write(marker.encode())
                process = subprocess.Popen(cmd, start_new_session=True, stdout=server_log, stderr=subprocess.STDOUT)
            record["pid"] = int(process.pid)
            _record_server_startup(out_dir, record)
            print(f"[CARLA] server output: {log_path}", flush=True)
            print(f"[CARLA] isolated RPC probe log: {probe_log_path}", flush=True)
            with open(probe_log_path, "a") as probe_log:
                probe_log.write(marker)
            wait_for_carla_ready(args.host, int(args.port), timeout_s=timeout, process=process,
                                 probe_log_path=probe_log_path)
            time.sleep(10.0)
            if process.poll() is not None:
                raise RuntimeError(f"CARLA launcher exited with code {process.returncode} during map streaming")
            if town and not server_town_matches(args, town):
                print(f"[CARLA] startup map differs; loading {town} once through the API", flush=True)
                if not load_town_in_child(args, town):
                    raise RuntimeError(f"CARLA did not load {town}")
            record["phase"] = "ready"
            _record_server_startup(out_dir, record)
            return
        except KeyboardInterrupt:
            if process is not None:
                _stop_carla_launch(process)
            print("[CARLA] startup interrupted; training checkpoints retained", flush=True)
            raise
        except (OSError, RuntimeError, TimeoutError) as exc:
            last_error = str(exc)
            record.update(phase="failed", error=last_error,
                          returncode=process.poll() if process is not None else None)
            _record_server_startup(out_dir, record)
            print(f"[CARLA] startup attempt {attempt}/{attempts} failed: {last_error}", flush=True)
            if process is not None:
                _stop_carla_launch(process)
            tail = _carla_startup_log_tail(log_path, offset)
            if tail:
                print(f"[CARLA] current-attempt log tail:\n{tail}", flush=True)
            # A missing/nonexecutable launch command will not heal by retrying.
            if isinstance(exc, (FileNotFoundError, PermissionError)):
                break
            if attempt < attempts:
                time.sleep(10.0)
    raise SystemExit(f"[CARLA] server startup failed; checkpoint retained. Last error: {last_error}. "
                     f"Inspect {log_path} and {os.path.join(out_dir, 'server_startup.json')}; "
                     "fix the server issue and rerun the SAME training command and --out-dir.")


class TownSwitch(Exception):
    """Raised when a different map is needed. The worker saves its progress and exits with code 75;
    the supervisor, which owns no Traffic Manager, loads the map and starts the worker again.
    A Traffic Manager lives inside the client process, and a long server-side map load makes its
    threads time out and abort that process, so maps are never loaded by the training process."""

    def __init__(self, town: str):
        super().__init__(town)
        self.town = town


TOWN_SWITCH_EXIT = 75
SIM_DOWN_EXIT = 70


class SimulatorDown(RuntimeError):
    """The CARLA server stopped answering: a call waited the full client time-out (300 s)."""


def server_unresponsive(env: object, e: BaseException) -> bool:
    """True when an exception means the CARLA server hangs. The flag makes env.close() skip the actor
    clean-up: after a frozen tick every destroy() waited 300 s, and 50 actors hung the worker for hours."""
    msg = str(e).lower()
    down = isinstance(e, SimulatorDown) or bool(getattr(env, "_server_down", False)) or \
        ("time-out" in msg and "waiting for the simulator" in msg)
    if down and env is not None:
        try:
            env._server_down = True
            env.client.set_timeout(2.0)
        except BaseException:
            pass
    return down


def current_server_town(host: str, port: int) -> str:
    # A short-lived in-process client can leave C++ threads alive after return.
    # Keep map queries isolated too, including the post-load verification.
    return str(_carla_probe_in_child(host, int(port), timeout_s=60.0, include_map=True)["map"])


def server_town_matches(args: argparse.Namespace, town: str) -> bool:
    try:
        wait_for_carla_ready(args.host, int(args.port), timeout_s=60.0)
        return town_base(current_server_town(args.host, int(args.port))) == town_base(town)
    except Exception as exc:
        print(f"[MAP] could not verify {town} at {args.host}:{args.port}: {exc}", flush=True)
        return False


def load_town_in_child(args: argparse.Namespace, town: str) -> bool:
    """client.load_world() in a throw-away child process: a C++ time-out inside load_world aborts
    the whole process, so it must never run inside the supervisor."""
    import subprocess
    code = ("import carla,sys;c=carla.Client(sys.argv[1],int(sys.argv[2]));c.set_timeout(600.0);"
            "w=c.get_world();s=w.get_settings();s.synchronous_mode=False;s.fixed_delta_seconds=None;"
            "w.apply_settings(s);c.load_world(sys.argv[3],reset_settings=True)")
    try:
        env = os.environ.copy()
        env["PYTHONPATH"] = os.pathsep.join([str(p) for p in sys.path if p] +
                                          ([env["PYTHONPATH"]] if env.get("PYTHONPATH") else []))
        rc = subprocess.call([sys.executable, "-c", code, args.host, str(int(args.port)), map_path(town)],
                             timeout=900, env=env)
        if rc != 0:
            print(f"[MAP] loader exited {rc}; requested={town}", flush=True)
            return False
    except Exception as exc:
        print(f"[MAP] loader failed: {exc}", flush=True)
        return False
    time.sleep(10.0)
    return server_town_matches(args, town)


def prepare_town(args: argparse.Namespace, town: str) -> None:
    """Make the CARLA server run `town` before the worker starts (each town in its own server run).
    With --carla-cmd the server is restarted with the map on its command line. Without it, one map
    load is tried in a child process; if that fails, the run waits until CARLA is restarted by hand
    with the printed command."""
    if server_town_matches(args, town):
        return
    if args.carla_cmd:
        restart_carla_server(args, town)
        print(f"[SUPERVISOR] CARLA running {town}")
        return
    if bool(getattr(args, "allow_map_load", False)):
        print(f"[SUPERVISOR] loading {town} in a child process (this can take a few minutes)")
        if load_town_in_child(args, town):
            print(f"[SUPERVISOR] {town} loaded")
            return
    print("\n" + "=" * 78 + f"\n[SUPERVISOR] >>> restart CARLA in {town} (in your CARLA shell: Ctrl+C, then)\n"
          f"    ./CarlaUE4.sh {map_path(town)} -quality-level=Low -nosound -carla-rpc-port={int(args.port)}\n"
          f"  Training is saved and continues by itself once CARLA runs {town_base(town)}.\n" + "=" * 78)
    while not server_town_matches(args, town):
        time.sleep(15.0)
    print(f"[SUPERVISOR] CARLA running {town}")


# ======================================================
# Environment manager
# ======================================================
class EnvManager:
    def __init__(self, args: argparse.Namespace, cfg: Config):
        self.args, self.cfg = args, cfg
        self.env: Optional[CaSTCarlaEnv] = None
        self.town: Optional[str] = None
        self.detector: Optional[BEVDetector] = None
        self.device = torch.device("cpu")

    def get(self, town: str) -> CaSTCarlaEnv:
        if self.env is not None and self.town == town:
            return self.env
        self.close()
        wait_for_carla_ready(self.args.host, self.args.port, timeout_s=300.0)
        if town_base(current_server_town(self.args.host, self.args.port)) != town_base(town):
            raise TownSwitch(town)
        for attempt in range(4):
            try:
                env = CaSTCarlaEnv(self.args.host, self.args.port, town, self.cfg)
                if town_base(env.town_name) != town_base(town):
                    env.close()
                    raise TownSwitch(town)
                env.set_detector(self.detector, self.device)
                self.env, self.town = env, town
                try:
                    with open(os.path.join(self.cfg.out_dir, "active_town.json"), "w") as f:
                        json.dump({"town": town}, f)      # the supervisor restarts CARLA in this town after a crash
                except BaseException:
                    pass
                return env
            except TownSwitch:
                raise
            except BaseException as e:
                print(f"[WARN] env build for {town} failed ({attempt + 1}/4): {e}")
                time.sleep(5.0 * (attempt + 1))
        raise RuntimeError(f"could not build the environment for {town}")

    def reset(self, cond: Conditions) -> Tuple[CaSTCarlaEnv, Dict[str, np.ndarray], Dict[str, object]]:
        last: Optional[BaseException] = None
        for attempt in range(4):
            env = self.get(cond.town)
            env.set_detector(self.detector, self.device)
            env.set_conditions(cond)
            try:
                seed = None if cond.route_seed is None else int(cond.route_seed) + 7919 * attempt
                obs, info = env.reset(seed=seed)
                return env, obs, info
            except TownSwitch:
                raise
            except BaseException as e:
                last = e
                print(f"[WARN] reset failed ({attempt + 1}/4) for {cond.key()}: {e}")
                self.close()
                time.sleep(3.0)
        raise RuntimeError(f"reset failed for {cond.key()}: {last}")

    def close(self) -> None:
        if self.env is not None:
            try:
                self.env.close()
            except BaseException:
                pass
        self.env, self.town = None, None


# ======================================================
# Labeler: delayed interventions with fixed ego actions (Sec. 3.4)
# ======================================================
class Labeler:
    def __init__(self, cfg: Config, nulls: Optional[NullBank], collect_nulls: bool):
        self.cfg, self.nulls, self.collect_nulls = cfg, nulls, collect_nulls
        self.pending: List[Dict[str, object]] = []
        self.actions: List[np.ndarray] = []
        self.seconds = 0.0

    def push_action(self, a: np.ndarray) -> None:
        self.actions.append(np.asarray(a, np.float32))

    def schedule(self, t: int, scene: PrivScene, ids: np.ndarray, kinds: np.ndarray, obs: Dict[str, np.ndarray],
                 extra: Optional[Dict[str, object]] = None) -> None:
        self.pending.append({"t": t, "scene": scene, "ids": ids, "kinds": kinds, "x0": obs["scalars"].copy(),
                             "edges": obs["edges"].copy(), "mask": obs["mask"].copy(), "extra": extra or {}})

    def ready(self, final: bool = False) -> List[Dict[str, object]]:
        H = int(self.cfg.intervention_horizon)
        out, keep = [], []
        for item in self.pending:
            t = int(item["t"])
            if not final and len(self.actions) < t + H:
                keep.append(item)
                continue
            t0 = time.perf_counter()
            acts = self.actions[t:t + H]
            if not acts:
                continue
            acts_arr = np.stack(acts + [acts[-1]] * (H - len(acts)), 0)
            scene: PrivScene = item["scene"]
            by_id = {e.eid: k for k, e in enumerate(scene.entities)}
            idxs = sorted({by_id[int(i)] for i in item["ids"] if int(i) in by_id})
            eff = entity_effects(scene, acts_arr, self.cfg, idxs)
            if self.collect_nulls and self.nulls is not None:
                for dist, f, m in null_probe_effects(scene, acts_arr, self.cfg, 2):
                    self.nulls.add(dist, f, m)
                for dist, f, m in null_probe_effects(scene, acts_arr, self.cfg, 1, far=True):
                    self.nulls.add(dist, f, m, far=True)
            slots = {}
            for slot, eid in enumerate(item["ids"]):
                eid = int(eid)
                if item["mask"][slot] < 0.5 or eid not in eff:
                    continue
                ent = scene.entities[by_id[eid]]
                dist = math.hypot(ent.x - scene.ego_x, ent.y - scene.ego_y)
                slots[slot] = (eff[eid][0], eff[eid][1], dist, ent.role)
            item["slots"] = slots
            self.seconds += time.perf_counter() - t0
            out.append(item)
        self.pending = keep
        return out


def oracle_causal_labels(env: CaSTCarlaEnv, obs: Dict[str, np.ndarray], action: np.ndarray, nulls: NullBank,
                         cfg: Config, null_mode: str = "matched") -> np.ndarray:
    """Ground-truth causal effect from the simulator for the oracle rows of Table 5 (constant current action)."""
    scene = env.privileged_scene()
    ids, _ = env.slot_info()
    acts = np.repeat(np.asarray(action, np.float32)[None], int(cfg.intervention_horizon), 0)
    by_id = {e.eid: k for k, e in enumerate(scene.entities)}
    eff = entity_effects(scene, acts, cfg, sorted({by_id[int(i)] for i in ids if int(i) in by_id}))
    out = np.zeros((cfg.max_entities,), np.float32)
    for slot, eid in enumerate(ids):
        if obs["mask"][slot] > 0.5 and int(eid) in eff:
            ent = scene.entities[by_id[int(eid)]]
            out[slot] = nulls.rank(eff[int(eid)][0], math.hypot(ent.x - scene.ego_x, ent.y - scene.ego_y),
                                   False, null_mode == "far")
    return out


def motion_target(cfg: Config, ids: np.ndarray, kinds: np.ndarray, ego0: Tuple[float, float, float],
                  pos0: Dict[int, Tuple[float, float]], ego1: Tuple[float, float, float],
                  pos1: Dict[int, Tuple[float, float]]) -> Tuple[np.ndarray, np.ndarray]:
    """Target of Eq. (5): displacement of the relative position of entity i over the motion horizon, ego frame at t."""
    tgt = np.zeros((cfg.max_entities, 2), np.float32)
    msk = np.zeros((cfg.max_entities,), np.float32)
    yaw = math.radians(ego0[2])
    fwd = np.array([math.cos(yaw), math.sin(yaw)])
    right = np.array([-math.sin(yaw), math.cos(yaw)])
    for slot, eid in enumerate(ids):
        eid = int(eid)
        if eid < 0 or int(kinds[slot]) == KIND_SIGNAL or eid not in pos0 or eid not in pos1:
            continue
        r0 = np.array(pos0[eid]) - np.array(ego0[:2])
        r1 = np.array(pos1[eid]) - np.array(ego1[:2])
        dlt = r1 - r0
        tgt[slot] = [float(np.dot(dlt, fwd)) / cfg.motion_scale_m, float(np.dot(dlt, right)) / cfg.motion_scale_m]
        msk[slot] = 1.0
    return tgt, msk


def phi_proxy(info: Dict[str, object], cfg: Config) -> float:
    ttc = safe_float(info.get("time_to_conflict", 999.0), 999.0)
    clear = max(0.0, safe_float(info.get("nearest_entity_dist", cfg.entity_max_dist)) - 2.0 * EGO_RADIUS_M)
    return 0.5 * min(ttc / cfg.ttc_norm_s, 1.0) + 0.5 * min(clear / cfg.clearance_norm_m, 1.0)


# ======================================================
# Coverage helpers
# ======================================================
@torch.no_grad()
def coverage_dk(agent: CaSTAgent, obs: Dict[str, np.ndarray]) -> Tuple[np.ndarray, np.ndarray]:
    cfg = agent.cfg
    if agent.cov is None:
        return np.zeros((cfg.max_entities, cfg.knn_k), np.float32), np.zeros((cfg.knn_k,), np.float32)
    ed = torch.from_numpy(obs["edges"]).to(agent.device)
    hs = agent.h_source(ed)
    dk = agent.cov.kdist_source(hs).cpu().numpy().astype(np.float32)
    sc = torch.from_numpy(obs["scene"][None]).to(agent.device)
    dk0 = agent.scene_cov.kdist_source(sc)[0].cpu().numpy().astype(np.float32)
    return dk, dk0


def teacher_label(env: CaSTCarlaEnv, cfg: Config) -> np.ndarray:
    """Clean action of the privileged teacher in the current state (the behaviour-cloning target)."""
    a = expert_drive_action(env, env.privileged_scene(), cfg)
    return clean_action(a)


class TeacherPerturbation:
    """Executed teacher action = clean label + temporally correlated perturbation (DART).
    About every 8 s a small steering (0.04-0.12) or throttle (0.1-0.25) disturbance lasting 0.3 to
    1.0 s pushes the car off the ideal path; the teacher then corrects, so the data contains
    recovery manoeuvres. The label stored for behaviour cloning is always the clean teacher action.
    A disturbance is never started, and a running one is cancelled, while the teacher brakes or an
    obstacle is closer than 25 m: the earlier version could add throttle on top of a braking label
    (the larger command wins), which drove the teacher into the car ahead."""

    def __init__(self, cfg: Config):
        self.cfg = cfg
        self.left = 0
        self.offset = np.zeros(3, np.float32)
        self.late_left = 0                     # remaining steps of a late-braking demonstration
        self.braking = False                   # the teacher label braked on the previous step
        self.cruise_throttle = 0.0             # last executed throttle before the teacher started braking

    def _late_brake_safe(self, gap: float, v: float, delay_s: float) -> bool:
        """After coasting for delay_s the teacher must still stop with 4 m/s^2 and 3 m to spare."""
        return gap - v * delay_s > v * v / (2.0 * 4.0) + 3.0

    def __call__(self, label: np.ndarray, hazard_gap: float = 1e9, ego_v: Optional[float] = None) -> np.ndarray:
        dt = float(self.cfg.dt)
        label = np.asarray(label, np.float32)
        brake_now = float(label[1]) > 0.15
        # Late-braking demonstration (DART for the longitudinal command). The teacher brakes early, so the
        # data almost never contains "stopped car or red line close ahead, still at speed, brake hard":
        # exactly the states in which the source policy drove on (it held +0.1..+0.4 throttle toward
        # visible stopped cars and crossed red lines in 40% of its own episodes). At a braking onset the
        # executed command keeps the cruise throttle for 0.4-1.4 s when the teacher can still stop safely;
        # then the teacher brakes. The stored label is always the clean teacher action.
        p_late = float(getattr(self.cfg, "late_brake_prob", 0.0))
        if ego_v is not None and p_late > 0.0 and hazard_gap < 1e8:
            v = max(float(ego_v), 0.0)
            if brake_now and not self.braking and self.late_left <= 0 and v > 2.0 and random.random() < p_late:
                delay = random.uniform(0.4, 1.4)
                if self._late_brake_safe(float(hazard_gap), v, delay):
                    self.late_left = int(round(delay / dt))
            self.braking = brake_now
            if self.late_left > 0:
                self.late_left -= 1
                if brake_now and self._late_brake_safe(float(hazard_gap), v, 0.0):
                    self.left = 0
                    return clean_action(np.array([min(self.cruise_throttle, 0.35), 0.0, float(label[2])], np.float32))
                self.late_left = 0
        else:
            self.braking = brake_now
        if not brake_now and float(label[1]) <= 0.02:
            self.cruise_throttle = float(label[0])
        if float(label[1]) > 0.02 or hazard_gap < 25.0:
            self.left = 0
            return clean_action(label + np.array([0.0, 0.0, np.random.normal(0.0, 0.01)], np.float32))
        if self.left <= 0 and random.random() < dt / 8.0:
            self.left = int(random.uniform(0.3, 1.0) / dt)
            if random.random() < 0.7:
                self.offset = np.array([0.0, 0.0, random.choice([-1.0, 1.0]) * random.uniform(0.04, 0.12)], np.float32)
            else:
                self.offset = np.array([random.uniform(0.1, 0.25), 0.0, 0.0], np.float32)
        a = label + np.random.normal(0.0, 1.0, 3).astype(np.float32) * np.array([0.02, 0.0, 0.01], np.float32)
        if self.left > 0:
            a = a + self.offset
            self.left -= 1
        return clean_action(a)


class InterventionGuard:
    """Teacher take-over rule of the hand-over phase (HG-DAgger, Kelly et al. 2019).

    The policy drives; the teacher takes control for 1.5 s when the policy (i) leaves the lane
    (|lateral offset| > 0.8 m or heading error > 15 deg), (ii) does not brake although the teacher
    brakes hard (teacher brake > 0.3 and policy longitudinal command 0.3 above the teacher's),
    (iii) heads for a standing object (cone, parked or stopped car) less than 2 s ahead on its own
    heading line without braking, or (iv) stands still for 2 s although the teacher would drive. Linear blending of teacher and
    policy actions was used before; it under-steered in sharp turns (the policy's share of the
    steering pulled the car wide) and hit kerbs and poles."""

    def __init__(self, cfg: Config):
        self.cfg = cfg
        self.left = 0
        self.still = 0

    def reset_still(self) -> None:
        self.left, self.still = 0, 0

    def check(self, scene: PrivScene, expert: np.ndarray, action: np.ndarray) -> str:
        if self.left > 0:
            self.left -= 1
            return "hold"
        why = ""
        try:
            _, d, head = route_frame(scene, np.array([scene.ego_x]), np.array([scene.ego_y]))
            herr = abs(wrap_pi(math.radians(scene.ego_yaw_deg) - float(head[0])))
            if abs(float(d[0])) > 0.8 or herr > math.radians(15.0):
                why = "lane"
        except BaseException:
            pass
        e_lon, p_lon = float(expert[0]) - float(expert[1]), float(action[0]) - float(action[1])
        if not why and float(expert[1]) > 0.3 and p_lon > e_lon + 0.3:
            why = "brake"
        if not why and scene.ego_v > 1.0 and p_lon > -0.2:
            # A standing object (cone, parked or stopped car) on the ego's own heading line less than 2 s
            # ahead, while the policy does not brake: the policy is about to hit it, wherever the route is.
            yaw = math.radians(scene.ego_yaw_deg)
            cy, sy = math.cos(yaw), math.sin(yaw)
            for e in scene.entities:
                if e.kind == KIND_SIGNAL or math.hypot(e.vx, e.vy) > 0.5:
                    continue
                dx, dy = e.x - scene.ego_x, e.y - scene.ego_y
                lon, lat = dx * cy + dy * sy, -dx * sy + dy * cy
                gap = lon - (e.radius if e.radius > 0.0 else KIND_RADIUS_M[e.kind]) - EGO_RADIUS_M
                if 0.0 < gap < 2.0 * scene.ego_v and abs(lat) < EGO_HALF_WIDTH_M + 0.3 + lateral_extent(e, yaw):
                    why = "obstacle"
                    break
        if scene.ego_v < 0.3 and float(expert[0]) > 0.15 and p_lon < 0.1:
            self.still += 1
        else:
            self.still = 0
        if not why and self.still >= int(round(2.0 / float(self.cfg.dt))):
            why, self.still = "start", 0
        if why:
            self.left = int(round(1.5 / float(self.cfg.dt))) - 1
        return why


def policy_to_env(a: np.ndarray) -> np.ndarray:
    """[longitudinal, steer] -> [throttle, brake, steer]."""
    lon, st = float(a[0]), float(a[1])
    return np.array([max(lon, 0.0), max(-lon, 0.0), st], np.float32)


def env_to_policy(a: np.ndarray) -> np.ndarray:
    """[throttle, brake, steer] -> [longitudinal, steer]."""
    return np.array([float(a[0]) - float(a[1]), float(a[2])], np.float32)


def clean_action(a: np.ndarray) -> np.ndarray:
    a = np.asarray(a, np.float32).copy()
    a[0:2] = np.clip(a[0:2], 0.0, 1.0)
    a[2] = float(np.clip(a[2], -1.0, 1.0))
    if a[0] > 0.0 and a[1] > 0.0:
        if a[0] >= a[1]:
            a[1] = 0.0
        else:
            a[0] = 0.0
    return a.astype(np.float32)


def bc_schedule(step: int, args: argparse.Namespace) -> float:
    """Behaviour-cloning weight: full during the teacher phase, then linear decay to a small floor
    (--bc-final) at the end of source training, so SAC drives the improvement while the teacher
    labels keep the policy from drifting to degenerate behaviour such as stopping."""
    w0, w1, t0, t1 = float(args.bc_weight), float(args.bc_final), int(args.teacher_steps), int(args.source_steps)
    if step < t0:
        return w0
    frac = min(1.0, (step - t0) / max(t1 - t0, 1))
    return float(w0 + (w1 - w0) * frac)


def demo_fraction(step: int, args: argparse.Namespace) -> float:
    """Share of teacher transitions in every source-stage batch after the teacher phase: --demo-frac
    (0.5) at the start of the hand-over, decaying linearly to --demo-frac-final (0.2) at the end of
    the source stage. During the teacher phase every transition is a teacher transition anyway."""
    t0, t1 = int(args.teacher_steps), int(args.source_steps)
    if step < t0:
        return 0.0
    frac = min(1.0, (step - t0) / max(t1 - t0, 1))
    return float(args.demo_frac + (args.demo_frac_final - args.demo_frac) * frac)


def teacher_blend(step: int, args: argparse.Namespace) -> float:
    """Teacher weight b of the executed action a = b a_teacher + (1 - b) a_policy (in the policy's
    [longitudinal, steer] space), as in the route-guidance curriculum of the CVPR 2026 code.
    b = 1 for the first --teacher-steps steps, then falls linearly from --blend-start to 0 over
    --handover-steps, so the policy takes control gradually while the car keeps driving safely."""
    t0, h = int(args.teacher_steps), int(args.handover_steps)
    if step < t0:
        return 1.0
    if h <= 0:
        return 0.0
    return float(max(0.0, float(args.blend_start) * (1.0 - (step - t0) / h)))


def sample_exploration_action(cfg: Config) -> np.ndarray:
    throttle = float(np.random.uniform(0.18, 0.62))
    brake = 0.0 if np.random.rand() < 0.90 else float(np.random.uniform(0.08, 0.32))
    if brake > 0.0:
        throttle = 0.0
    return np.array([throttle, brake, float(np.random.uniform(-0.35, 0.35))], dtype=np.float32)


@dataclass
class EpisodeSummary:
    steps: int = 0
    success: bool = False
    collision: bool = False
    rc: float = 0.0
    mean_U: float = 0.0
    mean_ubar: float = 0.0
    min_phi: float = 1.0
    km: float = 0.0
    reward: float = 0.0
    seconds_label: float = 0.0
    reason: str = ""
    jam_removed: int = 0
    hit: str = ""
    ubar_parts: str = ""                 # mean unclipped ubar components and fraction of saturated steps
    trace: str = ""                      # last 3 s before a failure: speed, lead token, policy vs teacher
    parts: str = ""
    interventions: int = 0
    lead_n: int = 0
    aborted: bool = False                # the simulator failed mid-episode (never counts as validated)
    lead_seen: int = 0
    det_true: int = 0
    det_found: int = 0
    near_true: int = 0
    near_found: int = 0
    int_reasons: Dict[str, int] = field(default_factory=dict)
    learning: Dict[str, float] = field(default_factory=dict)
    actor_updates: int = 0
    critic_updates: int = 0
    value_audit: Dict[str, object] = field(default_factory=dict)
    takeover_labels: int = 0
    target_expert_labels: int = 0


def save_value_audit(directory, episode, summ):
    if not summ.value_audit:
        return
    os.makedirs(directory, exist_ok=True)
    path = os.path.join(directory, f"{int(episode):06d}.json")
    # Episode-indexed replacement prevents duplicate records on checkpoint replay.
    with open(path + ".tmp", "w") as f:
        json.dump(dict(summ.value_audit, episode=int(episode)), f, indent=2, allow_nan=False)
    os.replace(path + ".tmp", path)
    a = summ.value_audit
    if a["mean_q_executed"] is not None:
        gap = a["mean_q_minus_soft_return"]
        gap_text = "censored" if gap is None else f"{gap:+.2f}"
        print(f"[VALUE AUDIT] executed_Q={a['mean_q_executed']:.2f} "
              f"observed_G={a['mean_observed_return']:.2f} "
              f"entropy_adjusted_G={a['mean_entropy_adjusted_return']:.2f} gap={gap_text} "
              f"teacher_fraction={a['teacher_fraction']:.2f} "
              f"synthetic_takeovers={summ.takeover_labels} target_labels={summ.target_expert_labels} "
              f"(behavior diagnostic, not frozen-policy calibration) | {path}")


# ======================================================
# Training episode: source stage and target stage
# ======================================================
def train_episode(mgr: EnvManager, agent: CaSTAgent, cond: Conditions, replay: CaSTReplay, args: argparse.Namespace,
                  cfg: Config, *, global_step: int, labeler: Optional[Labeler], dataset: Optional[CausalDataset],
                  domain: int, source_buffer: Optional[CaSTReplay], nulls: Optional[NullBank],
                  growth: Optional[List[Tuple[np.ndarray, np.ndarray]]]) -> Tuple[EpisodeSummary, int]:
    env, obs, info = mgr.reset(cond)
    sp = agent.spec
    perturb = TeacherPerturbation(cfg)
    guard = InterventionGuard(cfg)
    use_cov = sp.is_cast() and agent.cov is not None
    dk, dk0 = coverage_dk(agent, obs) if use_cov else (None, None)
    support = 1.0 if cond.in_source_support else 0.0
    pending_motion: List[Tuple[int, int, np.ndarray, np.ndarray, Tuple, Dict]] = []
    ego_hist: List[Tuple[float, float, float]] = []
    pos_hist: List[Dict[int, Tuple[float, float]]] = []
    summ = EpisodeSummary()
    trace_rows: List[Tuple] = []
    Us, ubs, ub_parts = [], [], []
    parts_sum: Dict[str, float] = {}
    learning_sum: Dict[str, float] = {}
    learning_n: Dict[str, int] = {}
    audit_rewards, audit_bonuses, audit_samples = [], [], []
    audit_start_step, teacher_steps = global_step, 0
    last_terminal, last_ridx = False, None
    c_or = None
    t = 0
    tau_c, tau_r = agent.tau_c(), agent.tau_rho()
    Hm = int(cfg.motion_horizon_steps)
    while True:
        ids, kinds = env.slot_info()
        ego_hist.append(env.ego_pose())
        pos_hist.append(env.true_positions())
        if sp.oracle_causal and nulls is not None:
            # Optimizer updates can change the action since next-state labels were stored.
            # The online oracle must describe the current policy, not its previous parameters.
            c_or = oracle_causal_labels(env, obs, agent.act(obs, True), nulls, cfg, sp.null_mode)
        q = agent.step_quantities(obs, dk, dk0, support, c_or)
        expert = None
        teacher = False
        rejected_action = None
        if domain == 0:
            agent.bc_weight = bc_schedule(global_step, args)
            # RL ramps in over the hand-over but is capped (source_rl_max): with the full RL weight, brake
            # take-overs rose from about 5% to 35% of steps while the imitation batches showed no braking
            # misses (rgbd9: mean target; rgbd7: 9% to 20% with the min-of-two target), so the RL term was
            # overriding the braking labels on the road. The critic still trains on every transition.
            agent.actor_rl_weight = float(cfg.source_rl_max) * float(np.clip(
                (global_step - int(args.teacher_steps)) / max(int(args.handover_steps), 1), 0.0, 1.0))
            agent.normalize_q = True
            # The temperature is tuned only once the policy drives; during pure imitation the entropy
            # of the policy says nothing about exploration, and tuning there drove alpha_0 to its cap.
            agent.tune_alpha = global_step >= int(args.teacher_steps)
            expert = teacher_label(env, cfg)          # DAgger label for every source state
            _sc = env.privileged_scene()
            hazard_gap = float(expert_lead(_sc, _sc.entities, v_cruise=cfg.expert_target_speed_ms)[0])
            _lg, _, _lid = expert_lead(_sc, _sc.entities, include_signals=False, predict=False)
            if _lid >= 0 and _lg < 20.0 and any(e.eid == _lid and e.kind == KIND_VEHICLE for e in _sc.entities):
                # Is the car the teacher reacts to (within 20 m) among the detector's tokens?
                summ.lead_n += 1
                summ.lead_seen += int(_lid in set(int(i) for i in ids))
            blend = teacher_blend(global_step, args)
            teacher = blend >= 1.0
            ego_slow = 1.0 if _sc.ego_v < 1.0 else 0.0
        else:
            agent.tune_alpha = True
            # Capped like the source stage. With weight 1 the critic, optimistic on target states (value
            # audit gap +10 to +45), pulled the actor to brake at standstill with a free road in heavy rain
            # (rgbd14 adaptation: 6 of the last 15 episodes stuck at spawn, teacher +0.6, policy -0.99).
            agent.actor_rl_weight = float(cfg.target_rl_max)
            agent.normalize_q = True         # same RL-term scale as in the source stage
            blend, ego_slow, hazard_gap = 0.0, 0.0, 1e9
            if target_expert_eligible(domain, q["ubar"], cfg):
                expert = teacher_label(env, cfg)
                ego_slow = float(env.privileged_scene().ego_v < 1.0)
        if teacher:
            action = perturb(expert, hazard_gap, float(_sc.ego_v) if domain == 0 else None)
        elif domain == 0 and global_step < int(args.start_steps):
            action = sample_exploration_action(cfg)
        else:
            # Source stage: the policy's mean action (the hand-over corrects it, behaviour cloning trains
            # it); sampled exploration noise there produced random braking and needless take-overs.
            action = agent.act(obs, deterministic=(domain == 0), std_cap=float(cfg.target_explore_std))
            if blend > 0.0:
                # Hand-over phase (HG-DAgger): the policy drives and the teacher takes over for 1.5 s
                # whenever the policy is about to fail; every state keeps its teacher label.
                why = guard.check(_sc, expert, action)
                if why:
                    if takeover_label_enabled(why, cfg.takeover_loss_weight):
                        rejected_action = env_to_policy(np.asarray(action, np.float32)).copy()
                    action = clean_action(expert)
                    teacher = True
                    summ.interventions += 1
                    summ.int_reasons[why] = summ.int_reasons.get(why, 0) + 1
            elif domain == 0:
                guard.reset_still()
        if labeler is not None and random.random() < float(sp.int_frac):
            rel = {}
            for slot in range(cfg.max_entities):
                if obs["mask"][slot] > 0.5:
                    occ = env.tokens_with_occlusion(int(ids[slot])) if (int(ids[slot]) >= 0 and int(kinds[slot]) != KIND_SIGNAL) else None
                    rel[slot] = agent.reliance(obs, slot, int(kinds[slot]), occ)
            labeler.schedule(t, env.privileged_scene(), ids, kinds, obs,
                             {"reliance": rel, "attn": q["w"] * max(float(obs["mask"].sum()), 1.0)})
        acted_on = obs                       # the observation behind the last action (for the failure log)
        try:
            _lt = expert if expert is not None else teacher_label(env, cfg)
            _ld, _lc, _lk, _li, _ls2 = lead_token(obs, cfg)
            _pa = env_to_policy(np.asarray(action, np.float32))
            _att = float(q["w"][_li]) * max(float(obs["mask"].sum()), 1.0) if _li >= 0 else float("nan")
            trace_rows.append((t, float(env.privileged_scene().ego_v), _ld, _lc, _lk, float(_pa[0]),
                               float(_lt[0]) - float(_lt[1]), _ls2, _att, float(_pa[1]), float(_lt[2])))
            if len(trace_rows) > int(round(3.0 / cfg.dt)) + 1:
                del trace_rows[0]
        except BaseException:
            pass
        try:
            next_obs, _, term, trunc, info = env.step(action)
        except Exception as e:
            print(f"[WARN] step failed: {e}; ending episode")
            summ.aborted = True
            summ.reason = "aborted"
            if last_ridx is not None:
                replay.data["episode_end"][last_ridx] = 1.0
                replay.data["truncated"][last_ridx] = 1.0
            down = server_unresponsive(env, e)
            mgr.close()
            if down:
                raise SimulatorDown(str(e))
            break
        command = np.asarray(info.get("command_action", action), np.float32)
        reward = build_reward_from_info(info, q["ubar"], cfg=cfg)
        for key, val in (("rp", cfg.w_p * safe_float(info.get("rp", 0.0))), ("rs", cfg.w_s * (safe_float(info.get("rs", 1.0)) - 1.0)),
                         ("rc", cfg.w_c * safe_float(info.get("rc", 0.0))), ("ru", -cfg.w_u * float(q["ubar"])),
                         ("lane", -safe_float(info.get("phi_lane", 0.0))),
                         ("red", cfg.r_red_light * len(info.get("red_light_events", []))),
                         ("end", {"collision": cfg.r_collision, "off_route": cfg.r_offroute, "goal": cfg.r_goal, "stuck": cfg.r_stuck}.get(
                             str(info.get("term_reason", "")), 0.0))):
            parts_sum[key] = parts_sum.get(key, 0.0) + val
        stored_action = env_to_policy(command)
        teacher_steps += int(teacher)
        last_terminal = bool(term)
        summ.takeover_labels += int(rejected_action is not None)
        summ.target_expert_labels += int(domain == 1 and expert is not None)
        if cfg.value_audit_stride > 0:
            audit_rewards.append(float(reward))
            audit_bonuses.append(float(q["audit_alpha"] * agent.entropy_scale() * q["policy_entropy"]))
            if t % cfg.value_audit_stride == 0 or term or trunc:
                # No parameter update has occurred since the action was selected.
                audit_samples.append(dict(agent.audit_action_values(obs, stored_action), t=t,
                                          teacher=bool(teacher), uncertainty=float(q["ubar"])))
        ndk, ndk0 = coverage_dk(agent, next_obs) if use_cov else (None, None)
        row = {"scalars": obs["scalars"], "edges": obs["edges"], "mask": obs["mask"], "scene": obs["scene"],
               "actions": stored_action, "rewards": reward, "terminals": float(bool(term)),
               "next_scalars": next_obs["scalars"], "next_edges": next_obs["edges"], "next_mask": next_obs["mask"],
               "next_scene": next_obs["scene"], "support": support, "next_support": support,
               "teacher": 1.0 if teacher else 0.0, "slow": ego_slow,
               # Labels are kept inside +-0.95: a label of exactly -1 (full brake) drives the tanh-squashed
               # mean to saturation, the log-density then explodes, and the entropy reading becomes meaningless.
               "expert_action": np.clip(env_to_policy(expert), -0.95, 0.95) if expert is not None
               else np.zeros(cfg.policy_action_dim, np.float32),
               "has_expert": 1.0 if expert is not None else 0.0}
        row.update({"rejected_action": rejected_action if rejected_action is not None
                    else np.zeros(cfg.policy_action_dim, np.float32),
                    "has_rejected": float(rejected_action is not None), "domain": float(domain),
                    "episode_end": float(bool(term or trunc)), "truncated": float(bool(trunc)),
                    "label_uncertainty": float(q["ubar"])})
        if use_cov:
            row.update({"dk": dk, "next_dk": ndk, "dk0": dk0, "next_dk0": ndk0})
            if c_or is not None:
                # next-state labels from the post-step scene and token slots, before the row can be sampled
                # Failure is not evidence of zero causal effect. Propagate it rather than
                # inserting fabricated labels, including for bootstrapped truncations.
                c_or_next = oracle_causal_labels(env, next_obs, agent.act(next_obs, True), nulls, cfg,
                                                 sp.null_mode)
                row["c_or"] = c_or
                row["next_c_or"] = c_or_next
        ridx = replay.add(row)
        last_ridx = ridx
        pending_motion.append((t, ridx, ids, kinds))
        if len(pending_motion) >= Hm:
            # displacement over exactly Hm steps: from the state of step t0 to the state after step t0+Hm-1
            t0, ri0, ids0, k0 = pending_motion.pop(0)
            tgt, msk = motion_target(cfg, ids0, k0, ego_hist[t0], pos_hist[t0], env.ego_pose(), env.true_positions())
            replay.set_motion(ri0, tgt, msk)
        if labeler is not None:
            labeler.push_action(command)
            for item in labeler.ready():
                store_labels(item, dataset, domain)
        # per-step quantities of Sec. 3.5
        U = float(q["ell0_t"] + (q["c_hat"] * q["ell_t"]).sum())
        Us.append(U)
        ubs.append(q["ubar"])
        ub_parts.append((q.get("ub_motion", 0.0), q.get("ub_critic", 0.0), q.get("ub_scene", 0.0),
                         float(q["ubar"] >= 0.999)))
        summ.min_phi = min(summ.min_phi, phi_proxy(info, cfg))
        if use_cov and growth is not None and "rho" in q:
            sel = (q["c_hat"] >= tau_c) & (q["rho"] < tau_r) & (obs["mask"] > 0.5)
            hs = agent.h_source(torch.from_numpy(obs["edges"][sel]).to(agent.device)).cpu().numpy() if sel.any() else \
                np.zeros((0, cfg.z_dim), np.float32)
            sc = obs["scene"][None] if q.get("rho0", 1.0) < tau_r else np.zeros((0, cfg.scene_dim), np.float32)
            growth.append((hs, sc))
        summ.reward += reward
        # updates
        if len(replay) >= max(cfg.batch_size, int(args.update_after)) and (domain == 1 or global_step >= int(args.update_after)):
            for _ in range(int(args.updates_per_step)):
                sb = source_buffer.sample(cfg.batch_size, agent.device) if source_buffer is not None and len(source_buffer) > cfg.batch_size else None
                if domain == 0:
                    batch = replay.sample_mixed(cfg.batch_size, agent.device, demo_fraction(global_step, args))
                    batch = augment_token_uncertainty(batch, float(cfg.sigma_aug_prob))
                else:
                    batch = replay.sample(cfg.batch_size, agent.device)
                act_step = True
                if domain == 1:
                    # Target stage: the actor followed the gradient of a critic fitted on a few thousand
                    # target transitions from the first update on, at the source learning rate, and plain
                    # fine-tuning drove off the road within one or two episodes (steering it never had
                    # to learn under RL). Critic warm-up and a policy delay keep it on the road.
                    agent.target_updates += 1
                    act_step = (agent.target_updates > int(cfg.target_critic_warmup)
                                and agent.target_updates % max(int(cfg.target_policy_delay), 1) == 0)
                # Separate source and target imitation batches. Only explicitly
                # uncertainty-gated target rows have labels in target replay.
                bc_replay = replay if domain == 0 else source_buffer
                bc_batch = bc_replay.sample_expert(cfg.batch_size, agent.device) if act_step and bc_replay is not None else None
                target_bc = (replay.sample_expert(cfg.batch_size, agent.device)
                             if domain == 1 and act_step and cfg.target_state_bc_weight > 0.0 else None)
                update_stats = agent.update(batch, sb, actor_step=act_step, bc_batch=bc_batch,
                                            target_bc_batch=target_bc)
                summ.critic_updates += 1
                summ.actor_updates += int(act_step)
                for key in ("critic_q", "td_abs", "td_target", "batch_reward", "critic_spread",
                            "target_selection_offset", "target_before_cap", "takeover_loss", "takeover_batch_n"):
                    value = float(update_stats.get(key, float("nan")))
                    if math.isfinite(value):
                        learning_sum[key] = learning_sum.get(key, 0.0) + value
                        learning_n[key] = learning_n.get(key, 0) + 1
                if act_step:
                    for key in ("L_BC", "L_source_BC", "brake_miss", "start_miss", "source_brake_miss",
                                "source_start_miss", "policy_std", "entropy", "L_target_BC",
                                "target_brake_miss", "target_start_miss", "L_A", "L_I", "q"):
                        value = float(update_stats.get(key, float("nan")))
                        if math.isfinite(value):
                            learning_sum[key] = learning_sum.get(key, 0.0) + value
                            learning_n[key] = learning_n.get(key, 0) + 1
        if domain == 0 and global_step > 0 and global_step % 10000 == 0:
            refit_v_calibrator(agent, replay)
        global_step += 1
        t += 1
        obs, dk, dk0 = next_obs, ndk, ndk0
        if term or trunc:
            break
    if labeler is not None:
        for item in labeler.ready(final=True):
            store_labels(item, dataset, domain)
        labeler.actions = []
        summ.seconds_label = labeler.seconds
    summ.steps = t
    summ.success = bool(info.get("success", False))
    summ.collision = bool(info.get("collision", False))
    summ.rc = safe_float(info.get("route_completion_pct", 0.0))
    summ.reason = "aborted" if summ.aborted else str(info.get("term_reason", ""))
    try:
        summ.det_true, summ.det_found = env.detection_recall_counts()
        summ.near_true, summ.near_found = env.near_recall_counts()
    except BaseException:
        pass
    summ.hit = str(info.get("collision_kind", "")).replace("collision_", "")
    if summ.reason in ("stuck", "blocked") and hasattr(env, "describe_wait"):
        summ.hit = env.describe_wait()
    if summ.hit and info.get("collision_where"):
        summ.hit += "," + str(info["collision_where"])
    if summ.reason in ("collision", "stuck", "off_route"):
        summ.trace = format_trace(trace_rows, cfg.dt)
    if summ.reason in ("collision", "stuck", "blocked", "off_route"):
        # Separates "the policy did not see it" (perception) from "it saw it and still drove" (control).
        try:
            summ.hit = (summ.hit + "," if summ.hit else "") + describe_tokens(acted_on, cfg)
        except BaseException:
            pass
    summ.jam_removed = int(info.get("jam_removed", 0))
    summ.km = safe_float(info.get("distance_driven_m", 0.0)) / 1000.0
    summ.mean_U = float(np.mean(Us)) if Us else 0.0
    summ.mean_ubar = float(np.mean(ubs)) if ubs else 0.0
    if ub_parts:
        m = np.mean(np.asarray(ub_parts, np.float64), 0)
        summ.ubar_parts = f"motion={m[0]:.2f} critic={m[1]:.2f} scene={m[2]:.2f} sat={100.0 * m[3]:.0f}%"
    summ.parts = " ".join((f"{k}={v / max(t, 1):+.3f}" if k != "end" else f"end={v:+.1f}") for k, v in parts_sum.items())
    summ.learning = {key: value / learning_n[key] for key, value in learning_sum.items()}
    if cfg.value_audit_stride > 0:
        summ.value_audit = build_value_audit(audit_rewards, audit_bonuses, audit_samples, cfg.gamma,
            last_terminal and not summ.aborted,
            {"training_revision": TRAINING_REVISION, "value_training_options": value_training_options(cfg),
             "start_step": audit_start_step, "end_step": global_step, "domain": domain,
             "town": cond.town, "weather": cond.weather, "end_reason": summ.reason,
             "teacher_fraction": teacher_steps / max(t, 1), "actor_updates": summ.actor_updates,
             "jam_removed": summ.jam_removed, "synthetic_takeovers": summ.takeover_labels,
             "target_expert_labels": summ.target_expert_labels, "learning_mean": summ.learning,
             "action_mode": "deterministic_source_or_teacher" if domain == 0 else "stochastic_target",
             "checkpoint_note": "May be overwritten when resuming the last complete checkpoint."})
    return summ, global_step


def light_token_status(e: np.ndarray, cfg: Config) -> str:
    """Colour the policy sees in one signal token: red, yellow, green or unknown (map_camera v2 flag)."""
    sd = int(cfg.sem_dim)
    if float(e[4 + sd + 1]) > 0.5:
        return "unknown"
    code = float(e[4 + sd])
    return "red" if code > 0.75 else "yellow" if code > 0.25 else "green"


def perceived_light(obs: Dict[str, np.ndarray], cfg: Config) -> Tuple[str, float]:
    """(colour, distance m) of the traffic-light token in an observation; ("none", -1) without one."""
    ed, ms = obs["edges"], obs["mask"]
    for i in range(len(ms)):
        if ms[i] >= 0.5 and int(np.argmax(ed[i, 4:4 + cfg.sem_dim])) == KIND_SIGNAL:
            return light_token_status(ed[i], cfg), float(ed[i, 4 + cfg.sem_dim + 2]) * float(cfg.entity_max_dist)
    return "none", -1.0


def describe_tokens(obs: Dict[str, np.ndarray], cfg: Config) -> str:
    """What the policy saw at the last step: the nearest object token and the traffic-light token."""
    names = {KIND_VEHICLE: "car", KIND_WALKER: "walker", KIND_STATIC: "static"}
    ed, ms = obs["edges"], obs["mask"]
    near, light = None, "none"
    for i in range(len(ms)):
        if ms[i] < 0.5:
            continue
        kind = int(np.argmax(ed[i, 4:4 + cfg.sem_dim]))
        d = float(math.hypot(ed[i, 0], ed[i, 1]) * cfg.entity_max_dist)
        if kind == KIND_SIGNAL:
            light = f"{light_token_status(ed[i], cfg)}@{d:.0f}m"
        elif near is None or d < near[1]:
            near = (names.get(kind, "obj"), d)
    tok = f"{near[0]}@{near[1]:.1f}m" if near else "none"
    return f"policy_saw:n={int(ms.sum())},nearest={tok},light={light}"


def teacher_idm_lon(v: torch.Tensor, gap: torch.Tensor, v_lead: torch.Tensor, cfg: Config) -> torch.Tensor:
    """Longitudinal command of expert_action (IDM, the same constants) for one obstacle on the route."""
    v0 = max(float(cfg.expert_target_speed_ms), 0.1)
    a_max, b_comf, t_head, s0 = 1.5, 3.0, 1.2, 3.0
    free = a_max * (1.0 - (v / v0).pow(4))
    s_star = (s0 + v * t_head + v * (v - v_lead) / (2.0 * math.sqrt(a_max * b_comf))).clamp(min=0.0)
    acc = free - a_max * (s_star / gap.clamp(min=0.1)).pow(2)
    lon = torch.where(acc > 0.0, (acc / 2.5).clamp(0.0, 1.0), -(-acc / 6.0).clamp(0.0, 1.0))
    hold = (v < 0.3) & (v_lead < 0.5) & (gap < s0 + 3.0)
    return torch.where(hold, torch.full_like(lon, -0.4), lon)


def augment_hazard(b: Dict[str, torch.Tensor], cfg: Config) -> Dict[str, torch.Tensor]:
    """Counterfactual hazard insertion for imitation batches. In a share cfg.hazard_aug_prob of the labelled
    states with nothing on the route within 35 m, a confidently detected car (stopped in 60% of cases,
    else slower than the ego) or a standing pedestrian is inserted 6-30 m ahead in the lane, and the label
    becomes the more cautious of the teacher's label and the teacher's IDM command for that obstacle.
    Both policies drove into a clearly detected stopped car (sigma^2 0.07-0.16, closing at 4.4 m/s, throttle
    +0.2 throughout): only 0.6% of the labelled adaptation states had the teacher braking for a car within
    20 m, mostly near red lights, so braking was learned from the light and the scene, not from the car
    (with the car removed from the input the adapted policy still braked in 88% of those states, and with a
    car inserted on a free road it braked in 4%)."""
    p = float(getattr(cfg, "hazard_aug_prob", 0.0))
    if p <= 0.0 or "expert_action" not in b:
        return b
    ed, ms = b["edges"], b["mask"]
    B, N, E = ed.shape
    sd, emd = int(cfg.sem_dim), float(cfg.entity_max_dist)
    dev = ed.device
    kind = ed[..., 4:4 + sd].argmax(-1)
    lat_m = ed[..., 4 + sd + 1] * 3.0 * 3.5
    arc_m = ed[..., 4 + sd + 2] * emd
    on_route = (ms > 0.5) & (kind != KIND_SIGNAL) & (lat_m.abs() <= 1.75) & (arc_m > 0.0) & (arc_m < 35.0)
    v2 = int(getattr(cfg, "hazard_aug_version", 2)) >= 2
    if v2:
        # Version 2: also in traffic (in front of the nearest obstacle on the route, at least 3 m before it),
        # and half of the insertions at close range. The rgbd14d policy braked for an isolated car on a free
        # road but still rolled into stopped cars in queues, after starting at a green light, and when the
        # detection flickered in rain, all at centre distances under 8 m.
        nearest = torch.where(on_route, arc_m, torch.full_like(arc_m, 1e3)).min(1).values
        upper = (nearest - 3.0).clamp(max=30.0)
        pick = (upper >= 5.0) & (b["has_expert"] > 0.5) & (torch.rand(B, device=dev) < p)
    else:
        pick = (~on_route.any(1)) & (b["has_expert"] > 0.5) & (torch.rand(B, device=dev) < p)
    if not bool(pick.any()):
        return b
    idx = torch.nonzero(pick).squeeze(1)
    k = idx.numel()
    v = (b["scalars"][idx, 0] * 10.0).clamp(min=0.0)
    walker = torch.rand(k, device=dev) < 0.3
    if v2:
        hi = upper[idx]
        close = torch.rand(k, device=dev) < 0.5
        hi = torch.where(close, hi.clamp(max=12.0), hi)
        dist = 4.8 + torch.rand(k, device=dev) * (hi - 4.8)
    else:
        dist = torch.empty(k, device=dev).uniform_(6.0, 30.0)
    v_lead = torch.where(walker | (torch.rand(k, device=dev) < 0.6), torch.zeros(k, device=dev),
                         torch.rand(k, device=dev) * v)
    radius = torch.where(walker, torch.full_like(dist, KIND_RADIUS_M[KIND_WALKER]),
                         torch.full_like(dist, KIND_RADIUS_M[KIND_VEHICLE]))
    gap = dist - radius - EGO_RADIUS_M
    lon = teacher_idm_lon(v, gap, v_lead, cfg)
    y_m = torch.empty(k, device=dev).uniform_(-0.6, 0.6)
    tok = torch.zeros(k, E, device=dev)
    tok[:, 0] = dist / emd
    tok[:, 1] = y_m / emd
    tok[:, 2] = (v_lead - v) / 15.0                   # relative velocity along the ego axis
    tok[torch.arange(k, device=dev), 4 + torch.where(walker, torch.full_like(idx, KIND_WALKER),
                                                    torch.full_like(idx, KIND_VEHICLE))] = 1.0
    tok[:, 4 + sd + 1] = (y_m / 3.5).clamp(-3.0, 3.0) / 3.0
    tok[:, 4 + sd + 2] = (dist / emd).clamp(-1.0, 1.0)
    tok[:, -1] = torch.empty(k, device=dev).uniform_(0.05, 0.4)
    ms_k = ms[idx]
    empty = ms_k < 0.5
    far = (ed[idx, :, 0:2].pow(2).sum(-1) * (~empty).float()).argmax(1)
    slot = torch.where(empty.any(1), empty.float().argmax(1), far)
    out = dict(b)
    e2, m2, a2 = ed.clone(), ms.clone(), b["expert_action"].clone()
    e2[idx, slot] = tok
    m2[idx, slot] = 1.0
    a2[idx, 0] = torch.minimum(a2[idx, 0], lon)
    out["edges"], out["mask"], out["expert_action"] = e2, m2, a2
    return out


def augment_token_uncertainty(b: Dict[str, torch.Tensor], p: float) -> Dict[str, torch.Tensor]:
    """Source-stage augmentation: for a share p of the samples, every token's detection variance
    (last edge feature) is multiplied by U(1, 5), clipped at 0.98; the teacher label is unchanged.
    Rain, fog, glare and night lower the detector's confidence and make tracks coast, so the variance
    of the very car or pedestrian ahead rises. A policy trained only on confident source detections
    then gave that token little attention and drove into it (pre-crash traces: lead visible for 3 s,
    teacher braking fully, policy holding throttle). The decision to brake must not depend on it."""
    if p <= 0.0:
        return b
    out = dict(b)
    for key in ("edges", "next_edges"):
        if key not in b:
            continue
        e = b[key].clone()
        B, N = e.shape[0], e.shape[1]
        pick = torch.rand(B, 1, device=e.device) < p
        scale = torch.empty(B, N, device=e.device).uniform_(1.0, 5.0)
        e[..., -1] = torch.where(pick, (e[..., -1] * scale).clamp(max=0.98), e[..., -1])
        out[key] = e
    return out


def lead_token(obs: Dict[str, np.ndarray], cfg: Config) -> Tuple[float, float, str, int, float]:
    """Nearest object token ahead on the route (route-relative offset within 1.75 m): arc distance (m),
    closing speed along the ego axis (m/s, positive = approaching), kind, slot, detection variance.
    (inf, 0, '', -1, 0) if none."""
    ed, ms = obs["edges"], obs["mask"]
    best = (float("inf"), 0.0, "", -1, 0.0)
    sd = cfg.sem_dim
    for i in range(len(ms)):
        if ms[i] < 0.5:
            continue
        kind = int(np.argmax(ed[i, 4:4 + sd]))
        if kind == KIND_SIGNAL:
            continue
        lat_m = float(ed[i, 4 + sd + 1]) * 3.0 * 3.5
        arc_m = float(ed[i, 4 + sd + 2]) * float(cfg.entity_max_dist)
        if abs(lat_m) <= 1.75 and 0.0 < arc_m < best[0]:
            best = (arc_m, -float(ed[i, 2]) * 15.0, {KIND_VEHICLE: "car", KIND_WALKER: "walker"}.get(kind, "obj"),
                    i, float(ed[i, -1]))
    return best


def format_trace(rows: List[Tuple], dt: float) -> str:
    """rows: (t, v, lead_arc, closing, kind, policy_lon, teacher_lon, lead_sigma2, lead_attention_share,
    policy_steer, teacher_steer) of the last 3 s, shown every 0.5 s. The attention share is the lead's
    weight times the number of tokens (1 = average attention)."""
    if not rows:
        return ""
    step = max(int(round(0.5 / dt)), 1)
    t_end = rows[-1][0]
    parts = []
    for r in rows[::-1][::step][::-1]:
        t, v, d, cv, kind, a, te = r[:7]
        s2, att, st, tst = (list(r[7:11]) + [float("nan")] * 4)[:4]
        lead = f"{kind}@{d:.1f}m/{cv:+.1f}(s2={s2:.2f},att={att:.1f})" if np.isfinite(d) else "none"
        parts.append(f"t{(t - t_end) * dt:+.1f}s v={v:.1f} lead={lead} act={a:+.2f}/{st:+.2f} teacher={te:+.2f}/{tst:+.2f}")
    return " | ".join(parts)


def store_labels(item: Dict[str, object], dataset: Optional[CausalDataset], domain: int) -> None:
    if dataset is None:
        return
    extra = item.get("extra", {})
    rel = extra.get("reliance", {})
    attn = extra.get("attn", np.zeros(len(item["mask"])))
    for slot, (e_full, e_marg, dist, _role) in item["slots"].items():
        dataset.add(item["x0"], item["edges"][slot], e_full, e_marg, dist, float(rel.get(slot, 0.0)),
                    float(np.clip(attn[slot], 0.0, 1.0)), domain)


@torch.no_grad()
def refit_v_calibrator(agent: CaSTAgent, replay: CaSTReplay, n: int = 20000) -> None:
    """Fit the monotone map g of Eq. (6) and the aleatoric scale on held-out source states."""
    if len(replay) < 100:
        return
    vals, sig = [], []
    bs = min(1000, len(replay))
    for _ in range(max(1, min(n, len(replay)) // bs)):
        b = replay.sample(bs, agent.device)
        mu, _, _, h = agent.actor.dist_params(b["scalars"], b["edges"], b["mask"])
        vals.append(agent.epistemic_v(b["scalars"], b["edges"], b["mask"], mu).cpu().numpy())
        _, logvar = agent.actor.motion_head(h, b["edges"])
        sig.append(logvar.exp()[b["mask"] > 0.5].cpu().numpy())
    agent.v_cal.fit_from_values(np.concatenate(vals))
    fit_sig_ref(agent, np.concatenate(sig))


def fit_sig_ref(agent: CaSTAgent, sig2: np.ndarray) -> None:
    """The motion head predicts a variance in m^2; 1 - exp(-sigma^2) saturated (median source token about
    0.6, decision uncertainty u_bar about 0.7 in the source domain and pinned at 1.0 in the target), so
    nothing that depends on u_bar (temperature, cautious value, reward term, coverage-weighted terms)
    could respond to the domain. The variance is divided by sig_ref, chosen so that the median source
    token gives 0.1: higher-than-usual motion uncertainty raises u_bar, ordinary source states do not."""
    v = np.asarray(sig2, np.float64).reshape(-1)
    v = v[np.isfinite(v) & (v > 0)]
    if v.size < 10:
        return
    agent.sig_ref = max(float(np.median(v)) / -math.log(0.9), 1e-4)
    agent.sig_ref_fitted = True


# ======================================================
# Perception stage: pre-train the camera-based BEV detector on source data, then freeze it
# ======================================================
def detector_train_step(det: BEVDetector, opt: torch.optim.Optimizer, prev: np.ndarray, cur: np.ndarray,
                        labels: List[np.ndarray], tl: np.ndarray, device: torch.device) -> Dict[str, float]:
    det.train()
    with torch.no_grad():
        prev_l = det(det.preprocess(torch.from_numpy(prev).to(device)), None)["lifted"]
    out = det(det.preprocess(torch.from_numpy(cur).to(device)), prev_l)
    loss, st = detector_loss(out, detector_targets(det, labels, tl, device))
    opt.zero_grad(set_to_none=True)
    loss.backward()
    torch.nn.utils.clip_grad_norm_(det.parameters(), 10.0)
    opt.step()
    st["loss"] = float(loss.detach())
    return st


@torch.no_grad()
def best_threshold(det: "BEVDetector", frames, cfg: Config, device: torch.device) -> float:
    """Heat-map threshold with the best vehicle F1 on the validation frames."""
    det.eval()
    outs = []
    for prev, cur, lab, _ in frames:
        prev_l = det(det.preprocess(torch.from_numpy(prev[None]).to(device)), None)["lifted"]
        outs.append((det(det.preprocess(torch.from_numpy(cur[None]).to(device)), prev_l), lab))
    best = (-1.0, float(cfg.det_threshold))
    for thr in (0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.5):
        tp = fp = fn = 0
        for out, lab in outs:
            dets = [d for d in det.decode(out, thr=thr, k=32) if d["kind"] == KIND_VEHICLE]
            truth = [(j, int(r[0]), float(r[1]), float(r[2])) for j, r in enumerate(lab) if int(r[0]) == KIND_VEHICLE]
            ids = match_detections(dets, truth)
            m = sum(1 for i in ids if i >= 0)
            tp, fp, fn = tp + m, fp + len(dets) - m, fn + len(truth) - m
        f1 = 2.0 * tp / max(2 * tp + fp + fn, 1)
        if f1 > best[0]:
            best = (f1, thr)
    return float(best[1])


@torch.no_grad()
def evaluate_detector(det: BEVDetector, frames: List[Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]],
                      cfg: Config, device: torch.device) -> List[List[object]]:
    det.eval()
    stats = {k: {"tp": 0, "fp": 0, "fn": 0, "err": []} for k in DET_CLASSES}
    tl_ok, tl_n = 0, 0
    for prev, cur, lab, tl in frames:
        prev_l = det(det.preprocess(torch.from_numpy(prev[None]).to(device)), None)["lifted"]
        out = det(det.preprocess(torch.from_numpy(cur[None]).to(device)), prev_l)
        dets = det.decode(out, thr=float(cfg.det_threshold), k=32)
        truth = [(j, int(r[0]), float(r[1]), float(r[2])) for j, r in enumerate(lab)]
        ids = match_detections(dets, truth)
        for d, i in zip(dets, ids):
            st = stats[d["kind"]]
            if i >= 0:
                st["tp"] += 1
                st["err"].append(math.hypot(d["x"] - lab[i][1], d["y"] - lab[i][2]))
            else:
                st["fp"] += 1
        matched = {i for i in ids if i >= 0}
        for j, r in enumerate(lab):
            if int(r[0]) in stats and j not in matched:
                stats[int(r[0])]["fn"] += 1
        tl_n += 1
        tl_ok += int(int(torch.argmax(out["tl_logits"][0])) == int(tl[0]))
    rows = []
    for k, st in stats.items():
        rec = st["tp"] / max(st["tp"] + st["fn"], 1)
        pre = st["tp"] / max(st["tp"] + st["fp"], 1)
        rows.append([KIND_NAMES[k], st["tp"] + st["fn"], 100.0 * rec, 100.0 * pre,
                     float(np.mean(st["err"])) if st["err"] else float("nan")])
    rows.append(["traffic_light_state", tl_n, 100.0 * tl_ok / max(tl_n, 1), float("nan"), float("nan")])
    # vehicle recall by distance (what matters for braking)
    for lo, hi in ((0.0, 15.0), (15.0, 25.0), (25.0, 40.0)):
        tp = tot = 0
        for prev, cur, lab, tl in frames:
            prev_l = det(det.preprocess(torch.from_numpy(prev[None]).to(device)), None)["lifted"]
            out = det(det.preprocess(torch.from_numpy(cur[None]).to(device)), prev_l)
            dets = [d for d in det.decode(out, thr=float(cfg.det_threshold), k=32) if d["kind"] == KIND_VEHICLE]
            truth = [(j, int(r[0]), float(r[1]), float(r[2])) for j, r in enumerate(lab)
                     if int(r[0]) == KIND_VEHICLE and lo <= math.hypot(float(r[1]), float(r[2])) < hi]
            ids = match_detections(dets, truth)
            tp += sum(1 for i in ids if i >= 0)
            tot += len(truth)
        rows.append([f"vehicle_{int(lo)}-{int(hi)}m", tot, 100.0 * tp / max(tot, 1), float("nan"), float("nan")])
    return rows


@torch.no_grad()
def evaluate_segnet(net: SegNet, frames: List[Tuple], cfg: Config, device: torch.device
                    ) -> Tuple[List[List[object]], List[List[object]]]:
    """Object recall/precision of the full RGB-D pipeline (segmentation, lifting, clustering) on held-out
    frames against visible ground truth, vehicle recall by distance, light-state accuracy, and pixel IoU."""
    net.eval()
    nc = len(SEG_CLASSES)
    inter, union = np.zeros(nc), np.zeros(nc)
    stats = {k: {"tp": 0, "fp": 0, "fn": 0, "err": []} for k in DET_CLASSES}
    bins = {(0.0, 15.0): [0, 0], (15.0, 25.0): [0, 0], (25.0, 40.0): [0, 0]}
    tl_ok = tl_n = tl_false = tl_none = 0
    for rgb, dep, seg, lab, tl_lab, every in frames:
        probs, _ = net.infer(rgb, dep, device)
        pred = probs.argmax(1)
        for c in range(1, nc):
            inter[c] += float(((pred == c) & (seg == c)).sum())
            union[c] += float(((pred == c) | (seg == c)).sum())
        dets, tl = rgbd_detect(probs, dep, cfg)
        vis = [(int(r[0]), float(r[1]), float(r[2])) for r in lab]
        ids, ign = match_with_ignore(dets, vis, every)
        for dd, i, g in zip(dets, ids, ign):
            st = stats[dd["kind"]]
            if i >= 0:
                st["tp"] += 1
                st["err"].append(math.hypot(dd["x"] - vis[i][1], dd["y"] - vis[i][2]))
            elif not g:
                st["fp"] += 1
        matched = {i for i in ids if i >= 0}
        for j, (k, x, y) in enumerate(vis):
            if k in stats and j not in matched:
                stats[k]["fn"] += 1
            if k == KIND_VEHICLE:
                r = math.hypot(x, y)
                for (lo, hi), cnt in bins.items():
                    if lo <= r < hi:
                        cnt[0] += int(j in matched)
                        cnt[1] += 1
        st_lab = int(tl_lab[0])
        st_pred = 0 if tl is None else {1.0: 1, 0.5: 2, 0.0: 3}[float(tl[1])]
        if st_lab > 0:                       # a light controls the ego lane within braking range
            tl_n += 1
            tl_ok += int(st_lab == st_pred)
        else:
            tl_none += 1
            tl_false += int(st_pred > 0)
    rows = []
    for k, st in stats.items():
        rows.append([KIND_NAMES[k], st["tp"] + st["fn"], 100.0 * st["tp"] / max(st["tp"] + st["fn"], 1),
                     100.0 * st["tp"] / max(st["tp"] + st["fp"], 1),
                     float(np.mean(st["err"])) if st["err"] else float("nan")])
    # recall column: correct light state when a light controls the ego lane; for the second row, the
    # share of frames without such a light in which a light token (any state) is reported
    rows.append(["traffic_light_state", tl_n, 100.0 * tl_ok / max(tl_n, 1), float("nan"), float("nan")])
    rows.append(["light_token_without_light", tl_none, 100.0 * tl_false / max(tl_none, 1), float("nan"), float("nan")])
    for (lo, hi), (tp, tot) in bins.items():
        rows.append([f"vehicle_{int(lo)}-{int(hi)}m", tot, 100.0 * tp / max(tot, 1), float("nan"), float("nan")])
    iou = [[SEG_CLASSES[c], int(union[c]), 100.0 * inter[c] / max(union[c], 1.0)] for c in range(1, nc)]
    return rows, iou


def perception_collection_templates(args: argparse.Namespace, suites: Dict[str, Suite]) -> List[Conditions]:
    """Complete non-resumable perception collection before switching policy training towns.

    Randomly drawing all source towns here used to raise TownSwitch and discard
    the unsaved network/frame buffer repeatedly. Only source data are used.
    """
    templates = suites["source"].templates
    if not templates:
        raise ValueError("No source templates available for perception collection")
    requested = town_base(str(args.source_towns).split(",")[0].strip())
    selected = [c for c in templates if town_base(c.town) == requested]
    if not selected:
        selected = [c for c in templates if c.town == templates[0].town]
    print(f"[PERCEPTION] collecting only {selected[0].town}; policy source towns={args.source_towns}")
    return selected


def run_rgbd_perception_stage(args: argparse.Namespace, cfg: Config, device: torch.device, suites: Dict[str, Suite],
                              mgr: EnvManager, out_dir: str) -> None:
    """Pre-train the RGB-D segmentation network on source-domain frames collected while the teacher
    drives, then freeze it (the perception front end of --perception rgbd)."""
    pdir = os.path.join(out_dir, "perception")
    os.makedirs(pdir, exist_ok=True)
    if os.path.exists(os.path.join(pdir, "DONE")):
        model_path = os.path.join(pdir, "segnet.pt")
        if os.path.isfile(model_path) and os.path.getsize(model_path) > 0:
            print("[PERCEPTION] segmentation network exists; skipping")
            return
        print("[PERCEPTION] DONE marker has no usable segnet.pt; rebuilding perception")
        os.remove(os.path.join(pdir, "DONE"))
    set_global_seed(int(args.seed) + 7)
    net = SegNet(cfg).to(device)
    opt = torch.optim.Adam(net.parameters(), lr=float(cfg.det_lr))
    mgr.detector = None
    cap, H, W, NC = int(args.perception_buffer), int(cfg.cam_height), int(cfg.cam_width), len(CAMERA_LAYOUT)
    stride = max(1, int(args.perception_stride))
    from numpy.lib.format import open_memmap
    paths = {k: os.path.join(pdir, f"{k}.npy") for k in ("rgb", "depth", "seg")}
    rgb_b = open_memmap(paths["rgb"], mode="w+", dtype=np.uint8, shape=(cap, NC, H, W, 3))
    dep_b = open_memmap(paths["depth"], mode="w+", dtype=np.float16, shape=(cap, NC, H, W))
    seg_b = open_memmap(paths["seg"], mode="w+", dtype=np.uint8, shape=(cap, NC, H, W))
    scn_b = np.zeros((cap, cfg.scene_dim), np.float32)
    tl_cam = np.full((cap,), -1, np.int8)          # camera with the most light-state pixels in the frame
    counts = np.zeros(len(SEG_CLASSES), np.float64)
    val: List[Tuple] = []
    n = ptr = step = it = 0
    last: Dict[str, float] = {}
    val_every = max(1, int(args.perception_steps) // max(int(args.perception_val), 1))
    w_cls = torch.tensor(SEG_DEFAULT_WEIGHTS, dtype=torch.float32, device=device)
    collection_templates = perception_collection_templates(args, suites)

    def class_weights() -> torch.Tensor:
        if counts[0] <= 0 or counts[1:].sum() <= 0:
            return w_cls
        f = counts / counts.sum()
        w = np.clip(np.sqrt(f[0] / np.maximum(f, 1e-9)), 1.0, 25.0)
        return torch.tensor(w, dtype=torch.float32, device=device)

    def train_iter(bs: int) -> Dict[str, float]:
        fi = np.random.randint(0, n, size=bs)
        ci = np.random.randint(0, NC, size=bs)
        # Light-state pixels are about 0.01% of all pixels: a quarter of every batch shows one.
        with_tl = np.nonzero(tl_cam[:n] >= 0)[0]
        if with_tl.size:
            k = bs // 4
            pick = with_tl[np.random.randint(0, with_tl.size, size=k)]
            fi[:k], ci[:k] = pick, tl_cam[pick]
        order = np.argsort(fi)
        fi, ci = fi[order], ci[order]
        net.train()
        loss, st = segnet_batch_loss(net, rgb_b[fi, ci], dep_b[fi, ci], seg_b[fi, ci], scn_b[fi], w_cls, device, True)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(net.parameters(), 10.0)
        opt.step()
        return st

    while step < int(args.perception_steps):
        base = random.choice(collection_templates)
        cond = Conditions(base.town, base.weather, random.randint(cfg.train_npc_min, cfg.train_npc_max),
                          random.randint(cfg.train_walker_min, cfg.train_walker_max), 0.0,
                          "coupled" if random.random() < float(args.barrel_source_frac) else "none", "none", True)
        env, obs, info = mgr.reset(cond)
        perturb = TeacherPerturbation(cfg)
        while True:
            try:
                _, _, term, trunc, info = env.step(perturb(teacher_label(env, cfg)))
            except BaseException as e:
                print(f"[WARN] perception step failed: {e}")
                down = server_unresponsive(env, e)
                mgr.close()
                if down:
                    raise SimulatorDown(str(e))
                break
            step += 1
            cur, dep = env.last_imgs, env.last_depth
            if cur is not None and dep is not None:
                if step % val_every == 0 and len(val) < int(args.perception_val):
                    seg = env.rgbd_seg_labels()
                    lab, tl = env.perception_labels()
                    if seg is not None:
                        val.append((cur.copy(), dep.astype(np.float32), seg, lab, tl, env.all_objects_ego()))
                elif step % stride == 0:
                    seg = env.rgbd_seg_labels()
                    if seg is not None:
                        rgb_b[ptr], dep_b[ptr], seg_b[ptr] = cur, dep.astype(np.float16), seg
                        scn_b[ptr] = env._weather_scene_vec()
                        n_tl = ((seg >= 4) & (seg <= 6)).reshape(seg.shape[0], -1).sum(1)
                        tl_cam[ptr] = int(np.argmax(n_tl)) if int(n_tl.max()) >= 4 else -1
                        counts += np.bincount(seg.reshape(-1), minlength=len(SEG_CLASSES))[:len(SEG_CLASSES)]
                        ptr, n = (ptr + 1) % cap, min(n + 1, cap)
            if n >= 16 and step % 2 == 0:
                if it % 500 == 0:
                    w_cls = class_weights()
                last = train_iter(8)
                it += 1
            if term or trunc or step >= int(args.perception_steps):
                break
        frac = counts / max(counts.sum(), 1.0)
        print(f"[PERCEPTION] steps={step}/{args.perception_steps} frames={n} val={len(val)} iters={it} "
              + " ".join(f"{k}={v:.3f}" for k, v in last.items())
              + " | label px%: " + " ".join(f"{SEG_CLASSES[c]}={100 * frac[c]:.3f}" for c in range(1, len(SEG_CLASSES)))
              + f" | frames with a light state: {int((tl_cam[:n] >= 0).sum())}")
    mgr.close()
    if counts[4:7].sum() == 0:
        print("[WARN] no traffic-light state pixels were labelled (TrafficLight.get_light_boxes unavailable?); "
              "the light token will be missing")
    w_cls = class_weights()
    total = int(args.perception_iters)
    for j in range(total):
        if j == int(0.7 * total):
            for g in opt.param_groups:
                g["lr"] = float(cfg.det_lr) * 0.1
        if n >= 1:
            last = train_iter(16)
        if (j + 1) % 1000 == 0:
            print(f"[PERCEPTION] extra iters {j + 1}/{total} " + " ".join(f"{k}={v:.3f}" for k, v in last.items()))
    torch.save(net.state_dict(), os.path.join(pdir, "segnet.pt"))
    rows, iou = evaluate_segnet(net, val, cfg, device)
    for pth in paths.values():
        try:
            os.remove(pth)
        except BaseException:
            pass
    os.makedirs(os.path.join(out_dir, "results"), exist_ok=True)
    write_csv(os.path.join(out_dir, "results", "perception_eval.csv"),
              ["class", "n_ground_truth", "recall_pct", "precision_pct", "mean_position_error_m"], rows)
    write_csv(os.path.join(out_dir, "results", "perception_pixel_iou.csv"), ["class", "n_pixels_union", "iou_pct"], iou)
    for r in rows:
        print(f"[PERCEPTION] {r[0]:>20s}: n={r[1]} recall={r[2]:.1f}% precision={r[3]:.1f}% pos_err={r[4]:.2f} m")
    open(os.path.join(pdir, "DONE"), "w").write("ok")


def load_detector(out_dir: str, cfg: Config, device: torch.device) -> Optional[nn.Module]:
    if str(cfg.perception_mode) == "rgbd":
        path = os.path.join(out_dir, "perception", "segnet.pt")
        if not os.path.exists(path):
            return None
        net = SegNet(cfg).to(device)
        net.load_state_dict(torch.load(path, map_location=device, weights_only=False))
        net.eval()
        for p_ in net.parameters():
            p_.requires_grad_(False)
        return net
    path = os.path.join(out_dir, "perception", "detector.pt")
    if not os.path.exists(path):
        return None
    det = BEVDetector(cfg).to(device)
    det.load_state_dict(torch.load(path, map_location=device, weights_only=False))
    det.eval()
    tpath = os.path.join(out_dir, "perception", "det_threshold.json")
    if os.path.exists(tpath):
        cfg.det_threshold = float(json.load(open(tpath))["det_threshold"])
    for p_ in det.parameters():
        p_.requires_grad_(False)
    return det


def run_perception_stage(args: argparse.Namespace, cfg: Config, device: torch.device, suites: Dict[str, Suite],
                         mgr: EnvManager, out_dir: str) -> None:
    """Pre-train the BEV detector on source-domain frames collected while the teacher drives
    (Sec. 4.1: pre-trained on source data and kept frozen during transfer)."""
    if str(cfg.perception_mode) == "rgbd":
        run_rgbd_perception_stage(args, cfg, device, suites, mgr, out_dir)
        return
    pdir = os.path.join(out_dir, "perception")
    os.makedirs(pdir, exist_ok=True)
    if os.path.exists(os.path.join(pdir, "DONE")):
        model_path = os.path.join(pdir, "detector.pt")
        if os.path.isfile(model_path) and os.path.getsize(model_path) > 0:
            print("[PERCEPTION] detector exists; skipping")
            return
        print("[PERCEPTION] DONE marker has no usable detector.pt; rebuilding perception")
        os.remove(os.path.join(pdir, "DONE"))
    set_global_seed(int(args.seed) + 7)
    det = BEVDetector(cfg).to(device)
    opt = torch.optim.Adam(det.parameters(), lr=float(cfg.det_lr))
    mgr.detector = None
    cap, H, W, NC = int(args.perception_buffer), int(cfg.cam_height), int(cfg.cam_width), len(CAMERA_LAYOUT)
    stride = max(1, int(args.perception_stride))
    # Frame pairs (previous, current) every `stride` steps over the WHOLE collection, kept on disk.
    # Before, a 2500-frame ring buffer held only the last one or two episodes, so the final training
    # iterations overfitted them while the validation frames came from the first episodes.
    from numpy.lib.format import open_memmap
    buf_path = os.path.join(pdir, "frames.npy")
    pairs = open_memmap(buf_path, mode="w+", dtype=np.uint8, shape=(cap, 2, NC, H, W, 3))
    labs: List[Optional[np.ndarray]] = [None] * cap
    tls = np.zeros((cap, 2), np.float32)
    val: List[Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]] = []
    n, ptr, step, it, kept = 0, 0, 0, 0, 0
    last = {}
    val_every = max(1, int(args.perception_steps) // max(int(args.perception_val), 1))
    collection_templates = perception_collection_templates(args, suites)

    def sample(bs: int):
        if n < 1:
            return None
        idx = np.sort(np.random.randint(0, n, size=bs))
        return pairs[idx, 0], pairs[idx, 1], [labs[i] for i in idx], tls[idx]

    while step < int(args.perception_steps):
        base = random.choice(collection_templates)
        cond = Conditions(base.town, base.weather, random.randint(cfg.train_npc_min, cfg.train_npc_max),
                          random.randint(cfg.train_walker_min, cfg.train_walker_max), 0.0,
                          "coupled" if random.random() < float(args.barrel_source_frac) else "none", "none", True)
        env, obs, info = mgr.reset(cond)
        prev_img = None
        perturb = TeacherPerturbation(cfg)
        while True:
            try:
                _, _, term, trunc, info = env.step(perturb(teacher_label(env, cfg)))
            except BaseException as e:
                print(f"[WARN] perception step failed: {e}")
                down = server_unresponsive(env, e)
                mgr.close()
                if down:
                    raise SimulatorDown(str(e))
                break
            step += 1
            cur = env.last_imgs
            if cur is not None and prev_img is not None:
                if step % val_every == 0 and len(val) < int(args.perception_val):
                    lab, tl = env.perception_labels()
                    val.append((prev_img, cur.copy(), lab, tl))
                elif step % stride == 0:
                    lab, tl = env.perception_labels()
                    pairs[ptr, 0], pairs[ptr, 1] = prev_img, cur
                    labs[ptr], tls[ptr] = lab, tl
                    ptr, n, kept = (ptr + 1) % cap, min(n + 1, cap), kept + 1
            if cur is not None:
                prev_img = cur.copy()
            if n >= 32 and step % 2 == 0:
                bt = sample(8)
                if bt is not None:
                    last = detector_train_step(det, opt, *bt, device)
                    it += 1
            if term or trunc or step >= int(args.perception_steps):
                break
        print(f"[PERCEPTION] steps={step}/{args.perception_steps} pairs={n} val={len(val)} iters={it} "
              + " ".join(f"{k}={v:.3f}" for k, v in last.items()))
    mgr.close()
    total = int(args.perception_iters)
    for j in range(total):
        if j == int(0.7 * total):
            for g in opt.param_groups:
                g["lr"] = float(cfg.det_lr) * 0.1        # final 30% at a tenth of the learning rate
        bt = sample(16)
        if bt is not None:
            last = detector_train_step(det, opt, *bt, device)
        if (j + 1) % 1000 == 0:
            print(f"[PERCEPTION] extra iters {j + 1}/{total} " + " ".join(f"{k}={v:.3f}" for k, v in last.items()))
    torch.save(det.state_dict(), os.path.join(pdir, "detector.pt"))
    thr = best_threshold(det, val, cfg, device)
    cfg.det_threshold = thr
    with open(os.path.join(pdir, "det_threshold.json"), "w") as f:
        json.dump({"det_threshold": thr}, f)
    print(f"[PERCEPTION] detection threshold (best vehicle F1 on validation frames) = {thr:.2f}")
    rows = evaluate_detector(det, val, cfg, device)
    try:
        os.remove(buf_path)        # the frame buffer is only needed for training (Linux keeps the mapping valid)
    except BaseException:
        pass
    os.makedirs(os.path.join(out_dir, "results"), exist_ok=True)
    write_csv(os.path.join(out_dir, "results", "perception_eval.csv"),
              ["class", "n_ground_truth", "recall_pct", "precision_pct", "mean_position_error_m"], rows)
    open(os.path.join(pdir, "DONE"), "w").write("ok")


# ======================================================
# Source stage (Algorithm 1, lines 1-3)
# ======================================================
def run_source_stage(args: argparse.Namespace, cfg: Config, device: torch.device, suites: Dict[str, Suite],
                     mgr: EnvManager, out_dir: str) -> None:
    """Source stage of Algorithm 1 (lines 1-3) with a teacher warm start, town blocks, and checkpoints."""
    sdir = os.path.join(out_dir, "source")
    os.makedirs(sdir, exist_ok=True)
    done = os.path.join(sdir, "DONE")
    if os.path.exists(done):
        print("[SOURCE] artifacts exist; skipping the source stage")
        want = str(cfg.token_encoder)
        import copy as _copy
        probe = _copy.copy(cfg)
        match_token_encoder(probe, os.path.join(sdir, "source_agent.pt"))
        if str(probe.token_encoder) != want:
            print(f"[WARN] the existing source agent uses token_encoder={probe.token_encoder}, not the current "
                  f"default {want}: this run continues the OLD architecture. For the new one, use a fresh "
                  f"--out-dir (or delete {sdir} and the runs directory).")
        return
    set_global_seed(int(args.seed))
    # recover an interrupted checkpoint first, so the encoder form is read from the checkpoint that resumes
    ckpt_recover(os.path.join(sdir, "ckpt"))
    match_token_encoder(cfg, os.path.join(sdir, "ckpt", "agent.pt"))
    agent = CaSTAgent(cfg, device, RunSpec(name="source", suite="source", method="finetune", seed=int(args.seed),
                                           budget=0, int_frac=float(cfg.source_intervention_fraction), q=cfg.cast_q))
    # Headroom for the last episode, which may run past source_steps (the ring buffer must not overwrite
    # the first teacher demonstrations).
    replay = CaSTReplay(max(int(args.source_replay), int(args.source_steps) + 10) + int(cfg.max_episode_steps)
                        + int(cfg.near_goal_timeout_extension_steps) + int(cfg.blocked_timeout_extension_steps),
                        cfg, with_coverage=False)
    nulls = NullBank(cfg)
    dataset = CausalDataset(cfg)
    labeler = Labeler(cfg, nulls, collect_nulls=True)
    ck = os.path.join(sdir, "ckpt")
    ckpt_recover(ck)
    os.makedirs(ck, exist_ok=True)
    state_path = os.path.join(ck, "state.json")
    step, ep, wall_prev, label_prev = 0, 0, 0.0, 0.0
    town, block_left, town_blocks = None, 0, 0
    if os.path.exists(state_path):
        with open(state_path) as state_file:
            st = json.load(state_file)
        step, ep, wall_prev, label_prev = int(st["step"]), int(st["ep"]), float(st["wall_s"]), float(st["label_s"])
        town, block_left = st.get("town"), int(st.get("block_left", 0))
        town_blocks = int(st.get("town_blocks", 0))
        agent.load(os.path.join(ck, "agent.pt"))
        replay.load_all(os.path.join(ck, "replay.npz"))
        dataset.load(os.path.join(ck, "dataset.npz"))
        nulls.load_state_dict(torch.load(os.path.join(ck, "nulls.pt"), weights_only=False))
        load_rng_state(os.path.join(ck, "rng.pt"))
        print(f"[SOURCE] resumed from checkpoint: step={step} episode={ep} replay={len(replay)}")
    t_start = time.perf_counter()
    towns = list(dict.fromkeys(c.town for c in suites["source"].templates))

    def checkpoint() -> None:
        tmp = ckpt_begin(ck)
        agent.save(os.path.join(tmp, "agent.pt"))
        replay.save_all(os.path.join(tmp, "replay.npz"))
        dataset.save(os.path.join(tmp, "dataset.npz"))
        torch.save(nulls.state_dict(), os.path.join(tmp, "nulls.pt"))
        torch.save(rng_state(), os.path.join(tmp, "rng.pt"))
        with open(os.path.join(tmp, "state.json"), "w") as f:
            json.dump({"step": step, "ep": ep, "wall_s": wall_prev + time.perf_counter() - t_start,
                       "label_s": label_prev + labeler.seconds, "town": town, "block_left": block_left,
                       "town_blocks": town_blocks}, f)
        ckpt_commit(ck)

    while step < int(args.source_steps):
        if town is None or block_left <= 0:
            # Episodes are run in blocks per town so that the map is not reloaded every episode.
            # Cover every source layout in each cycle, including teacher/handover
            # phases; random block selection can omit a town throughout warm-up.
            town = towns[town_blocks % len(towns)]
            town_blocks += 1
            block_left = int(args.town_block)
        block_left -= 1
        base = random.choice([c for c in suites["source"].templates if c.town == town])
        cond = Conditions(base.town, base.weather, random.randint(cfg.train_npc_min, cfg.train_npc_max),
                          random.randint(cfg.train_walker_min, cfg.train_walker_max), 0.0,
                          "coupled" if random.random() < float(args.barrel_source_frac) else "none", "none", True)
        try:
            summ, step = train_episode(mgr, agent, cond, replay, args, cfg, global_step=step, labeler=labeler,
                                       dataset=dataset, domain=0, source_buffer=None, nulls=nulls, growth=None)
        except TownSwitch:
            block_left += 1
            checkpoint()
            raise
        ep += 1
        save_value_audit(os.path.join(sdir, "value_audit"), ep, summ)
        if dataset.n >= 512 and nulls.count() >= 200:
            agent.train_causal_head(dataset, nulls, steps=200)
        share = teacher_blend(step, args)
        phase = "teacher" if share >= 1.0 else (f"blend({share:.2f})" if share > 0.0 else "policy")
        if share > 0.0 and share < 1.0:
            phase = "handover"
        n_int = sum(v for k, v in summ.int_reasons.items() if k != "hold")
        teach = (f" teacher={100.0 * summ.interventions / max(summ.steps, 1):.0f}%[" +
                 ",".join(f"{k}:{v}" for k, v in sorted(summ.int_reasons.items()) if k != "hold") + "]") if n_int else ""
        if summ.lead_n:
            teach += f" lead_car_seen={100.0 * summ.lead_seen / summ.lead_n:.0f}%"
        if summ.det_true:
            teach += f" det_recall={100.0 * summ.det_found / summ.det_true:.0f}%"
        if summ.near_true:
            teach += f" det_recall_25m={100.0 * summ.near_found / summ.near_true:.0f}%"
        teach += f" demo={demo_fraction(step, args):.2f}" if share < 1.0 else ""
        print(f"[SOURCE] ep={ep} steps={step}/{args.source_steps} phase={phase}{teach} town={town_base(town)} "
              f"RC={summ.rc:.1f}% success={int(summ.success)} coll={int(summ.collision)} end={summ.reason}{'(' + summ.hit + ')' if summ.hit else ''} "
              f"len={summ.steps} unjammed={summ.jam_removed} bc={agent.bc_weight:.2f} "
              f"bc_err={agent.last_stats.get('L_BC', float('nan')):.3f} "
              f"brake_n={agent.last_stats.get('brake_n', 0):.0f} "
              f"brake_miss={agent.last_stats.get('brake_miss', float('nan')):.2f} "
              f"brake_weak={agent.last_stats.get('brake_weak', float('nan')):.2f} "
              f"start_miss={agent.last_stats.get('start_miss', float('nan')):.2f} alpha0={agent.alpha0():.3f} "
              f"entropy={agent.last_stats.get('entropy', float('nan')):.2f} Q={agent.last_stats.get('q', float('nan')):.1f} "
              f"clip={100.0 * agent.last_stats.get('clip_frac', 0.0):.1f}% "
              f"return={summ.reward:.1f} per-step[{summ.parts}] labels={dataset.n} nulls={nulls.count()}")
        print(f"    learning: actor_updates={summ.actor_updates} critic_updates={summ.critic_updates} "
              f"mean_Q={summ.learning.get('critic_q', float('nan')):.2f} "
              f"TD_abs={summ.learning.get('td_abs', float('nan')):.2f} "
              f"TD_target={summ.learning.get('td_target', float('nan')):.2f} "
              f"target_offset={summ.learning.get('target_selection_offset', float('nan')):.3f} "
              f"critic_spread={summ.learning.get('critic_spread', float('nan')):.3f} "
              f"rl_weight={agent.actor_rl_weight:.2f} "
              f"std={summ.learning.get('policy_std', float('nan')):.4f}")
        if summ.trace:
            print(f"    trace: {summ.trace}")
        if ep % int(args.ckpt_every) == 0:
            checkpoint()
    checkpoint()
    refit_v_calibrator(agent, replay)
    agent.train_causal_head(dataset, nulls, steps=3000)
    agent.save(os.path.join(sdir, "source_agent.pt"))
    dataset.save(os.path.join(sdir, "causal_dataset.npz"))
    torch.save(nulls.state_dict(), os.path.join(sdir, "null_bank.pt"))
    replay.save(os.path.join(sdir, "source_snapshot.npz"), int(args.snapshot_rows))
    with open(os.path.join(sdir, "source_stats.json"), "w") as f:
        json.dump({"wall_s": wall_prev + time.perf_counter() - t_start, "label_s": label_prev + labeler.seconds,
                   "episodes": ep, "steps": step, "teacher_steps": int(args.teacher_steps)}, f, indent=2)
    open(done, "w").write("ok")


# ======================================================
# Source artifacts: causal heads, banks, open-set bank, prototypes
# ======================================================
class SourceArtifacts:
    def __init__(self, out_dir: str, cfg: Config, device: torch.device):
        self.dir = os.path.join(out_dir, "source")
        self.cfg, self.device = cfg, device
        self.agent_path = os.path.join(self.dir, "source_agent.pt")
        self.nulls = NullBank(cfg)
        self.nulls.load_state_dict(torch.load(os.path.join(self.dir, "null_bank.pt"), weights_only=False))
        self.dataset = CausalDataset(cfg)
        self.dataset.load(os.path.join(self.dir, "causal_dataset.npz"))
        self.snapshot = CaSTReplay.load(os.path.join(self.dir, "source_snapshot.npz"), cfg)
        self._heads: Dict[str, Dict] = {}
        self._banks: Dict[str, Tuple[CoverageModel, CoverageModel]] = {}
        self.bank_fallbacks: Dict[str, float] = {}
        match_token_encoder(cfg, self.agent_path)
        self.src = CaSTAgent(cfg, device)
        self.src.load(self.agent_path)
        stats_path = os.path.join(self.dir, "source_stats.json")
        self.stats = json.load(open(stats_path)) if os.path.exists(stats_path) else {}

    def head(self, relevance: str, null_mode: str) -> Dict:
        key = f"{relevance}_{null_mode}"
        if key in self._heads:
            return self._heads[key]
        path = os.path.join(self.dir, f"causal_head_{key}.pt")
        if os.path.exists(path):
            sd = torch.load(path, map_location=self.device, weights_only=False)
        else:
            tmp = CaSTAgent(self.cfg, self.device, RunSpec(name="h", suite="", method="cast", seed=0, budget=0,
                                                          relevance=relevance, null_mode=null_mode))
            tmp.train_causal_head(self.dataset, self.nulls, steps=4000)
            sd = tmp.causal_head.state_dict()
            torch.save(sd, path)
        self._heads[key] = sd
        return sd

    @torch.no_grad()
    def banks(self, spec: RunSpec) -> Tuple[CoverageModel, CoverageModel]:
        key = f"{spec.relevance}_{spec.null_mode}_q{spec.q}_k{spec.k}_m{int(spec.causal_mask)}"
        if key in self._banks:
            return self._banks[key]
        cfg, dev = self.cfg, self.device
        head = CausalHead(cfg).to(dev)
        head.load_state_dict(self.head(spec.relevance, spec.null_mode))
        snap = self.snapshot
        n = len(snap)
        cut = int(0.8 * n)
        W = self.src.actor.enc.W
        hs, scores, inb, sc_bank, sc_ref = [], [], [], [], []
        for lo in range(0, n, 4096):
            idx = np.arange(lo, min(n, lo + 4096))
            b = snap.sample(0, dev, idx)
            valid = b["mask"] > 0.5
            hs.append(W(b["edges"])[valid])
            scores.append(head(b["scalars"], b["edges"])[valid])
            inb.append(torch.from_numpy(idx < cut).to(dev).unsqueeze(1).expand_as(valid)[valid])
            sc_bank.append(b["scene"][torch.from_numpy(idx < cut).to(dev)])
            sc_ref.append(b["scene"][torch.from_numpy(idx >= cut).to(dev)])
        h_all, s_all, in_all = torch.cat(hs, 0), torch.cat(scores, 0), torch.cat(inb, 0)
        if spec.causal_mask:
            keep = s_all >= 1.0 - spec.q
            need = max(20 * spec.k, 500)
            if int(keep.sum()) < need and h_all.shape[0] > 0:
                # Too few interactions pass tau_c (short source training): keep the highest-scoring ones.
                kk = min(need, h_all.shape[0])
                thr = torch.topk(s_all, kk).values[-1]
                keep = s_all >= thr
                print(f"[WARN] coverage bank {key}: only a few interactions reach tau_c; "
                      f"using the top {int(keep.sum())} by causal score (threshold {float(thr):.3f})")
                self.bank_fallbacks[key] = float(thr)
        else:
            keep = torch.ones_like(s_all, dtype=torch.bool)
        feats = [h_all[keep & in_all]]
        refs = [h_all[keep & ~in_all]]

        def make(bank_l, ref_l, cap, nref) -> CoverageModel:
            bank = torch.cat(bank_l, 0)
            ref = torch.cat(ref_l, 0)
            g = torch.Generator().manual_seed(0)       # the same bank in every process and resume
            if bank.shape[0] > cap:
                bank = bank[torch.randperm(bank.shape[0], generator=g)[:cap].to(dev)]
            if ref.shape[0] > nref:
                ref = ref[torch.randperm(ref.shape[0], generator=g)[:nref].to(dev)]
            if bank.shape[0] < spec.k + 1 or ref.shape[0] < 10:
                raise RuntimeError("too few source interactions to build the coverage bank; train longer")
            kb = KNNBank(bank, spec.k)
            ref_d = torch.sort(kb.kdist(ref)[:, -1]).values
            return CoverageModel(bank, ref_d, spec.k)

        out = (make(feats, refs, cfg.bank_size, cfg.n_reference), make(sc_bank, sc_ref, cfg.bank_size, cfg.n_reference))
        self._banks[key] = out
        return out

    @torch.no_grad()
    def openset_bank(self, q: float) -> Tuple[KNNBank, float]:
        snap, dev = self.snapshot, self.device
        n = len(snap)
        idx = np.random.default_rng(0).permutation(n)
        b1 = snap.sample(0, dev, idx[: min(20000, int(0.8 * n))])
        b2 = snap.sample(0, dev, idx[int(0.8 * n):][:5000])
        keep = self.src.actor.actor_scalar_keep
        f1 = self.src.actor.enc(b1["scalars"] * keep, b1["edges"], b1["mask"])[0]
        f2 = self.src.actor.enc(b2["scalars"] * keep, b2["edges"], b2["mask"])[0]
        bank = KNNBank(f1, self.cfg.knn_k)
        thr = float(torch.quantile(bank.kdist(f2)[:, -1], 1.0 - q))
        return bank, thr

    @torch.no_grad()
    def prototypes(self, k: int = 16) -> Tuple[torch.Tensor, torch.Tensor]:
        snap, dev = self.snapshot, self.device
        idx = np.random.default_rng(0).permutation(len(snap))[:20000]
        b = snap.sample(0, dev, idx)
        z, _, _ = self.src.actor.enc.pooled(b["scalars"] * self.src.actor.actor_scalar_keep, b["edges"], b["mask"])
        cent = z[torch.from_numpy(np.random.default_rng(1).permutation(z.shape[0])[:k]).to(dev)].clone()
        for _ in range(25):
            a = torch.cdist(z, cent).argmin(1)
            for j in range(k):
                if (a == j).any():
                    cent[j] = z[a == j].mean(0)
        a = torch.cdist(z, cent).argmin(1)
        prior = torch.bincount(a, minlength=k).float()
        return cent, prior / prior.sum()


def make_run_agent(spec: RunSpec, art: SourceArtifacts, cfg: Config, device: torch.device) -> CaSTAgent:
    cfg.token_encoder = art.cfg.token_encoder        # run configs are copies; match the source agent
    agent = CaSTAgent(cfg, device, spec)
    agent.load(art.agent_path)
    if not agent.sig_ref_fitted:
        # source agents saved before the aleatoric scale existed: fit it once on the source snapshot
        if getattr(art, "_sig_ref", None) is None:
            refit_v_calibrator(agent, art.snapshot)
            art._sig_ref = (float(agent.sig_ref), agent.v_cal.state_dict())
            print(f"[CaST] aleatoric scale fitted on the source snapshot: sig_ref = {agent.sig_ref:.3f} m^2")
        agent.sig_ref, agent.sig_ref_fitted = art._sig_ref[0], True
        agent.v_cal.load_state_dict(art._sig_ref[1])
    for g in agent.actor_opt.param_groups:
        g["lr"] = float(cfg.target_actor_lr)
    agent.target_updates = 0
    agent.freeze_source_copies()
    agent.causal_head.load_state_dict(art.head(spec.relevance, spec.null_mode))
    if spec.is_cast():
        c, sc = art.banks(spec)
        agent.cov, agent.scene_cov = c.clone(), sc.clone()
    if spec.method == "openset_entropy":
        agent.src_feat_bank, agent.openset_thr = art.openset_bank(spec.q)
    if cfg.target_bc_weight > 0.0 or spec.method in ("uniform", "contrastive_filter", "ot_filter"):
        agent.source_buffer = art.snapshot
    if spec.method == "deconf":
        cent, prior = art.prototypes()
        agent.actor.enc.proto.protos.copy_(cent)
        agent.actor.enc.proto.prior.copy_(prior)
    return agent


# ======================================================
# Template samplers (Sec. 3.5, Eq. 26) and adaptation baselines
# ======================================================
def least_visited_town(templates, curve):
    towns = list(dict.fromkeys(c.town for c in templates))
    if not towns:
        raise ValueError("No adaptation towns available")
    return min(towns, key=lambda town: sum(row.get("town") == town for row in curve))


def undercovered_weather(templates, curve, town, minimum):
    if minimum <= 0:
        return None
    weathers = list(dict.fromkeys(c.weather for c in templates if c.town == town))
    counts = {w: sum(r.get("town") == town and r.get("weather") == w for r in curve) for w in weathers}
    pending = [w for w in weathers if counts[w] < minimum]
    return min(pending, key=counts.get) if pending else None


class TemplateSampler:
    def __init__(self, kind: str, templates: List[Conditions], cfg: Config, eps: float, temp: float):
        self.kind = kind
        self.templates = list(templates)
        self.base_n = len(templates)
        self.score = [1.0] * len(templates)
        self.count = [0] * len(templates)
        self.eps, self.temp = float(eps), float(temp)
        self.last = 0

    def sample(self, town: Optional[str] = None, weather: Optional[str] = None) -> Conditions:
        """Eq. (27). With a town given, the distribution is restricted to the templates of that town
        (episodes run in town blocks so that the map is not reloaded every episode)."""
        if self.kind == "uniform":
            cand = [i for i in range(self.base_n) if (town is None or self.templates[i].town == town)
                    and (weather is None or self.templates[i].weather == weather)]
            if not cand:
                raise ValueError(f"No adaptation templates for town={town} weather={weather}")
            self.last = random.choice(cand)
        else:
            cand = [i for i in range(len(self.templates)) if (town is None or self.templates[i].town == town)
                    and (weather is None or self.templates[i].weather == weather)]
            if not cand:
                raise ValueError(f"No adaptation templates for town={town} weather={weather}")
            s = np.asarray([self.score[i] for i in cand], np.float64)
            p = np.exp((s - s.max()) / max(self.temp, 1e-6))
            p = (1.0 - self.eps) * p / p.sum() + self.eps / len(s)
            self.last = cand[int(np.random.choice(len(cand), p=p))]
        return self.templates[self.last]

    def state(self) -> Dict[str, object]:
        return {"templates": [c.__dict__ for c in self.templates], "score": self.score, "count": self.count,
                "base_n": self.base_n, "last": self.last}

    def load(self, d: Dict[str, object]) -> None:
        self.templates = [Conditions(**c) for c in d["templates"]]
        self.score, self.count, self.base_n = list(d["score"]), list(d["count"]), int(d["base_n"])
        self.last = int(d.get("last", 0))  # older checkpoints resample before updating
        if not 0 <= self.last < len(self.templates):
            raise ValueError("checkpoint contains an invalid sampler template index")

    def update(self, summ: EpisodeSummary) -> None:
        if self.kind == "uniform":
            return
        if self.kind == "cast":
            val = summ.mean_U
        elif self.kind == "uncertainty":
            val = summ.mean_ubar
        else:  # safety_critical: criticality of the episode
            val = 1.0 if summ.collision else 1.0 - summ.min_phi
        i = self.last
        self.count[i] += 1
        self.score[i] += (val - self.score[i]) / max(self.count[i], 1) if self.count[i] > 1 else (val - self.score[i])
        if self.kind == "cast" and len(self.templates) < 3 * self.base_n and val >= np.quantile(self.score, 0.75):
            self.expand(i)

    def expand(self, i: int) -> None:
        """Local counterfactual expansion around a template with high unresolved causal mass."""
        c = self.templates[i]
        weather = random.choice(ADVERSE_WEATHERS) if c.weather in ADVERSE_WEATHERS else c.weather
        child = Conditions(c.town, weather, int(np.clip(c.npc + random.randint(-2, 2), 0, 40)),
                           int(np.clip(c.walkers + random.randint(0, 3), 0, 40)),
                           float(np.clip(c.degradation + random.choice([-0.1, 0.0, 0.1]), 0.0, 1.0)),
                           c.barrel_mode, c.queue_mode, c.in_source_support)
        self.templates.append(child)
        self.score.append(self.score[i])
        self.count.append(0)


def run_adaptation(spec: RunSpec, art: SourceArtifacts, suites: Dict[str, Suite], mgr: EnvManager,
                   args: argparse.Namespace, cfg: Config, device: torch.device, run_dir: str) -> CaSTAgent:
    ckpt = os.path.join(run_dir, "agent.pt")
    agent = make_run_agent(spec, art, cfg, device)
    if os.path.exists(os.path.join(run_dir, "ADAPT_DONE")):
        agent.load(ckpt)
        load_growth(agent, run_dir)
        return agent
    set_global_seed(1000 + int(spec.seed))
    suite = suites[spec.suite]
    replay = CaSTReplay(min(int(cfg.replay_size), max(1, spec.budget) * int(cfg.max_episode_steps)), cfg,
                        with_coverage=spec.is_cast())
    dataset = None
    labeler = None
    if spec.is_cast():
        dataset = CausalDataset(cfg)
        dataset.load(os.path.join(art.dir, "causal_dataset.npz"))
        labeler = Labeler(cfg, art.nulls, collect_nulls=False)
    sampler = TemplateSampler(spec.sampler, suite.templates, cfg, spec.eps, cfg.sampler_temp)
    curve: List[Dict[str, object]] = []
    t0 = time.perf_counter()
    label_s, label_prev, wall_prev, ep0 = 0.0, 0.0, 0.0, 0
    town, block_left = None, 0
    ck = os.path.join(run_dir, "ckpt")
    retry_cond: Optional[Conditions] = None
    ckpt_recover(ck)
    state_path = os.path.join(ck, "state.json")
    if os.path.exists(state_path):
        with open(state_path) as state_file:
            st = json.load(state_file)
        ep0, curve, wall_prev = int(st["ep"]), list(st["curve"]), float(st["wall_s"])
        town, block_left, label_prev = st.get("town"), int(st.get("block_left", 0)), float(st.get("label_s", 0.0))
        label_s = label_prev
        sampler.load(st["sampler"])
        if st.get("pending_condition") is not None:
            if "last" not in st["sampler"]:
                raise ValueError("pending condition has no saved sampler index")
            retry_cond = Conditions(**st["pending_condition"])
            if retry_cond.town != town:
                raise ValueError("pending condition does not match the checkpoint town")
        agent.load(os.path.join(ck, "agent.pt"))
        replay.load_all(os.path.join(ck, "replay.npz"))
        if dataset is not None and os.path.exists(os.path.join(ck, "dataset.npz")):
            dataset.load(os.path.join(ck, "dataset.npz"))
        load_growth(agent, ck)
        # The random streams saved with the checkpoint are restored; checkpoints written before they were
        # saved fall back to a seed that differs per resume point.
        if not load_rng_state(os.path.join(ck, "rng.pt")):
            set_global_seed(1000 + int(spec.seed) + 7919 * ep0)
        print(f"[ADAPT {spec.run_id()}] resumed at episode {ep0}")

    def checkpoint(ep_done: int) -> None:
        tmp = ckpt_begin(ck)
        agent.save(os.path.join(tmp, "agent.pt"))
        replay.save_all(os.path.join(tmp, "replay.npz"))
        if dataset is not None:
            dataset.save(os.path.join(tmp, "dataset.npz"))
        save_growth(agent, tmp)
        torch.save(rng_state(), os.path.join(tmp, "rng.pt"))
        with open(os.path.join(tmp, "state.json"), "w") as f:
            json.dump({"ep": ep_done, "curve": curve, "wall_s": wall_prev + time.perf_counter() - t0,
                       "label_s": label_prev + (labeler.seconds if labeler is not None else 0.0),
                       "town": town, "block_left": block_left, "sampler": sampler.state(),
                       "pending_condition": retry_cond.__dict__ if retry_cond is not None else None},
                      f, default=str)
        ckpt_commit(ck)

    ep, aborted_retries = ep0, 0
    while ep < int(spec.budget):
        if retry_cond is not None:
            cond = retry_cond
        else:
            cond = None
        if cond is None and (town is None or block_left <= 0):
            town = (least_visited_town(suite.templates, curve) if cfg.balanced_adaptation
                    else sampler.sample().town)
            # blocks of --town-block episodes, but at most B/4 (B=5: 1, B=10: 2, B=20: 5, B=40: 5)
            block_left = max(1, min(int(args.town_block), int(spec.budget) // 4))
        if cond is None:
            block_left -= 1
            weather = undercovered_weather(suite.templates, curve, town, cfg.adapt_min_weather_episodes)
            cond = with_distractors(sampler.sample(town, weather), suite.distractors)
            # Reserve this block slot exactly once, including across map switches/restarts.
            retry_cond = cond
        growth: List[Tuple[np.ndarray, np.ndarray]] = []
        try:
            summ, _ = train_episode(mgr, agent, cond, replay, args, cfg, global_step=10 ** 9, labeler=labeler,
                                    dataset=dataset, domain=1,
                                    source_buffer=agent.source_buffer,
                                    nulls=art.nulls, growth=growth if spec.is_cast() else None)
        except TownSwitch:
            checkpoint(ep)
            raise
        if summ.aborted:
            # A simulator error (not the policy) ended the episode: the same condition is driven again and
            # does not use the budget, update the template sampler, or count for coverage growth.
            # Transitions collected before the error stay in the replay buffer and the updates made on them
            # are kept. After three failures in a row the run stops as a simulator failure (the supervisor
            # restarts CARLA and resumes from this checkpoint); it is never counted as an episode.
            aborted_retries += 1
            if aborted_retries > 2:
                checkpoint(ep)
                raise SimulatorDown(f"episode {ep + 1} of {spec.run_id()} aborted {aborted_retries} times in a row")
            retry_cond = cond
            print(f"[ADAPT {spec.run_id()}] ep={ep + 1}: aborted by a simulator error, driving the same "
                  f"condition again ({aborted_retries}/2)")
            continue
        aborted_retries = 0
        sampler.update(summ)
        retry_cond = None
        if spec.is_cast() and growth:
            valid = (not summ.aborted) and (not summ.collision) and summ.min_phi >= cfg.validation_margin
            if valid or not spec.validated_growth:
                hs = np.concatenate([g[0] for g in growth], 0)
                sc = np.concatenate([g[1] for g in growth], 0)
                if hs.shape[0]:
                    agent.cov.add_points(torch.from_numpy(hs).to(device))
                if sc.shape[0]:
                    agent.scene_cov.add_points(torch.from_numpy(sc).to(device))
        if spec.is_cast() and dataset is not None and labeler is not None:
            agent.train_causal_head(dataset, art.nulls, steps=100)
            label_s = label_prev + labeler.seconds
        if spec.method == "ace" and len(replay) > 200:
            n = len(replay)
            agent.update_ace_weights(replay.data["actions"][:n], replay.data["rewards"][:n])
        curve.append({"episode": ep + 1, "mean_U": summ.mean_U, "mean_ubar": summ.mean_ubar, "rc": summ.rc,
                      "success": int(summ.success), "collision": int(summ.collision), "km": summ.km,
                      "template": cond.key(), "town": cond.town, "weather": cond.weather,
                      "end_reason": summ.reason, "jam_removed": summ.jam_removed,
                      "q": agent.last_stats.get("q"),
                      "clip_frac": agent.last_stats.get("clip_frac", 0.0),
                      "source_bc": agent.last_stats.get("L_source_BC"),
                      "source_brake_miss": agent.last_stats.get("source_brake_miss"),
                      "source_start_miss": agent.last_stats.get("source_start_miss"),
                      "critic_q": agent.last_stats.get("critic_q"),
                      "learning_mean": summ.learning, "actor_updates": summ.actor_updates,
                      "critic_updates": summ.critic_updates,
                      "synthetic_takeovers": summ.takeover_labels, "target_expert_labels": summ.target_expert_labels,
                      "target_state_bc": agent.last_stats.get("L_target_BC"),
                      "value_audit": {k: v for k, v in summ.value_audit.items() if k != "samples"}})
        save_value_audit(os.path.join(run_dir, "value_audit"), ep + 1, summ)
        det = (f"det_recall={100.0 * summ.det_found / max(summ.det_true, 1):.0f}% "
               f"det_recall_25m={100.0 * summ.near_found / max(summ.near_true, 1):.0f}% ") if summ.det_true else ""
        print(f"[ADAPT {spec.run_id()}] ep={ep + 1}/{spec.budget} town={town_base(cond.town)} weather={cond.weather} "
              f"{det}RC={summ.rc:.1f}% "
              f"coll={int(summ.collision)} end={summ.reason}{'(' + summ.hit + ')' if summ.hit else ''} "
              f"U={summ.mean_U:.3f} ubar={summ.mean_ubar:.3f} "
              f"mean_critic_Q={summ.learning.get('critic_q', float('nan')):.2f} "
              f"TD_abs={summ.learning.get('td_abs', float('nan')):.2f} "
              f"actor_updates={summ.actor_updates} critic_updates={summ.critic_updates} "
              f"src_bc={agent.last_stats.get('L_source_BC', float('nan')):.3f} "
              f"target_bc={summ.learning.get('L_target_BC', float('nan')):.3f} "
              f"L_A={summ.learning.get('L_A', float('nan')):.3f} L_I={summ.learning.get('L_I', float('nan')):.3f} "
              f"actor_Q={summ.learning.get('q', float('nan')):.2f} "
              f"target_offset={summ.learning.get('target_selection_offset', float('nan')):.3f} "
              f"clip={100.0 * agent.last_stats.get('clip_frac', 0.0):.1f}%"
              f"{' (' + summ.ubar_parts + ')' if summ.ubar_parts else ''}")
        if summ.trace:
            print(f"    trace: {summ.trace}")
        # Save the final episode too: interrupted export should not repeat training.
        checkpoint(ep + 1)
        ep += 1
    agent.save(ckpt)
    save_growth(agent, run_dir)
    with open(os.path.join(run_dir, "adapt.json"), "w") as f:
        json.dump({"spec": spec.__dict__, "curve": curve, "wall_s": wall_prev + time.perf_counter() - t0, "label_s": label_s,
                   "motion_nll": float(getattr(agent, "motion_nll_ema", float("nan")))},
                  f, indent=2, default=str)
    with open(os.path.join(run_dir, "ADAPT_DONE"), "w") as done_file:
        done_file.write("ok")
    return agent


def save_growth(agent: CaSTAgent, run_dir: str) -> None:
    if agent.cov is None:
        return
    d = {"z": agent.cov._add_cat.cpu() if agent.cov._add_cat is not None else None,
         "b": agent.scene_cov._add_cat.cpu() if agent.scene_cov._add_cat is not None else None}
    torch.save(d, os.path.join(run_dir, "growth.pt"))


def load_growth(agent: CaSTAgent, run_dir: str) -> None:
    p = os.path.join(run_dir, "growth.pt")
    if agent.cov is None or not os.path.exists(p):
        return
    d = torch.load(p, weights_only=False)
    if d.get("z") is not None:
        agent.cov.add_points(d["z"].to(agent.device))
    if d.get("b") is not None:
        agent.scene_cov.add_points(d["b"].to(agent.device))



# ======================================================
# Evaluation (Sec. 4)
# ======================================================
IS_PENALTY = {"collision_walker": 0.50, "collision_vehicle": 0.60, "collision_static": 0.65, "red_light": 0.70}


def expert_drive_action(env: CaSTCarlaEnv, scene: PrivScene, cfg: Config) -> np.ndarray:
    """Privileged expert driver for the reference row of Table 1: IDM longitudinal control of the
    expert (Sec. 3.4), combined conservatively with route-guidance speed and steering."""
    lon = np.asarray(expert_action(scene, scene.entities, cfg), dtype=np.float32)
    guide = np.asarray(env.route_guidance_action(), dtype=np.float32)
    if lon.shape != (3,) or guide.shape != (3,) or not (np.isfinite(lon).all() and np.isfinite(guide).all()):
        raise ValueError("Teacher produced an invalid action; refusing to store a corrupt label")
    # IDM handles predicted actors; route guidance handles turn/lane/endpoint speed.
    # Either controller may require braking. Never discard its brake request or
    # send simultaneous throttle/brake (the environment otherwise picks a winner).
    longitudinal = min(float(lon[0] - lon[1]), float(guide[0] - guide[1]))
    return np.array([np.clip(longitudinal, 0.0, 1.0),
                     np.clip(-longitudinal, 0.0, 1.0), np.clip(guide[2], -1.0, 1.0)], np.float32)


def bin_index(rho: float, edges: Sequence[float]) -> int:
    for i in range(len(edges) - 1):
        if edges[i] <= rho < edges[i + 1]:
            return i
    return len(edges) - 2


def event_metrics(events: List[Dict[str, object]], s: np.ndarray, v: np.ndarray, red: np.ndarray,
                  lead: np.ndarray, dt: float) -> Dict[str, float]:
    """False stops and missed stops (barrel suite) and hesitation (queue suite), Table 2."""
    out = {"barrel_events": 0, "false_stops": 0, "red_events": 0, "missed_stops": 0, "queue_events": 0, "hesitations": 0}
    if len(s) == 0:
        return out
    stop_len = int(round(1.0 / dt))
    for ev in events:
        se = float(ev["s"])
        typ = ev["type"]
        reached = bool(np.any(s >= se - 25.0))
        if not reached:
            continue
        if typ == "barrel":
            out["barrel_events"] += 1
            win = (s >= se - 25.0) & (s <= se + 10.0)
            slow = win & (v < 0.5) & (lead > 12.0)
            run, best = 0, 0
            for f in slow:
                run = run + 1 if f else 0
                best = max(best, run)
            out["false_stops"] += int(best >= stop_len)
        elif typ == "red_stop" and not bool(ev.get("barrels", False)):
            out["red_events"] += 1
            win = (s >= se - 2.0) & (s <= se + 8.0)
            out["missed_stops"] += int(bool(np.any(red[win])))
        elif typ == "green_queue":
            out["queue_events"] += 1
            appr = (s >= se - 30.0) & (s < se - 25.0)
            win = (s >= se - 25.0) & (s <= se) & (lead > 12.0)
            if appr.any() and win.any():
                va = float(np.mean(v[appr]))
                if va > 1.0 and float(np.min(v[win])) < 0.3 * va:
                    out["hesitations"] += 1
    return out


def evaluate_suite(agent: Optional[CaSTAgent], ectx: Optional[CaSTAgent], suite: Suite, mgr: EnvManager,
                   cfg: Config, *, scripted: bool, proxy_W: Optional[nn.Module], nulls: Optional[NullBank],
                   label_frac: float = 0.02, partial_path: Optional[str] = None) -> Dict[str, object]:
    """Evaluate a policy (agent) or the privileged expert (agent=None) on the evaluation routes of a suite."""
    routes_out: List[Dict[str, object]] = []
    steps_out: Dict[str, List] = {k: [] for k in ("route", "ubar", "ubar_noscene", "bin", "fail", "coll", "dist_m",
                                                  "ttc", "missed", "frame_dropped")}
    ident: List[Dict[str, object]] = []
    acc_pairs: List[Tuple[float, float]] = []
    gap_rows: List[Tuple[int, float]] = []
    share_rows: List[float] = []
    reliance_dis: List[float] = []
    edges_bins = list(cfg.coverage_bins)
    head_ms: List[float] = []
    done_routes = 0
    if partial_path is not None and os.path.exists(partial_path):
        pr = json.load(open(partial_path))
        routes_out, steps_out, ident = pr["routes"], pr["steps"], pr["ident"]
        acc_pairs = [tuple(x) for x in pr["acc_pairs"]]
        gap_rows = [tuple(x) for x in pr["gap"]]
        share_rows, reliance_dis, head_ms = pr["share_uncovered"], pr["reliance_distractor"], pr["head_ms_list"]
        done_routes = len(routes_out)
        print(f"[EVAL {suite.name}] resuming after {done_routes} finished routes")

    def pack() -> Dict[str, object]:
        return {"routes": routes_out, "steps": steps_out, "ident": ident, "acc_pairs": acc_pairs, "gap": gap_rows,
                "share_uncovered": share_rows, "reliance_distractor": reliance_dis,
                "head_ms": float(np.mean(head_ms)) if head_ms else float("nan"), "head_ms_list": head_ms}

    for ri, cond in enumerate(suite.eval_routes):
        if ri < done_routes:
            continue
        marks = (len(ident), len(share_rows), len(gap_rows), len(reliance_dis), len(head_ms))
        for attempt in range(3):
            aborted = False
            env, obs, info = mgr.reset(cond)
            support = 1.0 if cond.in_source_support else 0.0
            labeler = Labeler(cfg, None, collect_nulls=False) if (nulls is not None and agent is not None) else None
            rec = {k: [] for k in ("s", "v", "red", "lead", "ttc", "coll", "ubar", "ubar_ns", "bin", "missed", "fd", "dL")}
            prev_dist, s_ego = 0.0, 0.0
            t = 0
            while True:
                ids, kinds = env.slot_info()
                scene = env.privileged_scene()
                lead_gap, _, lead_id = expert_lead(scene, scene.entities, include_signals=False, predict=False)
                missed = bool(lead_id >= 0 and lead_gap < 30.0 and lead_id not in set(int(i) for i in ids))
                ub, ub_ns, st_bin = 0.0, 0.0, -1
                q = None
                if agent is not None:
                    t_h = time.perf_counter()
                    dkm, dk0m = coverage_dk(agent, obs) if agent.spec.is_cast() else (None, None)
                    q = agent.step_quantities(obs, dkm, dk0m, support)
                    head_ms.append(1000.0 * (time.perf_counter() - t_h))
                    ub, ub_ns = q["ubar"], q["ubar_noscene"]
                    action = agent.act(obs, deterministic=True)
                else:
                    action = expert_drive_action(env, scene, cfg)
                if ectx is not None:
                    dke, dk0e = coverage_dk(ectx, obs)
                    qe = ectx.step_quantities(obs, dke, dk0e, support)
                    causal = (qe["c_hat"] >= 1.0 - ectx.spec.q) & (obs["mask"] > 0.5)
                    rho_state = float(qe["rho"][causal].min()) if causal.any() else float(qe["rho0"])
                    st_bin = bin_index(rho_state, edges_bins)
                    if causal.any():
                        share_rows.extend([float(r < ectx.spec.q) for r in qe["rho"][causal]])
                    if proxy_W is not None and causal.any():
                        ed = torch.from_numpy(obs["edges"][causal]).to(ectx.device)
                        with torch.no_grad():
                            gap = (proxy_W(ed) - ectx.W_S(ed)).norm(dim=-1).cpu().numpy()
                        for r_, g_ in zip(qe["rho"][causal], gap):
                            gap_rows.append((bin_index(float(r_), edges_bins), float(g_)))
                # identification and reliance on distractors (scripted suites)
                if scripted and agent is not None and q is not None:
                    in_win = any(float(ev["s"]) - 25.0 <= s_ego <= float(ev["s"]) + 2.0 for ev in env.scenario_events)
                    if in_win:
                        nvalid = max(float(obs["mask"].sum()), 1.0)
                        role_of = {e.eid: e.role for e in scene.entities}
                        for slot in range(cfg.max_entities):
                            if obs["mask"][slot] < 0.5:
                                continue
                            eid = int(ids[slot])
                            role = "cause" if eid == lead_id and lead_gap < 20.0 else role_of.get(eid, "other")
                            if role not in ("cause", "distractor"):
                                continue
                            occ = env.tokens_with_occlusion(eid) if (t % 2 == 0 and eid >= 0 and int(kinds[slot]) != KIND_SIGNAL) else None
                            r_i = agent.reliance(obs, slot, int(kinds[slot]), occ) if t % 2 == 0 else float("nan")
                            if role == "distractor" and np.isfinite(r_i):
                                reliance_dis.append(r_i)
                            ident.append({"route": ri, "t": t, "slot": slot, "role": role, "c_hat": float(q["c_hat"][slot]),
                                          "attn": float(q["w"][slot] * nvalid), "reliance": r_i,
                                          "kind": int(kinds[slot]), "e_full": float("nan"), "dist": float("nan")})
                        if labeler is not None and t % 5 == 0:
                            labeler.schedule(t, scene, ids, kinds, obs, {"ident_t": t, "route": ri})
                elif labeler is not None and random.random() < label_frac:
                    labeler.schedule(t, scene, ids, kinds, obs, {"acc": q["c_hat"].copy() if q is not None else None})
                try:
                    obs, _, term, trunc, info = env.step(action)
                except BaseException as e:
                    print(f"[WARN] eval step failed: {e}")
                    down = server_unresponsive(env, e)
                    mgr.close()
                    if down:
                        raise SimulatorDown(str(e))
                    aborted = True
                    break
                if labeler is not None:
                    labeler.push_action(np.asarray(info.get("command_action", action), np.float32))
                dist_now = safe_float(info.get("distance_driven_m", 0.0))
                s_ego = safe_float(info.get("route_s", s_ego))
                rec["s"].append(s_ego)
                rec["v"].append((dist_now - prev_dist) / cfg.dt)
                prev_dist = dist_now
                rec["red"].append(bool(info.get("red_light_violation", False)))
                rec["lead"].append(lead_gap)
                rec["ttc"].append(safe_float(info.get("time_to_conflict", 999.0), 999.0))
                rec["coll"].append(bool(info.get("collision", False)))
                rec["ubar"].append(ub)
                rec["ubar_ns"].append(ub_ns)
                rec["bin"].append(st_bin)
                rec["missed"].append(missed)
                rec["fd"].append(bool(info.get("frame_dropped", False)))
                rec["dL"].append(abs(safe_float(info.get("dL", 0.0))))
                t += 1
                if term or trunc:
                    break
            if not aborted:
                break
            # A simulator failure is not a driving failure: drop the partial route and drive it again.
            del ident[marks[0]:], share_rows[marks[1]:], gap_rows[marks[2]:], reliance_dis[marks[3]:], head_ms[marks[4]:]
            if attempt < 2:
                print(f"[EVAL {suite.name}] route {ri + 1}: simulator failure, retrying ({attempt + 2}/3)")
        if aborted:
            print(f"[EVAL {suite.name}] route {ri + 1}: failed three times; scored as driven so far")
        # delayed labels: causal-head accuracy and identification ground truth
        if labeler is not None:
            for item in labeler.ready(final=True):
                ex = item.get("extra", {})
                for slot, (e_full, _em, dist, _role) in item["slots"].items():
                    if ex.get("acc") is not None:
                        acc_pairs.append((float(ex["acc"][slot]), e_full, dist))
                    if "ident_t" in ex:
                        for row in ident:
                            if row["route"] == ri and row["t"] == ex["ident_t"] and row["slot"] == slot:
                                row["e_full"], row["dist"] = e_full, dist
        # route metrics
        rc = safe_float(info.get("route_completion_pct", 0.0))
        infr = list(env.infractions)
        is_score = float(np.prod([IS_PENALTY.get(k, 1.0) for k, _ in infr])) if infr else 1.0
        km = max(safe_float(info.get("distance_driven_m", 0.0)) / 1000.0, 1e-6)
        det_true, det_found = env.detection_recall_counts()
        arr = {k: np.asarray(v) for k, v in rec.items()}
        evm = event_metrics(env.scenario_events, arr["s"], arr["v"], arr["red"], arr["lead"], cfg.dt) if scripted else {}
        n = len(arr["ttc"])
        fail = np.zeros(n, bool)
        bad = arr["coll"] | arr["red"] | (arr["ttc"] < 1.0) if n else np.zeros(0, bool)
        horizon = int(round(2.0 / cfg.dt))
        if n:
            idx_bad = np.nonzero(bad)[0]
            for j in idx_bad:
                fail[max(0, j - horizon):j + 1] = True
        routes_out.append({"route": ri, "conditions": cond.key(), "RC": rc, "IS": is_score, "DS": rc * is_score,
                           "success": int(bool(info.get("success", False))), "km": km,
                           "collisions": sum(1 for k, _ in infr if k.startswith("collision")),
                           "red_lights": sum(1 for k, _ in infr if k == "red_light"),
                           "min_ttc": float(np.min(arr["ttc"])) if n else 999.0,
                           "cte": float(np.mean(arr["dL"])) if n else 0.0,
                           "det_true": det_true, "det_found": det_found, **evm})
        for j in range(n):
            steps_out["route"].append(ri)
            steps_out["ubar"].append(float(arr["ubar"][j]))
            steps_out["ubar_noscene"].append(float(arr["ubar_ns"][j]))
            steps_out["bin"].append(int(arr["bin"][j]))
            steps_out["fail"].append(bool(fail[j]))
            steps_out["coll"].append(bool(arr["coll"][j]))
            steps_out["dist_m"].append(float(max(arr["v"][j], 0.0) * cfg.dt))
            steps_out["ttc"].append(float(arr["ttc"][j]))
            steps_out["missed"].append(bool(arr["missed"][j]))
            steps_out["frame_dropped"].append(bool(arr["fd"][j]))
        print(f"[EVAL {suite.name}] route {ri + 1}/{len(suite.eval_routes)} town={town_base(cond.town)} "
              f"weather={cond.weather} DS={rc * is_score:.1f} RC={rc:.1f} IS={is_score:.2f} "
              f"success={routes_out[-1]['success']}")
        if partial_path is not None:
            with open(partial_path + ".tmp", "w") as f:
                json.dump(pack(), f)
            os.replace(partial_path + ".tmp", partial_path)
    return pack()


def make_eval_context(art: SourceArtifacts, cfg: Config, device: torch.device) -> CaSTAgent:
    """Default CaST coverage and causal head used to bin states for every method (Fig. 3)."""
    spec = RunSpec(name="eval", suite="", method="cast", seed=0, budget=0, q=cfg.cast_q, k=cfg.knn_k)
    return make_run_agent(spec, art, cfg, device)



# ======================================================
# Statistics (Sec. 4.1: bootstrap intervals, paired Wilcoxon, Holm)
# ======================================================
def bootstrap_ci(vals: Sequence[float], n_boot: int = 2000, seed: int = 0) -> Tuple[float, float]:
    x = np.asarray(vals, np.float64)
    if x.size < 2:
        return (float(x.mean()) if x.size else float("nan"),) * 2
    rng = np.random.default_rng(seed)
    means = x[rng.integers(0, x.size, size=(n_boot, x.size))].mean(1)
    return float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def wilcoxon_p(x: Sequence[float], y: Sequence[float]) -> float:
    d = np.asarray(x, np.float64) - np.asarray(y, np.float64)
    d = d[d != 0.0]
    if d.size < 2:
        return 1.0
    try:
        from scipy.stats import wilcoxon
        return float(wilcoxon(d).pvalue)
    except Exception:
        pass
    ranks = np.argsort(np.argsort(np.abs(d))) + 1.0
    w_plus = ranks[d > 0].sum()
    n = d.size
    mu = n * (n + 1) / 4.0
    sd = math.sqrt(n * (n + 1) * (2 * n + 1) / 24.0)
    z = (w_plus - mu) / max(sd, 1e-9)
    return float(math.erfc(abs(z) / math.sqrt(2.0)))


def holm(pvals: Sequence[float]) -> List[float]:
    p = np.asarray(pvals, np.float64)
    order = np.argsort(p)
    m = len(p)
    adj = np.empty(m)
    running = 0.0
    for rank, i in enumerate(order):
        running = max(running, min(1.0, (m - rank) * p[i]))
        adj[i] = running
    return adj.tolist()


def auroc(scores: Sequence[float], labels: Sequence[bool]) -> float:
    s = np.asarray(scores, np.float64)
    y = np.asarray(labels, bool)
    npos, nneg = int(y.sum()), int((~y).sum())
    if npos == 0 or nneg == 0:
        return float("nan")
    order = np.argsort(s, kind="mergesort")
    ranks = np.empty(len(s))
    ranks[order] = np.arange(1, len(s) + 1)
    _, inv, cnt = np.unique(s, return_inverse=True, return_counts=True)
    sums = np.bincount(inv, weights=ranks)
    ranks = (sums / cnt)[inv]
    return float((ranks[y].sum() - npos * (npos + 1) / 2.0) / (npos * nneg))


def spearman(a: Sequence[float], b: Sequence[float]) -> float:
    a, b = np.asarray(a, np.float64), np.asarray(b, np.float64)
    if a.size < 3:
        return float("nan")
    ra, rb = np.argsort(np.argsort(a)), np.argsort(np.argsort(b))
    return float(np.corrcoef(ra, rb)[0, 1])


# ======================================================
# Experiment plan (one spec list per stage)
# ======================================================
ABLATIONS: List[Tuple[str, Dict[str, object]]] = [
    ("w/o coverage gate, Eq. (16)", {"gate": False}),
    ("w/o epistemic channel, Eq. (18)", {"epistemic": False}),
    ("Floor as penalty on sigma_hat^2 instead of Eq. (18)", {"floor_penalty": True}),
    ("w/o scene-level coverage, Eq. (15)", {"scene_term": False}),
    ("w/o cautious value, Eq. (19)", {"cautious": False}),
    ("Coverage on h_theta instead of h_S", {"coverage_space": "theta"}),
    ("w/o causal mask in coverage", {"causal_mask": False}),
    ("Reliance instead of effect, Eq. (23)", {"relevance": "reliance"}),
    ("Margin-only effect, Eq. (21)", {"relevance": "margin"}),
    ("Far-away null probes instead of Eq. (22)", {"null_mode": "far"}),
    ("w/o invariance, Eq. (25)", {"invariance": False}),
    ("Uniform sampling instead of Eq. (27)", {"sampler": "uniform"}),
    ("w/o validated coverage growth, Eq. (28)", {"validated_growth": False}),
    ("Full CaST", {}),
    ("Oracle causal entities", {"oracle_causal": True}),
    ("Oracle coverage", {"oracle_coverage": True}),
    ("Oracle causal entities and coverage", {"oracle_causal": True, "oracle_coverage": True}),
]
SENSITIVITY: List[Tuple[str, float]] = [("q", 0.01), ("q", 0.10), ("k", 5), ("k", 20), ("kappa", 0.5), ("kappa", 2.0),
                                         ("eps", 0.1), ("eps", 0.4), ("int_frac", 0.005)]
SHORTCUT_ROWS: List[Tuple[str, str, Dict[str, object]]] = [
    ("Uniform alignment [Borhan et al.]", "uniform", {}),
    ("Causality-aware entropy [ACE]", "ace", {}),
    ("Latent de-confounding [CausalVAD]", "deconf", {}),
    ("Attention as relevance", "cast", {"relevance": "attention"}),
    ("Reliance as relevance", "cast", {"relevance": "reliance"}),
    ("Margin-only effect", "cast", {"relevance": "margin"}),
    ("CaST (this work)", "cast", {}),
]
SAMPLERS = ("uniform", "uncertainty", "safety_critical", "cast")


def cast_spec(suite: str, seed: int, budget: int, cfg: Config, **kw) -> RunSpec:
    base = dict(name="cast", suite=suite, method="cast", seed=seed, budget=budget, sampler="cast", q=cfg.cast_q,
                k=cfg.knn_k, kappa=cfg.kappa_pess, eps=cfg.eps_ordinary, int_frac=cfg.intervention_fraction)
    base.update(kw)
    return RunSpec(**base)


def base_spec(method: str, suite: str, seed: int, budget: int, cfg: Config, **kw) -> RunSpec:
    if method == "cast":
        return cast_spec(suite, seed, budget, cfg, **kw)
    return RunSpec(name=method, suite=suite, method=method, seed=seed, budget=0 if method == "source_only" else budget,
                   q=cfg.cast_q, k=cfg.knn_k, kappa=cfg.kappa_pess, eps=cfg.eps_ordinary,
                   int_frac=cfg.intervention_fraction, **kw)


def plan(stage: str, P: Dict[str, object], cfg: Config) -> List[RunSpec]:
    B, s1, s3 = int(P["budget"]), int(P["seeds_table1"]), int(P["seeds"])
    out: List[RunSpec] = []
    if stage == "pilot":
        out.append(cast_spec("A", int(cfg.seed), B, cfg, eval_suites=("A",)))
    elif stage == "proxy":
        out.append(RunSpec(name="proxy", suite="C", method="finetune", seed=0, budget=B * int(P["proxy_mult"]),
                           q=cfg.cast_q, k=cfg.knn_k, tag="proxy"))
    elif stage == "expert":
        out.append(RunSpec(name="expert", suite="C", method="expert", seed=0, budget=0,
                           eval_suites=("source", "A", "L", "C")))
    elif stage == "table1":
        for suite in ("A", "L", "C"):
            # CaST first, then the baselines: the method's own runs finish (and can be checked) early.
            # The order changes no result; every run has its own seed and directory.
            for m in sorted(TRANSFER_METHODS, key=lambda x: x != "cast"):
                for seed in range(s1):
                    ev = [suite]
                    if suite == "C" and seed < s3 and m in ("uniform", "ace", "deconf", "cast"):
                        ev += ["barrel", "queue"]
                    if suite == "C" and seed < s3 and m == "cast":
                        ev += ["source"]
                    out.append(base_spec(m, suite, seed, B, cfg, eval_suites=tuple(ev)))
    elif stage == "table2":
        for _, m, kw in SHORTCUT_ROWS:
            for seed in range(s3):
                out.append(base_spec(m, "C", seed, B, cfg, eval_suites=("C", "barrel", "queue"), **kw))
    elif stage == "fig5":
        for smp in SAMPLERS:
            for b in P["budgets"]:
                for seed in range(s3):
                    out.append(cast_spec("C", seed, int(b), cfg, sampler=smp, eval_suites=("C",)))
    elif stage == "ablation":
        for _, kw in ABLATIONS:
            for seed in range(s3):
                out.append(cast_spec("C", seed, B, cfg, eval_suites=("C", "barrel"), **kw))
    elif stage == "sensitivity":
        for seed in range(s3):
            out.append(cast_spec("C", seed, B, cfg, eval_suites=("C", "source")))
            for prm, val in SENSITIVITY:
                out.append(cast_spec("C", seed, B, cfg, eval_suites=("C", "source"), **{prm: val}))
    return out


# ======================================================
# Run executor
# ======================================================
class Context:
    def __init__(self, args, cfg, device, suites, mgr, out_dir):
        self.args, self.cfg, self.device, self.suites, self.mgr, self.out = args, cfg, device, suites, mgr, out_dir
        self._art: Optional[SourceArtifacts] = None
        self._ectx: Optional[CaSTAgent] = None

    @property
    def art(self) -> SourceArtifacts:
        if self._art is None:
            self._art = SourceArtifacts(self.out, self.cfg, self.device)
        return self._art

    @property
    def ectx(self) -> CaSTAgent:
        if self._ectx is None:
            self._ectx = make_eval_context(self.art, self.cfg, self.device)
        return self._ectx

    def run_dir(self, spec: RunSpec) -> str:
        d = os.path.join(self.out, "runs", spec.run_id())
        os.makedirs(d, exist_ok=True)
        return d

    def proxy_W(self) -> Optional[nn.Module]:
        P = self.args.profile_dict
        spec = RunSpec(name="proxy", suite="C", method="finetune", seed=0, budget=int(P["budget"]) * int(P["proxy_mult"]),
                       q=self.cfg.cast_q, k=self.cfg.knn_k, tag="proxy")
        path = os.path.join(self.out, "runs", spec.run_id(), "agent.pt")
        if not os.path.exists(path):
            return None
        tmp = CaSTAgent(self.cfg, self.device)
        tmp.load(path)
        return copy_module(tmp.actor.enc.W)


def execute(spec: RunSpec, C: Context) -> None:
    import copy as _copy
    run_cfg = _copy.copy(C.cfg)
    run_cfg.knn_k = int(spec.k)
    run_dir = C.run_dir(spec)
    missing = [s for s in spec.eval_suites if not os.path.exists(os.path.join(run_dir, f"eval_{s}.json"))]
    needs_adapt = spec.method not in ("expert", "source_only") and not os.path.exists(os.path.join(run_dir, "ADAPT_DONE"))
    if not missing and not needs_adapt:
        return
    print(f"[RUN] {spec.run_id()}  eval={list(spec.eval_suites)}")
    agent: Optional[CaSTAgent] = None
    if spec.method == "source_only":
        agent = make_run_agent(spec, C.art, run_cfg, C.device)
    elif spec.method != "expert":
        agent = run_adaptation(spec, C.art, C.suites, C.mgr, C.args, run_cfg, C.device, run_dir)
    for es in missing:
        scripted = es in ("barrel", "queue")
        proxy = C.proxy_W() if (es == "C" and spec.is_cast()) else None
        partial = os.path.join(run_dir, f"eval_{es}.partial.json")
        res = evaluate_suite(agent, C.ectx, C.suites[es], C.mgr, run_cfg, scripted=scripted, proxy_W=proxy,
                             nulls=C.art.nulls if (agent is not None and spec.is_cast()) else None,
                             partial_path=partial)
        with open(os.path.join(run_dir, f"eval_{es}.json"), "w") as f:
            json.dump(res, f)
        if os.path.exists(partial):
            os.remove(partial)


def load_eval(C: Context, spec: RunSpec, suite: str) -> Optional[Dict[str, object]]:
    p = os.path.join(C.out, "runs", spec.run_id(), f"eval_{suite}.json")
    return json.load(open(p)) if os.path.exists(p) else None


def load_adapt(C: Context, spec: RunSpec) -> Optional[Dict[str, object]]:
    p = os.path.join(C.out, "runs", spec.run_id(), "adapt.json")
    return json.load(open(p)) if os.path.exists(p) else None


# ======================================================
# Aggregation into the CSV files used by the paper
# ======================================================
def write_csv(path: str, header: List[str], rows: List[List[object]]) -> None:
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(header)
        for r in rows:
            w.writerow([f"{x:.4f}" if isinstance(x, float) else x for x in r])
    print(f"[CSV] {path} ({len(rows)} rows)")


def pooled_routes(C: Context, specs: List[RunSpec], suite: str) -> List[Tuple[int, Dict[str, object]]]:
    out = []
    for sp in specs:
        ev = load_eval(C, sp, suite)
        if ev:
            out.extend((sp.seed, r) for r in ev["routes"])
    return out


def driving_row(routes: List[Tuple[int, Dict[str, object]]]) -> Dict[str, float]:
    if not routes:
        return {}
    ds = [r["DS"] for _, r in routes]
    km = sum(r["km"] for _, r in routes)
    lo, hi = bootstrap_ci(ds)
    return {"DS": float(np.mean(ds)), "DS_lo": lo, "DS_hi": hi, "DS_ci": (hi - lo) / 2.0,
            "SR": 100.0 * float(np.mean([r["success"] for _, r in routes])),
            "RC": float(np.mean([r["RC"] for _, r in routes])), "IS": float(np.mean([r["IS"] for _, r in routes])),
            "coll_km": sum(r["collisions"] for _, r in routes) / max(km, 1e-6),
            "red_km": sum(r["red_lights"] for _, r in routes) / max(km, 1e-6),
            "min_ttc": float(np.mean([min(r["min_ttc"], 10.0) for _, r in routes])),
            "cte": float(np.mean([r["cte"] for _, r in routes])), "n_routes": len(routes),
            "det_recall": sum(r["det_found"] for _, r in routes) / max(sum(r["det_true"] for _, r in routes), 1)}


def paired_p(a: List[Tuple[int, Dict]], b: List[Tuple[int, Dict]]) -> float:
    da = {(s, r["route"]): r["DS"] for s, r in a}
    db = {(s, r["route"]): r["DS"] for s, r in b}
    keys = sorted(set(da) & set(db))
    return wilcoxon_p([da[k] for k in keys], [db[k] for k in keys]) if keys else float("nan")


def ident_scores(rows: List[Dict[str, object]], predictor: str, tau: float) -> Tuple[float, float]:
    rows = [r for r in rows if r["role"] in ("cause", "distractor")]
    if not rows:
        return float("nan"), float("nan")
    if predictor == "c_hat":
        pred = [r["c_hat"] >= tau for r in rows]
    elif predictor == "reliance":
        pred = [np.isfinite(r["reliance"]) and r["reliance"] >= 0.05 for r in rows]
    else:
        pred = [r["attn"] >= 1.0 for r in rows]
    cause = [r["role"] == "cause" for r in rows]
    tp = sum(p and c for p, c in zip(pred, cause))
    npred, ncause = sum(pred), sum(cause)
    return (100.0 * tp / npred if npred else float("nan")), (100.0 * tp / ncause if ncause else float("nan"))


def aggregate(C: Context, P: Dict[str, object]) -> None:
    cfg = C.cfg
    res_dir = os.path.join(C.out, "results")
    os.makedirs(res_dir, exist_ok=True)
    s1, s3, B = int(P["seeds_table1"]), int(P["seeds"]), int(P["budget"])
    nulls = C.art.nulls if os.path.exists(os.path.join(C.out, "source", "DONE")) else None

    # ---------------- Table 1 ----------------
    rows, ref_rows = [], []
    for suite in ("A", "L", "C"):
        uni = pooled_routes(C, [base_spec("uniform", suite, s, B, cfg) for s in range(s1)], suite)
        block = []
        for m in TRANSFER_METHODS:
            routes = pooled_routes(C, [base_spec(m, suite, s, B, cfg) for s in range(s1)], suite)
            d = driving_row(routes)
            if not d:
                continue
            p = paired_p(routes, uni) if m != "uniform" else float("nan")
            block.append([suite, m, len({s for s, _ in routes}), d["DS"], d["DS_lo"], d["DS_hi"], d["DS_ci"], d["SR"],
                          d["coll_km"], d["RC"], d["IS"], d["min_ttc"], d["cte"], d["red_km"], p])
        rows.extend(block)
    # Holm correction across the three suites, per method
    for m in TRANSFER_METHODS:
        idx = [i for i, r in enumerate(rows) if r[1] == m and np.isfinite(r[-1])]
        if idx:
            adj = holm([rows[i][-1] for i in idx])
            for i, a in zip(idx, adj):
                rows[i] = rows[i] + [a, int(a < 0.05)]
    rows = [r if len(r) == 17 else r + [float("nan"), 0] for r in rows]
    write_csv(os.path.join(res_dir, "table1_transfer.csv"),
              ["suite", "method", "n_seeds", "DS", "DS_ci_lo", "DS_ci_hi", "DS_ci_halfwidth", "SR", "coll_per_km", "RC",
               "IS", "mean_min_ttc_s", "cte_m", "red_per_km", "p_wilcoxon_vs_uniform", "p_holm", "significant"], rows)
    expert = RunSpec(name="expert", suite="C", method="expert", seed=0, budget=0)
    for suite in ("source", "A", "L", "C"):
        d = driving_row(pooled_routes(C, [expert], suite))
        cast_routes = pooled_routes(C, [cast_spec(suite if suite != "source" else "C", s, B, cfg) for s in range(s1)],
                                    suite)
        rec = driving_row(cast_routes).get("det_recall", float("nan")) if cast_routes else float("nan")
        ref_rows.append([suite, d.get("DS", float("nan")), d.get("SR", float("nan")), d.get("coll_km", float("nan")),
                         100.0 * rec if np.isfinite(rec) else rec])
    write_csv(os.path.join(res_dir, "table1_reference.csv"),
              ["suite", "expert_DS", "expert_SR", "expert_coll_per_km", "detection_recall_pct"], ref_rows)

    # ---------------- Fig. 3 reliability, coverage shift, missed detections ----------------
    edges = list(cfg.coverage_bins)
    fig3 = []
    for m in ("uniform", "openset_entropy", "cast"):
        evs = [load_eval(C, base_spec(m, "C", s, B, cfg), "C") for s in range(s1)]
        evs = [e for e in evs if e]
        if not evs:
            continue
        for bi in range(len(edges) - 1):
            ub, fl, coll, dist = [], [], 0, 0.0
            min_ttc = []
            for e in evs:
                st = e["steps"]
                sel = [j for j, b_ in enumerate(st["bin"]) if b_ == bi]
                ub += [st["ubar"][j] for j in sel]
                fl += [st["fail"][j] for j in sel]
                coll += sum(1 for j in sel if st["coll"][j])
                dist += sum(st["dist_m"][j] for j in sel)
                per_route: Dict[int, float] = {}
                for j in sel:
                    per_route[st["route"][j]] = min(per_route.get(st["route"][j], 999.0), st["ttc"][j])
                min_ttc += [min(v, 10.0) for v in per_route.values()]
            gaps = [g for e in evs for b_, g in e.get("gap", []) if b_ == bi] if m == "cast" else []
            fig3.append([m, f"[{edges[bi]:.2f},{min(edges[bi + 1], 1.0):.2f})", len(ub), auroc(ub, fl),
                         coll / max(dist / 1000.0, 1e-6), float(np.mean(min_ttc)) if min_ttc else float("nan"),
                         float(np.mean(gaps)) if gaps else float("nan")])
    write_csv(os.path.join(res_dir, "fig3_reliability.csv"),
              ["method", "coverage_bin", "n_states", "failure_auroc", "coll_per_km", "mean_min_ttc_s",
               "embedding_gap_proxy"], fig3)
    shift = []
    for suite in ("source", "A", "L", "C"):
        vals = []
        for s in range(s1):
            sp = cast_spec("C" if suite == "source" else suite, s, B, cfg)
            e = load_eval(C, sp, suite)
            if e:
                vals += e["share_uncovered"]
        shift.append([suite, 100.0 * float(np.mean(vals)) if vals else float("nan"), len(vals)])
    write_csv(os.path.join(res_dir, "coverage_shift.csv"), ["suite", "pct_causal_with_rho_below_q", "n_interactions"], shift)
    md = []
    for s in range(s1):
        e = load_eval(C, cast_spec("C", s, B, cfg), "C")
        if not e:
            continue
        st = e["steps"]
        miss = [j for j, f in enumerate(st["missed"]) if f]
        md.append([s, driving_row([(s, r) for r in e["routes"]])["det_recall"] * 100.0, len(miss),
                   float(np.mean([st["ubar"][j] for j in miss])) if miss else float("nan"),
                   float(np.mean([st["ubar_noscene"][j] for j in miss])) if miss else float("nan")])
    write_csv(os.path.join(res_dir, "missed_detection.csv"),
              ["seed", "detection_recall_pct_C", "n_missed_causal_states", "ubar_with_scene_term",
               "ubar_without_scene_term"], md)

    # ---------------- Table 2 shortcut tests ----------------
    t2 = []
    for label, m, kw in SHORTCUT_ROWS:
        specs = [base_spec(m, "C", s, B, cfg, **kw) for s in range(s3)]
        agg = {"barrel_events": 0, "false_stops": 0, "red_events": 0, "missed_stops": 0, "queue_events": 0, "hesitations": 0}
        rel_b, rel_q, ident = [], [], []
        for sp in specs:
            for sname in ("barrel", "queue"):
                e = load_eval(C, sp, sname)
                if not e:
                    continue
                for r in e["routes"]:
                    for k in agg:
                        agg[k] += int(r.get(k, 0))
                (rel_b if sname == "barrel" else rel_q).extend(e["reliance_distractor"])
                ident.extend(e["ident"])
        predictor = "c_hat" if m == "cast" else "attention"
        prec, rec = ident_scores(ident, predictor, 1.0 - cfg.cast_q)
        t2.append([label, 100.0 * agg["false_stops"] / max(agg["barrel_events"], 1),
                   100.0 * agg["missed_stops"] / max(agg["red_events"], 1),
                   float(np.mean(rel_b)) if rel_b else float("nan"),
                   100.0 * agg["hesitations"] / max(agg["queue_events"], 1),
                   float(np.mean(rel_q)) if rel_q else float("nan"), prec, rec,
                   agg["barrel_events"], agg["red_events"], agg["queue_events"]])
    write_csv(os.path.join(res_dir, "table2_shortcut.csv"),
              ["method", "false_stop_pct", "missed_stop_pct", "reliance_barrels", "hesitation_pct", "reliance_queue",
               "precision_pct", "recall_pct", "n_barrel_events", "n_red_events", "n_queue_events"], t2)

    # ---------------- causal head accuracy, null calibration, cost ----------------
    ch = []
    for s in range(s3):
        sp = cast_spec("C", s, B, cfg)
        e = load_eval(C, sp, "C")
        a = load_adapt(C, sp)
        eb = load_eval(C, sp, "barrel")
        if not e or nulls is None:
            continue
        pairs = [(c, nulls.rank(ef, d)) for c, ef, d in e["acc_pairs"]]
        mae = float(np.mean([abs(c - y) for c, y in pairs])) if pairs else float("nan")
        rho_s = spearman([c for c, _ in pairs], [y for _, y in pairs]) if pairs else float("nan")
        prec_m = prec_f = float("nan")
        if eb:
            lab = [r for r in eb["ident"] if np.isfinite(r.get("e_full", float("nan")))]
            for far in (False, True):
                rows_ = [dict(r, c_hat=nulls.rank(r["e_full"], r["dist"], False, far)) for r in lab]
                pr, _ = ident_scores(rows_, "c_hat", 1.0 - cfg.cast_q)
                if far:
                    prec_f = pr
                else:
                    prec_m = pr
        overhead = 100.0 * a["label_s"] / max(a["wall_s"], 1e-6) if a else float("nan")
        ch.append([s, mae, rho_s, len(pairs), prec_m, prec_f, overhead, e.get("head_ms", float("nan"))])
    src_over = 100.0 * C.art.stats.get("label_s", 0.0) / max(C.art.stats.get("wall_s", 1.0), 1e-6) if nulls else float("nan")
    write_csv(os.path.join(res_dir, "causal_head.csv"),
              ["seed", "mae", "spearman", "n_labels", "precision_matched_nulls_pct", "precision_far_nulls_pct",
               "label_overhead_target_pct", "policy_and_monitor_ms_per_step"], ch)
    write_csv(os.path.join(res_dir, "cost.csv"), ["stage", "wall_s", "label_s", "label_overhead_pct"],
              [["source", C.art.stats.get("wall_s", float("nan")) if nulls else float("nan"),
                C.art.stats.get("label_s", float("nan")) if nulls else float("nan"), src_over]])

    # ---------------- Fig. 5 adaptation budget ----------------
    f5, curve = [], []
    for smp in SAMPLERS:
        for b in P["budgets"]:
            specs = [cast_spec("C", s, int(b), cfg, sampler=smp) for s in range(s3)]
            d = driving_row(pooled_routes(C, specs, "C"))
            if d:
                f5.append([smp, int(b), d["DS"], d["DS_lo"], d["DS_hi"], d["SR"], d["coll_km"]])
            if int(b) == max(int(x) for x in P["budgets"]):
                for sp in specs:
                    a = load_adapt(C, sp)
                    if a:
                        for row in a["curve"]:
                            curve.append([smp, sp.seed, row["episode"], row["mean_U"], row["mean_ubar"], row["rc"],
                                          row["collision"]])
    write_csv(os.path.join(res_dir, "fig5_budget.csv"), ["sampler", "budget", "DS", "DS_ci_lo", "DS_ci_hi", "SR",
                                                         "coll_per_km"], f5)
    write_csv(os.path.join(res_dir, "fig5_curve.csv"), ["sampler", "seed", "episode", "mean_U", "mean_ubar", "RC",
                                                        "collision"], curve)

    # ---------------- Table 5 ablation ----------------
    t5 = []
    for label, kw in ABLATIONS:
        specs = [cast_spec("C", s, B, cfg, **kw) for s in range(s3)]
        d = driving_row(pooled_routes(C, specs, "C"))
        if not d:
            continue
        ub, fl, fs, be, nll = [], [], 0, 0, []
        for sp in specs:
            e = load_eval(C, sp, "C")
            if e:
                st = e["steps"]
                ub += [st["ubar"][j] for j, b_ in enumerate(st["bin"]) if b_ == 0]
                fl += [st["fail"][j] for j, b_ in enumerate(st["bin"]) if b_ == 0]
            eb = load_eval(C, sp, "barrel")
            if eb:
                fs += sum(int(r.get("false_stops", 0)) for r in eb["routes"])
                be += sum(int(r.get("barrel_events", 0)) for r in eb["routes"])
            a = load_adapt(C, sp)
            if a:
                nll.append(a.get("motion_nll", float("nan")))
        t5.append([label, d["DS"], d["DS_ci"], d["coll_km"], auroc(ub, fl), 100.0 * fs / max(be, 1),
                   float(np.nanmean(nll)) if any(np.isfinite(x) for x in nll) else float("nan")])
    write_csv(os.path.join(res_dir, "table5_ablation.csv"),
              ["variant", "DS", "DS_ci_halfwidth", "coll_per_km", "auroc_lowest_bin", "false_stop_pct", "motion_nll"], t5)

    # ---------------- Fig. 6 sensitivity ----------------
    f6 = []
    defaults = {"q": cfg.cast_q, "k": cfg.knn_k, "kappa": cfg.kappa_pess, "eps": cfg.eps_ordinary,
                "int_frac": cfg.intervention_fraction}
    for prm in ("q", "k", "kappa", "eps", "int_frac"):
        vals = sorted({defaults[prm]} | {v for p_, v in SENSITIVITY if p_ == prm})
        for v in vals:
            kw = {} if v == defaults[prm] else {prm: v}
            specs = [cast_spec("C", s, B, cfg, **kw) for s in range(s3)]
            d = driving_row(pooled_routes(C, specs, "C"))
            ds_src = driving_row(pooled_routes(C, specs, "source"))
            if d:
                f6.append([prm, v, d["DS"], d["DS_lo"], d["DS_hi"], d["RC"], d["coll_km"], ds_src.get("DS", float("nan"))])
    write_csv(os.path.join(res_dir, "fig6_sensitivity.csv"),
              ["parameter", "value", "DS_C", "DS_ci_lo", "DS_ci_hi", "RC_C", "coll_per_km_C", "DS_source"], f6)
    with open(os.path.join(res_dir, "NOT_PRODUCED_BY_CAR_PY.txt"), "w") as f:
        f.write("Table 3 (Bench2Drive, 220 official routes) needs the Bench2Drive leaderboard harness, and Table 4\n"
                "(NAVSIM real-data check) needs the NAVSIM devkit with PDM-Closed. Neither runs inside car.py.\n")


# ======================================================
# Profiles and CLI
# ======================================================
PROFILES: Dict[str, Dict[str, object]] = {
    # Quick end-to-end check of the whole chain on a real CARLA server (about one hour).
    "smoke": {"perception_steps": 400, "perception_iters": 200, "source_steps": 3000, "teacher_steps": 1500, "handover_steps": 1000, "start_steps": 0, "update_after": 300, "budget": 2,
              "eval_routes": 4, "scripted_routes": 2, "seeds_table1": 1, "seeds": 1, "budgets": [1, 2],
              "proxy_mult": 2, "route_length": 150.0, "snapshot_rows": 3000, "source_replay": 3010},
    # Protocol of Sec. 4.1: 1.2x10^5 source steps: 4x10^4 teacher steps, 6x10^4 hand-over steps (the policy
    # drives, the teacher takes over when needed), and 2x10^4 steps of the policy alone.
    # Main adaptation budget is 10; proxy and Fig. 5 retain their explicitly different budgets.
    "paper": {"perception_steps": 20000, "perception_iters": 20000, "source_steps": 120000, "teacher_steps": 40000,
              "handover_steps": 60000, "start_steps": 0, "update_after": 2048, "budget": 10,
              "eval_routes": 20, "scripted_routes": 20, "seeds_table1": 5, "seeds": 3, "budgets": [5, 10, 20, 40],
              "proxy_mult": 10, "route_length": 500.0, "snapshot_rows": 50000, "source_replay": 120010},
    # Experimental extended budget. Does not change traffic, success criteria,
    # evaluation assistance, or the number of evaluation routes.
    # Shortest source schedule that still has all three phases: 40k teacher (with late-braking
    # demonstrations), 80k hand-over, 30k policy alone. The earlier 120k "paper" schedule reached the same
    # policy-alone success as the 300k "robust" one (81-83%); steering by the controller removes a whole
    # output the policy had to learn.
    "fast": {"perception_steps": 20000, "perception_iters": 20000, "source_steps": 150000, "teacher_steps": 40000,
             "handover_steps": 80000, "start_steps": 0, "update_after": 2048, "budget": 50,
             "eval_routes": 20, "scripted_routes": 20, "seeds_table1": 5, "seeds": 3, "budgets": [10, 20, 30, 50],
             "proxy_mult": 10, "route_length": 500.0, "snapshot_rows": 100000, "source_replay": 150010},
    "robust": {"perception_steps": 20000, "perception_iters": 20000, "source_steps": 300000, "teacher_steps": 60000,
               "handover_steps": 120000, "start_steps": 0, "update_after": 2048, "budget": 50,
               "eval_routes": 20, "scripted_routes": 20, "seeds_table1": 5, "seeds": 3, "budgets": [10, 20, 30, 50],
               "proxy_mult": 10, "route_length": 500.0, "snapshot_rows": 100000, "source_replay": 300010},
}
# car.py trains; car_eval.py evaluates (one town per CARLA run) and writes the paper CSV files.
ADAPT_STAGES = ("pilot", "proxy", "table1", "table2", "fig5", "ablation", "sensitivity")
ALL_STAGES = ("perception", "source") + ADAPT_STAGES
EVAL_STAGES = ("source_eval", "expert", "table1", "table2", "fig5", "ablation", "sensitivity")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="CaST training: Causal-Support Transfer for safety-aware E2E driving "
                                            "(CARLA 0.9.15). Evaluation: car_eval.py")
    p.add_argument("--host", default="localhost")
    p.add_argument("--port", type=int, default=2200)
    p.add_argument("--tm-port", type=int, default=8000)
    p.add_argument("--out-dir", default="./cast_results")
    p.add_argument("--profile", choices=sorted(PROFILES), default="paper")
    p.add_argument("--route-length", type=float, default=None, metavar="METRES",
                   help="road arc-length target for training and evaluation (e.g. 200); default: profile value. "
                        "Scales route tolerance and base episode limit, not total source/perception steps. "
                        "On resume, affects future routes only; use a new --eval-tag for evaluation.")
    p.add_argument("--stages", default="all", help="comma list of " + ",".join(ALL_STAGES) + " or 'all'")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--source-towns", default="Town10HD_Opt",
                   help="comma-separated policy source towns, visited in rotation; perception is collected in the first")
    p.add_argument("--layout-towns", default="Town01,Town02,Town03,Town04,Town05",
                   help="unseen towns of the layout and combined suites (add Town12,Town13 if installed)")
    p.add_argument("--barrel-source-frac", type=float, default=0.3,
                   help="share of source episodes in which barrels precede every forced stop (Sec. 4.3)")
    p.add_argument("--updates-per-step", type=int, default=1)
    p.add_argument("--source-steps", type=int, default=0, help="override the profile")
    p.add_argument("--teacher-steps", type=int, default=-1, help="override the profile")
    p.add_argument("--bc-weight", type=float, default=2.0, help="behaviour-cloning weight during the teacher phase")
    p.add_argument("--bc-final", type=float, default=2.0,
                   help="group-balanced imitation weight at the end of the source stage")
    p.add_argument("--demo-frac", type=float, default=0.5,
                   help="share of teacher transitions in each source-stage batch at the start of the hand-over")
    p.add_argument("--demo-frac-final", type=float, default=0.2,
                   help="share of teacher transitions in each batch at the end of the source stage")
    p.add_argument("--blend-start", type=float, default=0.9,
                   help="teacher weight of the blended action right after the teacher phase")
    p.add_argument("--handover-steps", type=int, default=-1,
                   help="steps after the teacher phase over which the teacher weight of the blended action falls "
                        "from --blend-start to 0 (default: the profile value)")
    p.add_argument("--budget", type=int, default=0, help="main adaptation episodes (paper: 10, robust: 50)")
    p.add_argument("--adapt-town-schedule", choices=("profile", "balanced", "sampler"), default="profile",
                   help="robust defaults to balanced town blocks; paper retains its sampler")
    p.add_argument("--adapt-min-weather-episodes", type=int, default=None,
                   help="initial coverage per weather/town before unrestricted sampling (robust: 2, others: 0)")
    p.add_argument("--red-light-penalty", type=float, default=None,
                   help="nonpositive reward per verified crossing (robust: -5, others: 0)")
    p.add_argument("--target-bc-weight", type=float, default=2.0,
                   help="source expert imitation during adaptation; 0 disables this experimental anchor")
    p.add_argument("--align-metric", choices=("action", "kl"), default="action",
                   help="distance for CaST alignment/invariance: squared squashed-action gap (default) or legacy KL")
    p.add_argument("--critic-target", choices=("mean", "lcb", "min2"), default="lcb",
                   help="target ensemble reduction; mean experiment, lcb mean-minus-disagreement, min2 legacy ablation")
    p.add_argument("--critic-disagreement-penalty", type=float, default=0.3, help="nonnegative LCB multiplier")
    p.add_argument("--lateral-control", choices=("controller", "policy"), default="controller",
                   help="controller: route-tracking steering, the policy sets the speed; policy: learned steering")
    p.add_argument("--light-source", choices=("map_camera", "v2i", "camera"), default="map_camera",
                   help="traffic-light token: map association + camera colour (default), map + broadcast state "
                        "(v2i, report separately), or camera only (earlier behaviour)")
    p.add_argument("--light-token-version", type=int, choices=(1, 2), default=2,
                   help="map_camera light token: 2 (light-head association, 2 s colour memory, unknown colour flagged; "
                        "default) or 1 (rgbd12 checkpoints: unknown colour encoded like yellow)")
    p.add_argument("--use-safety-shield", action="store_true",
                   help="car_eval.py: rule-based safety shield with privileged lights and actors (red-light brake, "
                        "front-vehicle brake, TTC caution, stuck release), as in the CVPR 2026 protocol. Every "
                        "activation counts as an intervention; results go to <eval-tag>_shield. Not for training.")
    p.add_argument("--eval-light-override", action="store_true",
                   help="car_eval.py diagnostic: allow --light-source to differ from the checkpoint's (results go to "
                        "<eval-tag>_light-<source>; not a paper result)")
    p.add_argument("--late-brake-prob", type=float, default=0.5,
                   help="teacher phase: probability that a braking onset is demonstrated 0.4-1.4 s late (0: off)")
    p.add_argument("--source-rl-max", type=float, default=0.25,
                   help="maximum weight of the actor's RL term in the source stage (1.0: earlier behaviour)")
    p.add_argument("--hazard-aug-prob", type=float, default=0.25,
                   help="share of free-road imitation states given an inserted stopped or slow car (or standing "
                        "pedestrian) ahead, labelled with the teacher's IDM braking (0: rgbd14 and earlier)")
    p.add_argument("--hazard-aug-version", type=int, choices=(1, 2), default=2,
                   help="hazard insertion: 2 (default) also in traffic, closer than the nearest obstacle, half of the "
                        "insertions 4.8-12 m centre distance; 1 (rgbd14d) free road only, 6-30 m")
    p.add_argument("--target-rl-max", type=float, default=0.25,
                   help="weight of the actor's RL term in the adaptation stage (1.0: rgbd14 and earlier)")
    p.add_argument("--takeover-loss-weight", type=float, default=1.0,
                   help="opt-in synthetic terminal critic penalty for newly rejected source handover actions")
    p.add_argument("--takeover-penalty", type=float, default=-5.0, help="synthetic terminal reward, strictly negative")
    p.add_argument("--target-state-bc-weight", type=float, default=0.0,
                   help="opt-in privileged expert imitation on uncertain target states; separate assisted setting")
    p.add_argument("--target-state-bc-threshold", type=float, default=0.35,
                   help="collection-time ubar threshold in [0,1] for target expert labels")
    p.add_argument("--value-audit-stride", type=int, default=20,
                   help="executed-action Q sampling interval for per-episode return audits; 0 disables")
    p.add_argument("--learning-smoke-test", action="store_true",
                   help="run CPU tensor/backpropagation checks and exit; no CARLA server or training output")
    p.add_argument("--target-actor-lr", type=float, default=3e-5)
    p.add_argument("--target-explore-std", type=float, default=0.05,
                   help="maximum pre-tanh policy std for both acting and SAC updates; must match on resume/evaluation")
    p.add_argument("--uncertainty-fusion", choices=("mean", "sum"), default="mean",
                   help="mean avoids additive saturation; sum restores the original fusion")
    p.add_argument("--eval-routes", type=int, default=0, help="override the profile")
    p.add_argument("--perception", choices=("camera", "rgbd", "gt"), default="camera",
                   help="camera: six cameras and the frozen BEV detector; rgbd: six RGB and six depth cameras, "
                        "a frozen RGB-D segmentation network, 3D clustering and tracking; gt: noisy ground-truth "
                        "tokens with distance/fog-dependent dropout, the observation model of the CVPR 2026 code "
                        "(no perception stage). Use a separate --out-dir for each mode.")
    p.add_argument("--perception-buffer", type=int, default=4000,
                   help="frame pairs kept on disk for detector training (about 1.2 MB each)")
    p.add_argument("--perception-stride", type=int, default=5,
                   help="store one frame pair every this many perception steps")
    p.add_argument("--perception-val", type=int, default=300, help="held-out source frames for perception_eval.csv")
    p.add_argument("--town-block", type=int, default=5, help="target-stage episodes per town before the map changes (B=20: four blocks)")
    p.add_argument("--ckpt-every", type=int, default=5, help="episodes between checkpoints")
    p.add_argument("--no-rendering", action="store_true", help="not allowed: the cameras need rendering")
    p.add_argument("--no-follow-view", action="store_true", help="do not move the spectator camera with the ego car")
    p.add_argument("--carla-cmd", default="",
                   help="command that starts the CARLA server, e.g. '~/CARLA_0.9.15/CarlaUE4.sh -carla-rpc-port=2200'; "
                        "if given, the supervisor restarts the server after a crash")
    p.add_argument("--carla-startup-attempts", type=int, default=3,
                   help="bounded server-launch attempts per recovery/map switch; preserves checkpoints")
    p.add_argument("--carla-startup-timeout", type=float, default=600.0,
                   help="RPC readiness deadline in seconds per server launch")
    p.add_argument("--repair-town03-tm-cache", action="store_true",
                   help="back up loose Town03_Opt TM binaries before restarting; CARLA 0.9.15 issue #7210")
    p.add_argument("--carla-root", default="",
                   help="local CARLA installation for cache repair; otherwise inferred from --carla-cmd")
    p.add_argument("--max-startup-crashes", type=int, default=2,
                   help="stop after this many native Traffic Manager startup failures (default: 2)")
    p.add_argument("--max-restarts", type=int, default=100)
    p.add_argument("--allow-map-load", action="store_true",
                   help="without --carla-cmd, try client.load_world() before asking for a CARLA restart in the new town")
    p.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    p.add_argument("--cpu", action="store_true")
    p.add_argument("--debug", action="store_true")
    return p.parse_args()


def make_cfg(args: argparse.Namespace, P: Dict[str, object]) -> Config:
    cfg = Config()
    cfg.tm_port = int(args.tm_port)
    if not 1 <= cfg.tm_port <= 65535 or cfg.tm_port in (int(args.port), int(args.port) + 1, int(args.port) + 2):
        raise ValueError("--tm-port must be a valid port distinct from the CARLA RPC/streaming ports")
    cfg.seed = int(args.seed)
    cfg.debug_mode = bool(args.debug)
    cfg.mode_name = "train"
    schedule = getattr(args, "adapt_town_schedule", "profile")
    cfg.balanced_adaptation = schedule == "balanced" or (schedule == "profile" and args.profile in ("robust", "fast"))
    weather_floor = getattr(args, "adapt_min_weather_episodes", None)
    cfg.adapt_min_weather_episodes = int((2 if args.profile in ("robust", "fast") else 0) if weather_floor is None else weather_floor)
    if cfg.adapt_min_weather_episodes < 0:
        raise ValueError("--adapt-min-weather-episodes must be nonnegative")
    red_penalty = getattr(args, "red_light_penalty", None)
    cfg.r_red_light = float((-5.0 if args.profile in ("robust", "fast") else 0.0) if red_penalty is None else red_penalty)
    if not math.isfinite(cfg.r_red_light) or cfg.r_red_light > 0.0:
        raise ValueError("--red-light-penalty must be finite and nonpositive")
    cfg.target_bc_weight = float(getattr(args, "target_bc_weight", cfg.target_bc_weight))
    cfg.critic_target = str(getattr(args, "critic_target", cfg.critic_target))
    cfg.align_metric = str(getattr(args, "align_metric", cfg.align_metric))
    cfg.lateral_control = str(getattr(args, "lateral_control", cfg.lateral_control))
    cfg.light_source = str(getattr(args, "light_source", cfg.light_source))
    cfg.light_token_version = int(getattr(args, "light_token_version", cfg.light_token_version))
    cfg.eval_light_override = bool(getattr(args, "eval_light_override", False))
    cfg.use_safety_shield = bool(getattr(args, "use_safety_shield", False))
    cfg.hazard_aug_version = int(getattr(args, "hazard_aug_version", cfg.hazard_aug_version))
    for key in ("critic_disagreement_penalty", "source_rl_max", "target_rl_max", "hazard_aug_prob", "late_brake_prob", "takeover_loss_weight", "takeover_penalty",
                "target_state_bc_weight", "target_state_bc_threshold"):
        setattr(cfg, key, float(getattr(args, key, getattr(cfg, key))))
    cfg.value_audit_stride = int(getattr(args, "value_audit_stride", cfg.value_audit_stride))
    finite_options = [v for v in value_training_options(cfg).values() if isinstance(v, (int, float))]
    if (cfg.critic_target not in ("mean", "lcb", "min2") or not all(math.isfinite(v) for v in finite_options)
            or min(cfg.critic_disagreement_penalty, cfg.takeover_loss_weight, cfg.target_state_bc_weight) < 0
            or not 0.0 <= cfg.source_rl_max <= 1.0 or not 0.0 <= cfg.target_rl_max <= 1.0
            or not 0.0 <= cfg.hazard_aug_prob <= 1.0 or not 0.0 <= cfg.late_brake_prob <= 1.0
            or cfg.takeover_penalty >= 0 or not 0 <= cfg.target_state_bc_threshold <= 1
            or cfg.value_audit_stride < 0):
        raise ValueError("Invalid value-learning options: weights finite/nonnegative, penalty negative, "
                         "target-state BC threshold in [0,1], audit stride nonnegative")
    cfg.target_actor_lr = float(getattr(args, "target_actor_lr", cfg.target_actor_lr))
    cfg.target_explore_std = float(getattr(args, "target_explore_std", cfg.target_explore_std))
    cfg.uncertainty_fusion = str(getattr(args, "uncertainty_fusion", cfg.uncertainty_fusion))
    if (not all(math.isfinite(x) for x in (cfg.target_bc_weight, cfg.target_actor_lr, cfg.target_explore_std))
            or cfg.target_bc_weight < 0 or cfg.target_actor_lr <= 0 or not 1e-4 <= cfg.target_explore_std <= 1.0):
        raise ValueError("target BC must be nonnegative, actor LR positive, and policy std cap in [0.0001, 1.0]; all finite")
    if int(getattr(args, "town_block", 5)) <= 0 or int(getattr(args, "ckpt_every", 5)) <= 0:
        raise ValueError("--town-block and --ckpt-every must be positive")
    cfg.no_rendering_mode = bool(args.no_rendering)
    cfg.follow_ego_view = not bool(args.no_follow_view)
    cfg.out_dir = os.path.abspath(args.out_dir)
    cfg.perception_mode = str(getattr(args, "perception", "camera"))
    if cfg.perception_mode == "gt":
        cfg.use_cameras = False
    if cfg.perception_mode == "rgbd":
        cfg.use_depth = True
    cfg.replay_size = max(int(P["source_replay"]), 200000)
    route_override = getattr(args, "route_length", None)
    rtl = float(P["route_length"] if route_override is None else route_override)
    if not math.isfinite(rtl) or rtl < 50.0:
        raise ValueError("--route-length must be a finite road arc-length of at least 50 metres")
    if abs(rtl - 500.0) > 1e-6:
        scale = rtl / 500.0
        cfg.route_target_length_m = rtl
        cfg.route_target_tolerance_m = max(5.0, 10.0 * scale)
        cfg.route_soft_min_length_m = rtl - cfg.route_target_tolerance_m
        cfg.route_soft_max_length_m = rtl + cfg.route_target_tolerance_m
        cfg.min_route_length_m = max(50.0, rtl - 2.0 * cfg.route_target_tolerance_m)
        cfg.candidate_goal_min_dist_m = min(cfg.candidate_goal_min_dist_m, rtl * 0.2)
        cfg.max_episode_steps = max(1500, int(6000 * scale))
    if route_override is not None:
        print(f"[ROUTE] target={cfg.route_target_length_m:g}+/-{cfg.route_target_tolerance_m:g} m "
              f"(road arc-length), base limit={cfg.max_episode_steps} steps "
              f"({cfg.max_episode_steps * cfg.dt:g} simulated seconds); traffic/near-goal extensions still apply")
    if args.profile == "smoke":
        cfg.null_total, cfg.bank_size, cfg.n_reference = 500, 5000, 1000
        cfg.intervention_fraction = 0.05
        cfg.batch_size = 128
    return cfg


def write_source_eval(C: "Context", P: Dict[str, object]) -> None:
    """Driving quality of the source policy on 20 routes per suite (source, A, L, C), seed 0."""
    rows = []
    for suite in ("source", "A", "L", "C"):
        sp = base_spec("source_only", "source" if suite == "source" else suite, 0, 0, C.cfg)
        d = driving_row(pooled_routes(C, [sp], suite))
        if d:
            towns = sorted({r["conditions"].split("|")[0] for _, r in pooled_routes(C, [sp], suite)})
            rows.append([suite, d["n_routes"], " ".join(town_base(t) for t in towns), d["DS"], d["DS_lo"], d["DS_hi"],
                         d["SR"], d["RC"], d["IS"], d["coll_km"], d["red_km"], d["min_ttc"], d["cte"]])
    os.makedirs(os.path.join(C.out, "results"), exist_ok=True)
    write_csv(os.path.join(C.out, "results", "source_policy_eval.csv"),
              ["suite", "n_routes", "towns", "DS", "DS_ci_lo", "DS_ci_hi", "SR", "RC", "IS", "coll_per_km", "red_per_km",
               "mean_min_ttc_s", "cte_m"], rows)


def setup_run(args: argparse.Namespace, avail: Optional[Sequence[str]] = None) -> Tuple[Dict[str, object], Config, torch.device, str,
                                                                    Dict[str, Suite], "EnvManager", "Context"]:
    """Profile, configuration, suites, environment manager, and run context (shared by car.py and car_eval.py)."""
    P = dict(PROFILES[args.profile])
    for k_cli, k_p in (("source_steps", "source_steps"), ("budget", "budget"), ("eval_routes", "eval_routes")):
        if int(getattr(args, k_cli)) > 0:
            P[k_p] = int(getattr(args, k_cli))
    if int(args.teacher_steps) >= 0:
        P["teacher_steps"] = int(args.teacher_steps)
    if int(args.handover_steps) >= 0:
        P["handover_steps"] = int(args.handover_steps)
    args.handover_steps = int(P.get("handover_steps", 10000))
    P["teacher_steps"] = min(int(P["teacher_steps"]), int(P["source_steps"]))
    P["source_replay"] = max(int(P["source_replay"]), int(P["source_steps"]) + 10)
    args.profile_dict = P
    args.source_steps, args.start_steps, args.update_after = P["source_steps"], P["start_steps"], P["update_after"]
    args.teacher_steps = P["teacher_steps"]
    args.perception_steps, args.perception_iters = P["perception_steps"], P["perception_iters"]
    args.eval_routes, args.scripted_routes = P["eval_routes"], P["scripted_routes"]
    args.snapshot_rows, args.source_replay = P["snapshot_rows"], P["source_replay"]
    cfg = make_cfg(args, P)
    P["route_length"] = float(cfg.route_target_length_m)
    if cfg.no_rendering_mode and cfg.use_cameras:
        raise SystemExit("--no-rendering disables the cameras of the perception front end; run without it")
    set_global_seed(int(args.seed))
    device = torch.device("cpu" if args.cpu or not torch.cuda.is_available() else "cuda")
    out = cfg.out_dir
    os.makedirs(out, exist_ok=True)
    os.makedirs(os.path.join(out, "results"), exist_ok=True)
    log: List[List[str]] = []
    from_server = avail is None
    avail = available_towns(args.host, args.port) if from_server else [town_base(t) for t in avail]
    suites = build_suites(args, cfg, avail, log)
    if from_server:
        write_csv(os.path.join(out, "results", "suites_used.csv"), ["suite", "town", "status"], log)
    mgr = EnvManager(args, cfg)
    mgr.device = device
    C = Context(args, cfg, device, suites, mgr, out)
    return P, cfg, device, out, suites, mgr, C


def train_spec(spec: RunSpec, C: "Context") -> None:
    """Training only: target-stage adaptation of one run (car_eval.py evaluates it later)."""
    if spec.method in ("expert", "source_only"):
        return
    run_dir = C.run_dir(spec)
    if os.path.exists(os.path.join(run_dir, "ADAPT_DONE")):
        return
    import copy as _copy
    run_cfg = _copy.copy(C.cfg)
    run_cfg.knn_k = int(spec.k)
    print(f"[TRAIN] {spec.run_id()}")
    run_adaptation(spec, C.art, C.suites, C.mgr, C.args, run_cfg, C.device, run_dir)


def _pending_town(run_dir: str) -> Optional[str]:
    """Town a paused adaptation run needs next (None if unknown: it is chosen when the run resumes)."""
    try:
        with open(os.path.join(run_dir, "ckpt", "state.json")) as state_file:
            st = json.load(state_file)
        pending = st.get("pending_condition")
        if pending is not None:
            return str(pending["town"])
        return str(st["town"]) if st.get("town") and int(st.get("block_left", 0)) > 0 else None
    except BaseException:
        return None


def adapt_all(args: argparse.Namespace, P: Dict[str, object], cfg: Config, C: "Context", stages: Sequence[str]) -> None:
    """Every target-stage adaptation run, scheduled by town so that CARLA changes map as rarely as possible.

    All runs that can continue in the town CARLA is running now are advanced until each one needs another
    town (its next block of --town-block episodes, chosen by Eq. 27) or is finished; only then is the map
    changed, to the town most runs are waiting for. Runs are independent, so this changes no result."""
    order = {"pilot": -1, "proxy": 0, "table1": 1, "ablation": 2, "table2": 3, "fig5": 4, "sensitivity": 5}
    specs: List[RunSpec] = []
    seen = set()
    for st in sorted(stages, key=lambda x: order.get(x, 9)):
        for sp in plan(st, P, cfg):
            if sp.method in ("expert", "source_only") or sp.run_id() in seen:
                continue
            seen.add(sp.run_id())
            specs.append(sp)
    while True:
        pending = [sp for sp in specs if not os.path.exists(os.path.join(C.out, "runs", sp.run_id(), "ADAPT_DONE"))]
        print(f"[PLAN] adaptation runs: {len(specs) - len(pending)}/{len(specs)} finished")
        if not pending:
            return
        current = town_base(current_server_town(args.host, args.port))
        need: Dict[str, int] = {}
        for sp in pending:
            t = _pending_town(os.path.join(C.out, "runs", sp.run_id()))
            if t is not None and town_base(t) != current:
                need[t] = need.get(t, 0) + 1
                continue
            try:
                train_spec(sp, C)
            except TownSwitch as ts:
                need[ts.town] = need.get(ts.town, 0) + 1
                C.mgr.close()
        if not need:
            continue
        nxt = max(need, key=lambda k: need[k])
        print(f"[PLAN] nothing more to do in {current}; runs waiting per town: "
              + ", ".join(f"{town_base(k)}={v}" for k, v in sorted(need.items())))
        raise TownSwitch(nxt)


def worker_main(args: argparse.Namespace) -> None:
    """Training only (perception, source stage, every target-stage adaptation). Evaluation is car_eval.py."""
    import faulthandler
    # A segmentation fault (exit code -11) cannot be caught in Python; this prints the Python stack of
    # every thread at the moment of the crash, which names the failing call (CARLA client, CUDA, ...).
    faulthandler.enable(all_threads=True)
    validate_training_directory(os.path.abspath(args.out_dir))
    P, cfg, device, out, suites, mgr, C = setup_run(args)
    validate_training_directory(out, cfg)
    stages = ALL_STAGES if args.stages == "all" else tuple(s.strip() for s in args.stages.split(","))
    bad = [st for st in stages if st not in ALL_STAGES]
    if bad:
        raise SystemExit(f"unknown training stage(s) {bad}; evaluation stages belong to car_eval.py")
    print(f"[CaST] TRAINING profile={args.profile} stages={stages} device={device} out={out}")
    print(f"[CaST] source steps={P['source_steps']} (teacher {P['teacher_steps']}, blend {args.handover_steps}), "
          f"target budget={P['budget']} episodes, follow view={cfg.follow_ego_view}")
    print(f"[LEARNING] {LEARNING_REVISION}: fusion={cfg.uncertainty_fusion}, "
          f"source BC={args.bc_weight}->{args.bc_final}, target source-BC={cfg.target_bc_weight}, "
          f"target actor lr={cfg.target_actor_lr:g}, shared policy std cap={cfg.target_explore_std:g}")
    print(f"[TRAINING PROTOCOL] {TRAINING_REVISION}: balanced_towns={cfg.balanced_adaptation}, "
          f"minimum visits per weather/town={cfg.adapt_min_weather_episodes}, red crossing reward={cfg.r_red_light:g}")
    print(f"[VALUE LEARNING] {json.dumps(value_training_options(cfg), sort_keys=True)}")
    if cfg.target_state_bc_weight > 0.0:
        print("[PROTOCOL] Expert-assisted target adaptation enabled; report separately from target RL without expert labels.")
    with open(os.path.join(out, "run_config.json"), "w") as f:
        json.dump({"training_revision": TRAINING_REVISION, "value_training_options": value_training_options(cfg),
                   "profile": P, "cfg": {k: v for k, v in cfg.__dict__.items()},
                   "args": {k: str(v) for k, v in vars(args).items()}}, f, indent=2, default=str)
    try:
        if "perception" in stages and cfg.perception_mode in ("camera", "rgbd"):
            run_perception_stage(args, cfg, device, suites, mgr, out)
        mgr.detector = load_detector(out, cfg, device) if cfg.perception_mode in ("camera", "rgbd") else None
        if cfg.perception_mode in ("camera", "rgbd") and mgr.detector is None and any(st != "perception" for st in stages):
            raise SystemExit("the frozen detector is missing: run the perception stage first")
        if "source" in stages:
            run_source_stage(args, cfg, device, suites, mgr, out)
        adapt_all(args, P, cfg, C, [st for st in ADAPT_STAGES if st in stages])
        print("[CaST] training finished; evaluate with car_eval.py")
    except TownSwitch as ts:
        mgr.close()
        with open(os.path.join(out, "town_request.json"), "w") as f:
            json.dump({"town": ts.town}, f)
        print(f"[CaST] next episode needs {ts.town}; progress saved, handing the map load to the supervisor")
        sys.exit(TOWN_SWITCH_EXIT)
    except SimulatorDown as e:
        if mgr.env is not None:
            mgr.env._server_down = True
        print(f"[CaST] CARLA stopped responding ({e}); exiting so the supervisor can recover")
        sys.exit(SIM_DOWN_EXIT)
    finally:
        mgr.close()


def supervisor_main(args: argparse.Namespace) -> None:
    """Run the experiment in a worker process and restart it after a crash.

    Each town runs in its own CARLA server run: when the worker needs another map it saves its
    progress and exits with code 75, and the supervisor restarts CARLA directly in that map
    (--carla-cmd) instead of calling client.load_world() on a long-running server. A CARLA
    TimeoutException inside a C++ client thread terminates the worker and cannot be caught in
    Python; every stage checkpoints its progress, so after a crash the supervisor restarts CARLA
    fresh in the same town and the worker continues where it stopped."""
    import subprocess
    out_dir = os.path.abspath(args.out_dir)
    if int(args.max_startup_crashes) < 1:
        raise SystemExit("--max-startup-crashes must be positive")
    startup_path = os.path.join(out_dir, "runtime_startup.json")

    def last_town() -> Optional[str]:
        for name in ("town_request.json", "active_town.json"):
            try:
                with open(os.path.join(out_dir, name)) as f:
                    return str(json.load(f)["town"])
            except Exception:
                continue
        return None

    if args.repair_town03_tm_cache:
        try:
            repair_town03_tm_cache(args)
        except (OSError, ValueError, RuntimeError) as exc:
            raise SystemExit(f"[TM REPAIR] {exc}") from exc
        restart_carla_server(args, last_town() or str(args.source_towns).split(',')[0])
    elif args.carla_cmd:
        try:
            wait_for_carla_ready(args.host, args.port, timeout_s=10.0)
        except Exception:
            restart_carla_server(args, last_town())
    cmd = [sys.executable, os.path.abspath(__file__)] + sys.argv[1:] + ["--worker"]
    restarts = 0
    startup_crashes = 0
    while True:
        try:
            # without --carla-cmd you restart CARLA yourself, so wait longer for it
            wait_for_carla_ready(args.host, args.port, timeout_s=600.0 if args.carla_cmd else 3600.0)
        except Exception as e:
            print(f"[SUPERVISOR] CARLA is not answering: {e}")
            if not args.carla_cmd:
                raise SystemExit("start CARLA (or pass --carla-cmd) and run the same command again")
            restarts += 1
            if restarts >= int(args.max_restarts):
                raise SystemExit("too many restarts (CARLA did not become ready)")
            restart_carla_server(args, last_town())
            continue
        # A prior worker's TM marker must not classify this worker's failure.
        try:
            os.remove(startup_path)
        except FileNotFoundError:
            pass
        try:
            rc = subprocess.call(cmd)
        except KeyboardInterrupt:
            print("[SUPERVISOR] interrupted; saved checkpoints retained")
            return
        if rc == 0:
            print("[SUPERVISOR] all stages finished")
            return
        if rc in (-2, -15, 130, 143):
            print("[SUPERVISOR] worker interrupted; saved checkpoints retained")
            return
        if rc == TOWN_SWITCH_EXIT:
            with open(os.path.join(out_dir, "town_request.json")) as f:
                town = str(json.load(f)["town"])
            prepare_town(args, town)
            try:
                with open(os.path.join(out_dir, "active_town.json"), "w") as f:
                    json.dump({"town": town}, f)
            except Exception:
                pass
            continue
        if rc == SIM_DOWN_EXIT:
            restarts += 1
            if restarts >= int(args.max_restarts):
                raise SystemExit("too many restarts (CARLA stopped responding repeatedly)")
            town = last_town() or "Town10HD_Opt"
            if args.carla_cmd:
                print("[SUPERVISOR] CARLA stopped responding; starting a fresh server")
                restart_carla_server(args, town)
                continue
            hung = set(_carla_server_pids(int(args.port)))
            print(f"[SUPERVISOR] CARLA stopped responding (a call waited 300 s). Close it and start it again in "
                  f"{town}, for example\n    ~/carla/CarlaUE4.sh {map_path(town)} -quality-level=Low -nosound "
                  f"-carla-rpc-port={int(args.port)}\n  The run continues from its last checkpoint once the new "
                  f"server answers (or pass --carla-cmd to let the supervisor restart CARLA itself).")
            while hung and set(_carla_server_pids(int(args.port))) & hung:
                time.sleep(10.0)          # the frozen server still runs: wait until it is gone
            continue
        # Python exceptions/configuration errors need diagnosis, not 100 restarts.
        if rc > 0:
            raise SystemExit(f"[SUPERVISOR] worker exited {rc}; fix the traceback and rerun the same command. "
                             "Existing checkpoints are retained.")
        try:
            with open(startup_path) as f:
                startup = json.load(f)
        except (OSError, ValueError):
            startup = {}
        if startup.get("phase") == "traffic_manager_start":
            startup_crashes += 1
            print(f"[SUPERVISOR] native TM startup failure {startup_crashes}/{args.max_startup_crashes}: "
                  f"map={startup.get('town')} tm_port={startup.get('tm_port')}", flush=True)
            if startup_crashes >= int(args.max_startup_crashes):
                raise SystemExit("Repeated native Traffic Manager startup failure. Training checkpoint retained. "
                                 f"Inspect {startup_path} and logs/carla_server.log. For CARLA 0.9.15 Town03_Opt, "
                                 "use --repair-town03-tm-cache --carla-root /path/to/carla with --carla-cmd. "
                                 "See https://github.com/carla-simulator/carla/issues/7210")
        else:
            startup_crashes = 0
        restarts += 1
        print(f"[SUPERVISOR] worker exited with code {rc}; restart {restarts}/{args.max_restarts} "
              f"(progress is resumed from the last checkpoint)")
        if restarts >= int(args.max_restarts):
            raise SystemExit("too many restarts")
        time.sleep(5.0)
        town = last_town()
        if args.carla_cmd:
            # A crashed client leaves the server in synchronous mode with orphan actors and a dead
            # Traffic Manager registered; a fresh server in the same town is the reliable recovery.
            restart_carla_server(args, town)
        elif town is not None:
            prepare_town(args, town)


def learning_smoke_test():
    """Real PyTorch integration check, runnable in the user's installed CARLA environment."""
    torch.manual_seed(123)
    cfg = Config()
    # max_entities is a read-only alias of this configurable field.
    cfg.max_entity_obs = 4
    cfg.takeover_loss_weight = 1.0
    cfg.target_state_bc_weight = 2.0
    cfg.sigma_aug_prob = 0.0
    device = torch.device("cpu")
    agent = CaSTAgent(cfg, device)
    agent.freeze_source_copies()
    agent.bc_weight = 0.0
    replay = CaSTReplay(8, cfg, with_coverage=False)
    for i in range(8):
        row = {key: np.zeros(shape, np.float32) for key, shape in replay.spec.items()}
        row.update(mask=np.ones(cfg.max_entities, np.float32),
                   next_mask=np.ones(cfg.max_entities, np.float32), rewards=float(i % 3) - 0.25,
                   terminals=float(i == 7), has_expert=1.0, slow=float(i % 2),
                   expert_action=np.array([-0.8 if i % 2 == 0 else 0.6, 0.0], np.float32),
                   has_rejected=float(i == 0), rejected_action=np.array([0.6, 0.1], np.float32))
        replay.add(row)
    batch = replay.sample(8, device, np.arange(8))
    actor_before = [p.detach().clone() for p in agent.actor.parameters()]
    critic_before = [p.detach().clone() for p in agent.critics.parameters()]
    stats = agent.update(batch, batch, actor_step=False, bc_batch=batch, target_bc_batch=batch)
    assert all(torch.equal(a, b) for a, b in zip(actor_before, agent.actor.parameters())), "warmup changed actor"
    assert any(not torch.equal(a, b) for a, b in zip(critic_before, agent.critics.parameters())), "critic did not update"
    assert stats["takeover_batch_n"] == 1.0 and stats["takeover_loss"] >= 0.0
    stats = agent.update(batch, batch, actor_step=True, bc_batch=batch, target_bc_batch=batch)
    assert any(not torch.equal(a, b) for a, b in zip(actor_before, agent.actor.parameters())), "actor did not update"
    assert math.isfinite(stats["L_target_BC"]) and math.isfinite(stats["L_source_BC"])
    assert math.isfinite(stats["critic_loss"]) and math.isfinite(stats["actor_loss"])
    assert all(torch.isfinite(p).all() for p in agent.actor.parameters()), "nonfinite actor weights"
    values = torch.tensor([[1., 4.], [2., 2.], [3., 0.]])
    assert torch.allclose(reduce_critic_target(values, "mean", 0.1), torch.tensor([2., 2.]))
    assert torch.all(reduce_critic_target(values, "lcb", 0.1) <= values.mean(0))
    obs = {k: replay.data[k][0] for k in ("scalars", "edges", "mask", "scene")}
    q = agent.step_quantities(obs, None, None, 1.0)
    v = agent.audit_action_values(obs, [0.0, 0.0])
    assert math.isfinite(q["policy_entropy"]) and all(math.isfinite(x) for x in v.values())
    print("[LEARNING SMOKE TEST] PASS: real CPU forward/backward, warmup isolation, target/source BC, "
          "synthetic penalty, ensemble reduction, entropy and Q audit. No driving validation.")


def main() -> None:
    args = parse_args()
    if getattr(args, "use_safety_shield", False) or getattr(args, "eval_light_override", False):
        raise SystemExit("--use-safety-shield and --eval-light-override are evaluation options: use them with car_eval.py")
    if args.learning_smoke_test:
        learning_smoke_test()
    elif args.worker:
        worker_main(args)
    else:
        supervisor_main(args)


if __name__ == "__main__":
    main()
