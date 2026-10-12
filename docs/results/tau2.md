# tau2-bench: Ratchet results

tau2-bench airline, retail and telecom run through Ratchet's worker loop: pass^k per run, with every planned, recorded and lost conversation counted and every stopped run listed with its reason.

Generated 2026-10-12 09:29 EDT by `python -m ratchet.bench.tau2 report` from the run records under the run directory; every number below is computed from those records.

Each conversation runs tau2-bench's own orchestrator, environment, user simulator and evaluator; only the agent's seat is taken by the Ratchet worker loop under the named harness. pass^k is tau2-bench's: the mean over tasks of C(c, k) / C(n, k) for c successes in n trials.

tau2-bench commit `4ce7c0397c1eb65c9bbe59aeacfe1ca44a1cd699` (data digest per run in its `config.json`).

## Runs

| domain | run | agent / user model | tasks x trials | recorded / planned | status | pass^1 | pass^2 | pass^3 | pass^4 |
|---|---|---|---|---|---|---|---|---|---|
| airline | `oss-full-4x` | `gpt-oss-120b` / `gpt-oss-120b` | 50 x 4 | 200 / 200 | complete | 56.0% | 46.0% | 39.5% | 34.0% |
| retail | `full-4x` | `gpt-5.6-luna` / `gpt-5.6-luna` | 114 x 4 | 269 / 456 | stopped |  |  |  |  |
| retail | `oss-full-4x` | `gpt-oss-120b` / `gpt-oss-120b` | 114 x 4 | 456 / 456 | complete | 53.5% | 40.5% | 32.7% | 27.2% |
| retail | `oss-smoke-5x1` | `gpt-oss-120b` / `gpt-oss-120b` | 5 x 1 | 5 / 5 | complete | 80.0% |  |  |  |
| retail | `smoke-5x1` | `gpt-5.6-luna` / `gpt-5.6-luna` | 5 x 1 | 5 / 5 | complete | 80.0% |  |  |  |
| telecom | `oss-full-4x` | `gpt-oss-120b` / `gpt-oss-120b` | 114 x 4 | 456 / 456 | complete | 69.7% | 56.0% | 47.6% | 42.1% |

### airline, harness `2a88a5824b61`, run `oss-full-4x`

- Agent model `gpt-oss-120b`, user simulator `gpt-oss-120b`, assertion judge `gpt-oss-120b`, temperature 0.0, seed 300 (trial seeds [626729, 373753, 361454, 1567]).
- Tasks: 50 of the domain's base set; trials: 4; conversations planned 200, recorded 200, completed 200, lost to the model endpoint after 2 retries 0.
- Successes: 112 of 200 recorded conversations.
- tau2-bench metrics over the 200 completed conversations: pass^1 56.0% | pass^2 46.0% | pass^3 39.5% | pass^4 34.0%.
- pass^k over all 200 recorded units, a lost unit counted as a failure: pass^1 56.0% | pass^2 46.0% | pass^3 39.5% | pass^4 34.0%.

| task id | split | successes / trials |
|---|---|---|
| `0` | train | 4 / 4 |
| `1` | train | 3 / 4 |
| `2` | heldout | 4 / 4 |
| `3` | train | 0 / 4 |
| `4` | train | 4 / 4 |
| `5` | train | 3 / 4 |
| `6` | heldout | 3 / 4 |
| `7` | train | 0 / 4 |
| `8` | heldout | 1 / 4 |
| `9` | train | 3 / 4 |
| `10` | train | 1 / 4 |
| `11` | train | 1 / 4 |
| `12` | train | 1 / 4 |
| `13` | heldout | 4 / 4 |
| `14` | train | 0 / 4 |
| `15` | train | 4 / 4 |
| `16` | heldout | 4 / 4 |
| `17` | train | 2 / 4 |
| `18` | heldout | 0 / 4 |
| `19` | heldout | 3 / 4 |
| `20` | train | 3 / 4 |
| `21` | train | 0 / 4 |
| `22` | heldout | 0 / 4 |
| `23` | train | 0 / 4 |
| `24` | heldout | 1 / 4 |
| `25` | heldout | 3 / 4 |
| `26` | heldout | 4 / 4 |
| `27` | train | 4 / 4 |
| `28` | train | 4 / 4 |
| `29` | heldout | 0 / 4 |
| `30` | heldout | 2 / 4 |
| `31` | heldout | 4 / 4 |
| `32` | heldout | 0 / 4 |
| `33` | train | 0 / 4 |
| `34` | train | 4 / 4 |
| `35` | heldout | 0 / 4 |
| `36` | train | 4 / 4 |
| `37` | heldout | 2 / 4 |
| `38` | train | 3 / 4 |
| `39` | train | 0 / 4 |
| `40` | train | 4 / 4 |
| `41` | train | 3 / 4 |
| `42` | train | 0 / 4 |
| `43` | train | 4 / 4 |
| `44` | heldout | 0 / 4 |
| `45` | heldout | 3 / 4 |
| `46` | train | 4 / 4 |
| `47` | train | 4 / 4 |
| `48` | heldout | 3 / 4 |
| `49` | train | 4 / 4 |

### retail, harness `6d9a2a325e67`, run `full-4x`

- Agent model `gpt-5.6-luna`, user simulator `gpt-5.6-luna`, assertion judge `gpt-5.6-luna`, temperature 0.0, seed 300 (trial seeds [626729, 373753, 361454, 1567]).
- Tasks: 114 of the domain's base set; trials: 4; conversations planned 456, recorded 269, completed 269, lost to the model endpoint after 2 retries 0.
- Successes: 185 of 269 recorded conversations.
- Stopped after 269 of 456 conversations: stopped by the coordinator's workstation-load rule (it would not finish by the 16:15 PT cut-off for workstation runs); not resumed, the gpt-oss-120b 4-trial runs took its place
- pass^k over all units: not computed, the run is not complete for every task and trial.

| task id | split | successes / trials |
|---|---|---|
| `0` | train | 3 / 3 |
| `1` | train | 3 / 3 |
| `2` | train | 1 / 3 |
| `3` | train | 3 / 3 |
| `4` | train | 3 / 3 |
| `5` | heldout | 2 / 3 |
| `6` | train | 1 / 3 |
| `7` | train | 0 / 3 |
| `8` | train | 2 / 3 |
| `9` | heldout | 0 / 3 |
| `10` | train | 3 / 3 |
| `11` | train | 1 / 3 |
| `12` | heldout | 2 / 3 |
| `13` | train | 3 / 3 |
| `14` | train | 1 / 3 |
| `15` | train | 3 / 3 |
| `16` | train | 3 / 3 |
| `17` | heldout | 3 / 3 |
| `18` | heldout | 2 / 3 |
| `19` | train | 2 / 3 |
| `20` | train | 3 / 3 |
| `21` | train | 2 / 3 |
| `22` | train | 3 / 3 |
| `23` | train | 3 / 3 |
| `24` | train | 3 / 3 |
| `25` | train | 3 / 3 |
| `26` | heldout | 3 / 3 |
| `27` | heldout | 0 / 3 |
| `28` | train | 2 / 2 |
| `29` | train | 3 / 3 |
| `30` | train | 2 / 2 |
| `31` | train | 1 / 2 |
| `32` | heldout | 1 / 3 |
| `33` | heldout | 3 / 3 |
| `34` | train | 2 / 3 |
| `35` | train | 3 / 3 |
| `36` | heldout | 2 / 2 |
| `37` | train | 1 / 2 |
| `38` | heldout | 1 / 2 |
| `39` | heldout | 1 / 3 |
| `40` | heldout | 2 / 3 |
| `41` | train | 1 / 3 |
| `42` | heldout | 2 / 2 |
| `43` | train | 2 / 3 |
| `44` | train | 3 / 3 |
| `45` | heldout | 0 / 2 |
| `46` | train | 1 / 2 |
| `47` | train | 2 / 2 |
| `48` | train | 3 / 3 |
| `49` | heldout | 2 / 3 |
| `50` | train | 3 / 3 |
| `51` | heldout | 1 / 2 |
| `52` | train | 0 / 2 |
| `53` | heldout | 2 / 2 |
| `54` | train | 1 / 2 |
| `55` | heldout | 2 / 2 |
| `56` | heldout | 2 / 2 |
| `57` | train | 2 / 2 |
| `58` | train | 1 / 2 |
| `59` | train | 0 / 2 |
| `60` | heldout | 1 / 2 |
| `61` | heldout | 1 / 2 |
| `62` | heldout | 2 / 2 |
| `63` | train | 2 / 2 |
| `64` | heldout | 1 / 2 |
| `65` | heldout | 2 / 2 |
| `66` | train | 2 / 2 |
| `67` | train | 1 / 2 |
| `68` | heldout | 1 / 2 |
| `69` | train | 2 / 2 |
| `70` | heldout | 2 / 2 |
| `71` | heldout | 0 / 2 |
| `72` | train | 1 / 2 |
| `73` | train | 2 / 2 |
| `74` | heldout | 0 / 2 |
| `75` | train | 1 / 2 |
| `76` | train | 2 / 2 |
| `77` | heldout | 2 / 2 |
| `78` | train | 2 / 2 |
| `79` | heldout | 1 / 2 |
| `80` | train | 2 / 2 |
| `81` | train | 1 / 2 |
| `82` | train | 2 / 2 |
| `83` | train | 2 / 2 |
| `84` | train | 1 / 2 |
| `85` | train | 2 / 2 |
| `86` | heldout | 1 / 2 |
| `87` | train | 1 / 2 |
| `88` | train | 2 / 2 |
| `89` | train | 1 / 2 |
| `90` | heldout | 1 / 2 |
| `91` | train | 0 / 2 |
| `92` | train | 2 / 2 |
| `93` | train | 1 / 2 |
| `94` | heldout | 2 / 2 |
| `95` | train | 2 / 2 |
| `96` | train | 1 / 2 |
| `97` | heldout | 2 / 2 |
| `98` | train | 0 / 2 |
| `99` | train | 2 / 2 |
| `100` | heldout | 0 / 2 |
| `101` | heldout | 1 / 2 |
| `102` | heldout | 2 / 2 |
| `103` | train | 1 / 2 |
| `104` | train | 1 / 2 |
| `105` | train | 0 / 2 |
| `106` | train | 2 / 2 |
| `107` | train | 0 / 2 |
| `108` | heldout | 2 / 2 |
| `109` | train | 0 / 2 |
| `110` | train | 2 / 2 |
| `111` | heldout | 1 / 2 |
| `112` | train | 1 / 2 |
| `113` | train | 1 / 2 |

### retail, harness `6d9a2a325e67`, run `oss-full-4x`

- Agent model `gpt-oss-120b`, user simulator `gpt-oss-120b`, assertion judge `gpt-oss-120b`, temperature 0.0, seed 300 (trial seeds [626729, 373753, 361454, 1567]).
- Tasks: 114 of the domain's base set; trials: 4; conversations planned 456, recorded 456, completed 456, lost to the model endpoint after 2 retries 0.
- Successes: 244 of 456 recorded conversations.
- tau2-bench metrics over the 456 completed conversations: pass^1 53.5% | pass^2 40.5% | pass^3 32.7% | pass^4 27.2%.
- pass^k over all 456 recorded units, a lost unit counted as a failure: pass^1 53.5% | pass^2 40.5% | pass^3 32.7% | pass^4 27.2%.

| task id | split | successes / trials |
|---|---|---|
| `0` | train | 3 / 4 |
| `1` | train | 4 / 4 |
| `2` | train | 2 / 4 |
| `3` | train | 3 / 4 |
| `4` | train | 0 / 4 |
| `5` | heldout | 0 / 4 |
| `6` | train | 4 / 4 |
| `7` | train | 3 / 4 |
| `8` | train | 3 / 4 |
| `9` | heldout | 4 / 4 |
| `10` | train | 4 / 4 |
| `11` | train | 1 / 4 |
| `12` | heldout | 4 / 4 |
| `13` | train | 3 / 4 |
| `14` | train | 1 / 4 |
| `15` | train | 4 / 4 |
| `16` | train | 4 / 4 |
| `17` | heldout | 4 / 4 |
| `18` | heldout | 0 / 4 |
| `19` | train | 2 / 4 |
| `20` | train | 0 / 4 |
| `21` | train | 3 / 4 |
| `22` | train | 2 / 4 |
| `23` | train | 3 / 4 |
| `24` | train | 3 / 4 |
| `25` | train | 4 / 4 |
| `26` | heldout | 4 / 4 |
| `27` | heldout | 1 / 4 |
| `28` | train | 1 / 4 |
| `29` | train | 4 / 4 |
| `30` | train | 0 / 4 |
| `31` | train | 0 / 4 |
| `32` | heldout | 0 / 4 |
| `33` | heldout | 3 / 4 |
| `34` | train | 2 / 4 |
| `35` | train | 0 / 4 |
| `36` | heldout | 0 / 4 |
| `37` | train | 2 / 4 |
| `38` | heldout | 0 / 4 |
| `39` | heldout | 3 / 4 |
| `40` | heldout | 4 / 4 |
| `41` | train | 0 / 4 |
| `42` | heldout | 4 / 4 |
| `43` | train | 2 / 4 |
| `44` | train | 3 / 4 |
| `45` | heldout | 0 / 4 |
| `46` | train | 4 / 4 |
| `47` | train | 3 / 4 |
| `48` | train | 4 / 4 |
| `49` | heldout | 2 / 4 |
| `50` | train | 4 / 4 |
| `51` | heldout | 3 / 4 |
| `52` | train | 3 / 4 |
| `53` | heldout | 4 / 4 |
| `54` | train | 0 / 4 |
| `55` | heldout | 3 / 4 |
| `56` | heldout | 0 / 4 |
| `57` | train | 3 / 4 |
| `58` | train | 3 / 4 |
| `59` | train | 0 / 4 |
| `60` | heldout | 3 / 4 |
| `61` | heldout | 4 / 4 |
| `62` | heldout | 4 / 4 |
| `63` | train | 4 / 4 |
| `64` | heldout | 3 / 4 |
| `65` | heldout | 4 / 4 |
| `66` | train | 1 / 4 |
| `67` | train | 0 / 4 |
| `68` | heldout | 3 / 4 |
| `69` | train | 4 / 4 |
| `70` | heldout | 2 / 4 |
| `71` | heldout | 1 / 4 |
| `72` | train | 0 / 4 |
| `73` | train | 4 / 4 |
| `74` | heldout | 1 / 4 |
| `75` | train | 4 / 4 |
| `76` | train | 0 / 4 |
| `77` | heldout | 2 / 4 |
| `78` | train | 3 / 4 |
| `79` | heldout | 2 / 4 |
| `80` | train | 4 / 4 |
| `81` | train | 0 / 4 |
| `82` | train | 0 / 4 |
| `83` | train | 0 / 4 |
| `84` | train | 2 / 4 |
| `85` | train | 3 / 4 |
| `86` | heldout | 1 / 4 |
| `87` | train | 0 / 4 |
| `88` | train | 4 / 4 |
| `89` | train | 1 / 4 |
| `90` | heldout | 0 / 4 |
| `91` | train | 1 / 4 |
| `92` | train | 4 / 4 |
| `93` | train | 0 / 4 |
| `94` | heldout | 3 / 4 |
| `95` | train | 4 / 4 |
| `96` | train | 1 / 4 |
| `97` | heldout | 2 / 4 |
| `98` | train | 1 / 4 |
| `99` | train | 2 / 4 |
| `100` | heldout | 4 / 4 |
| `101` | heldout | 0 / 4 |
| `102` | heldout | 2 / 4 |
| `103` | train | 2 / 4 |
| `104` | train | 0 / 4 |
| `105` | train | 0 / 4 |
| `106` | train | 1 / 4 |
| `107` | train | 3 / 4 |
| `108` | heldout | 2 / 4 |
| `109` | train | 0 / 4 |
| `110` | train | 0 / 4 |
| `111` | heldout | 4 / 4 |
| `112` | train | 3 / 4 |
| `113` | train | 4 / 4 |

### retail, harness `6d9a2a325e67`, run `oss-smoke-5x1`

- Agent model `gpt-oss-120b`, user simulator `gpt-oss-120b`, assertion judge `gpt-oss-120b`, temperature 0.0, seed 300 (trial seeds [626729]).
- Tasks: 5 of the domain's base set; trials: 1; conversations planned 5, recorded 5, completed 5, lost to the model endpoint after 2 retries 0.
- Successes: 4 of 5 recorded conversations.
- tau2-bench metrics over the 5 completed conversations: pass^1 80.0%.
- pass^k over all 5 recorded units, a lost unit counted as a failure: pass^1 80.0%.

| task id | split | successes / trials |
|---|---|---|
| `0` | train | 0 / 1 |
| `1` | train | 1 / 1 |
| `2` | train | 1 / 1 |
| `3` | train | 1 / 1 |
| `4` | train | 1 / 1 |

### retail, harness `6d9a2a325e67`, run `smoke-5x1`

- Agent model `gpt-5.6-luna`, user simulator `gpt-5.6-luna`, assertion judge `gpt-5.6-luna`, temperature 0.0, seed 300 (trial seeds [626729]).
- Tasks: 5 of the domain's base set; trials: 1; conversations planned 5, recorded 5, completed 5, lost to the model endpoint after 2 retries 0.
- Successes: 4 of 5 recorded conversations.
- tau2-bench metrics over the 5 completed conversations: pass^1 80.0%.
- pass^k over all 5 recorded units, a lost unit counted as a failure: pass^1 80.0%.

| task id | split | successes / trials |
|---|---|---|
| `0` | train | 1 / 1 |
| `1` | train | 0 / 1 |
| `2` | train | 1 / 1 |
| `3` | train | 1 / 1 |
| `4` | train | 1 / 1 |

### telecom, harness `f3caf6751556`, run `oss-full-4x`

- Agent model `gpt-oss-120b`, user simulator `gpt-oss-120b`, assertion judge `gpt-oss-120b`, temperature 0.0, seed 300 (trial seeds [626729, 373753, 361454, 1567]).
- Tasks: 114 of the domain's base set; trials: 4; conversations planned 456, recorded 456, completed 456, lost to the model endpoint after 2 retries 0.
- Successes: 318 of 456 recorded conversations.
- Stopped once after 239 conversations and resumed: stopped on the workstation by the coordinator's workstation-load rule (it would not finish by 16:15 PT); resumed with --resume on Hetzner box 178.105.114.53 (box_run snapshot 8c24fb93), where the remaining 217 conversations went to a pool of five HiPerGator gpt-oss-120b endpoints (same checkpoint and serve flags; workstation ports 8801, 8804-8807)
- tau2-bench metrics over the 456 completed conversations: pass^1 69.7% | pass^2 56.0% | pass^3 47.6% | pass^4 42.1%.
- pass^k over all 456 recorded units, a lost unit counted as a failure: pass^1 69.7% | pass^2 56.0% | pass^3 47.6% | pass^4 42.1%.

| task id | split | successes / trials |
|---|---|---|
| `[mobile_data_issue]airplane_mode_on|user_abroad_roaming_enabled_off[PERSONA:None]` | train | 4 / 4 |
| `[mobile_data_issue]airplane_mode_on|data_mode_off[PERSONA:None]` | train | 4 / 4 |
| `[mobile_data_issue]airplane_mode_on|bad_network_preference[PERSONA:Hard]` | train | 4 / 4 |
| `[mobile_data_issue]data_saver_mode_on|user_abroad_roaming_enabled_off[PERSONA:Easy]` | heldout | 4 / 4 |
| `[mobile_data_issue]bad_network_preference|user_abroad_roaming_enabled_off[PERSONA:Hard]` | train | 4 / 4 |
| `[mobile_data_issue]data_usage_exceeded|user_abroad_roaming_enabled_off[PERSONA:Easy]` | train | 4 / 4 |
| `[mobile_data_issue]data_mode_off|data_usage_exceeded[PERSONA:None]` | train | 4 / 4 |
| `[mobile_data_issue]data_saver_mode_on|data_usage_exceeded[PERSONA:Easy]` | train | 3 / 4 |
| `[mobile_data_issue]airplane_mode_on|bad_network_preference|user_abroad_roaming_enabled_off[PERSONA:Easy]` | train | 4 / 4 |
| `[mobile_data_issue]airplane_mode_on|data_saver_mode_on|user_abroad_roaming_disabled_on[PERSONA:None]` | heldout | 4 / 4 |
| `[mobile_data_issue]bad_network_preference|bad_vpn|user_abroad_roaming_enabled_off[PERSONA:Easy]` | train | 3 / 4 |
| `[mobile_data_issue]bad_vpn|data_saver_mode_on|user_abroad_roaming_disabled_on[PERSONA:None]` | train | 2 / 4 |
| `[mobile_data_issue]bad_network_preference|bad_vpn|user_abroad_roaming_disabled_on[PERSONA:None]` | train | 4 / 4 |
| `[mobile_data_issue]data_mode_off|data_usage_exceeded|user_abroad_roaming_disabled_off[PERSONA:Hard]` | train | 4 / 4 |
| `[mobile_data_issue]bad_network_preference|bad_vpn|user_abroad_roaming_disabled_off[PERSONA:Hard]` | heldout | 3 / 4 |
| `[mobile_data_issue]bad_network_preference|data_saver_mode_on|data_usage_exceeded[PERSONA:Hard]` | train | 4 / 4 |
| `[mobile_data_issue]airplane_mode_on|bad_network_preference|data_mode_off|user_abroad_roaming_disabled_on[PERSONA:Hard]` | train | 4 / 4 |
| `[mobile_data_issue]airplane_mode_on|bad_network_preference|data_usage_exceeded|user_abroad_roaming_disabled_off[PERSONA:Easy]` | train | 4 / 4 |
| `[mobile_data_issue]airplane_mode_on|bad_network_preference|data_mode_off|data_saver_mode_on[PERSONA:Hard]` | heldout | 4 / 4 |
| `[mobile_data_issue]bad_vpn|data_mode_off|data_usage_exceeded|user_abroad_roaming_disabled_off[PERSONA:None]` | heldout | 2 / 4 |
| `[mobile_data_issue]bad_network_preference|bad_vpn|data_usage_exceeded|user_abroad_roaming_disabled_off[PERSONA:Easy]` | train | 3 / 4 |
| `[mobile_data_issue]bad_network_preference|bad_vpn|data_mode_off|data_saver_mode_on[PERSONA:None]` | train | 4 / 4 |
| `[mobile_data_issue]airplane_mode_on|data_mode_off|data_saver_mode_on|data_usage_exceeded|user_abroad_roaming_enabled_off[PERSONA:Hard]` | heldout | 4 / 4 |
| `[mobile_data_issue]airplane_mode_on|bad_network_preference|data_mode_off|data_usage_exceeded|user_abroad_roaming_disabled_on[PERSONA:None]` | train | 3 / 4 |
| `[mobile_data_issue]airplane_mode_on|bad_network_preference|bad_vpn|data_saver_mode_on|user_abroad_roaming_disabled_on[PERSONA:Easy]` | train | 3 / 4 |
| `[mobile_data_issue]airplane_mode_on|bad_network_preference|data_saver_mode_on|data_usage_exceeded|user_abroad_roaming_disabled_off[PERSONA:None]` | train | 2 / 4 |
| `[mobile_data_issue]bad_network_preference|data_mode_off|data_saver_mode_on|data_usage_exceeded|user_abroad_roaming_disabled_off[PERSONA:Easy]` | train | 4 / 4 |
| `[mobile_data_issue]bad_network_preference|bad_vpn|data_saver_mode_on|data_usage_exceeded|user_abroad_roaming_disabled_off[PERSONA:Easy]` | heldout | 3 / 4 |
| `[mobile_data_issue]airplane_mode_on|bad_network_preference|bad_vpn|data_mode_off|data_usage_exceeded|user_abroad_roaming_enabled_off[PERSONA:Easy]` | train | 4 / 4 |
| `[mobile_data_issue]airplane_mode_on|bad_network_preference|bad_vpn|data_saver_mode_on|data_usage_exceeded|user_abroad_roaming_enabled_off[PERSONA:None]` | train | 3 / 4 |
| `[mobile_data_issue]airplane_mode_on|bad_network_preference|data_mode_off|data_saver_mode_on|data_usage_exceeded|user_abroad_roaming_disabled_on[PERSONA:None]` | train | 2 / 4 |
| `[mobile_data_issue]airplane_mode_on|bad_network_preference|bad_vpn|data_mode_off|data_saver_mode_on|user_abroad_roaming_disabled_off[PERSONA:Hard]` | train | 4 / 4 |
| `[mobile_data_issue]bad_network_preference|bad_vpn|data_mode_off|data_saver_mode_on|data_usage_exceeded|user_abroad_roaming_enabled_off[PERSONA:Easy]` | heldout | 2 / 4 |
| `[mobile_data_issue]airplane_mode_on|bad_network_preference|bad_vpn|data_mode_off|data_saver_mode_on|data_usage_exceeded|user_abroad_roaming_enabled_off[PERSONA:Easy]` | heldout | 4 / 4 |
| `[mobile_data_issue]airplane_mode_on|bad_network_preference|bad_vpn|data_mode_off|data_saver_mode_on|data_usage_exceeded|user_abroad_roaming_disabled_on[PERSONA:None]` | train | 3 / 4 |
| `[mobile_data_issue]airplane_mode_on|bad_network_preference|bad_vpn|data_mode_off|data_saver_mode_on|data_usage_exceeded|user_abroad_roaming_disabled_off[PERSONA:Hard]` | train | 3 / 4 |
| `[service_issue]airplane_mode_on|unseat_sim_card[PERSONA:None]` | train | 3 / 4 |
| `[service_issue]airplane_mode_on|lock_sim_card_pin[PERSONA:Easy]` | heldout | 4 / 4 |
| `[service_issue]airplane_mode_on|break_apn_settings[PERSONA:Hard]` | train | 2 / 4 |
| `[service_issue]airplane_mode_on|overdue_bill_suspension[PERSONA:None]` | heldout | 3 / 4 |
| `[service_issue]overdue_bill_suspension|unseat_sim_card[PERSONA:Easy]` | heldout | 3 / 4 |
| `[service_issue]contract_end_suspension|unseat_sim_card[PERSONA:Hard]` | heldout | 4 / 4 |
| `[service_issue]break_apn_settings|lock_sim_card_pin[PERSONA:None]` | heldout | 4 / 4 |
| `[service_issue]lock_sim_card_pin|overdue_bill_suspension[PERSONA:Easy]` | train | 3 / 4 |
| `[service_issue]contract_end_suspension|lock_sim_card_pin[PERSONA:Hard]` | train | 4 / 4 |
| `[service_issue]airplane_mode_on|lock_sim_card_pin|unseat_sim_card[PERSONA:Hard]` | train | 4 / 4 |
| `[service_issue]airplane_mode_on|break_apn_settings|unseat_sim_card[PERSONA:None]` | train | 2 / 4 |
| `[service_issue]airplane_mode_on|overdue_bill_suspension|unseat_sim_card[PERSONA:Easy]` | train | 4 / 4 |
| `[service_issue]airplane_mode_on|break_apn_settings|lock_sim_card_pin[PERSONA:None]` | train | 4 / 4 |
| `[service_issue]airplane_mode_on|lock_sim_card_pin|overdue_bill_suspension[PERSONA:Easy]` | heldout | 4 / 4 |
| `[service_issue]airplane_mode_on|break_apn_settings|overdue_bill_suspension[PERSONA:None]` | heldout | 4 / 4 |
| `[service_issue]break_apn_settings|overdue_bill_suspension|unseat_sim_card[PERSONA:Hard]` | train | 2 / 4 |
| `[service_issue]break_apn_settings|lock_sim_card_pin|overdue_bill_suspension[PERSONA:Easy]` | heldout | 4 / 4 |
| `[service_issue]break_apn_settings|contract_end_suspension|lock_sim_card_pin[PERSONA:Hard]` | heldout | 4 / 4 |
| `[service_issue]airplane_mode_on|break_apn_settings|lock_sim_card_pin|unseat_sim_card[PERSONA:None]` | heldout | 4 / 4 |
| `[service_issue]airplane_mode_on|lock_sim_card_pin|overdue_bill_suspension|unseat_sim_card[PERSONA:Easy]` | train | 4 / 4 |
| `[service_issue]airplane_mode_on|contract_end_suspension|lock_sim_card_pin|unseat_sim_card[PERSONA:Hard]` | heldout | 4 / 4 |
| `[service_issue]airplane_mode_on|break_apn_settings|overdue_bill_suspension|unseat_sim_card[PERSONA:None]` | train | 3 / 4 |
| `[service_issue]airplane_mode_on|break_apn_settings|contract_end_suspension|unseat_sim_card[PERSONA:Easy]` | train | 4 / 4 |
| `[service_issue]airplane_mode_on|break_apn_settings|lock_sim_card_pin|overdue_bill_suspension[PERSONA:Hard]` | heldout | 4 / 4 |
| `[service_issue]airplane_mode_on|break_apn_settings|contract_end_suspension|lock_sim_card_pin[PERSONA:None]` | heldout | 4 / 4 |
| `[service_issue]break_apn_settings|lock_sim_card_pin|overdue_bill_suspension|unseat_sim_card[PERSONA:Easy]` | train | 3 / 4 |
| `[service_issue]break_apn_settings|contract_end_suspension|lock_sim_card_pin|unseat_sim_card[PERSONA:Hard]` | heldout | 4 / 4 |
| `[service_issue]airplane_mode_on|break_apn_settings|lock_sim_card_pin|overdue_bill_suspension|unseat_sim_card[PERSONA:None]` | train | 4 / 4 |
| `[service_issue]airplane_mode_on|break_apn_settings|contract_end_suspension|lock_sim_card_pin|unseat_sim_card[PERSONA:Easy]` | heldout | 4 / 4 |
| `[mms_issue]airplane_mode_on|break_app_both_permissions[PERSONA:Hard]` | heldout | 2 / 4 |
| `[mms_issue]break_app_sms_permission|data_mode_off[PERSONA:None]` | heldout | 2 / 4 |
| `[mms_issue]break_app_storage_permission|data_usage_exceeded[PERSONA:Easy]` | train | 2 / 4 |
| `[mms_issue]break_app_both_permissions|data_usage_exceeded[PERSONA:Hard]` | train | 3 / 4 |
| `[mms_issue]break_apn_mms_setting|user_abroad_roaming_enabled_off[PERSONA:Hard]` | heldout | 4 / 4 |
| `[mms_issue]bad_network_preference|user_abroad_roaming_disabled_on[PERSONA:None]` | train | 4 / 4 |
| `[mms_issue]bad_network_preference|user_abroad_roaming_disabled_off[PERSONA:None]` | train | 4 / 4 |
| `[mms_issue]bad_network_preference|break_app_both_permissions[PERSONA:Easy]` | heldout | 4 / 4 |
| `[mms_issue]airplane_mode_on|bad_network_preference|data_usage_exceeded[PERSONA:None]` | train | 2 / 4 |
| `[mms_issue]airplane_mode_on|bad_wifi_calling|user_abroad_roaming_enabled_off[PERSONA:Easy]` | heldout | 2 / 4 |
| `[mms_issue]break_app_storage_permission|unseat_sim_card|user_abroad_roaming_disabled_on[PERSONA:Easy]` | train | 4 / 4 |
| `[mms_issue]break_app_both_permissions|unseat_sim_card|user_abroad_roaming_disabled_on[PERSONA:Hard]` | train | 1 / 4 |
| `[mms_issue]bad_wifi_calling|data_mode_off|data_usage_exceeded[PERSONA:Easy]` | train | 3 / 4 |
| `[mms_issue]bad_network_preference|data_mode_off|user_abroad_roaming_disabled_on[PERSONA:None]` | heldout | 3 / 4 |
| `[mms_issue]break_apn_mms_setting|data_mode_off|user_abroad_roaming_disabled_on[PERSONA:Hard]` | train | 3 / 4 |
| `[mms_issue]break_app_sms_permission|data_usage_exceeded|user_abroad_roaming_disabled_on[PERSONA:None]` | train | 4 / 4 |
| `[mms_issue]bad_network_preference|break_app_sms_permission|user_abroad_roaming_disabled_on[PERSONA:Hard]` | heldout | 0 / 4 |
| `[mms_issue]airplane_mode_on|break_app_both_permissions|data_usage_exceeded|user_abroad_roaming_disabled_off[PERSONA:None]` | heldout | 3 / 4 |
| `[mms_issue]airplane_mode_on|bad_network_preference|break_apn_mms_setting|data_usage_exceeded[PERSONA:Hard]` | train | 3 / 4 |
| `[mms_issue]airplane_mode_on|bad_network_preference|break_apn_mms_setting|break_app_storage_permission[PERSONA:None]` | train | 4 / 4 |
| `[mms_issue]bad_wifi_calling|break_apn_mms_setting|unseat_sim_card|user_abroad_roaming_enabled_off[PERSONA:Easy]` | train | 2 / 4 |
| `[mms_issue]break_apn_mms_setting|data_mode_off|data_usage_exceeded|user_abroad_roaming_disabled_on[PERSONA:Hard]` | train | 2 / 4 |
| `[mms_issue]bad_wifi_calling|break_apn_mms_setting|break_app_sms_permission|data_usage_exceeded[PERSONA:Hard]` | train | 2 / 4 |
| `[mms_issue]airplane_mode_on|bad_network_preference|bad_wifi_calling|break_app_storage_permission|unseat_sim_card[PERSONA:Hard]` | train | 2 / 4 |
| `[mms_issue]airplane_mode_on|bad_network_preference|break_app_storage_permission|data_mode_off|user_abroad_roaming_enabled_off[PERSONA:Easy]` | heldout | 3 / 4 |
| `[mms_issue]bad_wifi_calling|break_apn_mms_setting|data_mode_off|data_usage_exceeded|unseat_sim_card[PERSONA:None]` | train | 1 / 4 |
| `[mms_issue]bad_network_preference|break_app_sms_permission|data_mode_off|data_usage_exceeded|user_abroad_roaming_enabled_off[PERSONA:None]` | train | 1 / 4 |
| `[mms_issue]bad_network_preference|bad_wifi_calling|break_app_both_permissions|data_usage_exceeded|user_abroad_roaming_disabled_off[PERSONA:Hard]` | train | 1 / 4 |
| `[mms_issue]airplane_mode_on|bad_network_preference|break_apn_mms_setting|data_mode_off|unseat_sim_card|user_abroad_roaming_disabled_off[PERSONA:Easy]` | train | 2 / 4 |
| `[mms_issue]airplane_mode_on|bad_network_preference|bad_wifi_calling|data_usage_exceeded|unseat_sim_card|user_abroad_roaming_disabled_on[PERSONA:Easy]` | train | 1 / 4 |
| `[mms_issue]airplane_mode_on|bad_network_preference|break_app_both_permissions|data_usage_exceeded|unseat_sim_card|user_abroad_roaming_disabled_on[PERSONA:Hard]` | heldout | 2 / 4 |
| `[mms_issue]bad_network_preference|bad_wifi_calling|break_app_sms_permission|data_mode_off|unseat_sim_card|user_abroad_roaming_enabled_off[PERSONA:Hard]` | train | 0 / 4 |
| `[mms_issue]bad_network_preference|bad_wifi_calling|break_app_both_permissions|data_usage_exceeded|unseat_sim_card|user_abroad_roaming_disabled_off[PERSONA:None]` | train | 1 / 4 |
| `[mms_issue]bad_wifi_calling|break_apn_mms_setting|break_app_both_permissions|data_mode_off|data_usage_exceeded|user_abroad_roaming_disabled_off[PERSONA:None]` | heldout | 1 / 4 |
| `[mms_issue]airplane_mode_on|bad_wifi_calling|break_app_both_permissions|data_mode_off|data_usage_exceeded|unseat_sim_card|user_abroad_roaming_enabled_off[PERSONA:None]` | heldout | 0 / 4 |
| `[mms_issue]airplane_mode_on|bad_network_preference|bad_wifi_calling|break_apn_mms_setting|break_app_both_permissions|unseat_sim_card|user_abroad_roaming_enabled_off[PERSONA:Hard]` | heldout | 3 / 4 |
| `[mms_issue]airplane_mode_on|bad_network_preference|break_apn_mms_setting|break_app_both_permissions|data_mode_off|data_usage_exceeded|user_abroad_roaming_enabled_off[PERSONA:None]` | train | 0 / 4 |
| `[mms_issue]bad_network_preference|bad_wifi_calling|break_app_sms_permission|data_mode_off|data_usage_exceeded|unseat_sim_card|user_abroad_roaming_enabled_off[PERSONA:Easy]` | heldout | 2 / 4 |
| `[mms_issue]bad_network_preference|bad_wifi_calling|break_apn_mms_setting|break_app_storage_permission|data_mode_off|data_usage_exceeded|unseat_sim_card[PERSONA:Easy]` | train | 1 / 4 |
| `[mms_issue]airplane_mode_on|bad_network_preference|bad_wifi_calling|break_app_storage_permission|data_mode_off|data_usage_exceeded|unseat_sim_card|user_abroad_roaming_enabled_off[PERSONA:Hard]` | train | 0 / 4 |
| `[mms_issue]airplane_mode_on|bad_network_preference|bad_wifi_calling|break_app_sms_permission|data_mode_off|data_usage_exceeded|unseat_sim_card|user_abroad_roaming_disabled_off[PERSONA:None]` | train | 1 / 4 |
| `[mms_issue]airplane_mode_on|bad_network_preference|bad_wifi_calling|break_apn_mms_setting|break_app_storage_permission|data_mode_off|data_usage_exceeded|unseat_sim_card[PERSONA:Easy]` | train | 1 / 4 |
| `[mms_issue]airplane_mode_on|bad_network_preference|bad_wifi_calling|break_apn_mms_setting|break_app_storage_permission|data_usage_exceeded|unseat_sim_card|user_abroad_roaming_disabled_off[PERSONA:Easy]` | train | 3 / 4 |
| `[mms_issue]airplane_mode_on|bad_network_preference|bad_wifi_calling|break_apn_mms_setting|break_app_sms_permission|data_mode_off|data_usage_exceeded|unseat_sim_card|user_abroad_roaming_enabled_off[PERSONA:None]` | train | 0 / 4 |
| `[mms_issue]airplane_mode_on|bad_network_preference|bad_wifi_calling|break_apn_mms_setting|break_app_sms_permission|data_mode_off|data_usage_exceeded|unseat_sim_card|user_abroad_roaming_disabled_on[PERSONA:None]` | heldout | 1 / 4 |
| `[mms_issue]airplane_mode_on|bad_network_preference|bad_wifi_calling|break_apn_mms_setting|break_app_storage_permission|data_mode_off|data_usage_exceeded|unseat_sim_card|user_abroad_roaming_disabled_on[PERSONA:Easy]` | heldout | 0 / 4 |
| `[mms_issue]airplane_mode_on|bad_network_preference|bad_wifi_calling|break_apn_mms_setting|break_app_both_permissions|data_mode_off|data_usage_exceeded|unseat_sim_card|user_abroad_roaming_disabled_on[PERSONA:Hard]` | train | 0 / 4 |
| `[mms_issue]airplane_mode_on|bad_network_preference|bad_wifi_calling|break_apn_mms_setting|break_app_storage_permission|data_mode_off|data_usage_exceeded|unseat_sim_card|user_abroad_roaming_disabled_off[PERSONA:Easy]` | train | 0 / 4 |
| `[mms_issue]airplane_mode_on|bad_network_preference|bad_wifi_calling|break_apn_mms_setting|break_app_both_permissions|data_mode_off|data_usage_exceeded|unseat_sim_card|user_abroad_roaming_disabled_off[PERSONA:Hard]` | train | 0 / 4 |

## Evolve runs

Each evolve run measures the parent harness twice per generation on the held-out tasks (and once on the train tasks), lets the improver propose candidates from failed train conversations, evaluates some of them on the same tasks, and admits a candidate only through a signed decision whose checks all pass. A conversation lost to the model endpoint is not a pass: it counts as a failure in the passed counts and as invalid evidence in the held-out check, which then fails.

### Evolve run `evolve/retail/oss-evolve-1`

- agent model `gpt-oss-120b`, user model `gpt-oss-120b`, improver model `gpt-5.6-terra`.
- Stopped (`STOPPED.txt`): Stopped at 18:49:04 EDT 2026-10-11 by the coordinator's workstation-load rule: the candidate evaluation would not finish by the 16:15 PT cut-off for workstation runs. Generation 1's two parent runs had finished; candidate 2c8f6781c4af had 46 conversations recorded and no decision was made. Re-run on a box as oss-evolve-2. (Recorded in evolve-retail-oss-evolve-1.log: "STOPPED 2026-10-11T18:49:04 by the coordinator's workstation-load rule (candidate evaluation would not finish by 16:15 PT); re-run on a box as oss-evolve-2".)
- No decision was made.

Parent runs (the parent harness measured twice per generation; train is run in the first only):

| gen | parent | run | held-out passed | train passed | lost to the endpoint |
|---|---|---|---|---|---|
| 1 | `6d9a2a325e67` | 1 | 21/40 | 42/74 | 0 of 114 |
| 1 | `6d9a2a325e67` | 2 | 27/40 | not run | 0 of 40 |

Candidates (every evaluated candidate, with its decision):

| gen | candidate | kind | held-out passed | train passed | lost to the endpoint | decision | failed checks |
|---|---|---|---|---|---|---|---|
| | none | | | | | | |

- Candidate `2c8f6781c4af`: 46 recorded conversations that no finished evaluation counts (held-out passed 8/14, train passed 23/32). No decision rests on them.

### Evolve run `box/evolve-retail/evolve/retail/oss-evolve-2`

- agent model `gpt-oss-120b`, user model `gpt-oss-120b`, improver model `gpt-5.6-terra`.
- Failed (`FAILED.txt`): Failed run: every improver call (gpt-5.6-terra through the local proxy, box port 18839 -> workstation 8787) was refused with HTTP 401, because the box launcher passed the agent key "unused" to the improver and the proxy accepts only "dummy-key". The generation record says so (note: improver model calls failed for 4 proposal task(s)); it proposed and evaluated no candidate, so this run says nothing about the improver. The parent's two runs (train 40/74, held-out 20/40 and 19/40) are valid measurements. Launcher fixed 2026-10-11 (TAU2_IMPROVER_KEY).
- Generation 1: 0 candidates proposed, 0 evaluated, 0 admitted. Note: improver model calls failed for 4 proposal task(s).
- No decision was made.

Parent runs (the parent harness measured twice per generation; train is run in the first only):

| gen | parent | run | held-out passed | train passed | lost to the endpoint |
|---|---|---|---|---|---|
| 1 | `6d9a2a325e67` | 1 | 20/40 | 40/74 | 0 of 114 |
| 1 | `6d9a2a325e67` | 2 | 19/40 | not run | 0 of 40 |

Candidates (every evaluated candidate, with its decision):

| gen | candidate | kind | held-out passed | train passed | lost to the endpoint | decision | failed checks |
|---|---|---|---|---|---|---|---|
| | none | | | | | | |

### Evolve run `box/evolve-retail3/evolve/retail/oss-evolve-3`

- agent model `gpt-oss-120b`, user model `gpt-oss-120b`, improver model `gpt-5.6-terra`.
- Finished: 1 of 1 planned generations ran.
- Note (`NOTE.txt`): Candidates cf98c1481b70 and f7894282cb74 lost 38 and 109 of their 114 conversations to the agent endpoint between 01:04:32 and 01:13:54 PT on 2026-10-12 (box log, 08:04:32 to 08:13:54 UTC), each ending in 'OpenAIException - Connection error'; the box's tunnel log records connections to workstation port 8804, one of the two gpt-oss-120b endpoints, failing. Their counts (held-out 12 and 1 of 40) count those lost runs as failures and do not measure the candidates; their held-out checks had 15 and 38 tasks with no valid outcome, which alone fails the check. Candidate 5fcb50f1f4fc lost nothing: held-out 26 of 40 against the parent's 20 and 24, with 3 stable recoveries, rejected because 5 held-out tasks that passed in a parent run failed and 16 of the 67 must-succeed tasks that passed in a parent run regressed.
- Generation 1: 9 candidates proposed, 3 evaluated, 0 admitted. Note: improver model calls failed for 1 proposal task(s).
- 3 decisions, 0 admitted, signed with key `b61d7e39c111`.

Parent runs (the parent harness measured twice per generation; train is run in the first only):

| gen | parent | run | held-out passed | train passed | lost to the endpoint |
|---|---|---|---|---|---|
| 1 | `6d9a2a325e67` | 1 | 20/40 | 39/74 | 0 of 114 |
| 1 | `6d9a2a325e67` | 2 | 24/40 | not run | 0 of 40 |

Candidates (every evaluated candidate, with its decision):

| gen | candidate | kind | held-out passed | train passed | lost to the endpoint | decision | failed checks |
|---|---|---|---|---|---|---|---|
| 1 | `5fcb50f1f4fc` | procedure | 26/40 | 36/74 | 0 of 114 | rejected | heldout, no_regression |
| 1 | `cf98c1481b70` | prompt_edit | 12/40 | 31/74 | 38 of 114 | rejected | heldout, no_regression |
| 1 | `f7894282cb74` | procedure | 1/40 | 2/74 | 109 of 114 | rejected | heldout, no_regression |
