# FSQ receiver checkpoint audit

> Metrics come only from checkpoint-embedded metadata. Model and optimizer tensors were mapped to `meta`; no dataset re-evaluation was performed.

## Outcome

- Strict complete: **FALSE**
- Selected versions: 12; strict eligible: 0.
- Strict canonical checkpoint: `MISSING`.
- Per-version selection: `goal_best > best > latest > other`.

## Route and contract

| Version | Kind | Epoch | Route | Condition | Arch | Levels | K | D2 topology | Joint sender | Contract |
|---|---:|---:|---|---|---|---:|---:|---|---|---|
| cnn-fsq-k125-joint-index-v1 | best | 95 | joint_index | z1_x1 | cnn | 5x5x5 | 125 | unknown | MISSING | PASS |
| cnn-fsq-k4913-direct-cont-v1 | best | 75 | direct_q | z1_x1 | cnn | 17x17x17 | 4913 | unknown | MISSING | PASS |
| cnn-fsq-k4913-direct-hard-v1 | best | 20 | direct_q | z1_x1 | cnn | 17x17x17 | 4913 | unknown | MISSING | PASS |
| cnn-fsq-k4913-independent-d2-final-v6 | best | 5 | direct_q | z1_x1 | cnn | 17x17x17 | 4913 | independent-d2 | no | PASS |
| cnn-fsq-k4913-independent-d2-v5 | best | 35 | direct_q | z1_x1 | cnn | 17x17x17 | 4913 | independent-d2 | no | PASS |
| cnn-fsq-k4913-joint-predictable-v3 | best | 40 | direct_q | z1_x1 | cnn | 17x17x17 | 4913 | unknown | INVALID-v3-frozen | PASS |
| cnn-fsq-k4913-joint-predictable-v4 | best | 20 | direct_q | z1_x1 | cnn | 17x17x17 | 4913 | unknown | yes | PASS |
| cnn-fsq-k4913-preinit-oracleft-v2 | best | 30 | direct_q | z1_x1 | cnn | 17x17x17 | 4913 | unknown | MISSING | PASS |
| cnn-fsq-k4913-preinit-residual-anchor-v2 | best | 10 | direct_q | z1_x1 | cnn | 17x17x17 | 4913 | unknown | MISSING | PASS |
| cnn-fsq-k4913-rx-cont-jointft-v1 | best | 190 | direct_q | z1_x1 | cnn | 17x17x17 | 4913 | unknown | MISSING | PASS |
| cnn-fsq-k4913-rx-jointft-v1 | best | 75 | direct_q | z1_x1 | cnn | 17x17x17 | 4913 | unknown | MISSING | PASS |
| launcher-smoke | best | 1 | direct_q | z1_x1 | cnn | 17x17x17 | 4913 | unknown | MISSING | PASS |

## Reconstruction and prediction

| Version | PSNR x1 | PSNR oracle | PSNR pred | Delta oracle | Delta pred-x1 | Oracle gap | q MSE hard | q loss | Index acc |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| cnn-fsq-k125-joint-index-v1 | 21.7720 | 22.8760 | 21.9797 | 1.1040 | 0.2077 | 0.8963 | 0.3946 | 0.2831 | 0.0542 |
| cnn-fsq-k4913-direct-cont-v1 | 21.7720 | 22.9475 | 21.9543 | 1.1755 | 0.1823 | 0.9932 | 0.0547 | 0.0536 | 0.2728 |
| cnn-fsq-k4913-direct-hard-v1 | 21.7720 | 22.9475 | 21.8010 | 1.1755 | 0.0290 | 1.1465 | 0.0568 | 0.0557 | 0.2652 |
| cnn-fsq-k4913-independent-d2-final-v6 | 21.7720 | 22.7435 | 21.9707 | 0.9715 | 0.1986 | 0.7728 | 0.0048 | 0.0040 | 0.7202 |
| cnn-fsq-k4913-independent-d2-v5 | 21.7720 | 22.7435 | 22.0318 | 0.9715 | 0.2598 | 0.7117 | 0.0042 | 0.0035 | 0.7462 |
| cnn-fsq-k4913-joint-predictable-v3 | 21.7720 | 22.9755 | 21.9738 | 1.2034 | 0.2017 | 1.0017 | 0.0559 | 0.0547 | 0.2698 |
| cnn-fsq-k4913-joint-predictable-v4 | 21.7720 | 22.7435 | 21.9837 | 0.9715 | 0.2117 | 0.7598 | 0.0043 | 0.0035 | 0.7403 |
| cnn-fsq-k4913-preinit-oracleft-v2 | 21.7720 | 22.7653 | 21.9631 | 0.9933 | 0.1911 | 0.8022 | 0.0557 | 0.0545 | 0.2658 |
| cnn-fsq-k4913-preinit-residual-anchor-v2 | 21.7720 | 20.6543 | 21.8685 | -1.1178 | 0.0965 | -1.2142 | 0.0591 | 0.0581 | 0.2592 |
| cnn-fsq-k4913-rx-cont-jointft-v1 | 21.7720 | 20.9284 | 22.0217 | -0.8436 | 0.2497 | -1.0932 | 0.2717 | 0.2704 | 0.1077 |
| cnn-fsq-k4913-rx-jointft-v1 | 21.7720 | 20.1517 | 21.9379 | -1.6203 | 0.1659 | -1.7862 | 0.2665 | 0.2651 | 0.0981 |
| launcher-smoke | 20.2539 | 22.8976 | 18.7513 | 2.6437 | -1.5026 | 4.1463 | 0.2478 | 0.2482 | 0.0697 |

## Ablations and receiver audit

| Version | Condition-shuffle PSNR | Condition drop | Pred zero drop | Pred shuffle drop | Oracle zero drop | Oracle shuffle drop | Receiver-only audit |
|---|---:|---:|---:|---:|---:|---:|---:|
| cnn-fsq-k125-joint-index-v1 | MISSING | MISSING | MISSING | MISSING | MISSING | MISSING | 1.0000 |
| cnn-fsq-k4913-direct-cont-v1 | MISSING | MISSING | MISSING | MISSING | MISSING | MISSING | 1.0000 |
| cnn-fsq-k4913-direct-hard-v1 | MISSING | MISSING | MISSING | MISSING | MISSING | MISSING | 1.0000 |
| cnn-fsq-k4913-independent-d2-final-v6 | 21.7308 | 0.2399 | 2.0441 | 1.9676 | 2.9355 | 3.1577 | 1.0000 |
| cnn-fsq-k4913-independent-d2-v5 | 21.7073 | 0.3245 | 1.7804 | 2.2469 | 2.9355 | 3.1105 | 1.0000 |
| cnn-fsq-k4913-joint-predictable-v3 | MISSING | MISSING | 1.8989 | 2.0946 | MISSING | MISSING | 1.0000 |
| cnn-fsq-k4913-joint-predictable-v4 | MISSING | MISSING | 2.1757 | 2.3048 | 2.9355 | 3.0906 | 1.0000 |
| cnn-fsq-k4913-preinit-oracleft-v2 | MISSING | MISSING | 1.8968 | 1.8684 | MISSING | MISSING | 1.0000 |
| cnn-fsq-k4913-preinit-residual-anchor-v2 | MISSING | MISSING | 0.0923 | 0.0950 | MISSING | MISSING | 1.0000 |
| cnn-fsq-k4913-rx-cont-jointft-v1 | MISSING | MISSING | MISSING | MISSING | MISSING | MISSING | 1.0000 |
| cnn-fsq-k4913-rx-jointft-v1 | MISSING | MISSING | MISSING | MISSING | MISSING | MISSING | 1.0000 |
| launcher-smoke | MISSING | MISSING | MISSING | MISSING | MISSING | MISSING | 1.0000 |

## Strict gates

Thresholds: audit=1, condition/pred drops >=0.1 dB, oracle delta >=0.8 dB, oracle drops >=0.5 dB, predicted delta >=0.5 dB, with explicit topology and no-leakage contract.

| Version | Eligible | Canonical | Failures |
|---|---|---|---|
| cnn-fsq-k125-joint-index-v1 | FAIL | FAIL | condition_shuffle_drop, pred_drop_zero, pred_drop_shuffle, oracle_drop_zero, oracle_drop_shuffle, delta_x1, topology_contract_clear |
| cnn-fsq-k4913-direct-cont-v1 | FAIL | FAIL | condition_shuffle_drop, pred_drop_zero, pred_drop_shuffle, oracle_drop_zero, oracle_drop_shuffle, delta_x1, topology_contract_clear |
| cnn-fsq-k4913-direct-hard-v1 | FAIL | FAIL | condition_shuffle_drop, pred_drop_zero, pred_drop_shuffle, oracle_drop_zero, oracle_drop_shuffle, delta_x1, topology_contract_clear |
| cnn-fsq-k4913-independent-d2-final-v6 | FAIL | FAIL | delta_x1 |
| cnn-fsq-k4913-independent-d2-v5 | FAIL | FAIL | delta_x1 |
| cnn-fsq-k4913-joint-predictable-v3 | FAIL | FAIL | condition_shuffle_drop, oracle_drop_zero, oracle_drop_shuffle, delta_x1, topology_contract_clear, historical_sender_integrity |
| cnn-fsq-k4913-joint-predictable-v4 | FAIL | FAIL | condition_shuffle_drop, delta_x1, topology_contract_clear |
| cnn-fsq-k4913-preinit-oracleft-v2 | FAIL | FAIL | condition_shuffle_drop, oracle_drop_zero, oracle_drop_shuffle, delta_x1, topology_contract_clear |
| cnn-fsq-k4913-preinit-residual-anchor-v2 | FAIL | FAIL | condition_shuffle_drop, pred_drop_zero, pred_drop_shuffle, delta_oracle, oracle_drop_zero, oracle_drop_shuffle, delta_x1, topology_contract_clear |
| cnn-fsq-k4913-rx-cont-jointft-v1 | FAIL | FAIL | condition_shuffle_drop, pred_drop_zero, pred_drop_shuffle, delta_oracle, oracle_drop_zero, oracle_drop_shuffle, delta_x1, topology_contract_clear |
| cnn-fsq-k4913-rx-jointft-v1 | FAIL | FAIL | condition_shuffle_drop, pred_drop_zero, pred_drop_shuffle, delta_oracle, oracle_drop_zero, oracle_drop_shuffle, delta_x1, topology_contract_clear |
| launcher-smoke | FAIL | FAIL | condition_shuffle_drop, pred_drop_zero, pred_drop_shuffle, oracle_drop_zero, oracle_drop_shuffle, delta_x1, topology_contract_clear |

## Diagnostics

- `cnn-fsq-k125-joint-index-v1` (`MY-V2/jscc-f/explore-2/checkpoints-receiver/cnn-fsq-k125-joint-index-v1/fsq_receiver_joint_index_z1_x1_continuous_cnn_d3_l5x5x5_best.pth`): MISSING: condition_shuffle_drop; legacy no-condition audit cannot be strict; MISSING: explicit receiver_topology metadata.
- `cnn-fsq-k4913-direct-cont-v1` (`MY-V2/jscc-f/explore-2/checkpoints-receiver/cnn-fsq-k4913-direct-cont-v1/fsq_receiver_direct_q_z1_x1_continuous_cnn_d3_l17x17x17_best.pth`): MISSING: condition_shuffle_drop; legacy no-condition audit cannot be strict; MISSING: explicit receiver_topology metadata.
- `cnn-fsq-k4913-direct-hard-v1` (`MY-V2/jscc-f/explore-2/checkpoints-receiver/cnn-fsq-k4913-direct-hard-v1/fsq_receiver_direct_q_z1_x1_hard_cnn_d3_l17x17x17_best.pth`): MISSING: condition_shuffle_drop; legacy no-condition audit cannot be strict; MISSING: explicit receiver_topology metadata.
- `cnn-fsq-k4913-independent-d2-final-v6` (`MY-V2/jscc-f/explore-2/checkpoints-receiver/cnn-fsq-k4913-independent-d2-final-v6/fsq_receiver_direct_q_z1_x1_continuous_cnn_d3_l17x17x17_d2ft-oracle_independent-d2_best.pth`): checkpoint metadata contracts are explicit.
- `cnn-fsq-k4913-independent-d2-v5` (`MY-V2/jscc-f/explore-2/checkpoints-receiver/cnn-fsq-k4913-independent-d2-v5/fsq_receiver_direct_q_z1_x1_continuous_cnn_d3_l17x17x17_d2ft-oracle_independent-d2_best.pth`): checkpoint metadata contracts are explicit.
- `cnn-fsq-k4913-joint-predictable-v3` (`MY-V2/jscc-f/explore-2/checkpoints-receiver/cnn-fsq-k4913-joint-predictable-v3/fsq_receiver_direct_q_z1_x1_continuous_cnn_d3_l17x17x17_d2ft-oracle_best.pth`): KNOWN_INVALID: v3 requested joint sender training while E2/FSQ remained frozen; diagnostic only; MISSING: condition_shuffle_drop; legacy no-condition audit cannot be strict; MISSING: explicit receiver_topology metadata.
- `cnn-fsq-k4913-joint-predictable-v4` (`MY-V2/jscc-f/explore-2/checkpoints-receiver/cnn-fsq-k4913-joint-predictable-v4/fsq_receiver_direct_q_z1_x1_continuous_cnn_d3_l17x17x17_d2ft-oracle_best.pth`): MISSING: condition_shuffle_drop; legacy no-condition audit cannot be strict; MISSING: explicit receiver_topology metadata.
- `cnn-fsq-k4913-preinit-oracleft-v2` (`MY-V2/jscc-f/explore-2/checkpoints-receiver/cnn-fsq-k4913-preinit-oracleft-v2/fsq_receiver_direct_q_z1_x1_continuous_cnn_d3_l17x17x17_d2ft-oracle_best.pth`): MISSING: condition_shuffle_drop; legacy no-condition audit cannot be strict; MISSING: explicit receiver_topology metadata.
- `cnn-fsq-k4913-preinit-residual-anchor-v2` (`MY-V2/jscc-f/explore-2/checkpoints-receiver/cnn-fsq-k4913-preinit-residual-anchor-v2/fsq_receiver_direct_q_z1_x1_continuous_cnn_d3_l17x17x17_d2ft-residual_best.pth`): MISSING: condition_shuffle_drop; legacy no-condition audit cannot be strict; MISSING: explicit receiver_topology metadata.
- `cnn-fsq-k4913-rx-cont-jointft-v1` (`MY-V2/jscc-f/explore-2/checkpoints-receiver/cnn-fsq-k4913-rx-cont-jointft-v1/fsq_receiver_direct_q_z1_x1_continuous_cnn_d3_l17x17x17_d2ft-residual_best.pth`): MISSING: condition_shuffle_drop; legacy no-condition audit cannot be strict; MISSING: explicit receiver_topology metadata.
- `cnn-fsq-k4913-rx-jointft-v1` (`MY-V2/jscc-f/explore-2/checkpoints-receiver/cnn-fsq-k4913-rx-jointft-v1/fsq_receiver_direct_q_z1_x1_hard_cnn_d3_l17x17x17_d2ft-residual_best.pth`): MISSING: condition_shuffle_drop; legacy no-condition audit cannot be strict; MISSING: explicit receiver_topology metadata.
- `launcher-smoke` (`MY-V2/jscc-f/explore-2/checkpoints-receiver/launcher-smoke/fsq_receiver_direct_q_z1_x1_continuous_cnn_d3_l17x17x17_best.pth`): MISSING: condition_shuffle_drop; legacy no-condition audit cannot be strict; MISSING: explicit receiver_topology metadata.
