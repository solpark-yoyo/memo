#!/bin/bash
# =============================================================================
#  Gaussianity Trajectory (KL to N(0,1)) — run_eps_trajectory.sh 의 kl_div wrapper
#
#  run_eps_trajectory.sh 에 MEASURE=kl_div 를 넘겨서
#  각 denoising step 의 x_t gaussianity (KL(N(μ,σ²)‖N(0,1))) 를 측정.
#
#  환경: conda div_DM (ori_memo/ 에서 실행)
#    bash shells/others/run_eps_trajectory_gaussianity.sh
# =============================================================================

# SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
# MEASURE=kl_div bash "${SCRIPT_DIR}/run_eps_trajectory.sh" "$@" # kl div
MEASURE=cmp_l2 bash shells/others/run_eps_trajectory.sh # compact spectrum domain l2 norm
