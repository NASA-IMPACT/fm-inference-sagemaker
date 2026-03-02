#!/bin/bash
set -euo pipefail

##############################################################################
# Fair Build Timing Comparison: build_and_push.sh vs deploy_robust.sh
#
# Runs 4 tests:
#   1. build_and_push.sh  — cold cache (sequential builds)
#   2. deploy_robust.sh   — cold cache (parallel builds)
#   3. build_and_push.sh  — warm cache (sequential builds)
#   4. deploy_robust.sh   — warm cache (parallel builds)
#
# Both scripts build the same 7 images using identical Dockerfiles and
# --platform linux/amd64. The difference is parallel vs sequential builds
# and script overhead.
#
# Safety: Set SKIP_PRUNE=true to skip Docker prune between cold runs.
#         This avoids destroying images/volumes used by other services
#         on shared machines.
#
# Validation: Both scripts use set -e / set -euo pipefail, so exit code 0
# guarantees all 7 images were built. We additionally count build messages
# in the log as a sanity check.
##############################################################################

SKIP_PRUNE="${SKIP_PRUNE:-false}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RESULTS_FILE="$SCRIPT_DIR/build_timing_fair_comparison.txt"
LOG_DIR="$SCRIPT_DIR/build_logs"
mkdir -p "$LOG_DIR"

EXPECTED_IMAGES=7

header() {
    echo ""
    echo "============================================================"
    echo "  $1"
    echo "============================================================"
}

prune_all() {
    if [[ "$SKIP_PRUNE" == "true" ]]; then
        echo "  SKIP_PRUNE=true: Skipping Docker prune (shared environment safety)."
        echo "  Note: 'cold' results reflect whatever cache state exists."
        return 0
    fi
    echo "  Pruning all Docker images and build cache (cold start)..."
    docker system prune -af --volumes 2>&1 | tail -3
    docker builder prune -af 2>&1 | tail -3
    echo "  Prune complete."
}

validate_build() {
    local exit_code=$1
    local log_file=$2
    local script_name=$3

    if [[ $exit_code -ne 0 ]]; then
        echo "  FAIL: $script_name exited with code $exit_code"
        echo "  See log: $log_file"
        return 1
    fi

    # Count build messages as a sanity check
    local build_count=0
    if [[ "$script_name" == *"build_and_push"* ]]; then
        build_count=$(grep -c "Building temporary image to get digest:" "$log_file" || echo 0)
    else
        # deploy_robust.sh: count specific build messages
        grep -q "Building main inference image" "$log_file" && ((build_count++)) || true
        grep -q "Building base pipeline image" "$log_file" && ((build_count++)) || true
        grep -q "Building tile server" "$log_file" && ((build_count++)) || true
        for svc in floods burn_scars crop_classification surya_rollout; do
            grep -q "Building $svc" "$log_file" && ((build_count++)) || true
        done
    fi

    echo "  Exit code: 0 (success)"
    echo "  Build messages found: $build_count / $EXPECTED_IMAGES"

    if [[ $build_count -lt $EXPECTED_IMAGES ]]; then
        echo "  WARNING: Expected $EXPECTED_IMAGES build messages, found $build_count"
        echo "  See log: $log_file"
        return 1
    fi

    echo "  PASS: All $EXPECTED_IMAGES images built successfully"
    return 0
}

run_test() {
    local run_number=$1
    local script=$2
    local label=$3
    local cache_type=$4  # "cold" or "warm"

    header "Run $run_number: $label ($cache_type cache)"

    if [[ "$cache_type" == "cold" ]]; then
        prune_all
    else
        echo "  Warm cache: skipping prune"
    fi

    local log_file="$LOG_DIR/run${run_number}_$(basename "$script" .sh).log"

    echo "  Starting: $(date '+%Y-%m-%d %H:%M:%S')"
    local start_time
    start_time=$(date +%s)

    local exit_code=0
    SKIP_PUSH=true SKIP_DEPLOY=true bash "$SCRIPT_DIR/$script" >"$log_file" 2>&1 || exit_code=$?

    local end_time
    end_time=$(date +%s)
    local duration=$((end_time - start_time))
    local minutes=$((duration / 60))
    local seconds=$((duration % 60))

    echo "  Finished: $(date '+%Y-%m-%d %H:%M:%S')"
    echo "  Duration: ${minutes}m ${seconds}s ($duration seconds)"

    if validate_build "$exit_code" "$log_file" "$script"; then
        echo "$run_number|$label|$cache_type|PASS|${minutes}m ${seconds}s|${duration}" >>"$RESULTS_FILE"
    else
        echo "$run_number|$label|$cache_type|FAILED|${minutes}m ${seconds}s|${duration}" >>"$RESULTS_FILE"
        echo ""
        echo "  ERROR: Run $run_number failed. Check log: $log_file"
        echo "  Continuing with remaining tests..."
    fi
}

print_results() {
    header "RESULTS: Fair Build Timing Comparison"

    printf "%-5s  %-30s  %-6s  %-8s  %s\n" "Run" "Script" "Cache" "Status" "Time"
    printf "%-5s  %-30s  %-6s  %-8s  %s\n" "---" "------------------------------" "------" "--------" "--------"

    while IFS='|' read -r num label cache status time _rest; do
        printf "%-5s  %-30s  %-6s  %-8s  %s\n" "$num" "$label" "$cache" "$status" "$time"
    done <"$RESULTS_FILE"

    echo ""

    # Calculate speedup if we have valid cold and warm pairs
    local cold_bap="" cold_dr="" warm_bap="" warm_dr=""
    while IFS='|' read -r num label cache status _time seconds; do
        [[ "$status" != "PASS" ]] && continue
        case "$num" in
            1) cold_bap=$seconds ;;
            2) cold_dr=$seconds ;;
            3) warm_bap=$seconds ;;
            4) warm_dr=$seconds ;;
        esac
    done <"$RESULTS_FILE"

    if [[ -n "$cold_bap" && -n "$cold_dr" ]]; then
        local speedup
        speedup=$(awk "BEGIN {printf \"%.1f\", $cold_bap / $cold_dr}")
        echo "Cold cache speedup (deploy_robust / build_and_push): ${speedup}x"
    fi

    if [[ -n "$warm_bap" && -n "$warm_dr" ]]; then
        local speedup
        speedup=$(awk "BEGIN {printf \"%.1f\", $warm_bap / $warm_dr}")
        echo "Warm cache speedup (deploy_robust / build_and_push): ${speedup}x"
    fi

    echo ""
    echo "Full logs: $LOG_DIR/"
    echo "Results:   $RESULTS_FILE"
}

##############################################################################
# Main
##############################################################################

echo "Fair Build Timing Comparison"
echo "Date: $(date)"
echo "Both scripts build all 7 images with --platform linux/amd64"
echo "SKIP_PUSH=true SKIP_DEPLOY=true (build-only, no ECR interaction)"

# Clear previous results
>"$RESULTS_FILE"

run_test 1 "build_and_push.sh" "build_and_push (sequential)" "cold"
run_test 2 "deploy_robust.sh"  "deploy_robust (parallel)"    "cold"
run_test 3 "build_and_push.sh" "build_and_push (sequential)" "warm"
run_test 4 "deploy_robust.sh"  "deploy_robust (parallel)"    "warm"

print_results
