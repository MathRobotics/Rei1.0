#!/usr/bin/env bash
# Compatible with macOS Bash 3.2. Each version/suite has its own uv environment.
set -uo pipefail

root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)" || exit 1
cd "$root" || exit 1

usage() {
  cat <<'HELP'
Usage: scripts/test-python.sh [OPTIONS] [3.11 3.12 ...] [-- PYTEST_ARGS...]

Default versions: 3.11 3.12 3.13 3.14; any Python 3.11+ may be specified.
  --suite core       Full suite with core/dev dependencies (default).
                     Optional backend tests may skip; reasons appear in logs.
  --suite robotics   NumPy/Pinocchio checks used in CI; no Rust, no skips allowed.
  --suite all        Run both suites in separate environments.
  --env-dir PATH     Environment parent (default: .venv-python-tests).
  --log-dir PATH     Log parent (default: .python-test-results).
  -h, --help         Show this help.

Uses uv.lock without updating it. uv may download Python and dependencies.
Existing .venv is not used. Test environments are synchronized exactly.
Extra pytest arguments are appended to each test command; for example -- -x.
In core mode, -- tests/test_runtime_integrity.py selects a smaller test set.
Failures do not stop later versions. Any failure returns a nonzero exit code.
Each run saves per-case logs and summary.tsv in a new directory under log-dir.
HELP
}

suite=core
env_dir="$root/.venv-python-tests"
log_dir="$root/.python-test-results"
versions=()
pytest_args=()
while [[ $# -gt 0 ]]; do
  case "$1" in
    -h|--help) usage; exit 0 ;;
    --suite|--env-dir|--log-dir)
      if [[ $# -lt 2 || -z "$2" ]]; then
        echo "Missing value for $1" >&2; exit 2
      fi
      case "$1" in
        --suite) suite="$2" ;;
        --env-dir) env_dir="$2" ;;
        --log-dir) log_dir="$2" ;;
      esac
      shift 2 ;;
    --) shift; pytest_args=("$@"); break ;;
    *)
      if [[ ! "$1" =~ ^3\.([0-9]+)(\.[0-9]+)?$ ]]; then
        echo "Invalid Python version or option: $1 (expected e.g. 3.12 or 3.12.12)" >&2
        exit 2
      fi
      minor="${BASH_REMATCH[1]}"
      if (( 10#$minor < 11 )); then
        echo "Unsupported Python $1: Rei requires Python >=3.11." >&2; exit 2
      fi
      versions+=("$1")
      shift ;;
  esac
done
case "$suite" in
  core|robotics) suites=("$suite") ;;
  all) suites=(core robotics) ;;
  *) echo "Invalid suite: $suite (expected core, robotics or all)" >&2; exit 2 ;;
esac
if [[ ${#versions[@]} -eq 0 ]]; then
  versions=(3.11 3.12 3.13 3.14)
fi
if ! command -v uv >/dev/null 2>&1; then
  echo "uv is required and must be on PATH." >&2; exit 1
fi
mkdir -p "$env_dir" "$log_dir" || exit 1
env_dir="$(cd "$env_dir" && pwd)" || exit 1
log_dir="$(cd "$log_dir" && pwd)" || exit 1
run_dir="$log_dir/$(date +%Y%m%d-%H%M%S)-$$"
mkdir -p "$run_dir" || exit 1
summary="$run_dir/summary.tsv"
printf 'python\tsuite\tresult\texit_code\tlog\n' > "$summary" || exit 1
export MPLBACKEND=Agg
# Do not import dependencies from a different active/source environment.
unset PYTHONPATH PYTHONHOME

run_case() {
  local version="$1" selected_suite="$2"
  local environment="$env_dir/$version-$selected_suite"
  local python="$environment/bin/python"
  local groups=(--group dev)
  if [[ "$selected_suite" == robotics ]]; then
    groups+=(--group kots --group pinocchio)
  fi
  echo "=== Python $version / $selected_suite ==="
  echo "Environment: $environment"
  echo "Stage: locked dependency sync"
  UV_PROJECT_ENVIRONMENT="$environment" uv sync --locked --python "$version" \
    --no-default-groups "${groups[@]}" || return $?
  echo "Stage: interpreter and dependency check"
  "$python" -c 'import platform, sys, rei; print(platform.platform()); print(sys.version); print(sys.executable); print("Rei:", rei.__file__)' || return $?
  uv pip check --python "$python" || return $?
  uv pip list --python "$python" || return $?
  if [[ "$selected_suite" == core ]]; then
    echo "Stage: core tests (optional backend skips allowed)"
    "$python" -m pytest -q -rs ${pytest_args[@]+"${pytest_args[@]}"} || return $?
  else
    echo "Stage: portable robotics tests (Rust excluded; skips fail)"
    "$python" -m pytest -q -rs --fail-on-skip -k 'not rust' \
      tests/test_kots_perturbation.py tests/test_ioc_validation.py \
      tests/test_urdf_joint_locking.py tests/test_kots_compile_order.py \
      tests/test_model_file_spec.py tests/test_prepared_trajectory_maps.py \
      tests/test_trajectory_example_cli.py ${pytest_args[@]+"${pytest_args[@]}"} || return $?
    "$python" -m pytest -q -rs --fail-on-skip -k numpy \
      tests/test_kots_urdf_json_parity.py tests/test_kots_autodiff_batch.py \
      ${pytest_args[@]+"${pytest_args[@]}"} || return $?
  fi
}

trap 'exit 130' INT
trap 'exit 143' TERM
failed=0
for version in "${versions[@]}"; do
  for selected_suite in "${suites[@]}"; do
    log="$run_dir/$version-$selected_suite.log"
    run_case "$version" "$selected_suite" 2>&1 | tee "$log"
    statuses=("${PIPESTATUS[@]}")
    status="${statuses[0]}"
    if [[ "$status" -eq 0 && "${statuses[1]}" -ne 0 ]]; then
      status="${statuses[1]}"
    fi
    if [[ "$status" -eq 130 || "$status" -eq 143 ]]; then
      exit "$status"
    elif [[ "$status" -eq 0 ]]; then
      result=PASS
    else
      result=FAIL
      failed=1
    fi
    printf '%s\t%s\t%s\t%s\t%s\n' "$version" "$selected_suite" "$result" "$status" "$log" >> "$summary" || exit 1
  done
done
echo
echo '=== Results ==='
cat "$summary"
echo "Results saved: $run_dir"
exit "$failed"
