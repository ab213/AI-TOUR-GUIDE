#!/bin/bash
# thermal_watchdog.sh
# Monitors CPU temperature and manages run_benchmark.sh based on thresholds.

set -e

WD="$HOME/AI-TOUR-GUIDE/benchmarking"
BENCH_SH="$WD/run_benchmark.sh"
LOG_DIR="$WD/logs"
LOG_FILE="$LOG_DIR/thermal_watchdog_$(date +%Y%m%d_%H%M%S).log"

mkdir -p "$LOG_DIR"

# Thresholds (°C)
MAX_TEMP=90     # pause benchmark above this
RESUME_TEMP=55  # resume benchmark below this

# How often to sample (seconds)
INTERVAL=60

log() {
  echo "[$(date '+%Y-%m-%d %H:%M:%S')] $1" | tee -a "$LOG_FILE"
}

get_temp() {
  # Use vcgencmd; fall back to sysfs if needed
  if command -v vcgencmd >/dev/null 2>&1; then
    vcgencmd measure_temp | awk -F"=" '{print $2}' | tr -d "C'"
  elif [ -f /sys/class/thermal/thermal_zone0/temp ]; then
    awk '{printf "%.1f\n", $1/1000}' /sys/class/thermal/thermal_zone0/temp
  else
    echo "0"
  fi
}

is_benchmark_running() {
  pgrep -f "$BENCH_SH" >/dev/null 2>&1 || pgrep -f "benchmark_orchestrator.py" >/dev/null 2>&1
}

start_benchmark() {
  if is_benchmark_running; then
    log "Benchmark already running; not starting a second instance."
    return
  fi
  log "Starting benchmark: $BENCH_SH"
  cd "$WD"
  nohup "$BENCH_SH" >> "$LOG_DIR/benchmark_autorun.log" 2>&1 &
}

stop_benchmark() {
  log "Stopping benchmark processes (run_benchmark.sh + orchestrator)..."
  pkill -f "$BENCH_SH" 2>/dev/null || true
  pkill -f "benchmark_orchestrator.py" 2>/dev/null || true
}

log "Thermal watchdog started."
log "Max temp: $MAX_TEMP°C, resume temp: $RESUME_TEMP°C, interval: ${INTERVAL}s"

# Optional: start benchmark immediately when watchdog starts
start_benchmark

while true; do
  TEMP="$(get_temp)"
  # If parsing failed, skip this cycle
  if ! echo "$TEMP" | grep -Eq '^[0-9]+(\.[0-9]+)?$'; then
    log "Warning: could not read temperature, got '$TEMP'"
    sleep "$INTERVAL"
    continue
  fi

  log "Current CPU temp: ${TEMP}°C"

  if (( $(echo "$TEMP > $MAX_TEMP" | bc -l) )); then
    if is_benchmark_running; then
      log "Temp ${TEMP}°C > ${MAX_TEMP}°C: stopping benchmark to cool down."
      stop_benchmark
    fi
  elif (( $(echo "$TEMP < $RESUME_TEMP" | bc -l) )); then
    if ! is_benchmark_running; then
      log "Temp ${TEMP}°C < ${RESUME_TEMP}°C: safe to resume benchmark."
      start_benchmark
    fi
  fi

  sleep "$INTERVAL"
done

