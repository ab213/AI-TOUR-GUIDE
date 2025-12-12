#!/bin/bash
# run_benchmark.sh - Enhanced with nohup + error recovery
# Survives SSH disconnect, handles 12+ hour runs
# ONLY runs orchestrator + rd-usb; analysis is optional post-processing

set -e

# ============================================================================
# CONFIGURATION
# ============================================================================

# Default behavior: resume enabled & no cleanup after model
NO_RESUME=0
CLEANUP_AFTER_MODEL=0

# Parse flags
while [[ $# -gt 0 ]]; do
  case $1 in
    --no-resume)
      NO_RESUME=1
      shift
      ;;
    --cleanup-after-model)
      CLEANUP_AFTER_MODEL=1
      shift
      ;;
    *)
      echo "Unknown option: $1" >&2
      echo "Usage: $0 [--no-resume] [--cleanup-after-model]" >&2
      exit 1
      ;;
  esac
done

VENV_PY="/home/pi/morph_env/.venv/bin/python"
WD="$HOME/AI-TOUR-GUIDE/benchmarking"
LOG_DIR="$WD/logs"
LOG_FILE="$LOG_DIR/$(date +%Y%m%d_%H%M%S).log"
PY_SCRIPT="benchmark_orchestrator.py"
RD_USB_LOG="$LOG_DIR/rd_usb.log"
RD_USB_DIR="$HOME/rd-usb"
RD_USB_PIDFILE="$LOG_DIR/rd_usb.pid"
PY_PIDFILE="$LOG_DIR/orchestrator.pid"

# Create log directory
mkdir -p "$LOG_DIR"

# ============================================================================
# LOGGING
# ============================================================================
log() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $1" | tee -a "$LOG_FILE"
}

log_error() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] ✗ ERROR: $1" | tee -a "$LOG_FILE"
}

log_success() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] ✓ SUCCESS: $1" | tee -a "$LOG_FILE"
}

# ============================================================================
# CLEANUP ON EXIT/INTERRUPT
# ============================================================================
cleanup() {
    log "Cleaning up..."
    if [ -f "$RD_USB_PIDFILE" ]; then
        RD_PID=$(cat "$RD_USB_PIDFILE")
        log "Stopping rd-usb (PID $RD_PID)..."
        kill $RD_PID 2>/dev/null || true
        rm "$RD_USB_PIDFILE"
    fi
    if [ -f "$PY_PIDFILE" ]; then
        PY_PID=$(cat "$PY_PIDFILE")
        log "Orchestrator was PID $PY_PID"
        rm "$PY_PIDFILE"
    fi
    log_success "Cleanup complete"
}

trap cleanup EXIT INT TERM

# ============================================================================
# PREFLIGHT CHECKS
# ============================================================================
log "Starting benchmark orchestrator..."
log "Log file: $LOG_FILE"
log ""

if [ ! -f "benchmark_config.json" ]; then
    log_error "benchmark_config.json not found!"
    exit 1
fi

if [ ! -f "$PY_SCRIPT" ]; then
    log_error "$PY_SCRIPT not found!"
    exit 1
fi

if [ ! -d "$RD_USB_DIR" ]; then
    log_error "rd-usb directory not found at $RD_USB_DIR!"
    exit 1
fi

log "✓ Preflight checks passed"
log ""

# ============================================================================
# START RD-USB (Power Monitoring Daemon)
# ============================================================================
log "Starting rd-usb daemon (nohup, survives SSH disconnect)..."
cd "$RD_USB_DIR"
nohup "$VENV_PY" web.py --on-receive "$WD/record_pwr.py" --on-receive-interval 0 > "$RD_USB_LOG" 2>&1 &
RD_PID=$!
echo $RD_PID > "$RD_USB_PIDFILE"
sleep 2

if ! ps -p $RD_PID > /dev/null; then
    log_error "Failed to start rd-usb"
    cat "$RD_USB_LOG" >> "$LOG_FILE"
    exit 1
fi

log "✓ rd-usb started (PID $RD_PID)"
log "  Power data: $WD/results/rd_usb_samples.csv"
log ""

# ============================================================================
# MAIN ORCHESTRATOR RUN (with error recovery + retry logic)
# ============================================================================
cd "$WD"
MAX_RETRIES=3
RETRY_COUNT=0

while [ $RETRY_COUNT -lt $MAX_RETRIES ]; do
    log "Attempt $((RETRY_COUNT + 1))/$MAX_RETRIES: Starting orchestrator..."
    if [ "$NO_RESUME" -eq 1 ]; then
        log "Resume: DISABLED (fresh run, ignoring previous results.csv)"
    else
        log "Resume: ENABLED (will skip completed runs if results.csv exists)"
    fi
    log ""

    # Run with nohup so it survives SSH disconnect
    NO_RESUME="$NO_RESUME" nohup "$VENV_PY" "$PY_SCRIPT" > >(tee -a "$LOG_FILE") 2>&1 &
    PY_PID=$!
    echo $PY_PID > "$PY_PIDFILE"
    log "Orchestrator PID: $PY_PID"
    log ""

    if wait $PY_PID; then
        log_success "Orchestrator completed successfully"
        log ""
        break
    else
        EXIT_CODE=$?
        log_error "Orchestrator failed with exit code $EXIT_CODE"
        RETRY_COUNT=$((RETRY_COUNT + 1))

        if [ $RETRY_COUNT -lt $MAX_RETRIES ]; then
            log "Waiting 60s before retry..."
            sleep 60
            
            log "Clearing Python cache..."
            find "$WD" -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
            find "$WD" -name "*.pyc" -delete 2>/dev/null || true
            
            log "Checking system health..."
            df -h / | tail -1 | awk '{print "  Disk: " $4 " free"}'
            free -h | grep "^Mem" | awk '{print "  RAM: " $7 " free"}'
            
            log "Attempting network restart..."
            sudo systemctl restart networking 2>/dev/null || true
            sleep 5
        fi
    fi
done

log ""

if [ $RETRY_COUNT -eq $MAX_RETRIES ]; then
    log_error "Orchestrator failed after $MAX_RETRIES attempts"
    log "Check logs:"
    log "  Main: $LOG_FILE"
    log "  rd-usb: $RD_USB_LOG"
    exit 1
fi

# ============================================================================
# SUCCESS: RESULTS READY
# ============================================================================
log "Benchmark run complete!"
log ""
log "Results location:"
log "  Data: $WD/results/results.csv"
log "  Logs: $LOG_DIR/"
log "  Power: $WD/results/rd_usb_samples.csv"
log ""
log "Next steps (optional):"
log "  1. Analyze results: $VENV_PY analyze_benchmark.py $WD/results/results.csv"
log "  2. View summary: cat $WD/benchmark_analysis/SUMMARY_REPORT.txt"
log ""
log_success "All systems operational. Benchmark data ready for analysis."

