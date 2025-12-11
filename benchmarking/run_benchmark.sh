#!/bin/bash
set -e

# ============================================================================
# CONFIGURATION
# ============================================================================
WD="$HOME/AI-TOUR-GUIDE/benchmarking"
LOG_DIR="$WD/logs"
LOG_FILE="$LOG_DIR/$(date +%Y%m%d_%H%M%S).log"
PY_SCRIPT="benchmark_orchestrator.py"
RD_USB_LOG="$LOG_DIR/rd_usb.log"
RD_USB_DIR="$HOME/rd-usb"
RD_USB_PIDFILE="$LOG_DIR/rd_usb.pid"
PY_PIDFILE="$LOG_DIR/python.pid"

# Create log directory
mkdir -p "$LOG_DIR"

# ============================================================================
# LOGGING
# ============================================================================
log() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $1" | tee -a "$LOG_FILE"
}

log_error() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] ERROR: $1" | tee -a "$LOG_FILE"
}

log_success() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] SUCCESS: $1" | tee -a "$LOG_FILE"
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
        log "Python script PID was $PY_PID"
        rm "$PY_PIDFILE"
    fi
}

trap cleanup EXIT INT TERM

# ============================================================================
# PREFLIGHT CHECKS
# ============================================================================
log "Starting benchmark orchestrator..."

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

# ============================================================================
# START RD-USB
# ============================================================================
log "Starting rd-usb daemon..."
cd "$RD_USB_DIR"
python3 web.py --on-receive "$WD/record_pwr.py" --on-receive-interval 0 > "$RD_USB_LOG" 2>&1 &
RD_PID=$!
echo $RD_PID > "$RD_USB_PIDFILE"
sleep 2
if ! ps -p $RD_PID > /dev/null; then
    log_error "Failed to start rd-usb"
    cat "$RD_USB_LOG" >> "$LOG_FILE"
    exit 1
fi
log "✓ rd-usb started (PID $RD_PID)"

# ============================================================================
# MAIN BENCHMARK LOOP (with retries)
# ============================================================================
cd - > /dev/null
MAX_RETRIES=3
RETRY_COUNT=0

while [ $RETRY_COUNT -lt $MAX_RETRIES ]; do
    log "Attempt $((RETRY_COUNT + 1))/$MAX_RETRIES: Starting benchmark..."
    
    python3 "$PY_SCRIPT" > >(tee -a "$LOG_FILE") 2>&1 &
    PY_PID=$!
    echo $PY_PID > "$PY_PIDFILE"
    
    if wait $PY_PID; then
        log_success "Benchmark completed successfully"
        break
    else
        EXIT_CODE=$?
        log_error "Benchmark failed with exit code $EXIT_CODE"
        RETRY_COUNT=$((RETRY_COUNT + 1))
        
        if [ $RETRY_COUNT -lt $MAX_RETRIES ]; then
            log "Waiting 10s before retry..."
            sleep 10
            log "Restarting system services..."
            sudo systemctl restart networking 2>/dev/null || true
            sleep 5
        fi
    fi
done

if [ $RETRY_COUNT -eq $MAX_RETRIES ]; then
    log_error "Benchmark failed after $MAX_RETRIES attempts"
    log "Check logs at: $LOG_FILE and $RD_USB_LOG"
    exit 1
fi

log_success "All benchmarks completed!"
log "Results saved to $WD/benchmarking/results/results.csv"
log "Full log: $LOG_FILE"
