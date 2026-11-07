#!/bin/bash
# Four-protocol Gungnir pipeline with fail-closed, run-scoped publication.
set -uo pipefail

ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
PYTHON=${PYTHON:-/home/wct/anaconda3/envs/wct/bin/python}
RESULT_ROOT=${RESULT_ROOT:-/home/wct/Gungnir_result}
FRP_ROOT=${FRP_ROOT:-/home/wct/FRP-watcher/server_gungnir}
UPLOAD_SCRIPT=${UPLOAD_SCRIPT:-/home/wct/upload/run.sh}
LOCAL_IPV6=${LOCAL_IPV6:-2408:400a:76a:b00::9}
INTERFACE=${INTERFACE:-eth0}
ZMAP_BIN=${ZMAP_BIN:-zmap}
DNS_DOMAIN=${DNS_DOMAIN:-example.com}
SCAN_RATE=${SCAN_RATE:-0}
BGP_RATE=${BGP_RATE:-10}
AS_RATE=${AS_RATE:-0}
BATCH_TARGETS=${BATCH_TARGETS:-0}
ZMAP_WORKERS=${ZMAP_WORKERS:-2}
RESPONSE_THRESHOLD=${RESPONSE_THRESHOLD:-0.75}
TIMEOUT_MODEL=${TIMEOUT_MODEL:-7200}
TIMEOUT_ACTIVE=${TIMEOUT_ACTIVE:-21600}
TIMEOUT_TRAP=${TIMEOUT_TRAP:-43200}
SKIP_GENERATION=${SKIP_GENERATION:-0}
USE_LATEST_SNAPSHOTS=${USE_LATEST_SNAPSHOTS:-0}
UPLOAD_RESULTS=${UPLOAD_RESULTS:-0}
TASK_DATE=${RESULT_DATE:-$(date +%F)}
RUN_ID=${RUN_ID:-$(date -u +%Y%m%dT%H%M%SZ)-$$}
WORK_DIR="$RESULT_ROOT/$RUN_ID"
LOG_FILE="$WORK_DIR/run.log"
PROTOS=(ICMPv6 TCP80 TCP443 UDP53)
declare -A SOURCES

mkdir -p "$WORK_DIR"
exec > >(tee -a "$LOG_FILE") 2>&1
exec 9>/run/lock/gungnir_run_week.lock
if ! flock -n 9; then
    echo "[$(date '+%F %T')] another run_week.sh is still running; skip"
    exit 0
fi

log() { echo "[$(date '+%F %T')] $*"; }
fail() { log "ERROR: $*"; exit 1; }

run_required() {
    local stage=$1 timeout_seconds=$2
    shift 2
    log "START $stage: $*"
    if [ "$timeout_seconds" = 0 ]; then
        "$@"
    else
        timeout --signal=TERM --kill-after=60 "$timeout_seconds" "$@"
    fi
    local rc=$?
    if [ "$rc" -ne 0 ]; then
        fail "stage=$stage rc=$rc; no partial four-protocol result will be published"
    fi
    log "DONE $stage"
}

protocol_config() {
    case "$1" in
        ICMPv6) SCAN_PROTOCOL=icmp; SCAN_PORT=0 ;;
        TCP80) SCAN_PROTOCOL=tcp; SCAN_PORT=80 ;;
        TCP443) SCAN_PROTOCOL=tcp; SCAN_PORT=443 ;;
        UDP53) SCAN_PROTOCOL=udp; SCAN_PORT=53 ;;
        *) fail "unknown protocol $1" ;;
    esac
}

latest_snapshot() {
    local proto=$1
    find "$FRP_ROOT/$proto" -maxdepth 1 -type f \
        -name "????-??-??-$proto.txt" -print | sort | tail -n 1
}

publish_file() {
    local source=$1 destination=$2 temporary
    temporary=$(mktemp "$(dirname "$destination")/.$(basename "$destination").tmp.XXXXXX")
    if ! cp "$source" "$temporary" || ! chmod 0644 "$temporary" || ! mv "$temporary" "$destination"; then
        rm -f "$temporary"
        fail "cannot atomically publish $destination"
    fi
}

if [ "$USE_LATEST_SNAPSHOTS" = 1 ]; then
    SKIP_GENERATION=1
fi
[ -x "$PYTHON" ] || fail "Python is not executable: $PYTHON"
[ -f "$ROOT/data/latestv6.dat" ] || fail "missing BGP database: $ROOT/data/latestv6.dat"
cd "$ROOT" || fail "cannot enter $ROOT"
log "BEGIN run_id=$RUN_ID date=$TASK_DATE threshold=$RESPONSE_THRESHOLD snapshots=$USE_LATEST_SNAPSHOTS"

if [ "$SKIP_GENERATION" != 1 ]; then
    for proto in "${PROTOS[@]}"; do
        run_required "make_Prediction $proto" "$TIMEOUT_MODEL" \
            "$PYTHON" "$ROOT/make_Prediction.py" --target_type "$proto" \
            --seedfile "$ROOT/data/FRPseed/$proto.txt" \
            --Prediction_path "$ROOT/Prediction/Target_$proto.csv"
        run_required "Strategy $proto" "$TIMEOUT_MODEL" \
            "$PYTHON" "$ROOT/Strategy.py" --target_type "$proto" \
            --seedfile "$ROOT/data/FRPseed/$proto.txt" \
            --Prediction_path "$ROOT/Prediction/Target_$proto.csv" \
            --genarate_FRP_path "$ROOT/Prediction/PredictionFRP_$proto.txt"
    done
fi

# Freeze all four inputs before scanning so later publications cannot change this run.
for proto in "${PROTOS[@]}"; do
    if [ "$USE_LATEST_SNAPSHOTS" = 1 ]; then
        SOURCES[$proto]=$(latest_snapshot "$proto")
    else
        SOURCES[$proto]="$ROOT/Prediction/PredictionFRP_$proto.txt"
    fi
    [ -s "${SOURCES[$proto]}" ] || fail "missing or empty $proto input: ${SOURCES[$proto]}"
    log "INPUT $proto ${SOURCES[$proto]} sha256=$(sha256sum "${SOURCES[$proto]}" | awk '{print $1}')"
done

for proto in "${PROTOS[@]}"; do
    protocol_config "$proto"
    run_dir="$WORK_DIR/$proto"
    generated_dir="$run_dir/generated"
    zmap_dir="$run_dir/zmap"
    mkdir -p "$generated_dir" "$zmap_dir"
    source_file=${SOURCES[$proto]}
    active_input="$run_dir/input.txt"

    if [ "$USE_LATEST_SNAPSHOTS" = 1 ] || [ "${SHORTEN_INPUTS:-0}" = 1 ]; then
        run_required "shortest input $proto" 3600 \
            "$PYTHON" "$ROOT/tools/select_shortest_prefixes.py" \
            --input "$source_file" --output "$active_input"
    else
        cp "$source_file" "$active_input" || fail "cannot snapshot $proto input"
    fi

    active_file="$run_dir/active.txt"
    run_required "active $proto" "$TIMEOUT_ACTIVE" \
        "$PYTHON" "$ROOT/ipv6_bulk_zmap_scan_nibble.py" \
        --input "$active_input" --output-frp "$active_file" \
        --out-json "$run_dir/active.json" --run-id "$RUN_ID" \
        --protocol "$SCAN_PROTOCOL" --port "$SCAN_PORT" \
        --interface "$INTERFACE" --local-ipv6 "$LOCAL_IPV6" --zmap-bin "$ZMAP_BIN" \
        --dns-domain "$DNS_DOMAIN" --bgp-dat "$ROOT/data/latestv6.dat" \
        --generated-folder "$generated_dir" --zmap-folder "$zmap_dir" \
        --rate "$SCAN_RATE" --bgp-rate "$BGP_RATE" --as-rate "$AS_RATE" \
        --batch-targets "$BATCH_TARGETS" --workers "$ZMAP_WORKERS" \
        --response-threshold "$RESPONSE_THRESHOLD"

    trap_file="$run_dir/trap-full.txt"
    if [ -s "$active_file" ]; then
        cp "$active_file" "$trap_file" || fail "cannot initialize $proto trap output"
        run_required "trap $proto" "$TIMEOUT_TRAP" \
            "$PYTHON" "$ROOT/trap_det/trap.py" \
            --predict_path "$active_file" --bgp-dat "$ROOT/data/latestv6.dat" \
            --local_ipv6 "$LOCAL_IPV6" --generated_folder "$generated_dir" \
            --zmap_folder "$zmap_dir" --outputfrp "$trap_file" \
            --protocol "$SCAN_PROTOCOL" --port "$SCAN_PORT" --dns-domain "$DNS_DOMAIN" \
            --response-threshold "$RESPONSE_THRESHOLD" --batch-rate "$SCAN_RATE" \
            --bgp-rate "$BGP_RATE" --as-rate "$AS_RATE" --batch-targets "$BATCH_TARGETS"
    else
        : > "$trap_file"
        log "SKIP trap $proto: active result is empty"
    fi

    run_required "shortest output $proto" 3600 \
        "$PYTHON" "$ROOT/tools/select_shortest_prefixes.py" \
        --input "$trap_file" --output "$run_dir/final.txt"
    log "COUNTS $proto input=$(wc -l < "$active_input") active=$(wc -l < "$active_file") trap=$(wc -l < "$trap_file") final=$(wc -l < "$run_dir/final.txt")"
done

# Publication starts only after every protocol has completed successfully.
for proto in "${PROTOS[@]}"; do
    destination="$FRP_ROOT/$proto/$TASK_DATE-$proto.txt"
    mkdir -p "$(dirname "$destination")"
    publish_file "$WORK_DIR/$proto/final.txt" "$destination"
    log "PUBLISHED $proto $destination"
done

if [ "$UPLOAD_RESULTS" = 1 ]; then
    [ -x "$UPLOAD_SCRIPT" ] || fail "upload script is not executable: $UPLOAD_SCRIPT"
    run_required "upload four-protocol FRP" 900 bash "$UPLOAD_SCRIPT"
fi

log "COMPLETE run_id=$RUN_ID work_dir=$WORK_DIR uploaded=$UPLOAD_RESULTS"
