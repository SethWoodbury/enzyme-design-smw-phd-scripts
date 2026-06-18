#!/usr/bin/env bash
# =============================================================================
# MSA Builder (A3M) using HHblits (+ hhfilter) with UniRef30, fallback to BFD
# -----------------------------------------------------------------------------
# Usage:
#   ./build_msa.sh <in_fasta> <out_dir> <CPU> <MAXMEM_MB>
# Example:
#   ./build_msa.sh input.fasta out_msa 8 32000
#
# Notes:
# - MAXMEM is in **MB** for hhblits (-maxmem).
# - Requires HHsuite (hhblits, hhfilter) in PATH (or adjust HHLIB below).
# - Exits early if not on a "DB" node (in practice: if /local/databases AND
#   /databases are not present). Adjust policy as needed.
# - Core logic matches your original script.
# =============================================================================
#
# =============================================================================
# QUICK SUBMIT REFERENCE (Slurm one-liner)
# -----------------------------------------------------------------------------
# Submit this script as a job on a DB node, allocate CPUs/memory, and run:
#
# sbatch \
#   -J msa_job \
#   -C DB -p <your_partition> -t 00:20:00 -c 8 --mem=32G \
#   -o msa.%j.out -e msa.%j.err \
#   --wrap "module load hhsuite; /home/woodbuse/special_scripts/msa_tools/make_msa__CLEAN.sh /path/input.fasta /path/out_msa 8 32000"
#
# Notes:
#   - Replace <your_partition> with the correct partition on your cluster.
#   - -C DB requests nodes tagged with the DB feature (cluster-specific).
#   - -c 8 allocates 8 CPUs; pass the same "8" as the script's 3rd arg.
#   - --mem=32G allocates 32 GB; pass "32000" (MB) as the script's 4th arg.
#   - Update /path/input.fasta and /path/out_msa to your real paths.
#   - "module load hhsuite" can be removed if hhblits/hhfilter are already in PATH.
# =============================================================================

set -euo pipefail

# -------- Pretty logging helpers ------------------------------------------------
log()   { printf "[%s] %s\n" "$(date +'%F %T')" "$*"; }
warn()  { printf "\n\033[33m[WARN %s]\033[0m %s\n\n" "$(date +'%F %T')" "$*"; }
err()   { printf "\n\033[31m[ERROR %s]\033[0m %s\n\n" "$(date +'%F %T')" "$*" >&2; }
die()   { err "$*"; exit 1; }

# Optional: echo commands (comment to disable)
# set -x

# -------- Args ------------------------------------------------------------------
if [[ $# -lt 4 ]]; then
  cat <<'USAGE' >&2
Usage:
  build_msa.sh <in_fasta> <out_dir> <CPU> <MAXMEM_MB>

Example:
  build_msa.sh input.fasta out_msa 8 32000
USAGE
  exit 2
fi

in_fasta="$1"
out_dir="$2"
CPU="$3"           # threads -> hhblits -cpu
MEM="$4"           # MB      -> hhblits -maxmem

[[ -f "$in_fasta" ]] || die "Input FASTA not found: $in_fasta"
[[ "$CPU" =~ ^[0-9]+$ ]] || die "CPU must be an integer (got: $CPU)"
[[ "$MEM" =~ ^[0-9]+$  ]] || die "MAXMEM_MB must be an integer MB (got: $MEM)"

# -------- Environment / PATH ----------------------------------------------------
# If HHsuite is not already in PATH, set HHLIB here:
HHLIB_DEFAULT="/software/hhsuite/build/bin"
if ! command -v hhblits >/dev/null 2>&1; then
  if [[ -x "$HHLIB_DEFAULT/hhblits" ]]; then
    export PATH="$HHLIB_DEFAULT:$PATH"
  fi
fi

# Confirm tools exist
command -v hhblits  >/dev/null 2>&1 || die "hhblits not found in PATH"
command -v hhfilter >/dev/null 2>&1 || die "hhfilter not found in PATH"
command -v grep     >/dev/null 2>&1 || die "grep not found"
command -v realpath >/dev/null 2>&1 || warn "realpath not found (not critical)"

# -------- Databases / DB-node policy -------------------------------------------
# Your cluster mounts DBs under /local/databases or /databases on "DB" nodes.
PIPE_DIR="/local/databases"
if [[ ! -d "$PIPE_DIR" ]]; then
  PIPE_DIR="/databases"
  if [[ ! -d "$PIPE_DIR" ]]; then
    warn "############################################################################################
###   WARNING! Likely NOT on a DB node (no /local/databases or /databases present).        ###
###   Running here would degrade performance for everyone; exiting by policy.              ###
############################################################################################"
    exit 1
  else
    warn "Using PIPE_DIR=$PIPE_DIR (non-local path); ensure you're on a DB node (-C DB)."
  fi
fi

# Databases (adjust paths if different on your cluster)
DB_UR30="$PIPE_DIR/uniclust/UniRef30_2022_02"
DB_BFD="$PIPE_DIR/bfd/bfd_metaclust_clu_complete_id30_c90_final_seq.sorted_opt"

[[ -e "${DB_UR30}_hhm.ffdata" || -e "${DB_UR30}.cs219" || -e "${DB_UR30}" ]] || \
  warn "UniRef30 DB path looks odd: $DB_UR30 (ensure correct prefix for hhblits)"
[[ -e "${DB_BFD}_hhm.ffdata"  || -e "${DB_BFD}.cs219"  || -e "${DB_BFD}"  ]] || \
  warn "BFD DB path looks odd: $DB_BFD (ensure correct prefix for hhblits)"

# -------- Output layout ---------------------------------------------------------
mkdir -p "$out_dir"
mkdir -p "$out_dir/hhblits"
tmp_dir="$out_dir/hhblits"
out_prefix="$out_dir/t000_"

log "Input FASTA : $in_fasta"
log "Output dir  : $out_dir"
log "Threads     : $CPU"
log "MaxMem (MB) : $MEM"
log "DB root     : $PIPE_DIR"
log "UniRef30    : $DB_UR30"
log "BFD         : $DB_BFD"
log "tmp_dir     : $tmp_dir"
log "out_prefix  : $out_prefix"

# -------- SignalP block (disabled) ---------------------------------------------
# If you later enable SignalP, write outputs to $tmp_dir/signalp and set trim_fasta accordingly.
# For now, we use the input FASTA directly (matches your working behavior).
trim_fasta="$in_fasta"

# -------- HHblits command templates --------------------------------------------
HHBLITS_COMMON="-o /dev/null -mact 0.35 -maxfilt 100000000 -neffmax 20 -cov 25 -cpu $CPU -nodiff -realign_max 100000000 -maxseq 1000000 -maxmem $MEM -n 4"
HHBLITS_UR30="hhblits $HHBLITS_COMMON -d $DB_UR30"
HHBLITS_BFD="hhblits $HHBLITS_COMMON -d $DB_BFD"

# -------- Helper: count sequences in A3M ---------------------------------------
count_a3m() {
  # counts '>' headers in an a3m; returns 0 if file missing
  local f="$1"
  [[ -s "$f" ]] || { echo 0; return; }
  grep -c '^>' "$f" || echo 0
}

# -------- Main pipeline ---------------------------------------------------------
if [[ ! -s "${out_prefix}.msa0.a3m" ]]; then
  prev_a3m="$trim_fasta"

  # 1) Iterative searches against UniRef30
  for e in 1e-10 1e-6 1e-3; do
    log "Running HHblits vs UniRef30 (e=$e)"
    a3m_raw="$tmp_dir/t000_.${e}.a3m"
    a3m_cov75="$tmp_dir/t000_.${e}.id90cov75.a3m"
    a3m_cov50="$tmp_dir/t000_.${e}.id90cov50.a3m"

    if [[ ! -s "$a3m_raw" ]]; then
      # -v 0 (quiet); remove -v 0 to see HHblits chatter
      $HHBLITS_UR30 -i "$prev_a3m" -oa3m "$a3m_raw" -e "$e" -v 0
    else
      log "Reuse existing: $a3m_raw"
    fi

    # hhfilter at two coverage thresholds
    hhfilter -maxseq 100000 -id 90 -cov 75 -i "$a3m_raw" -o "$a3m_cov75"
    hhfilter -maxseq 100000 -id 90 -cov 50 -i "$a3m_raw" -o "$a3m_cov50"

    # Update previous a3m for the next iteration (more permissive set)
    prev_a3m="$a3m_cov50"

    # Count sequences
    n75=$(count_a3m "$a3m_cov75")
    n50=$(count_a3m "$a3m_cov50")
    log "Counts for e=$e → cov75: $n75 | cov50: $n50"

    # Select output if thresholds hit
    if (( n75 > 2000 )); then
      if [[ ! -s "${out_prefix}.msa0.a3m" ]]; then
        cp -v "$a3m_cov75" "${out_prefix}.msa0.a3m"
        log "Selected cov75 set (n=$n75) for msa0; stopping UniRef sweep."
        break
      fi
    elif (( n50 > 4000 )); then
      if [[ ! -s "${out_prefix}.msa0.a3m" ]]; then
        cp -v "$a3m_cov50" "${out_prefix}.msa0.a3m"
        log "Selected cov50 set (n=$n50) for msa0; stopping UniRef sweep."
        break
      fi
    fi
  done

  # 2) Fallback to BFD if UniRef30 didn’t reach thresholds
  if [[ ! -s "${out_prefix}.msa0.a3m" ]]; then
    e="1e-3"
    log "Running HHblits vs BFD (e=$e) — UniRef thresholds not met."
    a3m_raw="$tmp_dir/t000_.${e}.bfd.a3m"
    a3m_cov75="$tmp_dir/t000_.${e}.bfd.id90cov75.a3m"
    a3m_cov50="$tmp_dir/t000_.${e}.bfd.id90cov50.a3m"

    if [[ ! -s "$a3m_raw" ]]; then
      $HHBLITS_BFD -i "$prev_a3m" -oa3m "$a3m_raw" -e "$e" -v 0
    else
      log "Reuse existing: $a3m_raw"
    fi

    hhfilter -maxseq 100000 -id 90 -cov 75 -i "$a3m_raw" -o "$a3m_cov75"
    hhfilter -maxseq 100000 -id 90 -cov 50 -i "$a3m_raw" -o "$a3m_cov50"

    n75=$(count_a3m "$a3m_cov75")
    n50=$(count_a3m "$a3m_cov50")
    log "BFD counts → cov75: $n75 | cov50: $n50"

    if (( n75 > 2000 )); then
      cp -v "$a3m_cov75" "${out_prefix}.msa0.a3m"
      log "Selected BFD cov75 set for msa0."
    elif (( n50 > 4000 )); then
      cp -v "$a3m_cov50" "${out_prefix}.msa0.a3m"
      log "Selected BFD cov50 set for msa0."
    else
      log "BFD thresholds not reached; proceeding to final fallback."
    fi
  fi

  # 3) Final fallback — at least provide something
  if [[ ! -s "${out_prefix}.msa0.a3m" ]]; then
    cp -v "$prev_a3m" "${out_prefix}.msa0.a3m"
    warn "Thresholds not met; used last available A3M as msa0: ${out_prefix}.msa0.a3m"
  fi
else
  log "Output already exists: ${out_prefix}.msa0.a3m (skipping)"
fi

log "Done. Output: ${out_prefix}.msa0.a3m"
