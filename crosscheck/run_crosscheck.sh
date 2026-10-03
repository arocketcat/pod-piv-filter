#!/bin/bash
# run_crosscheck.sh
# Feed the same TIFFs to OpenPIV and (optionally) PIVlab, then score everything
# against the exact truth.
#
#   pip install openpiv                          # tested with 0.26.1
#   python synth_piv.py --out data --tiff        # from the package root, ~2 min
#   bash crosscheck/run_crosscheck.sh            # OpenPIV only
#   PIVLAB_DIR=/path/to/PIVlab-3.09 bash crosscheck/run_crosscheck.sh   # + PIVlab
#
# PIVlab: use release 3.09 (git tag 3.09). Newer releases changed piv_FFTmulti to
# name=value arguments. It should run unmodified in MATLAB (not tested: only
# GNU Octave was available here). Octave 8 needs the two
# small patches listed in OCTAVE_PATCHES.txt plus `pkg load image`; set
# PIVLAB_ENGINE=octave. Octave's interp2 cannot do spline image deformation on a
# deformed grid, so the driver uses linear deformation (PIVlab's own default).
set -e
HERE="$(cd "$(dirname "$0")" && pwd)"
ROOT="$(dirname "$HERE")"
DATA="${POD_PIV_DATA:-$ROOT/data}"
T="$DATA/tiff"
OUT="$HERE/out"
mkdir -p "$OUT"
cd "$ROOT"

# 1. filtered TIFFs, exactly as a user would make them for another program
[ -d "$T/pod" ] || python pod_filter.py "$T/dirty/*_A.tif" "$T/dirty/*_B.tif" --out "$T/pod" --modes auto
[ -d "$T/minsub" ] || python "$HERE/make_minsub.py" "$T"

# 2. OpenPIV
for c in clean dirty minsub pod; do
  [ -f "$OUT/openpiv_$c.npz" ] || python "$HERE/openpiv_run.py" "$T/$c" "$OUT/openpiv_$c.npz" 0 999
done

# 3. PIVlab (optional)
if [ -n "$PIVLAB_DIR" ]; then
  ENGINE="${PIVLAB_ENGINE:-matlab}"
  for spec in clean:clean:0 dirty:dirty:0 pod:pod:0 minsub:minsub:0 dirty_highpass:dirty:2 dirty_clahe:dirty:1; do
    name=${spec%%:*}; rest=${spec#*:}; dir=${rest%%:*}; prep=${rest#*:}
    out="$OUT/pivlab_$name.mat"
    [ -f "$out" ] && continue
    cmd="addpath('$HERE'); pivlab_run('$PIVLAB_DIR','$T/$dir','$out',0,999,$prep)"
    if [ "$ENGINE" = octave ]; then
      [ "$prep" = 1 ] && { echo "skipping CLAHE: Octave has no adapthisteq"; continue; }
      octave-cli --no-gui --eval "$cmd"
    else
      matlab -batch "$cmd"
    fi
  done
fi

# 4. score
python "$HERE/score_all.py"
