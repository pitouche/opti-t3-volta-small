#!/bin/bash
# build_t3v2.sh — build un .so candidat T3 v2 SANS toucher h2lib ni la prod.
#
# Usage : build_t3v2.sh <variant> [STEP] [LB_MINBLOCKS]
#   variant       : nom du .so de sortie (libggml-cuda-<variant>.so dans h2lib/lib)
#   STEP          : override FATTN_VS_STEP (défaut = 128, source intacte)
#   LB_MINBLOCKS  : 2e arg de __launch_bounds__ (défaut = 2, source intacte)
#
# Méthode : la source de travail est t3v2/ggml-cuda (copie de t3port).
#   1) .cuh restauré depuis fattn-mma-volta-small.ORIG.cuh, overrides sed appliqués
#   2) fattn.cu + 6 instances recompilés depuis t3v2/ggml-cuda (mêmes flags nvcc 18/10)
#   3) objects_t3v2.rsp = objects_t3.rsp (150 .o) avec fattn.t3.o + 6 instances substitués
#   4) link g++-13 (linkLibs.rsp du tree build-clean 18/10)
set -euo pipefail
VARIANT="${1:?usage: build_t3v2.sh <variant> [STEP] [LB_MINBLOCKS] [CUD_CUH]}"
STEP="${2:-}"
LB="${3:-}"

H2LIB=/home/guile/llama-experiments/h2/cupti129-test/e2/h2lib
E2=/home/guile/llama-experiments/h2/cupti129-test/e2
CUD_CUH="${4:-$E2/t3v2/fattn-mma-volta-small.ORIG.cuh}"
T3="$E2/t3v2/ggml-cuda"
BD="$H2LIB/build-clean/_deps/llama_cpp-build/ggml/src/ggml-cuda"
OUT="$H2LIB/lib/libggml-cuda-$VARIANT.so"
RSP="/tmp/objects_t3v2-$VARIANT.rsp"

[ -f "$H2LIB/fattn.t3.o" ] || { echo "ABORT: fattn.t3.o de base manquant"; exit 2; }
md5prod=$(md5sum /home/guile/llama-experiments/ollama-v100/lib/ollama/cuda_v12/libggml-cuda.so | cut -d' ' -f1 | cut -c1-8)
echo "prod md5 (inchangé pendant le build) : $md5prod"

# 1) .cuh : source candidate (défaut : ORIG intacte) puis overrides STEP/LB
cp "$CUD_CUH" "$T3/fattn-mma-volta-small.cuh"
if [ -n "$STEP" ]; then
  sed -i "s/^#define FATTN_VS_STEP[[:space:]]*128/#define FATTN_VS_STEP     $STEP/" "$T3/fattn-mma-volta-small.cuh"
  grep -q "FATTN_VS_STEP     $STEP" "$T3/fattn-mma-volta-small.cuh" || { echo "ABORT: sed STEP échoué"; exit 2; }
fi
if [ -n "$LB" ]; then
  sed -i "s/__launch_bounds__(FATTN_VS_NTHREADS, 2)/__launch_bounds__(FATTN_VS_NTHREADS, $LB)/" "$T3/fattn-mma-volta-small.cuh"
  grep -q "__launch_bounds__(FATTN_VS_NTHREADS, $LB)" "$T3/fattn-mma-volta-small.cuh" || { echo "ABORT: sed LB échoué"; exit 2; }
fi
echo "variant=$VARIANT STEP=${STEP:-128} LB=${LB:-2}"

cd "$BD"
# nvcc args de la recette 18/10 (compile_h2.sh) : nvcc = 1er elem, le reste = flags (sans source ni -o)
. "$E2/t3v2/nvcc_args.sh"
NVCC_BIN="${NVCC_ARGS[0]}"
[ -x "$NVCC_BIN" ] || { echo "ABORT: nvcc manquant: $NVCC_BIN"; exit 2; }
ARGS=()
SKIP=0
for a in "${NVCC_ARGS[@]:1}"; do
  if [ "$SKIP" = 1 ]; then SKIP=0; continue; fi
  case "$a" in
    *.cu) continue;;
    -o) SKIP=1; continue;;
  esac
  ARGS+=("$a")
done
[ ${#ARGS[@]} -ge 60 ] || { echo "ABORT: NVCC_ARGS mal extrait (${#ARGS[@]})"; exit 2; }

compile_one() { # $1 src, $2 dst
  "$NVCC_BIN" "${ARGS[@]}" --ptxas-options=-v -x cu -c "$1" -o "$2"
}

OBJDIR=/tmp/t3v2-objs-$VARIANT
mkdir -p "$OBJDIR"
rm -f "$OBJDIR"/*.o

# 2) recompilation (ptxas -v pour les registres)
compile_one "$T3/fattn.cu" "$OBJDIR/fattn.o" 2> "$OBJDIR/fattn.ptxas"
echo "fattn.o OK"
for f in "$T3/template-instances/fattn-mma-volta-small-instance-"*.cu; do
  out=$(basename "$f" .cu)
  compile_one "$f" "$OBJDIR/$out.o" 2> "$OBJDIR/$out.ptxas"
  echo "$out.o OK"
done
echo "--- ptxas registres (instance q8_0 d256, variant $VARIANT) ---"
grep -E 'Used [0-9]+ registers' "$OBJDIR/fattn-mma-volta-small-instance-q8_0-q8_0-d256.ptxas" | head -4
grep -E 'spill' "$OBJDIR/fattn-mma-volta-small-instance-q8_0-q8_0-d256.ptxas" | head -2

# 3) rsp
python3 - "$RSP" "$OBJDIR" << 'PYEOF'
import sys
H2LIB = "/home/guile/llama-experiments/h2/cupti129-test/e2/h2lib"
rsp, objdir = sys.argv[1], sys.argv[2]
s = open(H2LIB + "/objects_t3.rsp").read()
inst = ["fattn-mma-volta-small-instance-%s" % x for x in [
  "q8_0-q8_0-d64","q8_0-q8_0-d128","q8_0-q8_0-d256",
  "f16-f16-d64","f16-f16-d128","f16-f16-d256"]]
s = s.replace('"%s/fattn.t3.o"' % H2LIB, '"%s/fattn.o"' % objdir)
for i in inst:
    s = s.replace('"%s/%s.o"' % (H2LIB, i), '"%s/%s.o"' % (objdir, i))
assert '"%s/fattn.o"' % objdir in s, "substitution fattn echouee"
n = s.count('.o"')
print("rsp OK : %d objets" % n)
open(rsp, "w").write(s)
PYEOF

# 4) link
/home/guile/llama-experiments/tools/gcc-13-bin/g++ -fPIC -shared -o "$OUT" \
  @"$RSP" @CMakeFiles/ggml-cuda.dir/linkLibs.rsp \
  -L"/home/guile/llama-experiments/tools/cuda-12.6/usr/local/cuda-12.6/targets/x86_64-linux/lib/stubs" \
  -L"/home/guile/llama-experiments/tools/cuda-12.6/usr/local/cuda-12.6/targets/x86_64-linux/lib"
echo "BUILD $VARIANT OK"
ls -la "$OUT" | awk '{print $5, $NF}'
md5sum "$OUT" | cut -d' ' -f1 | cut -c1-8
md5prod2=$(md5sum /home/guile/llama-experiments/ollama-v100/lib/ollama/cuda_v12/libggml-cuda.so | cut -d' ' -f1 | cut -c1-8)
[ "$md5prod" = "$md5prod2" ] && echo "prod inchangé : OK ($md5prod)" || { echo "!! PROD A CHANGE : $md5prod -> $md5prod2"; exit 3; }
