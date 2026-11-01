#!/bin/bash
# build_t3.sh v3 — eval pour que les backslash-quotes du flag QUANTS soient re-interpretes
set -e
H2LIB=/home/guile/llama-experiments/h2/cupti129-test/e2/h2lib
T3=/home/guile/llama-experiments/h2/cupti129-test/e2/t3port/ggml-cuda
BD="$H2LIB/build-clean/_deps/llama_cpp-build/ggml/src/ggml-cuda"
cd "$BD"

# ligne nvcc de reference (compile_h2.sh, prouve) sans le -c <src> -o <dst> de fin
REF=$(grep nvcc "$H2LIB/compile_h2.sh" | sed 's/ -x cu -c .*//')

compile_one() {
  # $1 = source .cu, $2 = output .o
  eval "$REF -x cu -c \"$1\" -o \"$2\""
}

# 1) fattn.cu (H2 + dispatch T3)
compile_one "$T3/fattn.cu" "$H2LIB/fattn.t3.o"
echo "fattn.t3.o OK"

# 2) 6 instances T3
for f in "$T3"/template-instances/fattn-mma-volta-small-instance-*.cu; do
  out=$(basename "$f" .cu)
  compile_one "$f" "$H2LIB/$out.o"
  echo "$out.o OK"
done

# 3) rsp: fattn.h2.o -> fattn.t3.o + ajouter les 6 instances
python3 - << "PYEOF"
H2LIB = "/home/guile/llama-experiments/h2/cupti129-test/e2/h2lib"
rsp = open(H2LIB + "/objects_h2.rsp").read()
assert H2LIB + "/fattn.h2.o" in rsp
rsp = rsp.replace(H2LIB + "/fattn.h2.o", H2LIB + "/fattn.t3.o")
inst = " ".join('"%s/%s.o"' % (H2LIB, x) for x in [
    "fattn-mma-volta-small-instance-q8_0-q8_0-d64",
    "fattn-mma-volta-small-instance-q8_0-q8_0-d128",
    "fattn-mma-volta-small-instance-q8_0-q8_0-d256",
    "fattn-mma-volta-small-instance-f16-f16-d64",
    "fattn-mma-volta-small-instance-f16-f16-d128",
    "fattn-mma-volta-small-instance-f16-f16-d256"])
rsp = rsp.rstrip()
assert rsp.endswith('"')
rsp = rsp + " " + inst
open(H2LIB + "/objects_t3.rsp", "w").write(rsp)
print("objects_t3.rsp OK:", rsp.count('.o'))
PYEOF

# 4) link
/home/guile/llama-experiments/tools/gcc-13-bin/g++ -fPIC -shared -o "$H2LIB/lib/libggml-cuda-t3.so" \
  @"$H2LIB/objects_t3.rsp" @CMakeFiles/ggml-cuda.dir/linkLibs.rsp \
  -L"/home/guile/llama-experiments/tools/cuda-12.6/usr/local/cuda-12.6/targets/x86_64-linux/lib/stubs" \
  -L"/home/guile/llama-experiments/tools/cuda-12.6/usr/local/cuda-12.6/targets/x86_64-linux/lib"
echo "BUILD T3 OK"
ls -la "$H2LIB/lib/libggml-cuda-t3.so"
sha256sum "$H2LIB/lib/libggml-cuda-t3.so"
