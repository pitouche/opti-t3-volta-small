# T3 — kernel flash-attention decode VOLTA_SMALL (rows 1-8)

**Verdict : GO — en prod** (md5 prod actuel `d4954502` le contient, avec lba4+cvec+vldg).

## Question

Sur V100 (Volta, 100 SM, HBM2 ~794 GB/s mesurés), le decode de Qwen3.8-27B
(KV q8_0, 128 têtes Q8, GQA 8, d_head 128, ctx jusqu'à 196 k) est limité par le
kernel FA decode `moe_2` (`fattn_mma_v1_q8_q8_q8_2`), qui traite `rows` lignes
par warp-block avec un pipeline Q8 tensor-core (m16n8k16). Avec MTP, la plupart
des verify passent avec `rows = n_accepted+1 ∈ 1..8` → la quasi-totalité du decode
suit ce chemin. Le kernel générique est sous-occupé à faible `rows`.

## Hypothèse

Le PR anyei #153 (upstream llama.cpp, `fattn_mma_volta_small`, m8n8k4) cible
exactement les rows 1-8 avec des tiles 16x16, un double-buffer de Q, un
prefetch K/V et une accumulation fp16. Porté sur notre fork : il doit battre le
genérique sur rows 1-8 et rester compétitif sur rows==4 (où H2, rows 4-only,
était déjà). **Caveat du port** : le kernel #153 n'exploite pas les MMA V100
(les MMA Q8 16x16 n'existent pas sur sm_70) → on l'a adapté aux `mma.m8n8k4`
fp16 de Volta (les mêmes que H2, mais tiles 16 et 8 rows max).

## Résultats (MTP OFF, .so T3 `d2df07b8` vs baseline `a4b6f440`)

| ctx | decode baseline | decode T3 | delta |
|---|---|---|---|
| 114 k | 17,15 t/s | 19,41 t/s | **+13,2 %** |
| 196 k | 28,78 t/s | 37,18 t/s | **+29,2 %** |

Microbench FA decode (rows 1-8, d=128, Q8) : ×2.0 à ×2.2 à long ctx (ex. rows=4,
ctx 131 072 : baseline 52,8 µs/it → 23,8 µs/it ; rows=1 : 63,8 → 18,7).
**rows==4 : T3 bat aussi H2** (d'ou l'étalement H2 OFF, voir §Décision).

Microbench par variant (ctx 98 304, rows=1) — les 3 raffinements post-port :

| variant | µs | vs T3 de base |
|---|---|---|
| T3 (2026-10-26) | 14,55 | — |
| + lba4 (`__launch_bounds__(32,2)`) | 13,56 | +6,8 % |
| + cvec (load K/V 128-bit) | 13,00 | +10,7 % |
| + vldg (double-buffer V, 2 blocks/SM) | 12,65 | +13,0 % |

E2E cumul des raffinements (MTP OFF, 98 k, .so `4fd152f9` vs `d2df07b8`) :
+1,7 % (lba4), +3,5 % (cvec), +2,8 % @196 k (vldg).

**Décision H2 OFF** (après T3 en prod) : T3 couvre rows 1-8, donc le chemin rows==4
de H2 est remplacé par T3 ; mesure .so T3 (H2 ON vs OFF) : **−15 à −17 % avec H2 ON**
(le rows==4 d'H2 est moins rapide que T3 rows==4) → `GGML_CUDA_VOLTA_Q8_FATTN_TC=0`.
Le gain n'est pas « T3 vs H2 » mais « pas de conflit de dispatch + T3 partout 1-8 ».
Mission H2 complète (protocoles, A/B du GO, CUPTI, journal) : repo `opti-h2`.

## Contenu

```
src/
├── fattn.cu                    # fork + gate rows 1-8 → fattn_mma_volta_small
├── fattn-mma-volta-small.cuh   # kernel m8n8k4 (port anyei #153 adapté Volta)
├── fattn-mma-volta-small.lba4.cuh   # variant launch_bounds(32,2) [déploied]
├── fattn-mma-volta-small.cvec.cuh   # variant load K 128-bit  [déploied]
├── fattn-mma-volta-small.vldg.cuh   # variant double-buffer V [déploied]
├── template-instances/fattn-mma-volta-small-instance-*.cu  (q8_0 d64/d128/d256)
└── build_t3.sh / build_t3v2.sh / nvcc_args.sh   # build du .so
results/                        # données brutes (jsonl/txt/verdicts ci-dessous)
bench/                          # microbench + benchs e2e + deploy scripts
```

## `.so` associés (md5 — voir `sofiles/` du meta-repo)

| md5 | état |
|---|---|
| `d2df07b8` | T3 de base (premier port, 26/10) |
| `fa3e75c2` | T3 + lba4 (24/10, avant rebase — historique) |
| `feaf05aa` | T3 + lba4 prod (25/10, premier déploiement prod) |
| `b014d026` | T3 + lba4 + cvec (26/10) |
| `4fd152f9` | T3 + lba4 + cvec + vldg (28/10) — dernier avant FA16 |
| `d4954502` | = `4fd152f9` + FA16-compact — **prod actuel** |

## Re-validation

1. **Microbench** (le plus rapide, sans service) :
   `python3 bench/mb_fa_bench.py --so <md5:.so> --rows 1 2 3 4 5 6 7 8 --ctx 98304`
   → compare à `results/mb_fa_baseline_d2df_20261022.jsonl` (baseline) et
   `results/mb_fa_t3v2_results.jsonl` (variant). Attention : le microbench ne
   capture pas l'interaction MTP (voir `results/fa_step_INVALID_note.md`).
2. **E2E MTP OFF** (gold standard kernel) :
   `python3 bench/bench/supervisor_bench.py run bench/t3_bench.py --detach`
   → compare `results/t3_bench_results.jsonl`.
3. **E2E MTP ON prod** : `t3_vldg_mtp_bench.py` (draft acceptance identique
   entre legs, obligatoire).

## Fichiers de référence

- `results/t3_bench_summary.md` — bench e2e T3 vs baseline (17/10)
- `results/t3_h2_bench_summary.md` — T3 vs H2 rows 1-8
- `results/t1t3_h2_bench_summary.md` — 3-way (H2 / T3 / T1+T3)
- `results/t3_grille_B_ILP_20261028.md` — grille ILP/occupancy (piste B)
- `results/mb_fa_t3v2_verdict.md`, `results/t3_vldg_verdict_20261028.md`
- `results/fa_step_INVALID_note.md` — ⚠️ le `--fa-step 0` du microbench est
  invalide (le kernel FA ne se déclenche pas au step 0, le step réel est le 1)

## Limites / caveats

- Le gain microbench (×2) ne se transfère pas tel quel en e2e (+13 à +29 %) :
  le kernel FA n'est pas le seul coût decode (GEMV mmq domine court ctx).
- rows>8 (n_acceptance>7) retombe sur le kernel générique — n_max=3 limite
  structurellement rows à ≤ 4 (voir `repos/mtp-nmax/`).
- Le `.cu` est un snapshot du fork à l'instant du build ; la source complète du
  fork n'est pas dans ce repo (droits/copyright de la base — voir README meta §6).
