# CUDA Streams Performance Report — OpenMat

## Ambiente di test

| Componente | Valore |
|---|---|
| GPU | NVIDIA GeForce RTX 4060 |
| VRAM | 8 GB |
| CUDA Toolkit | 11.5 (V11.5.119) |
| Driver | 535.309.01 |
| Compute capability | 8.9 (Ada Lovelace) |
| Test suite | `tests/test_stream_perf.cpp` |

> I numeri di questo documento sono quelli della RTX 4060. Le stesse suite sono state
> rieseguite su NVIDIA GB10 (DGX Spark, `sm_121`, CUDA 13.0) e **due conclusioni su cinque
> si invertono** — vedi [Appendice — Rerun su NVIDIA GB10](#appendice--rerun-su-nvidia-gb10-dgx-spark).

---

## Panoramica

I CUDA Streams sono un meccanismo del driver CUDA per accodare operazioni sulla GPU senza bloccare il thread host. Il confronto principale è tra due modalità:

- **Sync**: dopo ogni operazione viene chiamato `cudaDeviceSynchronize()`, che blocca il thread host fino al completamento del kernel.
- **Stream**: le operazioni vengono accodate su uno stream asincrono; il thread host prosegue immediatamente. La sincronizzazione avviene una sola volta, o su stream separati per lavoro indipendente.

---

## Test 1 — Single op latency

**Scenario:** 200 iterazioni di una singola `add` su tensori da 16 MB (4M float32). Ogni iterazione lancia un kernel e poi sincronizza immediatamente.

| Variante | Tempo medio/iter | Speedup |
|---|---|---|
| `operator+` (sync implicita) | 0.42 ms | 1.00× |
| `add(default_stream())` | 0.42 ms | 1.00× |
| `add(Stream s)` + sync esplicita | 0.40 ms | 1.05× |

**Risultato: nessun miglioramento significativo.**

Quando si sincronizza dopo ogni singola operazione, gli stream non portano benefici: il lavoro del kernel è identico e il round-trip host→GPU→host rimane lo stesso. La differenza di ~1–5% rientra nel rumore di misura. Questo conferma che gli stream non aggiungono overhead osservabile, ma non aiutano nemmeno se usati come un semplice wrapper intorno a ogni singola op.

---

## Test 2 — Sequential chain

**Scenario:** 100 operazioni `add` consecutive su tensori da 8 MB (2M float32), con dati dipendenti (ogni output alimenta l'input successivo).

| Variante | Tempo totale | Speedup |
|---|---|---|
| Sync dopo ogni op (100 sync) | 45.38 ms | 1.00× |
| Stream + 1 sync alla fine | 16.94 ms | **2.68×** |

**Risultato: miglioramento netto di 2.68×.**

Questo è il caso d'uso più impattante per gli stream. Con 100 sync esplicite, il thread host si blocca 100 volte aspettando la GPU; ogni stallo consuma tempo di CPU e introduce latenza di scheduling. Con uno stream unico e una sola sync finale:

1. Il driver CUDA riceve tutti i 100 kernel in sequenza senza interruzioni.
2. La GPU può eseguire pipeline interne e nascondere latenze di memoria.
3. Il thread host non è mai bloccato durante l'accodamento.

La dipendenza dati tra le op non impedisce la sovrapposizione host/GPU: il driver mantiene l'ordine di esecuzione dei kernel nello stesso stream garantendo la correttezza.

Il guadagno di ~45 ms → ~17 ms è quasi interamente attribuibile all'eliminazione di 99 round-trip host/GPU (ciascuno ~0.28 ms di overhead).

---

## Test 3 — Parallel fan-out

**Scenario:** K operazioni `mul` indipendenti su K tensori da 4 MB ciascuno (1M float32), comparando esecuzione sequenziale vs K stream paralleli.

| K | Sequential | Parallel streams | Speedup |
|---|---|---|---|
| 2 | 0.15 ms | 0.14 ms | 1.05× |
| 4 | 0.30 ms | 0.31 ms | 0.97× |
| 8 | 0.73 ms | 0.69 ms | 1.06× |
| 16 | 1.46 ms | 1.38 ms | 1.06× |

**Risultato: nessun miglioramento significativo (≈ pareggio).**

Il guadagno atteso dalla parallelizzazione su più stream non si materializza in modo rilevante. La ragione è legata all'architettura Ada Lovelace e alle dimensioni dei dati:

- Ogni singolo `mul` su 4 MB satura già il 100% della memoria di banda della RTX 4060 (~272 GB/s). I kernel successivi sullo stesso SM non trovano ulteriore parallelismo disponibile.
- L'RTX 4060 ha un singolo motore di compute; la concorrenza tra kernel è limitata rispetto a GPU server come A100/H100.
- Con operazioni memory-bound di questa taglia, aggiungere stream paralleli non scala: il collo di bottiglia è la larghezza di banda, non la latenza di scheduling.

Il caso in cui più stream paralleli danno speedup reale è con kernel compute-bound (matmul piccoli, fused ops complesse) o con operazioni di dimensioni ridotte che non saturano il bus.

---

## Test 4 — Compute + transfer overlap

**Scenario:** 20 round, ciascuno con un upload H2D da 16 MB + una `mul` su 16 MB. Confronto tra esecuzione serializzata (prima il trasferimento, poi il calcolo) e sovrapposta (i due stream procedono in parallelo).

| Variante | Tempo totale | Speedup |
|---|---|---|
| Serializzato (H2D → sync → compute → sync) | 37.21 ms | 1.00× |
| Sovrapposto (stream_copy ∥ stream_compute) | 33.11 ms | **1.12×** |

**Risultato: miglioramento del 12%.**

L'RTX 4060 dispone di un motore DMA (copy engine) dedicato che opera indipendentemente dagli SM di compute. Accodando il trasferimento su uno stream e il calcolo su un altro, le due operazioni si sovrappongono parzialmente:

```
Serializzato:   [H2D------][compute------][H2D------][compute------]
Sovrapposto:    [H2D------]
                    [compute------]
                             [H2D------]
                                 [compute------]
```

Il guadagno teorico massimo con overlap perfetto sarebbe ~2× (il tempo scende a `max(H2D, compute)` invece di `H2D + compute`). Il 12% osservato è inferiore al teorico per due ragioni:

1. H2D e compute hanno tempi simili (~1.5 ms ciascuno), quindi l'overlap è parziale.
2. C'è una piccola latenza di scheduling tra round successivi.

In scenari reali con pipeline di inferenza (carica batch → processa batch precedente) il guadagno è più pronunciato.

---

## Test 5 — Stream creation overhead

**Scenario:** 1000 iterazioni di una `mul` su 256 KB, comparando il riuso di uno stream esistente vs la creazione di un nuovo stream per ogni chiamata.

| Variante | Tempo medio/iter | Speedup |
|---|---|---|
| Riuso stream | 0.01 ms | 1.00× |
| Nuovo stream per chiamata | 0.01 ms | 0.83× |

**Risultato: overhead trascurabile a questa scala.**

Con kernel da 256 KB l'overhead di `cudaStreamCreate` (~10–30 µs tipicamente) è comparabile al tempo del kernel stesso, rendendo i due casi simili. In pratica, su kernel più brevi o in loop ad alta frequenza la differenza diventa rilevante: creare uno stream è un'operazione di sistema che coinvolge il driver, il context CUDA e l'allocazione di strutture interne. La best practice è creare gli stream una volta sola e riusarli.

---

## Conclusioni generali

| Scenario | Streams migliorano? | Speedup |
|---|---|---|
| Singola op, sync immediata | No | ~1× |
| Catena di op dipendenti, 1 sync finale | **Sì** | **2.68×** |
| Op indipendenti in parallelo (memory-bound) | No (pareggio) | ~1× |
| Overlap compute + trasferimento dati | **Sì** | **1.12×** |
| Stream creation overhead | Trascurabile | — |

Gli stream CUDA portano benefici concreti in due situazioni:

1. **Catene di operazioni**: eliminare N−1 sincronizzazioni da una sequenza di N op dipendenti è il guadagno più grande. Su 100 op consecutive il tempo si riduce di quasi 3×. Questo schema è comune in inference (forward pass layer-by-layer) e in preprocessing pipeline.

2. **Overlap compute/trasferimento**: sfruttare il copy engine separato della GPU per sovrapporre upload di dati con elaborazione è un guadagno reale, tipicamente 10–30% su GPU consumer, di più su GPU server con link NVLink o HBM.

Gli stream **non aiutano** quando:
- Il kernel è già memory-bound e satura la larghezza di banda disponibile (il collo di bottiglia non è lo scheduling).
- Si sincronizza comunque dopo ogni op (annulla il beneficio del dispatch asincrono).
- I kernel sono troppo grandi per essere co-schedulati sulla stessa GPU.

Il pattern più efficace nell'attuale implementazione OpenMat è accodare più operazioni sullo stesso stream e sincronizzare una volta sola alla fine del batch, come avviene naturalmente con l'API stream overload `tensor.op(args, stream)`.

---

# Appendice — Rerun su NVIDIA GB10 (DGX Spark)

Tutti i test sopra sono stati rieseguiti su una seconda architettura. **I risultati non
si trasferiscono**: due delle cinque conclusioni si invertono. Questa sezione riporta i
numeri GB10 accanto a quelli RTX 4060 e spiega perché divergono.

## Ambiente di test

| Componente | RTX 4060 (originale) | GB10 (DGX Spark) |
|---|---|---|
| GPU | GeForce RTX 4060 | NVIDIA GB10 |
| Compute capability | 8.9 (Ada Lovelace) | 12.1 (Blackwell, `sm_121`) |
| Memoria | 8 GB GDDR6 dedicata | unificata Grace–Blackwell |
| CUDA Toolkit | 11.5 | 13.0 (V13.0.88) |
| Driver | 535.309.01 | 580.173.02 |
| Compilatore | — | GCC 13.3 / CMake 3.28 |
| Build | Release | Release |

Suite complete: **12/12 passate** (`ctest`, 12.31 s totali). I numeri sotto sono la
mediana di 4 esecuzioni consecutive di `test_stream_perf`; la varianza è entro il ±3%
tranne dove indicato.

## Confronto sintetico

| Test | RTX 4060 | GB10 | Esito |
|---|---|---|---|
| 1 — Single op, sync ogni iter | 0.42 ms, 1.00× | 0.32 ms, 1.02× | invariato |
| 2 — Sequential chain (100 add, 8 MB) | 45.38 → 16.94 ms, **2.68×** | 14.6 → 14.2 ms, **1.03×** | **guadagno sparito** |
| 3a — Fan-out K=2 | ~1.05× | **1.62×** | invertito |
| 3b — Fan-out K=8 | ~1.06× | **3.1–3.4×** | invertito |
| 3c — Fan-out K=16 | ~1.06× | **1.25–1.42×** | invertito |
| 4 — Overlap compute/transfer | 37.21 → 33.11 ms, **1.12×** | 47.9 → 11.9 ms, **~4.0×** | molto amplificato |
| 5 — Stream creation overhead | trascurabile | trascurabile (0.01 ms) | invariato |

## Test 2 — la catena sequenziale non guadagna più

Il 2.68× sulla 4060 derivava interamente dall'eliminazione di 99 stalli host, a ~0.28 ms
ciascuno. Su GB10 un round-trip di sincronizzazione costa così poco che 100 sync valgono
complessivamente ~0.45 ms su un totale di ~14.6 ms: la catena è limitata dal lavoro dei
kernel, non dagli stalli host, e accodare tutto su un unico stream rende il 3%.

La conclusione originale — «gli stream aiutano quando smetti di sincronizzare» — resta
valida come principio, ma la sua *entità* dipende dal costo di una sync sulla piattaforma.
Su un sistema a memoria unificata con CPU e GPU sullo stesso package quel costo è quasi
azzerato, e con esso il guadagno.

## Test 3 — il fan-out parallelo ora paga

Sulla 4060 il fan-out era un pareggio: un solo `mul` da 4 MB saturava già la banda e la GPU
aveva una sola pipeline di compute. Su GB10 otto `mul` indipendenti su otto stream vanno
**3.3× più veloci** della versione sequenziale. La motivazione architetturale scritta per la
4060 (banda satura ⇒ niente da guadagnare) non descrive questa piattaforma.

Da notare l'andamento non monotono: K=2 dà 1.6×, K=4 crolla a 1.05×, K=8 sale a 3.3×, K=16
ritorna a ~1.3×. Le righe K=8 e K=16 sono anche le uniche instabili tra run (3.10–3.43× e
1.25–1.42×). Il test misura un singolo run senza warm-up per K, quindi parte di questa
irregolarità è misura, non hardware — vale la pena mediarlo su più ripetizioni prima di
trarne conclusioni architetturali.

## Test 4 — overlap 4×, ma per il motivo sbagliato

Il rapporto sembra ottimo, però il guadagno viene soprattutto dal fatto che il *baseline*
è peggiore: la variante serializzata impiega 47.9 ms su GB10 contro 37.2 ms sulla 4060,
mentre quella sovrapposta scende a 11.9 ms. È il ramo H2D a dominare, e infatti
`Stress.AsyncTransferBandwidth` misura solo **4.3–4.8 GB/s** su round-trip da 64 MB — una
cifra bassa per una piattaforma a memoria coerente. L'ipotesi più probabile è che i
trasferimenti passino da memoria host pageable con staging intermedio invece che da memoria
pinned. Se confermato, sistemare il path di trasferimento migliorerebbe il tempo assoluto e
al tempo stesso **ridurrebbe** questo 4×.

---

## `test_benchmarks` su GB10

| MatMul | fp32 ms | fp32 GFLOPS | fp16 ms | fp16 GFLOPS |
|---|---|---|---|---|
| 256² | 0.03 | 1024 | 0.04 | 900 |
| 512² | 0.18 | 1453 | 0.18 | 1458 |
| 1024² | 1.36 | 1578 | 1.36 | 1579 |
| 2048² | 12.10 | 1420 | 11.09 | 1549 |
| 4096² | 99.98 | 1375 | 89.81 | 1530 |

fp16 rende solo ~1.12× rispetto a fp32: atteso, dato che il kernel è scritto a mano e non
usa i tensor core.

Element-wise su 16M elementi:

| Op | ms | Gelem/s |
|---|---|---|
| float32 add | 3.79 | 4.42 |
| float32 mul | 3.70 | 4.54 |
| `scale_shift` | 3.68 | 4.56 |
| `fused_add_mul` | 3.36 | 5.00 |
| float16 add | 2.58 | 6.51 |
| float16 mul | 2.54 | 6.61 |

## `test_stress` su GB10

| Scenario | Risultato |
|---|---|
| 500 × 4 MB alloc+fill+free | 255.0 ms (0.51 ms/iter) |
| 64 tensori vivi contemporaneamente (2 MB) | 5.8 ms |
| 1000 × (add+mul) su 16M float | 639.4 ms — **78.7 GB/s** effettivi |
| 8 stream × 8 MB mul in parallelo | 3.1 ms |
| add 512 MB + 512 MB | 26.2 ms — **61.5 GB/s** |
| catena add profonda 200 (4 MB) | 13.2 ms |
| permute rank-6 [4⁶] ×1000 | 8.9 ms |
| permute rank-8 [4⁸] inverso | OK |
| add+mul 32 MB su CPU | 20.1 ms |
| 100 × 64 MB round-trip H2D+D2H | 3109.1 ms — **4.3 GB/s** |

## Python

`python/test_bindings.py` passa interamente (CPU e GPU) contro `OpenMat.so` in Release.
La suite `pytest` non è stata eseguita: su questa macchina `uv` non è installato e il
`python3` di sistema non ha `pytest`.

## Output grezzo (GB10, Release)

```
+-- Single op (16M add, 200 iters, sync-after-each)
|   variant                                 ms  speedup
+------------------------------------------------------
|   operator+ (sync)                     0.33 ms    1.00x
|   add(default_stream())                0.32 ms    1.04x
|   add(Stream s) + sync                 0.31 ms    1.05x
+------------------------------------------------------

+-- Sequential chain (100 adds, 8MB, single run)
|   variant                                 ms  speedup
+------------------------------------------------------
|   sync after each op                  15.26 ms    1.00x
|   stream + 1 sync                     15.24 ms    1.00x
+------------------------------------------------------

+-- Fan-out: K independent muls (4MB each)
|   variant                                 ms  speedup
+------------------------------------------------------
|   K=2 sequential                       0.22 ms    1.00x
|   K=2 parallel streams                 0.14 ms    1.64x
|   K=4 sequential                       0.27 ms    1.00x
|   K=4 parallel streams                 0.25 ms    1.06x
|   K=8 sequential                       1.92 ms    1.00x
|   K=8 parallel streams                 0.57 ms    3.34x
|   K=16 sequential                      2.52 ms    1.00x
|   K=16 parallel streams                2.01 ms    1.25x
+------------------------------------------------------

+-- Compute+transfer overlap (20 rounds, 16MB)
|   variant                                 ms  speedup
+------------------------------------------------------
|   serialised (H2D then compute)       47.77 ms    1.00x
|   overlapped (2 streams)              12.50 ms    3.82x
+------------------------------------------------------

+-- Stream creation overhead (1000 iters, 256KB mul)
|   variant                                 ms  speedup
+------------------------------------------------------
|   reuse stream                         0.01 ms    1.00x
|   new stream per call                  0.01 ms    0.71x
+------------------------------------------------------
```
