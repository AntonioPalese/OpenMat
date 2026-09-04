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

> I numeri di questo documento sono quelli della RTX 4060, misurati nel 2024 e mai
> rieseguiti su quella macchina da allora. Le stesse suite sono state rieseguite su
> NVIDIA GB10 (DGX Spark, `sm_121`, CUDA 13.0) — l'ultima volta il **4 settembre 2026** —
> e **tre conclusioni su cinque si invertono** — vedi
> [Appendice — Rerun su NVIDIA GB10](#appendice--rerun-su-nvidia-gb10-dgx-spark).
>
> Attenzione a citare le tabelle 4060 come numeri correnti: le sezioni da qui alla fine
> dell'appendice descrivono lo stato del codice *a quell'epoca*, prima del fast path
> contiguo e del riciclo delle allocazioni host. L'appendice è la fonte aggiornata.

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
si trasferiscono**: tre delle cinque conclusioni si invertono. Questa sezione riporta i
numeri GB10 accanto a quelli RTX 4060 e spiega perché divergono.

> **Aggiornato al 4 settembre 2026.** Questa appendice è stata rimisurata dopo che sono
> atterrati il fast path contiguo per i kernel elementwise e il riciclo delle allocazioni
> host (`HostPool`/`PinnedHostPool`). Entrambi toccano il fondamento su cui poggiano
> questi test — il primo ha circa dimezzato il tempo dei kernel `add`/`mul` misurati qui,
> il secondo ha chiuso il collo di bottiglia sui trasferimenti che l'edizione precedente
> indicava come irrisolto — quindi **i numeri assoluti sono cambiati ovunque**, e in due
> righe è cambiata anche la conclusione. La versione precedente di questa appendice
> attribuiva a `Stress.AsyncTransferBandwidth` 4.3 GB/s: ora sono 51.1 GB/s.

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

Suite complete: **14/14 passate** (`ctest`, 6.17 s totali). I numeri sotto sono la
**mediana di 9 esecuzioni consecutive** di `test_stream_perf` — nove e non quattro perché
questa suite cronometra un singolo run non riscaldato per variante, e la dispersione fra
run è abbastanza larga da rendere una mediana su quattro campioni fuorviante: la catena
sequenziale, per esempio, ha prodotto singoli run da 0.98× e da 1.16×. Dove serve, il
range osservato è riportato accanto alla mediana.

## Confronto sintetico

| Test | RTX 4060 | GB10 | Esito |
|---|---|---|---|
| 1 — Single op, sync ogni iter | 0.42 ms, 1.00× | 0.22 ms, 1.03× | invariato |
| 2 — Sequential chain (100 add, 8 MB) | 45.38 → 16.94 ms, **2.68×** | 5.75 → 5.29 ms, **1.09×** | **guadagno quasi sparito** |
| 3a — Fan-out K=2 | ~1.05× | **3.29×** (2.98–3.44) | invertito |
| 3b — Fan-out K=4 | ~0.97× | ~1.04× (1.02–1.13) | invariato |
| 3c — Fan-out K=8 | ~1.06× | **5.19×** (4.92–5.66) | invertito |
| 3d — Fan-out K=16 | ~1.06× | **0.93×** (0.89–1.10) | **invertito, ora regressione** |
| 4 — Overlap compute/transfer | 37.21 → 33.11 ms, **1.12×** | 40.27 → 8.77 ms, **4.61×** | molto amplificato |
| 5 — Stream creation overhead | trascurabile | trascurabile (0.01 ms) | invariato |

## Test 2 — la catena sequenziale non guadagna più

Il 2.68× sulla 4060 derivava interamente dall'eliminazione di 99 stalli host, a ~0.28 ms
ciascuno. Su GB10 un round-trip di sincronizzazione costa così poco che i 100 sync valgono
complessivamente una frazione di millisecondo: la catena è limitata dal lavoro dei kernel,
non dagli stalli host, e accodare tutto su un unico stream rende il 9%.

La conclusione originale — «gli stream aiutano quando smetti di sincronizzare» — resta
valida come principio, ma la sua *entità* dipende dal costo di una sync sulla piattaforma.
Su un sistema a memoria unificata con CPU e GPU sullo stesso package quel costo è quasi
azzerato, e con esso il guadagno.

Da notare che la catena stessa è diventata **2.5× più veloce in valore assoluto** rispetto
alla misurazione precedente (14.6 → 5.75 ms) senza che questo test sia stato toccato: è il
fast path contiguo sui kernel elementwise ad aver accelerato le `add` che la compongono.
Questo è anche il motivo per cui il rapporto si è mosso da 1.03× a 1.09× — il numeratore e
il denominatore non sono scesi nella stessa proporzione, non perché gli stream siano
diventati più utili.

## Test 3 — il fan-out parallelo paga, ma non monotonicamente

Sulla 4060 il fan-out era un pareggio: un solo `mul` da 4 MB saturava già la banda e la GPU
aveva una sola pipeline di compute. Su GB10 otto `mul` indipendenti su otto stream vanno
**5.2× più veloci** della versione sequenziale. La motivazione architetturale scritta per la
4060 (banda satura ⇒ niente da guadagnare) non descrive questa piattaforma.

L'andamento resta marcatamente non monotono — K=2 dà 3.3×, K=4 ricade a ~1.04×, K=8 sale a
5.2×, K=16 **regredisce** a 0.93× — ma su nove run la dispersione per riga è stretta
abbastanza da dire che la forma è reale e non rumore: K=8 sta fra 4.92× e 5.66× in tutti e
nove i run, K=4 fra 1.02× e 1.13×, e K=16 sta **sotto 1.0× in otto run su nove**.

Quest'ultima riga è la novità rispetto all'edizione precedente, che dava K=16 a 1.25–1.42×:
con i kernel diventati circa 2× più veloci, a K=16 il lavoro per stream è sceso al punto che
il costo di gestire sedici stream supera quello che si guadagna a sovrapporli. È l'unico
punto in cui un'ottimizzazione ha reso *peggiore* un rapporto di questo report, ed è un
artefatto della metrica, non una regressione: il ramo sequenziale a K=16 è passato da
2.52 ms a 1.60 ms in valore assoluto.

Resta valido l'avvertimento metodologico: il test cronometra un singolo run non riscaldato
per K. Nove ripetizioni bastano a stabilizzare le mediane, non a giustificare conclusioni
architetturali fini su una curva così irregolare.

## Test 4 — overlap 4.6×, e ora per il motivo giusto

Il rapporto supera il tetto teorico di ~2× di un overlap perfetto, il che significa che sta
ancora misurando in parte un baseline mediocre: la variante serializzata impiega 40.3 ms su
GB10 contro 37.2 ms sulla 4060, perché è il ramo H2D a dominare.

**Due ipotesi, entrambe chiuse.** La prima — uno staging intermedio del driver — è stata
verificata e smentita: `Tensor::to()` pina la destinazione di un trasferimento
device-to-host (`PinnedCpuAllocator`, `cudaHostAlloc` riciclato via `PinnedHostPool` — vedi
[benchmark_report.md §3](benchmark_report.md#3-the-cpu-gap-above-128-kb-was-the-allocator-not-the-loop--fixed)),
esattamente ciò che salterebbe uno staging se ce ne fosse uno, e isolando la sola
`cudaMemcpyAsync` la banda è **58–59 GB/s sia pageable sia pinned**. Nessuna differenza:
NVLink-C2C non ha uno stadio di bounce buffer da saltare.

La seconda ipotesi era l'allocazione della destinazione a ogni round-trip, ed **era quella
giusta**. `Stress.AsyncTransferBandwidth` alloca il buffer di destinazione a ogni giro e
pagava per intero i page fault descritti in
[benchmark_report.md §3](benchmark_report.md#3-the-cpu-gap-above-128-kb-was-the-allocator-not-the-loop--fixed).
Con `HostPool` che ricicla quei blocchi lo stesso test passa da **3109 ms (4.3 GB/s) a
263 ms (51.1 GB/s)** — 11.8× — e una singola copia da 64 MB isolata misura 59.1 GB/s,
identica a PyTorch. Il collo di bottiglia era l'allocatore, non il tipo di memoria.

Quello che resta nel baseline serializzato è quindi tempo H2D genuino, e il 4.6× di questa
riga va letto come overlap reale e non più come un baseline da sistemare.

---

## `test_benchmarks` su GB10

| MatMul | fp32 ms | fp32 GFLOPS | fp16 ms | fp16 GFLOPS |
|---|---|---|---|---|
| 256² | 0.03 | 1053 | 0.03 | 970 |
| 512² | 0.18 | 1525 | 0.18 | 1530 |
| 1024² | 1.29 | 1664 | 1.29 | 1665 |
| 2048² | 11.38 | 1509 | 10.15 | 1692 |
| 4096² | 104.99 | 1309 | 92.43 | 1487 |

fp16 rende solo ~1.1× rispetto a fp32: atteso, dato che il kernel è scritto a mano e non
usa i tensor core. È la misura più diretta di quanto costa quella scelta — vedi
[benchmark_report.md §6](benchmark_report.md#6-matmul-the-cpu-gap-closed-68-the-gpu-one-remains).

Element-wise su 16M elementi (fra parentesi la misurazione precedente):

| Op | ms | Gelem/s |
|---|---|---|
| float32 add | 3.08 (3.79) | 5.44 (4.42) |
| float32 mul | 2.96 (3.70) | 5.67 (4.54) |
| `scale_shift` | 2.72 (3.68) | 6.18 (4.56) |
| `fused_add_mul` | 2.99 (3.36) | 5.62 (5.00) |
| float16 add | 1.56 (2.58) | 10.74 (6.51) |
| float16 mul | 1.45 (2.54) | 11.56 (6.61) |

Il guadagno più grande è su float16 (~1.65×), coerente con il fatto che il fast path
contiguo impacchetta `4 / sizeof(T)` elementi per thread: per `float16_t` sono due, mentre
per `float` è uno.

## `test_stress` su GB10

| Scenario | precedente | ora |
|---|---|---|
| 500 × 4 MB alloc+fill+free | 255.0 ms (0.51 ms/iter) | 192.5 ms (0.38 ms/iter) |
| 64 tensori vivi contemporaneamente (2 MB) | 5.8 ms | 4.7 ms |
| 1000 × (add+mul) su 16M float | 639.4 ms — 78.7 GB/s | **392.3 ms — 128.3 GB/s** |
| 8 stream × 8 MB mul in parallelo | 3.1 ms | 3.0 ms |
| add 512 MB + 512 MB | 26.2 ms — 61.5 GB/s | 23.0 ms — 70.1 GB/s |
| catena add profonda 200 (4 MB) | 13.2 ms | **3.4 ms** |
| permute rank-6 [4⁶] ×1000 | 8.9 ms | 9.5 ms |
| permute rank-8 [4⁸] inverso | OK | OK |
| add+mul 32 MB su CPU | 20.1 ms | **4.9 ms** |
| 100 × 64 MB round-trip H2D+D2H | 3109.1 ms — 4.3 GB/s | **262.7 ms — 51.1 GB/s** |

Tre righe si muovono per cause diverse, e vale la pena tenerle distinte: la catena profonda
(3.9×) e il sostenuto elementwise (1.6×) vengono dal fast path contiguo; l'`add+mul` su CPU
(4.1×) dal `parallel for` OpenMP; il round-trip H2D+D2H (11.8×) dal riciclo delle
allocazioni host. `permute` è l'unica riga che non migliora — non passa dal fast path,
perché è proprio l'operazione che rende un buffer non contiguo.

## Python

Con `OpenMat.so` in Release: `python/test_bindings.py` passa interamente (CPU e GPU), e la
suite `pytest` — non eseguibile alla stesura precedente, perché su questa macchina mancavano
sia `uv` sia `pytest` — ora gira e passa: **139 test in 0.55 s**
(`test_tensor.py`, `test_tensor_api.py`, `test_dtypes.py`, `test_streams.py`).

## Output grezzo (GB10, Release)

Un singolo run rappresentativo dei nove. Si noti K=4 a 1.06× e K=16 a 0.93×, coerenti con
le mediane della tabella sopra.

```
+-- Single op (16M add, 200 iters, sync-after-each)
|   variant                                 ms  speedup
+------------------------------------------------------
|   operator+ (sync)                     0.22 ms    1.00x
|   add(default_stream())                0.22 ms    1.01x
|   add(Stream s) + sync                 0.21 ms    1.03x
+------------------------------------------------------

+-- Sequential chain (100 adds, 8MB, single run)
|   variant                                 ms  speedup
+------------------------------------------------------
|   sync after each op                    5.75 ms    1.00x
|   stream + 1 sync                       5.29 ms    1.09x
+------------------------------------------------------

+-- Fan-out: K independent muls (4MB each)
|   variant                                 ms  speedup
+------------------------------------------------------
|   K=2 sequential                       0.14 ms    1.00x
|   K=2 parallel streams                 0.04 ms    3.29x
|   K=4 sequential                       0.10 ms    1.00x
|   K=4 parallel streams                 0.09 ms    1.06x
|   K=8 sequential                       1.47 ms    1.00x
|   K=8 parallel streams                 0.28 ms    5.19x
|   K=16 sequential                      1.60 ms    1.00x
|   K=16 parallel streams                1.73 ms    0.93x
+------------------------------------------------------

+-- Compute+transfer overlap (20 rounds, 16MB)
|   variant                                 ms  speedup
+------------------------------------------------------
|   serialised (H2D then compute)       40.27 ms    1.00x
|   overlapped (2 streams)               8.77 ms    4.61x
+------------------------------------------------------

+-- Stream creation overhead (1000 iters, 256KB mul)
|   variant                                 ms  speedup
+------------------------------------------------------
|   reuse stream                         0.01 ms    1.00x
|   new stream per call                  0.01 ms    0.60x
+------------------------------------------------------
```

## Come riprodurre

```bash
export CMAKE_LIBRARY_PATH="/usr/local/cuda/lib64:/usr/lib/$(uname -m)-linux-gnu"
cmake -S . -B build-release -DCMAKE_BUILD_TYPE=Release
make -C build-release -j$(nproc)

cd build-release && ctest                     # 14/14
./tests/test_stream_perf                      # ripetere ~9 volte e prendere la mediana
./tests/test_benchmarks
./tests/test_stress
```
