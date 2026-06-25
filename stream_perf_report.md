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
