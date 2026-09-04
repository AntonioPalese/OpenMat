# OpenMat — Roadmap delle implementazioni future

Questo documento elenca le funzionalità del framework, il loro stato attuale e il lavoro ancora da fare, raggruppate per area. Ogni voce include una breve descrizione tecnica e i file coinvolti.

**Legenda:** ✅ implementato · ⚠️ implementato parzialmente o con divergenze rispetto al progetto originale · ❌ da fare

Ultimo allineamento al codice: 4 settembre 2026 (in-place / `out=` aggiunti, §8; benchmark rieseguiti su GB10, Release; vedi [`benchmark_report.md`](../benchmark_report.md)).

---

## 1. Operazioni matematiche

### 1.1 Operazioni element-wise aggiuntive ❌

`add`, `sub`, `mul`, `div` (tensore⊕tensore e tensore⊕scalare) e `matmul` esistono. Mancano le seguenti, che seguono esattamente lo stesso pattern dei macro in `binary_op_macros.cuh` / `binary_op_macros.h`. Per ognuna basta definire kernel + functor CPU + metodo su `Tensor`.

| Operazione | Espressione | Note |
|---|---|---|
| `pow(rhs)` | `lhs[i] ^ rhs[i]` | Usare `powf` per float, `pow` per double |
| `min(rhs)` | `min(lhs[i], rhs[i])` | Usare `fminf` su GPU. Attenzione: il nome collide con `min()` riduzione già esistente — servirà un overload su `const Tensor&` |
| `max(rhs)` | `max(lhs[i], rhs[i])` | Come sopra |
| `abs()` | `\|x[i]\|` | Unaria, usare `fabsf` su GPU |
| `sqrt()` | `√x[i]` | Unaria, usare `sqrtf` su GPU |
| `exp()` | `e^x[i]` | Unaria, usare `expf` su GPU |
| `log()` | `ln(x[i])` | Unaria, usare `logf` su GPU |

Nota: le unarie (`abs`, `sqrt`, `exp`, `log`) **non** richiedono il macchinario dei binary op — conviene implementarle come functor in `fused_op.cuh` esposti via `Tensor::apply`, esattamente come `relu` e `sigmoid`. Il costo è un functor più quattro istanziazioni esplicite, contro cinque file da toccare. Esiste già un functor `Pow<T>` in `fused_op.cuh`, ma non è istanziato né esposto su `Tensor`.

**File da modificare (via `apply`, percorso consigliato):**
- `headers/ops/kernels/fused_op.cuh` — il nuovo functor
- `src/ops/kernels/fused_op.cu` — le istanziazioni esplicite per `float`, `int`, `char`, `float16_t`
- `headers/tensor.cuh` / `headers/tensor.inl` — l'overload `(const Stream&)` più il delegate senza stream

**File da modificare (via binary op, per le binarie):**
- `headers/ops/kernels/binary_op_macros.cuh` — `DEFINE_BINARY_OP_KERNEL_H` / `DEFINE_BINARY_OP_LAUNCH_H`
- `src/ops/kernels/binary_ops.cu` — `DEFINE_BINARY_OP_KERNEL_K1/K2/K3/K4/ND` + `DEFINE_BINARY_OP_LAUNCH` + `DEFINE_BINARY_OP_LAUNCH_FRW_DEC`
- `headers/ops/cpu/binary_op_macros.h` + `src/ops/cpu/binary_ops.cpp` — lato CPU
- `headers/tensor.cuh` / `headers/tensor.inl` — metodi pubblici
- `headers/kernel_launcher.h` / `.inl` — **solo** se serve anche la free function senza stream (il dispatch è ormai legacy: `Tensor` chiama i launcher direttamente)

---

### 1.2 Operazioni di riduzione ⚠️

Le riduzioni a scalare sono implementate; le riduzioni lungo un asse no.

| Operazione | Risultato | Stato |
|---|---|---|
| `sum()` | scalare | ✅ |
| `mean()` | scalare | ✅ |
| `min()` | scalare | ✅ |
| `max()` | scalare | ✅ |
| `sum(axis)` | tensore rank-1 | ❌ |
| `mean(axis)` | tensore rank-1 | ❌ |

Implementate in [`headers/ops/kernels/reduce_gpu.cuh`](../headers/ops/kernels/reduce_gpu.cuh) (riduzione a due fasi: tree-reduction in shared memory per blocco + warp shuffle con `__shfl_down_sync`) e [`headers/ops/cpu/reduce_cpu.h`](../headers/ops/cpu/reduce_cpu.h). Suite di test: `test_reductions`.

**Limite noto:** le riduzioni sono **sincrone** e restituiscono uno scalare host — non hanno overload `(const Stream&)`, a differenza di quasi tutto il resto della libreria. Aggiungerli richiede una variante che scriva il risultato in un `Tensor` di dimensione 1 sul device, lasciando la copia su host al chiamante.

**Prestazioni, aggiornamento settembre 2026.** `reduce_sum_cpu` accumulava in un singolo registro: la dipendenza FP loop-carried impediva la vettorizzazione automatica (senza `-ffast-math`) e il loop girava a un'addizione per *latenza* FP invece che per throughput — 7.7 GB/s contro la riduzione SIMD pairwise di NumPy. Spezzando l'accumulo su **8 lane indipendenti** ricombinate a coppie alla fine si rompe la catena: **36.9 GB/s a 16M elementi, 1.06× più veloce di NumPy** invece che 4× più lento. È una ristrutturazione a livello sorgente, non un pragma, ed è anche leggermente più accurata (i parziali sommati alla fine hanno magnitudine simile).

Resta single-thread: PyTorch è ancora 2.8× più veloce a 16M perché distribuisce la riduzione su 20 core. Un `parallel for` con clausola `reduction(+:)` è il passo successivo non ancora tentato — da non confondere con la clausola `simd reduction(min:)/(max:)` su `min`/`max`, che è stata misurata e **rimossa** (§6.4).

`min`/`max` restano il divario aperto più grande sulla superficie elementwise CPU: 2.7× più lenti di NumPy a 16M, 6.8× a 1M.

Le riduzioni per asse richiedono un kernel diverso da quello attuale: una griglia in cui ogni blocco riduce una "colonna" lungo l'asse scelto, con gli stride del tensore a determinare il passo.

---

### 1.3 Operazioni di confronto element-wise ❌

| Operazione | Espressione |
|---|---|
| `eq(rhs)` | `lhs[i] == rhs[i]` → tensore di `int` (0/1) |
| `lt(rhs)` | `lhs[i] < rhs[i]` |
| `gt(rhs)` | `lhs[i] > rhs[i]` |
| `clamp(min, max)` | `min ≤ x[i] ≤ max` |

Seguono il pattern binario esistente ma restituiscono `Tensor<int>` (maschera booleana). Questo è il primo caso nella libreria di un'operazione il cui tipo di ritorno differisce da quello degli operandi: `launch_apply_binary_op` è templato su un solo `T` per tutti e tre i view, quindi servirà un launcher separato con `T` in ingresso e `int` in uscita.

`clamp` invece è unaria e ricade direttamente su `Tensor::apply` con un functor `Clamp<T>{lo, hi}`.

---

## 2. Operazioni sulle dimensioni

### 2.1 Reshape e view ⚠️

```cpp
Tensor<T> reshape(const std::vector<size_t>& new_shape) const;  // ✅
Tensor<T> flatten() const;    // ✅ reshape({size()})
Tensor<T> squeeze(size_t axis) const;    // ✅
Tensor<T> unsqueeze(size_t axis) const;  // ✅
```

**Divergenza rispetto al progetto originale.** Il piano era di renderle zero-copy per tensori contigui. L'implementazione in [`headers/tensor.inl`](../headers/tensor.inl) fa invece una **deep copy**: `reshape` costruisce `Tensor out(*this)` (il copy-ctor alloca e copia) e poi riscrive `m_Shape`/`m_Stride`. Gli altri tre delegano a `reshape`.

Questa è una scelta deliberata, non un bug: `Tensor<T>` possiede il proprio buffer tramite un `unique_ptr<Allocator<T>>` e non esiste nella libreria alcun meccanismo di ownership condivisa. Una view zero-copy richiederebbe che `Tensor` sappia di non possedere i dati, il che significa introdurre un `shared_ptr` sul buffer o un tipo `TensorRef` distinto. Conseguenza pratica: a differenza di NumPy/PyTorch, una scrittura su un tensore reshaped **non** è visibile attraverso l'originale.

Sono operazioni host-only: nessun kernel, nessun overload con stream. `squeeze` dell'ultimo asse rimasto produce shape `{1}`, non uno scalare.

Suite di test: `test_reshape`.

**Lavoro residuo:** decidere se introdurre view aliasing (e con quale modello di ownership) oppure documentare definitivamente la semantica a copia. È il prerequisito di 2.3.

---

### 2.2 Transpose e permute ✅

```cpp
Tensor<T> transpose() const;                        // solo rank 2, altrimenti throw
Tensor<T> permute(const std::vector<size_t>& axes) const;
Tensor<T> transpose(const Stream& s) const;
Tensor<T> permute(const std::vector<size_t>& axes, const Stream& s) const;
```

Implementate in [`headers/ops/cpu/transpose_cpu.h`](../headers/ops/cpu/transpose_cpu.h) e [`src/ops/kernels/transpose_gpu.cu`](../src/ops/kernels/transpose_gpu.cu), entrambe con percorso CPU e GPU reali e con overload sullo stream.

**Divergenza:** il piano prevedeva una permutazione dei soli stride (vista logica, zero-copy). Come per 2.1, l'implementazione **sposta i dati**. Il vantaggio non è banale: il risultato resta contiguo in row-major, quindi ogni operazione successiva conserva accessi coalescenti invece di ereditare uno stride pattern degenere.

`permute` valida gli assi sull'host (lunghezza == rank, in range, nessun duplicato) prima del dispatch. `launch_permute` riceve gli assi come `const size_t*` **host** e li copia in un `AxesBuf`, struct trivially-copyable passata **per valore** nel parameter block del kernel — stessa regola di `DeviceTensorView`: niente allocazioni device per metadati piccoli, e nessun membro puntatore, che sul device decadrebbe in un puntatore host inutilizzabile.

Suite di test: `test_transpose`.

---

### 2.3 Slice e indexing ❌

```cpp
Tensor<T> slice(size_t axis, size_t start, size_t end) const;
Tensor<T> operator[](size_t index) const; // selezione lungo il primo asse
```

Non implementate in C++. Esiste l'accesso a singolo elemento tramite `operator()(std::initializer_list<size_t>)`, che valida rank e bounds ma funziona **solo su tensori host** (dereferenzia `m_Data` direttamente). Sul lato Python, `__getitem__` / `__setitem__` accettano un indice completo e restituiscono uno scalare, non una sotto-vista.

Uno slice sarebbe naturalmente un view con pointer offset e shape/stride aggiornati — quindi dipende dalla decisione presa in 2.1 sul modello di ownership. Finché `Tensor` è strettamente owning, `slice` può solo essere una copia.

---

## 3. Gestione della memoria

### 3.1 Trasferimento CPU ↔ GPU come metodi `Tensor` ✅

```cpp
Tensor<T> to(const Device& target) const;   // ✅
Tensor<T> cpu() const;                      // ✅
Tensor<T> cuda() const;                     // ✅
Tensor<T> to(const Device& target, const Stream& s) const;  // ✅
Tensor<T> cpu(const Stream& s) const;                       // ✅
Tensor<T> cuda(const Stream& s) const;                      // ✅
```

`copyToHost` / `copyToDevice` restano disponibili per scrivere su un buffer raw fornito dal chiamante. Suite di test: `test_device_transfer`.

---

### 3.2 Inizializzazione dei tensori ⚠️

```cpp
static Tensor<T> zeros(const std::vector<size_t>& shape, const Device& dv);    // ✅
static Tensor<T> ones(const std::vector<size_t>& shape, const Device& dv);     // ✅
static Tensor<T> full(const std::vector<size_t>& shape, T value, const Device& dv);  // ✅
static Tensor<T> from_vector(const std::vector<T>& data, const std::vector<size_t>& shape, const Device& dv);  // ✅
static Tensor<T> from_vector(..., const Device& dv, const Stream& s);          // ✅
static Tensor<T> arange(T start, T stop, T step, const Device& dv);            // ❌
static Tensor<T> linspace(T start, T stop, size_t n, const Device& dv);        // ❌
```

`zeros`, `ones`, `full` sono costruite sopra `fill` — l'unico consumatore rimasto del vecchio percorso di dispatch `_fill` in `kernel_launcher.inl`.

`arange` e `linspace` mancano in C++. **Nota:** `Tensor.arange` esiste sul lato Python ([`python/openmat/tensor.py`](../python/openmat/tensor.py)), ma è implementata costruendo la lista in Python e passandola a `from_list` — quindi è O(n) sull'host e non ha equivalente nativo. Una `arange` C++ richiede un kernel dedicato (o un loop CPU) e renderebbe la versione Python un semplice wrapper.

Suite di test: `test_factory`.

---

### 3.3 Inizializzazione random ❌

```cpp
static Tensor<T> rand_uniform(const std::vector<size_t>& shape, T low, T high, const Device& dv);
static Tensor<T> rand_normal(const std::vector<size_t>& shape, T mean, T std, const Device& dv);
```

Su GPU si usa cuRAND (`curandGenerateUniform`, `curandGenerateNormal`). Aggiunge una dipendenza da `libcurand` in `CMakeLists.txt`.

Da valutare: cuRAND ha un proprio concetto di stream (`curandSetStream`), che va allineato all'`om::Stream` del tensore di destinazione perché la memoria allocata con `cudaMallocAsync` resti coerente con lo stream che la possiede. Serve inoltre una decisione sulla gestione del seed (generatore globale vs. per chiamata).

---

## 4. Fused operations

Area sostanzialmente completata. Il documento di riferimento è [`docs/fused_operations.md`](fused_operations.md).

### 4.1 Test per le API fused ✅

[`tests/test_fused_ops.cpp`](../tests/test_fused_ops.cpp) contiene 36 test: correttezza CPU su valori noti, correttezza GPU per rank 1/2/3, consistenza CPU↔GPU e equivalenza rispetto al calcolo non fuso in due passi (`FusedAddMulEquivalent`, `FusedMulAddEquivalent`, `ScaleShiftEquivalent`), più `ShapeMismatchThrows`.

### 4.2 Fused operations su CPU ✅

Il bug latente è risolto. `apply` e `apply_binary` in [`headers/tensor.inl`](../headers/tensor.inl) fanno branch su `device_type()` ed eseguono un loop host quando il tensore è su CPU. Non c'è una seconda implementazione da mantenere allineata: tutti i functor sono annotati `__host__ __device__`, quindi **lo stesso oggetto `op` guida sia il kernel sia il loop CPU**.

### 4.3 Functor `ReLU`, `Sigmoid`, `Tanh` ⚠️

`ReLU<T>` e `Sigmoid<T>` sono implementati, istanziati per `float`/`int`/`char`/`float16_t`, esposti come `Tensor::relu()` / `Tensor::sigmoid()` con i rispettivi overload sullo stream, e disponibili da Python. `Tanh` non è implementato.

Entrambi passano per `float` internamente (`static_cast<float>(x) > 0.0f`, `expf(-static_cast<float>(x))`): è ciò che li fa funzionare senza modifiche su `float16_t`.

### 4.4 Overload sullo stream per `apply_binary` ✅

Risolto come effetto collaterale di §8: `apply_binary` è stato riscritto sopra `apply_binary_out(rhs, op, out, s)`, che uno stream lo prende per costruzione, e le forme allocante e in-place ci delegano. `scale_shift`, `shift_scale` e i quattro `fused_*_*` restano sincroni sullo stream di default perché non hanno un overload proprio — chiamarli su uno stream significa passare per `apply`/`apply_binary` con lo stesso functor.

---

## 5. Matmul — estensioni

Stato attuale: `matmul` è **2D-only** in entrambi i backend. Rank != 2 o dimensioni interne incompatibili sollevano un'eccezione da `Tensor::matmul` e di nuovo dentro `matmul_cpu`. Nessun batching, nessun broadcasting. Ha l'overload `(const Tensor&, const Stream&)`.

**Prestazioni CPU, aggiornamento settembre 2026 ✅.** `matmul_cpu` era il triplo loop `ijk` da manuale, che indicizza `rhs(k, j)` lungo una colonna — un cache miss per iterazione interna — senza blocking, SIMD o threading: **1.81 GFLOP/s** a 1024³, 421× dietro NumPy. Ora è ordine `ikj` con tiling L2 a 128 e `omp parallel for` sulle righe di output (indipendenti fra loro). Scorrere `k` in mezzo fa sì che sia la riga di `rhs` sia quella di `dst` vengano spazzate in modo contiguo nel loop `j` più interno — stride unitario, quindi vettorizzabile — e il blocking `i`/`k`/`j` tiene ogni pannello residente in L2 durante l'accumulo.

| CPU matmul | OpenMat GFLOP/s | NumPy GFLOP/s | rapporto |
|---|---|---|---|
| 128³ | **196.6** | 189.2 | **1.04× più veloce** |
| 512³ | 140.6 | 666.8 | 4.7× più lento |
| 1024³ | 122.8 | 768.3 | 6.3× più lento |

**68× a 1024³**, e a 128³ supera NumPy (lì la chiamata è abbastanza piccola che l'overhead di dispatch di OpenBLAS pesa quanto il lavoro). Quel che resta è la distanza fra un loop C tiled e un microkernel scritto a mano: OpenBLAS fa blocking anche per L1 e per i registri, emette intrinsics NEON con software pipelining, e impacchetta i pannelli in buffer contigui. È una classe di implementazione diversa, non una differenza di tuning — ma ora è un fattore 6, non tre ordini di grandezza.

Sul lato GPU il divario resta **10.3×** rispetto a cuBLAS (1636 vs 16869 GFLOP/s a 1024³) e ha tre cause distinte, tutte ancora aperte: un solo elemento di output per thread (due load da shared memory per FMA, quindi il tetto è il rate di issue LDS, non quello FFMA — il register blocking 4×4 è la vittoria singola più grande), nessun double buffering (le due barriere per tile rendono il loop strettamente load → sync → compute → sync), e nessun uso dei tensor core. Quest'ultimo punto è misurabile direttamente: in `test_benchmarks` fp16 rende solo ~1.1× rispetto a fp32, dove i tensor core varrebbero diversi×. Dettagli in [`benchmark_report.md` §6](../benchmark_report.md#6-matmul-the-cpu-gap-closed-68-the-gpu-one-remains).

### 5.1 Batch matmul ❌

```cpp
// A: (B, M, K), B: (B, K, N) → C: (B, M, N)
Tensor<T> bmm(const Tensor<T>& rhs) const;
```

Il kernel batched esegue `B` matmul indipendenti in parallelo usando `blockIdx.z` come batch index. In alternativa si può usare `cublasGemmStridedBatchedEx`.

### 5.2 Integrazione con cuBLAS (opzionale) ❌

Per matrici grandi (≥ 512×512) cuBLAS è significativamente più veloce del kernel tiled attuale. Introdurre un path condizionale:

```cpp
// se min(M,N,K) >= CUBLAS_THRESHOLD → cublasSgemm
// altrimenti → launch_matmul (kernel tiled attuale)
```

Aggiunge la dipendenza `cublas` in `CMakeLists.txt`. Da tenere presente: cuBLAS è **column-major**, quindi la convenzione row-major della libreria va gestita scambiando gli operandi (`C^T = B^T · A^T`) invece di trasporre i dati. Inoltre `cublasSetStream` va allineato all'`om::Stream` passato, e cuBLAS copre solo `float`/`double`/`__half` — `int` e `char` restano sul kernel tiled.

---

## 6. Infrastruttura

### 6.1 Stream CUDA ✅

Implementato e diventato **il percorso di esecuzione canonico** della libreria, non un'aggiunta opzionale.

```cpp
auto c = a + b;         // delega a a.add(b, Stream::default_stream())
auto c = a.add(b, s);   // enqueue su s; il chiamante deve fare s.synchronize()
```

[`om::Stream`](../headers/stream.h) è un wrapper RAII move-only: il costruttore di default chiama `cudaStreamCreate` e possiede l'handle; `Stream(cudaStream_t)` avvolge un handle esistente senza possederlo; `Stream::default_stream()` restituisce un wrapper non-owning su `nullptr` — ed è così che l'API sincrona riusa il percorso asincrono con zero duplicazione.

**Invariante di ownership dello stream:** la memoria di `cudaMallocAsync` appartiene a un pool stream-ordered e va liberata sullo stream su cui è stata allocata. Per questo ogni `Tensor` memorizza `m_Stream` e il distruttore chiama `deallocate_async(m_Data, m_Stream.get())`. La violazione si manifesta come illegal memory access lontano dal punto di chiamata reale.

Regola per chi aggiunge operazioni: **implementare l'overload `(args, const Stream&)` e rendere quello senza stream un delegate di una riga.** Farlo al contrario rompe l'invariante su cui è costruito tutto `tensor.inl`.

Suite di test: `test_streams`, `test_allocator_stream`, `test_stress`, `test_stream_perf`. Numeri misurati in [`README.md`](../README.md) e [`stream_perf_report.md`](../stream_perf_report.md) (build Release — in Debug i numeri non significano nulla).

### 6.2 Gestione degli errori CUDA ✅

`CUDA_CHECK` in [`headers/cuda_defines.cuh`](../headers/cuda_defines.cuh) include **file e riga** nel messaggio, ed è affiancato da `CUDA_CALL(expr)` che controlla il valore di ritorno di una singola chiamata riportando anche l'espressione fallita.

Il limite di fondo — `cudaGetLastError()` dopo il lancio intercetta solo gli errori **sincroni** di lancio, mentre gli errori asincroni dentro il kernel (accessi fuori dai limiti, illegal address) emergono al successivo punto di sincronizzazione, arbitrariamente lontano dal kernel colpevole — è ora coperto da una **modalità debug**.

`CUDA_CHECK_LAUNCH(nome_kernel, stream)` sostituisce `CUDA_CHECK` in tutti e 13 i siti di lancio. Fa sempre il controllo sincrono; in modalità debug forza in più un `cudaStreamSynchronize` sullo stream del lancio, così l'errore asincrono viene attribuito al kernel e al sito che lo ha lanciato:

```
[CUDA ASYNC ERROR] kernel 'add_kernel_rank1' at src/ops/kernels/binary_ops.cu:10
  in void om::launch_add(...) [with T = float; ...]
  on stream 0xb0149ef4a290
  → an illegal memory access was encountered
  (caught by OPENMAT_DEBUG_SYNC forced synchronization; unset it to restore asynchronous execution)
```

Attivazione:

| | |
|---|---|
| a runtime, senza ricompilare | `export OPENMAT_DEBUG_SYNC=1` (accetta anche `true`/`yes`/`on`) |
| di default nella build | `cmake -DOM_DEBUG_SYNC=ON` — la variabile d'ambiente la sovrascrive in entrambe le direzioni, quindi `OPENMAT_DEBUG_SYNC=0` la disattiva |
| eliminata a compile time | `cmake -DOM_NO_DEBUG_SYNC=ON` — resta solo il controllo sincrono |

La sincronizzazione forzata serializza esattamente la sovrapposizione per cui gli stream esistono: è una modalità diagnostica, mai un default. Dopo un illegal access il contesto CUDA è comunque inutilizzabile — il punto è sapere **dove** è successo.

L'implementazione sta in un'unica unità di traduzione ([`src/cuda_debug.cpp`](../src/cuda_debug.cpp)), non `inline` nell'header: entrambi gli switch sono condizionali del preprocessore, e una definizione inline darebbe a un consumatore compilato con impostazioni diverse una seconda definizione in conflitto — violazione della ODR che il linker risolve scegliendone una in silenzio. Il costo di una chiamata non inline è nulla rispetto ai microsecondi di un lancio di kernel.

Resta possibile (non fatto): la cattura di uno stack trace al punto di lancio da riportare quando l'errore emerge dopo. Con `CUDA_CHECK_LAUNCH` è però molto meno necessaria — `compute-sanitizer --tool memcheck` copre il resto.

### 6.3 Stampa e serializzazione ❌

```cpp
std::ostream& operator<<(std::ostream& os, const Tensor<T>& t);
void save(const std::string& path) const;       // formato binario raw
static Tensor<T> load(const std::string& path); // load corrispondente
```

Nessuno dei tre esiste in C++. `operator<<` deve copiare i dati su host se il tensore è su GPU prima di stampare — cosa oggi già possibile con `.cpu()`.

Sul lato Python la lacuna è parzialmente coperta: `Tensor` ha `__repr__` / `__str__`, `tolist()`, `numpy()` e i buffer protocol (`__array_interface__` per i tensori host, `__cuda_array_interface__` per quelli CUDA), quindi la serializzazione è delegabile a NumPy.

### 6.4 Parallelismo CPU (OpenMP) ⚠️

Area non presente nella roadmap originale, aggiunta dopo di essa.

`DEFINE_BINARY_OPS_CPU` ([`headers/ops/cpu/binary_op_macros.h`](../headers/ops/cpu/binary_op_macros.h)) e `DEFINE_UNARY_OPS_CPU` ([`headers/ops/cpu/unary_op_macros.h`](../headers/ops/cpu/unary_op_macros.h)) — il lato CPU di `add`/`sub`/`mul`/`div`, sia tensore⊕tensore sia tensore⊕scalare — usano ora `#pragma omp parallel for schedule(static) if(_total > 65536)`. Essendo entrambi un unico punto di generazione macro, tutte le operazioni beneficiano insieme. Sotto soglia il loop resta scalare (nessun costo di fork/join pagato sulle dimensioni piccole, misurato entro il 5%); sopra soglia, **1.7–11.6×** più veloce su una macchina di riferimento a 20 thread, al punto da superare NumPy in valore assoluto di 2.9–3.1× a 16M elementi. Il guadagno è massimo a 1M (5.3–11.6×), dove il working set sta in L2 e il tetto era il loop scalare; a 16M si comprime a ~2×, che è il sistema di memoria a parlare. `matmul_cpu` ([`headers/ops/cpu/matmul_cpu.h`](../headers/ops/cpu/matmul_cpu.h)) aveva già ricevuto lo stesso trattamento in precedenza (tiling `ikj` + `omp parallel for` sulle righe). Dati completi: [`benchmark_report.md` §7](../benchmark_report.md#7-cpu-elementwise-ops-were-single-threaded--openmp-closes-most-of-it).

**Tentativo non riuscito, documentato apposta per non essere ritentato senza ri-misurare:** `#pragma omp simd reduction(min:)`/`reduction(max:)` su `reduce_min_cpu`/`reduce_max_cpu` ([`headers/ops/cpu/reduce_cpu.h`](../headers/ops/cpu/reduce_cpu.h)) è stato aggiunto e poi **rimosso** — misurato ~1.6× più lento del semplice loop scalare a 16M elementi (isolato, stesso corpo di loop, stesse flag `-O3 -march=native -fopenmp`), perché GCC 13 vettorizza già automaticamente l'idioma branch-and-select sotto `-O3` da solo (confermato con `-fopt-info-vec-optimized`), e la clausola `reduction` forza una lowering diversa e peggiore sopra un loop già vettorizzato. `reduce_sum_cpu` non è coinvolto: il suo guadagno viene da una ristrutturazione a livello sorgente (8 accumulatori indipendenti), non da un pragma, e resta invariato.

`apply`/`apply_binary` ([`headers/tensor.inl`](../headers/tensor.inl), §4.2) **non** hanno ricevuto lo stesso trattamento — restano un loop scalare a singolo thread su CPU indipendentemente dalla dimensione del tensore, quindi `relu`, `sigmoid`, `scale_shift`, `shift_scale` e i quattro `fused_*` non beneficiano ancora del parallelismo che hanno `add`/`sub`/`mul`/`div`. Verificato misurando a 1 e a 20 thread: tempi identici (`relu` 2.343 vs 2.344 ms a 16M, `x*s+t` 2.356 vs 2.314 ms).

Vale la pena notare che **battono comunque NumPy** — `relu` 2.22×, `x*s+t` 3.49× a 16M — grazie alla sola fusione (§4), il che rende questo un'occasione mancata più che un difetto. È segnato ⚠️ (non ❌) perché la forma del loop è identica e lo stesso `if(_total > N)` si applicherebbe senza modifiche: è lavoro meccanico rimasto indietro, non un problema di design. Sulla base dei numeri di §7.1 dovrebbe valere circa 2× a 16M.

**Divario aperto residuo:** `reduce_min_cpu`/`reduce_max_cpu` restano 2.7× più lenti di NumPy a 16M (6.8× a 1M) — il più grande rimasto sulla superficie elementwise CPU, ora che `reduce_sum_cpu` è stato sistemato (§1.2). La leva non ancora provata è un `parallel for` con `reduction(min:)`/`(max:)` **fra thread**, che è una modifica diversa dalla clausola `simd` misurata e rimossa qui sopra.

---

## 7. Layer Python ✅

Area non presente nella roadmap originale, completata dopo di essa.

Il package è un binding **ctypes** (non pybind) sopra il layer C-ABI compilato dentro `OpenMat.so`. Il confine è [`src/python/openmat_capi.cpp`](../src/python/openmat_capi.cpp); la superficie per-dtype vive in [`src/python/openmat_capi_impl.inc`](../src/python/openmat_capi_impl.inc), **un solo corpo incluso due volte** con `OM_T`/`OM_SFX` diversi. Aggiungere un dtype = aggiungere un blocco `#define`/`#include`/`#undef`, a patto che i kernel siano istanziati per quel tipo.

Coperto oggi: `float32` e `int32`, l'intera superficie tensoriale (metadati, indexing, factory, aritmetica, riduzioni, shape ops, transpose/permute, fused ops), gli stream, e la API runtime dtype-independent (`om_cuda_device_count`, `om_cuda_is_available`, `om_device_synchronize`, `om_stream_*`).

**Gli stream sono reference-counted al confine C** (`StreamBox`), non in Python. È deliberato: il collector ciclico di Python finalizza i membri di un ciclo in ordine arbitrario, quindi un oggetto `Stream` potrebbe essere distrutto prima dei tensori la cui memoria appartiene ancora a quello stream. Tenere una reference Python non basta — due tentativi in quel senso hanno prodotto segfault sotto `gc.collect()`.

Suite di test: `python/tests/test_tensor.py`, `test_tensor_api.py`, `test_dtypes.py`, `test_streams.py`.

**Lavoro residuo:**
- `Tensor<double>` e `Tensor<char>` non sono esportati. `char` è immediato (i kernel ci sono); `double` richiede prima le istanziazioni GPU, che oggi non esistono per nessuna operazione.
- `__getitem__` restituisce solo scalari — lo slicing dipende da 2.3.
- Le riduzioni non accettano `stream` (dipende da 1.2).

---

## 8. Operazioni in-place e destinazione fornita dal chiamante ✅

Ogni metodo di `Tensor` era `const` e allocava il risultato. In un ciclo di training questo significa un'allocazione **e** una deallocazione per operazione per iterazione, e un picco di memoria pari a tutto il grafo dell'espressione invece che al working set. PyTorch espone `add_`, `mul_` e `out=` esattamente per questo.

I launcher accettavano già una `dst` separata, quindi il lavoro era tutto sulla superficie pubblica. Ogni operazione ha ora tre forme, di cui **una sola è un'implementazione vera**:

```cpp
auto c = a.add(b, s);        // alloca il risultato, poi chiama add_out
a.add_out(b, out, s);        // ← il corpo: scrive in una destinazione che esiste già
a.add_(b, s);                // == a.add_out(b, a, s)
```

`add_out` restituisce `Tensor&` (la destinazione) e `add_` restituisce `*this`, quindi le chiamate si concatenano; le forme senza stream sono delegate di una riga con `Stream::default_stream()`. Ci sono anche `operator+=`/`-=`/`*=`/`/=`.

**Famiglie coperte:** `add`/`sub`/`mul`/`div` (tensore e scalare), `apply`, `apply_binary`, `relu`, `sigmoid`, `fill_`. `matmul`, `transpose` e `permute` hanno solo `_out`: i loro kernel leggono elementi che non scrivono, quindi una destinazione che condivide il buffer con un operando leggerebbe valori già sovrascritti — `_check_alias_none` lo rifiuta al call site invece di restituire un risultato plausibile e sbagliato.

**Perché la famiglia elementwise *può* fare aliasing:** il loop CPU, il fast path contiguo GPU e i kernel per rank leggono l'indice i e scrivono l'indice i, quindi `dst == lhs` è corretto esattamente quanto un buffer separato. Questo dipende dal fatto che ogni buffer sia una singola sequenza piatta: `_check_alias_elementwise` ricontrolla `is_contiguous()`, così il giorno in cui una view con stride potrà puntare a una regione *diversa* della stessa allocazione (§2.1b) l'operazione solleva un'eccezione invece di calcolare la cosa sbagliata.

Conseguenza collaterale: i tre kernel di [`contiguous.cuh`](../headers/ops/kernels/contiguous.cuh) hanno perso `__restrict__`, che è precisamente la promessa che l'in-place viola. Misurato, non è costato nulla (`add` a 16M da 230.2-233.4 a 228.6-230.7 GB/s, `relu` da 232.9-234.6 a 233.4-235.5): il percorso read-only viene dall'`__ldg` esplicito in `device_load`, non da inferenza sul restrict.

Il `_out` è anche il punto in cui vive ora la validazione degli operandi (shape e device controllati prima del dispatch), quindi un `add` GPU fra shape diverse solleva un'eccezione invece di leggere oltre la fine di un buffer come faceva prima.

**Numeri** ([`benchmark_report.md` §9](../benchmark_report.md#9-every-op-allocated-its-own-result--in-place-and-out-forms-added), harness `scripts/bench_inplace.py`): 1.2-1.6× sotto i ~64 K elementi su entrambi i backend, dove l'allocazione è una frazione grande dell'operazione, che si riduce a 1.04× a 16 M dove domina il kernel e `HostPool`/`cudaMallocAsync` stanno già riciclando il blocco. L'argomento sulla memoria invece non si riduce: `tests/test_inplace.cpp` e `python/tests/test_inplace.py` verificano che `data_ptr` non cambi lungo un ciclo di 100 passi — la proprietà che un'implementazione corretta ma ri-allocante fallirebbe superando ogni controllo sui valori.

**Non fatto:** `matmul` batch/in-place resta fuori (§5.1), e i quattro `fused_*_*` non hanno una forma in-place dedicata — `apply_binary_(rhs, BinaryCompose<…>{…})` con lo stesso functor è il modo di ottenerla.

---

## Priorità suggerite

| Done | Priorità | Item |
|---|---|---|
| ✅ | Alta | 4.2 — CPU path per fused ops (bug latente) |
| ✅ | Alta | 4.1 — Test per le API fused |
| ✅ | Alta | 3.1 — `.to()` / `.cpu()` / `.cuda()` |
| ✅ | Alta | 3.2 — `zeros`, `ones`, `full`, `from_vector` |
| ✅ | Media | 1.2 — Riduzioni a scalare (`sum`, `mean`, `min`, `max`) |
| ✅ | Media | 2.1 — `reshape`, `flatten`, `squeeze`, `unsqueeze` (a copia) |
| ✅ | Media | 4.3 — `relu`, `sigmoid` |
| ✅ | Media | 2.2 — `transpose`, `permute` |
| ✅ | Bassa | 6.1 — CUDA streams |
| ✅ | — | 7 — Layer Python (ctypes, float32 + int32, stream) |
| ✅ | — | 6.2 — Errori CUDA: file/riga, `CUDA_CALL`, e `OPENMAT_DEBUG_SYNC` per gli errori asincroni |
| ⚠️ | — | 6.4 — Parallelismo CPU (OpenMP) su `add`/`sub`/`mul`/`div`, soglia 65536 elementi; `apply`/`apply_binary` non ancora esteso |
| ✅ | — | 6.5 — Fast path contiguo per i kernel elementwise: indicizzazione lineare al posto del layout per rank, ogni rank alla banda del rank 1 (verificato: spread 0.4% su `add`, 1.5% su `relu`, rank 1–5) |
| ✅ | — | 6.6 — `matmul_cpu` `ikj` + tiling L2 + OpenMP: 1.81 → 123 GFLOP/s a 1024³ (§5) |
| ✅ | — | 6.7 — `reduce_sum_cpu` a 8 lane: 7.7 → 36.9 GB/s, ora davanti a NumPy (§1.2) |
| ✅ | — | 6.8 — `HostPool`/`PinnedHostPool`: round-trip 100 × 64 MB da 4.3 a 51.1 GB/s |
| ✅ | — | 8 — Operazioni in-place (`add_`, `mul_`, `relu_`, `fill_`) e overload `_out` con destinazione fornita dal chiamante |
| ✅ | — | 4.4 — Overload stream per `apply_binary` (arrivato con §8: `apply_binary_out` prende lo stream per costruzione) |
| | **Alta** | 1.1 — Unarie `abs`, `sqrt`, `exp`, `log` via `apply` (costo basso, alto valore) |
| | **Alta** | 6.4b — OpenMP su `apply`/`apply_binary`: stesso `if(_total > 65536)` di 6.4, ~2× atteso a 16M su tutta la famiglia fused |
| | Media | 6.4c — `min`/`max` con `parallel for reduction(min:)/(max:)` fra thread (2.7× dietro NumPy); **non** la clausola `simd`, già misurata e rimossa |
| | Media | 6.4d — `sum` multi-thread con `reduction(+:)` (PyTorch è 2.8× avanti a 16M grazie a questo) |
| | Media | 5.3 — Register blocking 4×4 nel kernel matmul GPU: due load da shared per FMA sono il tetto attuale, 10.3× dietro cuBLAS |
| | Media | 6.3 — `operator<<`, `save`, `load` |
| | Media | 1.2b — Riduzioni per asse `sum(axis)`, `mean(axis)` |
| | Media | 3.2b — `arange` / `linspace` nativi in C++ |
| | Media | 1.3 — Operazioni di confronto (richiede un launcher con tipo di uscita diverso) |
| | Bassa | 2.1b — Decisione sul modello di ownership per le view (sblocca 2.3); il fast path di 6.5 è già protetto da `TensorView::is_contiguous()`, quindi una view con stride non compatti ricade sui kernel per rank invece di leggere gli elementi sbagliati |
| | Bassa | 2.3 — `slice`, `operator[]` |
| | Bassa | 5.1 — Batch matmul (`bmm`) |
| | Bassa | 5.2 — Integrazione cuBLAS |
| | Bassa | 3.3 — Inizializzazione random (cuRAND) |
