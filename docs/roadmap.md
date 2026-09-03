# OpenMat — Roadmap delle implementazioni future

Questo documento elenca le funzionalità del framework, il loro stato attuale e il lavoro ancora da fare, raggruppate per area. Ogni voce include una breve descrizione tecnica e i file coinvolti.

**Legenda:** ✅ implementato · ⚠️ implementato parzialmente o con divergenze rispetto al progetto originale · ❌ da fare

Ultimo allineamento al codice: settembre 2025.

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

### 4.4 Overload sullo stream per `apply_binary` ❌

Asimmetria residua: `apply` ha l'overload `(Op, const Stream&)`, `apply_binary` no. Di conseguenza `scale_shift`, `shift_scale` e i quattro `fused_*_*` sono chiamate sincrone sullo stream di default, mentre `relu` e `sigmoid` no.

Il lavoro è meccanico e segue `apply`: accettare `const Stream& s`, costruire l'output con il costruttore privato `Tensor(shape, device, Stream)` e passare `s.get()` al launcher — `launch_apply_binary_op` accetta già un `cudaStream_t`.

---

## 5. Matmul — estensioni

Stato attuale: `matmul` è **2D-only** in entrambi i backend. Rank != 2 o dimensioni interne incompatibili sollevano un'eccezione da `Tensor::matmul` e di nuovo dentro `matmul_cpu`. Nessun batching, nessun broadcasting. Ha l'overload `(const Tensor&, const Stream&)`.

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
| | **Alta** | 4.4 — Overload stream per `apply_binary` (chiude l'asimmetria di 6.1) |
| | **Alta** | 1.1 — Unarie `abs`, `sqrt`, `exp`, `log` via `apply` (costo basso, alto valore) |
| | Media | 6.3 — `operator<<`, `save`, `load` |
| | Media | 1.2b — Riduzioni per asse `sum(axis)`, `mean(axis)` |
| | Media | 3.2b — `arange` / `linspace` nativi in C++ |
| | Media | 1.3 — Operazioni di confronto (richiede un launcher con tipo di uscita diverso) |
| | Bassa | 2.1b — Decisione sul modello di ownership per le view (sblocca 2.3) |
| | Bassa | 2.3 — `slice`, `operator[]` |
| | Bassa | 5.1 — Batch matmul (`bmm`) |
| | Bassa | 5.2 — Integrazione cuBLAS |
| | Bassa | 3.3 — Inizializzazione random (cuRAND) |
