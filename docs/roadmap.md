# OpenMat — Roadmap delle implementazioni future

Questo documento elenca le funzionalità da aggiungere al framework, ordinate per priorità e raggruppate per area. Ogni voce include una breve descrizione tecnica e i file coinvolti.

---

## 1. Operazioni matematiche mancanti

### 1.1 Operazioni element-wise

Le seguenti operazioni seguono esattamente lo stesso pattern di `add`, `sub`, `mul`, `div` già esistenti in `binary_op_macros.cuh` / `binary_op_macros.h`. Per ognuna basta definire kernel + CPU functor + dispatch.

| Operazione | Espressione | Note |
|---|---|---|
| `pow(rhs)` | `lhs[i] ^ rhs[i]` | Usare `powf` per float, `pow` per double |
| `min(rhs)` | `min(lhs[i], rhs[i])` | Usare `fminf` su GPU |
| `max(rhs)` | `max(lhs[i], rhs[i])` | Usare `fmaxf` su GPU |
| `abs()` | `|x[i]|` | Unaria, usare `fabsf` su GPU |
| `sqrt()` | `√x[i]` | Unaria, usare `sqrtf` su GPU |
| `exp()` | `e^x[i]` | Unaria, usare `expf` su GPU |
| `log()` | `ln(x[i])` | Unaria, usare `logf` su GPU |

**File da modificare:**
- `headers/ops/kernels/binary_op_macros.cuh` — aggiungere `DEFINE_BINARY_OP_KERNEL_K1/K2/K3/K4/ND` + `DEFINE_BINARY_OP_LAUNCH` + `DEFINE_BINARY_OP_LAUNCH_FRW_DEC`
- `src/ops/kernels/binary_ops.cu` — translation unit per i nuovi kernel
- `headers/ops/cpu/binary_op_macros.h` — `DEFINE_BINARY_OPS_CPU`
- `src/ops/cpu/binary_ops.cpp` — implementazione CPU
- `headers/kernel_launcher.h` — `DEFINE_DEVICE_DISPATCH_BINARY_H` per ognuno
- `headers/kernel_launcher.inl` — `DEFINE_DEVICE_DISPATCH_BINARY_INL`
- `headers/tensor.cuh` / `headers/tensor.inl` — metodi pubblici `pow()`, `min()`, ecc.

---

### 1.2 Operazioni di riduzione

Oggi non esiste nessuna operazione che riduca un tensore a un singolo scalare o a un tensore di dimensione inferiore.

| Operazione | Risultato | Descrizione |
|---|---|---|
| `sum()` | scalare | Somma tutti gli elementi |
| `mean()` | scalare | Media di tutti gli elementi |
| `min()` | scalare | Elemento minimo |
| `max()` | scalare | Elemento massimo |
| `sum(axis)` | tensore rank-1 | Somma lungo una dimensione |
| `mean(axis)` | tensore rank-1 | Media lungo una dimensione |

Le riduzioni su GPU richiedono un pattern a due fasi (riduzione in shared memory per blocco, poi riduzione tra blocchi). Il pattern classico usa `__shared__` con tree-reduction e warp shuffle (`__shfl_down_sync`).

**File da creare:**
- `headers/ops/kernels/reduce_gpu.cuh`
- `src/ops/kernels/reduce_gpu.cu`
- `headers/ops/cpu/reduce_cpu.h`
- `src/ops/cpu/reduce_cpu.cpp`

---

### 1.3 Operazioni di confronto element-wise

| Operazione | Espressione |
|---|---|
| `eq(rhs)` | `lhs[i] == rhs[i]` → tensore di `int` (0/1) |
| `lt(rhs)` | `lhs[i] < rhs[i]` |
| `gt(rhs)` | `lhs[i] > rhs[i]` |
| `clamp(min, max)` | `min ≤ x[i] ≤ max` |

Seguono il pattern binario esistente ma restituiscono `Tensor<int>` (maschera booleana).

---

## 2. Operazioni sulle dimensioni

### 2.1 Reshape e view

```cpp
Tensor<T> reshape(const std::vector<size_t>& new_shape) const;
Tensor<T> flatten() const;   // reshape({size()})
Tensor<T> squeeze(size_t axis) const;   // rimuove una dimensione di size 1
Tensor<T> unsqueeze(size_t axis) const; // aggiunge una dimensione di size 1
```

`reshape` è zero-copy se il tensore è contiguo in memoria (i.e., i stride sono row-major standard): basta costruire un nuovo `TensorView` con la nuova shape. Se non è contiguo, occorre copiare.

**File da modificare:** `headers/tensor.cuh`, `headers/tensor.inl`.

---

### 2.2 Transpose

```cpp
Tensor<T> transpose() const;               // solo per rank 2
Tensor<T> permute(std::vector<size_t> axes) const; // ordine arbitrario degli assi
```

`transpose` di una matrice 2D può essere implementata come permutazione degli stride (vista logica) senza copiare dati. `permute` generalizza questo a rank N.

**File da creare:** `headers/ops/kernels/transpose_gpu.cuh`, `src/ops/kernels/transpose_gpu.cu`.

---

### 2.3 Slice e indexing

```cpp
Tensor<T> slice(size_t axis, size_t start, size_t end) const;
Tensor<T> operator[](size_t index) const; // selezione lungo il primo asse
```

Uno slice è un `TensorView` con pointer offset e shape/stride aggiornati, zero-copy per tensori contigui.

---

## 3. Gestione della memoria

### 3.1 Trasferimento CPU ↔ GPU come metodi `Tensor`

Oggi `copyToHost` e `copyToDevice` scrivono su un buffer raw (`T*`) passato dall'utente. Mancano metodi che restituiscano un nuovo `Tensor` sul device di destinazione:

```cpp
Tensor<T> to(const Device& target) const;
Tensor<T> cpu() const;   // shorthand per to(Device("cpu:0"))
Tensor<T> cuda() const;  // shorthand per to(Device("cuda:0"))
```

**File da modificare:** `headers/tensor.cuh`, `headers/tensor.inl`.

---

### 3.2 Inizializzazione dei tensori

Oggi l'unico modo per inizializzare i valori è `fill(value)` o assegnare elemento per elemento dal host. Mancano:

```cpp
static Tensor<T> zeros(const std::vector<size_t>& shape, const Device& dv);
static Tensor<T> ones(const std::vector<size_t>& shape, const Device& dv);
static Tensor<T> full(const std::vector<size_t>& shape, T value, const Device& dv);
static Tensor<T> arange(T start, T stop, T step, const Device& dv);
static Tensor<T> linspace(T start, T stop, size_t n, const Device& dv);
static Tensor<T> from_vector(const std::vector<T>& data, const std::vector<size_t>& shape, const Device& dv);
```

`zeros`, `ones`, `full` sono triviali sopra `fill`. `arange` e `linspace` richiedono un kernel dedicato (o un loop CPU).

---

### 3.3 Inizializzazione random

```cpp
static Tensor<T> rand_uniform(const std::vector<size_t>& shape, T low, T high, const Device& dv);
static Tensor<T> rand_normal(const std::vector<size_t>& shape, T mean, T std, const Device& dv);
```

Su GPU si usa cuRAND (`curandGenerateUniform`, `curandGenerateNormal`). Aggiunge una dipendenza da `libcurand`.

---

## 4. Fused operations — estensioni

### 4.1 Test per le nuove API

Le API `apply`, `scale_shift`, `shift_scale`, `apply_binary`, `fused_add_mul`, ecc. non hanno test in `tests/tensor_ops_test.cpp`. Aggiungere:

- Correttezza numerica su valori noti (CPU non applicabile — `launch_apply_op` è solo CUDA)
- Verifica che il risultato di `a.fused_add_mul(b, 2.0f)` sia identico a `(a + b) * 2.0f` calcolato in due passi separati
- Test per rank 1, 2, 3 per coprire i kernel rank-specializzati

---

### 4.2 Fused operations su CPU

`launch_apply_op` e `launch_apply_binary_op` esistono solo come kernel CUDA. Se il tensore è su CPU, chiamarli causa undefined behavior (il kernel viene lanciato su dati host). Aggiungere un path CPU:

```cpp
// In tensor.inl — apply():
if (this->device_type() == DEVICE_TYPE::CPU) {
    size_t n = out.size();
    for (size_t i = 0; i < n; ++i)
        out_view[i] = op(src_view[i]);
} else {
    launch_apply_op<value_type>(this->view(), out.view(), op);
}
```

---

### 4.3 Functor `ReLU`, `Sigmoid`, `Tanh`

Aggiungere i functor unari più comuni in `fused_op.cuh` e le relative istanziazioni esplicite:

```cpp
template <typename T>
struct ReLU {
    __device__ T operator()(T x) const { return x > static_cast<T>(0) ? x : static_cast<T>(0); }
};

template <typename T>
struct Sigmoid {
    __device__ T operator()(T x) const { return static_cast<T>(1) / (static_cast<T>(1) + expf(-float(x))); }
};
```

E i metodi corrispondenti su `Tensor<T>`:
```cpp
Tensor<value_type> relu() const;
Tensor<value_type> sigmoid() const;
```

---

## 5. Matmul — estensioni

### 5.1 Batch matmul

```cpp
// A: (B, M, K), B: (B, K, N) → C: (B, M, N)
Tensor<T> bmm(const Tensor<T>& rhs) const;
```

Il kernel batched esegue `B` matmul indipendenti in parallelo usando la dimensione `blockIdx.z` per il batch index. In alternativa si può usare `cublasGemmStridedBatchedEx`.

---

### 5.2 Integrazione con cuBLAS (opzionale)

Per matrici grandi (≥ 512×512) cuBLAS è significativamente più veloce del kernel tiled attuale. Introdurre un path cuBLAS condizionale:

```cpp
// In kernel_launcher.h — matmul_dispatch<CUDA>:
// se min(M,N,K) >= CUBLAS_THRESHOLD → cublasSgemm
// altrimenti → launch_matmul (kernel tiled attuale)
```

Aggiunge la dipendenza `cublas` in `CMakeLists.txt`.

---

## 6. Infrastruttura

### 6.1 Stream CUDA

Tutti i kernel oggi chiamano `cudaDeviceSynchronize()` dopo ogni lancio — questo serializza l'intera GPU dopo ogni operazione. Introdurre il supporto a CUDA streams per permettere overlapping di operazioni:

```cpp
class Stream {
    cudaStream_t m_stream;
public:
    Stream();
    ~Stream();
    void synchronize() const;
    cudaStream_t get() const { return m_stream; }
};
```

Ogni `launch_*` accetterebbe un parametro `cudaStream_t stream = 0` opzionale. La sincronizzazione diventerebbe responsabilità dell'utente o di un contesto.

---

### 6.2 Gestione degli errori CUDA

`CUDA_CHECK` in `cuda_defines.cuh` controlla solo dopo il lancio del kernel. Gli errori asincroni nei kernel (out-of-bounds in debug, divisione per zero, ecc.) vengono catturati solo al `cudaDeviceSynchronize()` successivo e il messaggio di errore non include stack trace. Miglioramento: usare `cudaGetLastError()` + `cudaPeekAtLastError()` e lanciare eccezioni con file/line.

---

### 6.3 Stampa e serializzazione

```cpp
std::ostream& operator<<(std::ostream& os, const Tensor<T>& t);
void save(const std::string& path) const;       // formato binario raw
static Tensor<T> load(const std::string& path); // corrispondente load
```

`operator<<` deve copiare i dati su host se il tensore è su GPU prima di stampare.

---

## Priorità suggerite

|Done| Priorità | Item |
|---|---|---|
|x| Alta | 4.2 — CPU path per fused ops (bug latente) |
|x| Alta | 4.1 — Test per le API fused |
|x| Alta | 3.1 — `.to()` / `.cpu()` / `.cuda()` |
|x| Alta | 3.2 — `zeros`, `ones`, `from_vector` |
|x| Media | 1.2 — Riduzioni (`sum`, `mean`, `max`) |
|x| Media | 2.1 — `reshape`, `flatten` |
| | Media | 4.3 — `relu`, `sigmoid` |
| | Media | 2.2 — `transpose`, `permute` |
| | Bassa | 6.1 — CUDA streams |
| | Bassa | 5.2 — cuBLAS integration |
| | Bassa | 3.3 — Inizializzazione random (cuRAND) |
