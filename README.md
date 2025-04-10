<p align="center">
  <img src="images/logo.png" alt="OpenMat Logo" width="300"/>
</p>

# OpenMat


**OpenMat** is a didactic open-source framework for performing **matrix operations accelerated with CUDA**, designed to help developers learn and explore **low-level programming in C/C++ and Python**, with a special focus on **GPU computing** and **performance-critical applications**.

## 🧠 Purpose

This project is meant as both a **learning tool** and a **performance benchmark**. It provides:

- Implementations of common matrix operations using **CUDA C/C++**.
- Custom **memory allocators** tailored for matrix data structures.
- Opportunities to integrate **AI models** or numerical methods into **high-performance GPU workflows**.

## 🔥 Key Features

- 🚀 GPU-accelerated matrix operations using **NVIDIA CUDA**.
- 🔧 Custom memory allocators for efficient matrix memory handling.
- 🧪 Python bindings planned, for safe and efficient GPU programming.
- 📚 Compliant with official CUDA programming practices (see [CUDA C Programming Guide](./CUDA_C_Programming_Guide.pdf)).
- 🐧 Developed and tested on **Linux** for full compatibility with NVIDIA toolchains.

## 📦 Components

- `src/`: CUDA C/C++ code for matrix operations and allocators.
- `include/`: Header files and kernel function declarations.
- `python/`: (WIP) Python FFI layer and safe wrappers over CUDA code.
- `docs/`: Supporting documentation and implementation notes.
- `examples/`: Example usage, tests, and performance comparisons.

## 🔍 Current Development Focus

- ✅ Building a foundational set of **matrix operation kernels** (add, mul, transpose).
- ✅ Implementing a **custom matrix memory allocator** using `cudaMalloc` / `cudaFree`.
- 🧪 Exploring **Python FFI** integration and safe abstractions over CUDA calls.
- 🔄 Aligning implementations with the official CUDA Programming Guide.

## 📖 Documentation

- Official NVIDIA [CUDA C Programming Guide](./CUDA_C_Programming_Guide.pdf) is included for reference.
- Inline code comments and markdown docs explain key CUDA concepts and implementation decisions.

## 🚧 Roadmap

- [x] Basic matrix addition, multiplication, transpose on GPU.
- [x] Memory allocator abstraction.
- [ ] Python bindings with zero-cost FFI.
- [ ] Benchmark suite comparing CPU vs CUDA vs multi-threaded CPU.
- [ ] Integration with AI modules for applied use cases (e.g. inference kernels).

## 🛠️ Build Instructions

Make sure you have:

- NVIDIA GPU with CUDA support
- CUDA Toolkit installed (e.g., `sudo apt install nvidia-cuda-toolkit`)
- `nvcc` available in your PATH

```bash
git clone https://github.com/yourusername/openmat.git
cd openmat
mkdir build && cd build
cmake ..
make
```

## 🤝 We're Looking for Contributors!

OpenMat is a growing project, and we’d love your help!

Whether you're a CUDA beginner, or a performance geek — contributions of all kinds are welcome. From kernel optimization to allocator design, from docs to test cases — there's room for you to make an impact.

🚀 **Open an issue, start a discussion, or send a PR — we’re excited to build this together.**

