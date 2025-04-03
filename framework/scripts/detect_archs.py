#!/usr/bin/env python3
import pycuda.driver as cuda
import pycuda.autoinit

archs = set()

for i in range(cuda.Device.count()):
    dev = cuda.Device(i)
    cc = dev.compute_capability()
    sm = f"{cc[0]}{cc[1]}"
    archs.add(sm)

print(";".join(sorted(archs)))
