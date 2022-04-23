import nums
from nums import numpy as nps
from nums.core import settings
from nums.core.application_manager import instance

import cupy as cp
import numpy as np

from statistics import mean

import time
import os
import sys
import csv

settings.device_grid_name = "packed"

### GLOBAL VARIABLES
MEMORY = 16e9 # IN GIGABYTES
NODES = 4# int(os.popen("nvidia-smi --query-gpu=name --format=csv,noheader | wc -l").read())
print("Nodes:", NODES)
TOTAL_MEMORY = NODES * MEMORY
THEORETICAL_PEAK_64 = (7.8 * 10 ** 12) * NODES
THEORETICAL_PEAK_32 = (15.7 * 10 ** 12) * NODES
THEORETICAL_PEAK_TENSOR = (125 * 10 ** 12) * NODES
THEORETICAL_PEAK_64_NO_FMA = 3916.8 * 10 ** 9

KILO = float(10 ** 3)
MEGA = float(10 ** 6)
GIGA = float(10 ** 9)
TERA = float(10 ** 12)

DTYPES = {
    nps.float32 : "float32",
    nps.float64 : "float64",
}

cp.random.seed(1337)
nps.random.seed(1337)

# FLOPS
def matmul_flops(n):
    return 2 * (n ** 3)

def elementwise_flops(n):
    return n

def run_nums(backend_name, mode, ns, dtype=None):
    settings.backend_name = backend_name
    
    nums.init()

    avg_times = []

    ### INITIALIZE LOGGER
    t0 = nps.random.rand(1)
    if dtype:
        t0 = t0.astype(dtype)
    fh = open("nums_" + mode + "_" + DTYPES[t0.dtype] + "_" + str(NODES) + "gpus.csv", "w", newline="")
    writer = csv.writer(fh)
    writer.writerow(["n", "time", "flops"])
    del t0

    for n in ns:
        if mode == "matmul":
            A = nps.random.rand(n, n)
            B = nps.random.rand(n, n)
        elif mode == "elementwise":
            A = nps.random.rand(n)
            B = nps.random.rand(n)

        if dtype:
            # Ex: dtype=nps.float32 passed into run_nums()
            A = A.astype(dtype)
            B = B.astype(dtype)
            
        A.touch()
        B.touch()
        print("Total memory to be used:", A.nbytes * 3)
        print("Total memory estimated usage: {} %".format((A.nbytes * 3) / TOTAL_MEMORY))
        print(A.shape, A.grid_shape, A.block_shape)

        times = []
        flops = []

        for i in range(15):
            time.sleep(1)
            if mode == "matmul":
                begin = time.time()
                C = A @ B
                C.touch()
                end = time.time()
                print(end - begin)
                times.append(end - begin)
                flops.append(matmul_flops(n) / (end - begin))
            elif mode == "elementwise":
                begin = time.time()
                C = A * B
                C.touch()
                end = time.time()
                print(end - begin)
                times.append(end - begin)
                flops.append(elementwise_flops(n) / (end - begin))

        # Throw away the first five results due to CuPy's Context initialization
        # More here: https://docs.cupy.dev/en/stable/user_guide/performance.html
        times.pop(0)
        times.pop(0)
        times.pop(0)
        times.pop(0)
        times.pop(0)

        avg_time = mean(times)
        avg_flops = mean(flops)
        
        writer.writerow([n, avg_time, avg_flops])

        del A, B, C

def run_cupy(mode, ns, dtype=None):
    cp_avg_times = []

    ### INITIALIZE LOGGER
    t0 = cp.random.rand(1, dtype=dtype)
    fh = open("cupy_" + mode + "_" + t0.dtype.base.name + "_" + str(NODES) + "gpus.csv", "w", newline="")
    writer = csv.writer(fh)
    writer.writerow(["n", "time", "flops"])
    del t0

    for n in ns:
        if mode == "matmul":
            A = cp.random.rand(n, n, dtype=dtype)
            B = cp.random.rand(n, n, dtype=dtype)
        elif mode == "elementwise":
            A = cp.random.rand(n, dtype=dtype)
            B = cp.random.rand(n, dtype=dtype)
       
        A.device.synchronize()
        B.device.synchronize()

        print("Total memory to be used:", A.nbytes * 3)
        print("Total memory estimated usage: {} %".format((A.nbytes * 3) / MEMORY))
        print(A.size)

        times = []
        flops = []

        for i in range(15):
            time.sleep(1)
            if mode == "matmul":
                begin = time.time()
                C = A @ B
                C.device.synchronize()
                end = time.time()
                print(end - begin)
                times.append(end - begin)
                flops.append(matmul_flops(n) / (end - begin))
            elif mode == "elementwise":
                begin = time.time()
                C = A * B
                C.device.synchronize()
                end = time.time()
                print(end - begin)
                times.append(end - begin)
                flops.append(elementwise_flops(n) / (end - begin))

        # Throw away the first five results due to CuPy's Context initialization
        # More here: https://docs.cupy.dev/en/stable/user_guide/performance.html
        times.pop(0)
        times.pop(0)
        times.pop(0)
        times.pop(0)
        times.pop(0)


        avg_time = mean(times)
        avg_flops = mean(flops)

        writer.writerow([n, avg_time, avg_flops])

        del A, B, C
    
    # cp_avg_flops = [flops(n) / (time * 10 ** 9) for n,time in zip(ns, cp_avg_times)]
    # avg_flops = [flops(n) / (time * 10 ** 9) for n, time in zip(ns_big, avg_times)]
    # print(ns, avg_times)
    # print(avg_flops)
    # plt.figure(figsize=(10, 10))
    # plt.plot(ns_big, avg_flops, label="NumS (8 V100s)")
    # plt.plot(ns, cp_avg_flops, label="CuPy Single Node")
    # # plt.plot(ns, [THEORETICAL_PEAK_64_NO_FMA for n in ns], label="Theoretical Peak No FMA", linestyle="--")
    # # plt.plot(ns, [THEORETICAL_PEAK_64 for n in ns], label="Theoretical Peak", linestyle="--")
    # plt.title(r"NumS GPU GFLOPS of Elementwise Operations on $n \times n$ Matrices")
    # plt.xscale("log")
    # # plt.yscale("log")
    # plt.xlabel(r"Matrix Size $n$ ($n \times n$)")
    # plt.ylabel("GFLOPS")
    # plt.xticks(ns_big)
    # plt.legend()
    # plt.savefig("NumS_GPU_FLOPS_elementwise.png", dpi=400)

    # plt.clf()

    # plt.plot(ns_big, avg_times, label="NumS (8 V100s)")
    # plt.plot(ns, cp_avg_times, label="CuPy Single Node")
    # plt.legend()
    # plt.savefig("test.png")


if __name__ == "__main__":
    ns = [1024, 2048, 4096, 8192, 16384] # This is waht single node can handle
    ns_big = ns + [32768]
    run_nums("gpu-intra", "matmul", ns_big)
    run_cupy("matmul", ns)
    # ns = [1 * 10 ** 8, 2 * 10 ** 8, 3 * 10 ** 8, 4 * 10 ** 8, 5 * 10 ** 8]
    # ns_big = ns + [6 * 10 ** 8, 7 * 10 ** 8, 8 * 10 ** 8, 9 * 10 ** 8, 10 ** 9]
    # run_nums("gpu-intra", "elementwise", ns_big)
    # run_cupy("elementwise", ns)
     
    