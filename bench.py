from mlp import run_test
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
MEMORY = 16e9  # IN GIGABYTES
NODES = int(
    os.popen("nvidia-smi --query-gpu=name --format=csv,noheader | wc -l").read()
)
TOTAL_MEMORY = NODES * MEMORY
THEORETICAL_PEAK_64 = (7.8 * 10**12) * NODES
THEORETICAL_PEAK_32 = (15.7 * 10**12) * NODES
THEORETICAL_PEAK_TENSOR = (125 * 10**12) * NODES
THEORETICAL_PEAK_64_NO_FMA = 3916.8 * 10**9

KILO = float(10**3)
MEGA = float(10**6)
GIGA = float(10**9)
TERA = float(10**12)

DTYPES = {
    "int32": nps.int32,
    "int64": nps.int64,
    "float16": nps.float16,
    "float32": nps.float32,
    "float64": nps.float64,
}

SEED = 1337
cp.random.seed(SEED)
nps.random.seed(SEED)

# FLOPS
def matmul_flops(n):
    return 2 * (n**3)


def elementwise_flops(n):
    return n


def run_nums(backend_name, mode, ns, dtype=None):
    settings.backend_name = backend_name
    global SEED

    nums.init()

    avg_times = []

    ### INITIALIZE LOGGER
    t0 = nps.random.rand(1)
    if dtype:
        t0 = t0.astype(DTYPES[dtype])
    fh = open(
        "data/nums_" + mode + "_" + t0.dtype.__name__ + "_" + str(NODES) + "gpus.csv",
        "w",
        newline="",
    )
    writer = csv.writer(fh)
    writer.writerow(["n", "time", "flops"])
    del t0

    for n in ns:
        SEED += 1
        nps.random.seed(SEED)
        if mode == "matmul":
            A = nps.random.rand(n, n)
            B = nps.random.rand(n, n)
        elif mode == "elementwise":
            A = nps.random.rand(n)
            B = nps.random.rand(n)

        if dtype:
            # Ex: dtype=nps.float32 passed into run_nums()
            A = A.astype(DTYPES[dtype])
            B = B.astype(DTYPES[dtype])

        A.touch()
        B.touch()
        print("Total memory to be used:", A.nbytes * 3)
        print(
            "Total memory estimated usage: {} %".format((A.nbytes * 3) / TOTAL_MEMORY)
        )
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

        # Throw away the first five results due to CuPy's context initialization
        # More here: https://docs.cupy.dev/en/stable/user_guide/performance.html
        times = times[5:]

        avg_time, avg_flops = mean(times), mean(flops)

        writer.writerow([n, avg_time, avg_flops])

        print("Average time:", avg_time)
        print("Average TFLOPS:", avg_flops / TERA)

        del A, B, C


def run_cupy(mode, ns, dtype=None):
    cp_avg_times = []
    global SEED

    ### INITIALIZE LOGGER
    t0 = cp.random.rand(1, dtype=DTYPES[dtype])
    fh = open(
        "data/cupy_" + mode + "_" + t0.dtype.base.name + "_" + str(NODES) + "gpus.csv",
        "w",
        newline="",
    )
    writer = csv.writer(fh)
    writer.writerow(["n", "time", "flops"])
    del t0

    for n in ns:
        SEED += 1
        cp.random.seed(SEED)
        if mode == "matmul":
            A = cp.random.rand(n, n, dtype=DTYPES[dtype])
            B = cp.random.rand(n, n, dtype=DTYPES[dtype])
        elif mode == "elementwise":
            A = cp.random.rand(n, dtype=DTYPES[dtype])
            B = cp.random.rand(n, dtype=DTYPES[dtype])

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

        # Throw away the first five results due to CuPy's context initialization
        # More here: https://docs.cupy.dev/en/stable/user_guide/performance.html
        times = times[5:]

        avg_time, avg_flops = mean(times), mean(flops)

        writer.writerow([n, avg_time, avg_flops])

        print("Average time:", avg_time)
        print("Average TFLOPS:", avg_flops / TERA)

        del A, B, C


"""
Usage:
    python3 bench.py <package> <benchmark> <dtype> <backend> 

Example:
    python3 bench.py nums matmul float64 gpu-intra
    python3 bench.py cupy elementwise float32
"""
if __name__ == "__main__":
    package = sys.argv[1]
    benchmark = sys.argv[2]
    dtype = sys.argv[3] 

    t0 = cp.random.rand(1, dtype=DTYPES[dtype])
    if package == "nums":
        backend = sys.argv[4]

    if benchmark == "matmul":
        base_ns = list(range(4000, 32000 * int(8 / t0.nbytes), 2000))
        ns_full = list(range(4000, 32000 * NODES * int(8 / t0.nbytes), 2000))
        ns = []
        for n in ns_full:
            usage = (n ** 2) * 3 * t0.nbytes / TOTAL_MEMORY
            if usage < 0.5:
                ns.append(n)
        print("Percentage of GPU memory to be used", (ns[-1] ** 2) * 3 * t0.nbytes / TOTAL_MEMORY)

    elif benchmark == "elementwise":
        base_ns = list(range(1 * 10 ** 8, 5 * int(8 / t0.nbytes) * 10 ** 8, 1 * 10 ** 8))
        ns = list(range(1 * 10 ** 8, NODES * int(8 / t0.nbytes) * 5 * 10 ** 8, 1 * 10 ** 8))
        print("Percentage of GPU memory to be used", ns[-1] * 3 * t0.nbytes / TOTAL_MEMORY)

    if package == "nums":
        run_nums(backend, benchmark, ns, dtype=dtype)
    elif package == "cupy":
        run_cupy(benchmark, base_ns, dtype=dtype)
