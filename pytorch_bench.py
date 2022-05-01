from pytorch_mlp import run_test

import numpy as np

from statistics import mean

import time
import os
import sys
import csv


### GLOBAL VARIABLES
MEMORY = 16e9 # IN GIGABYTES
NODES = 2# int(os.popen("nvidia-smi --query-gpu=name --format=csv,noheader | wc -l").read())
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

# FLOPS
def matmul_flops(n):
    return 2 * (n ** 3)

def elementwise_flops(n):
    return n

def run_torch(ns, dtype=None):

    avg_times = []

    ### INITIALIZE LOGGER
    t0 = np.random.rand(1)
    if dtype:
        t0 = t0.astype(dtype)
    fh = open("pytorch_" + str(NODES) + "gpus.csv", "w", newline="")
    writer = csv.writer(fh)
    writer.writerow(["n", "time", "flops"])
    del t0

    for n in ns:
        times = []
        flops = []

        for i in range(15):
            time.sleep(1)
            begin = time.time()
            run_test(n)
            # C.touch()
            end = time.time()
            print(end - begin)
            times.append(end - begin)
            flops.append(elementwise_flops(n) / (end - begin))

        times.pop(0)
        times.pop(0)
        times.pop(0)
        times.pop(0)
        times.pop(0)
        avg_time = mean(times)
        avg_flops = mean(flops)
        
        writer.writerow([n, avg_time, avg_flops])



if __name__ == "__main__":
    ns = [1024, 2048, 4096, 8192, 16384] # This is waht single node can handle
    ns_big = ns + [32768]
    # run_nums("gpu-intra", "matmul", ns_big)
    # run_cupy("matmul", ns)
    # ns = [1 * 10 ** 8, 2 * 10 ** 8, 3 * 10 ** 8, 4 * 10 ** 8, 5 * 10 ** 8]
    # ns_big = ns + [6 * 10 ** 8, 7 * 10 ** 8, 8 * 10 ** 8, 9 * 10 ** 8, 10 ** 9]
    # run_nums("gpu-intra", "elementwise", ns_big)
    # run_cupy("elementwise", ns)
    # run_cupy("mlp", ns_big)
    run_torch(ns_big)
     
    