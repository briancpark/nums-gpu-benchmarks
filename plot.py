import matplotlib.pyplot as plt
import pandas as pd

### GLOBAL VARIABLES
MEMORY = 16e9 # IN GIGABYTES
NODES = 1
TOTAL_MEMORY = NODES * MEMORY
THEORETICAL_PEAK_64 = (7.8 * 10 ** 12) * NODES
THEORETICAL_PEAK_32 = (15.7 * 10 ** 12) * NODES
THEORETICAL_PEAK_TENSOR = (125 * 10 ** 12) * NODES
THEORETICAL_PEAK_64_NO_FMA = 3916.8 * 10 ** 9

KILO = float(10 ** 3)
MEGA = float(10 ** 6)
GIGA = float(10 ** 9)
TERA = float(10 ** 12)

def plotter(is_nums, mode, dtype, num_gpus):
    if is_nums:
        name = "nums"
    else:
        name = "cupy"

    return pd.read_csv("{}_{}_{}_{}gpus.csv".format(name, mode, dtype, num_gpus))

    # cp_avg_flops = [flops(n) / (time * 10 ** 9) for n,time in zip(ns, cp_avg_times)]
    # avg_flops = [flops(n) / (time * 10 ** 9) for n, time in zip(ns_big, avg_times)]
    # print(ns, avg_times)
    # print(avg_flops)
    # plt.figure(figsize=(10, 10))
    # plt.plot(ns_big, avg_flops, label="NumS (8 V100s)")
    # plt.plot(ns, cp_avg_flops, label="CuPy Single Node")
    # # plt.plot(ns, [THEORETICAL_PEAK_64_NO_FMA for n in ns], label="Theoretical Peak No FMA", linestyle="--")
    # # # plt.plot(ns, [THEORETICAL_PEAK_64 for n in ns], label="Theoretical Peak", linestyle="--")

    # plt.plot(df["n"], df["flops"], label="NumS (" + str(num_gpus) + " V100s)")
    # plt.title(r"NumS GPU GFLOPS of Elementwise Operations on $n \times n$ Matrices")
    # # plt.xscale("log")
    # # plt.yscale("log")
    # plt.xlabel(r"Matrix Size $n$ ($n \times n$)")
    # plt.ylabel("GFLOPS")
    # # plt.xticks(ns_big)
    # plt.legend()
    # plt.savefig("NumS_GPU_FLOPS_elementwise.png", dpi=400)

    #     # plt.clf()
    # plt.plot(df["n"])

if __name__ == "__main__":
    df_cupy = plotter(is_nums=False, mode="elementwise", dtype="float64", num_gpus=1)

    df_nums_1 = plotter(is_nums=True, mode="elementwise", dtype="float64", num_gpus=1)
    df_nums_2 = plotter(is_nums=True, mode="elementwise", dtype="float64", num_gpus=2)
    df_nums_4 = plotter(is_nums=True, mode="elementwise", dtype="float64", num_gpus=4)
    df_nums_8 = plotter(is_nums=True, mode="elementwise", dtype="float64", num_gpus=8)

    plt.figure(figsize=(10, 10))
    plt.plot(df_cupy["n"], df_cupy["flops"], label="CuPy Single Node")
    plt.plot(df_nums_1["n"], df_nums_1["flops"], label="NumS (1 V100)")
    plt.plot(df_nums_2["n"], df_nums_2["flops"], label="NumS (2 V100s)")
    plt.plot(df_nums_4["n"], df_nums_4["flops"], label="NumS (4 V100s)")
    plt.plot(df_nums_8["n"], df_nums_8["flops"], label="NumS (8 V100s)")
    plt.title(r"NumS GPU FLOPS of Elementwise Operations on Size $n$ Array")
    plt.xlabel(r"Array Size $n$")
    plt.ylabel("FLOPS")
    plt.yscale("log")
    plt.xscale("log")
    plt.legend()
    plt.savefig("NumS_GPU_FLOPS_elementwise.png", dpi=400)
    plt.clf()


    df_cupy_mm = plotter(is_nums=False, mode="matmul", dtype="float64", num_gpus=1)

    df_nums_mm_1 = plotter(is_nums=True, mode="matmul", dtype="float64", num_gpus=1)
    df_nums_mm_2 = plotter(is_nums=True, mode="matmul", dtype="float64", num_gpus=2)

    plt.figure(figsize=(10, 10))
    plt.plot(df_cupy_mm["n"], df_cupy_mm["flops"], label="CuPy Single Node")
    plt.plot(df_nums_mm_1["n"], df_nums_mm_1["flops"], label="NumS (1 V100)")
    plt.plot(df_nums_mm_2["n"], df_nums_mm_2["flops"], label="NumS (2 V100s)")
    plt.plot(df_nums_mm_2["n"], [THEORETICAL_PEAK_64 for n in df_nums_mm_2["n"]], label="Theoretical Peak", linestyle="--")
    plt.title(r"NumS GPU FLOPS of Matrix Multiplication on $n \times n$ Matrix")
    plt.xlabel(r"Matrix $n$ Dimension ($n \times n$)")
    plt.ylabel("FLOPS")
    # plt.yscale("log")
    # plt.xscale("log")
    plt.legend()
    plt.savefig("NumS_GPU_FLOPS_matmul.png", dpi=400)
    plt.clf()

    

