import matplotlib.pyplot as plt
import pandas as pd

### GLOBAL VARIABLES
MEMORY = 16e9 # IN GIGABYTES
NODES = 1
TOTAL_MEMORY = NODES * MEMORY
THEORETICAL_PEAK_64 = 7.8 * 10**12
THEORETICAL_PEAK_32 = 15.7 * 10**12
THEORETICAL_PEAK_TENSOR = 125 * 10**12
THEORETICAL_PEAK_64_NO_FMA = 3.9168 * 10**12

EMPIRICAL_PEAK_64 = 7.0689 * 10**12
EMPIRICAL_PEAK_64_NO_FMA = 3.5358 * 10 ** 12

KILO = float(10 ** 3)
MEGA = float(10 ** 6)
GIGA = float(10 ** 9)
TERA = float(10 ** 12)

def plotter(is_nums, mode, dtype, num_gpus):
    if is_nums:
        name = "nums"
    else:
        name = "cupy"

    return pd.read_csv("data/{}_{}_{}_{}gpus.csv".format(name, mode, dtype, num_gpus))

if __name__ == "__main__":
    df_cupy = plotter(is_nums=False, mode="elementwise", dtype="float64", num_gpus=1)

    df_nums_1 = plotter(is_nums=True, mode="elementwise", dtype="float64", num_gpus=1)
    df_nums_2 = plotter(is_nums=True, mode="elementwise", dtype="float64", num_gpus=2)
    # df_nums_4 = plotter(is_nums=True, mode="elementwise", dtype="float64", num_gpus=4)
    # df_nums_8 = plotter(is_nums=True, mode="elementwise", dtype="float64", num_gpus=8)

    plt.figure(figsize=(7, 5))
    plt.plot(df_cupy["n"], df_cupy["flops"] / TERA, label="CuPy Single Node")
    plt.plot(df_nums_1["n"], df_nums_1["flops"] / TERA, label="NumS (1 V100)")
    plt.plot(df_nums_2["n"], df_nums_2["flops"] / TERA, label="NumS (2 V100s)")
    # plt.plot(df_nums_4["n"], df_nums_4["flops"] / TERA, label="NumS (4 V100s)")
    # plt.plot(df_nums_8["n"], df_nums_8["flops"] / TERA, label="NumS (8 V100s)")
    plt.title(r"NumS GPU FLOPS of Elementwise Operations on Size $n$ Array")
    plt.xlabel(r"Array Size $n$")
    plt.ylabel("TFLOPS")
    plt.yscale("log")
    plt.xscale("log")
    plt.legend()
    plt.savefig("figures/NumS_GPU_TFLOPS_elementwise.png", dpi=400)
    plt.clf()


    df_cupy_mm = plotter(is_nums=False, mode="matmul", dtype="float64", num_gpus=1)

    df_nums_mm_1 = plotter(is_nums=True, mode="matmul", dtype="float64", num_gpus=1)
    df_nums_mm_2 = plotter(is_nums=True, mode="matmul", dtype="float64", num_gpus=2)
    df_nums_mm_4 = plotter(is_nums=True, mode="matmul", dtype="float64", num_gpus=4)
    # df_nums_mm_8 = plotter(is_nums=True, mode="matmul", dtype="float64", num_gpus=8)

    plt.figure(figsize=(8, 6))
    plt.plot(df_cupy_mm["n"], df_cupy_mm["flops"] / TERA, label="CuPy Single Node")
    plt.plot(df_nums_mm_1["n"], df_nums_mm_1["flops"] / TERA, label="NumS (1 V100)")
    plt.plot(df_nums_mm_2["n"], df_nums_mm_2["flops"] / TERA, label="NumS (2 V100s)")
    plt.plot(df_nums_mm_4["n"], df_nums_mm_4["flops"] / TERA, label="NumS (4 V100s)")
    # plt.plot(df_nums_mm_8["n"], df_nums_mm_8["flops"] / TERA, label="NumS (8 V100s)")
    plt.plot(df_nums_mm_4["n"], [THEORETICAL_PEAK_64  / TERA for n in df_nums_mm_4["n"]], label="Theoretical Peak", linestyle="--")
    plt.plot(df_nums_mm_4["n"], [THEORETICAL_PEAK_64 * 2 / TERA for n in df_nums_mm_4["n"]], label="Theoretical Peak 2X", linestyle="--")
    plt.plot(df_nums_mm_4["n"], [THEORETICAL_PEAK_64 * 4 / TERA for n in df_nums_mm_4["n"]], label="Theoretical Peak 4X", linestyle="--")
    # plt.plot(df_nums_mm_8["n"], [THEORETICAL_PEAK_64 * 8 / TERA for n in df_nums_mm_8["n"]], label="Theoretical Peak 8X", linestyle="--")
    plt.plot(df_nums_mm_4["n"], [EMPIRICAL_PEAK_64  / TERA for n in df_nums_mm_4["n"]], label="Empirical Peak", linestyle="--")
    plt.plot(df_nums_mm_4["n"], [EMPIRICAL_PEAK_64 * 2 / TERA for n in df_nums_mm_4["n"]], label="Empirical Peak 2X", linestyle="--")
    plt.plot(df_nums_mm_4["n"], [EMPIRICAL_PEAK_64 * 4 / TERA for n in df_nums_mm_4["n"]], label="Empirical Peak 4X", linestyle="--")
    plt.title(r"NumS GPU DGEMM TFLOPS on $n \times n$ Matrix")
    plt.xlabel(r"Matrix $n$ Dimension ($n \times n$)")
    plt.ylabel("TFLOPS")
    plt.legend()
    plt.savefig("figures/NumS_GPU_TFLOPS_DGEMM.png", dpi=400)
    plt.clf()

    
    print("CuPy")
    print("Percentage of Empirical Peak (1 V100s): ", df_cupy_mm["flops"].max() / EMPIRICAL_PEAK_64)
    print("Percentage of Theoretical Peak (1 V100s): ", df_cupy_mm["flops"].max() /  THEORETICAL_PEAK_64)
    
    print("NumS")
    print("Percentage of Empirical Peak (1 V100s): ", df_nums_mm_1["flops"].max() / EMPIRICAL_PEAK_64)
    print("Percentage of Empirical Peak (2 V100s): ", df_nums_mm_2["flops"].max() / (EMPIRICAL_PEAK_64 * 2))
    print("Percentage of Empirical Peak (4 V100s): ", df_nums_mm_4["flops"].max() / (EMPIRICAL_PEAK_64 * 4))

    print("Percentage of Theoretical Peak (1 V100s): ", df_nums_mm_1["flops"].max() /  THEORETICAL_PEAK_64)
    print("Percentage of Theoretical Peak (2 V100s): ", df_nums_mm_2["flops"].max() / (THEORETICAL_PEAK_64 * 2))
    print("Percentage of Theoretical Peak (4 V100s): ", df_nums_mm_4["flops"].max() / (THEORETICAL_PEAK_64 * 4))




    df_cupy_mm = plotter(is_nums=False, mode="matmul", dtype="float32", num_gpus=1)

    df_nums_mm_1 = plotter(is_nums=True, mode="matmul", dtype="float32", num_gpus=1)
    df_nums_mm_2 = plotter(is_nums=True, mode="matmul", dtype="float32", num_gpus=2)
    df_nums_mm_4 = plotter(is_nums=True, mode="matmul", dtype="float32", num_gpus=4)
    # df_nums_mm_8 = plotter(is_nums=True, mode="matmul", dtype="float32", num_gpus=8)

    plt.figure(figsize=(8, 6))
    plt.plot(df_cupy_mm["n"], df_cupy_mm["flops"] / TERA, label="CuPy Single Node")
    plt.plot(df_nums_mm_1["n"], df_nums_mm_1["flops"] / TERA, label="NumS (1 V100)")
    plt.plot(df_nums_mm_2["n"], df_nums_mm_2["flops"] / TERA, label="NumS (2 V100s)")
    plt.plot(df_nums_mm_4["n"], df_nums_mm_4["flops"] / TERA, label="NumS (4 V100s)")
    # plt.plot(df_nums_mm_8["n"], df_nums_mm_8["flops"] / TERA, label="NumS (8 V100s)")
    plt.plot(df_nums_mm_4["n"], [THEORETICAL_PEAK_32 / TERA for n in df_nums_mm_4["n"]], label="Theoretical Peak", linestyle="--")
    plt.plot(df_nums_mm_4["n"], [THEORETICAL_PEAK_32 * 2 / TERA for n in df_nums_mm_4["n"]], label="Theoretical Peak 2X", linestyle="--")
    plt.plot(df_nums_mm_4["n"], [THEORETICAL_PEAK_32 * 4 / TERA for n in df_nums_mm_4["n"]], label="Theoretical Peak 4X", linestyle="--")
    # plt.plot(df_nums_mm_8["n"], [THEORETICAL_PEAK_32 * 8 / TERA for n in df_nums_mm_8["n"]], label="Theoretical Peak 8X", linestyle="--")
    plt.title(r"NumS GPU SGEMM TFLOPS on $n \times n$ Matrix")
    plt.xlabel(r"Matrix $n$ Dimension ($n \times n$)")
    plt.ylabel("TFLOPS")
    plt.legend()
    plt.savefig("figures/NumS_GPU_TFLOPS_SGEMM.png", dpi=400)
    plt.clf()

    print("Percentage of Theoretical Peak (1 V100s): ", df_nums_mm_1["flops"].max() /  THEORETICAL_PEAK_32)
    print("Percentage of Theoretical Peak (2 V100s): ", df_nums_mm_2["flops"].max() / (THEORETICAL_PEAK_32 * 2))
    print("Percentage of Theoretical Peak (4 V100s): ", df_nums_mm_4["flops"].max() / (THEORETICAL_PEAK_32 * 4))


    df_cupy_mm = plotter(is_nums=False, mode="matmul", dtype="float16", num_gpus=1)

    df_nums_mm_1 = plotter(is_nums=True, mode="matmul", dtype="float16", num_gpus=1)
    df_nums_mm_2 = plotter(is_nums=True, mode="matmul", dtype="float16", num_gpus=2)
    df_nums_mm_4 = plotter(is_nums=True, mode="matmul", dtype="float16", num_gpus=4)
    # df_nums_mm_8 = plotter(is_nums=True, mode="matmul", dtype="float16", num_gpus=8)

    plt.figure(figsize=(8, 6))
    plt.plot(df_cupy_mm["n"], df_cupy_mm["flops"] / TERA, label="CuPy Single Node")
    plt.plot(df_nums_mm_1["n"], df_nums_mm_1["flops"] / TERA, label="NumS (1 V100)")
    plt.plot(df_nums_mm_2["n"], df_nums_mm_2["flops"] / TERA, label="NumS (2 V100s)")
    plt.plot(df_nums_mm_4["n"], df_nums_mm_4["flops"] / TERA, label="NumS (4 V100s)")
    # plt.plot(df_nums_mm_8["n"], df_nums_mm_8["flops"] / TERA, label="NumS (8 V100s)")
    plt.plot(df_nums_mm_4["n"], [THEORETICAL_PEAK_TENSOR / TERA for n in df_nums_mm_4["n"]], label="Theoretical Peak", linestyle="--")
    plt.plot(df_nums_mm_4["n"], [THEORETICAL_PEAK_TENSOR * 2 / TERA for n in df_nums_mm_4["n"]], label="Theoretical Peak 2X", linestyle="--")
    plt.plot(df_nums_mm_4["n"], [THEORETICAL_PEAK_TENSOR * 4 / TERA for n in df_nums_mm_4["n"]], label="Theoretical Peak 4X", linestyle="--")
    # plt.plot(df_nums_mm_4["n"], [THEORETICAL_PEAK_TENSOR * 8 / TERA for n in df_nums_mm_4["n"]], label="Theoretical Peak 8X", linestyle="--")
    plt.title(r"NumS GPU FP16 GEMM TFLOPS  on $n \times n$ Matrix")
    plt.xlabel(r"Matrix $n$ Dimension ($n \times n$)")
    plt.ylabel("TFLOPS")
    plt.legend()
    plt.savefig("figures/NumS_GPU_TFLOPS_FP16GEMM.png", dpi=400)
    plt.clf()

    print("Percentage of Theoretical Peak (1 V100s): ", df_nums_mm_1["flops"].max() /  THEORETICAL_PEAK_TENSOR)
    print("Percentage of Theoretical Peak (2 V100s): ", df_nums_mm_2["flops"].max() / (THEORETICAL_PEAK_TENSOR * 2))
    print("Percentage of Theoretical Peak (4 V100s): ", df_nums_mm_4["flops"].max() / (THEORETICAL_PEAK_TENSOR * 4))

