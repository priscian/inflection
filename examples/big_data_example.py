"""
Big data examples demonstrating performance with large datasets.
Shows that EDE/BEDE methods are efficient even with millions of points.
"""

import numpy as np
import time
import matplotlib.pyplot as plt
from inflection import check_curve, ede, bede


def big_data_clean():
    """Test with large clean dataset (1 million points)."""
    print("=" * 60)
    print("Big Data Test - 1,000,001 Points (Clean)")
    print("=" * 60)

    ## f(x) = 500 + 500 * tanh(x - 500)
    x = np.linspace(0, 1000, 1000001)
    y = 500 + 500 * np.tanh(x - 500)

    print(f"Dataset size: {len(x):,} points")

    cc = check_curve(x, y)

    ## Time EDE method
    start = time.time()
    res_ede = ede(x, y, cc["index"])
    ede_time = time.time() - start
    print(f"\nEDE result: {res_ede["chi"]:.6f}")
    print(f"EDE time: {ede_time:.3f} seconds")

    ## Time BEDE method
    start = time.time()
    res_bede = bede(x, y, cc["index"])
    bede_time = time.time() - start
    print(f"\nBEDE result: {res_bede["iplast"]:.6f}")
    print(f"BEDE iterations: {len(res_bede["iters"]["n"])}")
    print(f"BEDE time: {bede_time:.3f} seconds")

    ## Show iteration details
    print("\nBEDE iteration details:")
    for i in range(len(res_bede["iters"]["n"])):
        print(f"  {i+1}: n={res_bede["iters"]["n"][i]:7d}, "
            f"EDE={res_bede["iters"]["EDE"][i]:.6f}")

    return res_bede


def big_data_noisy():
    """Test with large noisy dataset."""
    print("\n" + "=" * 60)
    print("Big Data Test with Noise")
    print("=" * 60)

    np.random.seed(666)

    ## Use fewer points for noisy example to save memory
    x = np.linspace(0, 1000, 100001)
    y = 500 + 500 * np.tanh(x - 500)
    ## Add significant noise
    y_noisy = y + np.random.uniform(-50, 50, len(y))

    print(f"Dataset size: {len(x):,} points")
    print(f"Noise level: U(-50, 50)")

    cc = check_curve(x, y_noisy)

    ## Time methods
    start = time.time()
    res_ede = ede(x, y_noisy, cc["index"])
    ede_time = time.time() - start

    start = time.time()
    res_bede = bede(x, y_noisy, cc["index"])
    bede_time = time.time() - start

    print(f"\nResults (true value = 500.000):")
    print(f"  EDE:  {res_ede["chi"]:.3f} (time: {ede_time:.3f}s)")
    print(f"  BEDE: {res_bede["iplast"]:.3f} (time: {bede_time:.3f}s, "
        f"{len(res_bede["iters"]["n"])} iterations)")

    return x, y_noisy, res_bede


def big_data_asymmetric():
    """Test with large dataset and asymmetric range."""
    print("\n" + "=" * 60)
    print("Big Data Test with Asymmetry")
    print("=" * 60)

    np.random.seed(666)

    ## Asymmetric range [0, 700] instead of [0, 1000]
    x = np.linspace(0, 700, 70001)
    y = 500 + 500 * np.tanh(x - 500)
    y_noisy = y + np.random.uniform(-50, 50, len(y))

    print(f"Dataset size: {len(x):,} points")
    print(f"Range: [{x[0]}, {x[-1]}] (asymmetric around IP=500)")

    cc = check_curve(x, y_noisy)

    res_bede = bede(x, y_noisy, cc["index"])

    print(f"\nBEDE result: {res_bede["iplast"]:.3f}")
    print(f"BEDE iterations: {len(res_bede["iters"]["n"])}")
    print("\nNote: Even with asymmetry and noise, BEDE converges close to true value.")

    ## Plot zoomed region
    fig, ax = plt.subplots(figsize = (10, 6))

    ## Plot only around inflection point
    mask = (x > 490) & (x < 510)
    ax.scatter(x[mask], y_noisy[mask], alpha = 0.3, s = 1, label = "Noisy data")
    ax.axvline(x = 500, color = "green", linestyle = ":", linewidth = 2, label = "True IP")
    ax.axvline(x = res_bede["iplast"], color = "red", linestyle = "--", linewidth = 2,
        label = f"BEDE: {res_bede["iplast"]:.3f}")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_title(f"Big Data: Zoomed View Around Inflection Point ({len(x):,} total points)")
    ax.legend()
    ax.grid(True, alpha = 0.3)

    plt.tight_layout()
    plt.show()

    return res_bede


def performance_comparison():
    """Compare performance across different dataset sizes."""
    print("\n" + "=" * 60)
    print("Performance Scaling Analysis")
    print("=" * 60)

    sizes = [100, 500, 1000, 5000, 10000, 50000, 100000]
    ede_times = []
    bede_times = []
    bede_iters = []

    for n in sizes:
        x = np.linspace(0, 1000, n)
        y = 500 + 500 * np.tanh(x - 500)

        cc = check_curve(x, y)

        ## Time EDE
        start = time.time()
        ede(x, y, cc["index"])
        ede_times.append(time.time() - start)

        ## Time BEDE
        start = time.time()
        res = bede(x, y, cc["index"])
        bede_times.append(time.time() - start)
        bede_iters.append(len(res["iters"]["n"]))

    ## Print results
    print("\n{:<10} {:<12} {:<12} {:<10}".format("Size", "EDE (s)", "BEDE (s)", "BEDE iters"))
    print("-" * 45)
    for i, n in enumerate(sizes):
        print(f"{n:<10,} {ede_times[i]:<12.4f} {bede_times[i]:<12.4f} {bede_iters[i]:<10}")

    print("\nKey observations:")
    print("- EDE is O(n) in time complexity")
    print("- BEDE iterations remain roughly constant regardless of data size")
    print("- Both methods scale well to large datasets")


if __name__ == "__main__":
    ## Run examples
    big_data_clean()
    big_data_noisy()
    big_data_asymmetric()
    performance_comparison()
