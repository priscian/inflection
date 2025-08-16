"""
Fisher-Pry sigmoid curve examples from the R vignettes.
Demonstrates inflection point detection on tanh functions.
"""

import numpy as np
import matplotlib.pyplot as plt
from inflection import check_curve, ese, ede, bese, bede


def fisher_pry_total_symmetry():
    """Example with total symmetry (no asymmetry in data range)."""
    print("=" * 60)
    print("Fisher-Pry Sigmoid with Total Symmetry")
    print("=" * 60)

    ## f(x) = 5 + 5 * tanh(x - 5), inflection point at x = 5
    x = np.linspace(2, 8, 501)  ## Symmetric around x = 5
    y = 5 + 5 * np.tanh(x - 5)

    ## Check curve
    cc = check_curve(x, y)
    print(f"Curve type: {cc["ctype"]}")
    print(f"Index: {cc["index"]}")

    ## Apply methods
    res_ese = ese(x, y, cc["index"])
    res_ede = ede(x, y, cc["index"])
    res_bese = bese(x, y, cc["index"])
    res_bede = bede(x, y, cc["index"])

    print(f"\nResults (true value = 5.000):")
    print(f"  ESE:  {res_ese["chi"]:.4f}")
    print(f"  EDE:  {res_ede["chi"]:.4f}")
    print(f"  BESE: {res_bese["iplast"]:.4f} ({len(res_bese["iters"]["n"])} iterations)")
    print(f"  BEDE: {res_bede["iplast"]:.4f} ({len(res_bede["iters"]["n"])} iterations)")

    return x, y, res_bese, res_bede


def fisher_pry_with_noise():
    """Example with added uniform noise."""
    print("\n" + "=" * 60)
    print("Fisher-Pry Sigmoid with Noise")
    print("=" * 60)

    np.random.seed(666)
    x = np.linspace(2, 8, 501)
    y = 5 + 5 * np.tanh(x - 5)
    ## Add uniform noise U(-0.05, 0.05)
    y_noisy = y + np.random.uniform(-0.05, 0.05, len(y))

    cc = check_curve(x, y_noisy)

    ## Apply methods
    res_ese = ese(x, y_noisy, cc["index"])
    res_ede = ede(x, y_noisy, cc["index"])
    res_bese = bese(x, y_noisy, cc["index"])
    res_bede = bede(x, y_noisy, cc["index"])

    print(f"\nResults with noise (true value = 5.000):")
    print(f"  ESE:  {res_ese["chi"]:.4f}")
    print(f"  EDE:  {res_ede["chi"]:.4f}")
    print(f"  BESE: {res_bese["iplast"]:.4f} ({len(res_bese["iters"]["n"])} iterations)")
    print(f"  BEDE: {res_bede["iplast"]:.4f} ({len(res_bede["iters"]["n"])} iterations)")

    return x, y_noisy, res_bese, res_bede


def fisher_pry_left_asymmetry():
    """Example with data left asymmetry."""
    print("\n" + "=" * 60)
    print("Fisher-Pry Sigmoid with Left Asymmetry")
    print("=" * 60)

    ## Start from x=4.2 instead of symmetric range
    x = np.linspace(4.2, 8, 301)
    y = 5 + 5 * np.tanh(x - 5)

    cc = check_curve(x, y)

    ## Apply methods
    res_ese = ese(x, y, cc["index"])
    res_ede = ede(x, y, cc["index"])
    res_bese = bese(x, y, cc["index"])
    res_bede = bede(x, y, cc["index"])

    print(f"\nResults with left asymmetry (true value = 5.000):")
    print(f"  ESE:  {res_ese["chi"]:.4f} (theoretical ~4.70)")
    print(f"  EDE:  {res_ede["chi"]:.4f} (theoretical ~5.09)")
    print(f"  BESE: {res_bese["iplast"]:.4f}")
    print(f"  BEDE: {res_bede["iplast"]:.4f}")

    ## Print iteration details for BESE
    print("\nBESE iterations:")
    for i in range(len(res_bese["iters"]["n"])):
        print(f"  {i + 1}: n={res_bese["iters"]["n"][i]:3d}, "
            f"[{res_bese["iters"]["a"][i]:.3f}, {res_bese["iters"]["b"][i]:.3f}], "
            f"ESE={res_bese["iters"]["ESE"][i]:.4f}")

    return x, y, res_bese, res_bede


def plot_results():
    """Create visualization of all three cases."""
    fig, axes = plt.subplots(1, 3, figsize = (15, 5))

    ## Total symmetry
    x1, y1, bese1, bede1 = fisher_pry_total_symmetry()
    axes[0].plot(x1, y1, "b-", linewidth = 2)
    axes[0].axvline(x = 5, color = "green", linestyle = ":", label = "True")
    axes[0].axvline(x = bese1["iplast"], color = "cyan", linestyle = "--", label = f"BESE: {bese1["iplast"]:.3f}")
    axes[0].axvline(x = bede1["iplast"], color = "magenta", linestyle = "--", label = f"BEDE: {bede1["iplast"]:.3f}")
    axes[0].set_title("Total Symmetry")
    axes[0].set_xlabel("x")
    axes[0].set_ylabel("y")
    axes[0].legend()
    axes[0].grid(True, alpha = 0.3)

    ## With noise
    x2, y2, bese2, bede2 = fisher_pry_with_noise()
    axes[1].scatter(x2, y2, alpha = 0.3, s = 1)
    axes[1].axvline(x = 5, color = "green", linestyle = ":", label = "True")
    axes[1].axvline(x = bese2["iplast"], color = "cyan", linestyle = "--", label = f"BESE: {bese2["iplast"]:.3f}")
    axes[1].axvline(x = bede2["iplast"], color = "magenta", linestyle = "--", label = f"BEDE: {bede2["iplast"]:.3f}")
    axes[1].set_title("With Noise")
    axes[1].set_xlabel("x")
    axes[1].set_ylabel("y")
    axes[1].legend()
    axes[1].grid(True, alpha = 0.3)

    ## Left asymmetry
    x3, y3, bese3, bede3 = fisher_pry_left_asymmetry()
    axes[2].plot(x3, y3, "b-", linewidth = 2)
    axes[2].axvline(x = 5, color = "green", linestyle = ":", label = "True")
    axes[2].axvline(x = bese3["iplast"], color = "cyan", linestyle = "--", label = f"BESE: {bese3["iplast"]:.3f}")
    axes[2].axvline(x = bede3["iplast"], color = "magenta", linestyle = "--", label = f"BEDE: {bede3["iplast"]:.3f}")
    axes[2].set_title("Left Asymmetry")
    axes[2].set_xlabel("x")
    axes[2].set_ylabel("y")
    axes[2].legend()
    axes[2].grid(True, alpha = 0.3)

    plt.suptitle("Fisher-Pry Sigmoid: Different Scenarios", fontsize = 14)
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    ## Run all examples
    fisher_pry_total_symmetry()
    fisher_pry_with_noise()
    fisher_pry_left_asymmetry()

    ## Create plots
    print("\n" + "=" * 60)
    print("Creating visualization...")
    print("=" * 60)
    plot_results()
