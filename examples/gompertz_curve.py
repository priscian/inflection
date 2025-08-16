"""
Gompertz curve examples from the R vignettes.
Demonstrates inflection point detection on asymmetric sigmoid curves.
"""

import numpy as np
import matplotlib.pyplot as plt
from inflection import check_curve, ese, ede, bese, bede


def gompertz_no_noise():
    """Gompertz curve without noise."""
    print("=" * 60)
    print("Gompertz Curve (Asymmetric Sigmoid) - No Noise")
    print("=" * 60)

    ## f(x) = 10 * exp(-exp(5) * exp(-x))
    ## Inflection point at x = 5
    x = np.linspace(3.5, 8, 501)
    y = 10 * np.exp(-np.exp(5) * np.exp(-x))

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


def gompertz_with_noise():
    """Gompertz curve with uniform noise."""
    print("\n" + "=" * 60)
    print("Gompertz Curve with Noise")
    print("=" * 60)

    np.random.seed(666)
    x = np.linspace(3.5, 8, 501)
    y = 10 * np.exp(-np.exp(5) * np.exp(-x))
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

    ## Show convergence details
    print("\nBESE convergence:")
    for i in range(min(5, len(res_bese["iters"]["n"]))):
        print(f"  Iteration {i+1}: ESE = {res_bese["iters"]["ESE"][i]:.6f}")

    return x, y_noisy, res_bese, res_bede


def compare_symmetric_vs_asymmetric():
    """Compare Fisher-Pry (symmetric) vs Gompertz (asymmetric)."""
    print("\n" + "=" * 60)
    print("Comparing Symmetric vs Asymmetric Sigmoids")
    print("=" * 60)

    x = np.linspace(2, 8, 501)

    ## Fisher-Pry (symmetric)
    y_fisher = 5 + 5 * np.tanh(x - 5)

    ## Gompertz (asymmetric)
    y_gompertz = 10 * np.exp(-np.exp(5) * np.exp(-x))

    ## Normalize both to [0, 1] for comparison
    y_fisher_norm = (y_fisher - y_fisher.min()) / (y_fisher.max() - y_fisher.min())
    y_gompertz_norm = (y_gompertz - y_gompertz.min()) / (y_gompertz.max() - y_gompertz.min())

    ## Plot
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize = (12, 5))

    ## Plot both curves
    ax1.plot(x, y_fisher_norm, "b-", label = "Fisher-Pry (symmetric)", linewidth = 2)
    ax1.plot(x, y_gompertz_norm, "r-", label = "Gompertz (asymmetric)", linewidth = 2)
    ax1.axvline(x = 5, color = "green", linestyle = ":", alpha = 0.5, label = "True IP")
    ax1.set_xlabel("x")
    ax1.set_ylabel("Normalized y")
    ax1.set_title("Normalized Comparison")
    ax1.legend()
    ax1.grid(True, alpha = 0.3)

    ## Plot derivatives to show asymmetry
    dy_fisher = np.gradient(y_fisher_norm, x)
    dy_gompertz = np.gradient(y_gompertz_norm, x)

    ax2.plot(x, dy_fisher, "b-", label = "Fisher-Pry derivative", linewidth = 2)
    ax2.plot(x, dy_gompertz, "r-", label = "Gompertz derivative", linewidth = 2)
    ax2.axvline(x = 5, color = "green", linestyle = ":", alpha = 0.5, label = "True IP")
    ax2.set_xlabel("x")
    ax2.set_ylabel("dy/dx")
    ax2.set_title("First Derivatives (showing asymmetry)")
    ax2.legend()
    ax2.grid(True, alpha = 0.3)

    plt.tight_layout()
    plt.show()

    print("\nNote: Gompertz curve shows clear asymmetry around the inflection point,")
    print("while Fisher-Pry (tanh) is perfectly symmetric.")


if __name__ == "__main__":
    ## Run examples
    gompertz_no_noise()
    gompertz_with_noise()
    compare_symmetric_vs_asymmetric()
