"""
Polynomial curve examples from the R vignettes.
Demonstrates inflection point detection on cubic polynomials.
"""

import numpy as np
import matplotlib.pyplot as plt
from inflection import check_curve, ese, ede, bese, bede


def cubic_polynomial_symmetric():
    """Symmetric 3rd order polynomial with total symmetry."""
    print("=" * 60)
    print("Cubic Polynomial - Symmetric Data Range")
    print("=" * 60)

    ## f(x) = -1/3 * x^3 + 5/2 * x^2 - 4 * x + 1/2
    ## Has inflection point at x = 2.5
    x = np.linspace(-2, 7, 501) # Symmetric around x = 2.5
    y = -1/3 * x**3 + 5/2 * x**2 - 4 * x + 1/2

    cc = check_curve(x, y)
    print(f"Curve type: {cc["ctype"]}")
    print(f"Index: {cc["index"]}")

    ## Apply methods
    res_ese = ese(x, y, cc["index"])
    res_ede = ede(x, y, cc["index"])
    res_bese = bese(x, y, cc["index"])
    res_bede = bede(x, y, cc["index"])

    print(f"\nResults (true value = 2.500):")
    print(f"  ESE:  {res_ese["chi"]:.4f}")
    print(f"  EDE:  {res_ede["chi"]:.4f}")
    print(f"  BESE: {res_bese["iplast"]:.4f}")
    print(f"  BEDE: {res_bede["iplast"]:.4f}")

    return x, y, res_bese


def cubic_polynomial_asymmetric():
    """Asymmetric 3rd order polynomial with data right asymmetry."""
    print("\n" + "=" * 60)
    print("Cubic Polynomial - Asymmetric Data Range")
    print("=" * 60)

    ## Same polynomial but with asymmetric range
    x = np.linspace(-2, 8, 501) # Right asymmetry
    y = -1/3 * x**3 + 5/2 * x**2 - 4 * x + 1/2

    cc = check_curve(x, y)

    ## Apply methods
    res_ese = ese(x, y, cc["index"])
    res_ede = ede(x, y, cc["index"])
    res_bese = bese(x, y, cc["index"])
    res_bede = bede(x, y, cc["index"])

    print(f"\nResults with asymmetry (true value = 2.500):")
    print(f"  ESE:  {res_ese["chi"]:.4f}")
    print(f"  EDE:  {res_ede["chi"]:.4f}")
    print(f"  BESE: {res_bese["iplast"]:.4f}")
    print(f"  BEDE: {res_bede["iplast"]:.4f}")

    ## Apply correction formula from Lemma 2.1 for 3rd order polynomials
    a, b = x[0], x[-1]
    chi_l = res_ese["j2"]
    chi_r = res_ese["j1"]
    if not np.isnan(chi_l) and not np.isnan(chi_r):
        corrected = 1/3 * x[chi_l] + 1/3 * x[chi_r] + 1/6 * a + 1/6 * b
        print(f"\nCorrected estimate (Lemma 2.1): {corrected:.4f}")

    return x, y, res_bese


def cubic_polynomial_with_noise():
    """Cubic polynomial with added noise."""
    print("\n" + "=" * 60)
    print("Cubic Polynomial with Noise")
    print("=" * 60)

    np.random.seed(666)
    x = np.linspace(-2, 7, 501)
    y = -1/3 * x**3 + 5/2 * x**2 - 4 * x + 1/2
    ## Add uniform noise U(-2, 2)
    y_noisy = y + np.random.uniform(-2, 2, len(y))

    cc = check_curve(x, y_noisy)

    ## Apply methods
    res_ese = ese(x, y_noisy, cc["index"])
    res_ede = ede(x, y_noisy, cc["index"])
    res_bese = bese(x, y_noisy, cc["index"])
    res_bede = bede(x, y_noisy, cc["index"])

    print(f"\nResults with noise (true value = 2.500):")
    print(f"  ESE:  {res_ese["chi"]:.4f}")
    print(f"  EDE:  {res_ede["chi"]:.4f}")
    print(f"  BESE: {res_bese["iplast"]:.4f}")
    print(f"  BEDE: {res_bede["iplast"]:.4f}")

    return x, y_noisy, res_bese


def plot_polynomial_examples():
    """Visualize all polynomial examples."""
    fig, axes = plt.subplots(1, 3, figsize = (15, 5))

    ## Symmetric
    x1, y1, bese1 = cubic_polynomial_symmetric()
    axes[0].plot(x1, y1, "b-", linewidth = 2)
    axes[0].axvline(x = 2.5, color = "green", linestyle = ":", label = "True")
    axes[0].axvline(x = bese1["iplast"], color = "cyan", linestyle = "--", label = f"BESE: {bese1["iplast"]:.3f}")
    axes[0].set_title("Symmetric Range")
    axes[0].set_xlabel("x")
    axes[0].set_ylabel("y")
    axes[0].legend()
    axes[0].grid(True, alpha = 0.3)

    ## Asymmetric
    x2, y2, bese2 = cubic_polynomial_asymmetric()
    axes[1].plot(x2, y2, "b-", linewidth = 2)
    axes[1].axvline(x = 2.5, color = "green", linestyle = ":", label = "True")
    axes[1].axvline(x = bese2["iplast"], color = "cyan", linestyle = "--", label = f"BESE: {bese2["iplast"]:.3f}")
    axes[1].set_title("Asymmetric Range")
    axes[1].set_xlabel("x")
    axes[1].set_ylabel("y")
    axes[1].legend()
    axes[1].grid(True, alpha = 0.3)

    ## With noise
    x3, y3, bese3 = cubic_polynomial_with_noise()
    axes[2].scatter(x3, y3, alpha = 0.3, s = 1)
    axes[2].axvline(x = 2.5, color = "green", linestyle = ":", label = "True")
    axes[2].axvline(x = bese3["iplast"], color = "cyan", linestyle = "--", label = f"BESE: {bese3["iplast"]:.3f}")
    axes[2].set_title("With Noise")
    axes[2].set_xlabel("x")
    axes[2].set_ylabel("y")
    axes[2].legend()
    axes[2].grid(True, alpha = 0.3)

    plt.suptitle("Cubic Polynomial: Different Scenarios", fontsize = 14)
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    ## Run examples
    cubic_polynomial_symmetric()
    cubic_polynomial_asymmetric()
    cubic_polynomial_with_noise()

    ## Create plots
    print("\n" + "=" * 60)
    print("Creating visualization...")
    print("=" * 60)
    plot_polynomial_examples()
