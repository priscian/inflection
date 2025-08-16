"""Basic usage examples for the inflection package."""

import numpy as np
import matplotlib.pyplot as plt
from inflection import check_curve, ese, ede, bese, bede, uik


def main():
    """Demonstrate basic usage of inflection package."""

    ## Create example sigmoid data
    print("Creating sigmoid data with inflection point at x=5...")
    x = np.linspace(0, 10, 501)
    y = 5 + 5 * np.tanh(x - 5)

    ## Add some noise
    np.random.seed(666)
    y_noisy = y + 0.1 * np.random.randn(len(y))

    ## Check curve type
    cc = check_curve(x, y_noisy)
    print(f"\nCurve type: {cc["ctype"]}")
    print(f"Index for methods: {cc["index"]}")

    ## Find inflection point using different methods
    print("\n" + "=" * 50)
    print("Finding inflection point with different methods:")
    print("="*50)

    ## ESE method
    ipese = ese(x, y_noisy, cc["index"])
    print(f"\nESE inflection point: {ipese["chi"]:.4f}")
    print(f"  Indices: j1={ipese["j1"]}, j2={ipese["j2"]}")

    ## EDE method
    ipede = ede(x, y_noisy, cc["index"])
    print(f"\nEDE inflection point: {ipede["chi"]:.4f}")
    print(f"  Indices: j1={ipede["j1"]}, j2={ipede["j2"]}")

    ## BESE method (iterative)
    ipbese = bese(x, y_noisy, cc["index"])
    print(f"\nBESE final inflection point: {ipbese["iplast"]:.4f}")
    print(f"  Number of iterations: {len(ipbese["iters"]["n"])}")
    print("  Iteration details:")
    for i in range(len(ipbese["iters"]["n"])):
        print(f"    {i+1}: n={ipbese["iters"]["n"][i]:4d}, "
            f"[{ipbese["iters"]["a"][i]:.3f}, {ipbese["iters"]["b"][i]:.3f}], "
            f"ESE={ipbese["iters"]["ESE"][i]:.4f}")

    ## BEDE method (iterative)
    ipbede = bede(x, y_noisy, cc["index"])
    print(f"\nBEDE final inflection point: {ipbede["iplast"]:.4f}")
    print(f"  Number of iterations: {len(ipbede["iters"]["n"])}")

    ## UIK method
    knee = uik(x, y_noisy)
    print(f"\nUIK knee point: {knee:.4f}")

    ## Plotting
    print("\n" + "=" * 50)
    print("Creating visualization...")
    print("=" * 50)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize = (12, 5))

    ## Plot 1: Original data with all methods
    ax1.scatter(x, y_noisy, alpha = 0.3, s = 1, label = "Noisy data")
    ax1.plot(x, y, "k-", alpha = 0.5, label = "True function")
    ax1.axvline(x = 5, color = "green", linestyle = ":", label = "True IP (5.0)")
    ax1.axvline(x = ipese["chi"], color = "blue", linestyle = "--", alpha = 0.7, label = f"ESE: {ipese["chi"]:.3f}")
    ax1.axvline(x = ipede["chi"], color = "red", linestyle = "--", alpha = 0.7, label = f"EDE: {ipede["chi"]:.3f}")
    ax1.axvline(x = ipbese["iplast"], color = "cyan", linestyle = "-", alpha = 0.7, label = f"BESE: {ipbese["iplast"]:.3f}")
    ax1.axvline(x = ipbede["iplast"], color = "magenta", linestyle = "-", alpha = 0.7, label = f"BEDE: {ipbede["iplast"]:.3f}")
    ax1.axvline(x = knee, color = "orange", linestyle = "-.", alpha = 0.7, label = f"UIK: {knee:.3f}")
    ax1.set_xlabel("x")
    ax1.set_ylabel("y")
    ax1.set_title("Inflection Point Detection - All Methods")
    ax1.legend(loc = "best", fontsize = 8)
    ax1.grid(True, alpha = 0.3)

    ## Plot 2: Zoomed in around inflection point
    zoom_range = 1.5
    zoom_mask = (x > 5 - zoom_range) & (x < 5 + zoom_range)
    x_zoom = x[zoom_mask]
    y_zoom = y[zoom_mask]
    y_noisy_zoom = y_noisy[zoom_mask]

    ax2.scatter(x_zoom, y_noisy_zoom, alpha = 0.5, s = 5, label = "Noisy data")
    ax2.plot(x_zoom, y_zoom, "k-", alpha = 0.5, label = "True function")
    ax2.axvline(x = 5, color = "green", linestyle = ":", linewidth = 2, label = "True IP")
    ax2.axvline(x = ipbese["iplast"], color = "cyan", linestyle = "-", alpha = 0.7, label = f"BESE: {ipbese["iplast"]:.3f}")
    ax2.axvline(x = ipbede["iplast"], color = "magenta", linestyle = "-", alpha = 0.7, label = f"BEDE: {ipbede["iplast"]:.3f}")
    ax2.axvline(x = knee, color = "orange", linestyle = "-.", alpha = 0.7, label = f"UIK: {knee:.3f}")
    ax2.set_xlabel("x")
    ax2.set_ylabel("y")
    ax2.set_title("Zoomed View Around Inflection Point")
    ax2.legend(loc = "best")
    ax2.grid(True, alpha = 0.3)

    plt.tight_layout()
    plt.show()

    ## Summary
    print("\n" + "=" * 50)
    print("Summary of Results:")
    print("="*50)
    print(f"True inflection point:    5.0000")
    print(f"ESE estimation:           {ipese["chi"]:.4f}  (error: {abs(ipese["chi"] - 5.0):.4f})")
    print(f"EDE estimation:           {ipede["chi"]:.4f}  (error: {abs(ipede["chi"] - 5.0):.4f})")
    print(f"BESE estimation:          {ipbese["iplast"]:.4f}  (error: {abs(ipbese["iplast"] - 5.0):.4f})")
    print(f"BEDE estimation:          {ipbede["iplast"]:.4f}  (error: {abs(ipbede["iplast"] - 5.0):.4f})")
    print(f"UIK estimation:           {knee:.4f}  (error: {abs(knee - 5.0):.4f})")
    print("\nNote: BESE and BEDE typically provide the most accurate results")
    print("through their iterative refinement process.")


if __name__ == "__main__":
    main()
