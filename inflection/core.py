"""
Core functions for inflection point detection.
Cross-platform compatible version using joblib for parallel processing.
"""

import numpy as np
import warnings
from typing import Dict, Optional
import sys
import platform

## Try to import joblib for parallel processing
try:
    from joblib import Parallel, delayed
    JOBLIB_AVAILABLE = True
except ImportError:
    JOBLIB_AVAILABLE = False
    warnings.warn(
        "joblib not installed. Parallel processing will not be available. "
        "Install with: pip install joblib",
        ImportWarning
    )


## Helper function for linear interpolation

# def lin2(x1, y1, x2, y2, x):
#     """Linear interpolation between two points."""
#     return y1 + (y2 - y1) * (x - x1) / (x2 - x1)

def lin2(x1, y1, x2, y2, x):
    """Linear interpolation between two points."""
    ## Handle case where x2 == x1 to avoid division by zero
    if np.isclose(x2, x1):
        return np.full_like(x, y1)
    return y1 + (y2 - y1) * (x - x1) / (x2 - x1)


## Function to find inflection point for a given index j
def findipl(x, y, j):
    """Compute left and right surface areas for a given index j."""
    n = len(x)

    ## sl (left surface)
    dxl = np.diff(x[0:j])
    fl = y[0:j] - lin2(x[0], y[0], x[j - 1], y[j - 1], x[0:j])
    sb2l = 0.5 * (fl[0:(j - 1)] + fl[1:j])
    sl = np.sum(dxl * sb2l)

    ## sr (right surface)
    dxr = np.diff(x[j - 1:n])
    fr = y[j - 1:n] - lin2(x[j - 1], y[j - 1], x[n - 1], y[n - 1], x[j - 1:n])
    sb2r = 0.5 * (fr[0:(len(fr) - 1)] + fr[1:len(fr)])
    sr = np.sum(dxr * sb2r)

    return j, x[j - 1], sl, sr


## Check curve convexity
def check_curve(x, y):
    """
    Function to decide for the convexity type of the curve.
    Returns a dictionary with "ctype" and "index"
    """
    N = len(x)

    ## Quantile indices
    j0 = 0
    j1 = int(np.quantile(range(N), 0.25))
    j2 = int(np.quantile(range(N), 0.50))
    j3 = int(np.quantile(range(N), 0.75))
    jn = N - 1

    ## Check convexity by computing s_l and s_r surfaces
    LR0 = findipl(x, y, j0 + 1)
    sl0, sr0 = LR0[2], LR0[3]

    LR1 = findipl(x, y, j1 + 1)
    sl1, sr1 = LR1[2], LR1[3]

    LR2 = findipl(x, y, j2 + 1)
    sl2, sr2 = LR2[2], LR2[3]

    LR3 = findipl(x, y, j3 + 1)
    sl3, sr3 = LR3[2], LR3[3]

    LRn = findipl(x, y, jn + 1)
    sln, srn = LRn[2], LRn[3]

    ## Analyze signs
    sleft = [sl1, sl2, sl3, sln]
    sright = [sr0, sr1, sr2, sr3]

    leftsigns = np.sign(sleft)
    rightsigns = np.sign(sright)

    uleft = np.unique(leftsigns)
    uright = np.unique(rightsigns)

    ## Determine left side convexity
    if len(uleft) == 1:
        cleft = "concave" if uleft[0] > 0 else "convex"
    else:
        cleft = "concave" if leftsigns[0] > 0 else "convex"

    ## Determine right side convexity
    if len(uright) == 1:
        cright = "concave" if uright[0] > 0 else "convex"
    else:
        cright = "concave" if rightsigns[-1] > 0 else "convex"

    ## Determine overall curve type
    ctype = f"{cleft}_{cright}"

    ## Set index for ESE/EDE methods
    if ctype == "convex_concave":
        index = 0
    elif ctype == "concave_convex":
        index = 1
    elif ctype == "convex_convex":
        index = 0
    elif ctype == "concave_concave":
        index = 1
    else:
        index = 0

    return {'ctype': ctype, 'index': index}


## EDE method (Extremum Distance Estimator)
def ede(x, y, index):
    """Output for EDE method as defined theoretically in papers by Christopoulos."""
    n = len(x)

    ## For convex/concave data (upward sigmoid) give index = 0
    ## For concave/convex data (downward sigmoid) give index = 1
    if index == 1:
        y = -y

    if n >= 4:
        LF = y - lin2(x[0], y[0], x[n - 1], y[n - 1], x)
        jf1 = np.argmin(LF)
        xf1 = x[jf1]
        jf2 = np.argmax(LF)
        xf2 = x[jf2]

        if jf2 < jf1:
            xfx = np.nan
        else:
            xfx = 0.5 * (xf1 + xf2)
    else:
        jf1 = np.nan
        jf2 = np.nan
        xfx = np.nan

    return {'j1': jf1, 'j2': jf2, 'chi': xfx}


## ESE method (Extremum Surface Estimator) with joblib
def ese(x, y, index, doparallel = False, n_jobs = -1):
    """
    Output for ESE method as defined theoretically in papers by Christopoulos.

    Parameters
    ----------
    x : np.ndarray
        X coordinates
    y : np.ndarray
        Y coordinates
    index : int
        0 for convex/concave, 1 for concave/convex
    doparallel : bool
        Use parallel processing (requires joblib)
    n_jobs : int
        Number of parallel jobs. -1 means use all processors.
        Only used if doparallel=True and joblib is available.

    Returns
    -------
    dict
        Dictionary with j1, j2, and chi (inflection point)
    """
    n = len(x)

    ## For convex/concave data (upward sigmoid) give index=0
    ## For concave/convex data (downward sigmoid) give index=1
    if index == 1:
        y = -y

    if n >= 4:
        ## Decide whether to use parallel processing
        use_parallel = doparallel and JOBLIB_AVAILABLE and n > 100  ## Only parallelize for larger datasets

        if doparallel and not JOBLIB_AVAILABLE:
            warnings.warn(
                "Parallel processing requested but joblib not installed. "
                "Using sequential computation. Install joblib with: pip install joblib",
                RuntimeWarning
            )

        if use_parallel:
            ## Use joblib for parallel processing (works on both Windows and Linux)
            try:
                ## Joblib handles the complexity of cross-platform parallel processing
                results = Parallel(n_jobs = n_jobs, backend = "loky")(
                    delayed(findipl)(x, y, i) for i in range(2, n)
                )
                slsr = np.array([[r[2], r[3]] for r in results])

            except Exception as e:
                ## Fallback to sequential if parallel fails
                warnings.warn(f"Parallel processing failed: {e}. Using sequential computation.")
                slsr = np.array([findipl(x, y, i)[2:4] for i in range(2, n)])
        else:
            ## Sequential computation
            slsr = np.array([findipl(x, y, i)[2:4] for i in range(2, n)])

        jl = np.argmin(slsr[:, 0]) + 2
        jr = np.argmax(slsr[:, 1]) + 2
        xl = x[jl - 1]
        xr = x[jr - 1]

        if jl - jr >= 2:
            xs = 0.5 * (xl + xr)
        else:
            xs = np.nan
    else:
        jl = np.nan
        jr = np.nan
        xs = np.nan

    return {'j1': jr, 'j2': jl, 'chi': xs}


## Faster BEDE method (Bisection EDE)
def bede(x, y, index, max_iter = 50):
    """
    Output for BEDE method -- iterative bisection version of EDE.

    Parameters
    ----------
    x : np.ndarray
        X coordinates
    y : np.ndarray
        Y coordinates
    index : int
        0 for convex/concave, 1 for concave/convex
    max_iter : int
        Maximum number of iterations (default 50, more than enough)

    Returns
    -------
    dict
        Dictionary with "iplast" and "iters"
    """
    import numpy as np

    EDE = []
    BEDE = []
    a = [x[0]]
    b = [x[-1]]
    nped = [len(x)]
    x2 = x.copy()
    y2 = y.copy()

    B = ede(x, y, index)
    EDE.append(B["chi"])
    BEDE.append(B["chi"])
    iplast = B["chi"]

    ## EDE iterations
    j = 0
    prev_chi = None
    convergence_tol = 1e-10 # Stop if chi changes less than this
    min_reduction_factor = 0.9 # Require at least 10% reduction in size

    while not np.isnan(B["chi"]) and j < max_iter:
        ## Check for convergence
        if prev_chi is not None:
            if abs(B["chi"] - prev_chi) < convergence_tol:
                ## Converged -- chi not changing significantly
                break

        prev_chi = B["chi"]

        ## Check if we can continue bisecting
        if B["j2"] >= B["j1"] + 3:
            j = j + 1

            ## Calculate new slice size
            old_size = len(x2)
            new_slice_size = B["j2"] - B["j1"] + 1

            ## Check if we"re actually reducing the problem size
            if new_slice_size > old_size * min_reduction_factor:
                ## Not reducing enough -- likely due to clean data
                ##   ... so force a more aggressive bisection, take middle 50% of current range
                mid_point = len(x2) // 2
                quarter = len(x2) // 4

                ## Ensure we still have enough points
                if quarter > 2:
                    start_idx = max(0, mid_point - quarter)
                    end_idx = min(len(x2), mid_point + quarter)
                    x2 = x2[start_idx:end_idx]
                    y2 = y2[start_idx:end_idx]
                else:
                    ## Too few points to continue
                    break
            else:
                ## Normal bisection
                x2 = x2[B["j1"]:B["j2"] + 1]
                y2 = y2[B["j1"]:B["j2"] + 1]

            ## Check if we have enough points to continue
            if len(x2) < 4:
                break

            B = ede(x2, y2, index)

            if not np.isnan(B["chi"]):
                ## Only record if we got a valid result
                ##   ... but check if the indices are valid for the current slice
                if B["j1"] < len(x2) and B["j2"] < len(x2):
                    a.append(x2[B["j1"]])
                    b.append(x2[B["j2"]])
                    nped.append(len(x2))
                    EDE.append(B["chi"])
                    BEDE.append(B["chi"])
                    iplast = B["chi"]
            else:
                break
        else:
            break

    ## Add warning if we hit max iterations
    if j >= max_iter:
        # import warnings
        warnings.warn(f"BEDE reached maximum iterations ({max_iter}). Result may not be fully converged.")

    ## Set output
    iters = {'n': nped, 'a': a, 'b': b, 'EDE': BEDE}
    return {'iplast': iplast, 'iters': iters}


## Original BEDE method (Bisection EDE)
def bede_orig(x, y, index):
    """
    Output for BEDE method -- iterative bisection version of EDE.

    Parameters
    ----------
    x : np.ndarray
        X coordinates
    y : np.ndarray
        Y coordinates
    index : int
        0 for convex/concave, 1 for concave/convex

    Returns
    -------
    dict
        Dictionary with "iplast" and "iters"
    """
    EDE = []
    BEDE = []
    a = [x[0]]
    b = [x[-1]]
    nped = [len(x)]
    x2 = x.copy()
    y2 = y.copy()

    B = ede(x, y, index)
    EDE.append(B["chi"])
    BEDE.append(B["chi"])
    iplast = B["chi"]

    ## EDE iterations
    j = 0
    while not np.isnan(B["chi"]):
        if B["j2"] >= B["j1"] + 3:
            j = j + 1
            x2 = x2[B["j1"]:B["j2"] + 1]
            y2 = y2[B["j1"]:B["j2"] + 1]
            B = ede(x2, y2, index)

            if not np.isnan(B["chi"]):
                a.append(x2[B["j1"]])
                b.append(x2[B["j2"]])
                nped.append(len(x2))
                EDE.append(B["chi"])
                BEDE.append(B["chi"])
                iplast = B["chi"]
            else:
                break
        else:
            break

    ## Set output
    iters = {'n': nped, 'a': a, 'b': b, 'EDE': BEDE}
    return {'iplast': iplast, 'iters': iters}


## BESE method (Bisection ESE)
def bese(x, y, index, doparallel = False, n_jobs = -1, max_iter = 50):
    """
    Output for BESE method - iterative bisection version of ESE.

    Parameters
    ----------
    x : np.ndarray
        X coordinates
    y : np.ndarray
        Y coordinates
    index : int
        0 for convex/concave, 1 for concave/convex
    doparallel : bool
        Use parallel processing for the first iteration (requires joblib)
    n_jobs : int
        Number of parallel jobs. -1 means use all processors.
    max_iter : int
        Maximum number of iterations (default 50)

    Returns
    -------
    dict
        Dictionary with "iplast" and "iters"
    """
    import numpy as np
    import warnings

    ESE = []
    BESE = []
    a = [x[0]]
    b = [x[-1]]
    npes = [len(x)]
    x2 = x.copy()
    y2 = y.copy()

    ## First iteration might benefit from parallel if dataset is large
    A = ese(x, y, index, doparallel, n_jobs)
    ESE.append(A["chi"])
    BESE.append(A["chi"])
    iplast = A["chi"]

    ## ESE iterations
    j = 0
    prev_chi = None
    convergence_tol = 1e-10  ## Stop if chi changes less than this
    min_reduction_factor = 0.9  ## Require at least 10% reduction in size
    stall_count = 0

    while not np.isnan(A["chi"]) and j < max_iter:
        ## Check for convergence
        if prev_chi is not None:
            if abs(A["chi"] - prev_chi) < convergence_tol:
                stall_count += 1
                if stall_count >= 3:  ## If chi hasn"t changed for 3 iterations
                    break
            else:
                stall_count = 0

        prev_chi = A["chi"]

        ## Check if we can continue bisecting
        if A["j2"] >= A["j1"] + 3:
            j = j + 1

            ## Calculate new slice size
            old_size = len(x2)
            new_slice_size = A["j2"] - A["j1"] + 1

            ## Check if we"re actually reducing the problem size
            if new_slice_size > old_size * min_reduction_factor:
                ## Not reducing enough - likely due to clean data
                ## Force a more aggressive bisection
                ## Take middle 50% of current range
                mid_point = len(x2) // 2
                quarter = len(x2) // 4

                ## Ensure we still have enough points
                if quarter > 2:
                    start_idx = max(0, mid_point - quarter)
                    end_idx = min(len(x2), mid_point + quarter)
                    x2 = x2[start_idx:end_idx]
                    y2 = y2[start_idx:end_idx]
                else:
                    ## Too few points to continue
                    break
            else:
                ## Normal bisection
                x2 = x2[A["j1"]:A["j2"] + 1]
                y2 = y2[A["j1"]:A["j2"] + 1]

            ## Check if we have enough points to continue
            if len(x2) < 4:
                break

            ## Don"t use parallel for iterations (datasets get smaller)
            ## Also, ESE is slower than EDE, so parallel might help less
            A = ese(x2, y2, index, doparallel = False)

            if not np.isnan(A["chi"]):
                ## Only record if we got a valid result
                ## But check if the indices are valid for the current slice
                if A["j1"] < len(x2) and A["j2"] < len(x2):
                    a.append(x2[A["j1"]])
                    b.append(x2[A["j2"]])
                    npes.append(len(x2))
                    ESE.append(A["chi"])
                    BESE.append(A["chi"])
                    iplast = A["chi"]
            else:
                break
        else:
            break

    ## Add warning if we hit max iterations
    if j >= max_iter:
        warnings.warn(f"BESE reached maximum iterations ({max_iter}). Result may not be fully converged.")

    ## Set output
    iters = {'n': npes, 'a': a, 'b': b, 'ESE': BESE}
    return {'iplast': iplast, 'iters': iters}


## BESE method (Bisection ESE)
def bese_orig(x, y, index, doparallel = False, n_jobs = -1):
    """
    Output for BESE method - iterative bisection version of ESE.

    Parameters
    ----------
    x : np.ndarray
        X coordinates
    y : np.ndarray
        Y coordinates
    index : int
        0 for convex/concave, 1 for concave/convex
    doparallel : bool
        Use parallel processing for the first iteration (requires joblib)
    n_jobs : int
        Number of parallel jobs. -1 means use all processors.

    Returns
    -------
    dict
        Dictionary with "iplast" and "iters"
    """
    ESE = []
    BESE = []
    a = [x[0]]
    b = [x[-1]]
    npes = [len(x)]
    x2 = x.copy()
    y2 = y.copy()

    ## First iteration might benefit from parallel if dataset is large
    A = ese(x, y, index, doparallel, n_jobs)
    ESE.append(A["chi"])
    BESE.append(A["chi"])
    iplast = A["chi"]

    ## ESE iterations
    j = 0
    while not np.isnan(A["chi"]):
        if A["j2"] >= A["j1"] + 3:
            j = j + 1
            x2 = x2[A["j1"]:A["j2"] + 1]
            y2 = y2[A["j1"]:A["j2"] + 1]
            ## Don"t use parallel for iterations (datasets get smaller)
            A = ese(x2, y2, index, doparallel = False)

            if not np.isnan(A["chi"]):
                a.append(x2[A["j1"]])
                b.append(x2[A["j2"]])
                npes.append(len(x2))
                ESE.append(A["chi"])
                BESE.append(A["chi"])
                iplast = A["chi"]
            else:
                break
        else:
            break

    ## Set output
    iters = {'n': npes, 'a': a, 'b': b, 'ESE': BESE}
    return {'iplast': iplast, 'iters': iters}


## EDECI method (EDE with Confidence Interval)
def edeci(x, y, index, k = 5):
    """Value and Chebyshev confidence interval for EDE."""
    tede = ede(x, y, index)
    m = tede["chi"]

    if not np.isnan(m):
        dy = np.diff(y)
        ss = 0.25 * np.sum(dy ** 2)
        s = np.sqrt(2 * ss / (len(y) - 1))

        ## Find CI
        xleft = m - k * s
        xright = m + k * s
    else:
        xleft = np.nan
        xright = np.nan

    out = {
        "j1": tede["j1"],
        "j2": tede["j2"],
        "chi": m,
        "k": k,
        f"chi-{k}*s": xleft,
        f"chi+{k}*s": xright
    }

    return out


## Combined ESE and EDE results
def findiplist(x, y, index, doparallel = False, n_jobs = -1):
    """
    Output for ESE, EDE methods combined.

    Parameters
    ----------
    x : np.ndarray
        X coordinates
    y : np.ndarray
        Y coordinates
    index : int
        0 for convex/concave, 1 for concave/convex
    doparallel : bool
        Use parallel processing for ESE (requires joblib)
    n_jobs : int
        Number of parallel jobs for ESE

    Returns
    -------
    dict
        Dictionary with ESE and EDE results
    """
    n = len(x)

    ## For convex/concave data (upward sigmoid) give index=0
    ## For concave/convex data (downward sigmoid) give index=1
    if index == 1:
        y = -y

    if n >= 4:
        A = ese(x, y, index, doparallel, n_jobs)
        B = ede(x, y, index)
    else:
        warnings.warn("Insufficient number of points, please provide at least 4 points!")
        A = {'j1': np.nan, 'j2': np.nan, 'chi': np.nan}
        B = {'j1': np.nan, 'j2': np.nan, 'chi': np.nan}

    out = {'ESE': A, 'EDE': B}
    return out


## UIK method (Unit Invariant Knee)
def uik(x, y):
    """Output for UIK method as defined in Christopoulos (2016)."""
    if len(x) <= 3:
        raise ValueError("Method is not applicable for such a small vector. Please give at least a 5 numbers vector")

    ## Check convexity or at least leading convexity
    cxv = check_curve(x, y)
    knee_result = ede(x, y, cxv["index"])
    knee = x[knee_result["j1"]]

    return knee


## Main test routine with UIK included
if __name__ == "__main__":
    import matplotlib.pyplot as plt
    import time

    ## Detect OS and joblib availability
    os_name = platform.system()
    print(f"Running on: {os_name}")
    print(f"Joblib available: {JOBLIB_AVAILABLE}")

    ## Create example sigmoid data
    x = np.linspace(0, 10, 501)
    y = 5 + 5 * np.tanh(x - 5)  ## Known inflection point at x=5

    ## Add some noise
    np.random.seed(666)
    y = y + 0.1 * np.random.randn(len(y))

    ## Check curve type
    cc = check_curve(x, y)
    print(f"\nCurve type: {cc["ctype"]}")
    print(f"Index for methods: {cc["index"]}")

    ## Test all methods
    print("\n=== Testing All Methods ===")

    ## ESE
    ipese = ese(x, y, cc["index"], doparallel = False)
    print(f"ESE inflection point: {ipese["chi"]:.3f}")

    ## EDE
    ipede = ede(x, y, cc["index"])
    print(f"EDE inflection point: {ipede["chi"]:.3f}")

    ## BESE
    ipbese = bese(x, y, cc["index"], doparallel = False)
    print(f"BESE final inflection point: {ipbese["iplast"]:.3f}")
    print(f"  Iterations: {len(ipbese["iters"]["n"])}")

    ## BEDE
    ipbede = bede(x, y, cc["index"])
    print(f"BEDE final inflection point: {ipbede["iplast"]:.3f}")
    print(f"  Iterations: {len(ipbede["iters"]["n"])}")

    ## UIK
    knee = uik(x, y)
    print(f"UIK knee point: {knee:.3f}")

    ## Plot results
    plt.figure(figsize = (10, 6))
    plt.scatter(x, y, alpha = 0.3, s = 1, label = "Data")
    plt.axvline(x = ipese["chi"], color = "blue", linestyle = "--", alpha = 0.7, label = f"ESE: {ipese["chi"]:.3f}")
    plt.axvline(x = ipede["chi"], color = "red", linestyle = "--", alpha = 0.7, label = f"EDE: {ipede["chi"]:.3f}")
    plt.axvline(x = ipbese["iplast"], color = "cyan", linestyle = "-", alpha = 0.7, label = f"BESE: {ipbese["iplast"]:.3f}")
    plt.axvline(x = ipbede["iplast"], color = "magenta", linestyle = "-", alpha = 0.7, label = f"BEDE: {ipbede["iplast"]:.3f}")
    plt.axvline(x = knee, color = "orange", linestyle = "-.", alpha = 0.7, label = f"UIK: {knee:.3f}")
    plt.axvline(x = 5, color = "green", linestyle = ":", linewidth = 2, label = "True: 5.000")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.title(f"Inflection Point Detection ({os_name})")
    plt.legend()
    plt.grid(True, alpha = 0.3)
    plt.tight_layout()
    plt.show()
