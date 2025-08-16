# API Reference

## Core Functions

### check_curve(x, y)
Determines the convexity type of a curve.

**Parameters:**
- `x` (np.ndarray): X coordinates of the curve
- `y` (np.ndarray): Y coordinates of the curve

**Returns:**
- dict: Contains "ctype" (curve type) and "index" (0 or 1 for methods)

### ese(x, y, index, doparallel = False, n_jobs = -1)
Extremum Surface Estimator method for finding inflection points.

**Parameters:**
- `x` (np.ndarray): X coordinates
- `y` (np.ndarray): Y coordinates
- `index` (int): 0 for convex/concave, 1 for concave/convex
- `doparallel` (bool): Use parallel processing (requires joblib)
- `n_jobs` (int): Number of parallel jobs (-1 for all cores)

**Returns:**
- dict: Contains "j1", "j2" (indices) and "chi" (inflection point)

### ede(x, y, index)
Extremum Distance Estimator method for finding inflection points.

**Parameters:**
- `x` (np.ndarray): X coordinates
- `y` (np.ndarray): Y coordinates
- `index` (int): 0 for convex/concave, 1 for concave/convex

**Returns:**
- dict: Contains "j1", "j2" (indices) and "chi" (inflection point)

### bese(x, y, index, doparallel = False, n_jobs = -1)
Bisection ESE—iterative refinement of ESE method.

**Parameters:**
- `x` (np.ndarray): X coordinates
- `y` (np.ndarray): Y coordinates
- `index` (int): 0 for convex/concave, 1 for concave/convex
- `doparallel` (bool): Use parallel processing
- `n_jobs` (int): Number of parallel jobs
- `max_iter` (int): Maximum number of iterations (default 50)

**Returns:**
- dict: Contains "iplast" (final inflection point) and "iters" (iteration details)

### bede(x, y, index)
Bisection EDE—iterative refinement of EDE method.

**Parameters:**
- `x` (np.ndarray): X coordinates
- `y` (np.ndarray): Y coordinates
- `index` (int): 0 for convex/concave, 1 for concave/convex
- `max_iter` (int): Maximum number of iterations (default 50)


**Returns:**
- dict: Contains "iplast" (final inflection point) and "iters" (iteration details)

### uik(x, y)
Unit Invariant Knee method for finding the knee/elbow point of a curve.

**Parameters:**
- `x` (np.ndarray): X coordinates (minimum 4 points)
- `y` (np.ndarray): Y coordinates

**Returns:**
- float: X-coordinate of the knee point

**Raises:**
- ValueError: If fewer than 4 points provided
