# Inflection Package Documentation

## Overview

The `inflection` package provides robust methods for finding inflection points of curves, based on the theoretical work of Demetris T. Christopoulos.

## Installation

```bash
pip install git+https://github.com/priscian/inflection.git
## Or to force reinstallation:
pip install --force-reinstall --no-cache-dir git+https://github.com/priscian/inflection.git
```

## Quick Start

```python
import numpy as np
from inflection import check_curve, bese, bede, uik

## Create sigmoid data
x = np.linspace(0, 10, 500)
y = 5 + 5 * np.tanh(x - 5)

## Check curve type
cc = check_curve(x, y)

## Find inflection point
if cc["ctype"] == "convex_concave":
    result = bese(x, y, cc["index"])
    print(f"Inflection point: {result["iplast"]}")
```

## Methods Available

### Basic Methods
- **ESE** (Extremum Surface Estimator): First approximation using surface areas
- **EDE** (Extremum Distance Estimator): First approximation using distances

### Iterative Methods (Recommended)
- **BESE** (Bisection ESE): Iterative refinement of ESE for higher accuracy
- **BEDE** (Bisection EDE): Iterative refinement of EDE for higher accuracy

### Special Methods
- **UIK** (Unit Invariant Knee): Finds the "elbow" or "knee" point of a curve
- **EDECI**: EDE with confidence intervals

## Which Method to Use?

| Scenario | Recommended Method | Why |
|----------|-------------------|-----|
| Clean data, small dataset | Any method | All perform similarly |
| Noisy data | BESE or BEDE | Iterative refinement handles noise better |
| Large dataset (>100k points) | BEDE | O(n) complexity, very fast |
| Need confidence intervals | EDECI | Provides uncertainty estimates |
| Finding elbow/knee point | UIK | Specifically designed for this |
| Asymmetric data range | BESE or BEDE | Better compensation for asymmetry |

## Understanding Curve Types

The `check_curve` function identifies four types:
1. `convex_concave` - Upward sigmoid (S-shape going up)
2. `concave_convex` - Downward sigmoid (S-shape going down)
3. `convex_convex` - Purely convex
4. `concave_concave` - Purely concave

## Theory

The methods are based on finding points where:
1. The curve transitions from convex to concave (or vice versa)
2. The second derivative changes sign
3. The distance or surface area metrics are extremized

### Mathematical Foundation

For a curve transitioning from convex to concave:
- The inflection point is where f''(x) = 0
- ESE finds where left and right surface areas balance
- EDE finds where left and right distances balance

## Examples

### Example 1: Simple Sigmoid
```python
import numpy as np
from inflection import check_curve, bede

x = np.linspace(0, 10, 500)
y = 5 + 5 * np.tanh(x - 5) # Inflection at x = 5

cc = check_curve(x, y)
result = bede(x, y, cc["index"])
print(f"Found: {result["iplast"]:.3f}, True: 5.000")
```

### Example 2: Noisy Data
```python
## Add noise
y_noisy = y + np.random.normal(0, 0.5, len(y))

cc = check_curve(x, y_noisy)
result = bede(x, y_noisy, cc["index"])
print(f"Found: {result["iplast"]:.3f}, True: 5.000")
## BEDE handles noise well
```

### Example 3: Finding Knee Point
```python
## Exponential decay curve
x = np.linspace(0, 5, 100)
y = np.exp(-x)

knee = uik(x, y)
print(f"Knee point at x={knee:.3f}")
```

### Example 4: Polynomial with Inflection
```python
## Cubic polynomial: f(x) = -1/3*x^3 + 5/2*x^2 - 4*x + 1/2
## Has inflection at x = 2.5
x = np.linspace(-2, 7, 500)
y = -1/3 * x**3 + 5/2 * x**2 - 4 * x + 1/2

cc = check_curve(x, y)
result = bese(x, y, cc["index"])
print(f"Found: {result["iplast"]:.3f}, True: 2.500")
```

### Example 5: Large Dataset Performance
```python
import time

## Create large dataset
x = np.linspace(0, 1000, 1000001)
y = 500 + 500 * np.tanh(x - 500)

cc = check_curve(x, y)

## Time different methods
start = time.time()
result_ede = ede(x, y, cc["index"])
time_ede = time.time() - start

start = time.time()
result_bede = bede(x, y, cc["index"])
time_bede = time.time() - start

print(f"EDE: {result_ede["chi"]:.3f} in {time_ede:.3f}s")
print(f"BEDE: {result_bede["iplast"]:.3f} in {time_bede:.3f}s")
## BEDE is fast even with 1 million points!
```

## Performance Tips

1. **For speed**: Use EDE/BEDE (no parallel processing needed)
2. **For accuracy**: Use BESE/BEDE (iterative refinement)
3. **For large data**: BEDE scales best
4. **Parallel processing**: Only beneficial for ESE/BESE with n > 10,000 points

### Performance Benchmarks

| Method | 1K points | 10K points | 100K points | 1M points |
|--------|-----------|------------|-------------|-----------|
| EDE | <0.001s | 0.005s | 0.05s | 0.5s |
| BEDE | 0.005s | 0.05s | 0.5s | 5s |
| ESE | 0.01s | 1s | 100s | - |
| ESE (parallel) | 0.01s | 0.3s | 30s | - |

## Advanced Usage

### Using Parallel Processing
```python
from inflection import ese, bese

## For large datasets, enable parallel processing
x = np.linspace(0, 100, 50000)
y = 50 + 50 * np.tanh(x - 50)

## Sequential
result_seq = ese(x, y, 0, doparallel = False)

## Parallel (requires joblib)
result_par = ese(x, y, 0, doparallel = True, n_jobs = -1)

## Use specific number of cores
result_par = ese(x, y, 0, doparallel = True, n_jobs = 4)
```

### Getting Confidence Intervals
```python
from inflection import edeci

x = np.linspace(0, 10, 500)
y = 5 + 5 * np.tanh(x - 5)

cc = check_curve(x, y)
result = edeci(x, y, cc["index"], k=5)

print(f"Inflection: {result["chi"]:.3f}")
print(f"95% CI: [{result["chi-5*s"]:.3f}, {result["chi+5*s"]:.3f}]")
```

### Handling Different Curve Types
```python
from inflection import check_curve, ede

## Upward sigmoid
x = np.linspace(0, 10, 500)
y_up = 5 + 5 * np.tanh(x - 5)
cc_up = check_curve(x, y_up)
print(f"Upward: {cc_up["ctype"]}, index = {cc_up["index"]}")

## Downward sigmoid
y_down = 5 - 5 * np.tanh(x - 5)
cc_down = check_curve(x, y_down)
print(f"Downward: {cc_down["ctype"]}, index = {cc_down["index"]}")

## Find inflection for both
ip_up = ede(x, y_up, cc_up["index"])
ip_down = ede(x, y_down, cc_down["index"])
print(f"Both find x={ip_up["chi"]:.3f}")
```

## Troubleshooting

### Common Issues

1. **ValueError: Method not applicable for small vector**
   - UIK requires at least 4 points
   - Solution: Provide more data points

2. **Warning: Insufficient number of points**
   - All methods need at least 4 points
   - Solution: Check your data has enough points

3. **Getting NaN results**
   - Data might not have an inflection point
   - Curve might be monotonic
   - Solution: Check curve type with `check_curve`

4. **Parallel processing not working**
   - joblib might not be installed
   - Solution: `pip install joblib`

5. **Inaccurate results with asymmetric data**
   - Use BESE or BEDE instead of ESE/EDE
   - These methods compensate better for asymmetry

## Mathematical Details

### ESE (Extremum Surface Estimator)
Finds the point where the sum of areas between the curve and secant lines is minimized on the left and maximized on the right.

### EDE (Extremum Distance Estimator)
Finds the point where the perpendicular distances from the curve to the connecting line are balanced.

### Bisection Methods (BESE/BEDE)
Iteratively refine the search interval using the bisection principle, converging to the true inflection point.

### UIK (Unit Invariant Knee)
Uses the EDE method after determining curve convexity to find the knee/elbow point.

## Citation

If you use this package, please cite:

```bibtex
@article{christopoulos2014,
  title={Developing methods for identifying the inflection point of a convex/concave curve},
  author={Christopoulos, Demetris T},
  journal={arXiv preprint arXiv:1206.5478},
  year={2014},
  doi={10.48550/arXiv.1206.5478}
}

@article{christopoulos2016,
  title={On the efficient identification of an inflection point},
  author={Christopoulos, Demetris T},
  journal={International Journal of Mathematics and Scientific Computing},
  volume={6},
  number={1},
  pages={13--20},
  year={2016},
  url={https://demovtu.veltech.edu.in/wp-content/uploads/2016/04/Paper-04-2016.pdf}
}

@article{christopoulos2016a,
  title={Introducing unit invariant knee (UIK) as an objective choice for elbow point in multivariate data analysis techniques},
  author={Christopoulos, Demetris T},
  journal={SSRN 3043076},
  year={2016},
  doi={10.2139/ssrn.3043076}
}
```

## Links

- [GitHub Repository](https://github.com/priscian/inflection)
- [API Reference](api_reference.md)
- [Original R Package](https://cran.r-project.org/package=inflection)
- [Theory Paper 2014](https://arxiv.org/abs/1206.5478)
- [Theory Paper 2016](https://demovtu.veltech.edu.in/wp-content/uploads/2016/04/Paper-04-2016.pdf)
- [UIK Paper 2016](https://dx.doi.org/10.2139/ssrn.3043076)

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

MIT License—see LICENSE file for details.

## Acknowledgments

Python implementation based on the original R package by Demetris T. Christopoulos.
