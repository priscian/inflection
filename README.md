# inflection - Python Package

[![Tests](https://github.com/priscian/inflection/actions/workflows/tests.yml/badge.svg)](https://github.com/priscian/inflection/actions/workflows/tests.yml)
[![Python Version](https://img.shields.io/pypi/pyversions/inflection)](https://pypi.org/project/inflection/)
[![License](https://img.shields.io/github/license/priscian/inflection)](https://github.com/priscian/inflection/blob/main/LICENSE)
[![OS](https://img.shields.io/badge/OS-Windows%20%7C%20Linux%20%7C%20macOS-blue)](https://github.com/priscian/inflection)

## 📖 Description

This package provides methods for finding the inflection point of a curve:
- **ESE** (Extremum Surface Estimator)
- **EDE** (Extremum Distance Estimator)
- **BESE** (Bisection ESE)
- **BEDE** (Bisection EDE)
- **UIK** (Unit Invariant Knee)

✅ **Cross-platform**: Works on Windows, Linux, and macOS\
✅ **Parallel processing**: Optional parallel computation using joblib\
✅ **Fast**: EDE/BEDE methods are O(n) complexity\
✅ **Robust**: Handles noisy data well

Based on the methods described in:
- Christopoulos DT. Developing methods for identifying the inflection point of a convex/concave curve. arXiv preprint arXiv:1206.5478, 2012. [dx.doi.org/10.48550/arXiv.1206.5478](https://dx.doi.org/10.48550/arXiv.1206.5478)
- Christopoulos DT. On the efficient identification of an inflection point. Int J Math Sci Comput 6(1):13–20, 2016. [demovtu.veltech.edu.in/wp-content/uploads/2016/04/Paper-04-2016.pdf](https://demovtu.veltech.edu.in/wp-content/uploads/2016/04/Paper-04-2016.pdf)
- Christopoulos DT. Introducing unit invariant knee (UIK) as an objective choice for elbow point in multivariate data analysis techniques. SSRN, 2016. [dx.doi.org/10.2139/ssrn.3043076](https://dx.doi.org/10.2139/ssrn.3043076)

## 🚀 Installation

### Standard Installation (with all features)
```bash
pip install git+https://github.com/priscian/inflection.git
## Or to force reinstallation:
pip install --force-reinstall --no-cache-dir git+https://github.com/priscian/inflection.git
```

### Install from source
```bash
git clone https://github.com/priscian/inflection.git
cd inflection
pip install -e .
```

## 📊 Quick Start

```python
import numpy as np
from inflection import check_curve, ese, ede, bese, bede, uik

## Create sigmoid data
x = np.linspace(0, 10, 500)
y = 5 + 5 * np.tanh(x - 5) # Inflection point at x=5

## Check curve type
cc = check_curve(x, y)
print(f"Curve type: {cc["ctype"]}") # "convex_concave"

## Find inflection point using different methods
ip_ese = ese(x, y, cc["index"])
print(f"ESE inflection point: {ip_ese["chi"]:.3f}")

ip_ede = ede(x, y, cc["index"])
print(f"EDE inflection point: {ip_ede["chi"]:.3f}")

## Use iterative methods for better accuracy
ip_bese = bese(x, y, cc["index"])
print(f"BESE inflection point: {ip_bese["iplast"]:.3f}")

ip_bede = bede(x, y, cc["index"])
print(f"BEDE inflection point: {ip_bede["iplast"]:.3f}")

## Use UIK method for knee detection
knee = uik(x, y)
print(f"UIK knee point: {knee:.3f}")
```

## ⚡ Performance Tips

### Method Selection by Dataset Size

| Dataset Size | Recommended Method | Parallel? | Typical Time |
|-------------|-------------------|-----------|--------------|
| < 1,000 | Any method | No | < 0.01s |
| 1,000–10,000 | EDE/BEDE | No | < 0.1s |
| 10,000–100,000 | EDE/BEDE | No | < 1s |
| > 100,000 | BEDE | No | < 5s |
| > 100,000 | BESE | Yes | < 10s |

### Parallel Processing

Enable parallel processing for ESE/BESE with large datasets:

```python
## Use all CPU cores
result = ese(x, y, index, doparallel = True, n_jobs = -1)

## Use specific number of cores
result = ese(x, y, index, doparallel = True, n_jobs = 4)
```

## 🧪 Testing

```bash
## Run all tests
pytest

## Run with coverage
pytest --cov=inflection tests/
```

## 📄 License

MIT License—see LICENSE file for details.

## 🙏 Acknowledgments

Python implementation based on the original R package by Demetris T. Christopoulos.
