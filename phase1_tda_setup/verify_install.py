"""Phase 1: TDA 라이브러리 설치 검증"""
import sys
print(f"Python: {sys.version}")

import numpy as np
import pandas as pd
import ripser
import persim
import gudhi

print(f"numpy: {np.__version__}")
print(f"pandas: {pd.__version__}")
print(f"gudhi: {gudhi.__version__}")
print(f"ripser: OK")
print(f"persim: OK")

# Quick smoke test: compute PH on small random data
from ripser import ripser as rips
data = np.random.randn(50, 3)
result = rips(data, maxdim=1)
print(f"\nSmoke test: PH computed on 50 random points in 3D")
print(f"  H0 features: {len(result['dgms'][0])}")
print(f"  H1 features: {len(result['dgms'][1])}")
print("\nAll TDA libraries working correctly!")
