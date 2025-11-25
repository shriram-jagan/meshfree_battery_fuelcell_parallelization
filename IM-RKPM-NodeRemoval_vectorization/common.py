try:
    print("Forcing use of NumPy backend by rasising a false error")
    raise ImportError  # Force use of regular NumPy instead of legate
    import cupynumeric as np
    import legate_sparse as sparse
    import legate_sparse.linalg as linalg
    from legate.timing import time
    from legate_sparse import csr_array
    from legate_sparse.linalg import spsolve

    use_legate = True
    print(f"Using legate")
except (RuntimeError, ImportError):
    from time import perf_counter_ns

    import numpy as np
    import scipy.sparse as sparse
    import scipy.sparse.linalg as linalg
    from scipy.sparse import csr_array
    from scipy.sparse.linalg import spsolve

    def time():
        return perf_counter_ns() / 1000.0

    def spmv(A, x, out):
        for i in range(A.shape[0]):
            begin, end = A.indptr[i], A.indptr[i + 1]
            indices = A.indices[begin:end]
            out[i] = max(x[indices].tolist())

    use_legate = False
    print(f"Using numpy")
try:
    from matplotlib import pyplot as plt

    use_matplotlib = True
except (RuntimeError, ImportError):
    print(f"Matplotlib not found")
    use_matplotlib = False
