try:
    # print("Forcing use of NumPy backend by raising a false error")
    # raise ImportError  # Force use of regular NumPy instead of legate
    import cupynumeric as np
    import legate_sparse as sparse
    import legate_sparse as sp  # alias for scipy.sparse compatibility
    import legate_sparse.linalg as linalg
    from legate.timing import time as legate_time

    # Try to import these, but they might not exist in legate_sparse yet
    from legate_sparse import block_array, csr_array, dia_array
    from legate_sparse.linalg import spsolve

    try:
        from legate_sparse import block_diag, vstack
    except ImportError:
        print(f"legate sparse does not have block_diag and vstack")
        print(f"Make sure to set USE_NUMPY_EQUIVALENTS to True in config.py")
        # These functions might not be implemented in legate_sparse yet
        block_diag = None
        vstack = None

    # Create a time.time() compatible function
    def time():
        return legate_time("us")

    use_legate = True
    print(f"Using legate")
except (RuntimeError, ImportError):
    from time import perf_counter_ns

    import numpy as np
    import scipy.sparse as sparse
    import scipy.sparse as sp  # alias for backward compatibility
    import scipy.sparse.linalg as linalg
    from scipy.sparse import block_array, block_diag, csr_array, dia_array, vstack
    from scipy.sparse.linalg import spsolve

    def time():
        return perf_counter_ns() / 1e3  # Convert nanoseconds to micro seconds

    def spmv(A, x, out):
        for i in range(A.shape[0]):
            begin, end = A.indptr[i], A.indptr[i + 1]
            indices = A.indices[begin:end]
            out[i] = max(x[indices].tolist())

    use_legate = False
    print(f"Using numpy")

# Export only the modules and functions that are actually used
__all__ = [
    "np",
    "sp",
    "sparse",
    "linalg",
    "csr_array",
    "dia_array",
    "block_diag",
    "vstack",
    "block_array",
    "spsolve",
    "time",
    "use_legate",
]

try:
    from matplotlib import pyplot as plt

    use_matplotlib = True
except (RuntimeError, ImportError):
    print(f"Matplotlib not found")
    use_matplotlib = False
