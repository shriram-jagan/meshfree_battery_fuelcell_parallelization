How to run:
- Create a conda environment with the following packages: numpy, scipy, matplotlib, tqdm, numba
- To run IM-RKPM-NodeRemoval_vectorization:
   - update the number of timesteps (`nt`) and the duration of simulation (`t`) in `main.py`
   - Do `python main.py`
- To run fuel_cell_3d:
   - Configuration is done in `config.py` and imports are done in `common.py`.
     To change the input parameters like domain, damage model, etc., see `config.py`.
   - Use the `run_legate.sh` script to run using legate. Note that for unsupported APIs,
     we still convert fallback to NumPy and SciPy APIs and this is set by the parameter
     `USE_NUMPY_EQUIVALENTS` in `config.py`. This must be set to True for `legate` runs
     and could be True or False for NumPy/SciPy runs.
   - To run using Numpy and SciPy, uncomment the following lines from `common.py`
       -> print("Forcing use of NumPy backend by raising a false error")
       -> raise ImportError  # Force use of regular NumPy instead of legate
     and then, set `USE_NUMPY_EQUIVALENTS` to False, and then run,
        `python ./main.py`

