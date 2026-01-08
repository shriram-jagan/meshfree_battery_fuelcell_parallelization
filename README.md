How to run:
- Create a conda environment with the following packages: numpy, scipy, matplotlib, tqdm, numba
- To run IM-RKPM-NodeRemoval_vectorization:
   - update the number of timesteps (`nt`) and the duration of simulation (`t`) in `main.py`
   - Do `python main.py`
- To run fuel_cell_3d:
   - Use the `run_legate.sh` script to run using legate.
   - The imports are performed in `common.py`, so if you prefer to prioritize importing 
     NumPy and SciPy over cuPyNumeric and Legate Sparse, 
     uncomment the first few lines in the try-except block in `common.py`

