import numpy as np

aa = np.array([1, 2])
bb = np.array([[1, 2], [3, 4], [5, 6]])
print(aa - bb)
print(np.linalg.norm(aa - bb, 2, axis=1))
print(np.where(np.linalg.norm(aa - bb, 2, axis=1) < 1e-10))

a = []
import matplotlib.pyplot as plt

a.append([[1, 2], [3, 4]])
a.append([[5, 6], [7, 8]])
plt.plot([1, 2], [3, 4], "bo", linestyle="-")
plt.show()
