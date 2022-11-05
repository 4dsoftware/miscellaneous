from sklearn.datasets import *
import time

st = time.time()
for _ in range(10):
    housing = fetch_california_housing()
    clear_data_home()
print(time.time() - st)