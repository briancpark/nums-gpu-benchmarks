import nums
import nums.numpy as nps
from nums.models.glms import LogisticRegression
from nums.core import settings
import time
import numpy as np
import pandas as pd
settings.device_grid_name = "packed"
# settings.backend_name = "gpu"

nums.init()


# Install HIGGS Dataset using (gitignored):
# wget https://archive.ics.uci.edu/ml/machine-learning-databases/00280/HIGGS.csv.gz
# Store either in SCRATCH or /tmp directory
# gzip -d HIGGS.csv.gz

# higgs_dataset = pd.read_csv("/data/briancpark/HIGGS.csv")
# higgs_dataset = higgs_dataset.
higgs_dataset = pd.read_csv("/data/briancpark/HIGGS.csv").to_numpy()
higgs_dataset = nps.array(higgs_dataset)
y, X = higgs_dataset[:, 0].astype(int), higgs_dataset[:, 1:]


TRIALS = 15
training_times = []
inference_times = []


# n = 100000

# X = nps.random.rand(n, 1000)
# # y = nps.random.randint(0, 1, (n))
# # y = y.astype(float)
# y = nps.random.rand(n)
print(X.shape, X.block_shape, X.grid_shape)
print(X.shape, y.shape)

for i in range(TRIALS):
    model = LogisticRegression(solver="newton")

    X.touch()
    y.touch()

    

    begin = time.time()
    model.fit(X, y)
    X.touch()
    y.touch()
    end = time.time()

    training_time = end - begin

    begin = time.time()
    y_pred = model.predict(X)
    y_pred.touch()
    end = time.time()

    inference_time = end - begin

    training_times.append(training_time)
    inference_times.append(inference_time)

    print("Training time:%.4f s, Inference time:%.4f s" % (training_time, inference_time))
print("accuracy", (nps.sum(y == y_pred) / X.shape[0]).get())


training_times = training_times[5:]
inference_times = inference_times[5:]

print(np.mean(training_times), np.mean(inference_times))