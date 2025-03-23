# 100-days-of-ML-by-CampusX
# ---------------------------------------------------------------------------------
This repository is used for Machine learning .. I am learning ML basics.  In 100 days of ML .I wanna cover this in 30 days. Every day note and learning output will be uploaded to this repo.
## 6th day:
### **Online Machine Learning**  
Online machine learning is a technique where a model learns from data incrementally, updating itself as new data arrives instead of training on a fixed dataset. It is useful when dealing with large-scale, continuously changing data streams.

---

### **2. When to Use Online Learning?**  
- When data arrives continuously (e.g., financial markets, IoT sensors).  
- When the dataset is too large to fit into memory.  
- When real-time predictions and adaptations are needed (e.g., recommendation systems).  
- When the data distribution changes over time (concept drift).  

---

### **3. How to Implement Online Learning?**  
#### **Using Scikit-Learn (Incremental Learning with Partial Fit)**
```python
from sklearn.linear_model import SGDClassifier
import numpy as np

# Simulated streaming data
X_batch1, y_batch1 = np.random.rand(100, 5), np.random.randint(0, 2, 100)
X_batch2, y_batch2 = np.random.rand(100, 5), np.random.randint(0, 2, 100)

# Initialize model
model = SGDClassifier(loss="log_loss")

# Train on first batch
model.partial_fit(X_batch1, y_batch1, classes=np.array([0, 1]))

# Train on next batch
model.partial_fit(X_batch2, y_batch2)
```

#### **Using River (Library for Online Learning)**
```python
from river import linear_model
from river import optim

model = linear_model.LogisticRegression(optimizer=optim.SGD(0.01))

for x, y in stream:  # Simulated streaming data
    model.learn_one(x, y)  # Learn from one instance at a time
```

---

### **4. Out-of-Core Learning**  
Out-of-core learning is a technique for training models on datasets too large to fit into memory. It loads and processes small chunks of data sequentially.

- **Tools:** `Dask`, `River`, `Vaex`, `scikit-learn` (`partial_fit`)  
- **Example using Pandas + Dask:**  
```python
import dask.dataframe as dd

df = dd.read_csv("large_dataset.csv", blocksize=1000000)
for batch in df.iter_partitions():
    model.partial_fit(batch.drop("target", axis=1), batch["target"])
```

---

### **5. Disadvantages of Online Learning**  
- **No reprocessing:** Model can't revisit old data unless explicitly stored.  
- **Sensitive to noise:** Sudden incorrect data can mislead the model.  
- **Difficult hyperparameter tuning:** No fixed dataset for validation.  
- **Risk of concept drift:** Requires monitoring for drastic changes.  

---

### **6. Batch vs Online Learning**  

| Feature            | Batch Learning                              | Online Learning                         |
|--------------------|---------------------------------|--------------------------------|
| **Data Handling**  | Uses entire dataset at once | Processes data incrementally |
| **Memory Usage**   | High (entire dataset loaded) | Low (only recent data needed) |
| **Training Speed** | Slower (depends on dataset size) | Faster (adapts on the go) |
| **Adaptability**   | Poor (fixed model) | High (updates continuously) |
| **Use Case**       | Stable environments | Streaming, dynamic environments |

Would you like a deeper dive into any of these topics? 🚀
