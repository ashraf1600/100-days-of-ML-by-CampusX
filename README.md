# 100-days-of-ML-by-CampusX
# -------------------------------------------------------------------------
This repository is used for Machine learning .. I am learning ML basics.  In 100 days of ML .I wanna cover this in 30 days. Every day note and learning output will be uploaded to this repo.
## 5th day:
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
# 6 th day:

### **Instance-Based Learning (Lazy Learning)**
Instance-based learning is a machine learning approach where the model **memorizes** training data and makes predictions by comparing new instances to stored examples rather than learning a general function.

#### **Key Features:**
- **No explicit model training** – Just stores examples and uses them for predictions.
- **Uses similarity measures** – Compares new data with stored instances using distance metrics (e.g., Euclidean distance).
- **Fast training but slow prediction** – Since the model does not generalize, it must search through stored data for every new prediction.

#### **Examples of Instance-Based Learning:**
- **K-Nearest Neighbors (KNN)** – Finds the k closest examples to classify new data.
- **Radial Basis Function Networks (RBFN)** – Uses stored examples as centers in a neural network.
- **Case-Based Reasoning (CBR)** – Solves problems by comparing them to past similar cases.

#### **Pros & Cons of Instance-Based Learning**
✔ **Adapts quickly to new data**  
✔ **Works well for non-linear problems**  
❌ **Requires large storage space**  
❌ **Slow prediction time** (since it searches all stored examples)  
❌ **Sensitive to noise and irrelevant features**  

---

### **Model-Based Learning (Eager Learning)**
Model-based learning is a machine learning approach where the model **learns a function** from training data and generalizes it to make predictions.

#### **Key Features:**
- **Learns a mathematical function** – Instead of storing examples, it builds a predictive model.
- **Training is computationally expensive** – Requires optimization and tuning.
- **Fast predictions** – Since the model is already trained, new predictions are quick.

#### **Examples of Model-Based Learning:**
- **Linear Regression** – Finds a straight-line relationship between input and output.
- **Decision Trees** – Learns a tree-like structure to classify data.
- **Neural Networks** – Uses layers of artificial neurons to learn complex patterns.
- **Support Vector Machines (SVM)** – Finds the best decision boundary to separate classes.

#### **Pros & Cons of Model-Based Learning**
✔ **Efficient for large datasets**  
✔ **Fast prediction speed**  
✔ **Less memory usage**  
❌ **Requires retraining for new data**  
❌ **May not capture complex patterns well (if the model is too simple)**  

---

### **Comparison: Instance-Based vs. Model-Based Learning**

| Feature                | **Instance-Based Learning** | **Model-Based Learning** |
|------------------------|--------------------------|--------------------------|
| **Training Time**      | Fast (just stores data)  | Slow (builds a model) |
| **Prediction Time**    | Slow (searches instances) | Fast (applies model) |
| **Memory Requirement** | High (stores data) | Low (stores only model parameters) |
| **Adaptability**       | Easy (new data can be added instantly) | Hard (requires retraining) |
| **Robustness to Noise** | Low (outliers affect results) | High (regularization can reduce noise effects) |
| **Best For**           | Small, dynamic datasets | Large, structured datasets |
## 7th day:
### **Different Challenges in Machine Learning**  

Machine Learning (ML) comes with various challenges that affect model performance, efficiency, and deployment. Below are key challenges categorized into different areas:

---

## **1. Data-Related Challenges**  

### **❌ Data Quality Issues**  
- **Noisy Data** – Data with irrelevant or misleading information.  
- **Missing Values** – Incomplete data can lead to biased models.  
- **Outliers** – Extreme values can distort the learning process.  

### **❌ Data Quantity Issues**  
- **Too Little Data** – Leads to overfitting and poor generalization.  
- **Imbalanced Data** – Some classes appear much more than others (e.g., fraud detection, medical diagnosis).  

### **❌ Feature Engineering & Selection**  
- Choosing the right features is crucial but challenging.  
- Poor feature selection can lead to underperforming models.  

---

## **2. Model-Related Challenges**  

### **❌ Overfitting & Underfitting**  
- **Overfitting** – The model learns too much from training data, losing generalization ability.  
- **Underfitting** – The model is too simple and fails to learn the underlying patterns.  

### **❌ Hyperparameter Tuning**  
- Finding the best parameters (e.g., learning rate, number of layers in neural networks) is complex and time-consuming.  

### **❌ Bias-Variance Tradeoff**  
- **High Bias (Underfitting)** – The model is too simple to capture patterns.  
- **High Variance (Overfitting)** – The model is too complex and captures noise.  

---

## **3. Computational & Resource Challenges**  

### **❌ High Computational Cost**  
- Training deep learning models requires powerful GPUs and large datasets.  
- Real-time applications demand fast inference times.  

### **❌ Scalability Issues**  
- Scaling models to handle massive datasets is difficult.  
- Distributed computing solutions (like Apache Spark) are often needed.  

---

## **4. Deployment & Maintenance Challenges**  

### **❌ Model Drift**  
- The world changes, and so does data. A model trained on old data may become obsolete (e.g., fraud detection models need frequent updates).  

### **❌ Interpretability & Explainability**  
- Black-box models (e.g., deep learning) are difficult to explain.  
- Some industries (healthcare, finance) require models to be interpretable.  

### **❌ Ethical & Bias Issues**  
- Bias in training data can lead to unfair decisions.  
- Example: AI hiring tools that discriminate based on gender or race.  

---

## **5. Real-World Constraints**  

### **❌ Privacy & Security**  
- Sensitive data (health, financial) requires strict security measures.  
- Federated Learning is a solution to train models without sharing data.  

### **❌ Data Labeling & Annotation**  
- Supervised learning models require labeled data, which is expensive and time-consuming to collect.  

---

### **Conclusion**  
Machine learning is powerful but challenging. Overcoming these challenges requires:  
✔ High-quality, diverse data.  
✔ Proper model selection and tuning.  
✔ Scalable computing solutions.  
✔ Ethical considerations in deployment.  
### **Challenges Related to Data in Machine Learning**  

Data is the backbone of machine learning. Poor data quality can lead to incorrect predictions and unreliable models. Here are some key data-related challenges:  

---

### **1. Insufficient Data**  
- **Problem:** Many machine learning models require large amounts of data to learn patterns effectively.  
- **Impact:** Models trained on small datasets may overfit or fail to generalize to new data.  
- **Solution:**  
  - Use data augmentation techniques (e.g., synthetic data generation).  
  - Collect more data through web scraping, surveys, or crowdsourcing.  
  - Transfer learning (use pre-trained models when data is limited).  

---

### **2. Non-Representative Data**  
- **Problem:** The dataset does not reflect real-world distributions.  
- **Example:** Training a facial recognition model on mostly Caucasian faces but deploying it in a diverse population.  
- **Impact:** The model will perform well only on certain groups, leading to biased and unfair predictions.  
- **Solution:**  
  - Ensure diverse data collection across all relevant groups.  
  - Use stratified sampling to maintain proper representation.  
  - Apply bias correction techniques.  

---

### **3. Imbalanced Data**  
- **Problem:** One class is significantly overrepresented compared to others.  
- **Example:** In fraud detection, fraudulent transactions are rare compared to normal ones.  
- **Impact:** The model may predict only the majority class and ignore the minority class.  
- **Solution:**  
  - Use oversampling (duplicate minority class instances) or undersampling (reduce majority class instances).  
  - Use techniques like SMOTE (Synthetic Minority Over-sampling Technique).  
  - Adjust model evaluation metrics (e.g., F1-score instead of accuracy).  

---

### **4. Noisy Data**  
- **Problem:** Data contains irrelevant, misleading, or incorrect information.  
- **Example:** Sensor data with random fluctuations, incorrect labels in a dataset.  
- **Impact:** The model learns incorrect patterns, reducing performance.  
- **Solution:**  
  - Use noise reduction techniques (e.g., filtering, smoothing).  
  - Perform careful data cleaning and validation.  
  - Use robust models that handle noise effectively (e.g., ensemble methods).  

---

### **5. Missing Data**  
- **Problem:** Some values in the dataset are missing.  
- **Example:** Missing customer age in a sales dataset.  
- **Impact:**  
  - Models may fail to train properly.  
  - Missing values can lead to biased results.  
- **Solution:**  
  - Use imputation (fill missing values using mean, median, mode, or predictive models).  
  - Remove instances with excessive missing values.  
  - Use models that handle missing data well (e.g., decision trees).  

---

### **6. High-Dimensional Data (Curse of Dimensionality)**  
- **Problem:** Too many features (columns) can make the model inefficient.  
- **Example:** Text or image data often has thousands of features.  
- **Impact:**  
  - Overfitting (the model memorizes noise instead of learning useful patterns).  
  - Increased computational complexity.  
- **Solution:**  
  - Use feature selection (choose only important features).  
  - Apply dimensionality reduction techniques (e.g., PCA, t-SNE, Autoencoders).  

---

### **7. Duplicate & Redundant Data**  
- **Problem:** The dataset contains duplicate entries or highly correlated features.  
- **Impact:**  
  - Increases dataset size unnecessarily.  
  - Can mislead the learning process by giving some features extra importance.  
- **Solution:**  
  - Remove duplicate rows and redundant features.  
  - Use correlation analysis to drop highly correlated variables.  

---

### **8. Data Privacy & Security Concerns**  
- **Problem:** Sensitive data (e.g., medical records, financial transactions) may not be available for training due to legal or ethical restrictions.  
- **Impact:**  
  - Limited data access slows down ML development.  
  - Unauthorized access can lead to data breaches.  
- **Solution:**  
  - Use privacy-preserving techniques (e.g., differential privacy, federated learning).  
  - Anonymize or encrypt sensitive data.  

---

### **Conclusion**  
✔ High-quality data is crucial for building robust ML models.  
✔ Cleaning and preprocessing data effectively can significantly improve model performance.  
✔ Combining domain expertise with data science techniques ensures better generalization.  
