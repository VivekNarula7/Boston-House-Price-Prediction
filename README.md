Yes, many of the techniques mentioned can be applied to regression problems. Here's how each technique fits into regression tasks:

### Techniques Applicable to Regression Problems

1. **Polynomial Regression**
   - **Description**: Extends linear regression by fitting a polynomial function to the data. Suitable for capturing non-linear relationships between features and the target variable.
   - **Use Case**: Effective when you suspect that the relationship between the features and the target is non-linear.

2. **Regularization Techniques**
   - **Ridge Regression (L2 Regularization)**: Adds a penalty on the magnitude of coefficients to prevent overfitting.
   - **Lasso Regression (L1 Regularization)**: Adds a penalty on the absolute value of coefficients, useful for feature selection.
   - **Elastic Net**: Combines L1 and L2 regularization, balancing between Ridge and Lasso.

3. **Decision Trees**
   - **Description**: Constructs a tree-like model of decisions. Can handle non-linear relationships and interactions.
   - **Use Case**: Suitable for capturing complex patterns in the data.

4. **Random Forest**
   - **Description**: An ensemble of decision trees that aggregates their predictions. Helps improve accuracy and reduces overfitting.
   - **Use Case**: Effective for handling complex datasets and interactions between features.

5. **Gradient Boosting Machines (GBM)**
   - **Description**: Builds models sequentially to correct the errors of previous models. Includes variants like XGBoost, LightGBM, and CatBoost.
   - **Use Case**: Provides high accuracy for complex regression problems by focusing on improving prediction errors iteratively.

6. **Support Vector Machines (SVM)**
   - **Description**: Can be used for regression (SVR) with different kernels to capture non-linear relationships.
   - **Use Case**: Suitable for complex relationships and high-dimensional data.

7. **K-Nearest Neighbors (KNN)**
   - **Description**: A non-parametric method that predicts values based on the closest training examples. 
   - **Use Case**: Simple and effective for small to medium-sized datasets or when data distribution is not known.

8. **Neural Networks**
   - **Description**: Models with multiple layers that can capture complex patterns. Variants include feedforward neural networks, CNNs, and deep learning models.
   - **Use Case**: Suitable for large datasets and capturing intricate relationships between features and the target variable.

9. **Ensemble Methods**
   - **Stacking**: Combines predictions from multiple models to enhance overall performance.
   - **Voting**: Aggregates predictions from different models to make a final prediction.
   - **Use Case**: Improves model performance by leveraging the strengths of multiple algorithms.

10. **Feature Engineering and Selection**
    - **Feature Engineering**: Creating new features or transforming existing ones to better represent the underlying data.
    - **Feature Selection**: Choosing the most relevant features to include in the model, reducing dimensionality, and improving performance.

### **Choosing the Right Technique**

- **Nature of Data**: Complex and non-linear relationships may benefit from models like Random Forest, Gradient Boosting, or Neural Networks.
- **Dataset Size**: Large datasets might benefit from advanced models like Neural Networks, while smaller datasets might be more suitable for models like Polynomial Regression or KNN.
- **Model Interpretability**: If interpretability is crucial, simpler models like Linear Regression or Decision Trees might be preferred.

Each technique has its strengths and weaknesses, and often a combination of methods (e.g., using ensemble methods or hybrid models) can yield the best results for a regression problem.
