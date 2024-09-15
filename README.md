# Boston Housing Price Prediction using Linear Regression

## Project Overview
This repository contains a project focused on predicting house prices in Boston using the **Boston Housing Dataset**. The project implements a **Linear Regression** model and evaluates its performance using standard metrics.

The goal of the project is to accurately predict house prices based on various features such as crime rate, number of rooms, and proximity to highways. The project involves data preprocessing, model training, and evaluation using Python libraries like **pandas**, **scikit-learn**, and **matplotlib**.

## Dataset
The dataset used in this project is the [Boston Housing Dataset](https://www.kaggle.com/c/boston-housing). It contains the following files:
- `train.csv`: Training data with 333 rows and 15 columns.
- `test.csv`: Testing data with 173 rows and 14 columns (no target variable).
- `submission.csv`: The expected format for submitting predicted values.

### Features
Key features of the dataset include:
- `crim`: Per capita crime rate.
- `nox`: Nitrogen oxides concentration.
- `rm`: Average number of rooms per dwelling.
- `medv`: Median value of owner-occupied homes (target variable).

For a detailed description, refer to the [Boston Housing Dataset on Kaggle](https://www.kaggle.com/c/boston-housing).

## Project Structure
The project is structured as follows:
- **Data Preprocessing**: Handling missing values, feature selection, and dataset splitting using `train_test_split`.
- **Model Implementation**: Using the `LinearRegression` model from `scikit-learn` to train and test the model.
- **Evaluation**: Calculating metrics like MSE, RMSE, MAE, and R² score to evaluate model performance.

## Installation
To run this project, you need Python 3.x and the following libraries:
```bash
pip install pandas scikit-learn matplotlib numpy
```

## Usage
1. **Clone the repository**:
   ```bash
   git clone https://github.com/yourusername/boston-housing-price-prediction.git
   cd boston-housing-price-prediction
   ```
2. **Run the project**:
   The main script is `boston_housing.py`, which handles data loading, preprocessing, model training, and evaluation.
   ```bash
   python boston_housing.py
   ```

## Model Training and Evaluation
The project uses **Linear Regression** for predicting house prices. The data is split into training and testing sets using a 67:33 ratio. Key evaluation metrics include:

- **Mean Squared Error (MSE)**
- **Root Mean Squared Error (RMSE)**
- **Mean Absolute Error (MAE)**
- **R² Score**

The evaluation results show that the model achieves a decent R² score and acceptable error metrics, considering the dataset's size.

## Results
| Metric             | Result  |
|--------------------|---------|
| R² Score           | 0.7451  |
| MSE                | 20.6271 |
| RMSE               | 4.5417  |
| MAE                | 3.4100  |

## Improvements
Due to the limitations of linear regression and the small dataset, the model's performance can be improved by:
- Increasing the sample size.
- Using more advanced models like **Ridge Regression**, **Decision Trees**, or **Neural Networks**.

## Conclusion
This project demonstrates the application of **Linear Regression** to predict house prices in Boston. While the model performs well, further improvements are possible by increasing the dataset size or using more complex algorithms.

## License
This project is licensed under the MIT License. See the `LICENSE` file for more details.

## Acknowledgments
- [Kaggle](https://www.kaggle.com/c/boston-housing) for providing the Boston Housing Dataset.
- Scikit-learn documentation for model implementation guides.

---

This readme provides an overview of your project based on the report details you shared. Let me know if you'd like to add or modify anything!
