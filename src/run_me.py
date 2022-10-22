import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.linear_model import Ridge
from sklearn.linear_model import LogisticRegression


def get_data(dataset: int, /, test: bool = False) -> pd.DataFrame:
    # Get the data directory pathname
    data_dir_pname = Path(Path(__file__).parents[1], 'data')

    # Get the csv path
    test_or_train = ('Test' if test else 'Train')
    fname = f'PS3_{dataset}_{test_or_train}.csv'
    pname = Path(data_dir_pname, fname)

    # Read and return the dataframe
    return pd.read_csv(pname)


def train_ridge_closed_form(alpha=1.0) -> np.ndarray:
    """Fit the PS3-1 training dataset to a ridge regression for a specified regularization constant.

    :param alpha: the regularization constant
    :return: A (2, 1) ndarray of fitted regression coefficients
    """
    # get the data from the csv file
    df_train = get_data(1, test=False)

    # Create matrices for observations of both predictors and responses
    x_mat = np.ones([df_train.shape[0], df_train.shape[1]])  # 50 rows, 2 columns. (One column is for the intercept.)
    y_mat = np.ones([df_train.shape[0], 1])  # 50 rows, 1 column.
    for r in range(x_mat.shape[0]):
        x_mat[r][1] = df_train.iat[r, 0]  # Populate x_mat
        y_mat[r][0] = df_train.iat[r, 1]  # Populate y_mat

    # Use the closed-form solution found at https://mlweb.loria.fr/book/en/ridgeregression.html
    x_mat_transpose = x_mat.transpose()
    theta = np.linalg.inv(x_mat_transpose @ x_mat + alpha * np.identity(x_mat.shape[1])) @ x_mat_transpose @ y_mat
    return theta


def train_ridge_sgd(alpha=1.0) -> np.ndarray:
    """Fit the PS3-1 training dataset to a ridge regression using stochastic gradient descent."""
    # get the data from the csv file
    df_train = get_data(1, test=False)
    train_X = df_train[['x']]
    train_y = df_train[['y']]

    # fit a model
    ridge_model = Ridge(alpha=alpha, tol=1e-9, solver='cholesky', random_state=42)
    ridge_model.fit(train_X, train_y)

    # convert intercept and coefficients to an ndarray and return it
    theta = np.asarray([[ridge_model.intercept_[0]], [ridge_model.coef_[0][0]]])
    return theta


def train_logistic_regression(x: pd.DataFrame, y: pd.DataFrame) -> np.ndarray:
    """Fit data to a logistic regression model.

    :param x: n-by-p matrix for predictors
    :param y: n-by-1 matrix for responses
    :return: sklearn LogisticRegression modle
    """
    clf = LogisticRegression(random_state=42).fit(x, y)
    # fix return value



def prob_1a() -> None:
    """The solution to problem 1a."""
    s = '----------\n' \
        'Problem 1a\n' \
        'The closed-form solution for a specified regularization \n' \
        'constant is train_ridge_closed_form().\n' \
        'Example coefficients for alpha=1.0, closed-form:\n' \
        + str(train_ridge_closed_form(alpha=1.0)) + '\n' \
        '\n'\
        'The stochastic gradient descent solution is train_ridge_sgd().\n'\
        'Example coefficients for alpha=1.0, SGD:\n'\
        + str(train_ridge_sgd(alpha=1.0))
    #  UNFINISHED UNFINISHED UNFINISHED, CAN WE DO SGD IN SKLEARN?
    print(s)


def prob_2a():



def main():
    prob_1a()


def testing():
    prob_1a()


if __name__ == '__main__':
    testing()
