import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import sklearn.metrics
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


def train_ridge_sgd(alpha=1.0, minibatch=10, n_epochs=1000) -> np.ndarray:
    """Fit the PS3-1 training dataset to a ridge regression using stochastic gradient descent."""
    # get the data from the csv file
    df_train = get_data(1, test=False)
    train_X = df_train[['x']]
    train_y = df_train[['y']]

    # fit a model
    clf = Ridge(alpha=alpha, tol=1e-9, solver='cholesky')
    clf.fit(df_train[['x']], df_train[['y']])

    for epoch in range(n_epochs):
        df_train = df_train.sample(frac=1, random_state=42)




def train_logistic_regression(x: pd.DataFrame, y: pd.DataFrame) -> LogisticRegression:
    """Fit data to a logistic regression model.

    :param x: n-by-p matrix for predictors.
    :param y: n-by-1 matrix for responses
    :return: sklearn LogisticRegression model
    """
    clf = LogisticRegression(penalty='none').fit(x, y)
    return clf


def savefig(fname: str) -> None:
    save_path = Path(Path(__file__).parents[1], 'figures', fname)
    print(f'Saving {save_path}')
    plt.savefig(save_path)


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
    s = '' \
      '----------\n' \
      'Problem 2a\n' \
      'The function train_logistic_regression() takes two pandas DataFrames, one for predictors \n' \
      'and one for response, and uses them to train a logistic regression model. It outputs a \n' \
      'LogisticRegression class from the sklearn library. (I convert this into a weight vector in \n' \
      'problem 2b manually, as it is much easier to evaluate the model against the test data with \n' \
      'the output I selected.)'
    print(s)


def prob_2b():
    print('----------')
    print('Problem 2b')

    # get the PS3-2 dataset
    df_train = get_data(2, test=False)

    # convert the responses and predictors to the format sklearn expects and get the fitted model
    x_train = df_train[['x1', 'x2']].to_numpy()
    y_train = df_train[['y']].to_numpy().ravel()
    clf = train_logistic_regression(x_train, y_train)

    # Find the weight vector for the PS3-2 training set
    intercept_ndarray = clf.intercept_.reshape(1, -1)
    coef_ndarray = clf.coef_.reshape(1, -1)
    weight_vector = np.concatenate((intercept_ndarray, coef_ndarray), axis=1)
    print(f'Weight vector for PS3-2 training data:\n{weight_vector}')

    # Evaluate the model on the PS3-2 test set and save the confusion matrix
    df_test = get_data(2, test=True)
    x_test = df_test[['x1', 'x2']].to_numpy()
    y_test = df_test[['y']].to_numpy().ravel()
    sklearn.metrics.ConfusionMatrixDisplay.from_estimator(clf, x_test, y_test)
    plt.title('Confusion Matrix for PS3-2 Test Set')
    savefig('ps3_2_test_confusion.png')
    plt.clf()

    # Plot the PS3-2 test data and decision boundary
    ax = sns.scatterplot(data=df_test, x='x1', y='x2', hue='y')
    theta0 = weight_vector[0, 0]
    theta1 = weight_vector[0, 1]
    theta2 = weight_vector[0, 2]
    x_decision = np.linspace(df_test['x1'].min(), df_test['x1'].max(), 50)
    y_decision = np.asarray([-(theta0 + theta1*x)/theta2 for x in x_decision])
    ax.plot(x_decision, y_decision, 'r')
    plt.title('PS3-2 test data with decision boundary')
    savefig('ps3_2_test_data_plot.png')
    plt.clf()


def prob_2c():
    print('----------')
    print('Problem 2c')

    # get the PS3-3 dataset
    df_train = get_data(3, test=False)

    # convert the responses and predictors to the format sklearn expects and get the fitted model
    x_train = df_train[['x1', 'x2']].to_numpy()
    y_train = df_train[['y']].to_numpy().ravel()
    clf = train_logistic_regression(x_train, y_train)

    # Find the weight vector for the PS3-3 training set
    intercept_ndarray = clf.intercept_.reshape(1, -1)
    coef_ndarray = clf.coef_.reshape(1, -1)
    weight_vector = np.concatenate((intercept_ndarray, coef_ndarray), axis=1)
    print(f'Weight vector for PS3-3 training data:\n{weight_vector}')

    # Evaluate the model on the PS3-3 test set and save the confusion matrix
    df_test = get_data(3, test=True)
    x_test = df_test[['x1', 'x2']].to_numpy()
    y_test = df_test[['y']].to_numpy().ravel()
    sklearn.metrics.ConfusionMatrixDisplay.from_estimator(clf, x_test, y_test)
    plt.title('Confusion Matrix for PS3-3 Test Set')
    savefig('ps3_3_test_confusion.png')
    plt.clf()

    # Plot the PS3-3 test data and decision boundary
    ax = sns.scatterplot(data=df_test, x='x1', y='x2', hue='y')
    theta0 = weight_vector[0, 0]
    theta1 = weight_vector[0, 1]
    theta2 = weight_vector[0, 2]
    x_decision = np.linspace(df_test['x1'].min(), df_test['x1'].max(), 50)
    y_decision = np.asarray([-(theta0 + theta1 * x) / theta2 for x in x_decision])
    ax.plot(x_decision, y_decision, 'r')
    plt.title('PS3-3 test data with decision boundary')
    savefig('ps3_3_test_data_plot.png')
    plt.clf()


def main():
    prob_1a()


def testing():
    train_ridge_sgd()


if __name__ == '__main__':
    testing()
