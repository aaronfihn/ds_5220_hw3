import pandas as pd
from pathlib import Path


def get_data(dataset: int, /, test: bool=False) -> pd.DataFrame:
    # Get the data directory pathname
    data_dir_pname = Path(Path(__file__).parents[1], 'data')

    # Get the csv path
    test_or_train = ('Test' if test else 'Train')
    fname = f'PS3_{dataset}_{test_or_train}.csv'
    pname = Path(data_dir_pname, fname)

    # Read and return the dataframe
    return pd.read_csv(pname)


def main():
    df = get_data(2, test=False)
    print(df)


if __name__ == '__main__':
    main()
