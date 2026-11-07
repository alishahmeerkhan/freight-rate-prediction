import argparse
import numpy as np
import pandas as pd
from joblib import dump

from xgboost import XGBRegressor
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import mean_absolute_error, root_mean_squared_error

TRAIN_FILE: str = 'data/train-test.csv'
DECEMBER_FILE: str = 'data/december-chart-inputs'

def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    drop_columns: list[str] = ['pickup', 'delivery', 'pickup_lat', 'pickup_lon', 
                    'delivery_lat', 'delivery_lon', 'date']

    df = df.drop(columns=drop_columns)
    print(f'\n\n{len(drop_columns)} columns dropped.')

    df['weight'] = df['weight'].fillna(df['weight'].mean())
    df['market_index'] = df['market_index'].fillna(df['market_index'].mean())
    print('\nMissing values have been filled.')

    encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
    encoded_array = encoder.fit_transform(df[['equipment']])
    encoded_df = pd.DataFrame(
        encoded_array, 
        columns=encoder.get_feature_names_out(), 
        index=df.index
    )
    df = pd.concat([df.drop('equipment', axis=1), encoded_df], axis=1)
    print('\nCategorical data encoding completed.')

    threshold = df['posted_rate'].quantile(0.97)
    df = df[df['posted_rate'] <= threshold]
    print('\nDropped outliers from the data.')

    return df

def train_model(df: pd.DataFrame) -> None:
    X = df.drop(columns=['load_id', 'posted_rate'])
    y = df['posted_rate']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    print('Training Scikit Model...')
    model = LinearRegression()
    model.fit(X_train, y_train)

    print('Training XGBoost Model...')
    model_xgb = XGBRegressor(n_estimators=180, learning_rate=0.05, max_depth=10, objective='reg:absoluteerror')
    model_xgb.fit(X_train, y_train)

    y_pred_sk = model.predict(X_test)
    y_pred_xgb = model_xgb.predict(X_test)
    print('\n\nEvaluation of Scikit Learn:')
    print(f'Mean Absolute Error:     {mean_absolute_error(y_test, y_pred_sk):.6f}')
    print(f'Root Mean Squared Error: {root_mean_squared_error(y_test, y_pred_sk):.6f}')

    print('\n\nEvaluation of XGBoost:')
    print(f'Mean Absolute Error:     {mean_absolute_error(y_test, y_pred_xgb):.6f}')
    print(f'Root Mean Squared Error: {root_mean_squared_error(y_test, y_pred_xgb):.6f}')

def save_model(model) -> None:
    pass

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Choose between training the model and filling the December Charts')

    parser.add_argument('--train', action='store_true', help='Train the model and save it.')
    parser.add_argument('--december', action='store_true', help='Fill the december fill for graph generation.')

    args = parser.parse_args()


    if not args.train and not args.december:
        print('Please enter the correct arguement.')
        print('> python main.py --train\t\t[To train the model]')
        print('> python main.py --december\t[To fill the december table]')        

    if args.train:
        print('MODE: Training model...')

        df: pd.DataFrame = pd.read_csv(TRAIN_FILE)

        print('Cleaning the data...')
        df = clean_data(df)

        print('Training the model...')
        train_model(df) 