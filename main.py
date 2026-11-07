import argparse
import numpy as np
import pandas as pd
from joblib import dump, load

from xgboost import XGBRegressor
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, root_mean_squared_error

TRAIN_FILE: str = 'data/train-test.csv'
VALIDATION_FILE: str = 'data/validation.csv'
VALIDATION_TEMPLATE: str = 'data/validation-predictions-template.csv'
DECEMBER_FILE: str = 'data/december-chart-inputs.csv'
MODEL_PATH: str = 'model/freight_rate_prediction.pkl'

def clean_data(df: pd.DataFrame, is_training: bool=True) -> pd.DataFrame:
    drop_columns: list[str] = ['pickup', 'delivery', 'pickup_lat', 'pickup_lon', 
                    'delivery_lat', 'delivery_lon', 'date']

    df = df.drop(columns=drop_columns)
    print(f'\n\n{len(drop_columns)} columns dropped.')

    df['weight'] = df['weight'].fillna(df['weight'].mean())
    df['market_index'] = df['market_index'].fillna(df['market_index'].mean())
    print('\nMissing values have been filled.')

    if is_training:
        encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
        encoded_array = encoder.fit_transform(df[['equipment']])
        encoded_df = pd.DataFrame(
            encoded_array, 
            columns=encoder.get_feature_names_out(), 
            index=df.index
        )
        df = pd.concat([df.drop('equipment', axis=1), encoded_df], axis=1)
        dump(encoder, 'model/encoder.pkl')
        print('\nCategorical data encoding completed and encoder saved at model/encoder.pkl.')
    else:
        encoder = load('model/encoder.pkl')
        encoded_array = encoder.transform(df[['equipment']])

        encoded_df = pd.DataFrame(
            encoded_array,
            columns=encoder.get_feature_names_out(),
            index=df.index
        )
        df = pd.concat([df.drop('equipment', axis=1), encoded_df], axis=1)
        print('\nCategorical data encoding applied from saved encoder.')

    if 'posted_rate' in df.columns:
        threshold = df['posted_rate'].quantile(0.97)
        df = df[df['posted_rate'] <= threshold]
        print('\nDropped outliers from the data.')

    return df

def train_model(df: pd.DataFrame) -> None:
    X = df.drop(columns=['load_id', 'posted_rate'])
    y = df['posted_rate']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    print('Training XGBoost Model...')
    model_xgb = XGBRegressor(n_estimators=180, learning_rate=0.05, max_depth=10, objective='reg:absoluteerror')
    model_xgb.fit(X_train, y_train)

    y_pred_xgb = model_xgb.predict(X_test)

    print('\n\nEvaluation of XGBoost:')
    print(f'Mean Absolute Error:     {mean_absolute_error(y_test, y_pred_xgb):.6f}')
    print(f'Root Mean Squared Error: {root_mean_squared_error(y_test, y_pred_xgb):.6f}')

    save_model(model_xgb)

def save_model(model) -> None:
    dump(model, MODEL_PATH)
    print(f'\n\nModel successfully saved at {MODEL_PATH}.')

def validation() -> None:
    df_validation: pd.DataFrame = pd.read_csv(VALIDATION_FILE)
    df_validation_template: pd.DataFrame = pd.read_csv(VALIDATION_TEMPLATE)

    print('Cleaning validation data...')
    df_validation = clean_data(df_validation, is_training=False)
    df_validation_copy = df_validation.drop(columns=['load_id'])

    print('Model Successfully loaded.')
    model = load(MODEL_PATH)

    predictions = model.predict(df_validation_copy)
    print('Prediction completed.')

    prediction_df = pd.DataFrame({
        'load_id': df_validation['load_id'],
        'predicted_rate': predictions
    })

    df_validation_template = df_validation_template.drop(columns=['predicted_rate'], errors='ignore')
    final_df = df_validation_template.merge(prediction_df, on='load_id', how='left')
    final_df.to_csv('validation_predictions.csv', index=False)
    print('Predicted file saved to validation_predictions.csv.')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Choose between training the model and filling the December Charts')

    parser.add_argument('--train', action='store_true', help='Train the model and save it.')
    parser.add_argument('--validation', action='store_true', help='Fill the validation csv')
    parser.add_argument('--december', action='store_true', help='Fill the december fill for graph generation.')

    args = parser.parse_args()


    if not args.train and not args.validation and not args.december:
        print('Please enter the correct arguement.')
        print('> python main.py --train\t[To train the model]')
        print('> python main.py --validation\t[To run validation predictions]')
        print('> python main.py --december\t[To fill the december table]')        

    if args.train:
        print('MODE: Training model...')

        df: pd.DataFrame = pd.read_csv(TRAIN_FILE)

        print('Cleaning the data...')
        df = clean_data(df, is_training=True)

        print('Training the model...')
        train_model(df) 

    elif args.validation:
        validation()

    elif args.december:
        pass