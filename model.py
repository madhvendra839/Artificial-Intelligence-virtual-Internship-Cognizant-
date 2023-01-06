import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
from sklearn.preprocessing import StandardScaler

def load_data(path: str = "/path/to/csv/"):

    df = pd.read_csv(f"{path}")
    df.drop(columns=["Unnamed: 0"], inplace=True, errors='ignore')
    return df

def create_target_and_predictors(
    data: pd.DataFrame = None, 
    target: str = "estimated_stock_pct"
):
    
    if target not in data.columns:
        raise Exception(f"Target: {target} is not present in the data")
    
    X = data.drop(columns=[target])
    y = data[target]
    return X, y

def train_algorithm_with_cross_validation(
    X: pd.DataFrame = None, 
    y: pd.Series = None
):
    
    accuracy = []

    for fold in range(0, K):

        model = RandomForestRegressor()
        scaler = StandardScaler()

        X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=SPLIT, random_state=42)

        scaler.fit(X_train)
        X_train = scaler.transform(X_train)
        X_test = scaler.transform(X_test)

        trained_model = model.fit(X_train, y_train)

        y_pred = trained_model.predict(X_test)

        mae = mean_absolute_error(y_true=y_test, y_pred=y_pred)
        accuracy.append(mae)
        print(f"Fold {fold + 1}: MAE = {mae:.3f}")

    print(f"Average MAE: {(sum(accuracy) / len(accuracy)):.2f}")


#main
df = load_data('new_data')
X,y = create_target_and_predictors(df)
train_algorithm_with_cross_validation(X,y)
