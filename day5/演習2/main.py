import os
import random
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
import pickle
import time
import great_expectations as gx

class DataLoader:
    """データロードを行うクラス"""

    @staticmethod
    def load_titanic_data(path=None):
        """Titanicデータセットを読み込む"""
        if path:
            return pd.read_csv(path)
        else:
            local_path = "data/Titanic.csv"
            if os.path.exists(local_path):
                return pd.read_csv(local_path)
            else:
                raise FileNotFoundError("Titanicデータセットが見つかりません。")

    @staticmethod
    def preprocess_titanic_data(data):
        """Titanicデータを前処理する"""
        data = data.copy()
        columns_to_drop = ["PassengerId", "Name", "Ticket", "Cabin"]
        data.drop(columns=[col for col in columns_to_drop if col in data.columns], axis=1, inplace=True)

        if "Survived" in data.columns:
            y = data["Survived"]
            X = data.drop("Survived", axis=1)
            return X, y
        else:
            return data, None

class DataValidator:
    """データバリデーションを行うクラス"""

    @staticmethod
    def validate_titanic_data(data):
        """Titanicデータセットの検証"""
        if not isinstance(data, pd.DataFrame):
            return False, ["データはpd.DataFrameである必要があります"]

        try:
            context = gx.get_context()
            data_source = context.data_sources.add_pandas("pandas")
            data_asset = data_source.add_dataframe_asset(name="pd dataframe asset")
            batch_definition = data_asset.add_batch_definition_whole_dataframe("batch definition")
            batch = batch_definition.get_batch(batch_parameters={"dataframe": data})

            required_columns = ["Pclass", "Sex", "Age", "SibSp", "Parch", "Fare", "Embarked"]
            missing_columns = [col for col in required_columns if col not in data.columns]
            if missing_columns:
                return False, [{"success": False, "missing_columns": missing_columns}]

            expectations = [
                gx.expectations.ExpectColumnDistinctValuesToBeInSet(column="Pclass", value_set=[1, 2, 3]),
                gx.expectations.ExpectColumnDistinctValuesToBeInSet(column="Sex", value_set=["male", "female"]),
                gx.expectations.ExpectColumnValuesToBeBetween(column="Age", min_value=0, max_value=100),
                gx.expectations.ExpectColumnValuesToBeBetween(column="Fare", min_value=0, max_value=600),
                gx.expectations.ExpectColumnDistinctValuesToBeInSet(column="Embarked", value_set=["C", "Q", "S", ""]),
            ]

            results = [batch.validate(expectation) for expectation in expectations]
            is_successful = all(result.success for result in results)
            return is_successful, results

        except Exception as e:
            return False, [{"success": False, "error": str(e)}]

class ModelTester:
    """モデルテストを行うクラス"""

    @staticmethod
    def create_preprocessing_pipeline():
        """前処理パイプラインを作成"""
        numeric_features = ["Age", "Fare", "SibSp", "Parch"]
        numeric_transformer = Pipeline(steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ])

        categorical_features = ["Pclass", "Sex", "Embarked"]
        categorical_transformer = Pipeline(steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore")),
        ])

        preprocessor = ColumnTransformer(transformers=[
            ("num", numeric_transformer, numeric_features),
            ("cat", categorical_transformer, categorical_features),
        ], remainder="drop")
        return preprocessor

    @staticmethod
    def train_model(X_train, y_train):
        """モデルを学習する"""
        preprocessor = ModelTester.create_preprocessing_pipeline()
        model = Pipeline(steps=[
            ("preprocessor", preprocessor),
            ("classifier", RandomForestClassifier(n_estimators=100, random_state=42)),
        ])
        model.fit(X_train, y_train)
        return model

    @staticmethod
    def evaluate_model(model, X_test, y_test):
        """モデルを評価する"""
        start_time = time.time()
        y_pred = model.predict(X_test)
        inference_time = time.time() - start_time
        accuracy = accuracy_score(y_test, y_pred)
        return {"accuracy": accuracy, "inference_time": inference_time}

    @staticmethod
    def save_model(model, path="models/titanic_model.pkl"):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump(model, f)

    @staticmethod
    def load_model(path="models/titanic_model.pkl"):
        with open(path, "rb") as f:
            return pickle.load(f)

if __name__ == "__main__":
    # データロード
    data = DataLoader.load_titanic_data()
    X, y = DataLoader.preprocess_titanic_data(data)

    # データバリデーション
    success, results = DataValidator.validate_titanic_data(X)
    if not success:
        print("データ検証に失敗しました。")
        exit(1)

    # モデルのトレーニングと評価
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = ModelTester.train_model(X_train, y_train)
    metrics = ModelTester.evaluate_model(model, X_test, y_test)

    print(f"精度: {metrics['accuracy']:.4f}")
    print(f"推論時間: {metrics['inference_time']:.4f}秒")

    # モデル保存
    ModelTester.save_model(model)
