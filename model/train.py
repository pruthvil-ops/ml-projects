import argparse
import glob
import os
import time
import warnings
from dataclasses import dataclass

import joblib
import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, RobustScaler
from xgboost import XGBClassifier

warnings.filterwarnings("ignore")


@dataclass
class TrainingResult:
    best_model_name: str
    best_f1: float
    best_accuracy: float
    model_path: str


class CICIDSTrainer:
    def __init__(self) -> None:
        self.target_column = None
        self.original_target_col = None

    def detect_target_column(self, df: pd.DataFrame) -> tuple[str, str]:
        possible_targets = {"label", "class", "attack"}
        for col in df.columns:
            if col.strip().lower() in possible_targets:
                return col.strip(), col

        last_col = df.columns[-1]
        return last_col.strip(), last_col

    @staticmethod
    def clean_column_names(df: pd.DataFrame) -> pd.DataFrame:
        cleaned = df.copy()
        cleaned.columns = [str(col).strip() for col in cleaned.columns]
        return cleaned

    @staticmethod
    def get_dataset_info(data_dir: str) -> tuple[list[str], list[dict], float]:
        csv_files = sorted(glob.glob(os.path.join(data_dir, "*.csv")))
        total_size = 0.0
        file_info: list[dict] = []
        for file_path in csv_files:
            size_mb = os.path.getsize(file_path) / (1024 * 1024)
            total_size += size_mb
            file_info.append({"file": os.path.basename(file_path), "size_mb": round(size_mb, 2)})
        return csv_files, file_info, total_size

    def load_and_sample_data(
        self,
        data_dir: str,
        sample_frac: float = 0.15,
        max_rows_per_file: int = 120_000,
        max_rows_per_class_per_file: int = 12_000,
    ) -> pd.DataFrame:
        csv_files = sorted(glob.glob(os.path.join(data_dir, "*.csv")))
        if not csv_files:
            raise ValueError(f"No CSV files found in {data_dir}")

        sampled_dfs: list[pd.DataFrame] = []
        for file_path in csv_files:
            print(f"Loading {os.path.basename(file_path)}")
            df = pd.read_csv(file_path, encoding="latin1", nrows=max_rows_per_file, low_memory=False)
            df = self.clean_column_names(df)

            if self.target_column is None:
                self.target_column, self.original_target_col = self.detect_target_column(df)
                print(f"Detected target column: {self.original_target_col}")

            if self.target_column not in df.columns:
                continue

            df = df.dropna(subset=[self.target_column])
            df = df.replace([np.inf, -np.inf], np.nan)
            df = df.dropna(axis=1, how="all")

            if df.empty:
                continue

            # Class-balanced sampling to reduce bias
            file_samples: list[pd.DataFrame] = []
            for _, group in df.groupby(self.target_column):
                target_n = max(1, int(len(group) * sample_frac))
                target_n = min(target_n, max_rows_per_class_per_file)
                sampled_group = group.sample(n=target_n, random_state=42) if len(group) > target_n else group
                file_samples.append(sampled_group)

            sampled = pd.concat(file_samples, ignore_index=True)
            sampled_dfs.append(sampled)
            print(f"Sampled {len(sampled)} rows from {os.path.basename(file_path)}")

        if not sampled_dfs:
            raise ValueError("No data could be loaded from dataset files")

        combined = pd.concat(sampled_dfs, ignore_index=True)
        print(f"Combined sampled dataset shape: {combined.shape}")
        return combined

    def preprocess_data(self, df: pd.DataFrame):
        target_col = self.target_column
        if target_col is None or target_col not in df.columns:
            raise ValueError("Target column not found in dataset")

        y = df[target_col]
        X = df.drop(columns=[target_col])
        X = self.clean_column_names(X)

        id_columns = [
            "Flow ID",
            "Source IP",
            "Destination IP",
            "Timestamp",
            "FlowID",
            "SourceIP",
            "DestinationIP",
        ]
        drop_cols = [col for col in id_columns if col in X.columns]
        if drop_cols:
            X = X.drop(columns=drop_cols)

        numeric_cols = X.select_dtypes(include=[np.number]).columns.tolist()
        if not numeric_cols:
            raise ValueError("No numeric feature columns available for training")

        X_numeric = X[numeric_cols]

        imputer = SimpleImputer(strategy="median")
        X_imputed = imputer.fit_transform(X_numeric)

        # Robust scaling is less sensitive to outliers than StandardScaler
        scaler = RobustScaler()
        X_scaled = scaler.fit_transform(X_imputed)

        le = LabelEncoder()
        y_encoded = le.fit_transform(y.astype(str))

        return X_scaled, y_encoded, scaler, imputer, le, numeric_cols

    @staticmethod
    def _class_weight_map(y_train: np.ndarray) -> dict[int, float]:
        unique, counts = np.unique(y_train, return_counts=True)
        total = counts.sum()
        n_classes = len(unique)
        return {int(cls): float(total / (n_classes * count)) for cls, count in zip(unique, counts)}

    def train_and_evaluate_models(self, X_train, X_test, y_train, y_test):
        class_weights = self._class_weight_map(y_train)

        models = {
            "LightGBM": LGBMClassifier(
                n_estimators=300,
                learning_rate=0.08,
                num_leaves=63,
                subsample=0.9,
                colsample_bytree=0.9,
                random_state=42,
                n_jobs=-1,
                objective="multiclass",
                class_weight=class_weights,
                verbose=-1,
            ),
            "XGBoost": XGBClassifier(
                n_estimators=250,
                learning_rate=0.08,
                max_depth=8,
                subsample=0.9,
                colsample_bytree=0.9,
                random_state=42,
                n_jobs=-1,
                eval_metric="mlogloss",
                tree_method="hist",
            ),
            "RandomForest": RandomForestClassifier(
                n_estimators=250,
                max_depth=None,
                class_weight="balanced_subsample",
                random_state=42,
                n_jobs=-1,
            ),
        }

        best_score = -1.0
        best_model = None
        best_model_name = ""
        best_accuracy = 0.0
        results: list[dict] = []

        for name, model in models.items():
            print(f"Training {name}")
            start_time = time.time()
            try:
                model.fit(X_train, y_train)
                y_pred = model.predict(X_test)

                accuracy = accuracy_score(y_test, y_pred)
                f1 = f1_score(y_test, y_pred, average="weighted")
                elapsed = time.time() - start_time

                results.append(
                    {
                        "Model": name,
                        "Accuracy": accuracy,
                        "F1_Score": f1,
                        "Training_Time_Seconds": round(elapsed, 2),
                    }
                )
                print(f"{name}: Accuracy={accuracy:.4f}, F1={f1:.4f}, Time={elapsed:.2f}s")

                if f1 > best_score:
                    best_score = float(f1)
                    best_accuracy = float(accuracy)
                    best_model = model
                    best_model_name = name
            except Exception as exc:
                print(f"{name} training failed: {exc}")

        if best_model is None:
            raise ValueError("All candidate models failed to train")

        return best_model, best_model_name, best_score, best_accuracy, results

    def train(self, data_dir: str, model_path: str, sample_frac: float = 0.15) -> TrainingResult:
        print("Network anomaly detection model training started")
        csv_files, _, total_size = self.get_dataset_info(data_dir)
        if not csv_files:
            raise ValueError(f"No CSV files found in {data_dir}")
        print(f"Dataset size: {total_size:.2f} MB across {len(csv_files)} files")

        data = self.load_and_sample_data(data_dir=data_dir, sample_frac=sample_frac)
        X, y, scaler, imputer, label_encoder, feature_names = self.preprocess_data(data)

        X_train, X_test, y_train, y_test = train_test_split(
            X,
            y,
            test_size=0.2,
            random_state=42,
            stratify=y,
        )

        best_model, best_model_name, best_f1, best_accuracy, results = self.train_and_evaluate_models(
            X_train, X_test, y_train, y_test
        )

        pipeline = {
            "model": best_model,
            "scaler": scaler,
            "imputer": imputer,
            "label_encoder": label_encoder,
            "feature_names": feature_names,
            "target_column": self.target_column,
            "training_date": time.strftime("%Y-%m-%d %H:%M:%S"),
            "metrics": {
                "best_model": best_model_name,
                "weighted_f1": best_f1,
                "accuracy": best_accuracy,
            },
        }

        os.makedirs(os.path.dirname(model_path) if os.path.dirname(model_path) else ".", exist_ok=True)
        joblib.dump(pipeline, model_path)

        pd.DataFrame(results).to_csv("model_training_results.csv", index=False)

        print("Training completed")
        print(f"Best model: {best_model_name}")
        print(f"Weighted F1: {best_f1:.4f}")
        print(f"Model saved: {model_path}")

        return TrainingResult(
            best_model_name=best_model_name,
            best_f1=best_f1,
            best_accuracy=best_accuracy,
            model_path=model_path,
        )


def train_with_sampling(data_dir: str, model_path: str, target_col: str = "Label", sample_frac: float = 0.15) -> float:
    trainer = CICIDSTrainer()
    result = trainer.train(data_dir=data_dir, model_path=model_path, sample_frac=sample_frac)
    return result.best_f1


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train CIC-IDS2017 anomaly detection model")
    parser.add_argument("--data-dir", type=str, default="data", help="Directory containing CIC-IDS CSV files")
    parser.add_argument("--model-path", type=str, default="model/cic_ids_model.pkl", help="Output model path")
    parser.add_argument("--sample-frac", type=float, default=0.15, help="Sampling fraction per class and file")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    trainer = CICIDSTrainer()
    trainer.train(data_dir=args.data_dir, model_path=args.model_path, sample_frac=args.sample_frac)


if __name__ == "__main__":
    main()
