from __future__ import annotations

import os
import glob
import threading
import time
from collections import Counter, deque
from datetime import datetime

import joblib
import numpy as np
import pandas as pd


class RealTimeMonitor:
    def __init__(self, model_path: str = "model/cic_ids_model.pkl", max_history: int = 1000) -> None:
        self.model_path = model_path
        self.max_history = max_history
        self.is_monitoring = False
        self._flows = deque(maxlen=max_history)
        self._thread: threading.Thread | None = None
        self._stop_event = threading.Event()
        self._pipeline = None
        self._benign_pool: list[dict] = []
        self._attack_pool: list[dict] = []

    def _load_model(self):
        if self._pipeline is None and os.path.exists(self.model_path):
            self._pipeline = joblib.load(self.model_path)
        return self._pipeline

    def _prepare_reference_traffic(self, max_rows_per_file: int = 3000) -> None:
        if self._benign_pool or self._attack_pool:
            return

        pipeline = self._load_model()
        if pipeline is None:
            return

        expected_features = list(pipeline.get("feature_names", []))
        if not expected_features:
            return

        data_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data")
        csv_files = sorted(glob.glob(os.path.join(data_dir, "*.csv")))
        if not csv_files:
            return

        benign_records: list[dict] = []
        attack_records: list[dict] = []

        for csv_path in csv_files:
            try:
                df = pd.read_csv(csv_path, encoding="latin1", low_memory=False, nrows=max_rows_per_file)
                df.columns = [str(c).strip() for c in df.columns]

                label_col = None
                for candidate in ["Label", "label", "Class", "class", "Attack", "attack"]:
                    if candidate in df.columns:
                        label_col = candidate
                        break
                if label_col is None:
                    continue

                for col in expected_features:
                    if col not in df.columns:
                        df[col] = 0

                feature_df = df[expected_features].replace([np.inf, -np.inf], np.nan).fillna(0)
                labels = df[label_col].astype(str).str.strip().str.upper()

                benign_mask = labels == "BENIGN"
                benign_rows = feature_df[benign_mask].head(800)
                attack_rows = feature_df[~benign_mask].head(800)

                if not benign_rows.empty:
                    benign_records.extend(benign_rows.to_dict(orient="records"))
                if not attack_rows.empty:
                    attack_records.extend(attack_rows.to_dict(orient="records"))
            except Exception:
                continue

        self._benign_pool = benign_records[:6000]
        self._attack_pool = attack_records[:6000]

    @staticmethod
    def _base_features() -> dict:
        return {
            "Flow Duration": int(np.random.randint(100, 10000)),
            "Total Fwd Packets": int(np.random.randint(1, 100)),
            "Total Backward Packets": int(np.random.randint(1, 50)),
            "Total Length of Fwd Packets": int(np.random.randint(100, 5000)),
            "Total Length of Bwd Packets": int(np.random.randint(100, 3000)),
            "Flow Bytes/s": int(np.random.randint(1000, 1000000)),
            "Flow Packets/s": int(np.random.randint(10, 1000)),
            "Flow IAT Mean": int(np.random.randint(100, 5000)),
            "Fwd IAT Total": int(np.random.randint(100, 10000)),
            "Fwd Packet Length Max": int(np.random.randint(100, 1500)),
            "Bwd Packet Length Max": int(np.random.randint(100, 1500)),
            "Packet Length Mean": int(np.random.randint(100, 1000)),
            "Packet Length Std": int(np.random.randint(50, 500)),
            "FIN Flag Count": int(np.random.randint(0, 2)),
            "SYN Flag Count": int(np.random.randint(0, 2)),
            "PSH Flag Count": int(np.random.randint(0, 2)),
            "ACK Flag Count": int(np.random.randint(0, 2)),
            "URG Flag Count": int(np.random.randint(0, 2)),
            "Down/Up Ratio": float(np.random.uniform(0.1, 10.0)),
            "Average Packet Size": int(np.random.randint(100, 1000)),
            "Avg Fwd Segment Size": int(np.random.randint(100, 1000)),
            "Avg Bwd Segment Size": int(np.random.randint(100, 1000)),
            "Init_Win_bytes_forward": int(np.random.randint(1000, 65535)),
            "Init_Win_bytes_backward": int(np.random.randint(1000, 65535)),
        }

    @staticmethod
    def _inject_attack_pattern(features: dict, attack_type: str) -> dict:
        updated = dict(features)

        if attack_type == "DDoS":
            updated["Flow Packets/s"] = int(np.random.randint(5000, 20000))
            updated["Flow Bytes/s"] = int(np.random.randint(5_000_000, 20_000_000))
            updated["Total Fwd Packets"] = int(np.random.randint(400, 2000))
            updated["Flow Duration"] = int(np.random.randint(10, 120))
            updated["Fwd IAT Total"] = int(np.random.randint(10, 120))

        elif attack_type == "PortScan":
            updated["Total Fwd Packets"] = int(np.random.randint(100, 500))
            updated["Total Backward Packets"] = int(np.random.randint(0, 6))
            updated["SYN Flag Count"] = 1
            updated["ACK Flag Count"] = 0
            updated["Flow Duration"] = int(np.random.randint(50, 250))

        elif attack_type == "Bot":
            updated["Flow IAT Mean"] = int(np.random.randint(5000, 15000))
            updated["Flow Packets/s"] = int(np.random.randint(100, 700))
            updated["Packet Length Mean"] = int(np.random.randint(200, 700))

        elif attack_type == "Web Attack":
            updated["Total Fwd Packets"] = int(np.random.randint(10, 60))
            updated["Total Backward Packets"] = int(np.random.randint(5, 40))
            updated["Fwd Packet Length Max"] = int(np.random.randint(500, 2000))
            updated["Bwd Packet Length Max"] = int(np.random.randint(100, 600))
            updated["PSH Flag Count"] = 1

        elif attack_type == "DoS":
            updated["Flow Packets/s"] = int(np.random.randint(1200, 8000))
            updated["Flow Bytes/s"] = int(np.random.randint(1_000_000, 8_000_000))
            updated["Total Fwd Packets"] = int(np.random.randint(200, 1000))
            updated["Flow Duration"] = int(np.random.randint(80, 500))
            updated["SYN Flag Count"] = 1

        return updated

    def _simulate_network_traffic(self) -> dict:
        self._prepare_reference_traffic()

        if self._benign_pool or self._attack_pool:
            pick_attack = bool(np.random.rand() < 0.35)
            if pick_attack and self._attack_pool:
                return dict(self._attack_pool[np.random.randint(0, len(self._attack_pool))])
            if self._benign_pool:
                return dict(self._benign_pool[np.random.randint(0, len(self._benign_pool))])
            if self._attack_pool:
                return dict(self._attack_pool[np.random.randint(0, len(self._attack_pool))])

        features = self._base_features()
        sampled_profile = np.random.choice(["BENIGN", "DDoS", "PortScan", "Bot", "Web Attack", "DoS"], p=[0.68, 0.08, 0.08, 0.06, 0.05, 0.05])
        if sampled_profile != "BENIGN":
            features = self._inject_attack_pattern(features, str(sampled_profile))
        return features

    def _predict_flow(self, features: dict) -> tuple[str, float, bool]:
        pipeline = self._load_model()
        if pipeline is None:
            return "BENIGN", 1.0, False

        expected_features = pipeline.get("feature_names", [])
        x_df = pd.DataFrame([features])
        for col in expected_features:
            if col not in x_df.columns:
                x_df[col] = 0
        x_df = x_df[expected_features]

        x_values = pipeline["imputer"].transform(x_df)
        x_values = pipeline["scaler"].transform(x_values)
        x_scaled_df = pd.DataFrame(x_values, columns=expected_features)

        pred_encoded = pipeline["model"].predict(x_scaled_df)[0]
        if hasattr(pipeline["model"], "predict_proba"):
            proba = float(np.max(pipeline["model"].predict_proba(x_scaled_df)[0]))
        else:
            proba = 1.0

        prediction = pipeline["label_encoder"].inverse_transform([pred_encoded])[0]
        is_anomaly = prediction != "BENIGN"
        return str(prediction), proba, bool(is_anomaly)

    def _monitor_loop(self, interval_seconds: float = 1.0):
        while not self._stop_event.is_set():
            features = self._simulate_network_traffic()
            prediction, confidence, is_anomaly = self._predict_flow(features)
            self._flows.append(
                {
                    "timestamp": datetime.now().isoformat(timespec="seconds"),
                    "prediction": prediction,
                    "attack_type": prediction if is_anomaly else "BENIGN",
                    "confidence": confidence,
                    "is_anomaly": is_anomaly,
                    "features": features,
                }
            )
            time.sleep(interval_seconds)

    def start_monitoring(self, interval_seconds: float = 1.0):
        if self.is_monitoring:
            return
        self.is_monitoring = True
        self._stop_event.clear()
        self._thread = threading.Thread(target=self._monitor_loop, args=(interval_seconds,), daemon=True)
        self._thread.start()

    def stop_monitoring(self):
        if not self.is_monitoring:
            return
        self.is_monitoring = False
        self._stop_event.set()
        if self._thread is not None:
            self._thread.join(timeout=2)
            self._thread = None

    def get_recent_flows(self, n: int = 50) -> list[dict]:
        if n <= 0:
            return []
        return list(self._flows)[-n:]

    def get_stats(self) -> dict:
        flows = list(self._flows)
        if not flows:
            return {
                "total_flows": 0,
                "anomaly_flows": 0,
                "anomaly_rate": 0.0,
                "attack_types": {},
                "latest_anomaly": None,
            }

        anomaly_flows = [f for f in flows if f["is_anomaly"]]
        attack_counts = Counter([f["prediction"] for f in anomaly_flows])
        latest_anomaly = anomaly_flows[-1] if anomaly_flows else None

        return {
            "total_flows": len(flows),
            "anomaly_flows": len(anomaly_flows),
            "anomaly_rate": len(anomaly_flows) / len(flows),
            "attack_types": dict(attack_counts),
            "latest_anomaly": latest_anomaly,
        }


realtime_monitor = RealTimeMonitor()
