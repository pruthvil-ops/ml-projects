import os
import sys
import time
from datetime import datetime

import joblib
import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from model.train import train_with_sampling
from monitoring.realtime_monitor import realtime_monitor
from utils.data_utils import get_dataset_info


st.set_page_config(page_title="Network Anomaly Detection", layout="wide")


def _predict_dataframe(df: pd.DataFrame, model_path: str = "model/cic_ids_model.pkl") -> pd.DataFrame:
    if not os.path.exists(model_path):
        raise FileNotFoundError("Trained model not found. Train the model first.")

    pipeline = joblib.load(model_path)
    features = pipeline["feature_names"]

    x_df = df.copy()
    x_df.columns = [str(c).strip() for c in x_df.columns]

    for col in features:
        if col not in x_df.columns:
            x_df[col] = 0

    x_df = x_df[features]
    x_df = x_df.replace([np.inf, -np.inf], np.nan)

    x_values = pipeline["imputer"].transform(x_df)
    x_values = pipeline["scaler"].transform(x_values)

    pred_encoded = pipeline["model"].predict(x_values)
    if hasattr(pipeline["model"], "predict_proba"):
        confidence = np.max(pipeline["model"].predict_proba(x_values), axis=1)
    else:
        confidence = np.ones(len(x_values))

    predictions = pipeline["label_encoder"].inverse_transform(pred_encoded)

    result = df.copy()
    result["Prediction"] = predictions
    result["Confidence"] = confidence
    result["Is_Anomaly"] = result["Prediction"].astype(str).str.strip().str.upper() != "BENIGN"
    return result


def main():
    st.title("Network Anomaly Detection System")

    page = st.sidebar.radio(
        "Navigation",
        ["Data Overview", "Model Training", "Prediction", "Live Dashboard", "Alerts"],
    )

    if page == "Data Overview":
        show_data_overview()
    elif page == "Model Training":
        show_model_training()
    elif page == "Prediction":
        show_prediction()
    elif page == "Live Dashboard":
        show_live_dashboard()
    elif page == "Alerts":
        show_alerts()


def show_data_overview():
    st.header("Dataset Overview")
    if not os.path.exists("data"):
        st.warning("Data directory not found.")
        return

    csv_files, file_info, total_size = get_dataset_info("data")
    st.write(f"Total dataset size: {total_size:.2f} MB")
    st.write(f"Number of files: {len(csv_files)}")

    if file_info:
        st.dataframe(pd.DataFrame(file_info), use_container_width=True)


def show_model_training():
    st.header("Model Training")
    sample_frac = st.slider("Sampling fraction", 0.05, 0.5, 0.15, 0.05)
    model_path = st.text_input("Model save path", "model/cic_ids_model.pkl")

    if st.button("Start Training", type="primary"):
        with st.spinner("Training model"):
            try:
                score = train_with_sampling("data", model_path, "Label", sample_frac)
                st.success(f"Training completed. Best weighted F1 score: {score:.4f}")
            except Exception as exc:
                st.error(f"Training failed: {exc}")


def show_prediction():
    st.header("Batch Prediction")
    uploaded_file = st.file_uploader("Upload network traffic CSV", type="csv")

    if uploaded_file is None:
        return

    try:
        new_data = pd.read_csv(uploaded_file)
        st.dataframe(new_data.head(), use_container_width=True)
    except Exception as exc:
        st.error(f"Unable to read file: {exc}")
        return

    if st.button("Predict", type="primary"):
        try:
            result = _predict_dataframe(new_data)
            st.session_state["last_prediction_result"] = result
        except Exception as exc:
            st.error(f"Prediction failed: {exc}")

    result = st.session_state.get("last_prediction_result")
    if result is None:
        return

    st.subheader("Prediction Results")
    st.dataframe(result[["Prediction", "Confidence", "Is_Anomaly"]].head(200), use_container_width=True)

    anomaly_count = int(result["Is_Anomaly"].sum())
    total = len(result)
    st.write(f"Detected anomalies: {anomaly_count}/{total}")

    # Always show overall class distribution
    full_breakdown = result["Prediction"].astype(str).value_counts().reset_index()
    full_breakdown.columns = ["Prediction", "Count"]
    fig_all = px.bar(full_breakdown, x="Prediction", y="Count", title="Prediction Distribution")
    st.plotly_chart(fig_all, use_container_width=True)

    # Show detected attack types clearly
    attack_only = result[result["Is_Anomaly"]].copy()
    if not attack_only.empty:
        attack_breakdown = attack_only["Prediction"].astype(str).value_counts().reset_index()
        attack_breakdown.columns = ["Attack Type", "Count"]
        st.subheader("Detected Attack Types")
        st.dataframe(attack_breakdown, use_container_width=True)

        fig_attack = px.pie(
            attack_breakdown,
            values="Count",
            names="Attack Type",
            title="Attack Type Distribution",
        )
        st.plotly_chart(fig_attack, use_container_width=True)
    else:
        st.info("No attack types detected in this file. All rows were predicted as BENIGN.")


def show_live_dashboard():
    st.header("Real-Time Monitoring")

    col1, col2 = st.columns(2)
    with col1:
        if st.button("Start Monitoring", type="primary"):
            realtime_monitor.start_monitoring()
    with col2:
        if st.button("Stop Monitoring"):
            realtime_monitor.stop_monitoring()

    stats = realtime_monitor.get_stats()
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total flows", stats.get("total_flows", 0))
    c2.metric("Anomaly flows", stats.get("anomaly_flows", 0))
    c3.metric("Anomaly rate", f"{stats.get('anomaly_rate', 0.0) * 100:.2f}%")
    latest_attack = "BENIGN"
    if stats.get("latest_anomaly"):
        latest_attack = str(stats["latest_anomaly"].get("prediction", "BENIGN"))
    c4.metric("Latest attack type", latest_attack)

    recent = realtime_monitor.get_recent_flows(50)
    if recent:
        df_recent = pd.DataFrame(recent)
        if "attack_type" not in df_recent.columns:
            df_recent["attack_type"] = np.where(df_recent["is_anomaly"], df_recent["prediction"], "BENIGN")
        st.dataframe(
            df_recent[["timestamp", "attack_type", "prediction", "confidence", "is_anomaly"]],
            use_container_width=True,
        )

        if stats.get("attack_types"):
            attack_df = pd.DataFrame(
                [{"Attack": k, "Count": v} for k, v in stats["attack_types"].items()]
            )
            fig = px.pie(attack_df, values="Count", names="Attack", title="Recent Attack Distribution")
            st.plotly_chart(fig, use_container_width=True)

    auto_refresh = st.checkbox("Auto refresh", value=True)
    refresh_rate = st.slider("Refresh rate (seconds)", 1, 10, 3)
    if auto_refresh and realtime_monitor.is_monitoring:
        time.sleep(refresh_rate)
        st.rerun()


def show_alerts():
    st.header("Recent Alerts")
    flows = realtime_monitor.get_recent_flows(200)
    alerts = [flow for flow in flows if flow.get("is_anomaly")]

    if not alerts:
        st.info("No anomalies detected yet.")
        return

    alert_df = pd.DataFrame(alerts)
    st.dataframe(alert_df[["timestamp", "prediction", "confidence"]], use_container_width=True)
    st.write(f"Last updated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")


if __name__ == "__main__":
    main()
