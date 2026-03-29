import gradio as gr
import joblib
import json
import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Load artefacts
# ---------------------------------------------------------------------------
ridge_model = joblib.load("ridge_model.pkl")
lasso_model = joblib.load("lasso_model.pkl")
scaler      = joblib.load("scaler.pkl")

with open("feature_cols.json") as f:
    FEATURE_COLS = json.load(f)

CONGESTION_LABELS = {
    0: "0 – Free Flow",
    1: "1 – Light",
    2: "2 – Moderate",
    3: "3 – Heavy",
    4: "4 – Severe / Gridlock",
}

# ---------------------------------------------------------------------------
# Prediction function
# ---------------------------------------------------------------------------
def predict(
    hour, weekday,
    lag1, lag2, lag3, lag6, lag24, roll3,
    avg_speed,
    weather, road_condition,
    is_holiday, school_in_session,
    event_flag, sensor_ok,
):
    row = {col: 0.0 for col in FEATURE_COLS}

    # Lag / rolling features
    row["lag1"]  = float(lag1)
    row["lag2"]  = float(lag2)
    row["lag3"]  = float(lag3)
    row["lag6"]  = float(lag6)
    row["lag24"] = float(lag24)
    row["roll3"] = float(roll3)

    # Speed
    row["avg_speed"] = float(avg_speed)

    # Cyclical time encoding  (matches notebook exactly)
    row["hour_sin"]    = np.sin(2 * np.pi * hour    / 24)
    row["hour_cos"]    = np.cos(2 * np.pi * hour    / 24)
    row["weekday_sin"] = np.sin(2 * np.pi * weekday / 7)
    row["weekday_cos"] = np.cos(2 * np.pi * weekday / 7)

    # Binary flags
    row["is_holiday"]        = 1.0 if is_holiday        else 0.0
    row["school_in_session"] = 1.0 if school_in_session else 0.0
    row["event_flag"]        = 1.0 if event_flag        else 0.0
    row["sensor_ok"]         = 1.0 if sensor_ok         else 0.0

    # One-hot weather
    for w in ["Clear", "Foggy", "Rainy", "Unknown"]:
        row[f"weather_{w}"] = 1.0 if weather == w else 0.0

    # One-hot road condition
    for r in ["Good", "Moderate", "Poor", "Unknown"]:
        row[f"road_{r}"] = 1.0 if road_condition == r else 0.0

    X = pd.DataFrame([row])[FEATURE_COLS]
    X_scaled = scaler.transform(X)

    # Regression: predicted vehicle count
    volume = ridge_model.predict(X_scaled)[0]
    volume = max(0, round(volume))

    # Classification: congestion level
    level       = int(lasso_model.predict(X_scaled)[0])
    confidence  = lasso_model.predict_proba(X_scaled)[0].max()
    label       = CONGESTION_LABELS.get(level, str(level))

    return (
        f"{volume} vehicles / hr",
        f"{label}  (confidence: {confidence:.1%})",
    )


# ---------------------------------------------------------------------------
# Gradio UI
# ---------------------------------------------------------------------------
with gr.Blocks(title="Traffic Ridge Predictor") as demo:
    gr.Markdown(
        """
        # Traffic Ridge Predictor
        Predicts **vehicle count** (RidgeCV regression) and **congestion level**
        (ElasticNet classification) for a given hour of traffic conditions.
        """
    )

    with gr.Row():
        with gr.Column():
            gr.Markdown("### Time")
            hour    = gr.Slider(0, 23, step=1, value=8,  label="Hour of Day (0–23)")
            weekday = gr.Slider(0, 6,  step=1, value=0,  label="Day of Week  (0=Mon … 6=Sun)")

        with gr.Column():
            gr.Markdown("### Past Traffic (lag features)")
            lag1  = gr.Number(value=300, label="lag1  – vehicles 1 hr ago")
            lag2  = gr.Number(value=280, label="lag2  – vehicles 2 hrs ago")
            lag3  = gr.Number(value=260, label="lag3  – vehicles 3 hrs ago")
            lag6  = gr.Number(value=200, label="lag6  – vehicles 6 hrs ago")
            lag24 = gr.Number(value=310, label="lag24 – vehicles 24 hrs ago")
            roll3 = gr.Number(value=280, label="roll3 – 3-hr rolling mean")

    with gr.Row():
        with gr.Column():
            gr.Markdown("### Road & Environment")
            avg_speed      = gr.Number(value=40.0, label="Average Speed (km/h)")
            weather        = gr.Dropdown(["Clear", "Foggy", "Rainy", "Unknown"],
                                         value="Clear", label="Weather")
            road_condition = gr.Dropdown(["Good", "Moderate", "Poor", "Unknown"],
                                         value="Good",  label="Road Condition")

        with gr.Column():
            gr.Markdown("### Flags")
            is_holiday        = gr.Checkbox(label="Public Holiday?")
            school_in_session = gr.Checkbox(label="School in Session?")
            event_flag        = gr.Checkbox(label="Active Road Event?")
            sensor_ok         = gr.Checkbox(label="Sensor OK?", value=True)

    predict_btn = gr.Button("Predict", variant="primary")

    with gr.Row():
        out_volume     = gr.Text(label="Predicted Vehicle Count")
        out_congestion = gr.Text(label="Congestion Level")

    predict_btn.click(
        fn=predict,
        inputs=[
            hour, weekday,
            lag1, lag2, lag3, lag6, lag24, roll3,
            avg_speed, weather, road_condition,
            is_holiday, school_in_session,
            event_flag, sensor_ok,
        ],
        outputs=[out_volume, out_congestion],
    )

    gr.Markdown(
        """
        ---
        **Model:** RidgeCV (regression) + LogisticRegressionCV ElasticNet (classification)
        **Trained on:** Hourly traffic sensor data with temporal cross-validation
        """
    )

demo.launch()
