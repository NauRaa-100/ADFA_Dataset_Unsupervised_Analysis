import gradio as gr
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
from pathlib import Path

# --------------------------
# Load data & models
# --------------------------
df = pd.read_csv("adfa_parsed.csv")
pipeline = joblib.load("unsup_iforest_pipeline.pkl")
rules = np.load("feature_rules.npy")  # feature-specific thresholds (optional)

# --------------------------
# Feature extraction
# --------------------------
def make_numeric_features(texts):
    X = pd.DataFrame()
    X["length"] = texts.apply(lambda x: len(str(x).split()))
    X["unique_calls"] = texts.apply(lambda x: len(set(str(x).split())))
    X["mean_call_log"] = texts.apply(lambda x: np.log1p(np.mean([int(t) for t in str(x).split() if t.isdigit()])))
    return X

# --------------------------
# Detection function
# --------------------------
def detect_log(text):
    df_text = pd.DataFrame({"text": [text]})
    X_feat = make_numeric_features(df_text)

    # ---------------------- Real-time plot ----------------------
    fig, ax = plt.subplots(figsize=(6,4))
    features = ["length","unique_calls","mean_call_log"]
    user_values = X_feat.iloc[0].values
    normal_means = [
        df["text"].apply(lambda x: len(str(x).split())).mean(),
        df["text"].apply(lambda x: len(set(str(x).split()))).mean(),
        np.log1p(df["text"].apply(lambda x: np.mean([int(t) for t in str(x).split() if t.isdigit()]))).mean()
    ]
    colors = ["blue" if user_values[i]<=normal_means[i]*1.5 else "red" for i in range(len(features))]

    ax.bar(features, user_values, color=colors)
    ax.plot(features, normal_means, "g--", label="Normal Avg")
    ax.set_title("Features for Input Log")
    ax.legend()
    plt.tight_layout()
    fig_path = "input_log_features.png"
    fig.savefig(fig_path)
    plt.close(fig)

    # ---------------------- Detection ----------------------
    status = "✔️ Normal"
    suggestions = []

    if user_values[0] > normal_means[0]*2:
        status = "❌ Suspicious (Feature rule: length)"
        suggestions.append(f"Length > typical max ({int(normal_means[0])})")
    if user_values[1] > normal_means[1]*2:
        status = "❌ Suspicious (Feature rule: unique_calls)"
        suggestions.append(f"Unique calls > typical max ({int(normal_means[1])})")
    if user_values[2] > np.log1p(1000):
        status = "❌ Suspicious (Feature rule: mean_call_log)"
        suggestions.append("Mean syscall ID too high")

    pred = pipeline.predict(X_feat)[0]  # 1 = normal, -1 = anomaly
    if pred == -1:
        status = "⚠️ Suspicious (IsolationForest)"
        suggestions.append("IsolationForest flagged anomaly")

    if suggestions:
        status += "\nSuggestions:\n" + "\n".join(suggestions)

    return status, fig_path

# --------------------------
# Load static visualizations
# --------------------------
def load_plot(name):
    path = Path(name)
    if path.exists():
        return str(path)
    return None

# --------------------------
# Dashboard
# --------------------------
css_style = """
.gradio-container {
    background: linear-gradient(135deg, #1a1a1a 0%, #2d2d2d 50%, #3a3a3a 100%);
    color: #e0e0e0;
    padding: 20px;
    font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
    border-radius: 10px;
    box-shadow: 0 4px 8px rgba(0,0,0,0.3);
    animation: fadeIn 1s ease-in-out;
}
@keyframes fadeIn {
    from { opacity: 0; transform: translateY(20px); }
    to { opacity: 1; transform: translateY(0); }
}
h1 {
    color: #ffffff;
    text-align: center;
    font-weight: 600;
    margin-bottom: 20px;
    text-shadow: 1px 1px 3px rgba(0,0,0,0.5);
    animation: slideIn 1.5s ease-out;
}
@keyframes slideIn {
    from { transform: translateX(-100%); opacity: 0; }
    to { transform: translateX(0); opacity: 1; }
}
.gradio-container .gr-button {
    background-color: #4a4a4a;
    color: #ffffff;
    border: 1px solid #666666;
    border-radius: 8px;
    padding: 10px 20px;
    font-weight: 500;
    transition: all 0.3s ease;
    box-shadow: 0 2px 4px rgba(0,0,0,0.2);
    cursor: pointer;
    position: relative;
    overflow: hidden;
}
.gradio-container .gr-button::before {
    content: '';
    position: absolute;
    top: 0;
    left: -100%;
    width: 100%;
    height: 100%;
    background: linear-gradient(90deg, transparent, rgba(255,255,255,0.2), transparent);
    transition: left 0.5s;
}
.gradio-container .gr-button:hover::before {
    left: 100%;
}
.gradio-container .gr-button:hover {
    background-color: #666666;
    transform: translateY(-2px) scale(1.05);
    box-shadow: 0 4px 8px rgba(0,0,0,0.4);
}
.gradio-container .gr-textbox {
    background-color: #2d2d2d;
    color: #e0e0e0;
    border: 1px solid #666666;
    border-radius: 5px;
    padding: 10px;
    font-size: 14px;
    transition: border-color 0.3s ease, box-shadow 0.3s ease;
}
.gradio-container .gr-textbox:focus {
    border-color: #888888;
    box-shadow: 0 0 5px rgba(136,136,136,0.5);
    transform: scale(1.02);
}
.gradio-container .gr-markdown {
    color: #e0e0e0;
    line-height: 1.6;
    animation: fadeInUp 2s ease-out;
}
@keyframes fadeInUp {
    from { opacity: 0; transform: translateY(30px); }
    to { opacity: 1; transform: translateY(0); }
}
.gradio-container .gr-tab {
    background-color: #2d2d2d;
    border-radius: 8px;
    margin-bottom: 10px;
    transition: background-color 0.3s ease, transform 0.3s ease;
}
.gradio-container .gr-tab:hover {
    background-color: #3a3a3a;
    transform: translateY(-5px);
}
.gradio-container .gr-tab-selected {
    background-color: #4a4a4a;
    border-bottom: 2px solid #888888;
    transform: scale(1.02);
}
.gradio-container .gr-image {
    border-radius: 8px;
    box-shadow: 0 2px 4px rgba(0,0,0,0.3);
    transition: transform 0.3s ease, box-shadow 0.3s ease;
}
.gradio-container .gr-image:hover {
    transform: scale(1.05);
    box-shadow: 0 4px 8px rgba(0,0,0,0.5);
}
"""

with gr.Blocks(title="ADFA Unsupervised Dashboard", css=css_style) as demo:

    gr.Markdown("# **ADFA Dataset Unsupervised Analysis**")

    with gr.Tab("🕵️ Real-time Detection"):
        input_txt = gr.Textbox(label="Paste Log", placeholder="Paste a log sequence here")
        output_lbl = gr.Textbox(label="Prediction")
        detect_btn = gr.Button("Detect")
        detect_img = gr.Image()

    with gr.Tab("📈 Visualizations"):
        gr.Markdown("### Dataset Visualizations")
        gr.Image(load_plot("char_length_dist.png"))
        gr.Image(load_plot("char_len_by_class.png"))
        gr.Image(load_plot("correlation_heatmap.png"))

        detect_btn.click(
            detect_log,
            inputs=input_txt,
            outputs=[output_lbl, detect_img]
        )

    # -------------------------- Footer note --------------------------
    gr.Markdown(
        "<p style='color:#ff6b6b; font-style:italic; text-align:center;'>⚠️ Note: The unsupervised model may occasionally make mistakes, especially on unusual log sequences.</p>"
    )

demo.launch()
