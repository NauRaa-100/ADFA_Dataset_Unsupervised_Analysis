
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
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
.gradio-container { 
    max-width: 1000px !important;
    margin: 0 auto !important;
    font-family: 'Inter', sans-serif !important;
    background: #000000 !important;
    color: #ffffff !important;
    padding: 30px !important;
    border-radius: 0 !important;
    box-shadow: none !important;
    line-height: 1.5 !important;
    animation: backgroundPulse 10s ease-in-out infinite !important;
}
@keyframes backgroundPulse {
    0%, 100% { background: #000000; }
    50% { background: #0a0a0a; }
}
/* Title styling */
h1, h2, h3 {
    font-weight: 600 !important;
    color: #ffffff !important;
    text-align: left !important;
    margin-bottom: 15px !important;
    font-size: 1.8rem !important;
    transition: color 0.3s ease !important;
    animation: titleGlow 3s ease-in-out infinite alternate !important;
}
@keyframes titleGlow {
    from { text-shadow: 0 0 5px rgba(0, 212, 255, 0.3); }
    to { text-shadow: 0 0 15px rgba(0, 212, 255, 0.7); }
}
h1 {
    font-size: 2rem !important;
    color: #00d4ff !important;
    margin-bottom: 10px !important;
    border-bottom: 3px solid #00d4ff !important;
    padding-bottom: 10px !important;
    text-shadow: 0 0 10px rgba(0, 212, 255, 0.5) !important;
    animation: slideInTitle 1s ease-out !important;
}
@keyframes slideInTitle {
    from { transform: translateX(-50px); opacity: 0; }
    to { transform: translateX(0); opacity: 1; }
}
h1:hover {
    color: #00aaff !important;
    animation: bounce 0.5s ease !important;
}
@keyframes bounce {
    0%, 20%, 50%, 80%, 100% { transform: translateY(0); }
    40% { transform: translateY(-10px); }
    60% { transform: translateY(-5px); }
}
/* Card style */
.card {
    padding: 30px !important;
    border-radius: 12px !important;
    background: linear-gradient(135deg, #1a1a1a, #2a2a2a) !important;
    border: 2px solid #00d4ff !important;
    box-shadow: 0 4px 15px rgba(0,0,0,0.3) !important;
    margin-bottom: 20px !important;
    transition: all 0.3s ease !important;
    animation: fadeInCard 1.5s ease-out !important;
}
@keyframes fadeInCard {
    from { opacity: 0; transform: translateY(20px); }
    to { opacity: 1; transform: translateY(0); }
}
.card:hover {
    transform: translateY(-5px) scale(1.02) !important;
    box-shadow: 0 8px 25px rgba(0, 212, 255, 0.2) !important;
    border-color: #00aaff !important;
    animation: cardHoverPulse 0.6s ease-in-out infinite alternate !important;
}
@keyframes cardHoverPulse {
    from { box-shadow: 0 8px 25px rgba(0, 212, 255, 0.2); }
    to { box-shadow: 0 8px 35px rgba(0, 212, 255, 0.5); }
}
/* KPI boxes */
.kpi-box {
    background: linear-gradient(135deg, #2a2a2a, #3a3a3a) !important;
    color: #ffffff !important;
    padding: 20px !important;
    border-radius: 12px !important;
    text-align: center !important;
    font-weight: 600 !important;
    font-size: 1.3rem !important;
    border: 2px solid #ff6b6b !important;
    box-shadow: 0 2px 10px rgba(0,0,0,0.2) !important;
    transition: all 0.3s ease !important;
    animation: kpiFadeIn 2s ease-out !important;
}
@keyframes kpiFadeIn {
    from { opacity: 0; transform: scale(0.9); }
    to { opacity: 1; transform: scale(1); }
}
.kpi-box:hover {
    background: linear-gradient(135deg, #3a3a3a, #4a4a4a) !important;
    transform: scale(1.05) rotate(2deg) !important;
    box-shadow: 0 4px 20px rgba(255,107,107, 0.3) !important;
    border-color: #ff8c8c !important;
    animation: kpiHoverShake 0.5s ease-in-out !important;
}
@keyframes kpiHoverShake {
    0%, 100% { transform: scale(1.05) rotate(2deg); }
    25% { transform: scale(1.05) rotate(-2deg); }
    75% { transform: scale(1.05) rotate(2deg); }
}
/* Plot frames */
.gradio_plot {
    border-radius: 12px !important;
    border: 2px solid #4ecdc4 !important;
    padding: 12px !important;
    background: linear-gradient(135deg, #1a1a1a, #2a2a2a) !important;
    box-shadow: 0 2px 10px rgba(0,0,0,0.2) !important;
    transition: all 0.3s ease !important;
    animation: plotSlideIn 2.5s ease-out !important;
}
@keyframes plotSlideIn {
    from { opacity: 0; transform: translateX(-30px); }
    to { opacity: 1; transform: translateX(0); }
}
.gradio_plot:hover {
    border-color: #5fd4c4 !important;
    box-shadow: 0 4px 20px rgba(78,205,196, 0.3) !important;
    transform: scale(1.02) !important;
    animation: plotHoverGlow 0.8s ease-in-out infinite alternate !important;
}
@keyframes plotHoverGlow {
    from { box-shadow: 0 4px 20px rgba(78,205,196, 0.3); }
    to { box-shadow: 0 4px 30px rgba(78,205,196, 0.6); }
}
/* Prediction field style */
.prediction-box {
    font-size: 1.8rem !important;
    font-weight: 700 !important;
    color: white !important;
    background: linear-gradient(135deg, #1a1a1a, #00aaff) !important;
    border-radius: 12px !important;
    padding: 20px !important;
    text-align: center !important;
    box-shadow: 0 4px 15px rgba(0, 212, 255, 0.3) !important;
    transition: all 0.3s ease !important;
    animation: predictionPulse 4s ease-in-out infinite !important;
}
@keyframes predictionPulse {
    0%, 100% { transform: scale(1); }
    50% { transform: scale(1.02); }
}
.prediction-box:hover {
    transform: scale(1.05) !important;
    box-shadow: 0 6px 25px rgba(0, 212, 255, 0.4) !important;
    animation: predictionHoverBounce 0.6s ease !important;
}
@keyframes predictionHoverBounce {
    0% { transform: scale(1.05); }
    50% { transform: scale(1.1); }
    100% { transform: scale(1.05); }
}
.prediction-box[style*="background: #dc3545"] {
    background: linear-gradient(135deg, #dc3545, #fd7e14) !important;
    box-shadow: 0 4px 15px rgba(220, 53, 69, 0.3) !important;
}
.prediction-box[style*="background: #dc3545"]:hover {
    box-shadow: 0 6px 25px rgba(220, 53, 69, 0.4) !important;
}
/* Input box styling */
.gradio-textbox textarea {
    border-radius: 10px !important;
    border: 2px solid #45b7d1 !important;
    padding: 10px !important;
    font-size: 1rem !important;
    background: linear-gradient(135deg, #2a2a2a, #3a3a3a) !important;
    color: #ffffff !important;
    transition: all 0.3s ease !important;
    animation: inputFadeIn 1s ease-out !important;
}
@keyframes inputFadeIn {
    from { opacity: 0; transform: translateY(10px); }
    to { opacity: 1; transform: translateY(0); }
}
.gradio-textbox textarea:focus {
    border-color: #5fb7d1 !important;
    box-shadow: 0 0 0 3px rgba(69,183,209, 0.2) !important;
    background: linear-gradient(135deg, #3a3a3a, #4a4a4a) !important;
    animation: inputFocusPulse 0.5s ease-in-out infinite alternate !important;
}
@keyframes inputFocusPulse {
    from { box-shadow: 0 0 0 3px rgba(69,183,209, 0.2); }
    to { box-shadow: 0 0 0 6px rgba(69,183,209, 0.1); }
}
/* Button styling for examples */
.gradio-examples button {
    background: linear-gradient(135deg, #444444, #555555) !important;
    color: white !important;
    border: 2px solid #96ceb4 !important;
    border-radius: 8px !important;
    padding: 10px 18px !important;
    font-weight: 500 !important;
    transition: all 0.3s ease !important;
    animation: buttonSlideIn 1.2s ease-out !important;
}
@keyframes buttonSlideIn {
    from { opacity: 0; transform: translateX(20px); }
    to { opacity: 1; transform: translateX(0); }
}
.gradio-examples button:hover {
    background: linear-gradient(135deg, #96ceb4, #a8d5b8) !important;
    color: #000000 !important;
    transform: translateY(-2px) scale(1.05) !important;
    box-shadow: 0 4px 15px rgba(150,206,180, 0.3) !important;
    animation: buttonHoverWiggle 0.4s ease-in-out !important;
}
@keyframes buttonHoverWiggle {
    0%, 100% { transform: translateY(-2px) scale(1.05); }
    25% { transform: translateY(-2px) scale(1.05) rotate(-2deg); }
    75% { transform: translateY(-2px) scale(1.05) rotate(2deg); }
}
/* Markdown styling */
.gradio-markdown {
    color: #cccccc !important;
    line-height: 1.6 !important;
    font-size: 1rem !important;
    animation: markdownFadeIn 2s ease-out !important;
}
@keyframes markdownFadeIn {
    from { opacity: 0; }
    to { opacity: 1; }
}
/* Separator */
hr {
    border: none !important;
    height: 2px !important;
    background: linear-gradient(90deg, #333333, #00d4ff, #333333) !important;
    margin: 10px 0 !important;
    animation: hrSlideIn 1.5s ease-out !important;
}
@keyframes hrSlideIn {
    from { width: 0%; }
    to { width: 100%; }
}
/* Responsive adjustments */
@media (max-width: 768px) {
    .gradio-container {
        padding: 10px !important;
    }
    h1 {
        font-size: 2.2rem !important;
    }
    .card {
        padding: 10px !important;
    }
}
.gradio-container .gr-button {
    background: linear-gradient(135deg, #444444, #555555) !important;
    color: white !important;
    border: 2px solid #ffeaa7 !important;
    border-radius: 8px !important;
    padding: 10px 18px !important;
    font-weight: 500 !important;
    transition: all 0.3s ease !important;
    animation: grButtonFadeIn 1.5s ease-out !important;
}
@keyframes grButtonFadeIn {
    from { opacity: 0; transform: scale(0.9); }
    to { opacity: 1; transform: scale(1); }
}
.gradio-container .gr-button:hover {
    background: linear-gradient(135deg, #ffeaa7, #fff4b8) !important;
    color: #000000 !important;
    transform: translateY(-2px) scale(1.05) !important;
    box-shadow: 0 4px 15px rgba(255,234,167, 0.3) !important;
    animation: grButtonHoverBounce 0.5s ease !important;
}
@keyframes grButtonHoverBounce {
    0% { transform: translateY(-2px) scale(1.05); }
    50% { transform: translateY(-4px) scale(1.1); }
    100% { transform: translateY(-2px) scale(1.05); }
}
.gradio-container .gr-textbox {
    border-radius: 10px !important;
    border: 2px solid #ff6b6b !important;
    padding: 10px !important;
    font-size: 1rem !important;
    background: linear-gradient(135deg, #2a2a2a, #3a3a3a) !important;
    color: #ffffff !important;
    transition: all 0.3s ease !important;
    animation: grTextboxFadeIn 1s ease-out !important;
}
@keyframes grTextboxFadeIn {
    from { opacity: 0; transform: translateY(10px); }
    to { opacity: 1; transform: translateY(0); }
}
.gradio-container .gr-textbox:focus {
    border-color: #ff8c8c
.gradio-container .gr-markdown {
    color: #cccccc !important;
    line-height: 1.6 !important;
    font-size: 1rem !important;
    animation: grMarkdownFadeIn 2s ease-out !important;
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
