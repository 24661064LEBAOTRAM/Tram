
import requests
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
import numpy as np
import plotly.graph_objects as go
import gradio as gr

SUPABASE_URL = "https://ocateixuzulwmrtxseom.supabase.co"
SUPABASE_API_KEY = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6Im9jYXRlaXh1enVsd21ydHhzZW9tIiwicm9sZSI6ImFub24iLCJpYXQiOjE3NTYyMTYyMTQsImV4cCI6MjA3MTc5MjIxNH0.w7PLqjLGj4hNNGDh81NwDodEUCCqVSVm_PL0FpYWif8"

headers = {
    "apikey": SUPABASE_API_KEY,
    "Authorization": f"Bearer {SUPABASE_API_KEY}",
    "Content-Type": "application/json"
}

url = f"{SUPABASE_URL}/rest/v1/order_details?select=quantity,unit_price"
response = requests.get(url, headers=headers)

if response.status_code != 200:
    raise Exception(f"Loi Supabase: {response.status_code}, {response.text}")

df = pd.DataFrame(response.json())
print("Du lieu:", df.shape)
print(df.head())

df["Revenue"] = df["quantity"] * df["unit_price"]
X = df[["quantity", "unit_price"]]
y = df["Revenue"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = LinearRegression()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

r2 = r2_score(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))

print(f"R2: {r2:.4f}, MAE: {mae:.4f}, RMSE: {rmse:.4f}")

if r2 >= 0.9:
    conclusion = "✅ Mo hinh rat tot (do khop cao, co the su dung thuc te)."
    color = "green"
elif r2 >= 0.7:
    conclusion = "🟩 Mo hinh kha tot, co the dung cho du bao so bo."
    color = "limegreen"
elif r2 >= 0.5:
    conclusion = "🟨 Mo hinh tam on, can them bien dau vao hoac lam sach du lieu."
    color = "orange"
else:
    conclusion = "🟥 Mo hinh yeu, can cai thien cau truc du lieu hoac thu mo hinh phi tuyen."
    color = "red"

def make_gauge():
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=r2 * 100,
        title={'text': "🎯 Do chinh xac mo hinh hoi quy (%)"},
        gauge={
            'axis': {'range': [0, 100]},
            'bar': {'color': color},
            'steps': [
                {'range': [0, 50], 'color': "lightcoral"},
                {'range': [50, 70], 'color': "khaki"},
                {'range': [70, 90], 'color': "lightgreen"},
                {'range': [90, 100], 'color': "green"}
            ],
            'threshold': {'line': {'color': "blue", 'width': 4}, 'value': r2 * 100}
        }
    ))
    fig.add_annotation(text=f"📋 Ket luan: {conclusion}",
        x=0.5, y=-0.3, showarrow=False, font=dict(size=14, color=color))
    fig.update_layout(height=400, margin=dict(t=60, b=120))
    return fig

def make_scatter():
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=y_test, y=y_pred, mode='markers',
        marker=dict(size=8, color='royalblue', opacity=0.6),
        name='Thuc te vs Du bao'
    ))
    fig.add_trace(go.Scatter(
        x=[y_test.min(), y_test.max()],
        y=[y_test.min(), y_test.max()],
        mode='lines', name='Duong hoan hao (y=x)',
        line=dict(color='red', dash='dash')
    ))
    fig.update_layout(
        title="📊 So sanh doanh thu thuc te va du bao",
        xaxis_title="Doanh thu thuc te",
        yaxis_title="Doanh thu du bao",
        annotations=[dict(
            text=f"📈 R²={r2:.2f} | MAE={mae:.2f} | RMSE={rmse:.2f}",
            x=0.5, y=-0.2, showarrow=False, xref="paper", yref="paper",
            font=dict(size=12, color=color)
        )]
    )
    return fig

with gr.Blocks(title="Bai 2 - Hoi quy tuyen tinh") as demo:
    gr.Markdown(f"## 📈 Bai 2: Du bao doanh thu bang Hoi quy tuyen tinh")
    gr.Markdown(f"**R² = {r2:.4f} | MAE = {mae:.4f} | RMSE = {rmse:.4f}**")
    gr.Plot(make_gauge)
    gr.Plot(make_scatter)

demo.launch(server_name="0.0.0.0", server_port=7861, share=True)
