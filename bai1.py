
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import plotly.graph_objects as go
import gradio as gr

df = pd.read_csv('Iris.csv')
X = df.iloc[:, :-1].values
y = df.iloc[:, -1].values

le = LabelEncoder()
y_encoded = le.fit_transform(y)

X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.3, random_state=42)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

model = SVC(kernel='rbf', C=1.0, gamma='scale', random_state=42)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

y_test_labels = le.inverse_transform(y_test)
y_pred_labels = le.inverse_transform(y_pred)

accuracy = accuracy_score(y_test, y_pred)
cm = confusion_matrix(y_test, y_pred)
report = classification_report(y_test, y_pred, target_names=le.classes_)

print(f"Do chinh xac: {round(accuracy * 100, 2)}%")
print(report)

if accuracy == 1.0:
    conclusion = "✅ Mo hinh SVM rat tot (co the dang overfit – can kiem tra them du lieu)."
    conclusion_color = "green"
elif accuracy >= 0.8:
    conclusion = "🟩 Mo hinh SVM hoat dong on, co the dung de du doan."
    conclusion_color = "limegreen"
elif accuracy >= 0.6:
    conclusion = "🟨 Mo hinh SVM tam on nhung can tinh chinh them tham so."
    conclusion_color = "orange"
else:
    conclusion = "🟥 Mo hinh SVM yeu, can cai thien hoac thay doi kernel."
    conclusion_color = "red"

def make_gauge():
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=accuracy * 100,
        title={'text': "🎯 Do chinh xac mo hinh SVM (%)"},
        gauge={
            'axis': {'range': [0, 100]},
            'bar': {'color': "green" if accuracy > 0.8 else "red"},
            'steps': [
                {'range': [0, 50], 'color': "lightcoral"},
                {'range': [50, 80], 'color': "khaki"},
                {'range': [80, 100], 'color': "lightgreen"}
            ],
            'threshold': {'line': {'color': "blue", 'width': 4}, 'value': accuracy * 100}
        }
    ))
    fig.add_annotation(text="🟥 0-50%: Mo hinh yeu", x=0.5, y=-0.18, showarrow=False, font=dict(size=12, color="red"))
    fig.add_annotation(text="🟨 50-80%: Trung binh", x=0.5, y=-0.25, showarrow=False, font=dict(size=12, color="orange"))
    fig.add_annotation(text="🟩 80-100%: Tot", x=0.5, y=-0.32, showarrow=False, font=dict(size=12, color="green"))
    fig.add_annotation(text=f"📋 Ket luan: {conclusion}", x=0.5, y=-0.45, showarrow=False, font=dict(size=14, color=conclusion_color))
    fig.update_layout(height=420, margin=dict(t=60, b=140))
    return fig

def make_cm():
    fig = go.Figure(data=go.Heatmap(
        z=cm, x=le.classes_, y=le.classes_,
        text=cm, texttemplate="%{text}",
        hoverongaps=False, colorscale='Viridis',
        colorbar=dict(title='So luong')
    ))
    fig.update_layout(
        title='📊 Ma tran nham lan mo hinh SVM theo ten loai',
        xaxis_title='Nhan du doan', yaxis_title='Nhan thuc te',
        annotations=[dict(text="💡 Mau cang dam = so du doan dung cang cao",
            x=0.5, y=-0.2, showarrow=False, xref="paper", yref="paper",
            font=dict(size=12, color="gray"))]
    )
    return fig

def make_bar():
    fig = go.Figure()
    fig.add_trace(go.Bar(name='Thuc te', x=np.arange(len(y_test_labels)), y=y_test_labels))
    fig.add_trace(go.Bar(name='Du doan', x=np.arange(len(y_pred_labels)), y=y_pred_labels))
    fig.update_layout(
        barmode='group',
        title='🔍 So sanh nhan thuc te va nhan du doan (SVM)',
        xaxis_title='Mau kiem tra', yaxis_title='Ten loai',
        annotations=[dict(text="🟦 Thanh xanh: nhan that | 🟥 Thanh do: nhan du doan",
            x=0.5, y=-0.25, showarrow=False, xref="paper", yref="paper",
            font=dict(size=12, color="gray"))]
    )
    return fig

with gr.Blocks(title="Bai 1 - SVM Iris") as demo:
    gr.Markdown(f"## 🌸 Bai 1: Phan biet loai hoa Dien Vi bang SVM\n**Do chinh xac: {round(accuracy*100, 2)}%**")
    gr.Markdown(f"```\n{report}\n```")
    gr.Plot(make_gauge)
    gr.Plot(make_cm)
    gr.Plot(make_bar)

demo.launch(server_name="0.0.0.0", server_port=7860, share=True)
