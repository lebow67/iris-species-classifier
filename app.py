import gradio as gr
import pandas as pd
import joblib

model = joblib.load("model_knn.pkl")
scaler = joblib.load("scaler.pkl")
expected_columns = joblib.load("columns.pkl")

def predict(petal_length,petal_width,sepal_length ,sepal_width):
    row ={
        "petal_length" : petal_length,
        "petal_width" : petal_width,
        "sepal_length" : sepal_length,
        "sepal_width" : sepal_width
    }

    input_df = pd.DataFrame([row])

    for col in expected_columns:
        if col not in input_df.columns:
            input_df[col] = 0
    input_df = input_df[expected_columns]

    scaled = scaler.transform(input_df)
    results = model.predict(scaled)[0]

    return f"predicted species : {results}"

demo = gr.Interface(
    fn=predict,
    inputs=[
        gr.Slider(1.0, 7.0, value=4.0, step=0.1, label="Petal Length (cm)"),
        gr.Slider(0.1, 2.5, value=1.3, step=0.1, label="Petal Width (cm)"),
        gr.Slider(4.0, 8.0, value=5.8, step=0.1, label="Sepal Length (cm)"),
        gr.Slider(2.0, 4.5, value=3.0, step=0.1, label="Sepal Width (cm)"),
    ],
    outputs=gr.Textbox(label="Prediction"),
    title="🌸 Iris Species Classifier",
    description="Adjust the petal and sepal measurements to predict the Iris species using a K-Nearest Neighbors model.",
    theme=gr.themes.Soft()
)

demo.launch()                
            