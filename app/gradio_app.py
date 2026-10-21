import gradio as gr
import requests
import os

API_URL = os.getenv("API_URL", "http://127.0.0.1:8000")


def clasificar(sepal_length, sepal_width, petal_length, petal_width):
    sample = {
        "sepal_length": sepal_length,
        "sepal_width": sepal_width,
        "petal_length": petal_length,
        "petal_width": petal_width
    }

    try:
        response = requests.post(f"{API_URL}/predict", json=sample, timeout=60)
    except requests.RequestException:
        raise gr.Error("No se pudo conectar a la API")

    if not response.ok:
        raise gr.Error(f"La API respondió con error: {response.json()}")

    return response.json()["probabilities"]


demo = gr.Interface(
    fn=clasificar,
    inputs=[
        gr.Slider(0.0, 10.0, value=5.8, step=0.1, label="Largo del sépalo (cm)"),
        gr.Slider(0.0, 10.0, value=5.8, step=0.1, label="Ancho del sépalo (cm)"),
        gr.Slider(0.0, 10.0, value=5.8, step=0.1, label="Largo del pétalo (cm)"),
        gr.Slider(0.0, 10.0, value=5.8, step=0.1, label="Ancho del pétalo (cm)"),
    ],
    outputs=gr.Label(num_top_classes=3, label="Especie predicha"),
    examples=[
        [5.1, 3.5, 1.4, 0.2], # setosa
        [7.0 ,3.2, 4.7, 1.4], # versicolor
        [6.3, 3.3, 6.0, 2.5] # virginica
    ],
    title="🌸 Clasificador de flores Iris (Despliegue Automático) 👽",
    description="Mueve los sliders con las medidas de la flor",
    flagging_mode="never"
)


if __name__ == "__main__":
    demo.launch()