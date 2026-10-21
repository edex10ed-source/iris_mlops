from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

# from iris_mlops import __version__
from iris_mlops.predict import load_model_info, predict


app = FastAPI(
    title="Iris API",
    description="API para predecir la especie de una flor a partir de sus medidas",
    version="0.1.0" #__version__
)


class IrisInput(BaseModel):
    sepal_length: float = Field(gt=0, le=10, description="Largo del sépalo (cm)")
    sepal_width: float = Field(gt=0, le=10, description="Ancho del sépalo (cm)")
    petal_length: float = Field(gt=0, le=10, description="Largo del pétalo (cm)")
    petal_width: float = Field(gt=0, le=10, description="Ancho del pétalo (cm)")


class PredictionOutput(BaseModel):
    species: str
    probabilities: dict[str, float]


# Endpoints
@app.get("/")
def health():
    return {"status": "ok", "message": "Iris API operativa"}

@app.get("/model-info")
def model_info():
    try:
        return load_model_info()
    except FileNotFoundError:
        raise HTTPException(status_code=503, detail="Modelo no encontrado o no entrenado")

@app.post("/predict", response_model=PredictionOutput)
def predict_species(sample: IrisInput):
    try:
        return predict(**sample.model_dump())
    except FileNotFoundError:
        raise HTTPException(status_code=503, detail="Modelo no encontrado o no entrenado")