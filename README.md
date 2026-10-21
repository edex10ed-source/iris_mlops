mkdir -p app data/raw docs models src/iris_mlops tests cd src/iris_mlops && touch api.py config.py data.py predict.py train.py cd ../../app && touch gradio_app.py

uv add pandas scikit-learn joblib "fastapi[standard]" pydantic gradio

-- Instalar el paquete / proyecto en modo editable para ir debuggeando uv pip install -e .

--Debug del modulo api.py uv run fastapi dev src/iris_mlops/api.py

--Levantar la interfaz uv run gradio_app.py

--Crear los tests unitarios uv add --dev pytest httpx2

--Correr test unitarios (-v: modo verboso) uv run pytest -v

--Crear inputs para validar más rápido (alias) --uv run preparar-datos

-- https://docs.github.com/en/actions