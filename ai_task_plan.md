# AI\_TASKS\_PLAN.md

## Objetivo

Crear una plataforma web completa para entrenar, gestionar y utilizar modelos de machine learning aplicados a sistemas HVAC. El desarrollo debe dividirse en tareas específicas para backend, frontend y configuración general.

---

## ✅ Tareas Generales

### TAREA 1: Clonar el repositorio

* Si no existe, crear un repositorio nuevo vacío.
* Clonar el repositorio localmente.

```bash
git clone [repository-url]
cd [repository-name]
```

---

## ⚙️ Backend

### TAREA 2: Estructurar carpetas backend

Crear esta estructura:

```
backend/
├── app/
│   ├── __init__.py
│   ├── models/
│   │   ├── __init__.py
│   │   └── model_manager.py
│   ├── routes/
│   │   ├── __init__.py
│   │   ├── training.py
│   │   └── prediction.py
│   └── utils/
│       ├── __init__.py
│       └── data_processor.py
├── config.py
└── requirements.txt
```

### TAREA 3: Crear `requirements.txt`

Agregar las dependencias:

```txt
fastapi
uvicorn
pandas
numpy
scikit-learn
tensorflow
python-dotenv
```

### TAREA 4: Crear `config.py`

Configurar rutas básicas de modelos y datos:

```python
import os

MODEL_SAVE_PATH = os.getenv("MODEL_SAVE_PATH", "./models/saved_models")
DATA_PATH = os.getenv("DATA_PATH", "./data")
```

### TAREA 5: Crear `data_processor.py`

* Funciones para leer CSV
* Separar variables input/output
* Normalización de datos

### TAREA 6: Crear `model_manager.py`

* Definir arquitectura del modelo
* Funciones de entrenamiento
* Guardar y cargar modelos

### TAREA 7: Crear `training.py`

* Ruta `POST /api/train`
* Ruta `GET /api/variables`
* Ruta `POST /api/models/save`
* Ruta `GET /api/models`

### TAREA 8: Crear `prediction.py`

* Ruta `POST /api/predict`
* Ruta `GET /api/models/{model_id}`
* Ruta `POST /api/models/{model_id}/predict`

### TAREA 9: Crear `main.py` en `app/`

Registrar rutas y lanzar FastAPI:

```python
from fastapi import FastAPI
from app.routes import training, prediction

app = FastAPI()
app.include_router(training.router)
app.include_router(prediction.router)
```

---

## 🧪 Datos y Modelos

### TAREA 10: Crear estructura de carpetas

```
models/
└── saved_models/

data/
└── data_limpia.csv
```

---

## 🌐 Frontend

### TAREA 11: Inicializar proyecto React

```bash
cd frontend
npx create-react-app .
npm install @mui/material @emotion/react @emotion/styled chart.js axios
```

### TAREA 12: Crear estructura de carpetas

```
frontend/
├── public/
├── src/
│   ├── components/
│   │   ├── Training/
│   │   └── Prediction/
│   ├── App.js
│   └── index.js
```

### TAREA 13: Crear componentes de Training

* Subida de datos CSV
* Selección de variables
* Configuración de hiperparámetros
* Botón de entrenamiento
* Visualización de métricas

### TAREA 14: Crear componentes de Prediction

* Cargar modelo
* Ingresar datos
* Ver resultados y exportar

### TAREA 15: Crear integración con API (Axios)

* Crear instancias de `axios`
* Funciones para consumir endpoints

---

## ⚙️ Configuración

### TAREA 16: Crear `.env` en backend

```env
MODEL_SAVE_PATH=./models/saved_models
DATA_PATH=./data
```

### TAREA 17: Crear script de inicio

* Backend: `uvicorn app.main:app --reload`
* Frontend: `npm start`

---

## ✅ Testing Final

### TAREA 18: Validar funcionalidades

* Cargar CSV y entrenar modelo
* Guardar modelo entrenado
* Usar modelo para predecir
* Visualización en frontend

---

## 🔁 DevOps (Opcional)

### TAREA 19: Crear archivo `.gitignore`

```txt
venv/
__pycache__/
.env
node_modules/
models/saved_models/
```

### TAREA 20: Crear instrucciones de despliegue (README.md)

* Instrucciones de instalación y ejecución
* Requisitos del sistema
