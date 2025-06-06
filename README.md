# HVAC Model Training and Prediction Platform

## Overview
This project provides a web-based platform for training, managing, and using machine learning models for HVAC (Heating, Ventilation, and Air Conditioning) system analysis. The platform allows users to train models on HVAC operational data, select input and output variables, save trained models, and make predictions using a user-friendly interface.

## Project Structure
```
├── backend/
│   ├── app/
│   │   ├── __init__.py
│   │   ├── models/
│   │   │   ├── __init__.py
│   │   │   └── model_manager.py
│   │   ├── routes/
│   │   │   ├── __init__.py
│   │   │   ├── training.py
│   │   │   └── prediction.py
│   │   └── utils/
│   │       ├── __init__.py
│   │       └── data_processor.py
│   ├── config.py
│   └── requirements.txt
├── frontend/
│   ├── public/
│   ├── src/
│   │   ├── components/
│   │   │   ├── Training/
│   │   │   └── Prediction/
│   │   ├── App.js
│   │   └── index.js
│   └── package.json
├── models/
│   └── saved_models/
├── data/
│   └── data_limpia.csv
└── README.md
```

## Features

### Training Interface
- Upload and preprocess HVAC operational data
- Select input and output variables from available columns
- Configure model hyperparameters
- Train and save models
- View training progress and metrics
- Export trained models

### Prediction Interface
- Load saved models
- Input variable selection
- Real-time predictions
- Visualization of results
- Export predictions

## Technical Requirements

### Backend
- Python 3.8+
- FastAPI
- TensorFlow 2.x
- Pandas
- NumPy
- scikit-learn

### Frontend
- React.js
- Material-UI
- Chart.js
- Axios

## Setup Instructions

1. Clone the repository
```bash
git clone [repository-url]
cd [repository-name]
```

2. Set up the backend
```bash
cd backend
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt
```

3. Set up the frontend
```bash
cd frontend
npm install
```

4. Configure environment variables
Create a `.env` file in the backend directory with:
```
MODEL_SAVE_PATH=./models/saved_models
DATA_PATH=./data
```

5. Run the application
```bash
# Terminal 1 (Backend)
cd backend
uvicorn app.main:app --reload

# Terminal 2 (Frontend)
cd frontend
npm start
```

## Data Format
The system expects CSV files with the following structure:
- Time-based features (hour, week, year)
- HVAC operational parameters
- Temperature readings
- Pressure readings
- Control signals

## Model Architecture
The default model architecture includes:
- Input layer with configurable dimensions
- Multiple dense layers with ReLU activation
- Output layer with linear activation
- Huber loss function
- Adam optimizer

## API Endpoints

### Training
- `POST /api/train`: Start model training
- `GET /api/models`: List available models
- `POST /api/models/save`: Save trained model
- `GET /api/variables`: Get available input/output variables

### Prediction
- `POST /api/predict`: Make predictions
- `GET /api/models/{model_id}`: Get model details
- `POST /api/models/{model_id}/predict`: Make predictions with specific model

## Contributing
1. Fork the repository
2. Create a feature branch
3. Commit your changes
4. Push to the branch
5. Create a Pull Request

## License
[Specify your license here]

## Contact
[Your contact information]

