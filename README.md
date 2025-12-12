# Movie Recommendation System - Production Deployment

A unified web application combining **NextItNet** (Sequential Deep Learning), **BiVAE** (Generative Collaborative Filtering), and **LLM-based** (Gemini AI) movie recommendations with role-based access control.

## Quick Start

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Configure Environment

Create `.env` file with your OpenRouter API key:

```env
OPENROUTER_API_KEY=your_api_key_here
MODEL=google/gemini-2.0-flash-001
```

### 3. Run the Application

```bash
python app.py
```

Access at: **http://localhost:5000**

**Note**: All necessary data and model files are already included in the `data/` and `models/` folders.

______________________________________________________________________

## User Roles

### **User** (Regular User)

- Browse and search 9,067 movies
- Build watch history
- Get personalized recommendations
- Access: User Interface tab only

### **Admin** (Administrator)

- All User features
- View system statistics
- Manage users and movies
- Access: User Interface + Admin View tabs

### **Data Scientist** (ML Engineer)

- All User features
- View/switch active model (NextItNet ↔ LLM)
- Monitor model performance metrics
- Train new models via web interface
- Access: All tabs (User Interface, Admin View, Data Scientist)

______________________________________________________________________

## Features

### Core Functionality

- **9,067 movies** from MovieLens dataset (100% vocabulary coverage)
- Triple recommendation engines:
  - **NextItNet**: Sequential recommendation (Session-based)
  - **BiVAE**: Generative Collaborative Filtering (Matrix Factorization based)
  - **LLM**: Content-based reasoning (Cold-start friendly)
- **Real-time recommendations** with fallback for cold-start users
- **Search with autocomplete**
- **Persistent watch history** (in-memory sessions)

### Data Scientist Tools

- **Dashboard**: Stats, active models table, production metrics, training form, comparison charts
- **Model Lab**: Detailed metrics for each model, expandable sections, action buttons
- **Live Training**: Start model training from web UI with progress monitoring
- **Model Switching**: Change active recommendation engine in real-time

### Technical

- **FastAPI** backend with async endpoints
- **PyTorch** for NextItNet model inference
- **Cornac** for BiVAE training and inference
- **OpenRouter/Gemini** for LLM recommendations
- **Role-based access control** with session storage
- **Responsive UI** with Tailwind CSS + Chart.js

______________________________________________________________________

## Architecture

```
production/
├── app.py                     # FastAPI server
├── config.py                  # Global configuration
├── local_config.py            # Model/data paths
├── data_manager.py            # Data handling
├── recommender_nextitnet.py   # NextItNet recommender
├── recommender_bivae.py       # BiVAE wrapper (Cornac)
├── recommender_llm.py         # LLM recommender
├── train_eval_bivae.py        # BiVAE training & evaluation script
├── training_manager.py        # Training job management
├── requirements.txt           # Python dependencies
├── .env                       # API keys (create manually)
├── data/                      # Datasets (included)
│   ├── vocab.pkl
│   ├── train.pkl
│   ├── val.pkl
│   ├── test.pkl
│   └── movies_metadata.csv
├── models/                    # Trained models
│   └── best_model.pth
    └── bivae_context/         # BiVAE saved model
│       └── BiVAECF/           # Cornac internal folder
│           └── 2025-12-12....pkl
├── static/                    # Frontend assets
│   ├── css/style.css
│   └── js/app.js
└── templates/                 # HTML templates
    ├── index.html
    └── login.html
```

______________________________________________________________________

## API Endpoints

### User Endpoints

- `POST /api/history/{user_id}` - Add movie to history
- `GET /api/history/{user_id}` - Get watch history
- `DELETE /api/history/{user_id}` - Clear history
- `GET /api/recommendations/{user_id}` - Get recommendations
- `GET /api/movies` - List all movies
- `GET /api/movies/search` - Search movies

### Admin Endpoints

- `GET /api/admin/stats` - System statistics
- `GET /api/admin/movies/all` - All movies (paginated)e
- `GET /api/admin/users` - All user sessions

### Data Scientist Endpoints

- `POST /api/model/switch` - Switch active model
- `POST /api/training/start` - Start training job
- `GET /api/training/status/{job_id}` - Get training progress
- `GET /api/training/jobs` - List all training jobs
- `DELETE /api/training/{job_id}` - Cancel training

______________________________________________________________________

## Model Details

### NextItNet v3.0

- **Architecture**: Dilated causal convolutions
- **Parameters**: 1,195,945
- **Vocabulary**: 9,067 items (9,066 movies + padding)
- **Performance**: NDCG@10 = 0.4105, HR@10 = 0.6620

### LLM Recommender v1.5

- **Provider**: OpenRouter (Gemini 2.5 Flash)
- **Type**: Content-based using AI
- **Latency**: ~1-2 seconds per request
- **Best for**: Cold-start users, content similarity

### BiVAE v1.0

- Type: Generative Collaborative Filtering
- Library: Cornac
- Architecture: Bayesian Generative Model with Variational Inference
- Input: User-Item Interaction Matrix
- Performance (Test Set):
- NDCG@10: 0.0557
- Recall@10: 0.0183
- MRR: 0.1323
- Training Time: ~738s (12 mins)

______________________________________________________________________

## Training New Models

1. Login as **Data Scientist**
1. Go to **Dashboard** tab
1. Fill **"Start New Training"** form
1. Monitor progress in browser console (F12)
1. New model saved after completion
1. Update `local_config.py` with new model path
1. Restart server to load new model

#### Train BiVAE

```python
python train_eval_bivae.py
```

- Trains the BiVAE model using Cornac.
- Evaluates on Test set.
- Saves the model to models/bivae_context/
  Run the evaluation and training script to generate a new model:

**Training Time**: 30 epochs ≈ 15-25 min (GPU), 1-2 hrs (CPU)

______________________________________________________________________

## Troubleshooting

### Server won't start

- Verify all data files exist in `data/` folder
- Check Python 3.8+ installed
- Install dependencies: `pip install -r requirements.txt`

### No recommendations

- Ensure at least 1 movie in watch history
- Check active model is "Ready" (shown in header)
- If only non-vocabulary movies watched → fallback recommendations shown

### Training fails

- Verify `data/train.pkl` and `data/val.pkl` exist
- Check GPU availability if "Use GPU" enabled
- Reduce batch_size if out of memory

______________________________________________________________________

## Requirements

- Python 3.8+
- PyTorch (CUDA recommended for training)
- FastAPI + Uvicorn
- Pandas, NumPy
- OpenRouter API key (for LLM features)

See `requirements.txt` for complete list.

______________________________________________________________________

## Credits

- **Dataset**: MovieLens 100K + TMDB metadata
- **Model**: NextItNet (dilated causal convolutions), BiVAE (Cornac)
- **LLM**: OpenRouter/Gemini 2.5 Flash
- **Framework**: FastAPI + PyTorch

**Developed for Intelligent Systems Course - Movie Recommendation Project**
