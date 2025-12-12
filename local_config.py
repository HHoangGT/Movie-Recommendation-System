"""
Local Configuration for Production Deployment
All paths point to files within the production folder
"""

from pathlib import Path

# Base directory (production folder)
BASE_DIR = Path(__file__).parent

# Model paths (local to production folder)
MODELS_DIR = BASE_DIR / "models"
DATA_DIR = BASE_DIR / "data"

MODEL_PATH = MODELS_DIR / "best_model.pth"
VOCAB_PATH = DATA_DIR / "vocab.pkl"
MOVIES_METADATA_PATH = DATA_DIR / "movies_metadata.csv"

# Model configuration (from training)
# IMPORTANT: This must match the model weights in models/best_model.pth
# Updated for model retrained with ALL 9,066 movies (no filter)
MODEL_CONFIG = {
    "num_items": 9067,  # 9066 movies + 1 padding
    "embedding_dim": 128,
    "num_blocks": 6,
    "kernel_size": 3,
    "dilations": [1, 2, 4, 1, 2, 4],
    "dropout": 0.2,
    "max_seq_length": 50,
}

# Recommendation settings
DEFAULT_TOP_K = 10
MAX_HISTORY_LENGTH = 50
MIN_HISTORY_LENGTH = 1


# Verify files exist
def verify_setup():
    """Verify all required files exist."""
    required_files = [MODEL_PATH, VOCAB_PATH, MOVIES_METADATA_PATH]

    missing = []
    for file_path in required_files:
        if not file_path.exists():
            missing.append(str(file_path))

    if missing:
        print("❌ Missing required files:")
        for f in missing:
            print(f"   - {f}")
        print("\n⚠️  Please run 'python setup_files.py' first!")
        return False

    print("✅ All required files found")
    return True
