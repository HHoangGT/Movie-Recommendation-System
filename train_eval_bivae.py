import os
import pickle
import cornac
from cornac.eval_methods import BaseMethod
from cornac.metrics import NDCG, Recall, MRR
from cornac.models import BiVAECF
from local_config import DATA_DIR, VOCAB_PATH, MODELS_DIR

os.makedirs(MODELS_DIR, exist_ok=True)


def load_data_from_pkl(file_name):
    """Load dữ liệu từ file pickle của NextItNet"""
    path = os.path.join(DATA_DIR, file_name)
    with open(path, "rb") as f:
        data = pickle.load(f)
    return data


def convert_to_uir(data, vocab_items, is_test=False):
    """
    Chuyển đổi dữ liệu chuỗi NextItNet sang dạng User-Item-Rating (UIR).
    NextItNet format: {'input_seqs': [[1,2],...], 'target_items': [3,...]}
    """
    uir_tuples = []
    users = set()

    # Duyệt qua từng user (mỗi dòng là 1 user/session)
    for user_idx, seq in enumerate(data["input_seqs"]):
        user_id = str(user_idx)
        users.add(user_id)

        for item_idx in seq:
            if item_idx != 0:  # Bỏ padding
                uir_tuples.append((user_id, item_idx, 1.0))

        target = data["target_items"][user_idx]
        if target != 0:
            uir_tuples.append((user_id, target, 1.0))

    return uir_tuples


def run_experiment():
    print("=" * 60)
    print("🧪 BẮT ĐẦU HUẤN LUYỆN VÀ ĐÁNH GIÁ BiVAE")
    print("=" * 60)

    # 1. Load Vocabulary để đồng bộ ID
    print("[1/5] Loading Vocabulary...")
    with open(VOCAB_PATH, "rb") as f:
        vocab = pickle.load(f)
    num_items = vocab["num_items"]
    print(f"   - Total Items in Vocab: {num_items}")

    # 2. Load và Convert Data
    print("\n[2/5] Loading & Converting Data (NextItNet format -> UIR)...")
    train_data = load_data_from_pkl("train.pkl")
    val_data = load_data_from_pkl("val.pkl")
    test_data = load_data_from_pkl("test.pkl")

    train_uir = convert_to_uir(train_data, num_items)
    val_uir = convert_to_uir(val_data, num_items)
    test_uir = convert_to_uir(test_data, num_items)

    print(f"   - Train interactions: {len(train_uir)}")
    print(f"   - Val interactions:   {len(val_uir)}")
    print(f"   - Test interactions:  {len(test_uir)}")

    print("   🛠  Adding Dummy User to enforce full item space...")
    all_items = set(range(num_items))
    present_items = {i for u, i, r in train_uir}
    missing_items = all_items - present_items

    for item_idx in missing_items:
        train_uir.append(("-1", item_idx, 1.0))

    # 3. Setup Evaluation Method
    print("\n[3/5] Setting up Evaluation Method...")

    eval_method = BaseMethod.from_splits(
        train_data=train_uir,
        test_data=test_uir,
        val_data=val_uir,
        exclude_unknowns=False,
        verbose=True,
        seed=123,
        item_image=None,  # Không dùng hình ảnh
    )

    # 4. Setup Model & Metrics
    print("\n[4/5] Configuring Model & Metrics...")

    # BiVAE Configuration
    bivae = BiVAECF(
        k=50,  # Latent dimension
        encoder_structure=[100],  # Hidden layer size
        act_fn="tanh",  # Activation function
        likelihood="pois",  # Poisson likelihood (tốt cho implicit feedback)
        n_epochs=30,  # Số vòng lặp (có thể tăng lên 100 nếu cần)
        batch_size=128,
        learning_rate=0.001,
        seed=123,
        use_gpu=False,
        verbose=True,
    )

    # Metrics: NDCG@10, Recall@10 (HR@10)
    metrics = [NDCG(k=10), Recall(k=10), MRR()]

    # 5. Run Experiment
    print("\n[5/5] Running Experiment...")
    exp = cornac.Experiment(
        eval_method=eval_method,
        models=[bivae],
        metrics=metrics,
        user_based=True,
    )

    exp.run()

    # 6. Save Model (Quan trọng cho Production)
    print("\n💾 Saving trained model for Production...")
    save_path = os.path.join(MODELS_DIR, "bivae_context")
    bivae.save(save_path, save_trainset=True)
    print(f"   ✅ Model saved to: {save_path}")


if __name__ == "__main__":
    run_experiment()
