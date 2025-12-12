import pickle
import os
import cornac
from local_config import DATA_DIR, VOCAB_PATH


def load_and_prep_data():
    print("🔄 Đang load dữ liệu từ hệ thống cũ...")

    # 1. Load Vocabulary (Để đảm bảo ID khớp nhau)
    # Chúng ta cần item2idx để biết hệ thống hiện tại map MovieID nào sang Index nào
    with open(VOCAB_PATH, "rb") as f:
        vocab = pickle.load(f)

    # idx2item: map từ Internal Index (0,1,2...) -> Movie ID gốc (1, 2, 94...)
    # item2idx: map từ Movie ID gốc -> Internal Index
    # idx2item = vocab["idx2item"]
    # item2idx = vocab["item2idx"]
    num_items = vocab["num_items"]

    print(f"✅ Đã load Vocab: {num_items} items.")

    # 2. Load Training Data (Dữ liệu chuỗi NextItNet)
    train_path = os.path.join(DATA_DIR, "train.pkl")
    with open(train_path, "rb") as f:
        train_data = pickle.load(f)

    # train_data có dạng {'input_seqs': [[1, 2], ...], 'target_items': [3, ...]}
    # Chúng ta cần chuyển nó thành list các bộ ba (User_ID, Item_Index, Rating)
    # Vì NextItNet không lưu UserID trong file pkl (nó chỉ lưu chuỗi),
    # ta sẽ giả định mỗi chuỗi tương ứng với một User Index ảo hoặc lấy từ session.
    # ĐỂ ĐƠN GIẢN VÀ HIỆU QUẢ CHO BIVAE:
    # Ta coi mỗi dòng trong input_seqs là một user ẩn danh.

    print("🔄 Đang chuyển đổi dữ liệu chuỗi sang dạng User-Item...")

    uir_tuples = []
    # Set để tránh duplicate (User A xem phim B nhiều lần chỉ tính là 1 tương tác tích cực)
    seen_interactions = set()

    for user_idx, seq in enumerate(train_data["input_seqs"]):
        # Lấy target item (phim tiếp theo user đã xem)
        target = train_data["target_items"][user_idx]

        # Thêm các phim trong lịch sử
        for item_idx in seq:
            if item_idx != 0:  # Bỏ qua padding (số 0)
                if (user_idx, item_idx) not in seen_interactions:
                    uir_tuples.append((str(user_idx), item_idx, 1.0))
                    seen_interactions.add((user_idx, item_idx))

        # Thêm target item (cũng là phim user đã xem/thích)
        if (user_idx, target) not in seen_interactions:
            uir_tuples.append((str(user_idx), target, 1.0))
            seen_interactions.add((user_idx, target))

    print(f"✅ Đã tạo {len(uir_tuples)} tương tác (User-Item).")

    # # 3. Tạo Cornac Dataset với Global Item IDs cố định
    # # Đây là bước QUAN TRỌNG NHẤT: Ép Cornac dùng không gian ID giống hệt NextItNet

    # # Tạo danh sách tất cả item indices có thể có (từ 0 đến num_items - 1)
    # # Điều này đảm bảo ma trận của Cornac sẽ có kích thước chính xác như NextItNet
    # all_item_indices = list(range(num_items))

    # # Cornac Dataset
    dataset = cornac.data.Dataset.from_uir(
        data=uir_tuples,
        seed=42,
        # Ép buộc dùng danh sách item này, không cho Cornac tự sinh ID mới
        # item_ids=all_item_indices
        # user_set=user_set,
        # item_set=item_set,
    )

    print(
        f"✅ Cornac Dataset Info: Users={dataset.num_users}, Items={dataset.num_items}"
    )

    print(
        f"✅ Cornac Dataset Info: Users={dataset.num_users}, Items={dataset.num_items}"
    )

    if dataset.num_items != num_items:
        print(
            f"⚠️ CẢNH BÁO: Số lượng item không khớp! (Cornac: {dataset.num_items} vs Vocab: {num_items})"
        )
        # Nếu lệch nhẹ do padding index 0, có thể chấp nhận được, nhưng cần lưu ý.

    return dataset


if __name__ == "__main__":
    load_and_prep_data()
