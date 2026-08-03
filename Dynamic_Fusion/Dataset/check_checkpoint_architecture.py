"""
check_checkpoint_architecture.py
---------------------------------
Xac dinh mot checkpoint .pt cua ETH_GBert la BI-MODAL (BERT+GCN, ban goc)
hay TRI-MODAL (BERT+GCN+Feature, ban co FeatureProjector + 4-gate
DynamicFusionLayer), chi bang cach doc shape cac tensor trong state_dict --
KHONG can chay lai model, khong can GPU.

Cach dung:
    python3 check_checkpoint_architecture.py /duong/dan/toi/ETH_GBert16_model_Dataset_MG_cle_sw0_vocab5400.pt

Chi can torch da cai (`pip install torch --break-system-packages` neu chua co).
"""

import sys
import torch


def diagnose(state_dict: dict) -> None:
    keys = list(state_dict.keys())

    feature_keys = [k for k in keys if "feature_projector" in k]
    fusion_keys = [k for k in keys if "dynamic_fusion_layer" in k]
    gcn_keys = [k for k in keys if "vocab_gcn" in k]

    print(f"Tong so tensor trong state_dict: {len(keys)}")
    print(f"  - Tensor thuoc vocab_gcn (GCN stream):        {len(gcn_keys)}")
    print(f"  - Tensor thuoc feature_projector:              {len(feature_keys)}")
    print(f"  - Tensor thuoc dynamic_fusion_layer:            {len(fusion_keys)}")
    print()

    if not fusion_keys:
        print("[KHONG XAC DINH DUOC] Khong tim thay 'dynamic_fusion_layer' trong "
              "checkpoint nay. Day co the la mot kien truc rat khac ban dang xem, "
              "hoac ten module da doi. Kiem tra thu cong danh sach key ben duoi.")
        print("\nMot vai key mau trong checkpoint:")
        for k in keys[:15]:
            print(" ", k, tuple(state_dict[k].shape))
        return

    # --- (A) Su co mat cua feature_projector la dau hieu manh nhat ---
    has_feature_stream = len(feature_keys) > 0

    # --- (B) Kiem tra kich thuoc dau vao cua gate_network (Linear dau tien) ---
    gate_in_key = None
    for k in fusion_keys:
        if k.endswith("gate_network.0.weight"):
            gate_in_key = k
            break

    gate_out_key = None
    for k in fusion_keys:
        if k.endswith("gate_network.2.weight"):
            gate_out_key = k
            break

    hidden_size = None
    num_gates = None
    if gate_in_key is not None:
        w = state_dict[gate_in_key]  # shape [hidden_dim, in_features]
        in_features = w.shape[1]
        hidden_size = w.shape[0]
        # in_features = hidden_dim * (2 hoac 3) tuy phien ban
        ratio = in_features / w.shape[0] if w.shape[0] else None
        print(f"gate_network.0.weight shape = {tuple(w.shape)}  "
              f"(in_features={in_features}, hidden_dim_of_gate={w.shape[0]})")
        print(f"  -> ty le in_features / hidden_dim_of_gate = {ratio:.2f} "
              f"(2.00 ~ bi-modal [BERT,GCN] | 3.00 ~ tri-modal [BERT,GCN,Feature])")

    if gate_out_key is not None:
        w2 = state_dict[gate_out_key]  # shape [num_gates, hidden_dim]
        num_gates = w2.shape[0]
        print(f"gate_network.2.weight shape = {tuple(w2.shape)}  "
              f"-> so nhanh gate dau ra = {num_gates} "
              f"(3 ~ ban goc/bi-modal [bert,gcn,mixed] | 4 ~ tri-modal [bert,gcn,feat,mixed])")

    print()
    print("=" * 70)
    if has_feature_stream and (num_gates == 4 or (gate_in_key and ratio and ratio >= 2.9)):
        print("KET LUAN: Checkpoint nay la TRI-MODAL "
              "(co feature_projector + gate 4 nhanh).")
        print("  -> Neu day la checkpoint cua large-bounded run, thi ket qua")
        print("     F1(pos) pure_test=0.7753 / overlap=0.2609 LA cua tri-modal.")
    elif not has_feature_stream and (num_gates == 3 or (gate_in_key and ratio and ratio < 2.5)):
        print("KET LUAN: Checkpoint nay la BI-MODAL "
              "(chi BERT+GCN, KHONG co feature_projector).")
        print("  -> Neu day la checkpoint cua large-bounded run, thi ket qua")
        print("     F1(pos) pure_test=0.7753 / overlap=0.2609 LA CUA BI-MODAL,")
        print("     KHONG PHAI cua kien truc tri-modal -- can chay lai de co so lieu")
        print("     that su cho tri-modal truoc khi dua vao bao cao/slide.")
    else:
        print("KET LUAN: Tin hieu chua nhat quan (co the la mot bien the khac).")
        print("  -> Doi chieu thu cong: has_feature_stream =", has_feature_stream,
              ", num_gates =", num_gates)

    print("=" * 70)
    print("\nToan bo key thuoc dynamic_fusion_layer / feature_projector:")
    for k in fusion_keys + feature_keys:
        print(" ", k, tuple(state_dict[k].shape))


def main():
    if len(sys.argv) != 2:
        print("Dung: python3 check_checkpoint_architecture.py <duong_dan_checkpoint.pt>")
        sys.exit(1)

    path = sys.argv[1]
    print(f"Dang doc checkpoint: {path}\n")
    ckpt = torch.load(path, map_location="cpu")

    # metadata truoc (giong nhu print trong train1.py khi luu)
    for meta_key in ("epoch", "perform_metrics", "recall", "precision", "valid_acc"):
        if meta_key in ckpt:
            print(f"  {meta_key}: {ckpt[meta_key]}")
    print()

    state_dict = ckpt["model_state"] if "model_state" in ckpt else ckpt
    diagnose(state_dict)


if __name__ == "__main__":
    main()