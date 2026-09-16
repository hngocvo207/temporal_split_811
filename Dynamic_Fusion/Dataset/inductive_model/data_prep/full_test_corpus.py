"""
Eval-full-pure_test (theo yêu cầu sau E2): dùng corpus 811,704 account
(pure_test 609,773 + overlap 201,931) build sẵn từ trước
(runs/expanded_test_attempt3/full_test_corpus/) -- không cần build gì thêm,
chỉ dùng để EVAL (không train), thay cho mẫu 5,000/609,773 quá nhỏ ở E2.

QUAN TRỌNG: file `test_y` gốc dùng isp_expanded.pkl (nhãn lan truyền 1-hop,
xem STATUS.md mục "phát hiện phụ" -- rủi ro circular cho nhánh graph). Dùng
`test_y_strict` thay thế (nhãn xác nhận độc lập, isp gốc) -- đã verify khớp
tuyệt đối 312 pure_test + 217 overlap dương, đúng canonical labels.pkl/
partition.pkl.

BERT_debug.md Task 1-2 (root-cause fix, xem STATUS.md muc "RA SOAT CODE"):
corpus gio luu raw_records (chua anonymize, khong self-address) thay vi text
da tokenize san. Render (dedup+anonymize+tokenize, data_prep.text_rendering)
xay ra voi seed CO DINH ("eval_full_test") -- eval chi duyet qua moi account
1 lan nen khong can bien thien qua "epoch" nhu train, nhung van bat buoc di
qua anonymize (khong bao gio de lo dia chi that cho BERT).

Cache ket qua render (text da tokenize, KHONG phai tensor -- nho hon nhieu,
xem RENDERED_CACHE_PATH) sau LAN CHAY DAU TIEN: render 811,704 document tu
raw_records moi lan mat vai phut (tokenize WordPiece song song), trong khi
train_e2_v2.py goi ham nay o MOI LAN KHOI DONG (de lay val subsample) -- neu
khong cache se cong them vai phut vao MOI lan chay script train, kho chiu khi
lap lai smoke-test/debug. Cache CHI dung khi max_examples=None (chay day du);
--max-examples (smoke-test) luon render truc tiep, khong dung/ghi cache.
"""
import pickle
import time
from pathlib import Path
from typing import List

import torch
from pytorch_pretrained_bert.tokenization import BertTokenizer

from data_prep.attempt3_corpus import MAX_SEQ_LEN, Example
from data_prep.labels_io import _load_addr_to_idx
from data_prep.text_rendering import encode_tokens, render_corpus

REPO_ROOT = Path(__file__).resolve().parents[3]
CORPUS_DIR = REPO_ROOT / "runs" / "expanded_test_attempt3" / "full_test_corpus"
RENDER_SEED = "eval_full_test"
RENDERED_CACHE_PATH = CORPUS_DIR / f"data_Dataset_MG_full_test.rendered_{RENDER_SEED}.pkl"


def _load_pickle(name: str):
    with open(CORPUS_DIR / f"data_Dataset_MG_full_test.{name}", "rb") as f:
        return pickle.load(f, encoding="latin1")


def load_full_test_examples(tokenizer: BertTokenizer = None, max_examples: int = None) -> List[Example]:
    tokenizer = tokenizer or BertTokenizer.from_pretrained("bert-base-uncased", do_lower_case=True)

    raw_records = _load_pickle("raw_records")  # dict address -> list[dict]
    doc_accounts = _load_pickle("doc_accounts")
    test_y = _load_pickle("test_y")  # propagated (isp_expanded) -- khop nhan corpus Attempt-3 dung cho E2
    test_y_strict = _load_pickle("test_y_strict")  # xac nhan doc lap (labels.pkl goc)
    test_partition = _load_pickle("test_partition")

    n = len(doc_accounts)
    assert n == len(test_y) == len(test_y_strict) == len(test_partition) == 811704

    global_a2i = _load_addr_to_idx()

    n_use = n if max_examples is None else min(max_examples, n)
    use_accounts = doc_accounts[:n_use]
    use_cache = max_examples is None and RENDERED_CACHE_PATH.exists()
    t0 = time.time()
    if use_cache:
        with open(RENDERED_CACHE_PATH, "rb") as f:
            rendered = pickle.load(f)
        print(f"  loaded {len(rendered):,} da-render tu cache {RENDERED_CACHE_PATH.name} ({time.time()-t0:.1f}s)")
    else:
        print(f"  rendering {len(use_accounts):,} document (dedup+anonymize+tokenize, seed={RENDER_SEED!r})...")
        rendered = render_corpus(raw_records, use_accounts, RENDER_SEED)
        print(f"  render xong ({time.time()-t0:.1f}s)")
        if max_examples is None:
            with open(RENDERED_CACHE_PATH, "wb") as f:
                pickle.dump(rendered, f, protocol=pickle.HIGHEST_PROTOCOL)
            print(f"  cached -> {RENDERED_CACHE_PATH.name} (lan chay sau se load thang, khong render lai)")
    encoded = {a: encode_tokens(rendered[a], tokenizer) for a in use_accounts}

    examples = []
    for i in range(n_use):
        raw_addr = doc_accounts[i]
        addr = raw_addr.lower()
        if addr not in global_a2i:
            raise ValueError(f"doc {i} address {addr} not in global address_to_index")
        input_ids, attention_mask = encoded[raw_addr]
        examples.append(Example(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=torch.zeros(MAX_SEQ_LEN, dtype=torch.long),
            label=int(test_y[i]),
            label_strict=int(test_y_strict[i]),
            global_idx=global_a2i[addr],
            address=addr,
            split=test_partition[i],  # 'pure_test' | 'overlap'
        ))
    return examples
