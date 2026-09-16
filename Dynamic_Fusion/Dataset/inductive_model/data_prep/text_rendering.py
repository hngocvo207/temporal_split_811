"""
BERT_debug.md Task 1-3: render account documents cho BERT KHONG BAO GIO de lo
1 chuoi hex dia chi vi that lam token -- day la fix cho root cause khien E2 v2
sap AUPRC full-scale (0.0495 vs graph-only 0.303, xem STATUS.md muc "RA SOAT
CODE TIM NGUYEN NHAN").

Kien truc: `mg_build_examples.py`/`mg_build_full_test_eval.py` gio chi luu RAW
per-account transaction records (counterparty THAT + amount + in_out +
timestamp + n-gram, KHONG co self-address -- Task 1, da sua o mg_build_examples.py).
Dedup (Task 3), anonymize dia chi doi tac (Task 2) va tokenize WordPiece xay ra
O DAY, tai thoi diem train/eval -- KHONG con o thoi diem build corpus tinh --
de moi epoch sinh ra 1 phep gan addr0/addr1/... KHAC NHAU cho cung 1 document,
chan model hoc thuoc 1 chuoi sub-token co dinh (BERT_debug.md Task 2, yeu cau
"BAT BUOC, khong phai fallback").

Dien giai pham vi random hoa (khac 1 diem so voi chu Y nguyen van cua
BERT_debug.md, ghi ro o day de minh bach): BERT_debug.md noi "random hoa
lai... moi lan document duoc sampler draw ra"; tieu chi validation di kem
CHI doi chieu QUA 2 EPOCH LIEN TIEP (khong yeu cau bien thien giua cac lan
draw trong CUNG 1 epoch) -- nen o day render lai 1 LAN/EPOCH (khong phai
1 lan/draw rieng le) van dat dung tieu chi da neu, va tranh phai tokenize
dong trong DataLoader moi batch (phuc tap/rui ro hon nhieu, ton CPU moi step
thay vi moi epoch). Neu sau khi do dac thay van khong du (P@100 van sap),
day la noi dau tien can xiet lai thanh random-per-draw.
"""
import multiprocessing as mp
import random
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import torch
from pytorch_pretrained_bert.tokenization import BertTokenizer

_TRI_MODEL_DIR = str(Path(__file__).resolve().parents[2] / "tri_model")
if _TRI_MODEL_DIR not in sys.path:
    sys.path.insert(0, _TRI_MODEL_DIR)
from utils import anonymize_addresses, clean_str  # noqa: E402

MAX_SEQ_LEN = 400  # khop attempt3_corpus.py
DEDUP_WINDOW_SECONDS = 72 * 3600  # BERT_debug.md Task 3: window_hours=72


def dedup_consecutive(records: List[dict], window_seconds: float = DEDUP_WINDOW_SECONDS) -> List[dict]:
    """BERT_debug.md Task 3: gop giao dich LIEN TIEP (ke nhau trong chuoi da
    sap xep thoi gian) cung counterparty that + cung in_out, timestamp cach
    nhau <= window_seconds -- cong don amount, giu timestamp dau tien, them
    field 'count'. Chi xu ly "continuous repetitiveness" (dung y BERT4ETH,
    Hu et al. WWW'23) -- "discontinuous repetitiveness" (doi tac lap lai
    nhung khong ke nhau) van con sau buoc nay, duoc giam bang Task 2's
    random-hoa ID qua epoch (documented rieng, khong phai fallback cho day)."""
    if not records:
        return []
    out = [dict(records[0], count=1)]
    for r in records[1:]:
        last = out[-1]
        same_party = r["counterparty"] == last["counterparty"] and r["in_out"] == last["in_out"]
        close_in_time = abs(r["timestamp"] - last["timestamp"]) <= window_seconds
        if same_party and close_in_time:
            last["amount"] = last["amount"] + r["amount"]
            last["count"] += 1
        else:
            out.append(dict(r, count=1))
    return out


def repetitiveness_ratio(records: List[dict]) -> float:
    """Metric do luong theo dung cach BERT4ETH dung: % giao dich co cung
    counterparty voi giao dich LIEN TRUOC no (tren cung 1 tai khoan). Dung de
    so sanh truoc/sau dedup_consecutive() -- xem yeu cau 'Do luong' Task 3."""
    if len(records) < 2:
        return 0.0
    n_repeat = sum(
        1 for i in range(1, len(records))
        if records[i]["counterparty"] == records[i - 1]["counterparty"]
    )
    return n_repeat / (len(records) - 1)


def _template(records: List[dict]) -> str:
    """Task 1: KHONG con field dia chi chinh chu (da loai tu luc build raw
    record, xem mg_build_examples.py::raw_records_for) -- chi con counterparty
    (da anonymize truoc khi goi ham nay), amount, in_out, n-gram."""
    parts = []
    for r in records:
        parts.append(
            f"counterparty: {r['counterparty']} amount: {r['amount']} in_out: {r['in_out']} "
            f"2-gram: {r['2gram']:.0f} 3-gram: {r['3gram']:.0f} "
            f"4-gram: {r['4gram']:.0f} 5-gram: {r['5gram']:.0f}"
        )
    return "  ".join(parts) if parts else "no_transactions"


def render_document(records: List[dict], tokenizer: BertTokenizer, rng: random.Random) -> str:
    """1 document (list record cua 1 account) -> chuoi WordPiece da tokenize,
    cach nhau boi space (giu dung dinh dang cu de attempt3_corpus.py::_encode
    chi can .split() la dung -- khong doi hop dong voi phia doc). KHONG BAO
    GIO dua dia chi hex that vao: Task 1 da loai self-address tu luc build
    raw record; Task 2 (anonymize_addresses) o day thay counterparty that
    bang addrN cuc bo truoc khi rendered thanh text."""
    deduped = dedup_consecutive(records)
    anon = anonymize_addresses(deduped, rng)
    sentence = _template(anon)
    sub_words = tokenizer.tokenize(clean_str(sentence))
    return " ".join(sub_words) if sub_words else "[UNK]"


def encode_tokens(doc: str, tokenizer: BertTokenizer) -> Tuple[torch.Tensor, torch.Tensor]:
    """Giong het attempt3_corpus.py::_encode() (pad/truncate ve MAX_SEQ_LEN) --
    tach rieng o day vi input dau vao gio la chuoi da tokenize TU render_document(),
    khong phai tu 1 pickle tinh."""
    tokens_a = doc.split()
    if len(tokens_a) > MAX_SEQ_LEN - 2:
        tokens_a = tokens_a[: MAX_SEQ_LEN - 2]
    tokens = ["[CLS]"] + tokens_a + ["[SEP]"]
    ids = tokenizer.convert_tokens_to_ids(tokens)
    n = len(ids)
    pad = MAX_SEQ_LEN - n
    input_ids = torch.tensor(ids + [0] * pad, dtype=torch.long)
    attention_mask = torch.tensor([1] * n + [0] * pad, dtype=torch.long)
    return input_ids, attention_mask


_worker_tokenizer = None
_worker_seed = None


def _init_worker(seed: int):
    global _worker_tokenizer, _worker_seed
    _worker_tokenizer = BertTokenizer.from_pretrained("bert-base-uncased", do_lower_case=True)
    _worker_seed = seed


def _render_one(item) -> str:
    addr, records = item
    # rng rieng cho tung (seed, addr) -- deterministic theo cap nay nen co
    # the tai lap de debug, nhung KHAC NHAU giua cac addr (khong dung chung
    # 1 rng tuan tu -- se lam thu tu xu ly song song anh huong ket qua) va
    # KHAC NHAU giua cac seed (epoch) khac nhau cho CUNG 1 addr -- dung yeu
    # cau Task 2.
    rng = random.Random(f"{_worker_seed}:{addr}")
    return render_document(records, _worker_tokenizer, rng)


def render_corpus(raw_records_by_addr: Dict[str, list], addrs: List[str], seed,
                   n_workers: int = 16) -> Dict[str, str]:
    """Render (dedup + anonymize + template + tokenize) toan bo `addrs` voi 1
    `seed` cho truoc -- goi lai voi seed KHAC o moi epoch (vd f"train_epoch{e}")
    de thoa man Task 2 (validation: 2 epoch lien tiep sinh 2 phep gan ID khac
    nhau cho cung 1 document). Dung 1 seed CO DINH (vd "eval") cho corpus eval
    (val subsample / full_test) -- eval chi duyet qua 1 lan nen khong can bien
    thien, nhung van phai di qua anonymize (khong bao gio de lo dia chi that)."""
    n_workers = max(1, min(n_workers, mp.cpu_count() or 1))
    items = [(a, raw_records_by_addr[a]) for a in addrs]
    if n_workers == 1 or len(items) < 2000:
        _init_worker(seed)
        results = [_render_one(it) for it in items]
    else:
        with mp.Pool(n_workers, initializer=_init_worker, initargs=(seed,)) as pool:
            results = pool.map(_render_one, items, chunksize=256)
    return {a: r for a, r in zip(addrs, results)}


def render_and_encode_corpus(raw_records_by_addr: Dict[str, list], addrs: List[str], seed,
                              tokenizer: BertTokenizer = None,
                              n_workers: int = 16) -> Dict[str, Tuple[torch.Tensor, torch.Tensor]]:
    """render_corpus() + encode_tokens() gop lai -- tra ve (input_ids,
    attention_mask) san sang gan thang vao Example. Tokenizer instance rieng
    o process chinh chi dung cho encode_tokens() (convert_tokens_to_ids, rat
    re) -- viec tokenize thuc su (ton CPU) da lam song song trong render_corpus()."""
    tokenizer = tokenizer or BertTokenizer.from_pretrained("bert-base-uncased", do_lower_case=True)
    rendered = render_corpus(raw_records_by_addr, addrs, seed, n_workers=n_workers)
    return {a: encode_tokens(rendered[a], tokenizer) for a in addrs}


def render_examples_inplace(examples, seed, tokenizer: BertTokenizer = None, n_workers: int = 16) -> None:
    """Tien loi cho train_eval/train_e2_v2.py: render + encode truc tiep tren
    1 list Example (moi item can .address/.records da set san, xem
    data_prep/e2_train_corpus.py), GAN .input_ids/.attention_mask TAI CHO --
    khong can dung rieng 1 dict raw_records, tai su dung .records da luu tren
    tung Example. Goi lai moi epoch voi 1 `seed` KHAC (vd f"train_epoch{e}")
    de thoa Task 2 (anonymize ID doi tac doi khac qua epoch)."""
    tokenizer = tokenizer or BertTokenizer.from_pretrained("bert-base-uncased", do_lower_case=True)
    raw_by_addr = {e.address: e.records for e in examples}
    addrs = [e.address for e in examples]
    encoded = render_and_encode_corpus(raw_by_addr, addrs, seed, tokenizer=tokenizer, n_workers=n_workers)
    for e in examples:
        e.input_ids, e.attention_mask = encoded[e.address]
