"""
Task B4, phần lấy giao dịch thô cho account hoàn toàn mới: dùng Etherscan REST
API trực tiếp qua `requests`, KHÔNG dùng package `etherscan` mà LMAE4Eth/etherscan.py
phụ thuộc -- package đó (PyPI `etherscan`) không có trong môi trường này và không
tự cài được (không phải lỗi mạng, package thật sự không tồn tại dưới tên đó trên
PyPI lúc kiểm tra). Vẫn tái dùng action name "txlist" từ
LMAE4Eth/Data/actions_enum.py (ActionsEnum.TXLIST) để nhất quán ý tưởng tham khảo,
nhưng gọi thẳng endpoint REST thay vì qua wrapper.

CHƯA TỰ TEST ĐƯỢC TRONG SANDBOX NÀY vì thiếu ETHERSCAN_API_KEY -- NHƯNG mạng ra
ngoài thật sự DÙNG ĐƯỢC (đã tự sửa lại 1 giả định sai trong lúc rà soát: từng
tưởng sandbox không có mạng, kiểm tra lại bằng `requests.get` thật thì có --
xem STATUS.md mục rà soát). Endpoint V1 (`api.etherscan.io/api`) đã bị Etherscan
deprecate (test thật: trả về lỗi "deprecated V1 endpoint"), phải dùng V2
(`api.etherscan.io/v2/api`, cần thêm `chainid`) -- đã sửa và verify V2 nhận
request đúng format (trả "Invalid API Key" thay vì lỗi deprecated, tức là parse
đúng, chỉ thiếu key thật). Vẫn cần API key thật + 1 địa chỉ đối chiếu thủ công
trên etherscan.io trước khi tin dùng production -- xem TODO ở cuối file.
"""
import os
import time
from typing import List

import requests

from model.new_account_features import RawTx

ETHERSCAN_API_URL = "https://api.etherscan.io/v2/api"
ETH_MAINNET_CHAIN_ID = 1


def fetch_raw_transactions(address: str, api_key: str = None, max_retries: int = 3) -> List[RawTx]:
    """Lấy toàn bộ giao dịch ETH thường (txlist) của 1 địa chỉ. Trả về list rỗng
    nếu địa chỉ chưa từng giao dịch (account thật sự mới tinh, chưa deploy/nhận
    ETH bao giờ) -- predict_account.py phải xử lý được trường hợp này (subgraph
    chỉ có đúng 1 node cô lập, confidence thấp nhất)."""
    api_key = api_key or os.environ.get("ETHERSCAN_API_KEY")
    if not api_key:
        raise RuntimeError(
            "ETHERSCAN_API_KEY chưa được set -- cần API key thật (miễn phí, đăng ký "
            "tại etherscan.io) để gọi B4 cho account mới. Không có key thì "
            "predict_account() chỉ dùng được cho Case A (account đã có trong graph)."
        )

    params = {
        "chainid": ETH_MAINNET_CHAIN_ID,
        "module": "account",
        "action": "txlist",  # == LMAE4Eth Data/actions_enum.py ActionsEnum.TXLIST
        "address": address,
        "startblock": 0,
        "endblock": 99999999,
        "sort": "asc",
        "apikey": api_key,
    }

    last_err = None
    for attempt in range(max_retries):
        try:
            resp = requests.get(ETHERSCAN_API_URL, params=params, headers={"User-Agent": ""}, timeout=15)
            resp.raise_for_status()
            payload = resp.json()
            break
        except (requests.RequestException, ValueError) as e:
            last_err = e
            time.sleep(0.5 * (attempt + 1))
    else:
        raise RuntimeError(f"Etherscan txlist request failed after {max_retries} attempts: {last_err}")

    if payload.get("status") == "0" and payload.get("message") != "No transactions found":
        raise RuntimeError(f"Etherscan API error for {address}: {payload.get('result')}")

    txs = []
    for tx in payload.get("result", []):
        txs.append(
            RawTx(
                from_addr=tx["from"],
                to_addr=tx["to"],
                value_eth=int(tx["value"]) / 1e18,  # wei -> ETH
                timestamp=int(tx["timeStamp"]),
            )
        )
    return txs


# TODO (chưa làm, cần API key + mạng thật để kiểm chứng):
#   1. Chạy fetch_raw_transactions() với 1 địa chỉ thật đã biết trước số lượng
#      giao dịch (đối chiếu thủ công trên etherscan.io) để xác nhận parsing đúng.
#   2. Etherscan giới hạn 10,000 tx/request (phân trang bằng page/offset) --
#      address có > 10k tx (sàn giao dịch, hợp đồng lớn) sẽ bị cắt, cần thêm
#      vòng lặp phân trang nếu B4 phải xử lý các địa chỉ dạng này.
#   3. txlist chỉ trả giao dịch ETH thường -- nếu cần cả internal tx/token
#      transfer để khớp đúng định nghĩa cạnh gốc trong edges.npz, phải gọi thêm
#      action="txlistinternal" (cũng có sẵn tên trong actions_enum.py).
