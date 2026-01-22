# Custom Algorithm Plugins

此目錄用於存放使用者自定義或新導入的演算法。

## 如何新增演算法 (How to Add)

1. 在此目錄下建立一個新的 `.py` 檔案 (例如 `my_algo.py`)。
2. 定義函數，名稱必須以 `op_` 開頭。
3. 函數簽名必須符合以下標準：

```python
import numpy as np
from typing import Dict, Optional

def op_my_algorithm_name(img: np.ndarray, params: Dict, debug: bool, context: Optional[Dict] = None) -> np.ndarray:
    """
    這裡寫演算法的說明文件。
    """
    # 1. 讀取參數 (params 來自 tech_lib.json)
    threshold = int(params.get('threshold', {}).get('default', 50))
    
    # 2. 處理影像
    # ... your logic here ...
    
    # 3. 回傳 BGR 影像
    return result_img
```

4. (目前版本) 手動在 `processor.py` 的 `operation_map` 中註冊它。
5. (未來版本) 系統將自動掃描此目錄並註冊。

## 注意事項

- 輸入 `img` 可能是 BGR 或 Grayscale，請使用 `ensure_gray` 或 `ensure_bgr` 輔助函數。
- 若演算法需要跨幀狀態 (如追蹤)，請使用 `context` 字典。
