|          | 檔案             | 功用                                       | 要改進的                 |
| -------- | ---------------- | ------------------------------------------ | ------------------------ |
| 已有檔案 | environment.py   | 定義2048如何運作                           |                          |
|          | main.py          | 主要的程式，使用（訓練、測試等）時都執行它 |                          |
|          | cfg.py           | 處理config                                 | 常用的cfg要設為attribute |
|          | network.py       | 定義神經網路                               |                          |
|          | replaybuffer.py  | 處理資料，selfplay時存資料、訓練時取資料   |                          |
|          | selfplay.py      | selfplay的流程                             |                          |
|          | sharedstorage.py | 負責把資料寫入檔案                         |                          |
|          | trainer.py       | 進行訓練                                   |                          |
| 待新增   | gamehistory.py   | 從self_play.py分離gamehistory              |                          |
# 主要考量
搜尋時用單線程。但訓練時一次開很多個worker一起selfplay，共用Evaluator。