# 2048-ai  
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
改成試圖用類似AlphaZero的方法訓練一個2048的模型，因為Muzero的方法似乎並沒有必要。  
參考來源:[別人的完整實作](https://github.com/werner-duvaud/muzero-general)、[Deep Mind 官方虛擬碼(偽代碼)](https://arxiv.org/src/1911.08265v2/anc/pseudocode.py)、[陽明交大論文](https://hdl.handle.net/11296/amnm56)
## 進度  
一開始的想法（bf33080以前）：  
把action分成兩個type，加入磚塊和移動，並且在game history都紀錄，在dynamics network中也是把兩種action視為一樣的一起做。
  
後來覺得隨機的action在搜尋時會很大程度的限制搜尋深度，嘗試把兩種action合併看待，只給神經網路移動的action，因為隨機的改變他應該也能學到，但後來發現[陽明交大論文](https://hdl.handle.net/11296/amnm56)，新增chance網路確實有必要。  
  
5a47818時發現optimizer影響很大，SGD就是始終無法fit，調高學習率反而會讓loss無法下降，Adam真的好用很多，不過反而要注意過擬和。  
  
**目前要打算大幅度重寫，比較不在意搜尋深度和訓練效率（不要硬是批次處理，簡化程式）**  
**未來規劃是先想好架構、把要做的事列出來再做**  
我覺得可以放棄MCTS時用多線程搜尋，因為多個線程存取同一棵樹會牽扯到很多麻煩的問題。  
因此可以一次使用多個MCTS，但共用一個神經網路，這樣可以加快推論速度。就像KataGo一樣。  
## todo
## 理解
1. value target: bootstrap---value is expectant reward, so expectant score=value+reward in past
## 提醒自己
network.Network及他的subclass輸出的hidden state都是scale過的，在Predictor、Manager、AbstractNetwork中都可以忽略scale hidden state
## 想法
參考[Katago](https://github.com/lightvector/KataGo)的架構
## Bugs
### bug1
偶爾一局遊戲結束時會停住，不知道發生什麼問題，然後很久很久之後出現這樣的錯誤訊息：  
[2022-03-31 22:18:36,289 E 28676 28964] gcs_server_address_updater.cc:76: Failed to receive the GCS address for 600 times without success. The worker will exit ungracefully. It is because GCS has died. It could be because there was an issue that kills GCS, such as high memory usage triggering OOM killer to kill GCS. Cluster will be highly likely unavailable if you see this log. Please check the log from gcs_server.err.
以上的bug應該是ray造成的，現在應該不會。  

### bug2
```
visits:[0, 21, 62, 17]
Played action: Left
  2|128|  8|  4|
----------------
 16| 64| 16|  8|
----------------
  8|  2| 32|  2|
----------------
  4| 16|  8|  4|
----------------
score= 1372
game length:143
flag play_game2
flag play_game3
flag self_play2
save_game1
save_game2
save_game3
save_game4
flag self_play3
done playing
Traceback (most recent call last):
  File "/home/kenny/桌面/tmpfs/2048-ai/main.py", line 470, in <module>
    muzero.train()
  File "/home/kenny/桌面/tmpfs/2048-ai/main.py", line 186, in train
    self.training_worker.run_update_weights(
  File "/home/kenny/桌面/tmpfs/2048-ai/trainer.py", line 51, in run_update_weights
    next_batch = replay_buffer.get_batch()
  File "/home/kenny/桌面/tmpfs/2048-ai/replay_buffer.py", line 94, in get_batch
    values, rewards, policies, actions = self.make_target(
  File "/home/kenny/桌面/tmpfs/2048-ai/replay_buffer.py", line 289, in make_target
    return 	(np.array(target_values,dtype=np.float32),
ValueError: setting an array element with a sequence. The requested array has an inhomogeneous shape after 1 dimensions. The detected shape was (11,) + inhomogeneous part.
```
以上的bug是因為compute target value在network改動後沒有改正，已修正
### bug3
reanalyze後的遊戲在產生數據時會需要很長的時間，是tolist、list的差別，已修正
### bug4
數據不正常，是因為合併action後，load game時沒有把add action也加入，所以reward都是0，已修正