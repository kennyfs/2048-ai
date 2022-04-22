# 2048-ai  
試圖用Muzero的方法訓練一個2048的模型。  
參考來源:[別人的完整實作](https://github.com/werner-duvaud/muzero-general)、[Deep Mind 官方虛擬碼(偽代碼)](https://arxiv.org/src/1911.08265v2/anc/pseudocode.py)、[陽明交大論文](https://hdl.handle.net/11296/amnm56)
## 進度  
一開始的想法（bf33080以前）：  
把action分成兩個type，加入磚塊和移動，並且在game history都紀錄，在dynamics network中也是把兩種action視為一樣的一起做。
  
後來覺得隨機的action在搜尋時會很大程度的限制搜尋深度，嘗試把兩種action合併看待，只給神經網路移動的action，因為隨機的改變他應該也能學到，但後來發現[陽明交大論文](https://hdl.handle.net/11296/amnm56)，新增chance網路確實有必要。  
  
5a47818時發現optimizer影響很大，SGD就是始終無法fit，調高學習率反而會讓loss無法下降，Adam真的好用很多，不過反而要注意過擬和。  
  
目前正在做Squeeze-and-Excitation、把原本合併的action分開（僅selfplay的node，紀錄的gamehistory保持不變）
## todo  
- [X] 修[bug](#Bugs)，目前沒什麼bug
- [X] network中prediction worker要處理多輸入
- [X] MCTS
- [ ] 增進訓練效率（加入chance之後時間大約變成兩三倍，目前懷疑是loss_fn呼叫chance network跟recurrent inference都各自多轉換一次action）
- [ ] 調整訓練參數讓輸出更理想  
- [ ] 運用多線程增進selfplay效率(似乎有點難)  
## 理解
1. value target: bootstrap---value is expectant reward, so expectant score=value+reward in past
## 提醒自己
network.Network及他的subclass輸出的hidden state都是scale過的，在Predictor、Manager、AbstractNetwork中都可以忽略scale hidden state
## 想法
參考陽明交大論文，以及KataGo，因為chance network要輸出空白的分佈，所以我想到可以像katago輸出領土分佈一樣，訓練一個網路，輸入hidden_state輸出observation，新增一個loss讓網路更趨近於環境。但後來想到hidden_state其實不一定要包含所有盤面上的資訊，而是側重某些方面，加入這個loss可能反而會讓模型沒辦法好好預測value reward policy，所以如果要加loss應該要謹慎。  
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