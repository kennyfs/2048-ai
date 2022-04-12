# 2048-ai  
試圖用Muzero的方法訓練一個2048的模型。  
參考來源:[別人的完整實作](https://github.com/werner-duvaud/muzero-general)、[Deep Mind 官方虛擬碼(偽代碼)](https://arxiv.org/src/1911.08265v2/anc/pseudocode.py)
## 進度  
（待補）
## todo  
- [ ] 調整訓練參數讓輸出更理想  
- [ ] 運用多線程增進selfplay效率(似乎有點難)  
- [X] 隨機產生遊戲
- [X] 修[bug](#Bugs)，目前沒什麼bug
- [ ] log更多資訊（如：隨機取樣輸出、lr）
## 理解
1. value target: bootstrap---value is expectant reward, so expectant score=value+reward in past
## 提醒自己
network.Network及他的subclass輸出的hidden state都是scale過的，在Predictor、Manager、AbstractNetwork中都可以忽略scale hidden state
## 想法
shared storage中儲存網路不一定要和其他資訊一起儲存  
非必要不要用ray，因為ray似乎無法把有asyncio的東西當作函數的參數（無法pickle），畢竟我就只有一個GPU，讓inference排隊，一起預測才比較重要  
  
replay buffer、shared storage必須同時用ray或不用ray，否則如果replay buffer用ray、shared storage是普通型態的參數，replay buffer會copy一份shared storage，無法正確修改。  
解決辦法：replay buffer不要管shared storage中的資訊，要用的時候(結束完selfplay更新)
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