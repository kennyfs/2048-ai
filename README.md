# 2048-ai  
試圖用Muzero的方法訓練一個2048的模型。  
參考來源:[別人的完整實作](https://github.com/werner-duvaud/muzero-general)、[Deep Mind 官方虛擬碼(偽代碼)](https://arxiv.org/src/1911.08265v2/anc/pseudocode.py)
## 進度  
幾乎完成，目前selfplay、replay buffer都應該沒問題，現在被一個奇怪的bug卡住，詳情見[bug輸出](#bug)
## todo  
- [ ] 整理my_config.py中的順序，讓人比較好理解  
- [ ] 確認資料處理沒問題（主要是replay buffer.py）  
- [ ] 完成ResNet實作  
- [ ] 完善checkpoint的儲存，main.py、network.py  
- [ ] 主要的流程控制(main.py)  
		
## 理解
1. value target: bootstrap---value is expectant reward, so expectant score=value+reward in past
## 提醒自己
network.Network及他的subclass輸出的hidden state都是scale過的，在Predictor、Manager、AbstractNetwork中都可以忽略scale hidden state
## 想法
shared storage中儲存網路不一定要和其他資訊一起儲存  
非必要不要用ray，因為ray似乎無法把有asyncio的東西當作函數的參數（無法pickle），畢竟我就只有一個GPU，讓inference排隊，一起預測才比較重要  
  
replay buffer、shared storage必須同時用ray或不用ray，否則如果replay buffer用ray、shared storage是普通型態的參數，replay buffer會copy一份shared storage，無法正確修改。  
解決辦法：replay buffer不要管shared storage中的資訊，要用的時候(結束完selfplay更新)
## bug
```
Traceback (most recent call last):
  File "/home/kenny/桌面/python3/2048-ai/main.py", line 436, in <module>
    muzero.train()
  File "/home/kenny/桌面/python3/2048-ai/main.py", line 175, in train
    self.training_worker.run_update_weights(
  File "/home/kenny/桌面/python3/2048-ai/trainer.py", line 70, in run_update_weights
    ) = self.update_weights(batch, shared_storage)
  File "/home/kenny/桌面/python3/2048-ai/trainer.py", line 171, in update_weights
    self.optimizer.minimize(loss_fn,self.model.trainable_variables)
  File "/home/kenny/.local/lib/python3.9/site-packages/keras/optimizer_v2/optimizer_v2.py", line 530, in minimize
    grads_and_vars = self._compute_gradients(
  File "/home/kenny/.local/lib/python3.9/site-packages/keras/optimizer_v2/optimizer_v2.py", line 574, in _compute_gradients
    loss = loss()
  File "/home/kenny/桌面/python3/2048-ai/trainer.py", line 125, in loss_fn
    _, _, _, hidden_state = self.model.recurrent_inference(
  File "/home/kenny/桌面/python3/2048-ai/network.py", line 151, in recurrent_inference
    hidden_state, reward=self.dynamics(hidden_state, action)
TypeError: dynamics() takes 2 positional arguments but 3 were given
```