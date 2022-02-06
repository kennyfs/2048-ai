# 2048-ai  
試圖用Muzero的方法訓練一個2048的模型。  
參考來源:[別人的完整實作](https://github.com/werner-duvaud/muzero-general)、[Deep Mind 官方虛擬碼(偽代碼)](https://arxiv.org/src/1911.08265v2/anc/pseudocode.py)
## 進度  
可能已經做完[self_play.py](self_play.py)中的MCTS。  
總之先把目前進度上傳，仍需要很多努力。
## todo  
- [x] 完成[self_play.py](self_play.py)中的SelfPlay  
- [x] [environment.py](environment.py)捨棄Game，全部改用Environment  
- [x] 訓練  
- [x] replay buffer重新加入priority  
- [x] shared storage(未確認引用位置是否寫得正確)  
- [x] Network應該要永遠輸出support版的value, reward，目前依照這個設定寫trainer
- [ ] 主要的流程控制  
    - [x] TensorBoard需要log的參數
		
## 理解
1. value target: bootstrap---value is expectant reward, so expectant score=value+reward in past
## 提醒自己
network.Network及他的subclass輸出的hidden state都是scale過的，在Predictor、Manager、AbstractNetwork中都可以忽略scale hidden state
## 作法
用pickle存：
    weights : ""
    {replay_buffer,num_played_games,num_played_steps,num_reanalysed_games}