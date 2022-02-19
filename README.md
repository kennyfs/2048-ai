# 2048-ai  
試圖用Muzero的方法訓練一個2048的模型。  
參考來源:[別人的完整實作](https://github.com/werner-duvaud/muzero-general)、[Deep Mind 官方虛擬碼(偽代碼)](https://arxiv.org/src/1911.08265v2/anc/pseudocode.py)
## 進度  
可能已經做完[self_play.py](self_play.py)中的MCTS。  
總之先把目前進度上傳，仍需要很多努力。
## todo  
- [ ] 除了shared storage、replay buffer都不要使用ray
    - [X] selfplay
    - [X] reanalyze
    - [ ] trainer
    - [ ] main中的log（因為沒有用ray，不會同步執行）
        - [ ] scalar
        - [ ] loss
- [ ] 整理myconfig中的順序，讓人比較好理解
- [ ] 主要的流程控制  
		
## 理解
1. value target: bootstrap---value is expectant reward, so expectant score=value+reward in past
## 提醒自己
network.Network及他的subclass輸出的hidden state都是scale過的，在Predictor、Manager、AbstractNetwork中都可以忽略scale hidden state
## 想法
shared storage中儲存網路不一定要和其他資訊一起儲存  
非必要不要用ray，因為ray似乎無法把有asyncio的東西當作函數的參數（無法pickle），畢竟我就只有一個GPU，讓inference排隊，一起預測才比較重要
## 作法
用pickle存：
    weights : ""
    {replay_buffer,num_played_games,num_played_steps,num_reanalysed_games}