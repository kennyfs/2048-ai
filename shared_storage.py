import copy
import os
import pickle

import ray
import tensorflow as tf


@ray.remote
class SharedStorage:
	"""
	Class which run in a dedicated thread to store the network weights and some information.
	is not necessarily responsible for save to file
	it saves:
	#from test_worker
	"total_reward",#score
	"game_length",
	"stdev_reward",

	#from trainer
	"training_step",
	"learning_rate",
	"total_loss",
	"value_loss",
	"reward_loss",
	"policy_loss",

	#from self_play_worker
	"num_played_games",
	"num_played_steps",
	"num_reanalysed_games",

	and weights from 
	"""

	def __init__(self, checkpoint, config):
		self.config = config
		self.current_checkpoint = copy.deepcopy(checkpoint)

	def save(self, replay_buffer=None):
		#replay_buffer is replay_buffer.ReplayBuffer.buffer, a dict {num_played_games: game_history}
		training_step = self.current_checkpoint['training_step']
		path = os.path.join(self.config.results_path, f"model-{training_step:06d}.pkl")
		with open(path,'wb') as F:
			pickle.dump(self.current_checkpoint,F)
		if replay_buffer:
			path = os.path.join(self.config.results_path, f"replay_buffer-{training_step:06d}.pkl")
			with open(path,'wb') as F:
				pickle.dump(replay_buffer,F)
		path = os.path.join(self.config.results_path, 'newest_training_step')
		with open(path, 'w') as F:
			F.write(f'{training_step:06d}')
	def get_checkpoint(self):
		return copy.deepcopy(self.current_checkpoint)

	def get_info(self, keys):
		if isinstance(keys, str):
			return self.current_checkpoint[keys]
		elif isinstance(keys, list):
			return {key: self.current_checkpoint[key] for key in keys}
		else:
			raise TypeError

	def set_info(self, keys, values=None):
		if isinstance(keys, str) and values is not None:
			self.current_checkpoint[keys] = values
		elif isinstance(keys, dict):
			self.current_checkpoint.update(keys)
		else:
			raise TypeError
