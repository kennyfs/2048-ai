import collections
import copy
import glob
import os
import pickle
import random
from statistics import mode
import sys
import time

import numpy as np
import ray
import tensorflow as tf

import environment
import my_config
import network
import replay_buffer
import self_play
import shared_storage
import trainer

bg   ="\x1b[48;5;"
word ="\x1b[38;5;"
end  ="m"
reset="\x1b[0m"
'''
assumption:
total training steps=1e5 or less
'''
def model_get(config):
	tmp_model=network.Network(config)
	return tmp_model.get_weights(),tmp_model.summary()
class MuZero:
	"""
	Main class to manage MuZero.

	Args:

		config (my_config.Config, optional): Override the default config of the game.

		split_resources_in (int, optional): Split the GPU usage when using concurent muzero instances.

	Example:
		>>> muzero = MuZero("cartpole")
		>>> muzero.train()
		>>> muzero.test(render=True)
	"""

	def __init__(self, config=None):
		# Load the game 
		self.Game=environment.Environment
		# Overwrite the config
		if config:
			self.config=config
		else:
			self.config=my_config.default_config()

		# Fix random generator seed
		seed=self.config.seed
		if seed==None:
			seed=random.randrange(2**32)
			print(f'seed was set to be {seed}.')
		self.config.seed=seed
		random.seed(seed)
		np.random.seed(seed)
		tf.random.set_seed(seed)
		'''
		# Manage GPUs
		if self.config.max_num_gpus == 0 and (
			self.config.selfplay_on_gpu
			or self.config.train_on_gpu
			or self.config.reanalyze_on_gpu
		):
			raise ValueError(
				"Inconsistent MuZeroConfig: max_num_gpus = 0 but GPU requested by selfplay_on_gpu or train_on_gpu or reanalyze_on_gpu."
			)
		if (
			self.config.selfplay_on_gpu
			or self.config.train_on_gpu
			or self.config.reanalyze_on_gpu
		):
			total_gpus = (
				self.config.max_num_gpus
				if self.config.max_num_gpus is not None
				else torch.cuda.device_count()
			)
		else:
			total_gpus = 0
		self.num_gpus = total_gpus / split_resources_in
		if 1 < self.num_gpus:
			self.num_gpus = math.floor(self.num_gpus)
		'''
		ray.init(ignore_reinit_error=True)

		# Checkpoint and replay buffer used to initialize workers
		self.checkpoint = {
			"weights": None,
			"total_reward": 0,
			"game_length": 0,
			"stdev_reward": 0,
			"training_step": 0,
			"learning_rate": 0,
			"total_loss": 0,
			"value_loss": 0,
			"reward_loss": 0,
			"policy_loss": 0,
			"num_played_games": 0,
			"num_played_steps": 0,
			"num_reanalyzed_games": 0,
		}
		self.replay_buffer = {}
		
		self.checkpoint["weights"] ,self.summary = copy.deepcopy(model_get(self.config))
		# Workers
		self.self_play_workers = None
		self.test_worker = None
		self.training_worker = None
		self.reanalyze_worker = None
		self.replay_buffer_worker = None
		self.shared_storage_worker = None
		self.file_writer=tf.summary.create_file_writer(self.config.results_path)

	def train(self, log_in_tensorboard=True):
		"""
		Spawn ray workers and launch the training.

		Args:
			log_in_tensorboard (bool): Start a testing worker and log its performance in TensorBoard.
		"""
		if log_in_tensorboard or self.config.save_model:
			os.makedirs(self.config.results_path, exist_ok=True)
		print(
			"\nTraining...\nRun tensorboard --logdir ./results and go to http://localhost:6006/ to see in real time the training performance.\n"
		)
		'''
		# Manage GPUs
		if 0 < self.num_gpus:
			num_gpus_per_worker = self.num_gpus / (
				self.config.train_on_gpu
				+ self.config.num_actors * self.config.selfplay_on_gpu
				+ log_in_tensorboard * self.config.selfplay_on_gpu
				+ self.config.reanalyze * self.config.reanalyze_on_gpu
			)
			if 1 < num_gpus_per_worker:
				num_gpus_per_worker = math.floor(num_gpus_per_worker)
		else:
			num_gpus_per_worker = 0
		'''
		#I only have 1 gpu, I don't know the default ray uses, but I'll just not use .options to specify num_gpus
		# Initialize workers
		model=network.Network(self.config)
		manager=network.Manager(self.config,model)
		predictor=network.Predictor(manager)
		self.training_worker = trainer.Trainer(self.checkpoint, model, self.config)

		self.shared_storage_worker = shared_storage.SharedStorage(self.checkpoint, self.config)

		self.replay_buffer_worker = replay_buffer.ReplayBuffer.remote(self.checkpoint, self.replay_buffer, self.config)

		if self.config.reanalyze:
			self.reanalyze_worker = replay_buffer.Reanalyze(self.checkpoint, model, self.config)
		self.self_play_worker = self_play.SelfPlay(predictor, self.Game, self.config, self.config.seed)

		# Launch workers
		counter=0
		training_counter=0
		try:
			while 1:
				self.self_play_worker.self_play(
					self.replay_buffer_worker, self.shared_storage_worker
				)
				print('done playing')
				self.training_worker.update_weights(
					self.replay_buffer_worker, self.shared_storage_worker
				)
				print('done training')
				self.reanalyze_worker.reanalyze.remote(
					self.replay_buffer_worker, self.shared_storage_worker
				)
				print('done reanalyzing')
				self.log_once(counter, training_counter)
		except KeyboardInterrupt:
			pass
		# Persist replay buffer to disk
		self.terminate()
		'''
		if log_in_tensorboard:
			self.log_once()'''

	def log_once(self, counter, training_counter):
		"""
		Keep track of the training performance.
		"""
		# Play a test game each time it logs.
		self.test_worker = self_play.SelfPlay(
			self.checkpoint,
			self.Game,
			self.config,
			self.config.seed + self.config.num_actors,
		)
		self.test_worker.self_play(
			None, self.shared_storage_worker, True
		)
		#"game_length", "total_reward", "stdev_reward" are set
		
		with self.file_writer.as_default():

			# Save hyperparameters to TensorBoard
			hp_table = [
				f"| {key} | {value} |" for key, value in self.config.__dict__.items()
			]
			tf.summary.text(
				"Hyperparameters",
				"| Parameter | Value |\n|-------|-------|\n" + "\n".join(hp_table),
				0,
			)
			# Save model representation
			tf.summary.text(
				"Model summary", self.summary, 0
			)
			# Loop for updating the training performance
			counter = 0
			keys=[
				#from test_worker
				"total_reward",#score
				"game_length",
				"stdev_reward",
				#from trainer
				"training_step",
				#"learning_rate",
				"total_loss",
				"value_loss",
				"reward_loss",
				"policy_loss",
				#from self_play_worker, not too important
				"num_played_games",
				"num_played_steps",
				"num_reanalyzed_games",
			]
			info = ray.get(self.shared_storage_worker.get_info.remote(keys))
			info = ray.get(self.shared_storage_worker.get_info.remote(keys))
			tf.summary.scalar(
				"1.Test_worker/1.Total_reward(Score)", info["total_reward"], counter,
			)
			tf.summary.scalar(
				"1.Test_worker/3.game_length", info["game_length"], counter,
			)
			tf.summary.scalar(
				"1.Test_worker/3.stdev_reward", info["stdev_reward"], counter,
			)
			tf.summary.scalar(
				"2.Self_play_worker/1.num_played_games", info["num_played_games"], counter
			)
			tf.summary.scalar(
				"2.Self_play_worker/2.num_played_steps", info["num_played_steps"], counter
			)
			tf.summary.scalar(
				"2.Self_play_worker/3.num_reanalyzed_games", info["num_reanalyzed_games"], counter
			)
			log_training_config=1
			if log_training_config==1:
				#log all loss
				for total_loss,value_loss,reward_loss,policy_loss in zip(info["total_loss"],info["value_loss"],info["reward_loss"],info["policy_loss"]):
					#tf.summary.scalar(
					#	"3.Trainer_worker/1.training_step", training_step, training_counter,
					#)
					#tf.summary.scalar(
					#	"3.Trainer_worker/2.learning_rate", learning_rate, training_counter
					#)
					tf.summary.scalar(
						"3.Trainer_worker/1.total_loss", total_loss, training_counter
					)
					tf.summary.scalar(
						"3.Trainer_worker/2.value_loss", value_loss, training_counter
					)
					tf.summary.scalar(
						"3.Trainer_worker/3.reward_loss", reward_loss, training_counter
					)
					tf.summary.scalar(
						"3.Trainer_worker/4.policy_loss", policy_loss, training_counter
					)
					training_counter+=1
			elif log_training_config==2:
				#log mean of loss
				length=len(total_loss)
				total_loss,value_loss,reward_loss,policy_loss=sum(info["total_loss"])/length,sum(info["value_loss"])/length,sum(info["reward_loss"])/length,sum(info["policy_loss"])/length#all are numpy array(size=())
				tf.summary.scalar(
					"3.Trainer_worker/1.training_step", info['training_step'], counter,
				)
				tf.summary.scalar(
					"3.Trainer_worker/2.total_loss", total_loss, counter
				)
				tf.summary.scalar(
					"3.Trainer_worker/3.value_loss", value_loss, counter
				)
				tf.summary.scalar(
					"3.Trainer_worker/4.reward_loss", reward_loss, counter
				)
				tf.summary.scalar(
					"3.Trainer_worker/5.policy_loss", policy_loss, counter
				)
			else:
				raise NotImplementedError
			print(
				f'Last test score: {info["total_reward"]:6d}. Training step: {info["training_step"]}/{self.config.training_steps}. Played games: {info["num_played_games"]}. Loss: {info["total_loss"]:.3f}',
				#end="\r",
			)
			counter += 1

	def terminate(self):
		"""
		Softly terminate the running tasks and garbage collect the workers.
		"""

		print("\n\nPersisting replay buffer games to disk...")
		replay_buffer = ray.get(self.replay_buffer_worker.get_buffer.remote())
		self.shared_storage_worker.save(replay_buffer)

		print("\nShutting down workers...")

		self.self_play_workers = None
		self.test_worker = None
		self.training_worker = None
		self.reanalyze_worker = None
		self.replay_buffer_worker = None
		self.shared_storage_worker = None

		print('done saving')

	def load_model(self, checkpoint_path=None, replay_buffer_path=None):
		"""
		Load a model and/or a saved replay buffer.

		Args:
			checkpoint_path (str): Path to model-{training_step}.pkl.

			replay_buffer_path (str): Path to replay_buffer-{training_step}.pkl
		"""
		# Load checkpoint
		if checkpoint_path:
			if os.path.exists(checkpoint_path):
				with open(checkpoint_path, "rb") as f:
					self.checkpoint = pickle.load(f)
				print(f"\nUsing checkpoint from {checkpoint_path}")
			else:
				print(f"\nThere is no model saved in {checkpoint_path}.")

		# Load replay buffer
		if replay_buffer_path:
			if os.path.exists(replay_buffer_path):
				with open(replay_buffer_path, "rb") as f:
					self.replay_buffer = pickle.load(f)

				print(f"\nInitializing replay buffer with {replay_buffer_path}")
			else:
				print(
					f"Warning: Replay buffer path '{replay_buffer_path}' doesn't exist.  Using empty buffer."
				)
				self.checkpoint["training_step"] = 0
				self.checkpoint["num_played_steps"] = 0
				self.checkpoint["num_played_games"] = 0
				self.checkpoint["num_reanalyzed_games"] = 0
	def load_model_menu(self):
		# Configure running options
		options = sorted(glob.glob(f"results/*/"),reverse=True) + ["Specify paths manually"]
		print()
		for i in range(len(options)):
			print(f"{i}. {options[i]}")

		choice = input("Enter a number to choose a model to load: ")
		valid_inputs = [str(i) for i in range(len(options))]
		while choice not in valid_inputs:
			choice = input("Invalid input, enter a number listed above: ")
		choice = int(choice)

		if choice == (len(options) - 1):
			# manual path option
			checkpoint_path = input(
				"Enter a path to the model.checkpoint, or ENTER if none: "
			)
			while checkpoint_path and not os.path.isfile(checkpoint_path):
				checkpoint_path = input("Invalid checkpoint path. Try again: ")
			replay_buffer_path = input(
				"Enter a path to the replay_buffer.pkl, or ENTER if none: "
			)
			while replay_buffer_path and not os.path.isfile(replay_buffer_path):
				replay_buffer_path = input("Invalid replay buffer path. Try again: ")
		else:
			#default to choose newest data
			with open(f'{options[choice]}newest_training_step','r') as F:
				newest_training_step=F.read()
			checkpoint_path = f"{options[choice]}model-{newest_training_step}.pkl"
			replay_buffer_path = f"{options[choice]}replay_buffer-{newest_training_step}.pkl"

		self.load_model(
			checkpoint_path=checkpoint_path, replay_buffer_path=replay_buffer_path,
		)
		
if __name__ == "__main__":
	if len(sys.argv) > 2 and sys.argv[2] in ('start','train'):
		# Train directly with "python muzero.py cartpole"
		muzero = MuZero()
		muzero.train()
	else:
		# Let user pick a game
		
		muzero = MuZero()

		while True:
			# Configure running options
			'''
			options = [
				"Train",
				"Load pretrained model",
				"Render some self play games",
				"Test the game manually",
				"Exit",
			]
			print()
			for i in range(len(options)):
				print(f"{i}. {options[i]}")

			choice = input("Enter a number to choose an action: ")
			valid_inputs = [str(i) for i in range(len(options))]
			while choice not in valid_inputs:
				choice = input("Invalid input, enter a number listed above: ")
			choice = int(choice)
			'''
			choice=0
			if choice == 0:
				muzero.train()
			elif choice == 1:
				muzero.load_model_menu()
			elif choice == 2:
				muzero.test(render=True, opponent="self", muzero_player=None)
			elif choice == 3:
				env = muzero.Game()
				env.reset()
				env.render()

				done = False
				while not done:
					action = env.human_to_action()
					observation, reward, done = env.step(action)
					print(f"\nAction: {env.action_to_string(action)}\nReward: {reward}")
					env.render()
			else:
				break
			print("\nDone")

	ray.shutdown()
