import numpy as np
import ray
import asyncio
class MCTS:
	"""
	Core Monte Carlo Tree Search algorithm.
	To decide on an action, we run N simulations, always starting at the root of
	the search tree and traversing the tree according to the UCB formula until we
	reach a leaf node.
	"""

	def __init__(self, config):
		self.config = config
	async def run(self,
			model,
			observation,
			legal_actions,
			add_exploration_noise,
			override_root_with=None):
		"""
		At the root of the search tree we use the representation function to obtain a
		hidden state given the current observation.
		We then run a Monte Carlo Tree Search using only action sequences and the model
		learned by the network.
		"""
		if override_root_with:
			root = override_root_with
			root_predicted_value = None
		else:
			root = Node(0)
			output=await model.initial_inference(observation)
			
			root_predicted_value=output.value
			reward=output.reward
			policy_logits=output.policy
			hidden_state=output.hidden_state
			if self.config.support:
				root_predicted_value = models.support_to_scalar(root_predicted_value, self.config.support).item()
				reward = models.support_to_scalar(reward, self.config.support).item()
			assert (
				legal_actions
			), f"Legal actions should not be an empty array. Got {legal_actions}."
			assert set(legal_actions).issubset(
				set(range(self.config.action_space_type0_size))
			), "Legal actions should be a subset of the action space."
			root.expand(
				legal_actions,
				reward,
				policy_logits,
				hidden_state,
			)

		if add_exploration_noise:
			root.add_exploration_noise(
				dirichlet_alpha=self.config.root_dirichlet_alpha,
				exploration_fraction=self.config.root_exploration_fraction,
			)

		min_max_stats = MinMaxStats()

		max_tree_depth = 0
		for _ in range(self.config.num_simulations):
			node = root
			search_path = [node]
			current_tree_depth = 0

			while node.expanded():
				current_tree_depth += 1
				action, node = self.select_child(node, min_max_stats)
				search_path.append(node)


			# Inside the search tree we use the dynamics function to obtain the next hidden
			# state given an action and the previous hidden state
			parent = search_path[-2]
			value, reward, policy_logits, hidden_state = model.recurrent_inference(
				parent.hidden_state,
				np.array([action])
			)
			if self.config.support:
				value = models.support_to_scalar(value, self.config.support).item()
				reward = models.support_to_scalar(reward, self.config.support).item()
			node.expand(
				self.config.action_space_type0_size,
				reward,
				policy_logits,
				hidden_state,
			)

			self.backpropagate(search_path, value, virtual_to_play, min_max_stats)

			max_tree_depth = max(max_tree_depth, current_tree_depth)

		extra_info = {
			"max_tree_depth": max_tree_depth,
			"root_predicted_value": root_predicted_value,
		}
		return root, extra_info

	def select_child(self, node, min_max_stats):
		"""
		Select the child with the highest UCB score.
		"""
		ucb=[self.ucb_score(node, child, min_max_stats) for action,child in node.children.items()]
		max_ucb = max(ucb)
		action = np.random.choice(
			[
				action
				for i,action in enumerate(node.children.keys())
				if ucb[i]==max_ucb
			]
		)
		return action, node.children[action]

	def ucb_score(self, parent, child, min_max_stats):
		"""
		The score for a node is based on its value, plus an exploration bonus based on the prior.
		"""
		pb_c = (
			math.log(
				(parent.visit_count + self.config.pb_c_base + 1) / self.config.pb_c_base
			)
			+ self.config.pb_c_init
		)
		pb_c *= math.sqrt(parent.visit_count) / (child.visit_count + 1)

		prior_score = pb_c * child.prior

		if child.visit_count > 0:
			# Mean value Q
			value_score = min_max_stats.normalize(
				child.reward
				+ self.config.discount
				* (child.value() if len(self.config.players) == 1 else -child.value())
			)
		else:
			value_score = 0

		return prior_score + value_score

	def backpropagate(self, search_path, value, to_play, min_max_stats):
		"""
		At the end of a simulation, we propagate the evaluation all the way up the tree
		to the root.
		"""
		if len(self.config.players) == 1:
			for node in reversed(search_path):
				node.value_sum += value
				node.visit_count += 1
				min_max_stats.update(node.reward + self.config.discount * node.value())

				value = node.reward + self.config.discount * value

		elif len(self.config.players) == 2:
			for node in reversed(search_path):
				node.value_sum += value if node.to_play == to_play else -value
				node.visit_count += 1
				min_max_stats.update(node.reward + self.config.discount * -node.value())

				value = (
					-node.reward if node.to_play == to_play else node.reward
				) + self.config.discount * value

		else:
			raise NotImplementedError("More than two player mode not implemented.")
		  
class Node:
	def __init__(self, prior):
		self.visit_count = 0
		self.to_play = -1
		self.prior = prior
		self.value_sum = 0
		self.children = {}
		self.hidden_state = None
		self.reward = 0

	def expanded(self):
		return len(self.children) > 0

	def value(self):
		if self.visit_count == 0:
			return 0
		return self.value_sum / self.visit_count

	def expand(self, actions, to_play, reward, policy_logits, hidden_state):
		"""
		We expand a node using the value, reward and policy prediction obtained from the
		neural network.
		"""
		assert type(actions) in (int,list)
		if type(actions)==int:
			actions=list(range(actions))
		self.reward = reward
		self.hidden_state = hidden_state

		policy_values = tf.nn.softmax(
			[policy_logits[0][a] for a in actions]
		).np()
		policy = {a: policy_values[i] for i, a in enumerate(actions)}
		for action, p in policy.items():
			self.children[action] = Node(p)

	def add_exploration_noise(self, dirichlet_alpha, exploration_fraction):
		"""
		At the start of each search, we add dirichlet noise to the prior of the root to
		encourage the search to explore new actions.
		"""
		actions = list(self.children.keys())
		noise = np.random.dirichlet([dirichlet_alpha] * len(actions))
		frac = exploration_fraction
		for a, n in zip(actions, noise):
			self.children[a].prior = self.children[a].prior * (1 - frac) + n * frac
