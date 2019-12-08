import tensorflow as tf
import numpy as np
#the use of this file will be in the mcts.py
#if I use tpu, optimizer must handle by "tf.tpu.CrossShardOptimizer" to cross more than one tpus.And then use it as GPU!!!
inputs=tf.keras.Input(shape=(4,4,18))#from 0 to 17, 0 to 131072
conv1=tf.keras.layers.Conv2D(
	filters=2,
	kernel_size=(2,1),padding="same",
	activation=None,
	kernel_regularizer=tf.keras.regularizers.l2(1e-4)
	)(inputs)
conv2=tf.keras.layers.Conv2D(
	filters=2,
	kernel_size=(1,2),padding="same",
	activation=None,
	kernel_regularizer=tf.keras.regularizers.l2(1e-4)
	)(inputs)
conv=tf.keras.layers.Concatenate(axis=-1)([conv1,conv2])#last axis, channel
conv1=tf.keras.layers.Conv2D(
	filters=2,
	kernel_size=(2,1),padding="same",
	activation=None,
	kernel_regularizer=tf.keras.regularizers.l2(1e-4)
	)(conv)
conv2=tf.keras.layers.Conv2D(
	filters=2,
	kernel_size=(1,2),padding="same",
	activation=None,
	kernel_regularizer=tf.keras.regularizers.l2(1e-4)
	)(conv)
conv=tf.keras.layers.Concatenate(axis=-1)([conv1,conv2])#last axis, channel
flatten=tf.keras.layers.Flatten()(conv)
policy=tf.keras.layers.Dense(units=10,kernel_regularizer=tf.keras.regularizers.l2(1e-4))(flatten)
policy=tf.keras.layers.Dense(units=4,activation='softmax',kernel_regularizer=tf.keras.regularizers.l2(1e-4),name='policy')(flatten)
'''
predictions = {
	'policy': tf.keras.layers.Softmax(name='policy_head')(policy_head_output),
	'value' : tf.identity(value_head_output, name='value_head')
}
if mode == tf.estimator.ModeKeys.PREDICT:
	return tf.estimator.EstimatorSpec(mode=mode, predictions=predictions)
policy_loss = tf.nn.softmax_cross_entropy_with_logits(labels=labels['policy'], logits=policy_head_output)# it does soft max to logits before cross entropy
#its output is an array, needing to reduce mean to a scalar
policy_loss = tf.reduce_mean(policy_loss)

value_loss = tf.losses.mean_squared_error(labels=labels['value'], predictions=value_head_output)
value_loss = tf.reduce_mean(value_loss)

regularizer = tf.keras.tf.keras.regularizers.l2(scale=1e-4)#as the alphagozero paper, c||θ||_2
regular_variables = tf.trainable_variables()#refers to the trainable things, including all the conv2d, batchnorm, and dense.
l2_loss = tf.contrib.layers.apply_regularization(regularizer, regular_variables)
loss = value_loss + policy_loss + l2_loss
loss = tf.identity(loss, name='loss')
def learning_rate_fn(global_step):
	#it can't be like this:
	
	#if global_step < 400:
	#	return 1e-2
	#elif global_step < 600:
	#	return 1e-3
	#return 1e-4
	
	boundaries = [400,600]
	learning_rates = [1e-2,1e-3,1e-4]
	lr = tf.train.piecewise_constant(global_step, boundaries, learning_rates)
if mode == tf.estimator.ModeKeys.TRAIN:
	global_step = tf.train.get_or_create_global_step()
	learning_rate = learning_rate_fn(global_step)
	
	optimizer = tf.train.MomentumOptimizer(
		learning_rate=learning_rate,
		momentum=0.9
	)#continue here
	train_op = optimizer.minimize(
		loss=loss,
		global_step=global_step#which is a variable
	)
	return tf.estimator.EstimatorSpec(mode=mode, loss=loss, train_op=train_op)
#no evaluate mode
def input_fn(features, labels, training=False, batch_size=256):
	"""An input function for training or predicting"""
	# Convert the inputs to a Dataset.
	dataset = tf.data.Dataset.from_tensor_slices((features, labels))

	# Shuffle and repeat if you are in training mode.
	if training:
		dataset = dataset.shuffle(1000).repeat()
		
	return dataset.batch(batch_size)
'''
class nn:
	def __init__(self,init):
		if init:
			self.model=tf.keras.Model(inputs=inputs,outputs=policy)
			self.model.compile(optimizer=tf.keras.optimizers.SGD(learning_rate=0.001,momentum=0.9,name='SGD'),loss='categorical_crossentropy',metrics=['accuracy'])
		else:
			self.model=None
	def train(self,data,label):
		self.model.fit(data,label,epochs=5,steps_per_epoch=1000,batch_size=32)
	def predict(self,data):
		return self.model.predict(data)
	def save(self,name):
		self.model.save(name)
	def load(self,name):
		self.model=tf.contrib.keras.models.load_model(name)
nn=nn(True)
x=np.random.randint(2,size=(1024,4,4,16)).astype('float32')
y=np.random.randint(100,size=(1024,4)).astype('float32')
for i in range(1024):
	total=sum(y[i])
	for j in range(4):
		y[i][j]/=total
nn.train(x,y)
nn.save('random.h5')
