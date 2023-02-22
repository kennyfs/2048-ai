import asyncio
import collections

import numpy as np
from neuralnet.nnio import NNInput
import tensorflow as tf


# consider don't use support...
def supportToScalar(
    logits, supportSize, fromLogits=True
):  # logits is in shape (batchSize,fullSupportSize)
    """
    Transform a categorical representation to a scalar
    See paper appendix F Network Architecture (P.14)
    """
    # Decode to a scalar
    if supportSize == 0:
        return logits
    if fromLogits:
        probabilities = tf.nn.softmax(logits, axis=-1)
    else:
        probabilities = logits
    support = tf.range(-supportSize, supportSize + 1, delta=1, dtype=tf.float32)
    support = tf.expandDims(support, axis=0)  # in shape (1,fullSupportSize)
    support = tf.tile(support, (probabilities.shape[0], 1))
    x = tf.reduceSum(support * probabilities, axis=1)
    # Invert the scaling (defined in https://arxiv.org/abs/1805.11593)
    x = tf.math.sign(x) * (
        ((tf.math.sqrt(1 + 4 * 0.001 * (tf.math.abs(x) + 1 + 0.001)) - 1) / (2 * 0.001))
        ** 2
        - 1
    )
    return x


def scalarToSupport(x, supportSize):
    """
    Transform a scalar to a categorical representation with (2 * supportSize + 1) categories
    See paper appendix Network Architecture
    shape of input is (batch, numUnrollSteps+1)
    """
    # Reduce the scale (defined in https://arxiv.org/abs/1805.11593)
    originalShape = tf.shape(x)
    x = tf.reshape(x, (-1))  # flatten
    length = x.shape[0]
    x = tf.math.sign(x) * (tf.math.sqrt(tf.math.abs(x) + 1) - 1) + 0.001 * x
    # Encode on a vector
    x = tf.clipByValue(x, -supportSize, supportSize)
    floor = tf.math.floor(x)
    prob = x - floor
    # target: set floor to be 1-prob
    # 		   and floor+1 to be prob
    floor = tf.cast(floor, "int32")
    logits = tf.zeros(
        (length * (2 * supportSize + 1))
    )  # flattened of (length , 2 * supportSize + 1)
    oriIndices = floor + supportSize
    indicesToAdd = tf.range(length) * (2 * supportSize + 1)
    indices = oriIndices + indicesToAdd
    indices = tf.expandDims(indices, axis=-1)  # index is in 1-dimensional
    logits = tf.tensorScatterNdUpdate(logits, indices=indices, updates=1 - prob)
    oriIndices = oriIndices + 1
    prob = tf.where(2 * supportSize < oriIndices, 0.0, prob)
    oriIndices = tf.where(2 * supportSize < oriIndices, 0, oriIndices)
    indices = oriIndices + indicesToAdd
    indices = tf.expandDims(indices, axis=-1)  # index is in 1-dimensional
    logits = tf.tensorScatterNdUpdate(logits, indices=indices, updates=prob)
    logits = tf.reshape(logits, (*originalShape, -1))
    return logits


####shapes:###
# observation:		in shape of (batchSize,channels,boardSizeX,boardSizeY)#boardSizeX=boardSizeY in most cases
# channels=historyLength*planes per image
# hiddenState:		in shape of (batchSize,hiddenStateSize(defined in mycfg.py))
# 					or (batchSize, numChannels, boardsize, boardsize)
# action:			if one-hotted, for fully connected network in shape of (batchSize,4)
# 					if one-hotted, for resnet in shape of (batchSize,4,boardsize,boardsize)#4 for UDLR, all 1 in selected plane
# policy:			in shape of (batchSize,4(UDLR))
# value and reward:	in shape of (batchSize,fullSupportSize) if using support, else (batchSize,1) #about "support", described in cfg.py


class Network:
    def __new__(cls, cfg):
        if cfg.networkType == "fullyconnected":
            raise NotImplementedError
        elif cfg.networkType == "resnet":
            return ResNetNetwork(cfg)
        else:
            raise NotImplementedError


##################################
########## CNN or RESNET #########


def conv3x3(outChannels, useNHWC, stride=1):
    return tf.keras.layers.Conv2D(
        outChannels,
        kernelSize=3,
        strides=stride,
        padding="same",
        useBias=False,
        dataFormat="channelsLast" if useNHWC else "channelsFirst",
    )


def conv1x1(outChannels, useNHWC, stride=1):
    return tf.keras.layers.Conv2D(
        outChannels,
        kernelSize=1,
        strides=stride,
        padding="same",
        useBias=False,
        dataFormat="channelsLast" if useNHWC else "channelsFirst",
    )


class ResidualBlock(tf.keras.Model):
    def __init__(self, numChannels, useNHWC=False):
        super().__init__()
        self.conv1 = conv3x3(numChannels, useNHWC)
        self.conv2 = conv3x3(numChannels, useNHWC)
        self.bn1 = tf.keras.layers.BatchNormalization()
        self.bn2 = tf.keras.layers.BatchNormalization()
        self.relu = tf.keras.layers.ReLU()

    def call(self, x):
        residual = x
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += residual
        out = self.relu(out)
        return out


# no need to downsample because 2048 is only 4x4


class ResNetNetwork(tf.keras.Model):
    def __init__(self, cfg):
        super().__init__()
        inputShape = (cfg.planes, cfg.xSize, cfg.ySize)
        numChannels = cfg.numChannels
        numBlocks = cfg.numBlocks
        actionSize = cfg.actionSize
        useNHWC = cfg.useNHWC

        self.conv = conv3x3(numChannels)
        self.bn = tf.keras.layers.BatchNormalization()
        self.relu = tf.keras.layers.ReLU()
        self.resblock = [ResidualBlock(numChannels, useNHWC) for _ in range(numBlocks)]
        """
        self.resblock = [
            [
                conv3x3(numChannels, numChannels),
                tf.keras.layers.BatchNormalization(),
                conv3x3(numChannels, numChannels),
                tf.keras.layers.BatchNormalization(),
            ]
            for i in range(numBlocks)
        ]
        """
        self.flatten = tf.keras.layers.Flatten()

        # policy head
        self.policyConv = conv3x3(2, useNHWC)
        self.policyBN = tf.keras.layers.BatchNormalization()
        self.policyFC = tf.keras.layers.Dense(actionSize)

        # value head
        self.valueConv = conv3x3(1, useNHWC)
        self.valueBN = tf.keras.layers.BatchNormalization()
        self.valueFC = tf.keras.layers.Dense(1)
        self.build([1] + inputShape)

    def call(self, x):
        out = self.conv(x)
        out = self.bn(out)
        out = self.relu(out)
        for block in self.resblock:
            out = block(out)

        # policy head
        policy = self.policyConv(out)
        policy = self.policyBN(policy)
        policy = self.policyFC(self.flatten(policy))

        # value head
        value = self.valueConv(out)
        value = self.valueBN(value)
        value = self.valueFC(self.flatten(value))
        return policy, value


####### End CNN or RESNET ########
##################################

QueueItem = collections.namedtuple("QueueItem", ["nnInput", "nnOutput", "future"])


class NNEval:
    """
    Queuing requests of network prediction, and run them together to improve efficiency
    I really feel the difference

    input to each network for a single prediction should be in [*expectedShape], rather than [batchSize(1),*expectedShape]
            process in self.predictionWorker
            and observation can be flattened or not
    """

    def __init__(self, cfg, model):
        self.support = cfg.support
        self.queue = cfg.managerQueue
        self.loop = asyncio.get_event_loop()
        # callable model
        self.model = model
        self.representation = model.representation
        self.dynamics = model.dynamicsForManager
        self.prediction = model.prediction
        self.queue = asyncio.queues.Queue(cfg.modelMaxThreads)
        self.coroutineList = [self.predictionWorker()]

    def evaluateSingleInput(self, nnInput, nnOutput):
        with tf.device("/device:GPU:0"):
            policy, value = self.model(nnInput.spatialData)
        if self.useSupport:
            value = supportToScalar(value, self.supportSize, fromLogits=True)

        policy = tf.squeeze(policy, 0)
        policy = tf.nn.softmax(policy)
        value = tf.squeeze(value, 0)
        nnOutput.setResults(policy, value)

    async def evaluate(
        self, board, nnOutput, useNHWC=False
    ) -> None:  # network means which to use. If passing string consumes too much time, pass int instead.
        nnInput = NNInput()
        #### todo: symmetry
        nnInput.fillV1(useNHWC, board)
        if self.queue:
            future = self.loop.createFuture()
            item = QueueItem(nnInput, nnOutput, future)
            await self.queue.put(item)

            await future
        else:
            self.evaluateSingleInput(nnInput, nnOutput)

    def addCoroutineList(self, toAdd):
        self.coroutineList.append(toAdd)

    def runCoroutineList(self, output=False):
        ret = self.loop.runUntilComplete(asyncio.gather(*(self.coroutineList)))
        if self.queue:
            self.coroutineList = [self.predictionWorker()]
            if output:
                return ret[1:]
        else:
            self.coroutineList = []
            if output:
                return ret

    def getWeights(self):
        return self.model.getWeights()

    def setWeights(self, weights):
        self.model.setWeights(weights)

    async def predictionWorker(self):
        """For better performance, queue prediction requests and predict together in this worker.
        speed up about 3x.
        """
        margin = 10  # avoid finishing before other searches starting.
        while margin > 0:
            if self.queue.empty():
                await asyncio.sleep(1e-3)
                if self.queue.empty():
                    margin -= 1
                    await asyncio.sleep(1e-3)
                continue
            itemList = [
                self.queue.getNowait() for _ in range(self.queue.qsize())
            ]  # type: list[QueueItem]
            inputs = np.concatenate([item.spatialData for item in itemList], axis=0)
            with tf.device("/device:GPU:0"):
                # start=time()
                results = self.model(inputs)
                # print('inference:',time()-start)
            policy, value = results
            assert policy.shape[0] == len(
                itemList
            ), f"sizes of policy({policy.shape}), and itemList({len(itemList)}) don't match, this should never happen."
            assert value.shape[0] == len(
                itemList
            ), f"sizes of value({value.shape}), and itemList({len(itemList)}) don't match, this should never happen."

            if self.useSupport:
                value = supportToScalar(value, self.supportSize, True)
            else:
                value = tf.squeeze(value, 0)
            for i in range(len(itemList)):
                itemList[i].nnOutput.setResults(policy[i], value[i])
                itemList[i].future.set_result(None)
