import collections


MAXIMUM_FLOAT_VALUE = float("inf")
KnownBounds = collections.namedtuple("KnownBounds", ["min", "max"])


class MinMaxStats:
    """A class that holds the min-max values of the tree."""

    def __init__(self, knownBounds=None):
        self.maximum = knownBounds.max if knownBounds else -MAXIMUM_FLOAT_VALUE
        self.minimum = knownBounds.min if knownBounds else MAXIMUM_FLOAT_VALUE

    def update(self, value: float):
        self.maximum = max(self.maximum, value)
        self.minimum = min(self.minimum, value)

    def normalize(self, value: float) -> float:
        if self.maximum > self.minimum:
            # We normalize only when we have set the maximum and minimum values.
            return (value - self.minimum) / (self.maximum - self.minimum)
        return value


class SearchParams:
    def __init__(
        self,
        numSimulations: int,
        pbCBase: float,
        pbCInit: float,
        addExplorationNoise: bool,
        dirichletAlpha: float,
        explorationFraction: float,
        discount: float,
    ):
        self.numSimulations = numSimulations
        self.pbCBase = pbCBase
        self.pbCInit = pbCInit
        self.addExplorationNoise = addExplorationNoise
        self.dirichletAlpha = dirichletAlpha
        self.explorationFraction = explorationFraction
        self.discount = discount
