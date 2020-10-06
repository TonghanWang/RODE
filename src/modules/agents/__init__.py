REGISTRY = {}

from .rnn_agent import RNNAgent
from .rode_agent import RODEAgent
REGISTRY["rnn"] = RNNAgent
REGISTRY["rode"] = RODEAgent
