REGISTRY = {}

from .rnn_agent import RNNAgent
from .rnn_ns_agent import RNNNSAgent
from .full_comm_agent import FullCommAgent
from .DGI2C_agent import DGI2CAgent

REGISTRY["rnn"] = RNNAgent
REGISTRY["rnn_ns"] = RNNNSAgent
REGISTRY["full_comm"] = FullCommAgent
REGISTRY["DGI2C"] = DGI2CAgent