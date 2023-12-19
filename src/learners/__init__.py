from .q_learner import QLearner
from .coma_learner import COMALearner
from .qtran_learner import QLearner as QTranLearner
from .actor_critic_learner import ActorCriticLearner
from .maddpg_learner import MADDPGLearner
from .ppo_learner import PPOLearner
from .dmaq_qatten_learner import DMAQ_qattenLearner
from .DGI2C_learner import DGI2CLearner
from .DGI2C_qplex_learner import DGI2CLearner as DGI2CQPLEXLearner

REGISTRY = {}

REGISTRY["q_learner"] = QLearner
REGISTRY["coma_learner"] = COMALearner
REGISTRY["qtran_learner"] = QTranLearner
REGISTRY["actor_critic_learner"] = ActorCriticLearner
REGISTRY["maddpg_learner"] = MADDPGLearner
REGISTRY["ppo_learner"] = PPOLearner
REGISTRY["dmaq_qatten_learner"] = DMAQ_qattenLearner
REGISTRY["DGI2C_learner"] = DGI2CLearner
REGISTRY["DGI2C_qplex_learner"] = DGI2CQPLEXLearner