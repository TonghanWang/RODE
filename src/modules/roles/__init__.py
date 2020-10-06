REGISTRY = {}

from .dot_role import DotRole
from .q_role import QRole
REGISTRY['dot'] = DotRole
REGISTRY['q'] = QRole
