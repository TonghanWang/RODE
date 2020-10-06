REGISTRY = {}

from .dot_selector import DotSelector
from .q_selector import QSelector

REGISTRY['dot'] = DotSelector
REGISTRY['q'] = QSelector
