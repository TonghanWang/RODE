REGISTRY = {}

from .basic_controller import BasicMAC
from .rode_controller import RODEMAC

REGISTRY["basic_mac"] = BasicMAC
REGISTRY['rode_mac'] = RODEMAC