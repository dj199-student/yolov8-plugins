"""注意力机制插件包"""

from .se import SEAttention
from .cbam import CBAM
from .eca import ECAAttention
from .ca import CoordAtt
from .gam import GAM
from .simam import SimAM
from .shuffle import ShuffleAttention
from .triplet import TripletAttention
from .da import DualAttention

__all__ = [
    "SEAttention",
    "CBAM",
    "ECAAttention",
    "CoordAtt",
    "GAM",
    "SimAM",
    "ShuffleAttention",
    "TripletAttention",
    "DualAttention",
]
