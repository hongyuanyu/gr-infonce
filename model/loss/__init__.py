# loss
from .loss_wrapper import DistributedLossWrapper
#from .part_triplet_loss import PartTripletLoss
from .center_loss import CenterLoss
from .cross_entropy_label_smooth import CrossEntropyLabelSmooth
from .triplet_loss import TripletLoss
from .contrastive_loss import ContrastiveLoss
from .cloth_loss import ClothLoss
from .reranking_graph import RerankingGraph
from .spcl import SPCLoss
from .infonce_loss import InfonceLoss
