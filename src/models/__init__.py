from .finetuned_model import FinetunedModelRead, FinetunedModelUpdate
from .inference import InferenceRead, InferenceUpdate
from .preloaded_event import PreloadedEventRead, PreloadedEventUpdate

# Update forward references with explicit globalns
InferenceRead.model_rebuild()
PreloadedEventRead.model_rebuild()
