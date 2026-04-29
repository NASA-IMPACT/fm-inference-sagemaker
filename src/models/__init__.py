from .finetuned_model import (
    FinetunedModelRead as FinetunedModelRead,
    FinetunedModelUpdate as FinetunedModelUpdate,
)
from .inference import (
    InferenceRead as InferenceRead,
    InferenceUpdate as InferenceUpdate,
)
from .preloaded_event import (
    PreloadedEventRead as PreloadedEventRead,
    PreloadedEventUpdate as PreloadedEventUpdate,
)

# Update forward references with explicit globalns
InferenceRead.model_rebuild()
PreloadedEventRead.model_rebuild()
