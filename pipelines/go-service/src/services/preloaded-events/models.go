package preloadedevents

import (
	"go-queue/services/queue"
	"github.com/google/uuid"
	"time"
)


type PreloadedEventsCustom struct {
	ID           uuid.UUID `json:"id"`
	EventName    string `json:"event_name"`
	EventDetails map[string]interface{} `json:"event_details"`
	CreatedAt    time.Time `json:"created_at"`
	InferenceID  *uuid.UUID `json:"inference_id"`
	Inference queue.Inference `json:"inference"`
}


type PreloadedEventsResponse struct {
	Body []*PreloadedEventsCustom
}





