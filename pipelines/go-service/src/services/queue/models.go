package queue

import (
	"go-queue/.gen/fm_finetuning/public/model"
	"github.com/google/uuid"
	"time"
)

type InferenceQueryType struct {
	BoundingBox [4]float64 `json:"bounding_box"`
	Dates       string     `json:"dates"`
}

type QueryType struct {
	Name              string             `json:"name"`
	Query             InferenceQueryType `json:"query"`
	FinetunedModelIds []string           `json:"finetuned_model_ids"`
}

type InferenceQueueInput struct {
	Body struct {
		QueryType
	}
}

type InferenceModelsOutput struct {
	Body []model.FinetunedModels
}

type InferenceOutput struct {
	Body model.Inferences
}

type QueueOrchestratorInput struct {
	InferenceId  uuid.UUID                `json:"inference_id"`
	ModelConfigs []map[string]interface{} `json:"model_configs"`
	Query        InferenceQueryType       `json:"query"`
}


type CustomFinetunedModel struct {
    ID uuid.UUID `json:"id"`
    Name string `json:"name"`
    SourceType string `json:"source_type"`
    SourceDetails map[string]interface{} `json:"source_details"`
    CreatedAt time.Time `json:"created_at"`
    DataConfig map[string]interface{} `json:"data_config"`

}

type Inference struct {
	Id uuid.UUID `json:"id"`
	Name string `json:"name"`
	Query InferenceQueryType `json:"query"`
	Status model.Inferencestatus `json:"status"`
	Results      map[string]interface{} `json:"results"`
	FinetunedModels []*CustomFinetunedModel `json:"finetuned_models"`
	CreatedAt time.Time `json:"created_at"`
	UpdatedAt time.Time `json:"updated_at"`
}

type InferenceOutputResponse struct {
	Body Inference
	
}


