package queue

import (
	"context"
	"fmt"
	"go-queue/.gen/fm_finetuning/public/model"
	"go-queue/auth"
	"go.temporal.io/sdk/client"
	"slices"
	"time"

	"database/sql"
	"github.com/danielgtaylor/huma/v2"
)

type Handler struct {
	temporal client.Client
	dbCli    *sql.DB
}

func NewHandler(tc client.Client, dbCli *sql.DB) *Handler {
	return &Handler{temporal: tc, dbCli: dbCli}
}

func (h *Handler) QueueInference(ctx context.Context, input *InferenceQueueInput) (*InferenceOutputResponse, error) {

	if len(input.Body.FinetunedModelIds) == 0 {

		return nil, huma.NewError(400, "No model is provided")
	}

	userGroups := auth.UserGroupsFromContext(ctx)
	email := auth.EmailFromContext(ctx)

	if input.Body.Name == "" {
		input.Body.Name = fmt.Sprintf("inference-%s", time.Now().UTC().Format("20060102_150405"))
	}

	finetunedModels, err := getFineTunedModelByIds(input.Body.FinetunedModelIds, h.dbCli)
	if err != nil {

		return nil, huma.NewError(400, fmt.Sprintf("Failed to get fine-tuned models: %s", err))
	}

	if len(finetunedModels) == 0 || len(finetunedModels) < len(input.Body.FinetunedModelIds) {

		return nil, huma.NewError(404, "One or more finetuned models not found")
	}

	var suryaResult interface{}
	for indx, finetunedModel := range finetunedModels {
		var sourceDetails = GetJsonRepresentationOfMap(finetunedModel.SourceDetails)
		modelId := fmt.Sprintf("%s", sourceDetails["model_id"])
		if !slices.Contains(userGroups, modelId) {
			return nil, huma.NewError(403, fmt.Sprintf("User does not have access to %s fine-tuned model", modelId))
		}
		port := fmt.Sprintf("%s", sourceDetails["port"])

		if finetunedModel.Name == "Surya" {

			err := getResultsInferSurya(modelId, port, input.Body.Query, suryaResult)
			if err != nil {
				return nil, huma.NewError(400, fmt.Sprintf("Failed to get surya results: %s", err))
			}
			// pop surya from the models to not queue it
			finetunedModels = slices.Delete(finetunedModels, indx, indx+1)
		}

	}

	inferenceOutput, err := insertInference(*input, model.Inferencestatus_Queued, email, h.dbCli)

	if err != nil {
		return nil, huma.NewError(500, fmt.Sprintf("%v", err))
	}

	var modelConfigs []map[string]interface{}

	var dataConfig map[string]interface{}
	var finetunedModelsCustom []*CustomFinetunedModel

	for _, finetunedModel := range finetunedModels {
		finetunedModelsCustom = append(finetunedModelsCustom, &CustomFinetunedModel{
			ID: finetunedModel.ID,
			Name: finetunedModel.Name,
			DataConfig: GetJsonRepresentationOfMap(*finetunedModel.DataConfig),
			SourceDetails: GetJsonRepresentationOfMap(finetunedModel.SourceDetails),
			SourceType: finetunedModel.SourceType.String(),
			CreatedAt: finetunedModel.CreatedAt,
			
		})

		dataConfig = GetJsonRepresentationOfMap(*finetunedModel.DataConfig)
		var sourceDetails = GetJsonRepresentationOfMap(finetunedModel.SourceDetails)

		modelConfig := make(map[string]interface{})

		modelConfig["data_config"] = dataConfig
		modelConfig["model_id"] = sourceDetails["model_id"]
		modelConfig["name"] = finetunedModel.Name
		modelConfig["timeseries"] = sourceDetails["timeseries"]
		modelConfig["port"] = sourceDetails["port"]
		modelConfigs = append(modelConfigs, modelConfig)
	}

	queueOrchestratorInput := QueueOrchestratorInput{
		InferenceId:  inferenceOutput.Body.ID,
		ModelConfigs: modelConfigs,
		Query:        input.Body.Query,
	}

	err = QueueInferenceTemporal(ctx, queueOrchestratorInput, h.temporal)
	if err != nil {
		return nil, huma.NewError(500, fmt.Sprintf("Can't queue inference: %v", err))
	}
	err = updateInferenceFinetunedModels(inferenceOutput.Body.ID, input.Body.FinetunedModelIds, h.dbCli)
	if err != nil {
		return nil, huma.NewError(500, fmt.Sprintf("Can't update inferenceModels tanle: %v", err))
	}
	var inferenceOutputResponse InferenceOutputResponse
	inferenceOutputResponse.Body.Id = inferenceOutput.Body.ID
	inferenceOutputResponse.Body.Name = inferenceOutput.Body.Name
	inferenceOutputResponse.Body.Query = input.Body.Query
	inferenceOutputResponse.Body.FinetunedModels = finetunedModelsCustom
	inferenceOutputResponse.Body.Status = inferenceOutput.Body.Status
	
	return &inferenceOutputResponse, nil

}
