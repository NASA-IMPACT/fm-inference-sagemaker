package queue

import (
	"bytes"
	"encoding/json"
	"fmt"
	"go-queue/.gen/fm_finetuning/public/model"
	. "go-queue/.gen/fm_finetuning/public/table"
	"log"
	"net/http"
	"strings"
	"time"

	"database/sql"
	. "github.com/go-jet/jet/v2/postgres"
	"github.com/google/uuid"
)

func getFineTunedModelByIds(ids []string, dbCli *sql.DB) ([]*model.FinetunedModels, error) {

	var finetunedModels []*model.FinetunedModels

	expressions := make([]Expression, len(ids))
	for i, id := range ids {
		uuidId, err := uuid.Parse(id)
		if err != nil {
			return nil, err
		}
		expressions[i] = UUID(uuidId)
	}

	selectStmt := FinetunedModels.SELECT(FinetunedModels.AllColumns).WHERE(FinetunedModels.ID.IN(expressions...))
	err := selectStmt.Query(dbCli, &finetunedModels)
	if err != nil {
		return nil, err
	}

	return finetunedModels, nil

}

func ParseInferenceQuery(queryStr string) InferenceQueryType {
	var q InferenceQueryType
	if err := json.Unmarshal([]byte(queryStr), &q); err != nil {
		fmt.Printf("failed to parse inference query: %v", err)
	}
	return q
}

func GetJsonRepresentationOfMap(variableString string) map[string]interface{} {
	var mapString map[string]interface{}
	err := json.Unmarshal([]byte(variableString), &mapString)
	if err != nil {
		fmt.Printf("%v", err)
	}
	return mapString
}

func GetFineTunedModelById(id string, dbCli *sql.DB) (*model.FinetunedModels, error) {
	var finetunedModel model.FinetunedModels
	uuid_id, err := uuid.Parse(id)
	if err != nil {
		return nil, err
	}

	selectStmt := FinetunedModels.SELECT(FinetunedModels.AllColumns).WHERE(FinetunedModels.ID.EQ(UUID(uuid_id)))

	err = selectStmt.Query(dbCli, &finetunedModel)
	if err != nil {
		return nil, fmt.Errorf("failed to fetch finetuned model by id: %s", id)
	}
	return &finetunedModel, nil

}

func updateInferenceFinetunedModels(inferenceID uuid.UUID, finetunedModelIDs []string, dbCli *sql.DB) error {
	// Start transaction
	tx, err := dbCli.Begin()
	if err != nil {
		return fmt.Errorf("failed to begin transaction: %w", err)
	}
	defer tx.Rollback() // Will be no-op if committed

	// 1. Delete existing relationships
	deleteStmt := InferenceFinetunedModel.
		DELETE().
		WHERE(InferenceFinetunedModel.InferenceID.EQ(UUID(inferenceID)))
	
	if _, err := deleteStmt.Exec(tx); err != nil {
		return fmt.Errorf("failed to delete existing relationships: %w", err)
	}

	// 2. Batch insert new relationships
	if len(finetunedModelIDs) > 0 {
		models := make([]model.InferenceFinetunedModel, len(finetunedModelIDs))
		for i, finetunedModelID := range finetunedModelIDs {
			uid, err := uuid.Parse(finetunedModelID)
			if err != nil {
				return fmt.Errorf("failed to parse finetunedModelID '%s': %w", finetunedModelID, err)
			}
			models[i] = model.InferenceFinetunedModel{
				InferenceID:      inferenceID,
				FinetunedModelID: uid,
			}
		}

		// Single INSERT with multiple rows
		insertStmt := InferenceFinetunedModel.
			INSERT(InferenceFinetunedModel.InferenceID, InferenceFinetunedModel.FinetunedModelID).
			MODELS(models)

		if _, err := insertStmt.Exec(tx); err != nil {
			return fmt.Errorf("failed to insert relationships: %w", err)
		}
	}

	// Commit transaction
	if err := tx.Commit(); err != nil {
		return fmt.Errorf("failed to commit transaction: %w", err)
	}

	return nil
}


func getAllFinetunedModels(dbCli *sql.DB) ([]*model.FinetunedModels, error) {
	var dest []*model.FinetunedModels
	selectStmt := FinetunedModels.SELECT(FinetunedModels.AllColumns).ORDER_BY(FinetunedModels.CreatedAt.DESC())
	err := selectStmt.Query(dbCli, &dest)
	if err != nil {
		return nil, err
	}
	return dest, nil
}

func getResultsInferSurya(model_id string, port string, query InferenceQueryType, results interface{}) error {

	jsonData, err := json.Marshal(query)
	if err != nil {
		log.Printf("%v", err)
		return err
	}
	serviceName := strings.Replace(model_id, "_", "-", -1)
	url := fmt.Sprintf("http://%s-service-lb:%s/api/v1/invocations", serviceName, port)
	resp, err := http.Post(url, "application/json", bytes.NewBuffer(jsonData))
	if err != nil {
		log.Printf("%v", err)
	}
	defer resp.Body.Close()

	return json.NewDecoder(resp.Body).Decode(results)

}

func insertInference(inferenceRequest InferenceQueueInput, status model.Inferencestatus, userEmail string, dbCli *sql.DB) (*InferenceOutput, error) {
	id, err := uuid.NewV7()
	if err != nil {
		return nil, err
	}
	query, err := json.Marshal(inferenceRequest.Body.Query)
	if err != nil {
		return nil, err
	}
	queryStr := string(query)

	inference := model.Inferences{
		ID:        id,
		Name:      inferenceRequest.Body.Name,
		CreatedAt: time.Now(),
		UpdatedAt: time.Now(),
		UserEmail: userEmail,
		Query:     &queryStr,
		Status:    status,
	}

	inferenceInsertStmt := Inferences.INSERT(Inferences.AllColumns).MODEL(inference)

	_, err = inferenceInsertStmt.Exec(dbCli)
	if err != nil {
		fmt.Println("Error inserting model:", err)
		return nil, err
	}

	var inferenceResult InferenceOutput
	inferenceResult.Body = inference

	return &inferenceResult, nil

}
