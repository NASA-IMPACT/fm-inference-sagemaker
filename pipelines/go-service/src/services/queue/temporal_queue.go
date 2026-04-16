package queue

import (

	//"os"
	"context"
	"fmt"

	"github.com/google/uuid"
	"go.temporal.io/sdk/client"
)

func QueueInferenceTemporal(ctx context.Context, input QueueOrchestratorInput, tc client.Client) error {

	workflowId, _ := uuid.NewV7()
	// TODO make workflow and task queue env variables
	workflowOptions := client.StartWorkflowOptions{
		ID:        fmt.Sprintf("orchestrator-%s", workflowId),
		TaskQueue: "orchestrator-tq", // must match worker's task queue

	}
	_, err := tc.ExecuteWorkflow(
		ctx,
		workflowOptions,
		"InferenceOrchestrator", // use the string name to avoid importing worker code
		input,
	)
	if err != nil {
		return err
	}

	return nil

}
