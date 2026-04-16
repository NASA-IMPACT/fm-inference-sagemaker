package preloadedevents

import (
	"context"
	"database/sql"
	"go-queue/.gen/fm_finetuning/public/model"
	. "go-queue/.gen/fm_finetuning/public/table"
	"go-queue/services/queue"
)

type preloadedEventJoinResult struct {
	model.PreloadedEvents

	Inference struct {
		model.Inferences
	}

	FinetunedModel struct {
		model.FinetunedModels
	}
}

func getAllPreloadedEventsWithJoin(ctx context.Context, dbCli *sql.DB) ([]*preloadedEventJoinResult, error) {
	var dest []*preloadedEventJoinResult

	selectStmt := PreloadedEvents.
		SELECT(
			PreloadedEvents.AllColumns,
			Inferences.AllColumns,
			FinetunedModels.AllColumns,
		).
		FROM(
			PreloadedEvents.
				LEFT_JOIN(Inferences, Inferences.ID.EQ(PreloadedEvents.InferenceID)).
				LEFT_JOIN(InferenceFinetunedModel, InferenceFinetunedModel.InferenceID.EQ(Inferences.ID)).
				LEFT_JOIN(FinetunedModels, FinetunedModels.ID.EQ(InferenceFinetunedModel.FinetunedModelID)),
		).
		ORDER_BY(PreloadedEvents.CreatedAt.DESC())

	err := selectStmt.QueryContext(ctx, dbCli, &dest)
	if err != nil {
		return nil, err
	}
	return dest, nil
}

func mapJoinResultsToResponse(rows []*preloadedEventJoinResult) []*PreloadedEventsCustom {
	eventMap := make(map[string]*PreloadedEventsCustom)
	var order []string

	for _, row := range rows {
		id := row.PreloadedEvents.ID.String()

		event, exists := eventMap[id]
		if !exists {
			event = &PreloadedEventsCustom{
				ID:           row.PreloadedEvents.ID,
				EventName:    row.PreloadedEvents.EventName,
				EventDetails: queue.GetJsonRepresentationOfMap(*row.PreloadedEvents.EventDetails),
				CreatedAt:    row.PreloadedEvents.CreatedAt,
				InferenceID:  row.PreloadedEvents.InferenceID,
			}

			if row.PreloadedEvents.InferenceID != nil {
				inf := row.Inference.Inferences
				event.Inference = queue.Inference{
					Id:        inf.ID,
					Name:      inf.Name,
					Status:    inf.Status,
					CreatedAt: inf.CreatedAt,
					UpdatedAt: inf.UpdatedAt,
				}
				if inf.Query != nil {
					event.Inference.Query = queue.ParseInferenceQuery(*inf.Query)
				}
				if inf.Results != nil {
					event.Inference.Results = queue.GetJsonRepresentationOfMap(*inf.Results)
				}
			}

			eventMap[id] = event
			order = append(order, id)
		}

		// Append finetuned model if present (handles the one-to-many from the JOIN)
		fm := row.FinetunedModel.FinetunedModels
		if fm.ID.String() != "00000000-0000-0000-0000-000000000000" {
			customFm := &queue.CustomFinetunedModel{
				ID:         fm.ID,
				Name:       fm.Name,
				SourceType: fm.SourceType.String(),
				CreatedAt:  fm.CreatedAt,
			}
			if fm.DataConfig != nil {
				customFm.DataConfig = queue.GetJsonRepresentationOfMap(*fm.DataConfig)
			}
			customFm.SourceDetails = queue.GetJsonRepresentationOfMap(fm.SourceDetails)
			event.Inference.FinetunedModels = append(event.Inference.FinetunedModels, customFm)
		}
	}

	result := make([]*PreloadedEventsCustom, 0, len(order))
	for _, id := range order {
		result = append(result, eventMap[id])
	}
	return result
}
