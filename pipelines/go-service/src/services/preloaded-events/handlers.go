package preloadedevents

import (
	"context"
	"database/sql"

	"github.com/danielgtaylor/huma/v2"
)

type Handler struct {
	dbCli *sql.DB
}

func NewHandler(dbCli *sql.DB) *Handler {
	return &Handler{dbCli: dbCli}
}

func (h *Handler) GetPreloadedEvents(ctx context.Context, input *struct{}) (*PreloadedEventsResponse, error) {
	rows, err := getAllPreloadedEventsWithJoin(ctx, h.dbCli)
	if err != nil {
		return nil, huma.NewError(500, "Failed to fetch preloaded events", err)
	}

	return &PreloadedEventsResponse{
		Body: mapJoinResultsToResponse(rows),
	}, nil
}
