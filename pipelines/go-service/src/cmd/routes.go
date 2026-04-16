package main

import (
	"database/sql"
	"go-queue/services/health"
	preloadedevents "go-queue/services/preloaded-events"
	"go-queue/services/queue"
	"net/http"

	"go.temporal.io/sdk/client"

	"github.com/danielgtaylor/huma/v2"
)

func generateRoutes(api huma.API, tc client.Client, dbCli *sql.DB, jwtAuth *JWTAuth) {

	huma.Register(api,
		huma.Operation{
			OperationID: "check-health",
			Description: "Check System Health",
			Path:        "/health",
			Method:      http.MethodGet,
		},
		health.GetHealth,
	)

	queueHandler := queue.NewHandler(tc, dbCli)

	huma.Register(api,
		huma.Operation{
			OperationID:   "queue-inference",
			Description:   "Queue an inference",
			Path:          "/queue",
			Method:        http.MethodPost,
			Middlewares:    huma.Middlewares{jwtAuth.Authenticate},
			Security:      []map[string][]string{{"bearerAuth": {}}},
		},
		queueHandler.QueueInference,
	)



}


func RegisterPreloadedEventsRoutes(api huma.API, dbCli *sql.DB) {
	preloadedEventsGroup := huma.NewGroup(api)
	
	preloadedEventsHandler := preloadedevents.NewHandler(dbCli)
	huma.Register(preloadedEventsGroup, huma.Operation{
		OperationID: "get-preloaded-events",
		Method:      http.MethodGet,
		Path:        "/preloaded-events",
		Summary:     "Get all preloaded events",
		Description: "Retrieve a list of all preloaded events.",
		Tags:        []string{"Preloaded Events"},
	}, preloadedEventsHandler.GetPreloadedEvents)

	// huma.Register(preloadedEventsGroup, huma.Operation{
	// 	OperationID: "create-preloaded-event",
	// 	Method:      http.MethodPost,
	// 	Path:        "/",
	// 	Summary:     "Create a preloaded event",
	// 	Description: "Create a new preloaded event",
	// 	Tags:        []string{"Preloaded Events"},
	// }, handlers.CreatePreloadedEvent)

	// huma.Register(preloadedEventsGroup, huma.Operation{
	// 	OperationID: "get-preloaded-event-by-id",
	// 	Method:      http.MethodGet,
	// 	Path:        "/{preloaded_event_id}",
	// 	Summary:     "Get a preloaded event by ID",
	// 	Description: "Get a preloaded event by ID",
	// 	Tags:        []string{"Preloaded Events"},
	// }, handlers.GetPreloadedEventById)

	// huma.Register(preloadedEventsGroup, huma.Operation{
	// 	OperationID: "get-models-by-preloaded-event-id",
	// 	Method:      http.MethodGet,
	// 	Path:        "/{preloaded_event_id}/models",
	// 	Summary:     "Get models by a preloaded event by ID",
	// 	Description: "Get models by a preloaded event by ID",
	// 	Tags:        []string{"Preloaded Events"},
	// }, handlers.GetModelsOfPreloadedEventByPreloadedEventId)

	// huma.Register(preloadedEventsGroup, huma.Operation{
	// 	OperationID: "delete-preloaded-event-by-id",
	// 	Method:      http.MethodDelete,
	// 	Path:        "/{preloaded_event_id}",
	// 	Summary:     "Delete a preloaded event by ID",
	// 	Description: "Delete a preloaded event by ID",
	// 	Tags:        []string{"Preloaded Events"},
	// }, handlers.DeletePreloadedEventById)

}
