package main

import (
	"database/sql"
	"log"
	"time"

	"go.temporal.io/sdk/client"
)

func closeOpenConns(tc client.Client, dbCli *sql.DB) {
	if tc != nil {
		tc.Close()
	}
	if dbCli != nil {
		dbCli.Close()
	}

}

func main() {

	cfg := config{
		title:             "A simple API for queueing inferences",
		version:           "1.0.0",
		timeout:           time.Second * 10,
		readTimeout:       time.Second * 5,
		writeTimeout:      time.Second * 5,
		idleTimeout:       time.Second * 30,
		addr:              ":8080",
		basePath:          getEnv("QUEUE_BASE_PATH","/api/predict/v1/go"),
		jwtSecret:         getEnv("JWT_SECRET_KEY", ""),
	}

	// Create a new router & API
	chiRouter := getChiRouter()

	databaseURL := getEnv("DATABASE_URL", "postgres://user:password@localhost:5432/mydb")

	// Create database client

	DbClient, err := sql.Open("postgres", databaseURL)
	if err != nil {
		log.Fatalf("Failed to connect to database: %v", err)
	}

	DbClient.SetMaxOpenConns(25)
	DbClient.SetMaxIdleConns(10)
	DbClient.SetConnMaxLifetime(5 * time.Minute)
	DbClient.SetConnMaxIdleTime(1 * time.Minute)

	err = DbClient.Ping()
	if err != nil {
		log.Fatalf("Failed to ping database: %v", err)
	}

	// Create Temporal client once
	tc, err := client.Dial(client.Options{
		HostPort:  getEnv("TEMPORAL_SERVER_URL", "localhost:7233"),
		Namespace: getEnv("TEMPORAL_NAMESPACE", "default"),
	})
	if err != nil {
		log.Printf("Failed to connect to Temporal: %v", err)
	}
	defer closeOpenConns(tc, DbClient)

	// Initialize JWT auth
	jwtAuth, err := NewJWTAuth(cfg.jwtSecret)
	if err != nil {
		log.Fatalf("Failed to initialize JWT auth: %v", err)
	}

	api := chiRouter.generateApi(cfg)

	generateRoutes(api, tc, DbClient, jwtAuth)
	// // PreloadedEvents
	// resourcePath := "preloaded_events"
	// path := fmt.Sprintf("%s/%s", cfg.basePath, resourcePath)
	RegisterPreloadedEventsRoutes(api, DbClient)

	err = chiRouter.run(cfg)
	if err != nil {
		log.Fatalf("Error starting server: %v", err)
	}
}
