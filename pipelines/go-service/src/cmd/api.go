package main

import (
	"context"
	"fmt"
	"github.com/danielgtaylor/huma/v2"
	"github.com/danielgtaylor/huma/v2/adapters/humachi"
	"github.com/go-chi/chi/v5"
	"github.com/go-chi/chi/v5/middleware"
	"log"
	"net/http"
	"os"
	"os/signal"
	"syscall"
	"strings"
)

type chiRouter struct {
	router *chi.Mux
}

func (chiR chiRouter) generateApi(cfg config) huma.API {
	r := chiR.router
	r.Use(middleware.RequestID)
	r.Use(middleware.RealIP)
	r.Use(middleware.Logger)
	r.Use(middleware.Recoverer)
	r.Use(middleware.Timeout(cfg.timeout))

	router := chi.NewRouter()
	basePath := cfg.basePath
	if basePath != "" {
		r.Mount(basePath, router)
	}

	serverURL := ""

	if basePath != "" {
		serverURL += basePath
	}

	config := huma.DefaultConfig(cfg.title, cfg.version)
	config.CreateHooks = []func(huma.Config) huma.Config{}
	config.Servers = []*huma.Server{{
		URL: serverURL,
	}}
	config.Components.SecuritySchemes = map[string]*huma.SecurityScheme{
		"bearerAuth": {
			Type:         "http",
			Scheme:       "bearer",
			BearerFormat: "JWT",
		},
	}

	return humachi.New(router, config)
}

func (app *chiRouter) run(cfg config) error {
	srv := &http.Server{
		Addr:         cfg.addr,
		Handler:      app.router,
		ReadTimeout:  cfg.readTimeout,
		WriteTimeout: cfg.writeTimeout,
		IdleTimeout:  cfg.idleTimeout,
	}

	// Start server in a goroutine
	errCh := make(chan error, 1)

	serverURL := fmt.Sprintf("http://%s%s/docs", cfg.addr, cfg.basePath)

	go func() {
		localServer := strings.ReplaceAll(serverURL, "http://:", "http://localhost:")
		log.Printf("Starting server on %s", localServer)
		if err := srv.ListenAndServe(); err != nil && err != http.ErrServerClosed {
			errCh <- err
		}
	}()

	// Wait for interrupt signal
	quit := make(chan os.Signal, 1)
	signal.Notify(quit, syscall.SIGINT, syscall.SIGTERM)

	select {
	case err := <-errCh:
		return fmt.Errorf("server failed to start: %w", err)
	case sig := <-quit:
		log.Printf("Received signal %s, shutting down...", sig)
	}

	// Give in-flight requests time to finish
	ctx, cancel := context.WithTimeout(context.Background(), cfg.timeout)
	defer cancel()

	if err := srv.Shutdown(ctx); err != nil {
		return fmt.Errorf("server forced to shutdown: %w", err)
	}

	log.Println("Server exited gracefully")
	return nil
}

func getChiRouter() *chiRouter {
	return &chiRouter{
		router: chi.NewRouter(),
	}
}
