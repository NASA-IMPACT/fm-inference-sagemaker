package main

import (
	"os"
	"time"
)

type config struct {
	title              string
	version            string
	timeout            time.Duration
	readTimeout        time.Duration
	writeTimeout       time.Duration
	idleTimeout        time.Duration
	addr               string
	basePath           string
	jwtSecret         string
}

func getEnv(key, fallback string) string {
	if v := os.Getenv(key); v != "" {
		return v
	}
	return fallback
}
