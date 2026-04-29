package health

import (
	"context"
)

func GetHealth(ctx context.Context, input *struct{}) (*HealthOutput, error) {
	resp := &HealthOutput{}
	resp.Body.Status = "Healthy"
	resp.Body.Code = 200
	return resp, nil
}
