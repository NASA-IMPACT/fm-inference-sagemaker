package health

type HealthOutput struct {
	Body struct {
		Status string `json:"status"`
		Code   int16  `json:"code"`
	}
}
