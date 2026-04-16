# Use the official Go image as the base image
FROM golang:1.26-alpine AS builder

# Set the working directory inside the container
WORKDIR /app
# Copy go mod and sum files
ADD go-service/src .
# Download dependencies
RUN go mod download



# Build the application
RUN CGO_ENABLED=0 GOOS=linux go build -a -installsuffix cgo -o main ./cmd/

FROM alpine:latest

# Create a non-root user
RUN addgroup -g 1001 appgroup && \
    adduser -D -u 1001 -G appgroup appuser

WORKDIR /home/appuser

# Switch to non-root user
USER appuser
# Copy the binary from the builder stage
COPY --from=builder /app/main .



# Expose the port the app runs on
EXPOSE 8080

# Command to run the executable
ENTRYPOINT ["./main"]
