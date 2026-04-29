package main

import (
	"fmt"
	"go-queue/auth"
	"log"
	"net/http"
	"strings"

	"github.com/MicahParks/keyfunc/v3"
	"github.com/danielgtaylor/huma/v2"
	"github.com/golang-jwt/jwt/v5"
)

// JWTAuth handles JWT token validation using either Cognito JWKS,
// a shared HMAC secret, or both.
type JWTAuth struct {
	jwks       keyfunc.Keyfunc
	issuer     string
	hmacSecret []byte
}

// NewJWTAuth creates a new JWTAuth instance. Cognito JWKS is configured
// when region and userPoolID are provided. Shared-secret HMAC is configured
// when jwtSecret is provided. At least one must be set.
func NewJWTAuth(jwtSecret string) (*JWTAuth, error) {
	a := &JWTAuth{}


	// Configure shared-secret HMAC if provided
	if jwtSecret != "" {
		a.hmacSecret = []byte(jwtSecret)
		log.Println("Using JWT shared-secret validation")
	}

	if a.jwks == nil && a.hmacSecret == nil {
		return nil, fmt.Errorf("no JWT validation method configured: set JWT_SECRET_KEY")
	}

	return a, nil
}

// Authenticate is a Huma middleware that validates the JWT Bearer token
// from the Authorization header and injects email and groups into the context.
func (j *JWTAuth) Authenticate(ctx huma.Context, next func(huma.Context)) {
	authHeader := ctx.Header("Authorization")
	if authHeader == "" {
		writeAuthError(ctx, http.StatusUnauthorized, "missing authorization header")
		return
	}

	parts := strings.SplitN(authHeader, " ", 2)
	if len(parts) != 2 || !strings.EqualFold(parts[0], "Bearer") {
		writeAuthError(ctx, http.StatusUnauthorized, "invalid authorization header format")
		return
	}

	tokenStr := parts[1]

	var token *jwt.Token
	var err error

	// Try Cognito JWKS if configured
	if j.jwks != nil {
		token, err = jwt.Parse(tokenStr, j.jwks.Keyfunc,
			jwt.WithIssuer(j.issuer),
			jwt.WithExpirationRequired(),
		)
	}

	// Try HMAC shared secret if JWKS not configured or failed
	if (token == nil || err != nil) && j.hmacSecret != nil {
		token, err = jwt.Parse(tokenStr, func(t *jwt.Token) (any, error) {
			if _, ok := t.Method.(*jwt.SigningMethodHMAC); !ok {
				return nil, fmt.Errorf("unexpected signing method: %v", t.Header["alg"])
			}
			return j.hmacSecret, nil
		}, jwt.WithExpirationRequired())
	}

	if err != nil || !token.Valid {
		log.Printf("JWT validation failed: %v", err)
		writeAuthError(ctx, http.StatusUnauthorized, "invalid or expired token")
		return
	}

	claims, ok := token.Claims.(jwt.MapClaims)
	if !ok {
		writeAuthError(ctx, http.StatusUnauthorized, "invalid token claims")
		return
	}

	email, _ := claims["email"].(string)

	var userGroups []string
	// Support both "cognito:groups" (Cognito tokens) and "groups" (shared-secret tokens)
	groupsClaim := claims["cognito:groups"]
	if groupsClaim == nil {
		groupsClaim = claims["groups"]
	}
	if groups, ok := groupsClaim.([]any); ok {
		for _, g := range groups {
			if s, ok := g.(string); ok {
				userGroups = append(userGroups, s)
			}
		}
	}

	reqCtx := auth.WithEmail(ctx.Context(), email)
	reqCtx = auth.WithUserGroups(reqCtx, userGroups)
	ctx = huma.WithContext(ctx, reqCtx)

	next(ctx)
}

func writeAuthError(ctx huma.Context, status int, detail string) {
	ctx.SetStatus(status)
	ctx.SetHeader("Content-Type", "application/json")
	fmt.Fprintf(ctx.BodyWriter(), `{"title":"Unauthorized","status":%d,"detail":"%s"}`, status, detail)
}
