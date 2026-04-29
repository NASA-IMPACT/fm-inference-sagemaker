package auth

import "context"


const (
	ctxKeyEmail      string = "email"
	ctxKeyUserGroups string = "userGroups"
)

// WithEmail stores the email in the context.
func WithEmail(ctx context.Context, email string) context.Context {
	return context.WithValue(ctx, ctxKeyEmail, email)
}

// WithUserGroups stores user groups in the context.
func WithUserGroups(ctx context.Context, groups []string) context.Context {
	return context.WithValue(ctx, ctxKeyUserGroups, groups)
}

// EmailFromContext extracts the email from the request context.
func EmailFromContext(ctx context.Context) string {
	email, _ := ctx.Value(ctxKeyEmail).(string)
	return email
}

// UserGroupsFromContext extracts the user groups from the request context.
func UserGroupsFromContext(ctx context.Context) []string {
	groups, _ := ctx.Value(ctxKeyUserGroups).([]string)
	return groups
}
