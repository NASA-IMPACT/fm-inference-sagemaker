from typing import List, TypeVar, Generic
from sqlalchemy.orm import Query
from fastapi import Response

T = TypeVar('T')


class PaginationHelper:
    """Helper class for handling pagination with response headers."""

    @staticmethod
    def paginate(
        query: Query,
        response: Response,
        base_url: str,
        skip: int = 0,
        limit: int = 10
    ) -> List[T]:
        """
        Paginate a SQLAlchemy query and set appropriate response headers.

        Args:
            query: SQLAlchemy query to paginate
            response: FastAPI Response object to set headers on
            base_url: Base URL for generating pagination links
            skip: Number of records to skip (offset)
            limit: Maximum number of records to return (page size)

        Returns:
            List of paginated results

        Sets the following response headers:
            - X-Total-Count: Total number of records
            - Link: RFC 8288 compliant pagination links (next, prev, first, last)
        """
        # Get total count
        total = query.count()

        # Set X-Total-Count header
        response.headers["X-Total-Count"] = str(total)

        # Generate Link header for pagination (RFC 8288)
        links = []

        # Next link
        if skip + limit < total:
            next_link = f'<{base_url}?skip={skip + limit}&limit={limit}>; rel="next"'
            links.append(next_link)

        # Previous link
        if skip > 0:
            previous_skip = max(0, skip - limit)
            previous_link = f'<{base_url}?skip={previous_skip}&limit={limit}>; rel="prev"'
            links.append(previous_link)

        # First link
        first_link = f'<{base_url}?skip=0&limit={limit}>; rel="first"'
        links.append(first_link)

        # Last link
        if total > 0:
            last_skip = max(0, ((total - 1) // limit) * limit)
            last_link = f'<{base_url}?skip={last_skip}&limit={limit}>; rel="last"'
            links.append(last_link)

        # Set Link header
        if links:
            response.headers["Link"] = ", ".join(links)

        # Apply pagination and return results
        return query.offset(skip).limit(limit).all()

    @staticmethod
    def get_pagination_params(skip: int = 0, limit: int = 10) -> tuple[int, int]:
        """
        Validate and normalize pagination parameters.

        Args:
            skip: Number of records to skip (offset)
            limit: Maximum number of records to return (page size)

        Returns:
            Tuple of (skip, limit) with validated values
        """
        # Ensure skip is non-negative
        skip = max(0, skip)

        # Ensure limit is between 1 and 100
        limit = max(1, min(100, limit))

        return skip, limit
