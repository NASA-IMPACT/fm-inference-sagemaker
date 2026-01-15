"""add base_fm to finetuned_models

Revision ID: f7e8d9c0a1b2
Revises: bcca3e997123
Create Date: 2026-01-13 11:38:17.000000

"""
from alembic import op
import sqlalchemy as sa
import sys
from pathlib import Path

# Add src directory to path to import models
sys.path.append(str(Path(__file__).parent.parent.parent / 'src'))
from db.models import BaseFM


# revision identifiers, used by Alembic.
revision = 'f7e8d9c0a1b2'
down_revision = 'bcca3e997123'
branch_labels = None
depends_on = None


def upgrade() -> None:
    # Create the BaseFM enum type if it doesn't exist
    # Using raw SQL for better control and compatibility
    connection = op.get_bind()

    # Check if enum type already exists (PostgreSQL specific)
    result = connection.execute(
        sa.text("SELECT EXISTS (SELECT 1 FROM pg_type WHERE typname = 'basefm')")
    ).scalar()

    if not result:
        # Create enum type using raw SQL for better compatibility
        connection.execute(
            sa.text("CREATE TYPE basefm AS ENUM ('surya', 'prithvi')")
        )

    # Add the base_fm column to finetuned_models table
    op.add_column('finetuned_models',
        sa.Column('base_fm', sa.Enum(BaseFM, name='basefm'), nullable=True)
    )


def downgrade() -> None:
    # Drop the base_fm column
    op.drop_column('finetuned_models', 'base_fm')

    # Drop the BaseFM enum type
    connection = op.get_bind()
    connection.execute(sa.text("DROP TYPE IF EXISTS basefm"))
