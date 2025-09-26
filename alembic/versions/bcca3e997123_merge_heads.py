"""merge heads

Revision ID: bcca3e997123
Revises: a1b2c3d4e5f6, de28ddce0e07
Create Date: 2025-09-15 10:43:06.173365

"""
from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision = 'bcca3e997123'
down_revision = ('a1b2c3d4e5f6', 'de28ddce0e07')
branch_labels = None
depends_on = None


def upgrade() -> None:
    pass


def downgrade() -> None:
    pass
