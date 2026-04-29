import os
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.ext.declarative import declarative_base
from dotenv import load_dotenv

load_dotenv()

# Database URL from environment variable
DATABASE_URL = os.getenv(
    "DATABASE_URL", "postgresql://fm_user:fm_password@localhost:5432/fm_finetuning"
)

# Create SQLAlchemy engine
engine = create_engine(
    DATABASE_URL,
    pool_size=10,
    max_overflow=20,
    pool_timeout=30,  # fail fast instead of hanging forever
    pool_pre_ping=True,  # detect stale connections
)

# Create Base class
Base = declarative_base()


# Dependency to get database session
def get_db():
    # Create SessionLocal class
    SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
