import os

from dotenv import load_dotenv


load_dotenv()


class Settings:
    """Lightweight settings reader that uses environment variables.

    This avoids requiring `pydantic.BaseSettings` (which may be unavailable
    or incompatible across environments). For local development the
    `DATABASE_URL` environment variable is respected; a safe default that
    points to the compose-mapped Postgres (host:127.0.0.1:5433) is used
    otherwise.
    """

    app_name: str = "fraud-detection-system"
    database_url: str = os.environ.get(
        "DATABASE_URL", "postgresql://protouser:protopass@127.0.0.1:5433/protodb"
    )


settings = Settings()
