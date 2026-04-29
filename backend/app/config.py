from pydantic import BaseSettings, PostgresDsn, Field


class Settings(BaseSettings):
    app_name: str = "fraud-detection-system"
    database_url: PostgresDsn = Field(
        "postgresql://postgres:password@localhost:5432/fraud_db",
        env="DATABASE_URL",
    )

    class Config:
        env_file = ".env"


settings = Settings()
