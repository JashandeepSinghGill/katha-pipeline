from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_prefix="KATHA_", env_file=".env", extra="ignore"
    )

    database_url: str = "postgresql://katha:katha@localhost:5432/katha"

    watch_dir: Path = Path("/srv/media/katha/incoming")
    work_dir: Path = Path("/srv/media/katha/work")

    sarvam_api_key: str = ""
    sarvam_model: str = "saaras:v3"
    sarvam_mode: str = "transcribe"
    sarvam_language: str = "pa-IN"
    sarvam_timeout_seconds: int = 7200
    sarvam_poll_seconds: int = 15
    num_speakers: int = 2

    align_language: str = "pan"
    align_batch_size: int = 8

    device: str = "cuda"
    denoise_enabled: bool = True
    denoise_bin: str = "deepFilter"
    denoise_model: str = "DeepFilterNet3"

    max_audio_seconds: int = 7200
    # 4h: a 2h file denoising slower than realtime on CPU
    subprocess_timeout_seconds: int = 14400
    max_attempts: int = 3


def get_settings() -> Settings:
    return Settings()
