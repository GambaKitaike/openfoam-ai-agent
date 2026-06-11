"""
設定管理 - .env ファイルや環境変数から設定を読み込む
"""
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
    )
    # LLM設定
    llm_provider: str = "openai"          # "openai" or "anthropic"
    llm_model: str = "gpt-4o"
    llm_model_mini: str = "gpt-4o-mini"   # case_scaffold 内の spec 変換など単発出力用
    openai_api_key: str = ""
    anthropic_api_key: str = ""

    # OpenFOAM設定
    openfoam_version: str = "2512"
    openfoam_root: str = "/usr/lib/openfoam/openfoam2512"

    # 出力設定
    default_output_dir: str = "./output"
