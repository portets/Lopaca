"""Runtime settings for Lopaca."""

from dataclasses import dataclass


@dataclass(frozen=True)
class Settings:
    server_url: str = "http://localhost:9931/v1"
    api_key: str = "sk-no-key-required"
    model: str = "local-model"
    temperature: float = 0.7
    # None lets llama-server use its configured generation limit.
    max_tokens: int | None = None
    request_timeout: int = 600
    connection_check_interval: int = 5
    preserve_thinking: bool = True
    max_context_chars: int = 12000
    chunk_chars: int = 1100
    chunk_overlap: int = 140
    enable_command_approval: bool = True
