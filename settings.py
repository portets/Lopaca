"""Runtime settings for Lopaca."""

from dataclasses import dataclass

CHAT_SORT_OPTIONS = (
    ("recently_used", "Recently Used"),
    ("creation_date", "Creation Date"),
    ("user_selectable", "User Selectable"),
)


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
    # Injected by llama-server when its reasoning budget runs out.
    reasoning_budget_message: str = "Reasoning budget exceeded, continuing..."
    chat_sort: str = "recently_used"
    max_context_chars: int = 12000
    chunk_chars: int = 1100
    chunk_overlap: int = 140
    code_agent: bool = True
    enable_command_approval: bool = True
    command_scan_chars: int = 262144
    command_scan_files: int = 8
