"""Runtime settings for Lopaca."""

from dataclasses import dataclass
from pathlib import Path

APP_DIR = Path.home() / ".local" / "share" / "lopaca"
CHATS_DIR = APP_DIR / "chats"

CHAT_SORT_OPTIONS = (
    ("recently_used", "Recently Used"),
    ("creation_date", "Creation Date"),
    ("user_selectable", "User Selectable"),
)
CONVERSATION_STYLE_OPTIONS = (
    ("casual", "Casual"),
    ("neutral", "Neutral"),
    ("technical", "Technical"),
)


@dataclass(frozen=True)
class Settings:
    # Model connection and generation.
    server_url: str = "http://localhost:9931/v1"
    api_key: str = "sk-no-key-required"
    model: str = "local-model"
    temperature: float = 0.7
    # None lets llama-server use its configured generation limit.
    max_tokens: int | None = None
    request_timeout: int = 600
    thinking_selection_timeout: int = 15
    max_tool_rounds: int = 6
    connection_check_interval: int = 5

    # Conversation defaults and global memory limits.
    preserve_thinking: bool = True
    global_memories: bool = True
    memory_core_chars: int = 2000
    memory_context_chars: int = 4000
    memory_records: int = 500
    memory_core_records: int = 20
    memory_edit_output_tokens: int = 800
    # Injected by llama-server when its reasoning budget runs out.
    reasoning_budget_message: str = "Reasoning budget exceeded, continuing..."
    chat_sort: str = "recently_used"
    conversation_style: str = "neutral"

    # Document retrieval and model-assisted indexing.
    max_context_chars: int = 12000
    chunk_chars: int = 1100
    chunk_overlap: int = 140
    rag_results: int = 5
    rag_candidates: int = 40
    rag_query_expansion: bool = False
    rag_expansion_timeout: int = 10
    rag_model_indexing: bool = False
    history_model_indexing: bool = False
    rag_indexing_timeout: int = 600
    rag_indexing_retry_interval: int = 60
    rag_indexing_batch_size: int = 8
    rag_indexing_batch_chars: int = 6000
    rag_indexing_reasoning_tokens: int = 1024
    rag_indexing_output_tokens: int = 192

    # Approved commands and inspection limits.
    enable_command_approval: bool = True
    command_scan_chars: int = 262144
    command_scan_files: int = 8


CHAT_SETTING_DEFAULTS = {
    "global_memories": Settings.global_memories,
    "preserve_reasoning": Settings.preserve_thinking,
    "rag_query_expansion": Settings.rag_query_expansion,
    "rag_model_indexing": Settings.rag_model_indexing,
    "history_model_indexing": Settings.history_model_indexing,
    "conversation_style": Settings.conversation_style,
}
