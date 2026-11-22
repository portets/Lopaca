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
class ModelDownload:
    name: str
    publisher: str
    quantization: str
    repository: str
    revision: str
    filename: str
    size: int
    sha256: str
    minimum_cuda_compute: int = 0
    vision_family: str = ""

    @property
    def url(self):
        return f"https://huggingface.co/{self.repository}/resolve/{self.revision}/{self.filename}"


_UNSLOTH_9B = "unsloth/Qwen3.5-9B-GGUF"
_UNSLOTH_9B_REVISION = "3885219b6810b007914f3a7950a8d1b469d598a5"
_UNSLOTH_27B = "unsloth/Qwen3.8-27B-GGUF"
_UNSLOTH_27B_REVISION = "4af51b3542f775b0a8e4b310bfb3f17e01c306c9"

MODEL_DOWNLOADS = {
# curl -L https://huggingface.co/unsloth/Qwen3.5-9B-GGUF/raw/main/Qwen3.5-9B-UD-Q4_K_XL.gguf
# to get size and sha256, change blob to raw in url
    "27b-nvfp4": ModelDownload(
        "Qwen 3.8 27B",
        "portets",
        "NVFP4 QAD",
        "portets/Qwen3.8-27b-NVFP4-QAD",
        "b420f63af95a528bc20322821b68278d945e31dd",
        "Qwen3.8-27B-NVFP4-QAD.gguf",
        18_993_871_040,
        "0b3882065b71d2c0e292f45ba45bc52e4ab413b10770fdaea1bc410dd0499681",
        minimum_cuda_compute=100,
        vision_family="3.8",
    ),
    "9b-q5": ModelDownload(
        "Qwen 3.5 9B",
        "Unsloth",
        "UD-Q5_K_XL",
        _UNSLOTH_9B,
        _UNSLOTH_9B_REVISION,
        "Qwen3.5-9B-UD-Q5_K_XL.gguf",
        6_743_680_224,
        "96cf42ddb97f9572410a72b9ed6f2299b1e887ee08da4c2a6c01e897cfa9f673",
        vision_family="3.5",
    ),
    "9b-q4": ModelDownload(
        "Qwen 3.5 9B",
        "Unsloth",
        "UD-Q4_K_XL",
        _UNSLOTH_9B,
        _UNSLOTH_9B_REVISION,
        "Qwen3.5-9B-UD-Q4_K_XL.gguf",
        5_966_095_584,
        "6f5d30666c2d8ae16a306e616d95341dcf3cc46810df84d7e6f5a7d1e4c1b293",
        vision_family="3.5",
    ),
    "9b-q3": ModelDownload(
        "Qwen 3.5 9B",
        "Unsloth",
        "UD-Q3_K_XL",
        _UNSLOTH_9B,
        _UNSLOTH_9B_REVISION,
        "Qwen3.5-9B-UD-Q3_K_XL.gguf",
        5_053_834_464,
        "aae0879e1be99ce93f0d56217f8185a399e25ad68a8ebbc095f362706283062f",
        vision_family="3.5",
    ),
    "27b-q8": ModelDownload(
        "Qwen 3.8 27B",
        "Unsloth",
        "UD-Q8_K_L",
        _UNSLOTH_27B,
        _UNSLOTH_27B_REVISION,
        "Qwen3.8-27B-UD-Q8_K_L.gguf",
        28_045_695_904,
        "2a13bba36d2efa213f9275abc430c40e4d914b145bfde976f30f1bb5b7e23ac2",
        vision_family="3.8",
    ),
    "27b-q6": ModelDownload(
        "Qwen 3.8 27B",
        "Unsloth",
        "UD-Q6_K_M",
        _UNSLOTH_27B,
        _UNSLOTH_27B_REVISION,
        "Qwen3.8-27B-UD-Q6_K_M.gguf",
        23_087_180_704,
        "3e253088108e612dac4fe845d73945234afb4e7095638711402b8317f8f65ca7",
        vision_family="3.8",
    ),
    "27b-q4m": ModelDownload(
        "Qwen 3.8 27B",
        "Unsloth",
        "UD-Q4_K_M",
        _UNSLOTH_27B,
        _UNSLOTH_27B_REVISION,
        "Qwen3.8-27B-UD-Q4_K_M.gguf",
        16_564_259_744,
        "1bce504dfecd84153666928a41d8a66706f0158d3a317bc5196b72819d7dc529",
        vision_family="3.8",
    ),
    "27b-q4s": ModelDownload(
        "Qwen 3.8 27B",
        "Unsloth",
        "UD-Q4_K_S",
        _UNSLOTH_27B,
        _UNSLOTH_27B_REVISION,
        "Qwen3.8-27B-UD-Q4_K_S.gguf",
        15_358_213_024,
        "75bc9c8adba2842e72f0ab5201aaa07133c5010b566305c09187fcbdcd364017",
        vision_family="3.8",
    ),
    "27b-q2": ModelDownload(
        "Qwen 3.8 27B",
        "Unsloth",
        "UD-Q2_K_XL",
        _UNSLOTH_27B,
        _UNSLOTH_27B_REVISION,
        "Qwen3.8-27B-UD-Q2_K_XL.gguf",
        9_948_666_784,
        "18f1187f38e7994b05c08cb45c674b90b7f17523687c83a4614fe493f7ea739d",
        vision_family="3.8",
    ),
}


EMBEDDING_DOWNLOAD = ModelDownload(
    "Qwen 3 Embedding 0.6B",
    "Qwen",
    "FP16",
    "Qwen/Qwen3-Embedding-0.6B-GGUF",
    "370f27d7550e0def9b39c1f16d3fbaa13aa67728",
    "Qwen3-Embedding-0.6B-f16.gguf",
    1_197_629_632,
    "421a27e58d165478cc7acb984a688c2aa41404968b0203e7cd743ece44c54340",
)
EMBEDDING_LEGACY_Q8 = ("Qwen3-Embedding-0.6B-Q8_0.gguf", 639_150_016)
EMBEDDING_CPU_AUTO_DOWNLOAD_MIN_RAM_GIB = 22 # for cpu-only main models
EMBEDDING_QUERY_INSTRUCTION = (
    "Given a search query, retrieve relevant passages that answer the query"
)
# Cosine cutoffs for semantic recall over the local embedding model's 256-dimensional vectors.
EMBEDDING_MEMORY_MINIMUM = 0.56
EMBEDDING_MEMORY_WITH_KEYWORD_MINIMUM = 0.62
EMBEDDING_HISTORY_MINIMUM = 0.54
EMBEDDING_HISTORY_WITH_KEYWORD_MINIMUM = 0.63
EMBEDDING_HISTORY_CANDIDATES = 24
EMBEDDING_DOCUMENT_CANDIDATES = 5
EMBEDDING_DOCUMENT_MINIMUM = 0.55
EMBEDDING_DOCUMENT_MARGIN = 0.07


@dataclass(frozen=True)
class VisionDownload:
    repository: str
    revision: str
    size: int
    sha256: str

    def url(self, precision):
        variant = "F16" if precision == "FP16" else precision
        return f"https://huggingface.co/{self.repository}/resolve/{self.revision}/mmproj-{variant}.gguf"


# These projectors match the Qwen 3.5 9B and Qwen 3.8 27B architectures above.
VISION_DOWNLOADS = {
    "3.5": {
        "FP16": VisionDownload(
            _UNSLOTH_9B,
            _UNSLOTH_9B_REVISION,
            918_166_080,
            "f70dc3509053962b0d0d3ee8a7eacebf5d60aa560cad78254ae8698516ae029f",
        ),
        "BF16": VisionDownload(
            _UNSLOTH_9B,
            _UNSLOTH_9B_REVISION,
            921_705_024,
            "853698ce7aa6c7ba732478bad280240969ddf7b0fcbf93900046f63903a83383",
        ),
    },
    "3.8": {
        "FP16": VisionDownload(
            _UNSLOTH_27B,
            _UNSLOTH_27B_REVISION,
            927_607_488,
            "cbb841a9ee0636b2ec172f5bb8df2ea8dfeb01e90fe7c6126581d662a0b4e43e",
        ),
        "BF16": VisionDownload(
            _UNSLOTH_27B,
            _UNSLOTH_27B_REVISION,
            931_146_432,
            "83ee4f4f205fa514161778c41df1ea14144faa0f713510893b63c2395f5c2d53",
        ),
    },
}


@dataclass(frozen=True)
class ModelRecommendation:
    balanced: str
    faster: str
    better: str
    backends: tuple[str, ...] = ()
    vram_gib: int = 0
    cuda_compute: int = 0
    ram_gib: int = 0


# First match wins. Discounts are for reported VRAM and RAM minor discrepancies.
MODEL_TOTAL_VRAM_DISCOUNT_GIB = 1
MODEL_FREE_VRAM_DISCOUNT_GIB = 2
MODEL_TOTAL_RAM_DISCOUNT_GIB = 2
MODEL_RECOMMENDATIONS = {
    "blackwell": ModelRecommendation(
        balanced="27b-nvfp4",
        faster="27b-q2",
        better="27b-q8",
        backends=("cuda",),
        vram_gib=12,
        cuda_compute=100,
    ),
    "gpu-12gb": ModelRecommendation(
        balanced="27b-q4m",
        faster="27b-q2",
        better="27b-q6",
        backends=("cuda", "vulkan"),
        vram_gib=12,
    ),
    "gpu-8gb": ModelRecommendation(
        balanced="27b-q2",
        faster="9b-q4",
        better="27b-q4s",
        backends=("cuda", "vulkan"),
        vram_gib=8,
    ),
    # Includes CPU, unknown VRAM, GPUs under 8 GiB, but with 32gb RAM.
    "small": ModelRecommendation(
        balanced="9b-q4",
        faster="9b-q3",
        better="27b-q4s",
        ram_gib=32,
    ),
    # Includes remainder.
    "smaller": ModelRecommendation(
        balanced="9b-q4",
        faster="9b-q3",
        better="9b-q5",
    ),
}


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
    max_tool_rounds: int = 12
    connection_check_interval: int = 5

    # Conversation defaults and global memory limits.
    preserve_thinking: bool = True
    global_memories: bool = True
    global_chat_histories: bool = True
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
    "rag_query_expansion": Settings.rag_query_expansion,
    "rag_model_indexing": Settings.rag_model_indexing,
    "history_model_indexing": Settings.history_model_indexing,
    "conversation_style": Settings.conversation_style,
}
