"""Runtime settings for Lopaca."""

from dataclasses import dataclass, field
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


_UNSLOTH_9B = "unsloth/Qwen3.5-9B-MTP-GGUF"
_UNSLOTH_9B_REVISION = "9716a636ee4bddc3fed678220b7a33dd2a4160ae"
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
        6_874_345_824,
        "bb0aa4bf2acf4b6d97eca051a7af91200729e9e97449ce98785af9e70f6d703c",
        vision_family="3.5",
    ),
    "9b-q4": ModelDownload(
        "Qwen 3.5 9B",
        "Unsloth",
        "UD-Q4_K_XL",
        _UNSLOTH_9B,
        _UNSLOTH_9B_REVISION,
        "Qwen3.5-9B-UD-Q4_K_XL.gguf",
        6_135_034_208,
        "362f85a2d7dbc0259e926d5ac33ca0d0f17fd3753496d65bfd2106384c929d3f",
        vision_family="3.5",
    ),
    "9b-q3": ModelDownload(
        "Qwen 3.5 9B",
        "Unsloth",
        "UD-Q3_K_XL",
        _UNSLOTH_9B,
        _UNSLOTH_9B_REVISION,
        "Qwen3.5-9B-UD-Q3_K_XL.gguf",
        5_241_319_776,
        "437bb5639136b1b48807a5427f320321133383319e134e1687f339b38fe1a050",
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
    "27b-q4xl": ModelDownload(
        "Qwen 3.8 27B",
        "Unsloth",
        "UD-Q4_K_XL",
        _UNSLOTH_27B,
        _UNSLOTH_27B_REVISION,
        "Qwen3.8-27B-UD-Q4_K_XL.gguf",
        17_659_407_264,
        "2a5f3565f39cf885103da591182c1266da64745737b42b452fd3e675369f1211",
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
EMBEDDING_QUERY_INSTRUCTION = (
    "Given a search query, retrieve relevant passages that answer the query"
)


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
            918_165_984,
            "5a40d1f771686432172a4981018a0d30d03a5aaf5793a5badd5416573362a232",
        ),
        "BF16": VisionDownload(
            _UNSLOTH_9B,
            _UNSLOTH_9B_REVISION,
            921_704_928,
            "73c90471c349fec797f3969b59249259318ee3fa855a684da7ef35a46f063a9e",
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


# llama-server b12359 launch defaults. Set tier-specific changes in launch_parameters below.
SERVER_LAUNCH_DEFAULTS = {
    "ctx-size": 8192,               # Lopaca default, llama-server default was 0
    "n-gpu-layers": "auto",
    "threads": -1,
    "threads-batch": -1,
    "batch-size": 2048,
    "ubatch-size": 512,
    "parallel": 1,                  # Lopaca default, llama-server default was -1
    "cache-ram": 8192,              # Change below
    "cache-type-k": "f16",
    "cache-type-v": "f16",
    "flash-attn": "auto",
    "fit": "on",
    "fit-ctx": 4096,
    "fit-target": 512,              # Lopaca default, llama-server default was 1024
    "reasoning": "auto",
    "reasoning-budget": 16384,      # Lopaca default, llama-server default was -1
    "spec-type": "draft-mtp",       # Lopaca default, llama-server default was none
    "spec-draft-n-max": 2,          # Lopaca default, llama-server default was 3
}

DEFAULT_TEMPERATURE = 0.8  # Lopaca's existing default for other models and external servers.
SERVER_SAMPLING_DEFAULTS = {
    "temperature": DEFAULT_TEMPERATURE,
    "top_p": 0.95,
    "top_k": 30,
    "min_p": 0.02,
    "presence_penalty": 0.5,
    "repetition_penalty": 1.0,
}
# Match model filenames by family and parameter count, across quantizations.
MODEL_SAMPLING_DEFAULTS = {
    "qwen3.8-27b": {
        "temperature": 0.9,
        "top_p": 0.9,
        "top_k": 20,
        "min_p": 0.0,
        "presence_penalty": 0.5,
        "repetition_penalty": 1.0,
    },
    "qwen3.5-9b": {
        "temperature": 0.7,
        "top_p": 0.95,
        "top_k": 20,
        "min_p": 0.0,
        "presence_penalty": 0.5,
        "repetition_penalty": 1.0,
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
    vision: bool = True
    embedding: bool = True
    launch_parameters: dict[str, int | float | str] = field(default_factory=dict)


# First match wins. Discounts are for reported VRAM and RAM minor discrepancies.
# Vision and embedding control automatic setup. Users can install either manually.
# Example tier override: launch_parameters={"ctx-size": 65536, "fit-target": 512}.
MODEL_TOTAL_VRAM_DISCOUNT_GIB = 1
MODEL_FREE_VRAM_DISCOUNT_GIB = 2
MODEL_TOTAL_RAM_DISCOUNT_GIB = 1
MODEL_RECOMMENDATIONS = {
    "blackwell-32gb-hiram": ModelRecommendation(
        balanced="27b-nvfp4",
        faster="27b-q2",
        better="27b-q8",
        vision=True,
        embedding=True,
        backends=("cuda",),
        vram_gib=32,
        ram_gib=32,
        cuda_compute=100,
        launch_parameters={
            "ctx-size": 262144,
            "fit-target": 1024,
            "cache-ram": 16384,
            "spec-draft-n-max": 3,
        }
    ),
    "blackwell-32gb-": ModelRecommendation(
        balanced="27b-nvfp4",
        faster="27b-q2",
        better="27b-q8",
        vision=True,
        embedding=True,
        backends=("cuda",),
        vram_gib=32,
        cuda_compute=100,
        launch_parameters={
            "ctx-size": 262144,
            "fit-target": 1024,
            "spec-draft-n-max": 3,
        }
    ),
    "blackwell-24gb-hiram": ModelRecommendation(
        balanced="27b-nvfp4",
        faster="27b-q2",
        better="27b-q8",
        vision=True,
        embedding=True,
        backends=("cuda",),
        vram_gib=24,
        ram_gib=32,
        cuda_compute=100,
        launch_parameters={
            "ctx-size": 131072,
            "fit-target": 1024,
            "cache-ram": 16384,
            "spec-draft-n-max": 3,
        }
    ),
    "blackwell-24gb": ModelRecommendation(
        balanced="27b-nvfp4",
        faster="27b-q2",
        better="27b-q6",
        vision=True,
        embedding=True,
        backends=("cuda",),
        vram_gib=24,
        cuda_compute=100,
        launch_parameters={
            "ctx-size": 131072,
            "fit-target": 1024,
            "spec-draft-n-max": 3,
        }
    ),
    "blackwell-12gb-hiram": ModelRecommendation(
        balanced="27b-nvfp4",
        faster="27b-q2",
        better="27b-q6",
        vision=True,
        embedding=True,
        backends=("cuda",),
        vram_gib=12,
        ram_gib=32,
        cuda_compute=100,
        launch_parameters={
            "ctx-size": 65536,
        }
    ),
    "blackwell-12gb": ModelRecommendation(
        balanced="27b-nvfp4",
        faster="27b-q2",
        better="27b-q6",
        vision=True,
        embedding=True,
        backends=("cuda",),
        vram_gib=12,
        cuda_compute=100,
        launch_parameters={
            "ctx-size": 32768,
            "cache-ram": 4096,
            "cache-type-k": "q8_0",
            "cache-type-v": "q8_0",
        }
    ),
    "gpu-32gb-hiram": ModelRecommendation(
        balanced="27b-q4xl",
        faster="27b-q2",
        better="27b-q8",
        vision=True,
        embedding=True,
        backends=("cuda", "vulkan"),
        vram_gib=32,
        ram_gib=32,
        launch_parameters={
            "ctx-size": 262144,
            "fit-target": 1024,
            "cache-ram": 16384,
            "spec-draft-n-max": 3,
        }
    ),
    "gpu-32gb": ModelRecommendation(
        balanced="27b-q4xl",
        faster="27b-q2",
        better="27b-q8",
        vision=True,
        embedding=True,
        backends=("cuda", "vulkan"),
        vram_gib=32,
        launch_parameters={
            "ctx-size": 262144,
            "fit-target": 1024,
            "spec-draft-n-max": 3,
        }
    ),
    "gpu-24gb-hiram": ModelRecommendation(
        balanced="27b-q4xl",
        faster="27b-q2",
        better="27b-q8",
        vision=True,
        embedding=True,
        backends=("cuda", "vulkan"),
        vram_gib=24,
        ram_gib=32,
        launch_parameters={
            "ctx-size": 131072,
            "fit-target": 1024,
            "cache-ram": 16384,
            "spec-draft-n-max": 3,
        }
    ),

    "gpu-24gb": ModelRecommendation(
        balanced="27b-q4xl",
        faster="27b-q2",
        better="27b-q6",
        vision=True,
        embedding=True,
        backends=("cuda", "vulkan"),
        vram_gib=24,
        launch_parameters={
            "ctx-size": 131072,
            "fit-target": 1024,
            "spec-draft-n-max": 3,
        }
    ),

    "gpu-12gb-hiram": ModelRecommendation(
        balanced="27b-q4m",
        faster="27b-q2",
        better="27b-q6",
        vision=True,
        embedding=True,
        backends=("cuda", "vulkan"),
        vram_gib=12,
        ram_gib=32,
        launch_parameters={
            "ctx-size": 65536,
        }
    ),
    "gpu-12gb": ModelRecommendation(
        balanced="27b-q4m",
        faster="27b-q2",
        better="27b-q6",
        vision=True,
        embedding=True,
        backends=("cuda", "vulkan"),
        vram_gib=12,
        launch_parameters={
            "ctx-size": 32768,
            "cache-ram": 4096,
            "cache-type-k": "q8_0",
            "cache-type-v": "q8_0",
        }
    ),
    "gpu-8gb-hiram": ModelRecommendation(
        balanced="27b-q2",
        faster="9b-q4",
        better="27b-q4s",
        vision=True,
        embedding=True,
        backends=("cuda", "vulkan"),
        vram_gib=8,
        ram_gib=32,
        launch_parameters={
            "ctx-size": 32768,
            "cache-ram": 4096,
        }
    ),
    "gpu-8gb": ModelRecommendation(
        balanced="27b-q2",
        faster="9b-q4",
        better="27b-q4s",
        vision=True,
        embedding=True,
        backends=("cuda", "vulkan"),
        vram_gib=8,
        launch_parameters={
            "ctx-size": 16384,
            "cache-ram": 2048,
            "cache-type-k": "q8_0",
            "cache-type-v": "q8_0",
        }
    ),
    "gpu-6gb-hiram": ModelRecommendation(
        balanced="27b-q2",
        faster="9b-q4",
        better="27b-q4s",
        vision=True,
        embedding=True,
        backends=("cuda", "vulkan"),
        vram_gib=6,
        ram_gib=32,
        launch_parameters={
            "ctx-size": 16384,
            "cache-ram": 4096,
        }
    ),
    "gpu-6gb": ModelRecommendation(
        balanced="9b-q4",
        faster="9b-q3",
        better="27b-q2",
        vision=True,
        embedding=True,
        backends=("cuda", "vulkan"),
        vram_gib=6,
        launch_parameters={
            "ctx-size": 16384,
            "cache-ram": 2048,
            "cache-type-k": "q8_0",
            "cache-type-v": "q8_0",
            "spec-draft-n-max": 1,
        }
    ),
    # Includes CPU, unknown VRAM, GPUs under 6 GiB, but with 32gb RAM.
    "small": ModelRecommendation(
        balanced="9b-q4",
        faster="9b-q3",
        better="27b-q2",
        vision=True,
        embedding=True,
        ram_gib=32,
        launch_parameters={
            "ctx-size": 16384,
            "cache-ram": 2048,
            "spec-draft-n-max": 1,
        }
    ),
    # Includes remainder.
    "smaller": ModelRecommendation(
        balanced="9b-q4",
        faster="9b-q3",
        better="9b-q5",
        vision=True,
        embedding=False,
        launch_parameters={
            "ctx-size": 8192,
            "cache-ram": 1024,
            "cache-type-k": "q8_0",
            "cache-type-v": "q8_0",
            "spec-draft-n-max": 1,
        }
    ),
}


@dataclass(frozen=True)
class Settings:
    # Model connection and generation.
    server_url: str = "http://localhost:9931/v1"
    api_key: str = "sk-no-key-required"
    model: str = "local-model"
    temperature: float = DEFAULT_TEMPERATURE
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
