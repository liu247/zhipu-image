"""zhipu-image-mcp:通用 MCP Server 库,调用智谱视觉模型进行图像理解。"""

from __future__ import annotations

from .config import (
    ENV_API_KEY,
    ENV_BASE_URL,
    ENV_IMAGE_ROOT,
    ENV_MODEL,
    ENV_TIMEOUT,
    get_api_key,
    get_base_url,
    get_image_root,
    get_model,
    get_timeout,
)
from .server import compare_images, describe_image, mcp, ocr_image
from .vision import (
    IMAGE_SUFFIXES,
    MAX_IMAGE_BYTES,
    MAX_IMAGES,
    VisionError,
    ZhipuVisionClient,
    build_messages,
    normalize_image,
)

__version__ = "0.1.0"

__all__ = [
    "__version__",
    # 配置
    "ENV_API_KEY",
    "ENV_BASE_URL",
    "ENV_IMAGE_ROOT",
    "ENV_MODEL",
    "ENV_TIMEOUT",
    "get_api_key",
    "get_base_url",
    "get_image_root",
    "get_model",
    "get_timeout",
    # MCP server
    "mcp",
    "describe_image",
    "ocr_image",
    "compare_images",
    # 独立客户端(不依赖 MCP)
    "ZhipuVisionClient",
    "VisionError",
    "build_messages",
    "normalize_image",
    # 本地文件校验常量
    "MAX_IMAGE_BYTES",
    "MAX_IMAGES",
    "IMAGE_SUFFIXES",
]
