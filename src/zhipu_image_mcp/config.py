"""运行配置。

所有配置项通过环境变量读取,支持运行时覆盖,方便在任意 MCP 客户端中
通过 ``env`` 字段注入,也方便在脚本中直接修改 ``os.environ`` 后重试。
"""

from __future__ import annotations

import os

DEFAULT_BASE_URL = "https://open.bigmodel.cn/api/paas/v4"
DEFAULT_MODEL = "glm-4v-flash"
DEFAULT_TIMEOUT = 60.0

#: 环境变量名一览(见 README「配置」一节)
ENV_API_KEY = "ZHIPU_API_KEY"
ENV_BASE_URL = "ZHIPU_BASE_URL"
ENV_MODEL = "ZHIPU_IMAGE_MODEL"
ENV_TIMEOUT = "ZHIPU_IMAGE_TIMEOUT"
#: 可选:本地图片目录白名单,设置后只允许读取该目录内的图片
ENV_IMAGE_ROOT = "ZHIPU_IMAGE_ROOT"


def get_api_key() -> str | None:
    """读取 ZHIPU_API_KEY;未设置时返回 None。"""
    return os.environ.get(ENV_API_KEY) or None


def get_image_root() -> str | None:
    """读取 ZHIPU_IMAGE_ROOT(可选);未设置时返回 None,表示不限制本地图片目录。"""
    return os.environ.get(ENV_IMAGE_ROOT) or None


def get_base_url() -> str:
    """读取 ZHIPU_BASE_URL(默认智谱开放平台 v4 端点),去掉尾部斜杠。"""
    return os.environ.get(ENV_BASE_URL, DEFAULT_BASE_URL).rstrip("/")


def get_model() -> str:
    """读取 ZHIPU_IMAGE_MODEL(默认 glm-4v-flash,免费)。"""
    return os.environ.get(ENV_MODEL, DEFAULT_MODEL)


def get_timeout() -> float:
    """读取 ZHIPU_IMAGE_TIMEOUT(秒),非法值回退默认 60s。"""
    try:
        return float(os.environ.get(ENV_TIMEOUT, DEFAULT_TIMEOUT))
    except (TypeError, ValueError):
        return DEFAULT_TIMEOUT
