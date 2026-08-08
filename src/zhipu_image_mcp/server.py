"""FastMCP server:通过智谱视觉模型(GLM-4V 系列)提供图像理解工具。

启动方式:

- stdio(默认,供 Claude Desktop / Cursor / Windsurf 等客户端注册)::

      python -m zhipu_image_mcp
      # 或已安装时: zhipu-image-mcp

- streamable HTTP(供远程调用)::

      zhipu-image-mcp --transport streamable-http --port 8000
"""

from __future__ import annotations

import argparse
import logging
from contextlib import asynccontextmanager
from typing import Any

from fastmcp import FastMCP

from . import config
from .vision import VisionError, ZhipuVisionClient

logger = logging.getLogger(__name__)

DEFAULT_DESCRIBE = "请详细描述这张图片的内容,包括主体、场景、文字、颜色等细节。"
DEFAULT_OCR = (
    "请识别这张图片中的所有文字,按原有顺序原样输出;"
    "如果没有文字,请回复「未检测到文字」。"
)
DEFAULT_COMPARE = "请对比分析这些图片的异同,并分别说明每张图的要点。"

_client: ZhipuVisionClient | None = None


@asynccontextmanager
async def _lifespan(_server: FastMCP):
    """服务生命周期:退出时关闭底层 httpx 连接。"""
    try:
        yield
    finally:
        global _client
        if _client is not None:
            await _client.close()
            _client = None


mcp = FastMCP(
    "zhipu-image",
    lifespan=_lifespan,
    # 工具抛异常时向客户端隐藏内部堆栈/细节
    mask_error_details=True,
)


def _get_client() -> ZhipuVisionClient:
    """惰性创建全局客户端(API Key 在请求时校验)。"""
    global _client
    if _client is None:
        _client = ZhipuVisionClient()
    return _client


async def _run(prompt: str, images: list[str], **kwargs: Any) -> str:
    """执行一次视觉调用;失败转为可读文本,不让工具崩溃。"""
    try:
        return await _get_client().chat(prompt, images, **kwargs)
    except VisionError as exc:
        # 已知错误(缺 Key、HTTP 错误等),信息可直接展示给用户
        return f"❌ {exc}"
    except Exception as exc:  # noqa: BLE001 - 兜底,未知异常不外泄细节
        logger.exception("调用智谱视觉模型时发生未知错误")
        return "❌ 发生未知错误,请查看服务端日志"


@mcp.tool()
async def describe_image(image: str, question: str = DEFAULT_DESCRIBE) -> str:
    """使用智谱视觉模型(默认 glm-4v-flash)描述一张图片的内容。

    Args:
        image: 图片来源,支持本地文件路径、file:// URL、http(s) URL 或 base64 data URL。
        question: 对图片的提问,默认为详细描述。
    """
    return await _run(question, [image])


@mcp.tool()
async def ocr_image(image: str) -> str:
    """识别图片中的文字(OCR),按原顺序输出文本。

    Args:
        image: 图片来源,支持本地文件路径、file:// URL、http(s) URL 或 base64 data URL。
    """
    return await _run(DEFAULT_OCR, [image])


@mcp.tool()
async def compare_images(images: list[str], question: str = DEFAULT_COMPARE) -> str:
    """让视觉模型同时分析多张图片(最多 8 张),支持对比异同、找差异、归纳总结。

    Args:
        images: 图片来源列表(本地路径 / file:// / http(s) / data URL),最多 8 张。
        question: 多图分析的具体问题。
    """
    return await _run(question, images)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        prog="zhipu-image-mcp",
        description="通用 MCP Server:调用智谱视觉模型进行图像理解。"
        "需要环境变量 ZHIPU_API_KEY(模型可用 ZHIPU_IMAGE_MODEL 覆盖,默认 glm-4v-flash)。",
    )
    parser.add_argument(
        "--transport",
        choices=["stdio", "streamable-http"],
        default="stdio",
        help="传输方式(默认 stdio)",
    )
    parser.add_argument("--host", default="127.0.0.1", help="streamable-http 监听地址")
    parser.add_argument("--port", type=int, default=8000, help="streamable-http 监听端口")
    args = parser.parse_args(argv)

    if args.transport == "stdio":
        # stdio 模式下 stdout 是 MCP 协议通道,banner 不能输出
        mcp.run(show_banner=False)
    else:
        mcp.run(transport="streamable-http", host=args.host, port=args.port)


if __name__ == "__main__":
    main()
