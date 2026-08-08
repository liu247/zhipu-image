"""智谱视觉模型客户端(OpenAI 兼容接口)。

支持三种图片输入,统一归一化为智谱 API 需要的 URL 字段:

- 本地文件路径(如 ``/path/to/a.jpg``、``C:\\a.png``)→ 自动读取并转 base64 data URL
- ``file://`` URL → 同上
- ``http(s)://`` URL → 原样透传(需公网可访问)
- ``data:image/...;base64,...`` → 原样透传

该客户端不依赖 MCP,可独立在任何 Python 脚本中使用::

    from zhipu_image_mcp.vision import ZhipuVisionClient

    client = ZhipuVisionClient(api_key="...")
    text = await client.chat("这张图里有什么?", ["/tmp/a.png"])
"""

from __future__ import annotations

import base64
import mimetypes
from pathlib import Path
from urllib.parse import urlparse
from urllib.request import url2pathname

import httpx

from . import config

#: 请求超时秒数
DEFAULT_TIMEOUT = config.DEFAULT_TIMEOUT
#: 多图模式下智谱视觉模型一次最多接受的图片数
MAX_IMAGES = 8
#: 本地单张图片大小上限(与智谱 glm-4v-flash 上传限制一致)
MAX_IMAGE_BYTES = 5 * 1024 * 1024
#: 本地文件只允许读取这些图片扩展名,防止任意文件被读走外发
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".gif", ".webp", ".bmp"}

_MIME_FALLBACK = "image/jpeg"
#: 常见图片扩展名 -> MIME(兜底 mimetypes,某些平台可能缺失)
_EXTRA_MIME = {
    ".jpg": "image/jpeg",
    ".jpeg": "image/jpeg",
    ".png": "image/png",
    ".gif": "image/gif",
    ".webp": "image/webp",
    ".bmp": "image/bmp",
}


class VisionError(RuntimeError):
    """视觉模型调用失败(网络、鉴权、模型报错等),message 可直接展示给用户。"""


def normalize_image(image: str) -> str:
    """把任意图片输入归一化为智谱 API 接受的 ``url`` 字符串。

    - data URL / http(s) URL 原样返回
    - ``file://`` URL 与本地路径 → 读取为 base64 data URL

    本地文件会校验:扩展名必须是图片格式、大小 ≤ 5MB,
    且(若设置了 ``ZHIPU_IMAGE_ROOT``)必须位于该目录内。
    """
    image = image.strip()
    if not image:
        raise VisionError("图片路径为空")

    lowered = image.lower()
    if lowered.startswith("data:"):
        return image
    if lowered.startswith(("http://", "https://")):
        return image

    path = image
    if lowered.startswith("file://"):
        parsed = urlparse(image)
        if parsed.netloc not in ("", "localhost"):
            raise VisionError(
                f"不支持带主机名的 file:// URL: {parsed.netloc!r}"
            )
        path = url2pathname(parsed.path)

    return _path_to_data_url(Path(path))


def _allowed_root() -> Path | None:
    """ZHIPU_IMAGE_ROOT 目录白名单;未设置时返回 None(不限制)。"""
    raw = config.get_image_root()
    if not raw:
        return None
    try:
        return Path(raw).expanduser().resolve()
    except OSError:
        return None


def _path_to_data_url(path: Path) -> str:
    # resolve() 跟随符号链接,白名单判断基于真实路径
    try:
        resolved = path.expanduser().resolve()
    except OSError as exc:
        raise VisionError(f"无法解析图片路径: {path}") from exc

    root = _allowed_root()
    if root is not None and not resolved.is_relative_to(root):
        raise VisionError(
            f"路径不在允许的图片目录(ZHIPU_IMAGE_ROOT)内: {path}"
        )

    if not resolved.is_file():
        raise VisionError(f"本地图片不存在: {resolved}")

    suffix = resolved.suffix.lower()
    if suffix not in IMAGE_SUFFIXES:
        raise VisionError(
            f"不支持的图片格式: {suffix or '(无扩展名)'}"
            f"(仅支持 {'/'.join(sorted(s.lstrip('.') for s in IMAGE_SUFFIXES))})"
        )

    size = resolved.stat().st_size
    if size > MAX_IMAGE_BYTES:
        raise VisionError(
            f"图片超过 {MAX_IMAGE_BYTES // (1024 * 1024)}MB 限制: {size / (1024 * 1024):.1f}MB"
        )

    mime = _EXTRA_MIME.get(suffix) or mimetypes.guess_type(resolved.name)[0]
    if not mime or not mime.startswith("image/"):
        mime = _MIME_FALLBACK
    encoded = base64.b64encode(resolved.read_bytes()).decode("ascii")
    return f"data:{mime};base64,{encoded}"


def build_messages(prompt: str, images: list[str]) -> list[dict]:
    """构造智谱 chat.completions 需要的 messages。"""
    if not images:
        raise VisionError("至少需要提供一张图片")
    if len(images) > MAX_IMAGES:
        raise VisionError(f"一次最多支持 {MAX_IMAGES} 张图片,当前 {len(images)} 张")

    content: list[dict] = [{"type": "text", "text": prompt}]
    for img in images:
        content.append(
            {"type": "image_url", "image_url": {"url": normalize_image(img)}}
        )
    return [{"role": "user", "content": content}]


class ZhipuVisionClient:
    """智谱视觉模型客户端。

    Args:
        api_key: 智谱 API Key;不传时读取 ``ZHIPU_API_KEY`` 环境变量。
        base_url: API 端点;不传时读取 ``ZHIPU_BASE_URL``。
        model: 模型名;不传时读取 ``ZHIPU_IMAGE_MODEL``(默认 ``glm-4v-flash``)。
        timeout: 请求超时秒数。
        transport: 可选,仅供测试注入(MockTransport),正常使用请勿传入。
    """

    def __init__(
        self,
        api_key: str | None = None,
        base_url: str | None = None,
        model: str | None = None,
        timeout: float | None = None,
        transport: httpx.AsyncBaseTransport | None = None,
    ) -> None:
        # API Key 校验延迟到 chat() 时进行,保证输入校验(文件格式等)优先报错
        self.api_key = api_key or config.get_api_key()
        self.base_url = (base_url or config.get_base_url()).rstrip("/")
        self.model = model or config.get_model()
        self.timeout = timeout if timeout is not None else config.get_timeout()
        self._client = httpx.AsyncClient(timeout=self.timeout, transport=transport)

    async def close(self) -> None:
        await self._client.aclose()

    async def __aenter__(self) -> "ZhipuVisionClient":
        return self

    async def __aexit__(self, *exc: object) -> None:
        await self.close()

    def _headers(self) -> dict[str, str]:
        return {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }

    def _endpoint(self) -> str:
        return f"{self.base_url}/chat/completions"

    async def chat(
        self,
        prompt: str,
        images: list[str],
        *,
        temperature: float = 0.2,
        max_tokens: int | None = None,
        top_p: float | None = None,
    ) -> str:
        """发送一张或多张图片 + 文本提示,返回模型文本回答。

        Raises:
            VisionError: 缺少 API Key、网络错误、HTTP 错误或模型返回异常时。
        """
        payload: dict = {
            "model": self.model,
            "messages": build_messages(prompt, images),
            "temperature": temperature,
        }
        if max_tokens is not None:
            payload["max_tokens"] = max_tokens
        if top_p is not None:
            payload["top_p"] = top_p

        # 先做输入校验(build_messages),再校验 Key,让文件/路径错误优先暴露
        if not self.api_key:
            raise VisionError(
                "缺少 API Key:请设置环境变量 ZHIPU_API_KEY,"
                "或在创建 ZhipuVisionClient 时传入 api_key"
            )

        try:
            resp = await self._client.post(
                self._endpoint(), headers=self._headers(), json=payload
            )
        except httpx.HTTPError as exc:
            raise VisionError(f"请求智谱 API 失败: {exc}") from exc

        if resp.status_code >= 400:
            detail = _extract_error_detail(resp)
            raise VisionError(
                f"智谱 API 返回 HTTP {resp.status_code}: {detail}"
            )

        try:
            data = resp.json()
        except ValueError as exc:
            raise VisionError(
                f"智谱 API 返回了非 JSON 响应(HTTP {resp.status_code})"
            ) from exc

        try:
            return data["choices"][0]["message"]["content"]
        except (KeyError, IndexError, TypeError) as exc:
            raise VisionError(
                f"响应格式异常,缺少 choices[0].message.content: {str(data)[:200]}"
            ) from exc


def _extract_error_detail(resp: httpx.Response) -> str:
    try:
        body = resp.json()
        error = body.get("error") or body
        if isinstance(error, dict):
            message = error.get("message") or error.get("msg") or error
            return str(message)
        return str(error)
    except ValueError:
        return resp.text[:300]
