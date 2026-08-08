"""vision.py 单元测试:图片归一化、消息构造、API 调用(用 MockTransport,不发真实请求)。"""

from __future__ import annotations

import base64

import httpx
import pytest

from zhipu_image_mcp import config
from zhipu_image_mcp.vision import (
    MAX_IMAGES,
    VisionError,
    ZhipuVisionClient,
    build_messages,
    normalize_image,
)


def make_client(handler):
    """构造注入 MockTransport 的客户端。"""
    return ZhipuVisionClient(
        api_key="test-key",
        transport=httpx.MockTransport(handler),
    )


def json_response(status: int = 200, **body) -> httpx.Response:
    return httpx.Response(status, json=body)


class TestNormalizeImage:
    def test_data_url_passthrough(self):
        url = "data:image/png;base64,aGVsbG8="
        assert normalize_image(url) == url

    def test_http_url_passthrough(self):
        for url in ("https://example.com/a.png", "http://example.com/b.jpg"):
            assert normalize_image(url) == url

    def test_local_file_to_data_url(self, tmp_path):
        img = tmp_path / "photo.png"
        img.write_bytes(b"\x89PNG fake")
        result = normalize_image(str(img))
        assert result.startswith("data:image/png;base64,")
        payload = result.split(",", 1)[1]
        assert base64.b64decode(payload) == b"\x89PNG fake"

    def test_non_image_extension_rejected(self, tmp_path):
        img = tmp_path / "secret.txt"
        img.write_bytes(b"password=123")
        with pytest.raises(VisionError, match="不支持的图片格式"):
            normalize_image(str(img))

    def test_oversized_image_rejected(self, tmp_path):
        from zhipu_image_mcp.vision import MAX_IMAGE_BYTES

        img = tmp_path / "big.png"
        img.write_bytes(b"x" * (MAX_IMAGE_BYTES + 1))
        with pytest.raises(VisionError, match="5MB"):
            normalize_image(str(img))

    def test_image_root_whitelist(self, tmp_path, monkeypatch):
        from zhipu_image_mcp import config

        root = tmp_path / "images"
        root.mkdir()
        inside = root / "ok.png"
        inside.write_bytes(b"x")
        outside = tmp_path / "outside.png"
        outside.write_bytes(b"x")

        monkeypatch.setenv(config.ENV_IMAGE_ROOT, str(root))
        assert normalize_image(str(inside)).startswith("data:image/png;base64,")
        with pytest.raises(VisionError, match="目录\(ZHIPU_IMAGE_ROOT\)内"):
            normalize_image(str(outside))
        monkeypatch.delenv(config.ENV_IMAGE_ROOT)

    def test_image_root_symlink_cannot_escape(self, tmp_path, monkeypatch):
        """白名单内指向白名单外的符号链接必须被拒绝。"""
        import os as _os

        from zhipu_image_mcp import config

        root = tmp_path / "images"
        root.mkdir()
        outside = tmp_path / "outside.png"
        outside.write_bytes(b"x")
        link = root / "evil.png"
        _os.symlink(outside, link)

        monkeypatch.setenv(config.ENV_IMAGE_ROOT, str(root))
        with pytest.raises(VisionError, match="目录\(ZHIPU_IMAGE_ROOT\)内"):
            normalize_image(str(link))
        monkeypatch.delenv(config.ENV_IMAGE_ROOT)

    def test_image_root_not_set_allows_anywhere(self, tmp_path, monkeypatch):
        from zhipu_image_mcp import config

        monkeypatch.delenv(config.ENV_IMAGE_ROOT, raising=False)
        img = tmp_path / "a.png"
        img.write_bytes(b"x")
        assert normalize_image(str(img)).startswith("data:image/png;base64,")

    def test_file_url(self, tmp_path):
        img = tmp_path / "a.png"
        img.write_bytes(b"x")
        assert normalize_image(f"file://{img}").startswith("data:image/png;base64,")

    def test_file_url_with_host_rejected(self):
        with pytest.raises(VisionError, match="带主机名"):
            normalize_image("file://evil-host/etc/passwd")

    def test_missing_file_raises(self):
        with pytest.raises(VisionError, match="不存在"):
            normalize_image("/no/such/file.png")

    def test_empty_raises(self):
        with pytest.raises(VisionError, match="为空"):
            normalize_image("  ")


class TestBuildMessages:
    def test_single_image(self, tmp_path):
        img = tmp_path / "a.png"
        img.write_bytes(b"x")
        messages = build_messages("看图", [str(img)])
        content = messages[0]["content"]
        assert content[0] == {"type": "text", "text": "看图"}
        assert content[1]["type"] == "image_url"
        assert content[1]["image_url"]["url"].startswith("data:image/png;base64,")

    def test_url_images(self):
        messages = build_messages("q", ["https://a.com/1.png", "https://a.com/2.png"])
        urls = [c["image_url"]["url"] for c in messages[0]["content"][1:]]
        assert urls == ["https://a.com/1.png", "https://a.com/2.png"]

    def test_no_image_raises(self):
        with pytest.raises(VisionError, match="至少"):
            build_messages("q", [])

    def test_too_many_images_raises(self):
        images = [f"https://a.com/{i}.png" for i in range(MAX_IMAGES + 1)]
        with pytest.raises(VisionError, match="最多"):
            build_messages("q", images)


class TestZhipuVisionClient:
    async def test_missing_api_key(self, monkeypatch):
        monkeypatch.delenv(config.ENV_API_KEY, raising=False)
        client = ZhipuVisionClient()
        try:
            with pytest.raises(VisionError, match="ZHIPU_API_KEY"):
                await client.chat("q", ["https://a.com/1.png"])
        finally:
            await client.close()

    async def test_chat_success(self):
        import json

        def handler(request: httpx.Request) -> httpx.Response:
            assert request.headers["Authorization"] == "Bearer test-key"
            body = json.loads(request.read().decode())
            assert body["model"] == "glm-4v-flash"
            assert body["messages"][0]["role"] == "user"
            content = body["messages"][0]["content"]
            assert content[0] == {"type": "text", "text": "描述"}
            assert content[1]["image_url"]["url"] == "https://a.com/1.png"
            return json_response(
                choices=[{"message": {"content": "图中有只猫"}}]
            )

        client = make_client(handler)
        try:
            text = await client.chat("描述", ["https://a.com/1.png"])
            assert text == "图中有只猫"
        finally:
            await client.close()

    async def test_http_error_raises_with_detail(self):
        def handler(_: httpx.Request) -> httpx.Response:
            return json_response(
                status=401,
                error={"message": "Invalid API key"},
            )

        client = make_client(handler)
        try:
            with pytest.raises(VisionError, match="401.*Invalid API key"):
                await client.chat("q", ["https://a.com/1.png"])
        finally:
            await client.close()

    async def test_bad_response_shape_raises(self):
        def handler(_: httpx.Request) -> httpx.Response:
            return json_response(choices=[])

        client = make_client(handler)
        try:
            with pytest.raises(VisionError, match="响应格式异常"):
                await client.chat("q", ["https://a.com/1.png"])
        finally:
            await client.close()

    async def test_network_error_raises(self):
        client = make_client(lambda _: (_ for _ in ()).throw(httpx.ConnectError("boom")))
        try:
            with pytest.raises(VisionError, match="请求智谱 API 失败"):
                await client.chat("q", ["https://a.com/1.png"])
        finally:
            await client.close()
