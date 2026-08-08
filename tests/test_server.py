"""server.py 测试:工具注册 + 工具函数的成功/失败路径(注入假客户端,不发真实请求)。"""

from __future__ import annotations

import asyncio

from zhipu_image_mcp import server
from zhipu_image_mcp.vision import VisionError

EXPECTED_TOOLS = {"describe_image", "ocr_image", "compare_images"}


class FakeClient:
    """记录调用参数的假视觉客户端。"""

    def __init__(self, reply: str = "模型回答") -> None:
        self.reply = reply
        self.calls: list[tuple[str, list[str]]] = []

    async def chat(self, prompt: str, images: list[str], **kwargs) -> str:
        self.calls.append((prompt, images))
        return self.reply


def test_tools_registered():
    tools = {t.name for t in asyncio.run(server.mcp.list_tools())}
    assert EXPECTED_TOOLS <= tools


async def test_describe_image_success(monkeypatch):
    fake = FakeClient()
    monkeypatch.setattr(server, "_get_client", lambda: fake)

    text = await server.describe_image("/tmp/a.png", question="它是什么颜色?")
    assert text == "模型回答"
    prompt, images = fake.calls[0]
    assert images == ["/tmp/a.png"]
    assert "颜色" in prompt


async def test_ocr_uses_ocr_prompt(monkeypatch):
    fake = FakeClient(reply="你好,世界")
    monkeypatch.setattr(server, "_get_client", lambda: fake)

    text = await server.ocr_image("https://a.com/ocr.png")
    assert text == "你好,世界"
    prompt, images = fake.calls[0]
    assert "文字" in prompt


async def test_compare_images(monkeypatch):
    fake = FakeClient()
    monkeypatch.setattr(server, "_get_client", lambda: fake)

    imgs = ["/tmp/1.png", "https://a.com/2.png"]
    await server.compare_images(imgs, question="两图差异?")
    prompt, images = fake.calls[0]
    assert images == imgs
    assert "差异" in prompt


async def test_vision_error_becomes_friendly_text(monkeypatch):
    class Boom:
        async def chat(self, *args, **kwargs):
            raise VisionError("缺少 API Key:请设置环境变量 ZHIPU_API_KEY")

    monkeypatch.setattr(server, "_get_client", lambda: Boom())
    text = await server.describe_image("/tmp/a.png")
    assert text.startswith("❌")
    assert "ZHIPU_API_KEY" in text


async def test_unexpected_error_becomes_friendly_text(monkeypatch):
    class Boom:
        async def chat(self, *args, **kwargs):
            raise RuntimeError("炸了")

    monkeypatch.setattr(server, "_get_client", lambda: Boom())
    text = await server.describe_image("/tmp/a.png")
    assert text.startswith("❌")
    assert "未知错误" in text
