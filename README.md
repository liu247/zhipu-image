# zhipu-image-mcp

通用 MCP Server 库:调用智谱(**GLM-4V 系列**)视觉模型进行图像理解。

- 支持**本地图片、`file://`、`http(s)` URL、base64 data URL** 四种输入,本地图片自动转 base64
- 默认使用 **`glm-4v-flash`**(免费),可随时切换到 `glm-4v-plus` / `glm-4.1v-thinking`
- **不绑定任何客户端**:以标准 stdio / streamable-http MCP 服务形式运行,注册方式见下文,也可以在任何 Python 脚本中直接当库用
- 只依赖 `mcp` + `httpx`,轻量、零框架耦合

## 安装

```bash
# 方式一:作为全局 CLI 安装
pip install zhipu-image-mcp        # 或: uv tool install zhipu-image-mcp

# 方式二:在某个项目里使用
uv add zhipu-image-mcp             # 或: pip install zhipu-image-mcp
```

也可以不安装,直接克隆后运行:

```bash
uv run --with zhipu-image-mcp zhipu-image-mcp
```

## 配置

| 环境变量 | 必填 | 默认值 | 说明 |
| --- | --- | --- | --- |
| `ZHIPU_API_KEY` | ✅ | — | 智谱开放平台 API Key([申请地址](https://open.bigmodel.cn/)) |
| `ZHIPU_IMAGE_MODEL` | — | `glm-4v-flash` | 视觉模型名 |
| `ZHIPU_BASE_URL` | — | `https://open.bigmodel.cn/api/paas/v4` | API 端点(一般不用改) |
| `ZHIPU_IMAGE_TIMEOUT` | — | `60` | 请求超时(秒) |
| `ZHIPU_IMAGE_ROOT` | — | (不限制) | 本地图片目录白名单,设置后只允许读取该目录内的图片 |

```bash
export ZHIPU_API_KEY=your_key_here
# 可选: export ZHIPU_IMAGE_MODEL=glm-4v-plus
```

## 启动

```bash
# stdio(默认,给桌面/编辑器客户端用)
zhipu-image-mcp            # 或: python -m zhipu_image_mcp

# streamable HTTP(远程调用)
zhipu-image-mcp --transport streamable-http --host 127.0.0.1 --port 8000
```

## 在任意 MCP 客户端注册

本库不自动写客户端配置文件,把下面任一示例填到你所用客户端的 MCP server 配置里即可。

### Claude Desktop

编辑 `~/Library/Application Support/Claude/claude_desktop_config.json`:

```json
{
  "mcpServers": {
    "zhipu-image": {
      "command": "zhipu-image-mcp",
      "env": { "ZHIPU_API_KEY": "your_key_here" }
    }
  }
}
```

### Cursor / Windsurf

项目根目录 `.mcp.json`(或 Cursor 设置 → MCP → Add global MCP server):

```json
{
  "mcpServers": {
    "zhipu-image": {
      "command": "zhipu-image-mcp",
      "env": { "ZHIPU_API_KEY": "your_key_here" }
    }
  }
}
```

### VS Code

`~/.vscode/mcp.json`(或项目 `.vscode/mcp.json`),配置同上。若 `zhipu-image-mcp` 不在 PATH,把 `command` 换成绝对路径或 `uv run zhipu-image-mcp`。

### Claude Code

```bash
claude mcp add zhipu-image -- zhipu-image-mcp
# 或通过环境变量传入 Key:
# claude mcp add --env ZHIPU_API_KEY=your_key_here zhipu-image -- zhipu-image-mcp
```

> 提示:如果全局未安装且不想装,把 `command` 替换为 `uv run --with zhipu-image-mcp zhipu-image-mcp`(需本机有 [uv](https://docs.astral.sh/uv/))。

## 提供的工具

| 工具 | 说明 |
| --- | --- |
| `describe_image(image, question?)` | 描述单张图片(默认详细描述;可自定义提问) |
| `ocr_image(image)` | 识别图片中的文字,按原顺序输出 |
| `compare_images(images, question?)` | 多张图片(≤8 张)同时分析:对比异同、找差异、归纳 |

所有 `image` 参数都支持:本地路径(`/tmp/a.png`)、`file://`、`http(s)://`、`data:image/...;base64,...`。

## 作为库独立使用

不依赖 MCP,在任意 Python 脚本/框架中直接调用:

```python
import asyncio
from zhipu_image_mcp import ZhipuVisionClient

async def main():
    async with ZhipuVisionClient(api_key="your_key_here") as client:
        text = await client.chat(
            "这张图里有什么?",
            ["/tmp/photo.png", "https://example.com/diagram.png"],
        )
        print(text)

asyncio.run(main())
```

## 模型切换

| 模型 | 特点 | 用途 |
| --- | --- | --- |
| `glm-4v-flash` | 免费、快 | 日常看图、OCR、常规描述(默认) |
| `glm-4v-plus` | 精度更高 | 需要细致理解的场景 |
| `glm-4.1v-thinking` | 带思考链 | 复杂推理、图表/题目解析 |

```bash
export ZHIPU_IMAGE_MODEL=glm-4v-plus
```

## 开发

```bash
uv venv --python 3.13 .venv
uv pip install -e ".[dev]"
uv run pytest
```

## 常见问题

- **工具返回 `❌ 缺少 API Key...`**:未设置 `ZHIPU_API_KEY`,检查客户端配置里的 `env`。
- **工具返回 `❌ 智谱 API 返回 HTTP 401...`**:Key 无效或余额不足。
- **本地图片不存在**:工具会提示路径不存在;网络图片需公网可访问。
- **图片上传限制**(glm-4v-flash):单张 ≤ 5MB,像素 ≤ 6000×6000,支持 jpg / png / jpeg。

## 安全说明

- 工具中的 `image` 参数由 AI/用户控制,本地文件只允许**图片扩展名**(jpg/jpeg/png/gif/webp/bmp)且 **≤ 5MB**,防止任意文件被读取并外发给智谱 API。
- 如需进一步限制,设置 `ZHIPU_IMAGE_ROOT` 为某个目录,此后只能读取该目录内的图片(基于解析后的真实路径判断,符号链接无法绕过)。
- 未知内部错误只记录到服务端日志,不会把堆栈/响应体细节返回给客户端。
