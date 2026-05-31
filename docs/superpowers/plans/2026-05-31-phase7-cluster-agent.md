# ShopMind AI Phase 7 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 构建 agent-cluster（MCP + Agent 集群），用 Claude 原生 Tool Use（非 LangGraph）将 6 个已有 Agent 封装为工具，接受自然语言任务描述，Claude 自主决策调用哪些 Agent、以什么顺序执行，支持跨 Agent 流水线。

**Architecture:** 不使用 LangGraph。`tools.py` 定义 6 个工具（每个 Agent 一个 dict），`executor.py` 负责 HTTP 调用对应 Agent，`orchestrator.py` 实现 Tool Use 循环（`ChatAnthropic.bind_tools` → 收到 tool_calls → execute → ToolMessage → 再次 invoke → 直到无 tool_calls）。每轮累计 token 用量，结果写入 PostgreSQL `cluster_jobs` 表。

**Tech Stack:** Python 3.11 · FastAPI · langchain-anthropic · langchain-core · httpx · psycopg2 · python-dotenv

---

## 参考资料

- 各 Agent 端口：cleaning=8001, customer=8002, media=8003, social=8004, clip=8005, training=8006
- Gateway 需配置：`AGENT_CLUSTER_URL=http://localhost:8007`

---

## 文件结构总览

```
agent-cluster/
├── main.py                    # FastAPI: POST /invoke, GET /health
├── config.py                  # env vars, AGENT_*_URL, get_llm()
├── tools.py                   # 6 个工具定义（dict 格式，Anthropic tool_use 兼容）
├── executor.py                # 按 tool_name 路由到对应 Agent HTTP /invoke
├── orchestrator.py            # Tool Use 循环：bind_tools → invoke → execute → repeat
├── utils/
│   ├── __init__.py
│   ├── startup.py             # 建 cluster_jobs 表
│   └── usage.py               # Token 统计（跨轮累计）
├── tests/
│   ├── __init__.py
│   ├── test_tools.py
│   ├── test_executor.py
│   ├── test_orchestrator.py
│   └── test_integration.py
├── requirements.txt
├── .env.example
└── .gitignore
```

---

## /invoke 端点契约

```
POST /invoke

Request:
{
  "task": "帮我生成一篇露营帐篷的小红书种草文案",
  "context": {}          # 可选，额外数据（如待清洗的商品列表）
}

Response:
{
  "job_id": "uuid",
  "result": "标题：露营新手必看！...\n\n正文：...",
  "agents_called": ["social_media"],
  "steps": [
    {
      "tool": "social_media",
      "input": {"product_name": "露营帐篷", "platform": "xiaohongshu", "tone": "种草"},
      "output": {"title": "...", "content": "...", "hashtags": [...]}
    }
  ],
  "usage": {"input_tokens": 1200, "output_tokens": 400}
}
```

---

## Task 1：项目初始化 + config + utils

**Files:**
- Create: `agent-cluster/requirements.txt`
- Create: `agent-cluster/.env.example`
- Create: `agent-cluster/.gitignore`
- Create: `agent-cluster/config.py`
- Create: `agent-cluster/utils/usage.py`
- Create: `agent-cluster/utils/startup.py`

- [ ] **Step 1: 创建目录结构**

```bash
cd /Users/liuyun/CCWorkSpace/shopmind-ai
mkdir -p agent-cluster/{utils,tests}
touch agent-cluster/utils/__init__.py
touch agent-cluster/tests/__init__.py
cd agent-cluster && git init && git checkout -b develop
git commit --allow-empty -m "chore: init agent-cluster"
```

- [ ] **Step 2: 创建 requirements.txt**

```
fastapi>=0.115.0
uvicorn[standard]>=0.30.0
langchain>=0.3.0
langchain-anthropic>=0.3.0
langchain-core>=0.3.0
httpx>=0.27.0
psycopg2-binary>=2.9.9
python-dotenv>=1.0.0
pytest>=8.0.0
pytest-asyncio>=0.24.0
```

- [ ] **Step 3: 创建 .env.example**

```
ANTHROPIC_API_KEY=your-key-here
CLAUDE_MODEL=claude-sonnet-4-6
DATABASE_URL=postgresql://user:pass@host/dbname?sslmode=require

AGENT_CLEANING_URL=http://localhost:8001
AGENT_CUSTOMER_URL=http://localhost:8002
AGENT_MEDIA_URL=http://localhost:8003
AGENT_SOCIAL_URL=http://localhost:8004
AGENT_CLIP_URL=http://localhost:8005
AGENT_TRAINING_URL=http://localhost:8006
```

- [ ] **Step 4: 创建 .gitignore**

```
__pycache__/
*.pyc
.env
.venv/
*.egg-info/
.pytest_cache/
```

- [ ] **Step 5: 创建 config.py**

```python
import os
from dotenv import load_dotenv

load_dotenv()

ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY", "")
CLAUDE_MODEL = os.getenv("CLAUDE_MODEL", "claude-sonnet-4-6")
DATABASE_URL = os.getenv("DATABASE_URL", "")

AGENT_URLS = {
    "data_cleaning":       os.getenv("AGENT_CLEANING_URL",  "http://localhost:8001"),
    "customer_service":    os.getenv("AGENT_CUSTOMER_URL",  "http://localhost:8002"),
    "media_library_search": os.getenv("AGENT_MEDIA_URL",   "http://localhost:8003"),
    "social_media":        os.getenv("AGENT_SOCIAL_URL",   "http://localhost:8004"),
    "live_clip":           os.getenv("AGENT_CLIP_URL",     "http://localhost:8005"),
    "sales_training":      os.getenv("AGENT_TRAINING_URL", "http://localhost:8006"),
}


def get_llm():
    from langchain_anthropic import ChatAnthropic
    return ChatAnthropic(model=CLAUDE_MODEL, api_key=ANTHROPIC_API_KEY)
```

- [ ] **Step 6: 创建 utils/usage.py**

```python
from typing import TypedDict

_INPUT_COST_PER_M = 3.0
_OUTPUT_COST_PER_M = 15.0


class TokenUsage(TypedDict):
    input_tokens: int
    output_tokens: int


def add(a: TokenUsage, response) -> TokenUsage:
    """Accumulate tokens from one LLM response into running total."""
    meta = getattr(response, "usage_metadata", None) or {}
    return {
        "input_tokens":  a["input_tokens"]  + meta.get("input_tokens", 0),
        "output_tokens": a["output_tokens"] + meta.get("output_tokens", 0),
    }


def zero() -> TokenUsage:
    return {"input_tokens": 0, "output_tokens": 0}


def cost_usd(usage: TokenUsage) -> float:
    return (
        usage["input_tokens"]  / 1_000_000 * _INPUT_COST_PER_M +
        usage["output_tokens"] / 1_000_000 * _OUTPUT_COST_PER_M
    )
```

- [ ] **Step 7: 创建 utils/startup.py**

```python
import psycopg2
from config import DATABASE_URL

_DDL = """
CREATE TABLE IF NOT EXISTS cluster_jobs (
    id            SERIAL PRIMARY KEY,
    job_id        VARCHAR(36) UNIQUE NOT NULL DEFAULT gen_random_uuid()::text,
    task          TEXT        NOT NULL,
    agents_called TEXT[]      DEFAULT '{}',
    result        TEXT,
    input_tokens  INTEGER     DEFAULT 0,
    output_tokens INTEGER     DEFAULT 0,
    created_at    TIMESTAMPTZ DEFAULT NOW()
);
"""


def check_and_init():
    if not DATABASE_URL:
        raise RuntimeError("DATABASE_URL is not set.")
    conn = psycopg2.connect(DATABASE_URL)
    try:
        with conn.cursor() as cur:
            cur.execute(_DDL)
        conn.commit()
        print("[startup] agent-cluster DB initialized.")
    finally:
        conn.close()


def save_job(job_id: str, task: str, agents_called: list[str],
             result: str, usage: dict):
    conn = psycopg2.connect(DATABASE_URL)
    try:
        with conn.cursor() as cur:
            cur.execute(
                "INSERT INTO cluster_jobs (job_id, task, agents_called, result, "
                "input_tokens, output_tokens) VALUES (%s,%s,%s,%s,%s,%s) "
                "ON CONFLICT (job_id) DO NOTHING",
                (job_id, task, agents_called, result,
                 usage.get("input_tokens", 0), usage.get("output_tokens", 0))
            )
        conn.commit()
    finally:
        conn.close()
```

- [ ] **Step 8: 安装并验证**

```bash
pip install -r requirements.txt -q
python -c "import fastapi, httpx, langchain_anthropic, langchain_core; print('imports OK')"
```

Expected: `imports OK`

- [ ] **Step 9: commit**

```bash
git add . && git commit -m "feat: project setup, config with AGENT_URLS, utils (startup + usage)"
```

---

## Task 2：tools.py（6 个工具定义）+ 测试

**Files:**
- Create: `agent-cluster/tools.py`
- Create: `agent-cluster/tests/test_tools.py`

- [ ] **Step 1: 写失败测试**

创建 `tests/test_tools.py`：

```python
def test_tools_list_has_six_entries():
    from tools import TOOLS
    assert len(TOOLS) == 6


def test_each_tool_has_required_fields():
    from tools import TOOLS
    for tool in TOOLS:
        assert "name" in tool, f"Tool missing 'name': {tool}"
        assert "description" in tool, f"Tool missing 'description': {tool}"
        assert "input_schema" in tool, f"Tool missing 'input_schema': {tool}"
        assert "type" in tool["input_schema"], f"input_schema missing 'type': {tool['name']}"
        assert "properties" in tool["input_schema"], f"input_schema missing 'properties': {tool['name']}"


def test_tool_names_match_agent_keys():
    from tools import TOOLS
    from config import AGENT_URLS
    tool_names = {t["name"] for t in TOOLS}
    agent_keys = set(AGENT_URLS.keys())
    assert tool_names == agent_keys, f"Tool names {tool_names} don't match agent keys {agent_keys}"


def test_social_media_tool_has_platform_enum():
    from tools import TOOLS
    social = next(t for t in TOOLS if t["name"] == "social_media")
    platform_prop = social["input_schema"]["properties"]["platform"]
    assert "enum" in platform_prop
    assert "xiaohongshu" in platform_prop["enum"]
```

- [ ] **Step 2: 运行，确认失败**

```bash
cd /Users/liuyun/CCWorkSpace/shopmind-ai/agent-cluster
python -m pytest tests/test_tools.py -v 2>&1 | tail -5
```

Expected: FAILED（ImportError）

- [ ] **Step 3: 创建 tools.py**

```python
TOOLS = [
    {
        "name": "data_cleaning",
        "description": "清洗电商商品或订单数据，修复格式、去重、标准化分类。适合数据混乱、格式不统一时使用。",
        "input_schema": {
            "type": "object",
            "properties": {
                "data": {
                    "type": "array",
                    "items": {"type": "object"},
                    "description": "待清洗的数据列表（商品/订单JSON数组）"
                },
                "data_type": {
                    "type": "string",
                    "enum": ["products", "orders", "customers"],
                    "description": "数据类型，默认products"
                }
            },
            "required": ["data"]
        }
    },
    {
        "name": "customer_service",
        "description": "回答客户关于退换货政策、物流时效、促销规则的问题，或查询客户订单状态。",
        "input_schema": {
            "type": "object",
            "properties": {
                "message": {
                    "type": "string",
                    "description": "客户的问题或查询内容"
                },
                "customer_id": {
                    "type": "string",
                    "description": "客户ID，用于查询个人账户数据，默认为1"
                }
            },
            "required": ["message"]
        }
    },
    {
        "name": "media_library_search",
        "description": "在商品素材库中语义搜索商品图片，返回匹配素材（图片URL+标签+描述+相似度分数）。",
        "input_schema": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "搜索关键词，如：红色跑鞋、户外防水帐篷"
                },
                "limit": {
                    "type": "integer",
                    "description": "返回结果数量，默认3"
                }
            },
            "required": ["query"]
        }
    },
    {
        "name": "social_media",
        "description": "为电商商品生成小红书/抖音/微信公众号营销文案，包含标题、正文和话题标签。",
        "input_schema": {
            "type": "object",
            "properties": {
                "product_name": {
                    "type": "string",
                    "description": "商品名称"
                },
                "platform": {
                    "type": "string",
                    "enum": ["xiaohongshu", "douyin", "wechat"],
                    "description": "目标发布平台"
                },
                "tone": {
                    "type": "string",
                    "enum": ["种草", "专业", "活泼"],
                    "description": "内容基调"
                },
                "key_features": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "商品核心特点列表，如[\"碳纤维中底\",\"轻量\"]"
                }
            },
            "required": ["product_name"]
        }
    },
    {
        "name": "live_clip",
        "description": "分析直播视频，用Whisper转录文字，识别精彩片段，用FFmpeg切片。适合直播录像的自动后期处理。",
        "input_schema": {
            "type": "object",
            "properties": {
                "video_url": {
                    "type": "string",
                    "description": "视频文件URL（MP4格式）"
                },
                "max_clips": {
                    "type": "integer",
                    "description": "最大切片数量，默认3"
                },
                "clip_duration": {
                    "type": "integer",
                    "description": "每个片段时长（秒），默认30"
                }
            },
            "required": ["video_url"]
        }
    },
    {
        "name": "sales_training",
        "description": "启动AI陪练销售对话，AI扮演指定类型的客户，与销售员进行角色扮演练习。",
        "input_schema": {
            "type": "object",
            "properties": {
                "product_name": {
                    "type": "string",
                    "description": "训练用的商品名称"
                },
                "scenario": {
                    "type": "string",
                    "enum": ["product_intro", "objection_handling", "closing"],
                    "description": "训练场景"
                },
                "customer_type": {
                    "type": "string",
                    "enum": ["skeptical", "hesitant", "price_sensitive"],
                    "description": "模拟客户类型"
                }
            },
            "required": ["product_name"]
        }
    },
]
```

- [ ] **Step 4: 运行，确认通过**

```bash
python -m pytest tests/test_tools.py -v 2>&1 | tail -8
```

Expected: 4 PASSED

- [ ] **Step 5: commit**

```bash
git add tools.py tests/test_tools.py
git commit -m "feat: 6 tool definitions for Claude Tool Use (one per agent)"
```

---

## Task 3：executor.py（HTTP 路由到各 Agent）+ 测试

**Files:**
- Create: `agent-cluster/executor.py`
- Create: `agent-cluster/tests/test_executor.py`

- [ ] **Step 1: 写失败测试**

创建 `tests/test_executor.py`：

```python
from unittest.mock import patch, MagicMock


def test_execute_social_media_sends_correct_payload():
    from executor import execute_tool
    with patch("executor.httpx.Client") as mock_cls:
        mock_client = MagicMock()
        mock_client.__enter__ = MagicMock(return_value=mock_client)
        mock_client.__exit__ = MagicMock(return_value=False)
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = {"title": "好鞋", "content": "...", "hashtags": []}
        mock_client.post.return_value = mock_resp
        mock_cls.return_value = mock_client

        result = execute_tool("social_media", {
            "product_name": "跑步鞋",
            "platform": "xiaohongshu",
            "tone": "种草"
        })

    assert result["title"] == "好鞋"
    call_json = mock_client.post.call_args[1]["json"]
    assert call_json["product_name"] == "跑步鞋"


def test_execute_media_library_search_sends_action():
    from executor import execute_tool
    with patch("executor.httpx.Client") as mock_cls:
        mock_client = MagicMock()
        mock_client.__enter__ = MagicMock(return_value=mock_client)
        mock_client.__exit__ = MagicMock(return_value=False)
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = {"results": []}
        mock_client.post.return_value = mock_resp
        mock_cls.return_value = mock_client

        execute_tool("media_library_search", {"query": "红色跑鞋", "limit": 3})

    call_json = mock_client.post.call_args[1]["json"]
    assert call_json["action"] == "search"
    assert call_json["query"] == "红色跑鞋"


def test_execute_unknown_tool_returns_error():
    from executor import execute_tool
    result = execute_tool("nonexistent_tool", {})
    assert "error" in result


def test_execute_handles_agent_service_down():
    from executor import execute_tool
    with patch("executor.httpx.Client") as mock_cls:
        mock_client = MagicMock()
        mock_client.__enter__ = MagicMock(return_value=mock_client)
        mock_client.__exit__ = MagicMock(return_value=False)
        mock_client.post.side_effect = Exception("connection refused")
        mock_cls.return_value = mock_client
        result = execute_tool("customer_service", {"message": "你好"})
    assert "error" in result
```

- [ ] **Step 2: 运行，确认失败**

```bash
python -m pytest tests/test_executor.py -v 2>&1 | tail -5
```

Expected: FAILED（ImportError）

- [ ] **Step 3: 创建 executor.py**

```python
import json
import httpx
from config import AGENT_URLS


def _build_payload(tool_name: str, tool_input: dict) -> dict:
    """Map tool input to the specific agent's /invoke payload format."""
    if tool_name == "data_cleaning":
        return {
            "data": tool_input.get("data", []),
            "data_type": tool_input.get("data_type", "products"),
        }
    elif tool_name == "customer_service":
        return {
            "message": tool_input.get("message", ""),
            "customer_id": tool_input.get("customer_id", "1"),
        }
    elif tool_name == "media_library_search":
        return {
            "action": "search",
            "query": tool_input.get("query", ""),
            "limit": tool_input.get("limit", 3),
        }
    elif tool_name == "social_media":
        return {
            "product_name": tool_input.get("product_name", ""),
            "platform": tool_input.get("platform", "xiaohongshu"),
            "tone": tool_input.get("tone", "种草"),
            "key_features": tool_input.get("key_features", []),
        }
    elif tool_name == "live_clip":
        return {
            "video_url": tool_input.get("video_url", ""),
            "max_clips": tool_input.get("max_clips", 3),
            "clip_duration": tool_input.get("clip_duration", 30),
        }
    elif tool_name == "sales_training":
        return {
            "action": "start",
            "product_name": tool_input.get("product_name", ""),
            "scenario": tool_input.get("scenario", "product_intro"),
            "customer_type": tool_input.get("customer_type", "hesitant"),
        }
    return tool_input


def execute_tool(tool_name: str, tool_input: dict) -> dict:
    """Call the corresponding agent service and return its response."""
    url = AGENT_URLS.get(tool_name)
    if not url:
        return {"error": f"Unknown tool: {tool_name}"}

    payload = _build_payload(tool_name, tool_input)
    try:
        with httpx.Client(timeout=60.0) as client:
            resp = client.post(f"{url}/invoke", json=payload)
            if resp.status_code == 200:
                return resp.json()
            return {"error": f"Agent returned HTTP {resp.status_code}"}
    except Exception as e:
        return {"error": str(e)}
```

- [ ] **Step 4: 运行，确认通过**

```bash
python -m pytest tests/test_executor.py -v 2>&1 | tail -7
```

Expected: 4 PASSED

- [ ] **Step 5: commit**

```bash
git add executor.py tests/test_executor.py
git commit -m "feat: executor routes tool calls to agent HTTP services"
```

---

## Task 4：orchestrator.py（Tool Use 循环）+ 测试

**Files:**
- Create: `agent-cluster/orchestrator.py`
- Create: `agent-cluster/tests/test_orchestrator.py`

- [ ] **Step 1: 写失败测试**

创建 `tests/test_orchestrator.py`：

```python
import json
from unittest.mock import patch, MagicMock
from langchain_core.messages import AIMessage, ToolMessage


def _make_ai_with_tool_call(tool_name: str, tool_args: dict, call_id: str = "call_001") -> AIMessage:
    return AIMessage(
        content="",
        tool_calls=[{"name": tool_name, "args": tool_args, "id": call_id}]
    )


def _make_ai_final(text: str) -> AIMessage:
    return AIMessage(content=text)


def test_orchestrate_single_tool_call():
    from orchestrator import orchestrate
    mock_llm = MagicMock()
    mock_llm.bind_tools.return_value = mock_llm
    mock_llm.invoke.side_effect = [
        _make_ai_with_tool_call("social_media", {"product_name": "跑步鞋"}, "c1"),
        _make_ai_final("已为您生成文案：好鞋推荐！"),
    ]
    with patch("orchestrator.get_llm", return_value=mock_llm), \
         patch("orchestrator.execute_tool") as mock_exec:
        mock_exec.return_value = {"title": "好鞋", "content": "推荐！", "hashtags": []}
        result = orchestrate("生成跑步鞋文案", {})

    assert "social_media" in result["agents_called"]
    assert result["result"] == "已为您生成文案：好鞋推荐！"
    assert len(result["steps"]) == 1
    assert result["steps"][0]["tool"] == "social_media"


def test_orchestrate_multi_tool_pipeline():
    from orchestrator import orchestrate
    mock_llm = MagicMock()
    mock_llm.bind_tools.return_value = mock_llm
    mock_llm.invoke.side_effect = [
        _make_ai_with_tool_call("data_cleaning", {"data": [], "data_type": "products"}, "c1"),
        _make_ai_with_tool_call("social_media", {"product_name": "跑步鞋"}, "c2"),
        _make_ai_final("数据已清洗，文案已生成。"),
    ]
    with patch("orchestrator.get_llm", return_value=mock_llm), \
         patch("orchestrator.execute_tool") as mock_exec:
        mock_exec.return_value = {"result": "ok"}
        result = orchestrate("先清洗数据再生成文案", {"data": []})

    assert "data_cleaning" in result["agents_called"]
    assert "social_media" in result["agents_called"]
    assert len(result["steps"]) == 2


def test_orchestrate_no_tool_needed():
    from orchestrator import orchestrate
    mock_llm = MagicMock()
    mock_llm.bind_tools.return_value = mock_llm
    mock_llm.invoke.return_value = _make_ai_final("这是一个简单问候，不需要调用任何工具。")
    with patch("orchestrator.get_llm", return_value=mock_llm), \
         patch("orchestrator.execute_tool") as mock_exec:
        result = orchestrate("你好", {})

    mock_exec.assert_not_called()
    assert result["agents_called"] == []
    assert "不需要" in result["result"]


def test_orchestrate_accumulates_usage():
    from orchestrator import orchestrate
    mock_resp1 = _make_ai_with_tool_call("social_media", {"product_name": "商品"}, "c1")
    mock_resp1.usage_metadata = {"input_tokens": 300, "output_tokens": 50}
    mock_resp2 = _make_ai_final("完成")
    mock_resp2.usage_metadata = {"input_tokens": 400, "output_tokens": 100}

    mock_llm = MagicMock()
    mock_llm.bind_tools.return_value = mock_llm
    mock_llm.invoke.side_effect = [mock_resp1, mock_resp2]
    with patch("orchestrator.get_llm", return_value=mock_llm), \
         patch("orchestrator.execute_tool", return_value={}):
        result = orchestrate("生成文案", {})

    assert result["usage"]["input_tokens"] == 700
    assert result["usage"]["output_tokens"] == 150
```

- [ ] **Step 2: 运行，确认失败**

```bash
python -m pytest tests/test_orchestrator.py -v 2>&1 | tail -5
```

Expected: FAILED（ImportError）

- [ ] **Step 3: 创建 orchestrator.py**

```python
import json
from langchain_core.messages import HumanMessage, ToolMessage
from config import get_llm
from tools import TOOLS
from executor import execute_tool
from utils.usage import zero, add

_MAX_ITERATIONS = 10


def orchestrate(task: str, context: dict) -> dict:
    """
    Run the Claude Tool Use loop until end_turn.
    Returns result text, list of agents called, steps detail, usage totals.
    """
    llm = get_llm().bind_tools(TOOLS)

    content = task
    if context:
        content += f"\n\n附加上下文：{json.dumps(context, ensure_ascii=False)}"

    messages = [HumanMessage(content=content)]
    agents_called: list[str] = []
    steps: list[dict] = []
    usage = zero()

    for _ in range(_MAX_ITERATIONS):
        response = llm.invoke(messages)
        usage = add(usage, response)
        messages.append(response)

        if not response.tool_calls:
            final_text = response.content if isinstance(response.content, str) else str(response.content)
            break

        tool_messages = []
        for tc in response.tool_calls:
            tool_name = tc["name"]
            tool_input = tc["args"]

            agents_called.append(tool_name)
            output = execute_tool(tool_name, tool_input)
            steps.append({"tool": tool_name, "input": tool_input, "output": output})

            tool_messages.append(ToolMessage(
                content=json.dumps(output, ensure_ascii=False),
                tool_call_id=tc["id"],
            ))

        messages.extend(tool_messages)
    else:
        final_text = "任务已执行，但达到最大迭代次数限制。"

    return {
        "result":        final_text,
        "agents_called": list(dict.fromkeys(agents_called)),
        "steps":         steps,
        "usage":         usage,
    }
```

- [ ] **Step 4: 运行，确认通过**

```bash
python -m pytest tests/test_orchestrator.py -v 2>&1 | tail -7
```

Expected: 4 PASSED

- [ ] **Step 5: commit**

```bash
git add orchestrator.py tests/test_orchestrator.py
git commit -m "feat: orchestrator Tool Use loop (bind_tools → execute → iterate → end_turn)"
```

---

## Task 5：main.py + 集成测试

**Files:**
- Create: `agent-cluster/main.py`
- Create: `agent-cluster/tests/test_integration.py`

- [ ] **Step 1: 创建 main.py**

```python
import uuid
from contextlib import asynccontextmanager
from fastapi import FastAPI
from pydantic import BaseModel
from utils.startup import check_and_init, save_job
from orchestrator import orchestrate


@asynccontextmanager
async def lifespan(app: FastAPI):
    check_and_init()
    yield


app = FastAPI(title="ShopMind Cluster Agent", version="1.0.0", lifespan=lifespan)


class InvokeRequest(BaseModel):
    task: str
    context: dict = {}


@app.post("/invoke")
def invoke(req: InvokeRequest):
    job_id = str(uuid.uuid4())
    result = orchestrate(req.task, req.context)

    save_job(
        job_id=job_id,
        task=req.task,
        agents_called=result["agents_called"],
        result=result["result"],
        usage=result["usage"],
    )

    return {
        "job_id":        job_id,
        "result":        result["result"],
        "agents_called": result["agents_called"],
        "steps":         result["steps"],
        "usage":         result["usage"],
    }


@app.get("/health")
def health():
    return {"status": "ok", "service": "agent-cluster"}
```

- [ ] **Step 2: 写集成测试**

创建 `tests/test_integration.py`：

```python
import pytest
from unittest.mock import patch, MagicMock
from fastapi.testclient import TestClient


@pytest.fixture
def client():
    with patch("utils.startup.check_and_init"), \
         patch("utils.startup.save_job"):
        import importlib
        import main as m
        importlib.reload(m)
        yield TestClient(m.app)


def test_invoke_returns_complete_response(client):
    with patch("main.orchestrate", return_value={
        "result": "已生成文案：好鞋推荐！",
        "agents_called": ["social_media"],
        "steps": [{"tool": "social_media", "input": {}, "output": {"title": "好鞋"}}],
        "usage": {"input_tokens": 500, "output_tokens": 150},
    }):
        resp = client.post("/invoke", json={
            "task": "生成跑步鞋小红书文案",
            "context": {},
        })
    assert resp.status_code == 200
    body = resp.json()
    assert "job_id" in body
    assert body["result"] == "已生成文案：好鞋推荐！"
    assert "social_media" in body["agents_called"]
    assert body["usage"]["input_tokens"] == 500


def test_invoke_assigns_unique_job_id(client):
    with patch("main.orchestrate", return_value={
        "result": "完成", "agents_called": [], "steps": [],
        "usage": {"input_tokens": 100, "output_tokens": 20},
    }):
        r1 = client.post("/invoke", json={"task": "任务"})
        r2 = client.post("/invoke", json={"task": "任务"})
    assert r1.json()["job_id"] != r2.json()["job_id"]


def test_invoke_with_context(client):
    with patch("main.orchestrate", return_value={
        "result": "数据已清洗", "agents_called": ["data_cleaning"],
        "steps": [], "usage": {"input_tokens": 300, "output_tokens": 80},
    }) as mock_orch:
        resp = client.post("/invoke", json={
            "task": "清洗这批数据",
            "context": {"data": [{"name": "商品A"}]},
        })
    assert resp.status_code == 200
    call_args = mock_orch.call_args
    assert call_args[0][1] == {"data": [{"name": "商品A"}]}


def test_health_endpoint(client):
    resp = client.get("/health")
    assert resp.status_code == 200
    assert resp.json()["service"] == "agent-cluster"
```

- [ ] **Step 3: 运行全部测试**

```bash
python -m pytest tests/ -v 2>&1 | tail -20
```

Expected: 全部 PASSED（约 16 个）

- [ ] **Step 4: 手动验证**

```bash
cp /Users/liuyun/CCWorkSpace/insurance-agent/.env .env
python -m uvicorn main:app --port 8007 --reload
curl -s http://localhost:8007/health
```

Expected: `{"status":"ok","service":"agent-cluster"}`

- [ ] **Step 5: commit**

```bash
git add main.py tests/test_integration.py
git commit -m "feat: FastAPI /invoke + cluster orchestration + 16 tests passing"
```

---

## Task 6：更新 shopmind-web（集群指挥页）

**Files:**
- Modify: `shopmind-web/app/dashboard/layout.tsx`
- Create: `shopmind-web/app/dashboard/cluster/page.tsx`
- Modify: `shopmind-web/app/dashboard/page.tsx` — 更新为 7/8

- [ ] **Step 1: 切换分支并创建目录**

```bash
cd /Users/liuyun/CCWorkSpace/shopmind-ai/shopmind-web
git checkout develop
mkdir -p app/dashboard/cluster
```

- [ ] **Step 2: 更新 layout.tsx — 加入集群入口**

将 navItems 改为（8 项）：

```typescript
const navItems = [
  { href: "/dashboard",           label: "概览",   emoji: "📊" },
  { href: "/dashboard/cleaning",  label: "数据清洗", emoji: "🧹" },
  { href: "/dashboard/customer",  label: "智能客服", emoji: "💬" },
  { href: "/dashboard/media",     label: "素材库",   emoji: "🖼️" },
  { href: "/dashboard/social",    label: "内容创作", emoji: "✍️" },
  { href: "/dashboard/clip",      label: "直播切片", emoji: "✂️" },
  { href: "/dashboard/training",  label: "销售培训", emoji: "🎯" },
  { href: "/dashboard/cluster",   label: "集群指挥", emoji: "🤖" },
];
```

- [ ] **Step 3: 更新 dashboard page.tsx（7/8）**

将 `6 / 8` 改为 `7 / 8`。

- [ ] **Step 4: 创建 app/dashboard/cluster/page.tsx**

```typescript
"use client";
import { useState } from "react";
import { api } from "@/lib/api";
import { Button } from "@/components/ui/button";
import { Textarea } from "@/components/ui/textarea";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";

type Step = {
  tool: string;
  input: Record<string, unknown>;
  output: Record<string, unknown>;
};

type ClusterResult = {
  job_id: string;
  result: string;
  agents_called: string[];
  steps: Step[];
  usage: { input_tokens: number; output_tokens: number };
};

const TOOL_LABELS: Record<string, string> = {
  data_cleaning:       "🧹 数据清洗",
  customer_service:    "💬 智能客服",
  media_library_search:"🖼️ 素材库",
  social_media:        "✍️ 内容创作",
  live_clip:           "✂️ 直播切片",
  sales_training:      "🎯 销售培训",
};

const EXAMPLES = [
  "帮我生成一篇关于马拉松跑鞋的小红书种草文案",
  "在素材库中搜索户外露营相关图片",
  "回答：退货政策是什么？",
  "先搜索跑步装备图片，再生成一篇小红书文案",
];

export default function ClusterPage() {
  const [task, setTask] = useState("");
  const [result, setResult] = useState<ClusterResult | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");

  async function handleRun() {
    if (!task.trim()) return;
    setLoading(true); setError("");
    try {
      const res = await api.invokeAgent("cluster", {
        task: task.trim(),
        context: {},
      }) as ClusterResult;
      setResult(res);
    } catch (e) {
      setError(e instanceof Error ? e.message : "执行失败");
    } finally {
      setLoading(false);
    }
  }

  return (
    <div className="max-w-4xl space-y-6">
      <div>
        <h1 className="text-2xl font-bold">🤖 集群指挥</h1>
        <p className="text-slate-500 text-sm mt-1">
          用自然语言描述任务，AI 自动调度最合适的 Agent 完成
        </p>
      </div>

      {/* 示例 */}
      <div className="flex flex-wrap gap-2">
        {EXAMPLES.map(ex => (
          <button key={ex} onClick={() => setTask(ex)}
            className="text-xs px-3 py-1 bg-slate-100 hover:bg-slate-200 rounded-full transition-colors">
            {ex}
          </button>
        ))}
      </div>

      {/* 输入区 */}
      <Card>
        <CardContent className="p-4 space-y-3">
          <Textarea
            value={task}
            onChange={e => setTask(e.target.value)}
            placeholder="描述你想完成的任务，例如：先清洗商品数据，然后生成小红书文案..."
            rows={4}
            className="resize-none"
          />
          <Button onClick={handleRun} disabled={loading || !task.trim()} className="w-full">
            {loading ? "AI 正在调度执行中（约15-60秒）..." : "🚀 执行任务"}
          </Button>
          {error && <p className="text-red-500 text-sm">{error}</p>}
        </CardContent>
      </Card>

      {/* 结果区 */}
      {result && (
        <div className="space-y-4">
          {/* 执行摘要 */}
          <div className="flex items-center gap-3 flex-wrap text-sm text-slate-500">
            <span className="font-mono text-xs bg-slate-100 px-2 py-0.5 rounded">
              {result.job_id.slice(0, 8)}...
            </span>
            {result.agents_called.map(a => (
              <Badge key={a} variant="secondary" className="text-xs">
                {TOOL_LABELS[a] ?? a}
              </Badge>
            ))}
            <span>Tokens: {result.usage.input_tokens + result.usage.output_tokens}</span>
          </div>

          {/* 执行步骤 */}
          {result.steps.length > 0 && (
            <Card>
              <CardHeader className="pb-2">
                <CardTitle className="text-sm">执行日志（{result.steps.length} 步）</CardTitle>
              </CardHeader>
              <CardContent className="space-y-2">
                {result.steps.map((step, i) => (
                  <div key={i} className="bg-slate-50 rounded p-3 text-xs font-mono">
                    <span className="text-slate-500">Step {i + 1}</span>
                    <span className="mx-2 text-slate-300">→</span>
                    <span className="text-blue-600">{TOOL_LABELS[step.tool] ?? step.tool}</span>
                    <span className="ml-2 text-slate-400">
                      {JSON.stringify(step.input).slice(0, 80)}...
                    </span>
                  </div>
                ))}
              </CardContent>
            </Card>
          )}

          {/* 最终结果 */}
          <Card>
            <CardHeader className="pb-2">
              <CardTitle className="text-sm">最终结果</CardTitle>
            </CardHeader>
            <CardContent>
              <pre className="text-sm text-slate-700 whitespace-pre-wrap bg-slate-50 rounded p-3 max-h-80 overflow-y-auto">
                {result.result}
              </pre>
            </CardContent>
          </Card>
        </div>
      )}
    </div>
  );
}
```

- [ ] **Step 5: 构建验证**

```bash
npm run build 2>&1 | tail -10
```

Expected: 构建成功，新增 `/dashboard/cluster` 路由。

- [ ] **Step 6: commit 并推送**

```bash
git add app/dashboard/layout.tsx app/dashboard/cluster/ app/dashboard/page.tsx
git commit -m "feat: cluster orchestration page (natural language → multi-agent) + 7/8"
git push origin develop
```

---

## Task 7：Gateway + 端到端测试 + 推送

- [ ] **Step 1: 注册 gateway**

```bash
echo "AGENT_CLUSTER_URL=http://localhost:8007" >> /Users/liuyun/CCWorkSpace/shopmind-ai/shopmind-gateway/.env
```

- [ ] **Step 2: 启动 cluster agent**

```bash
cd /Users/liuyun/CCWorkSpace/shopmind-ai/agent-cluster
nohup python -m uvicorn main:app --port 8007 > /tmp/cluster_agent.log 2>&1 &
sleep 4 && curl -s http://localhost:8007/health
```

Expected: `{"status":"ok","service":"agent-cluster"}`

- [ ] **Step 3: 重启 gateway 并测试**

```bash
pkill -f "uvicorn main:app --port 8000" 2>/dev/null || true; sleep 1
cd /Users/liuyun/CCWorkSpace/shopmind-ai/shopmind-gateway
nohup python -m uvicorn main:app --port 8000 > /tmp/gateway.log 2>&1 &
sleep 5

TOKEN=$(curl -s -X POST http://localhost:8000/auth/login \
  -H "Content-Type: application/json" \
  -d '{"username":"demo","password":"shopmind2026"}' \
  | python3 -c "import sys,json; print(json.load(sys.stdin)['access_token'])")

curl -s --max-time 90 -X POST http://localhost:8000/api/v1/cluster/invoke \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer $TOKEN" \
  -d '{"task":"在素材库搜索跑步鞋相关图片","context":{}}' \
  > /tmp/cluster_result.json

python3 -c "
import json
with open('/tmp/cluster_result.json') as f: r=json.load(f)
print('✅ job:', r.get('job_id','')[:8])
print('✅ agents:', r.get('agents_called'))
print('✅ tokens:', r.get('usage',{}).get('input_tokens',0)+r.get('usage',{}).get('output_tokens',0))
print('✅ result:', str(r.get('result',''))[:60])
"
```

Expected: 自动调用 `media_library_search` Agent，返回搜索结果。

- [ ] **Step 4: 推送 GitHub**

```bash
cd /Users/liuyun/CCWorkSpace/shopmind-ai/agent-cluster
git add . && git commit -m "chore: phase7 complete - cluster agent integrated with gateway"
gh repo create shopmind-ai/agent-cluster \
  --public \
  --description "ShopMind AI - MCP + Agent cluster: Claude Tool Use orchestrates 6 agents via natural language tasks" \
  --source=. --remote=origin --push

cd /Users/liuyun/CCWorkSpace/shopmind-ai/shopmind-gateway
git commit --allow-empty -m "chore: add AGENT_CLUSTER_URL to gateway config"
git push origin develop
```

---

## 自审通过项

- ✅ `AGENT_URLS` 在 `config.py` 定义，`executor.py` 从 config 导入 — 不重复
- ✅ `_build_payload` 为每个 tool_name 映射到正确的 `/invoke` payload 格式（media → `action: search`，training → `action: start`，cleaning → `data + data_type`）
- ✅ `orchestrator.py` 的 Tool Use 循环最多 `_MAX_ITERATIONS=10` 次，防止无限循环
- ✅ `utils/usage.py` 的 `add(a, response)` 函数名与 orchestrator 中调用 `add(usage, response)` 一致
- ✅ 集成测试 mock 的是 `main.orchestrate`（顶层函数），不需要深入 mock LLM
- ✅ test_orchestrator.py 测试了单工具、多工具、无工具、usage 累计 4 个场景
- ✅ shopmind-web navItems 共 8 项（全部 Agent），概览更新为 7/8
- ✅ 前端步骤日志截断显示（`JSON.stringify(step.input).slice(0, 80)`），防止大数据溢出
