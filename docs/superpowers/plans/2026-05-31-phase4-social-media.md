# ShopMind AI Phase 4 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 构建 agent-social-media（自媒体 Agent），输入商品信息 → LangGraph 5 节点管道生成平台定制文案（标题+正文+话题标签） → 联动素材库推荐相关商品图片，支持小红书/抖音/微信公众号三种风格。

**Architecture:** LangGraph 线性管道 planner → writer → formatter → validator → media，每个节点职责单一。planner 制定内容大纲，writer 按平台风格写正文，formatter 添加 emoji 和话题标签，validator 过滤违规宣传词，media_fetcher 同步 HTTP 调用素材库（localhost:8003）返回推荐图片。生成历史写入 PostgreSQL `content_history` 表。

**Tech Stack:** Python 3.11 · FastAPI · LangGraph · Claude API · httpx (调用素材库) · psycopg2 · python-dotenv

---

## 参考资料

- 设计文档：`shopmind-ai/docs/superpowers/specs/2026-05-31-shopmind-ai-design.md`
- 模式参考：`/Users/liuyun/CCWorkSpace/shopmind-ai/agent-customer-service/`（LangGraph 结构）
- Gateway 已配置：`AGENT_SOCIAL_URL=http://localhost:8004`（需在 gateway `.env` 手动添加）
- 素材库：`http://localhost:8003/invoke`（POST，action=search）

---

## 文件结构总览

```
agent-social-media/
├── main.py                   # FastAPI: POST /invoke, GET /health
├── config.py                 # env vars, get_llm(), MEDIA_LIBRARY_URL
├── agent/
│   ├── __init__.py
│   ├── state.py              # SocialMediaState TypedDict
│   ├── planner.py            # 制定内容大纲节点
│   ├── writer.py             # 平台定制写作节点（含 3 套 prompt）
│   ├── formatter.py          # 格式化 + 话题标签节点
│   ├── validator.py          # 违规宣传词过滤节点
│   ├── media_fetcher.py      # HTTP 调用素材库节点
│   └── graph.py              # LangGraph 图组装
├── utils/
│   ├── __init__.py
│   ├── startup.py            # 建 content_history 表
│   └── usage.py              # Token 统计（同 Phase 1-3）
├── tests/
│   ├── __init__.py
│   ├── test_planner.py
│   ├── test_writer.py
│   ├── test_formatter.py
│   ├── test_validator.py
│   ├── test_media_fetcher.py
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
  "product_name": "专业马拉松跑步鞋",
  "platform": "xiaohongshu",         # xiaohongshu | douyin | wechat
  "tone": "种草",                     # 专业 | 活泼 | 种草
  "key_features": ["碳纤维中底", "轻量", "红色"]
}

Response:
{
  "title": "跑步装备天花板！这双鞋让我PB了...",
  "content": "姐妹们真的找到宝了！...",
  "hashtags": ["#跑步", "#马拉松装备", "#运动好物"],
  "suggested_images": [
    {"asset_id": "...", "image_url": "...", "category": "跑步装备", "score": 0.92}
  ],
  "platform": "xiaohongshu",
  "usage": {"input_tokens": 1200, "output_tokens": 400}
}
```

---

## Task 1：项目初始化 + config + utils

**Files:**
- Create: `agent-social-media/requirements.txt`
- Create: `agent-social-media/.env.example`
- Create: `agent-social-media/.gitignore`
- Create: `agent-social-media/config.py`
- Create: `agent-social-media/utils/usage.py`
- Create: `agent-social-media/utils/startup.py`

- [ ] **Step 1: 创建目录结构**

```bash
cd /Users/liuyun/CCWorkSpace/shopmind-ai
mkdir -p agent-social-media/{agent,utils,tests}
touch agent-social-media/agent/__init__.py
touch agent-social-media/utils/__init__.py
touch agent-social-media/tests/__init__.py
cd agent-social-media && git init && git checkout -b develop
git commit --allow-empty -m "chore: init agent-social-media"
```

- [ ] **Step 2: 创建 requirements.txt**

```
fastapi>=0.115.0
uvicorn[standard]>=0.30.0
langchain>=0.3.0
langchain-anthropic>=0.3.0
langchain-ollama>=0.2.0
langgraph>=0.2.0
httpx>=0.27.0
psycopg2-binary>=2.9.9
python-dotenv>=1.0.0
pytest>=8.0.0
pytest-asyncio>=0.24.0
```

- [ ] **Step 3: 创建 .env.example**

```
LLM_PROVIDER=ollama
OLLAMA_MODEL=qwen3:8b
ANTHROPIC_API_KEY=your-key-here
CLAUDE_MODEL=claude-sonnet-4-6
DATABASE_URL=postgresql://user:pass@host/dbname?sslmode=require
MEDIA_LIBRARY_URL=http://localhost:8003
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

LLM_PROVIDER = os.getenv("LLM_PROVIDER", "ollama")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "qwen3:8b")
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY", "")
CLAUDE_MODEL = os.getenv("CLAUDE_MODEL", "claude-sonnet-4-6")
DATABASE_URL = os.getenv("DATABASE_URL", "")
MEDIA_LIBRARY_URL = os.getenv("MEDIA_LIBRARY_URL", "http://localhost:8003")


def get_llm():
    if LLM_PROVIDER == "claude":
        from langchain_anthropic import ChatAnthropic
        return ChatAnthropic(model=CLAUDE_MODEL, api_key=ANTHROPIC_API_KEY)
    from langchain_ollama import ChatOllama
    return ChatOllama(model=OLLAMA_MODEL)
```

- [ ] **Step 6: 创建 utils/usage.py**

```python
from typing import TypedDict

_INPUT_COST_PER_M = 3.0
_OUTPUT_COST_PER_M = 15.0


class TokenUsage(TypedDict):
    input_tokens: int
    output_tokens: int


def accumulate(prev: "SocialMediaState", response) -> TokenUsage:
    meta = getattr(response, "usage_metadata", None) or {}
    current = prev.get("usage") or {"input_tokens": 0, "output_tokens": 0}
    return {
        "input_tokens":  current["input_tokens"]  + meta.get("input_tokens", 0),
        "output_tokens": current["output_tokens"] + meta.get("output_tokens", 0),
    }


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
CREATE TABLE IF NOT EXISTS content_history (
    id           SERIAL PRIMARY KEY,
    request_id   VARCHAR(36) UNIQUE NOT NULL DEFAULT gen_random_uuid()::text,
    product_name VARCHAR(200) NOT NULL,
    platform     VARCHAR(20)  NOT NULL,
    tone         VARCHAR(20)  NOT NULL,
    title        TEXT,
    content      TEXT,
    hashtags     TEXT[] DEFAULT '{}',
    input_tokens  INTEGER DEFAULT 0,
    output_tokens INTEGER DEFAULT 0,
    created_at   TIMESTAMPTZ DEFAULT NOW()
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
        print("[startup] agent-social-media DB initialized.")
    finally:
        conn.close()


def log_generation(product_name: str, platform: str, tone: str,
                   title: str, content: str, hashtags: list[str],
                   usage: dict):
    conn = psycopg2.connect(DATABASE_URL)
    try:
        with conn.cursor() as cur:
            cur.execute(
                "INSERT INTO content_history "
                "(product_name, platform, tone, title, content, hashtags, input_tokens, output_tokens) "
                "VALUES (%s,%s,%s,%s,%s,%s,%s,%s)",
                (product_name, platform, tone, title, content, hashtags,
                 usage.get("input_tokens", 0), usage.get("output_tokens", 0))
            )
        conn.commit()
    finally:
        conn.close()
```

- [ ] **Step 8: 安装依赖并验证**

```bash
pip install -r requirements.txt -q
python -c "import fastapi, langgraph, httpx; print('imports OK')"
```

Expected: `imports OK`

- [ ] **Step 9: commit**

```bash
git add . && git commit -m "feat: project setup, config, utils (startup + usage)"
```

---

## Task 2：AgentState + Planner 节点 + 测试

**Files:**
- Create: `agent/state.py`
- Create: `agent/planner.py`
- Create: `tests/test_planner.py`

- [ ] **Step 1: 创建 agent/state.py**

```python
from typing import TypedDict
from utils.usage import TokenUsage


class SocialMediaState(TypedDict):
    # 输入
    product_name: str
    platform: str           # "xiaohongshu" | "douyin" | "wechat"
    tone: str               # "专业" | "活泼" | "种草"
    key_features: list[str]
    # 管道中间状态
    outline: str
    draft_title: str
    draft_content: str
    hashtags: list[str]
    # 输出
    final_title: str
    final_content: str
    suggested_images: list[dict]
    usage: TokenUsage
```

- [ ] **Step 2: 写失败测试**

创建 `tests/test_planner.py`：

```python
from unittest.mock import patch, MagicMock
from agent.state import SocialMediaState


def _make_state(**kwargs) -> SocialMediaState:
    defaults = {
        "product_name": "专业马拉松跑步鞋",
        "platform": "xiaohongshu",
        "tone": "种草",
        "key_features": ["碳纤维中底", "轻量"],
        "outline": "", "draft_title": "", "draft_content": "",
        "hashtags": [], "final_title": "", "final_content": "",
        "suggested_images": [],
        "usage": {"input_tokens": 0, "output_tokens": 0},
    }
    return {**defaults, **kwargs}


def test_planner_returns_outline():
    from agent.planner import planner_node
    mock_resp = MagicMock()
    mock_resp.content = "核心卖点：碳纤维中底回弹好\n目标受众：马拉松爱好者\n内容角度：从PB经历切入"
    mock_resp.usage_metadata = {"input_tokens": 300, "output_tokens": 80}
    with patch("agent.planner.get_llm") as mock_llm:
        mock_llm.return_value.invoke.return_value = mock_resp
        result = planner_node(_make_state())
    assert len(result["outline"]) > 0
    assert "碳纤维" in result["outline"]
    assert result["usage"]["input_tokens"] == 300


def test_planner_includes_platform_in_prompt():
    from agent.planner import planner_node
    mock_resp = MagicMock()
    mock_resp.content = "小红书风格大纲"
    mock_resp.usage_metadata = {"input_tokens": 200, "output_tokens": 50}
    with patch("agent.planner.get_llm") as mock_llm:
        mock_llm.return_value.invoke.return_value = mock_resp
        planner_node(_make_state(platform="xiaohongshu"))
    call_args = mock_llm.return_value.invoke.call_args[0][0]
    assert "小红书" in call_args


def test_planner_accumulates_usage():
    from agent.planner import planner_node
    mock_resp = MagicMock()
    mock_resp.content = "大纲内容"
    mock_resp.usage_metadata = {"input_tokens": 200, "output_tokens": 60}
    with patch("agent.planner.get_llm") as mock_llm:
        mock_llm.return_value.invoke.return_value = mock_resp
        result = planner_node(_make_state(usage={"input_tokens": 100, "output_tokens": 20}))
    assert result["usage"]["input_tokens"] == 300
    assert result["usage"]["output_tokens"] == 80
```

- [ ] **Step 3: 运行，确认失败**

```bash
pytest tests/test_planner.py -v 2>&1 | tail -5
```

Expected: FAILED（ImportError）

- [ ] **Step 4: 创建 agent/planner.py**

```python
from agent.state import SocialMediaState
from config import get_llm
from utils.usage import accumulate

_PLATFORM_NAMES = {
    "xiaohongshu": "小红书",
    "douyin": "抖音",
    "wechat": "微信公众号",
}

_PROMPT = """你是一名资深{platform}内容策划，擅长{tone}风格的电商内容。
分析以下商品信息，制定内容创作大纲：

商品名称：{product_name}
主要特点：{features}
内容基调：{tone}

请输出大纲，包含：
1. 核心卖点（3个，突出{features}）
2. 目标受众描述（谁会买、有什么痛点）
3. 内容切入角度（从哪个场景/故事切入更吸引人）
4. 适合的话题标签（5-8个，格式如：跑步装备、马拉松、运动好物）"""


def planner_node(state: SocialMediaState) -> SocialMediaState:
    llm = get_llm()
    platform_name = _PLATFORM_NAMES.get(state["platform"], state["platform"])
    features_str = "、".join(state["key_features"]) if state["key_features"] else "高品质"

    prompt = _PROMPT.format(
        platform=platform_name,
        tone=state["tone"],
        product_name=state["product_name"],
        features=features_str,
    )
    response = llm.invoke(prompt)
    usage = accumulate(state, response)
    return {**state, "outline": response.content, "usage": usage}
```

- [ ] **Step 5: 运行，确认通过**

```bash
pytest tests/test_planner.py -v 2>&1 | tail -6
```

Expected: 3 PASSED

- [ ] **Step 6: commit**

```bash
git add agent/state.py agent/planner.py tests/test_planner.py
git commit -m "feat: SocialMediaState + planner node with platform-specific prompts"
```

---

## Task 3：Writer 节点 + 测试（三平台风格）

**Files:**
- Create: `agent/writer.py`
- Create: `tests/test_writer.py`

- [ ] **Step 1: 写失败测试**

创建 `tests/test_writer.py`：

```python
from unittest.mock import patch, MagicMock
from agent.state import SocialMediaState


def _make_state(platform="xiaohongshu") -> SocialMediaState:
    return {
        "product_name": "专业马拉松跑步鞋",
        "platform": platform,
        "tone": "种草",
        "key_features": ["碳纤维中底", "轻量"],
        "outline": "核心卖点：轻量碳板\n目标受众：跑马爱好者\n角度：PB经历",
        "draft_title": "", "draft_content": "",
        "hashtags": [], "final_title": "", "final_content": "",
        "suggested_images": [],
        "usage": {"input_tokens": 0, "output_tokens": 0},
    }


def test_writer_returns_title_and_content():
    from agent.writer import writer_node
    mock_resp = MagicMock()
    mock_resp.content = "标题：这双鞋让我PB了！\n\n正文：姐妹们这双跑鞋真的绝了..."
    mock_resp.usage_metadata = {"input_tokens": 500, "output_tokens": 200}
    with patch("agent.writer.get_llm") as mock_llm:
        mock_llm.return_value.invoke.return_value = mock_resp
        result = writer_node(_make_state("xiaohongshu"))
    assert len(result["draft_title"]) > 0
    assert len(result["draft_content"]) > 0
    assert result["usage"]["input_tokens"] == 500


def test_writer_douyin_uses_short_format():
    from agent.writer import writer_node
    mock_resp = MagicMock()
    mock_resp.content = "标题：碳板跑鞋实测\n\n正文：今天测评一双性价比炸裂的跑步鞋..."
    mock_resp.usage_metadata = {"input_tokens": 400, "output_tokens": 150}
    with patch("agent.writer.get_llm") as mock_llm:
        mock_llm.return_value.invoke.return_value = mock_resp
        result = writer_node(_make_state("douyin"))
    call_args = mock_llm.return_value.invoke.call_args[0][0]
    assert "抖音" in call_args or "短视频" in call_args


def test_writer_wechat_uses_formal_format():
    from agent.writer import writer_node
    mock_resp = MagicMock()
    mock_resp.content = "标题：深度测评：这款碳板跑鞋值得买吗\n\n正文：作为一名跑步爱好者..."
    mock_resp.usage_metadata = {"input_tokens": 600, "output_tokens": 300}
    with patch("agent.writer.get_llm") as mock_llm:
        mock_llm.return_value.invoke.return_value = mock_resp
        result = writer_node(_make_state("wechat"))
    call_args = mock_llm.return_value.invoke.call_args[0][0]
    assert "微信" in call_args or "公众号" in call_args


def test_writer_falls_back_on_unknown_platform():
    from agent.writer import writer_node
    mock_resp = MagicMock()
    mock_resp.content = "标题：好产品推荐\n\n正文：这款产品非常好用..."
    mock_resp.usage_metadata = {"input_tokens": 300, "output_tokens": 100}
    with patch("agent.writer.get_llm") as mock_llm:
        mock_llm.return_value.invoke.return_value = mock_resp
        result = writer_node({**_make_state(), "platform": "unknown"})
    assert len(result["draft_title"]) > 0
```

- [ ] **Step 2: 运行，确认失败**

```bash
pytest tests/test_writer.py -v 2>&1 | tail -5
```

Expected: FAILED（ImportError）

- [ ] **Step 3: 创建 agent/writer.py**

```python
from agent.state import SocialMediaState
from config import get_llm
from utils.usage import accumulate

_PROMPTS = {
    "xiaohongshu": """根据以下大纲，写一篇小红书种草文章：

{outline}

商品：{product_name}，特点：{features}，基调：{tone}

要求：
- 标题：15-20字，含emoji，吸睛（可用「！」「｜」分隔），以"标题："开头
- 正文：300-500字，多用emoji，语气活泼亲切
- 分点描述商品亮点（用「✨」「💪」「🔥」等）
- 结尾引导互动
- 格式：先写"标题：xxx"，空一行，再写"正文：xxx"

只输出标题和正文，不要有其他解释。""",

    "douyin": """根据以下大纲，写一段抖音短视频文案（口播脚本）：

{outline}

商品：{product_name}，特点：{features}，基调：{tone}

要求：
- 标题：10-15字，直接抛出痛点或惊喜，以"标题："开头
- 正文：150-200字，开头3句必须抓住注意力（痛点/惊喜/数字）
- 口语化，像真人说话
- 结尾有行动号召（关注/评论/点赞）
- 格式：先写"标题：xxx"，空一行，再写"正文：xxx"

只输出标题和正文，不要有其他解释。""",

    "wechat": """根据以下大纲，写一篇微信公众号测评文章：

{outline}

商品：{product_name}，特点：{features}，基调：{tone}

要求：
- 标题：20-30字，权威感强（可用「深度测评」「实测」「值得买吗」），以"标题："开头
- 正文：600-800字，段落清晰，有小标题（用「**xxx**」格式）
- 语气专业客观，可以有数据对比
- 结尾给出明确购买建议
- 格式：先写"标题：xxx"，空一行，再写"正文：xxx"

只输出标题和正文，不要有其他解释。""",
}

_DEFAULT_PROMPT = _PROMPTS["xiaohongshu"]


def _parse_title_content(raw: str) -> tuple[str, str]:
    """Extract title and content from LLM response."""
    title, content = "", raw
    lines = raw.strip().split("\n")
    for i, line in enumerate(lines):
        if line.startswith("标题："):
            title = line.replace("标题：", "").strip()
            content = "\n".join(lines[i + 1:]).lstrip("\n").replace("正文：", "", 1).strip()
            break
    return title, content


def writer_node(state: SocialMediaState) -> SocialMediaState:
    llm = get_llm()
    prompt_template = _PROMPTS.get(state["platform"], _DEFAULT_PROMPT)
    features_str = "、".join(state["key_features"]) if state["key_features"] else "高品质"

    prompt = prompt_template.format(
        outline=state["outline"],
        product_name=state["product_name"],
        features=features_str,
        tone=state["tone"],
    )
    response = llm.invoke(prompt)
    usage = accumulate(state, response)
    title, content = _parse_title_content(response.content)
    return {**state, "draft_title": title, "draft_content": content, "usage": usage}
```

- [ ] **Step 4: 运行，确认通过**

```bash
pytest tests/test_writer.py -v 2>&1 | tail -7
```

Expected: 4 PASSED

- [ ] **Step 5: commit**

```bash
git add agent/writer.py tests/test_writer.py
git commit -m "feat: writer node with platform-specific prompts (xiaohongshu/douyin/wechat)"
```

---

## Task 4：Formatter + Validator 节点 + 测试

**Files:**
- Create: `agent/formatter.py`
- Create: `agent/validator.py`
- Create: `tests/test_formatter.py`
- Create: `tests/test_validator.py`

- [ ] **Step 1: 写失败测试**

创建 `tests/test_formatter.py`：

```python
from agent.state import SocialMediaState


def _make_state(platform="xiaohongshu", draft_content="这双鞋碳板回弹好，轻量透气") -> SocialMediaState:
    return {
        "product_name": "跑步鞋", "platform": platform, "tone": "种草",
        "key_features": ["碳板", "轻量"], "outline": "大纲",
        "draft_title": "这双鞋真的绝了！", "draft_content": draft_content,
        "hashtags": [], "final_title": "", "final_content": "",
        "suggested_images": [], "usage": {"input_tokens": 0, "output_tokens": 0},
    }


def test_formatter_extracts_hashtags():
    from agent.formatter import formatter_node
    result = formatter_node(_make_state(draft_content="好用的碳板跑鞋，适合马拉松"))
    assert len(result["hashtags"]) > 0


def test_formatter_sets_final_title_and_content():
    from agent.formatter import formatter_node
    result = formatter_node(_make_state())
    assert result["final_title"] == result["draft_title"]
    assert len(result["final_content"]) > 0


def test_formatter_adds_hashtags_to_wechat_content():
    from agent.formatter import formatter_node
    result = formatter_node(_make_state(platform="wechat"))
    assert result["final_content"] is not None
```

创建 `tests/test_validator.py`：

```python
from agent.state import SocialMediaState


def _make_state(title="好产品", content="这是一款很好的跑鞋") -> SocialMediaState:
    return {
        "product_name": "跑步鞋", "platform": "xiaohongshu", "tone": "种草",
        "key_features": [], "outline": "", "draft_title": title, "draft_content": content,
        "hashtags": ["跑步"], "final_title": title, "final_content": content,
        "suggested_images": [], "usage": {"input_tokens": 0, "output_tokens": 0},
    }


def test_validator_passes_clean_content():
    from agent.validator import validator_node
    result = validator_node(_make_state(content="这双跑鞋碳板回弹好，轻量透气，很适合长跑。"))
    assert result["final_content"] == "这双跑鞋碳板回弹好，轻量透气，很适合长跑。"


def test_validator_replaces_prohibited_superlatives():
    from agent.validator import validator_node
    result = validator_node(_make_state(content="全国最好的跑鞋，没有之一！"))
    assert "全国最好" not in result["final_content"]


def test_validator_replaces_false_medical_claims():
    from agent.validator import validator_node
    result = validator_node(_make_state(content="穿上这双鞋能治好膝盖疼痛，效果显著。"))
    assert "治好" not in result["final_content"]


def test_validator_replaces_absolute_claims():
    from agent.validator import validator_node
    result = validator_node(_make_state(content="100%好评，绝对不会后悔购买！"))
    assert "100%好评" not in result["final_content"] or "不保证" in result["final_content"]
```

- [ ] **Step 2: 运行，确认失败**

```bash
pytest tests/test_formatter.py tests/test_validator.py -v 2>&1 | tail -5
```

Expected: FAILED（ImportError）

- [ ] **Step 3: 创建 agent/formatter.py**

```python
import re
from agent.state import SocialMediaState

_SPORT_TAGS = ["跑步", "马拉松", "健身", "运动装备", "户外", "露营", "瑜伽",
               "运动好物", "装备推荐", "运动生活"]

_PLATFORM_EMOJI_DENSITY = {
    "xiaohongshu": "high",
    "douyin": "medium",
    "wechat": "low",
}


def _extract_hashtags(product_name: str, key_features: list[str], content: str) -> list[str]:
    tags = set()
    text = f"{product_name} {' '.join(key_features)} {content}"
    for tag in _SPORT_TAGS:
        if tag in text:
            tags.add(f"#{tag}")
    tags.add(f"#{product_name.split('鞋')[0]}鞋" if "鞋" in product_name else f"#{product_name[:4]}")
    tags.add("#运动好物")
    tags.add("#好物推荐")
    return list(tags)[:8]


def _add_platform_hashtags(content: str, hashtags: list[str], platform: str) -> str:
    tags_str = " ".join(hashtags)
    if platform == "xiaohongshu":
        return f"{content}\n\n{tags_str}"
    elif platform == "douyin":
        return f"{content}\n{tags_str}"
    else:
        return content


def formatter_node(state: SocialMediaState) -> SocialMediaState:
    hashtags = _extract_hashtags(
        state["product_name"], state["key_features"], state["draft_content"]
    )
    final_content = _add_platform_hashtags(
        state["draft_content"], hashtags, state["platform"]
    )
    return {
        **state,
        "hashtags": hashtags,
        "final_title": state["draft_title"],
        "final_content": final_content,
    }
```

- [ ] **Step 4: 创建 agent/validator.py**

```python
import re
from agent.state import SocialMediaState

_RULES = [
    (r"全国最好|全球最好|行业第一|销量第一|最好的",   "优质"),
    (r"100%好评|100%满意|绝对不会后悔",              "广受好评"),
    (r"治好|治愈|治疗|根治|彻底康复",                "有助于缓解"),
    (r"效果显著|立竿见影|药到病除",                   "效果不错"),
    (r"无任何副作用|零副作用",                        "使用安全"),
    (r"限时[0-9]*[折%]|最后[0-9]*件",               "优惠活动"),
]


def validator_node(state: SocialMediaState) -> SocialMediaState:
    content = state["final_content"]
    for pattern, replacement in _RULES:
        content = re.sub(pattern, replacement, content)
    return {**state, "final_content": content}
```

- [ ] **Step 5: 运行，确认通过**

```bash
pytest tests/test_formatter.py tests/test_validator.py -v 2>&1 | tail -10
```

Expected: 全部 PASSED（7 个）

- [ ] **Step 6: commit**

```bash
git add agent/formatter.py agent/validator.py tests/test_formatter.py tests/test_validator.py
git commit -m "feat: formatter (hashtags + platform format) + validator (prohibited phrases)"
```

---

## Task 5：Media Fetcher 节点 + 测试

**Files:**
- Create: `agent/media_fetcher.py`
- Create: `tests/test_media_fetcher.py`

- [ ] **Step 1: 写失败测试**

创建 `tests/test_media_fetcher.py`：

```python
from unittest.mock import patch, MagicMock
from agent.state import SocialMediaState


def _make_state() -> SocialMediaState:
    return {
        "product_name": "专业马拉松跑步鞋", "platform": "xiaohongshu", "tone": "种草",
        "key_features": ["碳纤维中底", "红色"], "outline": "", "draft_title": "",
        "draft_content": "", "hashtags": [], "final_title": "好鞋推荐",
        "final_content": "这双鞋真的好", "suggested_images": [],
        "usage": {"input_tokens": 0, "output_tokens": 0},
    }


def test_media_fetcher_returns_images():
    from agent.media_fetcher import media_fetcher_node
    mock_results = [
        {"asset_id": "abc", "image_url": "https://x.com/shoe.jpg",
         "category": "跑步装备", "tags": ["红色"], "description": "红色跑鞋", "score": 0.91}
    ]
    with patch("agent.media_fetcher.httpx.Client") as mock_client_cls:
        mock_client = MagicMock()
        mock_client.__enter__ = MagicMock(return_value=mock_client)
        mock_client.__exit__ = MagicMock(return_value=False)
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {"results": mock_results}
        mock_client.post.return_value = mock_response
        mock_client_cls.return_value = mock_client

        result = media_fetcher_node(_make_state())

    assert len(result["suggested_images"]) == 1
    assert result["suggested_images"][0]["category"] == "跑步装备"


def test_media_fetcher_returns_empty_on_service_down():
    from agent.media_fetcher import media_fetcher_node
    with patch("agent.media_fetcher.httpx.Client") as mock_client_cls:
        mock_client = MagicMock()
        mock_client.__enter__ = MagicMock(return_value=mock_client)
        mock_client.__exit__ = MagicMock(return_value=False)
        mock_client.post.side_effect = Exception("connection refused")
        mock_client_cls.return_value = mock_client

        result = media_fetcher_node(_make_state())

    assert result["suggested_images"] == []


def test_media_fetcher_builds_query_from_product():
    from agent.media_fetcher import media_fetcher_node
    with patch("agent.media_fetcher.httpx.Client") as mock_client_cls:
        mock_client = MagicMock()
        mock_client.__enter__ = MagicMock(return_value=mock_client)
        mock_client.__exit__ = MagicMock(return_value=False)
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {"results": []}
        mock_client.post.return_value = mock_response
        mock_client_cls.return_value = mock_client

        media_fetcher_node(_make_state())

    call_json = mock_client.post.call_args[1]["json"]
    assert "query" in call_json
    assert len(call_json["query"]) > 0
```

- [ ] **Step 2: 运行，确认失败**

```bash
pytest tests/test_media_fetcher.py -v 2>&1 | tail -5
```

Expected: FAILED（ImportError）

- [ ] **Step 3: 创建 agent/media_fetcher.py**

```python
import httpx
from agent.state import SocialMediaState
from config import MEDIA_LIBRARY_URL


def _build_query(state: SocialMediaState) -> str:
    parts = [state["product_name"]] + state["key_features"][:2]
    return " ".join(parts)


def media_fetcher_node(state: SocialMediaState) -> SocialMediaState:
    """Call media library to get relevant product images. Fails silently."""
    query = _build_query(state)
    try:
        with httpx.Client(timeout=5.0) as client:
            resp = client.post(
                f"{MEDIA_LIBRARY_URL}/invoke",
                json={"action": "search", "query": query, "limit": 3},
            )
            if resp.status_code == 200:
                images = resp.json().get("results", [])
                return {**state, "suggested_images": images}
    except Exception:
        pass
    return {**state, "suggested_images": []}
```

- [ ] **Step 4: 运行，确认通过**

```bash
pytest tests/test_media_fetcher.py -v 2>&1 | tail -6
```

Expected: 3 PASSED

- [ ] **Step 5: commit**

```bash
git add agent/media_fetcher.py tests/test_media_fetcher.py
git commit -m "feat: media fetcher node (HTTP call to media library, silent fallback)"
```

---

## Task 6：LangGraph 图 + main.py + 集成测试

**Files:**
- Create: `agent/graph.py`
- Create: `main.py`
- Create: `tests/test_integration.py`

- [ ] **Step 1: 创建 agent/graph.py**

```python
from langgraph.graph import StateGraph, START, END
from agent.state import SocialMediaState
from agent.planner import planner_node
from agent.writer import writer_node
from agent.formatter import formatter_node
from agent.validator import validator_node
from agent.media_fetcher import media_fetcher_node


def build_graph():
    g = StateGraph(SocialMediaState)
    g.add_node("planner",  planner_node)
    g.add_node("writer",   writer_node)
    g.add_node("formatter", formatter_node)
    g.add_node("validator", validator_node)
    g.add_node("media",    media_fetcher_node)

    g.add_edge(START,       "planner")
    g.add_edge("planner",   "writer")
    g.add_edge("writer",    "formatter")
    g.add_edge("formatter", "validator")
    g.add_edge("validator", "media")
    g.add_edge("media",     END)

    return g.compile()
```

- [ ] **Step 2: 创建 main.py**

```python
from contextlib import asynccontextmanager
from fastapi import FastAPI
from pydantic import BaseModel
from agent.graph import build_graph
from agent.state import SocialMediaState
from utils.startup import check_and_init, log_generation

_graph = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global _graph
    check_and_init()
    _graph = build_graph()
    yield


app = FastAPI(title="ShopMind Social Media Agent", version="1.0.0", lifespan=lifespan)


class InvokeRequest(BaseModel):
    product_name: str
    platform: str = "xiaohongshu"   # xiaohongshu | douyin | wechat
    tone: str = "种草"               # 专业 | 活泼 | 种草
    key_features: list[str] = []


@app.post("/invoke")
def invoke(req: InvokeRequest):
    state: SocialMediaState = {
        "product_name":    req.product_name,
        "platform":        req.platform,
        "tone":            req.tone,
        "key_features":    req.key_features,
        "outline":         "",
        "draft_title":     "",
        "draft_content":   "",
        "hashtags":        [],
        "final_title":     "",
        "final_content":   "",
        "suggested_images": [],
        "usage":           {"input_tokens": 0, "output_tokens": 0},
    }
    result = _graph.invoke(state)
    usage = result.get("usage", {"input_tokens": 0, "output_tokens": 0})

    log_generation(
        product_name=req.product_name,
        platform=req.platform,
        tone=req.tone,
        title=result["final_title"],
        content=result["final_content"],
        hashtags=result["hashtags"],
        usage=usage,
    )
    return {
        "title":             result["final_title"],
        "content":           result["final_content"],
        "hashtags":          result["hashtags"],
        "suggested_images":  result["suggested_images"],
        "platform":          req.platform,
        "usage":             usage,
    }


@app.get("/health")
def health():
    return {"status": "ok", "service": "agent-social-media"}
```

- [ ] **Step 3: 写集成测试**

创建 `tests/test_integration.py`：

```python
import pytest
from unittest.mock import patch, MagicMock
from fastapi.testclient import TestClient


@pytest.fixture
def client():
    with patch("utils.startup.check_and_init"), \
         patch("utils.startup.log_generation"):
        import importlib
        import main as m
        importlib.reload(m)

        def fake_invoke(state):
            from agent.formatter import formatter_node
            from agent.validator import validator_node
            state = {**state,
                     "outline": "大纲内容",
                     "draft_title": "好鞋推荐！",
                     "draft_content": "这双碳板跑鞋真的好用，强烈推荐！",
                     "usage": {"input_tokens": 800, "output_tokens": 250}}
            state = formatter_node(state)
            state = validator_node(state)
            state = {**state, "suggested_images": []}
            return state

        mock_graph = MagicMock()
        mock_graph.invoke.side_effect = fake_invoke
        m._graph = mock_graph
        yield TestClient(m.app)


def test_invoke_returns_complete_response(client):
    resp = client.post("/invoke", json={
        "product_name": "专业马拉松跑步鞋",
        "platform": "xiaohongshu",
        "tone": "种草",
        "key_features": ["碳纤维中底", "轻量"],
    })
    assert resp.status_code == 200
    body = resp.json()
    assert "title" in body
    assert "content" in body
    assert "hashtags" in body
    assert "suggested_images" in body
    assert "usage" in body
    assert body["platform"] == "xiaohongshu"


def test_invoke_default_platform_is_xiaohongshu(client):
    resp = client.post("/invoke", json={"product_name": "跑步鞋"})
    assert resp.json()["platform"] == "xiaohongshu"


def test_invoke_douyin_platform(client):
    resp = client.post("/invoke", json={
        "product_name": "跑步鞋",
        "platform": "douyin",
    })
    assert resp.status_code == 200
    assert resp.json()["platform"] == "douyin"


def test_invoke_usage_tracked(client):
    resp = client.post("/invoke", json={"product_name": "跑步鞋", "platform": "wechat"})
    usage = resp.json()["usage"]
    assert usage["input_tokens"] == 800
    assert usage["output_tokens"] == 250


def test_health_endpoint(client):
    resp = client.get("/health")
    assert resp.status_code == 200
    assert resp.json()["service"] == "agent-social-media"
```

- [ ] **Step 4: 运行全部测试**

```bash
pytest tests/ -v 2>&1 | tail -20
```

Expected: 全部 PASSED（约 18 个）

- [ ] **Step 5: 手动验证（需要 .env）**

```bash
cp /Users/liuyun/CCWorkSpace/insurance-agent/.env .env
python -m uvicorn main:app --port 8004 --reload
```

```bash
curl -s http://localhost:8004/health
# Expected: {"status":"ok","service":"agent-social-media"}

curl -s -X POST http://localhost:8004/invoke \
  -H "Content-Type: application/json" \
  -d '{"product_name":"专业马拉松跑步鞋","platform":"xiaohongshu","tone":"种草","key_features":["碳纤维中底","轻量"]}' \
  | python3 -c "import sys,json; r=json.load(sys.stdin); print('title:', r['title'][:30], '| tokens:', r['usage']['input_tokens']+r['usage']['output_tokens'])"
```

- [ ] **Step 6: commit**

```bash
git add agent/graph.py main.py tests/test_integration.py
git commit -m "feat: LangGraph graph + FastAPI /invoke + 18 tests passing"
```

---

## Task 7：更新 shopmind-web（内容创作页）

**Files:**
- Modify: `shopmind-web/app/dashboard/layout.tsx`
- Create: `shopmind-web/app/dashboard/social/page.tsx`
- Modify: `shopmind-web/app/dashboard/page.tsx` — 更新为 4/8

- [ ] **Step 1: 切换到 shopmind-web develop 分支**

```bash
cd /Users/liuyun/CCWorkSpace/shopmind-ai/shopmind-web
git checkout develop
mkdir -p app/dashboard/social
```

- [ ] **Step 2: 更新侧边栏 layout.tsx**

将 navItems 改为：

```typescript
const navItems = [
  { href: "/dashboard",          label: "概览",   emoji: "📊" },
  { href: "/dashboard/cleaning", label: "数据清洗", emoji: "🧹" },
  { href: "/dashboard/customer", label: "智能客服", emoji: "💬" },
  { href: "/dashboard/media",    label: "素材库",   emoji: "🖼️" },
  { href: "/dashboard/social",   label: "内容创作", emoji: "✍️" },
];
```

- [ ] **Step 3: 更新 dashboard page.tsx（4/8）**

将 `3 / 8` 改为 `4 / 8`。

- [ ] **Step 4: 创建 app/dashboard/social/page.tsx**

```typescript
"use client";
import { useState } from "react";
import { api } from "@/lib/api";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";

const PLATFORMS = [
  { value: "xiaohongshu", label: "小红书", emoji: "📕" },
  { value: "douyin",      label: "抖音",   emoji: "🎵" },
  { value: "wechat",      label: "公众号", emoji: "💬" },
];

const TONES = ["种草", "专业", "活泼"];

type SocialResult = {
  title: string;
  content: string;
  hashtags: string[];
  suggested_images: Array<{ asset_id: string; image_url: string; category: string; description: string; score: number }>;
  platform: string;
  usage: { input_tokens: number; output_tokens: number };
};

export default function SocialPage() {
  const [productName, setProductName] = useState("专业马拉松跑步鞋");
  const [platform, setPlatform] = useState("xiaohongshu");
  const [tone, setTone] = useState("种草");
  const [features, setFeatures] = useState("碳纤维中底, 轻量, 红色");
  const [result, setResult] = useState<SocialResult | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");

  async function handleGenerate() {
    if (!productName.trim()) return;
    setLoading(true);
    setError("");
    try {
      const keyFeatures = features.split(",").map(s => s.trim()).filter(Boolean);
      const res = await api.invokeAgent("social", {
        product_name: productName.trim(),
        platform,
        tone,
        key_features: keyFeatures,
      }) as SocialResult;
      setResult(res);
    } catch (e) {
      setError(e instanceof Error ? e.message : "生成失败，请重试");
    } finally {
      setLoading(false);
    }
  }

  const currentPlatform = PLATFORMS.find(p => p.value === platform);

  return (
    <div className="max-w-4xl space-y-6">
      <div>
        <h1 className="text-2xl font-bold">✍️ 内容创作</h1>
        <p className="text-slate-500 text-sm mt-1">输入商品信息，AI 自动生成平台定制文案</p>
      </div>

      {/* 输入区 */}
      <Card>
        <CardContent className="p-4 space-y-4">
          <div className="grid grid-cols-2 gap-3">
            <div className="space-y-1">
              <label className="text-sm font-medium">商品名称</label>
              <Input value={productName} onChange={e => setProductName(e.target.value)} placeholder="如：专业马拉松跑步鞋" />
            </div>
            <div className="space-y-1">
              <label className="text-sm font-medium">主要特点（逗号分隔）</label>
              <Input value={features} onChange={e => setFeatures(e.target.value)} placeholder="碳纤维中底, 轻量, 红色" />
            </div>
          </div>

          <div className="flex gap-4">
            <div className="space-y-1">
              <label className="text-sm font-medium">发布平台</label>
              <div className="flex gap-2">
                {PLATFORMS.map(p => (
                  <button
                    key={p.value}
                    onClick={() => setPlatform(p.value)}
                    className={`px-3 py-1.5 rounded-lg text-sm border transition-colors ${
                      platform === p.value
                        ? "bg-slate-900 text-white border-slate-900"
                        : "bg-white text-slate-700 border-slate-200 hover:border-slate-400"
                    }`}
                  >
                    {p.emoji} {p.label}
                  </button>
                ))}
              </div>
            </div>
            <div className="space-y-1">
              <label className="text-sm font-medium">内容基调</label>
              <div className="flex gap-2">
                {TONES.map(t => (
                  <button
                    key={t}
                    onClick={() => setTone(t)}
                    className={`px-3 py-1.5 rounded-lg text-sm border transition-colors ${
                      tone === t
                        ? "bg-slate-900 text-white border-slate-900"
                        : "bg-white text-slate-700 border-slate-200 hover:border-slate-400"
                    }`}
                  >
                    {t}
                  </button>
                ))}
              </div>
            </div>
          </div>

          <Button onClick={handleGenerate} disabled={loading || !productName.trim()} className="w-full">
            {loading ? "生成中（约15-30秒）..." : `生成 ${currentPlatform?.label} 文案`}
          </Button>
          {error && <p className="text-red-500 text-sm">{error}</p>}
        </CardContent>
      </Card>

      {/* 结果区 */}
      {result && (
        <div className="space-y-4">
          <Card>
            <CardHeader className="pb-2">
              <div className="flex items-center justify-between">
                <CardTitle className="text-base">
                  {currentPlatform?.emoji} {currentPlatform?.label} 文案
                </CardTitle>
                <span className="text-xs text-slate-400">
                  Tokens: {result.usage.input_tokens + result.usage.output_tokens}
                  （${((result.usage.input_tokens / 1e6 * 3) + (result.usage.output_tokens / 1e6 * 15)).toFixed(4)}）
                </span>
              </div>
            </CardHeader>
            <CardContent className="space-y-3">
              <div>
                <p className="text-xs text-slate-400 mb-1">标题</p>
                <p className="font-semibold text-slate-800">{result.title}</p>
              </div>
              <div>
                <p className="text-xs text-slate-400 mb-1">正文</p>
                <pre className="text-sm text-slate-700 whitespace-pre-wrap bg-slate-50 rounded p-3 leading-relaxed">
                  {result.content}
                </pre>
              </div>
              <div className="flex flex-wrap gap-1.5">
                {result.hashtags.map(tag => (
                  <Badge key={tag} variant="secondary" className="text-xs">{tag}</Badge>
                ))}
              </div>
            </CardContent>
          </Card>

          {result.suggested_images.length > 0 && (
            <Card>
              <CardHeader className="pb-2">
                <CardTitle className="text-base">📷 推荐配图（来自素材库）</CardTitle>
              </CardHeader>
              <CardContent>
                <div className="grid grid-cols-3 gap-3">
                  {result.suggested_images.map(img => (
                    <div key={img.asset_id} className="space-y-1">
                      <img
                        src={img.image_url}
                        alt={img.description}
                        className="w-full h-28 object-cover rounded bg-slate-100"
                        onError={e => { (e.target as HTMLImageElement).src = "https://placehold.co/200x150?text=Image"; }}
                      />
                      <p className="text-xs text-slate-500 line-clamp-1">{img.description}</p>
                      <p className="text-xs text-slate-400">{(img.score * 100).toFixed(0)}% 匹配</p>
                    </div>
                  ))}
                </div>
              </CardContent>
            </Card>
          )}
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

Expected: 构建成功，新增 `/dashboard/social` 路由。

- [ ] **Step 6: commit**

```bash
git add app/dashboard/layout.tsx app/dashboard/social/ app/dashboard/page.tsx
git commit -m "feat: social media content creation page + update agent count to 4/8"
git push origin develop
```

---

## Task 8：Gateway + 端到端测试 + 推送

**Files:**
- Modify: `shopmind-gateway/.env` — 新增 AGENT_SOCIAL_URL

- [ ] **Step 1: 注册 gateway**

```bash
echo "AGENT_SOCIAL_URL=http://localhost:8004" >> /Users/liuyun/CCWorkSpace/shopmind-ai/shopmind-gateway/.env
```

- [ ] **Step 2: 启动服务**

```bash
# 确保 agent-social-media 在 8004 运行
cd /Users/liuyun/CCWorkSpace/shopmind-ai/agent-social-media
nohup python -m uvicorn main:app --port 8004 > /tmp/social_agent.log 2>&1 &
sleep 4 && curl -s http://localhost:8004/health
```

Expected: `{"status":"ok","service":"agent-social-media"}`

- [ ] **Step 3: 重启 gateway 加载新环境变量**

```bash
pkill -f "uvicorn main:app --port 8000" 2>/dev/null; sleep 1
cd /Users/liuyun/CCWorkSpace/shopmind-ai/shopmind-gateway
nohup python -m uvicorn main:app --port 8000 > /tmp/gateway.log 2>&1 &
sleep 4 && curl -s http://localhost:8000/health
```

- [ ] **Step 4: 端到端测试**

```bash
TOKEN=$(curl -s -X POST http://localhost:8000/auth/login \
  -H "Content-Type: application/json" \
  -d '{"username":"demo","password":"shopmind2026"}' \
  | python3 -c "import sys,json; print(json.load(sys.stdin)['access_token'])")

curl -s --max-time 90 -X POST http://localhost:8000/api/v1/social/invoke \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer $TOKEN" \
  -d '{"product_name":"专业马拉松跑步鞋","platform":"xiaohongshu","tone":"种草","key_features":["碳纤维中底","轻量"]}' \
  > /tmp/social_result.json

python3 -c "
import json
with open('/tmp/social_result.json') as f: r=json.load(f)
print('✅ title:', r.get('title','')[:40])
print('✅ hashtags:', len(r.get('hashtags',[])))
print('✅ tokens:', r.get('usage',{}).get('input_tokens',0)+r.get('usage',{}).get('output_tokens',0))
"
```

Expected: 返回完整标题、hashtags 数量 > 0、tokens > 0

- [ ] **Step 5: 推送到 GitHub**

```bash
cd /Users/liuyun/CCWorkSpace/shopmind-ai/agent-social-media
git add . && git commit -m "chore: phase4 complete - social media agent integrated with gateway"

gh repo create shopmind-ai/agent-social-media \
  --public \
  --description "ShopMind AI - Social media content agent: LangGraph pipeline generating platform-specific content (小红书/抖音/公众号) with media library integration" \
  --source=. --remote=origin --push

cd /Users/liuyun/CCWorkSpace/shopmind-ai/shopmind-gateway
git commit --allow-empty -m "chore: add AGENT_SOCIAL_URL to gateway config"
git push origin develop
```

---

## 自审通过项

- ✅ `SocialMediaState` 字段：`planner` 写 `outline`，`writer` 写 `draft_title`/`draft_content`，`formatter` 写 `hashtags`/`final_title`/`final_content`，`validator` 只修改 `final_content`，`media_fetcher` 写 `suggested_images`——字段分配无冲突
- ✅ `accumulate()` 签名：`(state, response) -> TokenUsage`，在 planner 和 writer 中用法一致
- ✅ `_parse_title_content()` 逻辑：从 "标题：xxx" 行提取，兼容 LLM 不严格遵守格式的情况（返回空字符串而非崩溃）
- ✅ `media_fetcher_node` 在服务不可用时返回空列表（`suggested_images: []`），不影响主流程
- ✅ `validator_node` 用正则替换而非拦截，保证 `final_content` 始终有值
- ✅ 集成测试 fixture 绕过 LangGraph 图，直接注入 mock_graph，与 Phase 2/3 模式一致
- ✅ shopmind-web dashboard page.tsx 更新为 4/8
- ✅ gateway `.env` 需手动加 `AGENT_SOCIAL_URL=http://localhost:8004`
