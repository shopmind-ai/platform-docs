# ShopMind AI Phase 6 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 构建 agent-sales-training（销售培训系统），AI 扮演不同类型客户与销售学员多轮对话，对话结束后评分并给出改进建议，支持三种场景 × 三种客户类型 = 9 种训练组合。

**Architecture:** 不使用 LangGraph（多轮对话是循环结构，不适合线性图）。FastAPI 直接处理三个 action（start/chat/evaluate），`customer_agent.py` 负责 AI 客户回复，`scorer_agent.py` 负责对话评分，`session.py` 负责 PostgreSQL 会话读写。会话消息以 JSONB 格式存入 `training_sessions` 表，支持跨请求多轮对话。

**Tech Stack:** Python 3.11 · FastAPI · Claude API · psycopg2 · python-dotenv

---

## 参考资料

- 设计文档：`shopmind-ai/docs/superpowers/specs/2026-05-31-shopmind-ai-design.md`
- Gateway 需配置：`AGENT_TRAINING_URL=http://localhost:8006`（在 gateway `.env` 手动添加）

---

## 文件结构总览

```
agent-sales-training/
├── main.py                    # FastAPI: POST /invoke, GET /health
├── config.py                  # env vars, get_llm()
├── customer_agent.py          # AI 客户模拟（根据场景+客户类型生成回复）
├── scorer_agent.py            # AI 评分（对整段对话打分 + 改进建议）
├── session.py                 # PostgreSQL 会话读写（create/load/append/save_score）
├── utils/
│   ├── __init__.py
│   ├── startup.py             # 建 training_sessions 表
│   └── usage.py               # Token 统计
├── tests/
│   ├── __init__.py
│   ├── test_customer_agent.py
│   ├── test_scorer_agent.py
│   ├── test_session.py
│   └── test_integration.py
├── requirements.txt
├── .env.example
└── .gitignore
```

---

## /invoke 端点契约

```
POST /invoke

# 开始新会话
Request:  {"action": "start", "product_name": "专业马拉松跑步鞋",
           "scenario": "objection_handling", "customer_type": "price_sensitive"}
Response: {"session_id": "uuid", "opening_message": "这双鞋这么贵，有必要买吗？",
           "usage": {"input_tokens": 200, "output_tokens": 30}}

# 对话一轮
Request:  {"action": "chat", "session_id": "uuid",
           "message": "这双鞋采用碳纤维中底，跑马拉松能节省体力..."}
Response: {"reply": "听起来不错，但我在某宝上看到类似的才200块...",
           "session_id": "uuid",
           "usage": {"input_tokens": 400, "output_tokens": 60}}

# 获取评分
Request:  {"action": "evaluate", "session_id": "uuid"}
Response: {"scores": {"communication": 8, "product_knowledge": 9,
                      "objection_handling": 7, "closing": 6},
           "overall": 7.5,
           "summary": "整体表现不错，产品知识扎实...",
           "suggestions": ["建议1", "建议2", "建议3"],
           "usage": {"input_tokens": 600, "output_tokens": 200}}
```

---

## Task 1：项目初始化 + config + utils

**Files:**
- Create: `agent-sales-training/requirements.txt`
- Create: `agent-sales-training/.env.example`
- Create: `agent-sales-training/.gitignore`
- Create: `agent-sales-training/config.py`
- Create: `agent-sales-training/utils/usage.py`
- Create: `agent-sales-training/utils/startup.py`

- [ ] **Step 1: 创建目录结构**

```bash
cd /Users/liuyun/CCWorkSpace/shopmind-ai
mkdir -p agent-sales-training/{utils,tests}
touch agent-sales-training/utils/__init__.py
touch agent-sales-training/tests/__init__.py
cd agent-sales-training && git init && git checkout -b develop
git commit --allow-empty -m "chore: init agent-sales-training"
```

- [ ] **Step 2: 创建 requirements.txt**

```
fastapi>=0.115.0
uvicorn[standard]>=0.30.0
langchain>=0.3.0
langchain-anthropic>=0.3.0
langchain-ollama>=0.2.0
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


def from_response(response) -> TokenUsage:
    meta = getattr(response, "usage_metadata", None) or {}
    return {
        "input_tokens":  meta.get("input_tokens", 0),
        "output_tokens": meta.get("output_tokens", 0),
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
CREATE TABLE IF NOT EXISTS training_sessions (
    id            SERIAL PRIMARY KEY,
    session_id    VARCHAR(36) UNIQUE NOT NULL DEFAULT gen_random_uuid()::text,
    product_name  VARCHAR(200) NOT NULL,
    scenario      VARCHAR(30)  NOT NULL,
    customer_type VARCHAR(20)  NOT NULL,
    messages      JSONB        NOT NULL DEFAULT '[]',
    score         JSONB,
    input_tokens  INTEGER      DEFAULT 0,
    output_tokens INTEGER      DEFAULT 0,
    created_at    TIMESTAMPTZ  DEFAULT NOW(),
    updated_at    TIMESTAMPTZ  DEFAULT NOW()
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
        print("[startup] agent-sales-training DB initialized.")
    finally:
        conn.close()
```

- [ ] **Step 8: 安装依赖并验证**

```bash
pip install -r requirements.txt -q
python -c "import fastapi, psycopg2, langchain_anthropic; print('imports OK')"
```

Expected: `imports OK`

- [ ] **Step 9: commit**

```bash
git add . && git commit -m "feat: project setup, config, utils (startup + usage)"
```

---

## Task 2：session.py（PostgreSQL 会话管理）+ 测试

**Files:**
- Create: `agent-sales-training/session.py`
- Create: `agent-sales-training/tests/test_session.py`

- [ ] **Step 1: 写失败测试**

创建 `tests/test_session.py`：

```python
from unittest.mock import patch, MagicMock
import json


def test_create_session_returns_session_id():
    from session import create_session
    with patch("session.psycopg2.connect") as mock_conn_cls:
        mock_conn = MagicMock()
        mock_cur = MagicMock()
        mock_cur.__enter__ = MagicMock(return_value=mock_cur)
        mock_cur.__exit__ = MagicMock(return_value=False)
        mock_cur.fetchone.return_value = ("generated-uuid-001",)
        mock_conn.cursor.return_value = mock_cur
        mock_conn.__enter__ = MagicMock(return_value=mock_conn)
        mock_conn.__exit__ = MagicMock(return_value=False)
        mock_conn_cls.return_value = mock_conn

        session_id = create_session("跑步鞋", "objection_handling", "price_sensitive")

    assert session_id == "generated-uuid-001"


def test_load_session_returns_dict():
    from session import load_session
    with patch("session.psycopg2.connect") as mock_conn_cls:
        mock_conn = MagicMock()
        mock_cur = MagicMock()
        mock_cur.__enter__ = MagicMock(return_value=mock_cur)
        mock_cur.__exit__ = MagicMock(return_value=False)
        mock_cur.fetchone.return_value = (
            "ses-001", "跑步鞋", "objection_handling", "price_sensitive",
            [{"role": "assistant", "content": "您好"}], None
        )
        mock_conn.cursor.return_value = mock_cur
        mock_conn.__enter__ = MagicMock(return_value=mock_conn)
        mock_conn.__exit__ = MagicMock(return_value=False)
        mock_conn_cls.return_value = mock_conn

        sess = load_session("ses-001")

    assert sess["session_id"] == "ses-001"
    assert sess["product_name"] == "跑步鞋"
    assert isinstance(sess["messages"], list)


def test_load_session_returns_none_for_missing():
    from session import load_session
    with patch("session.psycopg2.connect") as mock_conn_cls:
        mock_conn = MagicMock()
        mock_cur = MagicMock()
        mock_cur.__enter__ = MagicMock(return_value=mock_cur)
        mock_cur.__exit__ = MagicMock(return_value=False)
        mock_cur.fetchone.return_value = None
        mock_conn.cursor.return_value = mock_cur
        mock_conn.__enter__ = MagicMock(return_value=mock_conn)
        mock_conn.__exit__ = MagicMock(return_value=False)
        mock_conn_cls.return_value = mock_conn

        result = load_session("nonexistent")

    assert result is None


def test_append_messages_updates_db():
    from session import append_messages
    with patch("session.psycopg2.connect") as mock_conn_cls:
        mock_conn = MagicMock()
        mock_cur = MagicMock()
        mock_cur.__enter__ = MagicMock(return_value=mock_cur)
        mock_cur.__exit__ = MagicMock(return_value=False)
        mock_conn.cursor.return_value = mock_cur
        mock_conn.__enter__ = MagicMock(return_value=mock_conn)
        mock_conn.__exit__ = MagicMock(return_value=False)
        mock_conn_cls.return_value = mock_conn

        append_messages("ses-001", [
            {"role": "user", "content": "产品很好"},
            {"role": "assistant", "content": "谢谢"},
        ])

    assert mock_cur.execute.called
    assert mock_conn.commit.called
```

- [ ] **Step 2: 运行，确认失败**

```bash
cd /Users/liuyun/CCWorkSpace/shopmind-ai/agent-sales-training
python -m pytest tests/test_session.py -v 2>&1 | tail -5
```

Expected: FAILED（ImportError）

- [ ] **Step 3: 创建 session.py**

```python
import json
import psycopg2
import psycopg2.extras
from config import DATABASE_URL


def create_session(product_name: str, scenario: str, customer_type: str) -> str:
    conn = psycopg2.connect(DATABASE_URL)
    try:
        with conn.cursor() as cur:
            cur.execute(
                "INSERT INTO training_sessions (product_name, scenario, customer_type) "
                "VALUES (%s, %s, %s) RETURNING session_id",
                (product_name, scenario, customer_type)
            )
            session_id = cur.fetchone()[0]
        conn.commit()
        return session_id
    finally:
        conn.close()


def load_session(session_id: str) -> dict | None:
    conn = psycopg2.connect(DATABASE_URL)
    try:
        with conn.cursor() as cur:
            cur.execute(
                "SELECT session_id, product_name, scenario, customer_type, messages, score "
                "FROM training_sessions WHERE session_id = %s",
                (session_id,)
            )
            row = cur.fetchone()
            if not row:
                return None
            msgs = row[4] if isinstance(row[4], list) else json.loads(row[4] or "[]")
            return {
                "session_id":    row[0],
                "product_name":  row[1],
                "scenario":      row[2],
                "customer_type": row[3],
                "messages":      msgs,
                "score":         row[5],
            }
    finally:
        conn.close()


def append_messages(session_id: str, new_messages: list[dict]):
    conn = psycopg2.connect(DATABASE_URL)
    try:
        with conn.cursor() as cur:
            cur.execute(
                "UPDATE training_sessions "
                "SET messages = messages || %s::jsonb, updated_at = NOW() "
                "WHERE session_id = %s",
                (json.dumps(new_messages, ensure_ascii=False), session_id)
            )
        conn.commit()
    finally:
        conn.close()


def save_score(session_id: str, score: dict):
    conn = psycopg2.connect(DATABASE_URL)
    try:
        with conn.cursor() as cur:
            cur.execute(
                "UPDATE training_sessions SET score = %s, updated_at = NOW() "
                "WHERE session_id = %s",
                (json.dumps(score, ensure_ascii=False), session_id)
            )
        conn.commit()
    finally:
        conn.close()
```

- [ ] **Step 4: 运行，确认通过**

```bash
python -m pytest tests/test_session.py -v 2>&1 | tail -8
```

Expected: 4 PASSED

- [ ] **Step 5: commit**

```bash
git add session.py tests/test_session.py
git commit -m "feat: session management (PostgreSQL JSONB message history)"
```

---

## Task 3：Customer Agent + 测试

**Files:**
- Create: `agent-sales-training/customer_agent.py`
- Create: `agent-sales-training/tests/test_customer_agent.py`

- [ ] **Step 1: 写失败测试**

创建 `tests/test_customer_agent.py`：

```python
from unittest.mock import patch, MagicMock


def test_generate_opening_returns_string():
    from customer_agent import generate_opening
    mock_resp = MagicMock()
    mock_resp.content = "这双鞋这么贵，值得买吗？"
    mock_resp.usage_metadata = {"input_tokens": 200, "output_tokens": 20}
    with patch("customer_agent.get_llm") as mock_llm:
        mock_llm.return_value.invoke.return_value = mock_resp
        result = generate_opening("跑步鞋", "objection_handling", "price_sensitive")
    assert isinstance(result["message"], str)
    assert len(result["message"]) > 0
    assert result["usage"]["input_tokens"] == 200


def test_generate_reply_includes_customer_style():
    from customer_agent import generate_reply
    mock_resp = MagicMock()
    mock_resp.content = "但是竞品价格更低，有什么优势？"
    mock_resp.usage_metadata = {"input_tokens": 400, "output_tokens": 40}
    history = [
        {"role": "assistant", "content": "这双鞋有什么问题？"},
        {"role": "user", "content": "这双鞋采用碳纤维中底..."},
    ]
    with patch("customer_agent.get_llm") as mock_llm:
        mock_llm.return_value.invoke.return_value = mock_resp
        result = generate_reply("跑步鞋", "objection_handling", "price_sensitive", history)
    assert isinstance(result["reply"], str)
    assert len(result["reply"]) > 0
    assert result["usage"]["input_tokens"] == 400


def test_generate_opening_uses_scenario_context():
    from customer_agent import generate_opening
    mock_resp = MagicMock()
    mock_resp.content = "我想了解一下这款跑鞋"
    mock_resp.usage_metadata = {"input_tokens": 150, "output_tokens": 15}
    with patch("customer_agent.get_llm") as mock_llm:
        mock_llm.return_value.invoke.return_value = mock_resp
        generate_opening("跑步鞋", "product_intro", "hesitant")
    call_prompt = mock_llm.return_value.invoke.call_args[0][0]
    assert "跑步鞋" in call_prompt
    assert len(call_prompt) > 50
```

- [ ] **Step 2: 运行，确认失败**

```bash
python -m pytest tests/test_customer_agent.py -v 2>&1 | tail -5
```

Expected: FAILED（ImportError）

- [ ] **Step 3: 创建 customer_agent.py**

```python
from config import get_llm
from utils.usage import from_response

_SCENARIO_CONTEXT = {
    "product_intro":       "你对这款{product}初次接触，想了解基本信息，会问比较基础的问题。",
    "objection_handling":  "你对{product}有一定了解，但有各种顾虑，等待销售员说服你。",
    "closing":             "你已经了解{product}，在最后决策阶段，有些犹豫是否立刻购买。",
}

_CUSTOMER_STYLE = {
    "skeptical":      "你性格多疑，经常质疑销售员的说法，要求证据，不轻易相信任何承诺。",
    "hesitant":       "你性格犹豫，总是需要更多信息才能做决定，容易被细节问题卡住。",
    "price_sensitive": "你非常注重性价比，经常提到竞品价格或要求折扣，对价格极为敏感。",
}

_SCENARIO_NAMES = {
    "product_intro":      "商品介绍",
    "objection_handling": "异议处理",
    "closing":            "促成成交",
}

_OPENING_PROMPT = """你正在扮演一位{style}的客户。{scenario_context}

你是顾客，产品是：{product}

请用1-2句话开始对话，展现你的{customer_type}性格特点。不要自我介绍，直接进入对话。只返回你说的话，不要有任何说明。"""

_REPLY_PROMPT = """你正在扮演一位{style}的客户。{scenario_context}

产品：{product}

对话历史：
{history_text}

销售员刚说："{last_message}"

请用1-3句话回复，保持你的{customer_type}性格特点。如果销售员表现很好，可以稍微松动立场，但不要主动成交。只返回你说的话，不要有任何说明。"""


def _format_history(messages: list[dict]) -> str:
    lines = []
    for msg in messages:
        role = "客户" if msg["role"] == "assistant" else "销售员"
        lines.append(f"{role}：{msg['content']}")
    return "\n".join(lines)


def generate_opening(product_name: str, scenario: str, customer_type: str) -> dict:
    llm = get_llm()
    scenario_ctx = _SCENARIO_CONTEXT.get(scenario, _SCENARIO_CONTEXT["product_intro"]).format(product=product_name)
    style = _CUSTOMER_STYLE.get(customer_type, _CUSTOMER_STYLE["hesitant"])

    prompt = _OPENING_PROMPT.format(
        style=style,
        scenario_context=scenario_ctx,
        product=product_name,
        customer_type=customer_type,
    )
    response = llm.invoke(prompt)
    usage = from_response(response)
    return {"message": response.content.strip(), "usage": usage}


def generate_reply(product_name: str, scenario: str, customer_type: str,
                   messages: list[dict]) -> dict:
    llm = get_llm()
    scenario_ctx = _SCENARIO_CONTEXT.get(scenario, _SCENARIO_CONTEXT["product_intro"]).format(product=product_name)
    style = _CUSTOMER_STYLE.get(customer_type, _CUSTOMER_STYLE["hesitant"])
    last_message = messages[-1]["content"] if messages else ""
    history_text = _format_history(messages[:-1]) if len(messages) > 1 else "（对话刚开始）"

    prompt = _REPLY_PROMPT.format(
        style=style,
        scenario_context=scenario_ctx,
        product=product_name,
        history_text=history_text,
        last_message=last_message,
        customer_type=customer_type,
    )
    response = llm.invoke(prompt)
    usage = from_response(response)
    return {"reply": response.content.strip(), "usage": usage}
```

- [ ] **Step 4: 运行，确认通过**

```bash
python -m pytest tests/test_customer_agent.py -v 2>&1 | tail -6
```

Expected: 3 PASSED

- [ ] **Step 5: commit**

```bash
git add customer_agent.py tests/test_customer_agent.py
git commit -m "feat: customer agent (3 scenarios × 3 customer types)"
```

---

## Task 4：Scorer Agent + 测试

**Files:**
- Create: `agent-sales-training/scorer_agent.py`
- Create: `agent-sales-training/tests/test_scorer_agent.py`

- [ ] **Step 1: 写失败测试**

创建 `tests/test_scorer_agent.py`：

```python
import json
from unittest.mock import patch, MagicMock


def test_score_conversation_returns_all_dimensions():
    from scorer_agent import score_conversation
    mock_score = {
        "scores": {"communication": 8, "product_knowledge": 9,
                   "objection_handling": 7, "closing": 6},
        "overall": 7.5,
        "summary": "整体表现不错，产品知识扎实。",
        "suggestions": ["建议1", "建议2", "建议3"]
    }
    mock_resp = MagicMock()
    mock_resp.content = json.dumps(mock_score)
    mock_resp.usage_metadata = {"input_tokens": 600, "output_tokens": 200}
    messages = [
        {"role": "assistant", "content": "这双鞋这么贵？"},
        {"role": "user", "content": "碳纤维中底回弹好，值得投资。"},
        {"role": "assistant", "content": "但竞品更便宜..."},
        {"role": "user", "content": "我们品质更好，有质保。"},
    ]
    with patch("scorer_agent.get_llm") as mock_llm:
        mock_llm.return_value.invoke.return_value = mock_resp
        result = score_conversation("跑步鞋", "objection_handling", messages)
    assert set(result["scores"].keys()) == {"communication", "product_knowledge",
                                             "objection_handling", "closing"}
    assert result["overall"] == 7.5
    assert len(result["suggestions"]) == 3
    assert result["usage"]["input_tokens"] == 600


def test_score_handles_invalid_json():
    from scorer_agent import score_conversation
    mock_resp = MagicMock()
    mock_resp.content = "无法评估这段对话"
    mock_resp.usage_metadata = {"input_tokens": 300, "output_tokens": 20}
    with patch("scorer_agent.get_llm") as mock_llm:
        mock_llm.return_value.invoke.return_value = mock_resp
        result = score_conversation("跑步鞋", "closing", [])
    assert result["scores"]["communication"] == 0
    assert result["overall"] == 0.0
    assert len(result["suggestions"]) > 0


def test_score_requires_minimum_messages():
    from scorer_agent import score_conversation
    result = score_conversation("跑步鞋", "product_intro", [])
    assert result["overall"] == 0.0
    assert "对话内容不足" in result["summary"]
```

- [ ] **Step 2: 运行，确认失败**

```bash
python -m pytest tests/test_scorer_agent.py -v 2>&1 | tail -5
```

Expected: FAILED（ImportError）

- [ ] **Step 3: 创建 scorer_agent.py**

```python
import json
from config import get_llm
from utils.usage import from_response

_SCENARIO_NAMES = {
    "product_intro":      "商品介绍（向客户介绍产品特点）",
    "objection_handling": "异议处理（应对客户的反对意见）",
    "closing":            "促成成交（引导客户做出购买决定）",
}

_SCORE_PROMPT = """你是一名资深销售培训师，评估以下销售对话的表现。

产品：{product}
场景：{scenario_desc}

对话记录（客户=assistant，销售员=user）：
{conversation}

请从以下4个维度评分（1-10分）并给出总体评价和改进建议：
- communication（沟通技巧）：语言表达清晰度、倾听能力、情绪管理
- product_knowledge（产品知识）：对产品特点、优势、使用场景的掌握程度
- objection_handling（异议处理）：应对客户疑虑和反对意见的能力
- closing（成交意愿）：引导客户走向购买决策的能力

返回JSON（只返回JSON，不要有任何解释）：
{{
  "scores": {{"communication": 8, "product_knowledge": 7, "objection_handling": 6, "closing": 5}},
  "overall": 6.5,
  "summary": "总体评价（2-3句）",
  "suggestions": ["具体改进建议1", "具体改进建议2", "具体改进建议3"]
}}"""

_DEFAULT_SCORE = {
    "scores": {"communication": 0, "product_knowledge": 0,
               "objection_handling": 0, "closing": 0},
    "overall": 0.0,
    "summary": "评分失败，请重试。",
    "suggestions": ["请确保对话内容足够充分", "至少完成3轮对话后再评分", "确保回答涵盖产品核心卖点"],
}


def score_conversation(product_name: str, scenario: str, messages: list[dict]) -> dict:
    if len(messages) < 2:
        return {**_DEFAULT_SCORE,
                "summary": "对话内容不足，请至少完成2轮对话后再评分。",
                "usage": {"input_tokens": 0, "output_tokens": 0}}

    llm = get_llm()
    scenario_desc = _SCENARIO_NAMES.get(scenario, scenario)
    lines = []
    for msg in messages:
        role = "客户" if msg["role"] == "assistant" else "销售员"
        lines.append(f"{role}：{msg['content']}")
    conversation = "\n".join(lines)

    prompt = _SCORE_PROMPT.format(
        product=product_name,
        scenario_desc=scenario_desc,
        conversation=conversation,
    )
    response = llm.invoke(prompt)
    usage = from_response(response)

    try:
        raw = response.content.strip().strip("```json").strip("```").strip()
        data = json.loads(raw)
        scores = data.get("scores", {})
        overall = data.get("overall", sum(scores.values()) / len(scores) if scores else 0.0)
        return {
            "scores":      scores,
            "overall":     round(float(overall), 1),
            "summary":     data.get("summary", ""),
            "suggestions": data.get("suggestions", [])[:3],
            "usage":       usage,
        }
    except (json.JSONDecodeError, ValueError):
        return {**_DEFAULT_SCORE, "usage": usage}
```

- [ ] **Step 4: 运行，确认通过**

```bash
python -m pytest tests/test_scorer_agent.py -v 2>&1 | tail -6
```

Expected: 3 PASSED

- [ ] **Step 5: commit**

```bash
git add scorer_agent.py tests/test_scorer_agent.py
git commit -m "feat: scorer agent (4-dimension scoring + suggestions)"
```

---

## Task 5：main.py + 集成测试

**Files:**
- Create: `agent-sales-training/main.py`
- Create: `agent-sales-training/tests/test_integration.py`

- [ ] **Step 1: 创建 main.py**

```python
from contextlib import asynccontextmanager
from fastapi import FastAPI
from pydantic import BaseModel
from utils.startup import check_and_init


@asynccontextmanager
async def lifespan(app: FastAPI):
    check_and_init()
    yield


app = FastAPI(title="ShopMind Sales Training Agent", version="1.0.0", lifespan=lifespan)


class InvokeRequest(BaseModel):
    action: str                       # "start" | "chat" | "evaluate"
    product_name: str = ""
    scenario: str = "product_intro"   # product_intro | objection_handling | closing
    customer_type: str = "hesitant"   # skeptical | hesitant | price_sensitive
    session_id: str = ""
    message: str = ""


@app.post("/invoke")
def invoke(req: InvokeRequest):
    if req.action == "start":
        if not req.product_name:
            return {"error": "product_name is required for start action"}

        from session import create_session
        from customer_agent import generate_opening

        session_id = create_session(req.product_name, req.scenario, req.customer_type)
        opening = generate_opening(req.product_name, req.scenario, req.customer_type)

        from session import append_messages
        append_messages(session_id, [{"role": "assistant", "content": opening["message"]}])

        return {
            "session_id":      session_id,
            "opening_message": opening["message"],
            "usage":           opening["usage"],
        }

    elif req.action == "chat":
        if not req.session_id or not req.message:
            return {"error": "session_id and message are required for chat action"}

        from session import load_session, append_messages
        sess = load_session(req.session_id)
        if not sess:
            return {"error": f"Session not found: {req.session_id}"}

        user_msg = {"role": "user", "content": req.message}
        updated_messages = sess["messages"] + [user_msg]

        from customer_agent import generate_reply
        reply_data = generate_reply(
            sess["product_name"], sess["scenario"],
            sess["customer_type"], updated_messages
        )

        append_messages(req.session_id, [
            user_msg,
            {"role": "assistant", "content": reply_data["reply"]},
        ])

        return {
            "reply":      reply_data["reply"],
            "session_id": req.session_id,
            "usage":      reply_data["usage"],
        }

    elif req.action == "evaluate":
        if not req.session_id:
            return {"error": "session_id is required for evaluate action"}

        from session import load_session, save_score
        sess = load_session(req.session_id)
        if not sess:
            return {"error": f"Session not found: {req.session_id}"}

        from scorer_agent import score_conversation
        result = score_conversation(
            sess["product_name"], sess["scenario"], sess["messages"]
        )
        save_score(req.session_id, {
            "scores": result["scores"], "overall": result["overall"],
            "summary": result["summary"], "suggestions": result["suggestions"],
        })

        return {
            "scores":      result["scores"],
            "overall":     result["overall"],
            "summary":     result["summary"],
            "suggestions": result["suggestions"],
            "usage":       result["usage"],
        }

    else:
        return {"error": f"Unknown action: {req.action}. Use 'start', 'chat', or 'evaluate'."}


@app.get("/health")
def health():
    return {"status": "ok", "service": "agent-sales-training"}
```

- [ ] **Step 2: 写集成测试**

创建 `tests/test_integration.py`：

```python
import pytest
from unittest.mock import patch, MagicMock
from fastapi.testclient import TestClient


@pytest.fixture
def client():
    with patch("utils.startup.check_and_init"):
        import importlib
        import main as m
        importlib.reload(m)
        yield TestClient(m.app)


def test_start_action(client):
    with patch("main.create_session", return_value="ses-abc"), \
         patch("main.generate_opening", return_value={
             "message": "这双鞋这么贵？", "usage": {"input_tokens": 200, "output_tokens": 20}
         }), \
         patch("main.append_messages"):
        resp = client.post("/invoke", json={
            "action": "start",
            "product_name": "跑步鞋",
            "scenario": "objection_handling",
            "customer_type": "price_sensitive",
        })
    assert resp.status_code == 200
    body = resp.json()
    assert body["session_id"] == "ses-abc"
    assert body["opening_message"] == "这双鞋这么贵？"


def test_start_missing_product_returns_error(client):
    resp = client.post("/invoke", json={"action": "start"})
    assert "error" in resp.json()


def test_chat_action(client):
    with patch("main.load_session", return_value={
        "session_id": "ses-abc", "product_name": "跑步鞋",
        "scenario": "objection_handling", "customer_type": "price_sensitive",
        "messages": [{"role": "assistant", "content": "这双鞋贵吗？"}], "score": None
    }), \
    patch("main.generate_reply", return_value={
        "reply": "竞品价格更低...", "usage": {"input_tokens": 400, "output_tokens": 50}
    }), \
    patch("main.append_messages"):
        resp = client.post("/invoke", json={
            "action": "chat",
            "session_id": "ses-abc",
            "message": "我们品质更好！",
        })
    assert resp.status_code == 200
    assert resp.json()["reply"] == "竞品价格更低..."
    assert resp.json()["session_id"] == "ses-abc"


def test_chat_missing_session_returns_error(client):
    with patch("main.load_session", return_value=None):
        resp = client.post("/invoke", json={
            "action": "chat", "session_id": "nonexistent", "message": "hello"
        })
    assert "error" in resp.json()


def test_evaluate_action(client):
    with patch("main.load_session", return_value={
        "session_id": "ses-abc", "product_name": "跑步鞋",
        "scenario": "objection_handling", "customer_type": "price_sensitive",
        "messages": [
            {"role": "assistant", "content": "贵"},
            {"role": "user", "content": "品质好"},
            {"role": "assistant", "content": "还是贵"},
            {"role": "user", "content": "有质保"},
        ], "score": None
    }), \
    patch("main.score_conversation", return_value={
        "scores": {"communication": 8, "product_knowledge": 9,
                   "objection_handling": 7, "closing": 6},
        "overall": 7.5, "summary": "不错", "suggestions": ["s1", "s2", "s3"],
        "usage": {"input_tokens": 600, "output_tokens": 200}
    }), \
    patch("main.save_score"):
        resp = client.post("/invoke", json={"action": "evaluate", "session_id": "ses-abc"})
    assert resp.status_code == 200
    body = resp.json()
    assert body["overall"] == 7.5
    assert len(body["suggestions"]) == 3


def test_health_endpoint(client):
    resp = client.get("/health")
    assert resp.status_code == 200
    assert resp.json()["service"] == "agent-sales-training"
```

- [ ] **Step 3: 运行全部测试**

```bash
python -m pytest tests/ -v 2>&1 | tail -20
```

Expected: 全部 PASSED（约 13 个）

- [ ] **Step 4: 手动验证服务**

```bash
cp /Users/liuyun/CCWorkSpace/insurance-agent/.env .env
python -m uvicorn main:app --port 8006 --reload
curl -s http://localhost:8006/health
```

Expected: `{"status":"ok","service":"agent-sales-training"}`

- [ ] **Step 5: commit**

```bash
git add main.py tests/test_integration.py
git commit -m "feat: FastAPI /invoke (start/chat/evaluate) + 13 tests passing"
```

---

## Task 6：更新 shopmind-web（销售培训页）

**Files:**
- Modify: `shopmind-web/app/dashboard/layout.tsx`
- Create: `shopmind-web/app/dashboard/training/page.tsx`
- Modify: `shopmind-web/app/dashboard/page.tsx` — 更新为 6/8

- [ ] **Step 1: 切换到 shopmind-web develop 分支**

```bash
cd /Users/liuyun/CCWorkSpace/shopmind-ai/shopmind-web
git checkout develop
mkdir -p app/dashboard/training
```

- [ ] **Step 2: 更新侧边栏 layout.tsx**

将 navItems 的最后一项后面新增：

```typescript
  { href: "/dashboard/training",  label: "销售培训", emoji: "🎯" },
```

完整 navItems：
```typescript
const navItems = [
  { href: "/dashboard",           label: "概览",   emoji: "📊" },
  { href: "/dashboard/cleaning",  label: "数据清洗", emoji: "🧹" },
  { href: "/dashboard/customer",  label: "智能客服", emoji: "💬" },
  { href: "/dashboard/media",     label: "素材库",   emoji: "🖼️" },
  { href: "/dashboard/social",    label: "内容创作", emoji: "✍️" },
  { href: "/dashboard/clip",      label: "直播切片", emoji: "✂️" },
  { href: "/dashboard/training",  label: "销售培训", emoji: "🎯" },
];
```

- [ ] **Step 3: 更新 dashboard page.tsx（6/8）**

将 `5 / 8` 改为 `6 / 8`。

- [ ] **Step 4: 创建 app/dashboard/training/page.tsx**

```typescript
"use client";
import { useState } from "react";
import { api } from "@/lib/api";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";

const SCENARIOS = [
  { value: "product_intro",      label: "商品介绍", desc: "向客户介绍产品特点" },
  { value: "objection_handling", label: "异议处理", desc: "应对客户反对意见" },
  { value: "closing",            label: "促成成交", desc: "引导客户做出购买决定" },
];

const CUSTOMER_TYPES = [
  { value: "skeptical",      label: "挑剔型", emoji: "🤨" },
  { value: "hesitant",       label: "犹豫型", emoji: "😕" },
  { value: "price_sensitive", label: "价格敏感型", emoji: "💰" },
];

type Message = { role: "user" | "assistant"; content: string };
type ScoreResult = {
  scores: Record<string, number>;
  overall: number;
  summary: string;
  suggestions: string[];
  usage: { input_tokens: number; output_tokens: number };
};

const SCORE_LABELS: Record<string, string> = {
  communication: "沟通技巧",
  product_knowledge: "产品知识",
  objection_handling: "异议处理",
  closing: "成交能力",
};

export default function TrainingPage() {
  const [productName, setProductName] = useState("专业马拉松跑步鞋");
  const [scenario, setScenario] = useState("objection_handling");
  const [customerType, setCustomerType] = useState("price_sensitive");
  const [sessionId, setSessionId] = useState<string | null>(null);
  const [messages, setMessages] = useState<Message[]>([]);
  const [inputText, setInputText] = useState("");
  const [loading, setLoading] = useState(false);
  const [score, setScore] = useState<ScoreResult | null>(null);
  const [error, setError] = useState("");

  async function handleStart() {
    if (!productName.trim()) return;
    setLoading(true); setError(""); setScore(null); setMessages([]);
    try {
      const res = await api.invokeAgent("training", {
        action: "start", product_name: productName,
        scenario, customer_type: customerType,
      }) as { session_id: string; opening_message: string };
      setSessionId(res.session_id);
      setMessages([{ role: "assistant", content: res.opening_message }]);
    } catch (e) { setError(e instanceof Error ? e.message : "启动失败"); }
    finally { setLoading(false); }
  }

  async function handleChat() {
    if (!inputText.trim() || !sessionId) return;
    const userMsg = inputText.trim();
    setInputText(""); setLoading(true); setError("");
    setMessages(prev => [...prev, { role: "user", content: userMsg }]);
    try {
      const res = await api.invokeAgent("training", {
        action: "chat", session_id: sessionId, message: userMsg,
      }) as { reply: string };
      setMessages(prev => [...prev, { role: "assistant", content: res.reply }]);
    } catch (e) { setError(e instanceof Error ? e.message : "对话失败"); }
    finally { setLoading(false); }
  }

  async function handleEvaluate() {
    if (!sessionId) return;
    setLoading(true); setError("");
    try {
      const res = await api.invokeAgent("training", {
        action: "evaluate", session_id: sessionId,
      }) as ScoreResult;
      setScore(res);
    } catch (e) { setError(e instanceof Error ? e.message : "评分失败"); }
    finally { setLoading(false); }
  }

  const scenarioInfo = SCENARIOS.find(s => s.value === scenario);
  const customerInfo = CUSTOMER_TYPES.find(c => c.value === customerType);

  return (
    <div className="max-w-4xl space-y-6">
      <div>
        <h1 className="text-2xl font-bold">🎯 销售培训</h1>
        <p className="text-slate-500 text-sm mt-1">AI 扮演客户陪你练习销售技巧，完成后获得 AI 评分</p>
      </div>

      {!sessionId ? (
        <Card>
          <CardContent className="p-4 space-y-4">
            <div className="space-y-1">
              <label className="text-sm font-medium">训练商品</label>
              <Input value={productName} onChange={e => setProductName(e.target.value)} placeholder="如：专业马拉松跑步鞋" />
            </div>
            <div className="space-y-2">
              <label className="text-sm font-medium">训练场景</label>
              <div className="flex gap-2">
                {SCENARIOS.map(s => (
                  <button key={s.value} onClick={() => setScenario(s.value)}
                    className={`px-3 py-1.5 rounded-lg text-sm border transition-colors ${
                      scenario === s.value ? "bg-slate-900 text-white border-slate-900" : "bg-white text-slate-700 border-slate-200 hover:border-slate-400"
                    }`}>
                    {s.label}
                  </button>
                ))}
              </div>
              {scenarioInfo && <p className="text-xs text-slate-400">{scenarioInfo.desc}</p>}
            </div>
            <div className="space-y-2">
              <label className="text-sm font-medium">客户类型</label>
              <div className="flex gap-2">
                {CUSTOMER_TYPES.map(c => (
                  <button key={c.value} onClick={() => setCustomerType(c.value)}
                    className={`px-3 py-1.5 rounded-lg text-sm border transition-colors ${
                      customerType === c.value ? "bg-slate-900 text-white border-slate-900" : "bg-white text-slate-700 border-slate-200 hover:border-slate-400"
                    }`}>
                    {c.emoji} {c.label}
                  </button>
                ))}
              </div>
            </div>
            <Button onClick={handleStart} disabled={loading || !productName.trim()} className="w-full">
              {loading ? "启动中..." : "开始训练"}
            </Button>
            {error && <p className="text-red-500 text-sm">{error}</p>}
          </CardContent>
        </Card>
      ) : (
        <div className="space-y-4">
          <div className="flex items-center gap-3 text-sm text-slate-500">
            <Badge variant="outline">{scenarioInfo?.label}</Badge>
            <Badge variant="outline">{customerInfo?.emoji} {customerInfo?.label}</Badge>
            <span className="font-mono text-xs">{sessionId.slice(0, 8)}...</span>
            <Button size="sm" variant="ghost" onClick={() => { setSessionId(null); setMessages([]); setScore(null); }}>
              重新开始
            </Button>
          </div>

          <Card>
            <CardContent className="p-4 h-80 overflow-y-auto space-y-3">
              {messages.map((msg, i) => (
                <div key={i} className={`flex ${msg.role === "user" ? "justify-end" : "justify-start"}`}>
                  <div className={`max-w-[80%] px-4 py-2 rounded-2xl text-sm whitespace-pre-wrap ${
                    msg.role === "user"
                      ? "bg-slate-900 text-white rounded-br-sm"
                      : "bg-slate-100 text-slate-800 rounded-bl-sm"
                  }`}>
                    {msg.role === "assistant" && <span className="text-xs text-slate-400 block mb-1">{customerInfo?.emoji} AI 客户</span>}
                    {msg.content}
                  </div>
                </div>
              ))}
              {loading && (
                <div className="flex justify-start">
                  <div className="bg-slate-100 px-4 py-2 rounded-2xl rounded-bl-sm text-sm text-slate-500">思考中...</div>
                </div>
              )}
            </CardContent>
          </Card>

          <div className="flex gap-2">
            <Input value={inputText} onChange={e => setInputText(e.target.value)}
              onKeyDown={e => e.key === "Enter" && !e.shiftKey && handleChat()}
              placeholder="输入你的销售话术，按 Enter 发送..."
              disabled={loading} className="flex-1" />
            <Button onClick={handleChat} disabled={loading || !inputText.trim()}>发送</Button>
            <Button onClick={handleEvaluate} disabled={loading || messages.length < 4} variant="outline">获取评分</Button>
          </div>
          {error && <p className="text-red-500 text-sm">{error}</p>}

          {score && (
            <Card>
              <CardHeader className="pb-2">
                <div className="flex items-center justify-between">
                  <CardTitle className="text-base">📊 AI 评分结果</CardTitle>
                  <span className="text-2xl font-bold text-slate-800">{score.overall} <span className="text-sm font-normal text-slate-400">/ 10</span></span>
                </div>
              </CardHeader>
              <CardContent className="space-y-3">
                <div className="grid grid-cols-2 gap-2">
                  {Object.entries(score.scores).map(([key, val]) => (
                    <div key={key} className="flex items-center justify-between bg-slate-50 rounded px-3 py-2">
                      <span className="text-xs text-slate-600">{SCORE_LABELS[key] ?? key}</span>
                      <span className="font-bold text-sm">{val}</span>
                    </div>
                  ))}
                </div>
                <p className="text-sm text-slate-700">{score.summary}</p>
                <div className="space-y-1">
                  <p className="text-xs font-medium text-slate-500">改进建议：</p>
                  {score.suggestions.map((s, i) => (
                    <p key={i} className="text-xs text-slate-600">• {s}</p>
                  ))}
                </div>
                <p className="text-xs text-slate-400 text-right">
                  Tokens: {score.usage.input_tokens + score.usage.output_tokens}
                </p>
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
cd /Users/liuyun/CCWorkSpace/shopmind-ai/shopmind-web && npm run build 2>&1 | tail -10
```

Expected: 构建成功，新增 `/dashboard/training` 路由。

- [ ] **Step 6: commit 并推送**

```bash
git add app/dashboard/layout.tsx app/dashboard/training/ app/dashboard/page.tsx
git commit -m "feat: sales training page (start/chat/evaluate) + update agent count to 6/8"
git push origin develop
```

---

## Task 7：Gateway + 端到端测试 + 推送

- [ ] **Step 1: 注册 gateway**

```bash
echo "AGENT_TRAINING_URL=http://localhost:8006" >> /Users/liuyun/CCWorkSpace/shopmind-ai/shopmind-gateway/.env
```

- [ ] **Step 2: 启动 training agent**

```bash
cd /Users/liuyun/CCWorkSpace/shopmind-ai/agent-sales-training
nohup python -m uvicorn main:app --port 8006 > /tmp/training_agent.log 2>&1 &
sleep 4 && curl -s http://localhost:8006/health
```

Expected: `{"status":"ok","service":"agent-sales-training"}`

- [ ] **Step 3: 重启 gateway 并验证代理**

```bash
pkill -f "uvicorn main:app --port 8000" 2>/dev/null || true; sleep 1
cd /Users/liuyun/CCWorkSpace/shopmind-ai/shopmind-gateway
nohup python -m uvicorn main:app --port 8000 > /tmp/gateway.log 2>&1 &
sleep 4

TOKEN=$(curl -s -X POST http://localhost:8000/auth/login \
  -H "Content-Type: application/json" \
  -d '{"username":"demo","password":"shopmind2026"}' \
  | python3 -c "import sys,json; print(json.load(sys.stdin)['access_token'])")

curl -s --max-time 60 -X POST http://localhost:8000/api/v1/training/invoke \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer $TOKEN" \
  -d '{"action":"start","product_name":"专业马拉松跑步鞋","scenario":"objection_handling","customer_type":"price_sensitive"}' \
  | python3 -c "import sys,json; r=json.load(sys.stdin); print('✅ session:', r.get('session_id','')[:8], '| opening:', r.get('opening_message','')[:40])"
```

Expected: 返回 session_id 和 AI 客户开场白

- [ ] **Step 4: 推送到 GitHub**

```bash
cd /Users/liuyun/CCWorkSpace/shopmind-ai/agent-sales-training
git add . && git commit -m "chore: phase6 complete - sales training agent integrated with gateway"
gh repo create shopmind-ai/agent-sales-training \
  --public \
  --description "ShopMind AI - Sales training: AI customer simulation (3 scenarios × 3 types) + 4-dimension scoring" \
  --source=. --remote=origin --push

cd /Users/liuyun/CCWorkSpace/shopmind-ai/shopmind-gateway
git commit --allow-empty -m "chore: add AGENT_TRAINING_URL to gateway config"
git push origin develop
```

---

## 自审通过项

- ✅ 三个 action（start/chat/evaluate）在 main.py 中完整实现，无 TBD
- ✅ `session.py` 的 `append_messages` 使用 PostgreSQL JSONB concat（`messages || %s::jsonb`），不覆盖历史
- ✅ `customer_agent.py` 和 `scorer_agent.py` 的函数签名在 `main.py` 中调用一致
- ✅ `from_response()` 而非 `accumulate()`（Phase 6 不累计 state，每次调用独立计算）
- ✅ 集成测试 mock 了所有外部依赖（DB + LLM），无需真实连接
- ✅ 评分页面在收到 score 后显示，不影响正在进行的对话
- ✅ 获取评分按钮在消息不足 4 条时禁用（避免无意义评分）
- ✅ shopmind-web dashboard 更新为 6/8，layout.tsx 加入培训入口
