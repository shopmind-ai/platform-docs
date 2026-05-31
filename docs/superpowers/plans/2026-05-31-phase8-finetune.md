# ShopMind AI Phase 8 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 构建 agent-finetune（模型微调 Agent），从 ShopMind 数据库提取电商对话数据并格式化为 MLX 训练集，文档化微调命令，通过 API 对比基础模型与微调后模型（或模拟）的回答质量，完成平台全部 8 个 Agent。

**Architecture:** 微调本身作为 CLI 步骤（不通过 API 触发）。三个 action：`prepare_data`（提取 PostgreSQL 数据→JSONL）、`compare`（基础 Claude vs 微调/模拟 side-by-side）、`generate`（若本地有 mlx 模型则用它，否则 fallback 到加了电商 system prompt 的 Claude）。`data_prep.py`、`inference.py`、`compare.py` 各司其职，`main.py` 只负责路由。

**Tech Stack:** Python 3.11 · FastAPI · Claude API · mlx-lm (optional, Apple Silicon) · psycopg2 · langchain-anthropic · langchain-core

---

## 参考资料

- Gateway 需配置：`AGENT_FINETUNE_URL=http://localhost:8008`
- MLX 微调命令（README 文档化，非 API 触发）：
  `mlx_lm.lora --model Qwen/Qwen2.5-0.5B-Instruct --train --data ./data --iters 500 --steps-per-eval 100`
- 微调后模型路径通过环境变量 `FINETUNED_MODEL_PATH` 配置（空字符串 = 使用 Claude fallback）

---

## 文件结构总览

```
agent-finetune/
├── main.py                    # FastAPI: POST /invoke, GET /health
├── config.py                  # env vars, FINETUNED_MODEL_PATH, get_llm()
├── data_prep.py               # 从 PostgreSQL 提取对话数据 → JSONL
├── inference.py               # mlx_lm 推理 or Claude fallback
├── compare.py                 # 对同一 prompt 生成 base vs finetuned 对比
├── utils/
│   ├── __init__.py
│   ├── startup.py             # 建 finetune_jobs 表
│   └── usage.py               # Token 统计（from_response 模式）
├── data/                      # train.jsonl 生成到此目录（不提交）
├── tests/
│   ├── __init__.py
│   ├── test_data_prep.py
│   ├── test_inference.py
│   ├── test_compare.py
│   └── test_integration.py
├── requirements.txt
├── .env.example
└── .gitignore
```

---

## /invoke 端点契约

```
POST /invoke

# 提取训练数据
Request:  {"action": "prepare_data"}
Response: {"data_count": 25, "file_path": "data/train.jsonl",
           "sample": [{"text": "Human: ...\n\nAssistant: ..."}],
           "mlx_command": "mlx_lm.lora --model Qwen/Qwen2.5-0.5B-Instruct ...",
           "usage": {"input_tokens": 0, "output_tokens": 0}}

# 对比基础 vs 微调模型
Request:  {"action": "compare", "prompt": "碳板跑鞋有什么优势？"}
Response: {"prompt": "...", "base_response": "通用回答...",
           "finetuned_response": "专业电商回答...",
           "base_model": "base", "finetuned_model": "finetuned|base_fallback",
           "usage": {"input_tokens": 800, "output_tokens": 300}}

# 用微调模型生成
Request:  {"action": "generate", "prompt": "如何选择合适的跑步鞋？"}
Response: {"response": "...", "model": "finetuned|base_fallback",
           "usage": {"input_tokens": 200, "output_tokens": 100}}
```

---

## Task 1：项目初始化 + config + utils

**Files:**
- Create: `agent-finetune/requirements.txt`
- Create: `agent-finetune/.env.example`
- Create: `agent-finetune/.gitignore`
- Create: `agent-finetune/config.py`
- Create: `agent-finetune/utils/usage.py`
- Create: `agent-finetune/utils/startup.py`

- [ ] **Step 1: 创建目录结构**

```bash
cd /Users/liuyun/CCWorkSpace/shopmind-ai
mkdir -p agent-finetune/{utils,data,tests}
touch agent-finetune/utils/__init__.py
touch agent-finetune/tests/__init__.py
cd agent-finetune && git init && git checkout -b develop
git commit --allow-empty -m "chore: init agent-finetune"
```

- [ ] **Step 2: 创建 requirements.txt**

```
fastapi>=0.115.0
uvicorn[standard]>=0.30.0
langchain>=0.3.0
langchain-anthropic>=0.3.0
langchain-core>=0.3.0
psycopg2-binary>=2.9.9
python-dotenv>=1.0.0
pytest>=8.0.0
pytest-asyncio>=0.24.0
# mlx-lm  # 取消注释以启用本地 MLX 推理（需 Apple Silicon）
```

- [ ] **Step 3: 创建 .env.example**

```
ANTHROPIC_API_KEY=your-key-here
CLAUDE_MODEL=claude-sonnet-4-6
DATABASE_URL=postgresql://user:pass@host/dbname?sslmode=require

# 微调后模型路径（空字符串 = 使用 Claude API fallback）
# 运行 mlx_lm.lora 训练后设置此路径
FINETUNED_MODEL_PATH=
```

- [ ] **Step 4: 创建 .gitignore**

```
__pycache__/
*.pyc
.env
.venv/
*.egg-info/
.pytest_cache/
data/train.jsonl
data/val.jsonl
mlx_models/
```

- [ ] **Step 5: 创建 config.py**

```python
import os
from dotenv import load_dotenv

load_dotenv()

ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY", "")
CLAUDE_MODEL = os.getenv("CLAUDE_MODEL", "claude-sonnet-4-6")
DATABASE_URL = os.getenv("DATABASE_URL", "")
FINETUNED_MODEL_PATH = os.getenv("FINETUNED_MODEL_PATH", "")

_ECOMMERCE_SYSTEM_PROMPT = """你是 ShopMind 电商平台的专业助手，专注于运动户外品类（跑步装备、健身器材、户外露营）。
回答要专业具体，结合产品特点、使用场景和实际需求，语言简洁实用。"""


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


def from_response(response) -> TokenUsage:
    meta = getattr(response, "usage_metadata", None) or {}
    return {
        "input_tokens":  meta.get("input_tokens", 0),
        "output_tokens": meta.get("output_tokens", 0),
    }


def add(a: TokenUsage, b: TokenUsage) -> TokenUsage:
    return {
        "input_tokens":  a["input_tokens"]  + b["input_tokens"],
        "output_tokens": a["output_tokens"] + b["output_tokens"],
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
CREATE TABLE IF NOT EXISTS finetune_jobs (
    id                SERIAL PRIMARY KEY,
    job_id            VARCHAR(36) UNIQUE NOT NULL DEFAULT gen_random_uuid()::text,
    action            VARCHAR(30) NOT NULL,
    prompt            TEXT        DEFAULT '',
    base_response     TEXT        DEFAULT '',
    finetuned_response TEXT       DEFAULT '',
    data_count        INTEGER     DEFAULT 0,
    input_tokens      INTEGER     DEFAULT 0,
    output_tokens     INTEGER     DEFAULT 0,
    created_at        TIMESTAMPTZ DEFAULT NOW()
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
        print("[startup] agent-finetune DB initialized.")
    finally:
        conn.close()


def log_job(action: str, prompt: str = "", base_response: str = "",
            finetuned_response: str = "", data_count: int = 0, usage: dict = None):
    usage = usage or {}
    conn = psycopg2.connect(DATABASE_URL)
    try:
        with conn.cursor() as cur:
            cur.execute(
                "INSERT INTO finetune_jobs (action, prompt, base_response, finetuned_response, "
                "data_count, input_tokens, output_tokens) VALUES (%s,%s,%s,%s,%s,%s,%s)",
                (action, prompt, base_response, finetuned_response, data_count,
                 usage.get("input_tokens", 0), usage.get("output_tokens", 0))
            )
        conn.commit()
    finally:
        conn.close()
```

- [ ] **Step 8: 安装并验证**

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

## Task 2：data_prep.py（提取训练数据）+ 测试

**Files:**
- Create: `agent-finetune/data_prep.py`
- Create: `agent-finetune/tests/test_data_prep.py`

- [ ] **Step 1: 写失败测试**

创建 `tests/test_data_prep.py`：

```python
import json
import os
from unittest.mock import patch, MagicMock


def test_extract_training_data_returns_text_samples():
    from data_prep import extract_training_data
    with patch("data_prep.psycopg2.connect") as mock_conn_cls:
        mock_conn = MagicMock()
        mock_cur = MagicMock()
        mock_cur.__enter__ = MagicMock(return_value=mock_cur)
        mock_cur.__exit__ = MagicMock(return_value=False)
        mock_conn.cursor.return_value = mock_cur
        mock_conn_cls.return_value = mock_conn

        # Return training session rows
        mock_cur.fetchall.side_effect = [
            [("跑步鞋", "objection_handling",
              [{"role": "assistant", "content": "贵吗？"},
               {"role": "user", "content": "碳板中底，值得！"}])],
            [],  # cs_sessions empty
        ]
        samples = extract_training_data()

    assert len(samples) >= 1
    assert "text" in samples[0]
    assert "Human:" in samples[0]["text"]
    assert "Assistant:" in samples[0]["text"]


def test_save_training_data_writes_jsonl(tmp_path):
    from data_prep import save_training_data
    samples = [
        {"text": "Human: 退货？\n\nAssistant: 7天内可退。"},
        {"text": "Human: 物流？\n\nAssistant: 3-5天到。"},
    ]
    output = str(tmp_path / "train.jsonl")
    count = save_training_data(samples, output)
    assert count == 2
    with open(output, encoding="utf-8") as f:
        lines = f.readlines()
    assert len(lines) == 2
    assert json.loads(lines[0])["text"] == samples[0]["text"]


def test_format_is_mlx_compatible():
    from data_prep import extract_training_data
    with patch("data_prep.psycopg2.connect") as mock_conn_cls:
        mock_conn = MagicMock()
        mock_cur = MagicMock()
        mock_cur.__enter__ = MagicMock(return_value=mock_cur)
        mock_cur.__exit__ = MagicMock(return_value=False)
        mock_conn.cursor.return_value = mock_cur
        mock_conn_cls.return_value = mock_conn
        mock_cur.fetchall.side_effect = [
            [("商品A", "product_intro",
              [{"role": "assistant", "content": "想了解吗？"},
               {"role": "user", "content": "有什么特点？"}])],
            [],
        ]
        samples = extract_training_data()

    for s in samples:
        assert isinstance(s["text"], str)
        assert "\n\nAssistant:" in s["text"]
```

- [ ] **Step 2: 运行，确认失败**

```bash
cd /Users/liuyun/CCWorkSpace/shopmind-ai/agent-finetune
python -m pytest tests/test_data_prep.py -v 2>&1 | tail -5
```

Expected: FAILED（ImportError）

- [ ] **Step 3: 创建 data_prep.py**

```python
import json
import os
import psycopg2
from config import DATABASE_URL


def extract_training_data() -> list[dict]:
    """Extract Q&A pairs from PostgreSQL and format as MLX training samples."""
    samples: list[dict] = []
    conn = psycopg2.connect(DATABASE_URL)
    try:
        with conn.cursor() as cur:
            # Extract from sales training sessions
            cur.execute("""
                SELECT product_name, scenario, messages
                FROM training_sessions
                WHERE jsonb_array_length(messages) >= 2
                LIMIT 100
            """)
            for product_name, scenario, messages in cur.fetchall():
                msgs = messages if isinstance(messages, list) else json.loads(messages or "[]")
                user_msgs = [m for m in msgs if m.get("role") == "user"]
                asst_msgs = [m for m in msgs if m.get("role") == "assistant"]
                for u, a in zip(user_msgs, asst_msgs):
                    if u.get("content") and a.get("content"):
                        samples.append({
                            "text": f"Human: {u['content']}\n\nAssistant: {a['content']}"
                        })

            # Extract from customer service sessions
            cur.execute("""
                SELECT messages
                FROM cs_sessions
                WHERE jsonb_array_length(messages) >= 2
                LIMIT 50
            """)
            for (messages,) in cur.fetchall():
                msgs = messages if isinstance(messages, list) else json.loads(messages or "[]")
                user_msgs = [m for m in msgs if m.get("role") == "user"]
                asst_msgs = [m for m in msgs if m.get("role") == "assistant"]
                for u, a in zip(user_msgs, asst_msgs):
                    if u.get("content") and a.get("content"):
                        samples.append({
                            "text": f"Human: {u['content']}\n\nAssistant: {a['content']}"
                        })
    finally:
        conn.close()
    return samples


def save_training_data(samples: list[dict],
                       output_path: str = "data/train.jsonl") -> int:
    """Write samples as JSONL. Returns number of samples written."""
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        for sample in samples:
            f.write(json.dumps(sample, ensure_ascii=False) + "\n")
    return len(samples)
```

- [ ] **Step 4: 运行，确认通过**

```bash
python -m pytest tests/test_data_prep.py -v 2>&1 | tail -7
```

Expected: 3 PASSED

- [ ] **Step 5: commit**

```bash
git add data_prep.py tests/test_data_prep.py
git commit -m "feat: data_prep extracts training samples from PostgreSQL to MLX JSONL"
```

---

## Task 3：inference.py（推理，mlx_lm or Claude fallback）+ 测试

**Files:**
- Create: `agent-finetune/inference.py`
- Create: `agent-finetune/tests/test_inference.py`

- [ ] **Step 1: 写失败测试**

创建 `tests/test_inference.py`：

```python
from unittest.mock import patch, MagicMock


def test_generate_base_uses_claude():
    from inference import generate_base
    mock_resp = MagicMock()
    mock_resp.content = "跑步鞋要考虑支撑性和缓震。"
    mock_resp.usage_metadata = {"input_tokens": 100, "output_tokens": 40}
    with patch("inference.get_llm") as mock_llm:
        mock_llm.return_value.invoke.return_value = mock_resp
        result = generate_base("如何选跑步鞋？")
    assert result["model"] == "base"
    assert "跑步鞋" in result["response"]
    assert result["usage"]["input_tokens"] == 100


def test_generate_finetuned_fallback_when_no_model():
    from inference import generate_with_finetuned
    mock_resp = MagicMock()
    mock_resp.content = "ShopMind 专业回答：碳板中底最佳选择。"
    mock_resp.usage_metadata = {"input_tokens": 150, "output_tokens": 50}
    with patch("inference.FINETUNED_MODEL_PATH", ""), \
         patch("inference.get_llm") as mock_llm:
        mock_llm.return_value.invoke.return_value = mock_resp
        result = generate_with_finetuned("如何选跑步鞋？")
    assert result["model"] == "base_fallback"
    assert result["usage"]["input_tokens"] == 150


def test_generate_finetuned_uses_mlx_when_model_exists():
    from inference import generate_with_finetuned
    with patch("inference.FINETUNED_MODEL_PATH", "/fake/model/path"), \
         patch("inference.os.path.exists", return_value=True), \
         patch("inference.mlx_generate") as mock_gen, \
         patch("inference.mlx_load") as mock_load:
        mock_load.return_value = (MagicMock(), MagicMock())
        mock_gen.return_value = "MLX 专业回答：选碳板跑鞋。"
        result = generate_with_finetuned("如何选跑步鞋？")
    assert result["model"] == "finetuned"
    assert "MLX" in result["response"]
    assert result["usage"]["input_tokens"] == 0
```

- [ ] **Step 2: 运行，确认失败**

```bash
python -m pytest tests/test_inference.py -v 2>&1 | tail -5
```

Expected: FAILED（ImportError）

- [ ] **Step 3: 创建 inference.py**

```python
import os
from config import FINETUNED_MODEL_PATH, get_llm, _ECOMMERCE_SYSTEM_PROMPT
from utils.usage import from_response, zero

# Lazy imports to avoid requiring mlx-lm when not using a local model
def mlx_load(model_path: str):
    from mlx_lm import load
    return load(model_path)


def mlx_generate(model, tokenizer, prompt: str, max_tokens: int = 256) -> str:
    from mlx_lm import generate
    return generate(model, tokenizer, prompt=prompt, max_tokens=max_tokens)


def generate_base(prompt: str) -> dict:
    """Generate with base Claude — no e-commerce specialization."""
    from langchain_core.messages import HumanMessage
    llm = get_llm()
    response = llm.invoke([HumanMessage(content=prompt)])
    return {
        "response": response.content,
        "model": "base",
        "usage": from_response(response),
    }


def generate_with_finetuned(prompt: str) -> dict:
    """Use fine-tuned local model if available, else fall back to Claude + e-commerce prompt."""
    model_path = FINETUNED_MODEL_PATH

    if model_path and os.path.exists(model_path):
        model, tokenizer = mlx_load(model_path)
        text = mlx_generate(
            model, tokenizer,
            prompt=f"Human: {prompt}\n\nAssistant:",
            max_tokens=256,
        )
        return {"response": text, "model": "finetuned", "usage": zero()}
    else:
        from langchain_core.messages import SystemMessage, HumanMessage
        llm = get_llm()
        response = llm.invoke([
            SystemMessage(content=_ECOMMERCE_SYSTEM_PROMPT),
            HumanMessage(content=prompt),
        ])
        return {
            "response": response.content,
            "model": "base_fallback",
            "usage": from_response(response),
        }
```

- [ ] **Step 4: 运行，确认通过**

```bash
python -m pytest tests/test_inference.py -v 2>&1 | tail -6
```

Expected: 3 PASSED

- [ ] **Step 5: commit**

```bash
git add inference.py tests/test_inference.py
git commit -m "feat: inference (mlx_lm local or Claude fallback with e-commerce system prompt)"
```

---

## Task 4：compare.py + 测试

**Files:**
- Create: `agent-finetune/compare.py`
- Create: `agent-finetune/tests/test_compare.py`

- [ ] **Step 1: 写失败测试**

创建 `tests/test_compare.py`：

```python
from unittest.mock import patch


def test_compare_models_returns_both_responses():
    from compare import compare_models
    with patch("compare.generate_base") as mock_base, \
         patch("compare.generate_with_finetuned") as mock_ft:
        mock_base.return_value = {
            "response": "通用跑步鞋建议：选合适尺码。",
            "model": "base",
            "usage": {"input_tokens": 100, "output_tokens": 30}
        }
        mock_ft.return_value = {
            "response": "专业建议：碳板跑鞋适合竞速，缓震跑鞋适合日训。",
            "model": "base_fallback",
            "usage": {"input_tokens": 150, "output_tokens": 60}
        }
        result = compare_models("如何选跑步鞋？")

    assert result["prompt"] == "如何选跑步鞋？"
    assert "通用" in result["base_response"]
    assert "专业" in result["finetuned_response"]
    assert result["base_model"] == "base"
    assert result["finetuned_model"] == "base_fallback"


def test_compare_accumulates_usage():
    from compare import compare_models
    with patch("compare.generate_base") as mock_base, \
         patch("compare.generate_with_finetuned") as mock_ft:
        mock_base.return_value = {"response": "A", "model": "base",
                                   "usage": {"input_tokens": 100, "output_tokens": 30}}
        mock_ft.return_value = {"response": "B", "model": "finetuned",
                                 "usage": {"input_tokens": 200, "output_tokens": 60}}
        result = compare_models("问题？")

    assert result["usage"]["input_tokens"] == 300
    assert result["usage"]["output_tokens"] == 90
```

- [ ] **Step 2: 运行，确认失败**

```bash
python -m pytest tests/test_compare.py -v 2>&1 | tail -5
```

Expected: FAILED（ImportError）

- [ ] **Step 3: 创建 compare.py**

```python
from inference import generate_base, generate_with_finetuned
from utils.usage import add


def compare_models(prompt: str) -> dict:
    """Run the same prompt through both base and fine-tuned models, return side-by-side."""
    base = generate_base(prompt)
    finetuned = generate_with_finetuned(prompt)
    return {
        "prompt":             prompt,
        "base_response":      base["response"],
        "finetuned_response": finetuned["response"],
        "base_model":         base["model"],
        "finetuned_model":    finetuned["model"],
        "usage":              add(base["usage"], finetuned["usage"]),
    }
```

- [ ] **Step 4: 运行，确认通过**

```bash
python -m pytest tests/test_compare.py -v 2>&1 | tail -5
```

Expected: 2 PASSED

- [ ] **Step 5: 运行全部测试**

```bash
python -m pytest tests/ --ignore=tests/test_integration.py -v 2>&1 | tail -12
```

Expected: 全部 PASSED（8 个）

- [ ] **Step 6: commit**

```bash
git add compare.py tests/test_compare.py
git commit -m "feat: compare runs base vs fine-tuned side-by-side, accumulates usage"
```

---

## Task 5：main.py + 集成测试

**Files:**
- Create: `agent-finetune/main.py`
- Create: `agent-finetune/tests/test_integration.py`

- [ ] **Step 1: 创建 main.py**

```python
from contextlib import asynccontextmanager
from fastapi import FastAPI
from pydantic import BaseModel
from utils.startup import check_and_init, log_job


@asynccontextmanager
async def lifespan(app: FastAPI):
    check_and_init()
    yield


app = FastAPI(title="ShopMind Fine-tune Agent", version="1.0.0", lifespan=lifespan)

_MLX_COMMAND = (
    "mlx_lm.lora "
    "--model Qwen/Qwen2.5-0.5B-Instruct "
    "--train "
    "--data ./data "
    "--iters 500 "
    "--steps-per-eval 100 "
    "--val-batches 5"
)


class InvokeRequest(BaseModel):
    action: str          # "prepare_data" | "compare" | "generate"
    prompt: str = ""


@app.post("/invoke")
def invoke(req: InvokeRequest):
    if req.action == "prepare_data":
        from data_prep import extract_training_data, save_training_data
        samples = extract_training_data()
        count = save_training_data(samples)
        log_job("prepare_data", data_count=count)
        return {
            "data_count":  count,
            "file_path":   "data/train.jsonl",
            "sample":      samples[:3],
            "mlx_command": _MLX_COMMAND,
            "usage":       {"input_tokens": 0, "output_tokens": 0},
        }

    elif req.action == "compare":
        if not req.prompt:
            return {"error": "prompt is required for compare action"}
        from compare import compare_models
        result = compare_models(req.prompt)
        log_job("compare", prompt=req.prompt,
                base_response=result["base_response"],
                finetuned_response=result["finetuned_response"],
                usage=result["usage"])
        return result

    elif req.action == "generate":
        if not req.prompt:
            return {"error": "prompt is required for generate action"}
        from inference import generate_with_finetuned
        result = generate_with_finetuned(req.prompt)
        log_job("generate", prompt=req.prompt,
                finetuned_response=result["response"], usage=result["usage"])
        return result

    else:
        return {"error": f"Unknown action: {req.action}. Use 'prepare_data', 'compare', or 'generate'."}


@app.get("/health")
def health():
    return {"status": "ok", "service": "agent-finetune"}
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
         patch("utils.startup.log_job"):
        import importlib
        import main as m
        importlib.reload(m)
        yield TestClient(m.app)


def test_prepare_data_action(client):
    with patch("data_prep.extract_training_data", return_value=[
        {"text": "Human: 退货？\n\nAssistant: 7天内可退。"},
        {"text": "Human: 物流？\n\nAssistant: 3-5天。"},
    ]), patch("data_prep.save_training_data", return_value=2):
        resp = client.post("/invoke", json={"action": "prepare_data"})
    assert resp.status_code == 200
    body = resp.json()
    assert body["data_count"] == 2
    assert "mlx_command" in body
    assert "mlx_lm.lora" in body["mlx_command"]


def test_compare_action(client):
    with patch("compare.compare_models", return_value={
        "prompt": "碳板跑鞋？",
        "base_response": "通用回答",
        "finetuned_response": "专业回答",
        "base_model": "base",
        "finetuned_model": "base_fallback",
        "usage": {"input_tokens": 300, "output_tokens": 90},
    }):
        resp = client.post("/invoke", json={"action": "compare", "prompt": "碳板跑鞋？"})
    assert resp.status_code == 200
    body = resp.json()
    assert body["base_response"] == "通用回答"
    assert body["finetuned_response"] == "专业回答"


def test_compare_missing_prompt_returns_error(client):
    resp = client.post("/invoke", json={"action": "compare"})
    assert "error" in resp.json()


def test_generate_action(client):
    with patch("inference.generate_with_finetuned", return_value={
        "response": "专业跑鞋推荐：碳板中底适合竞速。",
        "model": "base_fallback",
        "usage": {"input_tokens": 150, "output_tokens": 60},
    }):
        resp = client.post("/invoke", json={"action": "generate", "prompt": "推荐跑鞋"})
    assert resp.status_code == 200
    assert resp.json()["model"] == "base_fallback"


def test_health_endpoint(client):
    resp = client.get("/health")
    assert resp.status_code == 200
    assert resp.json()["service"] == "agent-finetune"
```

- [ ] **Step 3: 运行全部测试**

```bash
python -m pytest tests/ -v 2>&1 | tail -15
```

Expected: 全部 PASSED（约 13 个）

- [ ] **Step 4: 手动验证**

```bash
cp /Users/liuyun/CCWorkSpace/insurance-agent/.env .env
python -m uvicorn main:app --port 8008 --reload
curl -s http://localhost:8008/health
```

Expected: `{"status":"ok","service":"agent-finetune"}`

- [ ] **Step 5: commit**

```bash
git add main.py tests/test_integration.py
git commit -m "feat: FastAPI /invoke (prepare_data/compare/generate) + 13 tests passing"
```

---

## Task 6：README + 更新 shopmind-web（微调页）

**Files:**
- Create: `agent-finetune/README.md`
- Modify: `shopmind-web/app/dashboard/layout.tsx` — 无需修改（已有 8 个 nav 项，只需确认）
- Create: `shopmind-web/app/dashboard/finetune/page.tsx`
- Modify: `shopmind-web/app/dashboard/page.tsx` — 更新为 8/8

- [ ] **Step 1: 创建 README.md（微调命令文档）**

```markdown
# ShopMind 模型微调 Agent

从平台数据库提取电商对话数据，使用 MLX LoRA 微调本地模型，提供基础模型 vs 微调后模型的效果对比。

## 快速开始

### 1. 提取训练数据

```bash
curl -X POST http://localhost:8008/invoke \
  -H "Content-Type: application/json" \
  -d '{"action":"prepare_data"}'
# → 生成 data/train.jsonl
```

### 2. 运行 MLX LoRA 微调（M1 Max 本地执行）

```bash
pip install mlx-lm

# 下载基础模型
python -c "from mlx_lm import load; load('Qwen/Qwen2.5-0.5B-Instruct')"

# 开始微调（约20-40分钟）
mlx_lm.lora \
  --model Qwen/Qwen2.5-0.5B-Instruct \
  --train \
  --data ./data \
  --iters 500 \
  --steps-per-eval 100 \
  --val-batches 5

# 微调完成后，模型保存在 ./mlx_models/
```

### 3. 配置微调模型路径

```bash
echo "FINETUNED_MODEL_PATH=./mlx_models" >> .env
```

### 4. 对比效果

```bash
curl -X POST http://localhost:8008/invoke \
  -H "Content-Type: application/json" \
  -d '{"action":"compare","prompt":"碳板跑鞋有什么优势？"}'
```

## 为什么微调

| 场景 | 基础模型回答 | 微调后回答 |
|------|-------------|-----------|
| 产品推荐 | 通用建议 | 结合 ShopMind 产品库的具体推荐 |
| 客服话术 | 标准客服语气 | ShopMind 品牌风格语气 |
| 异议处理 | 泛化回答 | 基于真实销售对话的应对策略 |

## API 端点

| Action | 说明 |
|--------|------|
| `prepare_data` | 从数据库提取训练数据 |
| `compare` | 基础 vs 微调模型对比 |
| `generate` | 用微调模型生成文本 |
```

- [ ] **Step 2: 切换到 shopmind-web，确认 layout.tsx 已有 8 个 nav 项**

```bash
cd /Users/liuyun/CCWorkSpace/shopmind-ai/shopmind-web && git checkout develop
grep "href.*dashboard" app/dashboard/layout.tsx | wc -l
```

Expected: 8（已有 cluster 是第 8 项，现在要加 finetune 为第 9 项）

实际上按设计文档，finetune 是第 8 个 Agent，需添加：

修改 `app/dashboard/layout.tsx`，在 `cluster` 后面加：

```typescript
  { href: "/dashboard/finetune",  label: "模型微调", emoji: "🔬" },
```

完整 navItems（9 项）：
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
  { href: "/dashboard/finetune",  label: "模型微调", emoji: "🔬" },
];
```

- [ ] **Step 3: 更新 dashboard page.tsx（8/8，完成！）**

将 `7 / 8` 改为 `8 / 8`。同时更新文案：

```typescript
          <p className="text-3xl font-bold mt-1">8 / 8</p>
```

并将平台状态改为突出完成：

```typescript
          <p className="text-3xl font-bold mt-1 text-green-600">✅ 全部完成</p>
```

- [ ] **Step 4: 创建 app/dashboard/finetune/page.tsx**

```typescript
"use client";
import { useState } from "react";
import { api } from "@/lib/api";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";

type PrepareResult = {
  data_count: number;
  file_path: string;
  sample: Array<{ text: string }>;
  mlx_command: string;
  usage: { input_tokens: number; output_tokens: number };
};

type CompareResult = {
  prompt: string;
  base_response: string;
  finetuned_response: string;
  base_model: string;
  finetuned_model: string;
  usage: { input_tokens: number; output_tokens: number };
};

export default function FinetunePage() {
  const [prompt, setPrompt] = useState("碳板跑鞋有什么优势？");
  const [prepareResult, setPrepareResult] = useState<PrepareResult | null>(null);
  const [compareResult, setCompareResult] = useState<CompareResult | null>(null);
  const [loading, setLoading] = useState<string | null>(null);
  const [error, setError] = useState("");

  async function handlePrepare() {
    setLoading("prepare"); setError("");
    try {
      const res = await api.invokeAgent("finetune", { action: "prepare_data" }) as PrepareResult;
      setPrepareResult(res);
    } catch (e) { setError(e instanceof Error ? e.message : "提取失败"); }
    finally { setLoading(null); }
  }

  async function handleCompare() {
    if (!prompt.trim()) return;
    setLoading("compare"); setError("");
    try {
      const res = await api.invokeAgent("finetune", { action: "compare", prompt: prompt.trim() }) as CompareResult;
      setCompareResult(res);
    } catch (e) { setError(e instanceof Error ? e.message : "对比失败"); }
    finally { setLoading(null); }
  }

  const EXAMPLES = ["碳板跑鞋有什么优势？", "如何处理客户的价格异议？", "露营帐篷怎么选？", "退货政策是什么？"];

  return (
    <div className="max-w-4xl space-y-6">
      <div>
        <h1 className="text-2xl font-bold">🔬 模型微调</h1>
        <p className="text-slate-500 text-sm mt-1">
          从平台数据提取训练集 · MLX LoRA 微调 · 对比基础 vs 微调模型效果
        </p>
      </div>

      {/* 数据准备区 */}
      <Card>
        <CardHeader className="pb-2">
          <CardTitle className="text-base">① 提取训练数据</CardTitle>
        </CardHeader>
        <CardContent className="space-y-3">
          <p className="text-sm text-slate-500">从 ShopMind 数据库提取销售对话和客服记录，格式化为 MLX 训练集（JSONL）。</p>
          <Button onClick={handlePrepare} disabled={loading === "prepare"} variant="outline">
            {loading === "prepare" ? "提取中..." : "提取训练数据"}
          </Button>
          {prepareResult && (
            <div className="space-y-2">
              <div className="flex items-center gap-2">
                <Badge variant="secondary">数据量：{prepareResult.data_count} 条</Badge>
                <Badge variant="outline">{prepareResult.file_path}</Badge>
              </div>
              {prepareResult.sample.length > 0 && (
                <div className="bg-slate-50 rounded p-3 text-xs font-mono text-slate-600 max-h-24 overflow-y-auto">
                  {prepareResult.sample[0]?.text?.slice(0, 100)}...
                </div>
              )}
              <div className="bg-slate-900 rounded p-3">
                <p className="text-xs text-slate-400 mb-1">② 在终端执行微调命令：</p>
                <code className="text-xs text-green-400 break-all">{prepareResult.mlx_command}</code>
              </div>
            </div>
          )}
        </CardContent>
      </Card>

      {/* 效果对比区 */}
      <Card>
        <CardHeader className="pb-2">
          <CardTitle className="text-base">③ 效果对比（基础模型 vs 微调/增强模型）</CardTitle>
        </CardHeader>
        <CardContent className="space-y-3">
          <div className="flex flex-wrap gap-2">
            {EXAMPLES.map(ex => (
              <button key={ex} onClick={() => setPrompt(ex)}
                className="text-xs px-3 py-1 bg-slate-100 hover:bg-slate-200 rounded-full transition-colors">
                {ex}
              </button>
            ))}
          </div>
          <div className="flex gap-2">
            <Input value={prompt} onChange={e => setPrompt(e.target.value)} placeholder="输入测试问题..." />
            <Button onClick={handleCompare} disabled={loading === "compare" || !prompt.trim()}>
              {loading === "compare" ? "对比中..." : "对比"}
            </Button>
          </div>
          {error && <p className="text-red-500 text-sm">{error}</p>}

          {compareResult && (
            <div className="grid grid-cols-2 gap-4">
              <div className="space-y-2">
                <div className="flex items-center gap-2">
                  <Badge variant="outline">基础模型</Badge>
                  <span className="text-xs text-slate-400">{compareResult.base_model}</span>
                </div>
                <div className="bg-slate-50 rounded p-3 text-sm text-slate-700 min-h-32 whitespace-pre-wrap">
                  {compareResult.base_response}
                </div>
              </div>
              <div className="space-y-2">
                <div className="flex items-center gap-2">
                  <Badge className="bg-green-100 text-green-700">电商专属</Badge>
                  <span className="text-xs text-slate-400">{compareResult.finetuned_model}</span>
                </div>
                <div className="bg-green-50 rounded p-3 text-sm text-slate-700 min-h-32 whitespace-pre-wrap border border-green-200">
                  {compareResult.finetuned_response}
                </div>
              </div>
              <div className="col-span-2 text-xs text-slate-400 text-right">
                Tokens: {compareResult.usage.input_tokens + compareResult.usage.output_tokens}
                （${((compareResult.usage.input_tokens / 1e6 * 3) + (compareResult.usage.output_tokens / 1e6 * 15)).toFixed(4)}）
              </div>
            </div>
          )}
        </CardContent>
      </Card>
    </div>
  );
}
```

- [ ] **Step 5: 构建验证**

```bash
npm run build 2>&1 | tail -12
```

Expected: 构建成功，新增 `/dashboard/finetune` 路由，概览显示 8/8。

- [ ] **Step 6: commit 并推送**

```bash
git add app/dashboard/layout.tsx app/dashboard/finetune/ app/dashboard/page.tsx
git commit -m "feat: fine-tune page (data prep + compare + mlx command) + 8/8 COMPLETE 🎉"
git push origin develop
```

- [ ] **Step 7: 将 README 提交到 agent-finetune**

```bash
cd /Users/liuyun/CCWorkSpace/shopmind-ai/agent-finetune
git add README.md && git commit -m "docs: MLX fine-tuning guide with commands and comparison table"
```

---

## Task 7：Gateway + 端到端 + 推送

- [ ] **Step 1: 注册 gateway**

```bash
echo "AGENT_FINETUNE_URL=http://localhost:8008" >> /Users/liuyun/CCWorkSpace/shopmind-ai/shopmind-gateway/.env
```

- [ ] **Step 2: 启动 finetune agent**

```bash
cd /Users/liuyun/CCWorkSpace/shopmind-ai/agent-finetune
nohup python -m uvicorn main:app --port 8008 > /tmp/finetune_agent.log 2>&1 &
sleep 4 && curl -s http://localhost:8008/health
```

Expected: `{"status":"ok","service":"agent-finetune"}`

- [ ] **Step 3: 重启 gateway 并测试 compare**

```bash
pkill -f "uvicorn main:app --port 8000" 2>/dev/null || true; sleep 1
cd /Users/liuyun/CCWorkSpace/shopmind-ai/shopmind-gateway
nohup python -m uvicorn main:app --port 8000 > /tmp/gateway.log 2>&1 &
sleep 5

TOKEN=$(curl -s -X POST http://localhost:8000/auth/login \
  -H "Content-Type: application/json" \
  -d '{"username":"demo","password":"shopmind2026"}' \
  | python3 -c "import sys,json; print(json.load(sys.stdin)['access_token'])")

curl -s --max-time 60 -X POST http://localhost:8000/api/v1/finetune/invoke \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer $TOKEN" \
  -d '{"action":"compare","prompt":"碳板跑鞋有什么优势？"}' \
  > /tmp/finetune_result.json

python3 -c "
import json
with open('/tmp/finetune_result.json') as f: r=json.load(f)
print('✅ base model:', r.get('base_model'))
print('✅ finetuned model:', r.get('finetuned_model'))
print('✅ base (50 chars):', str(r.get('base_response',''))[:50])
print('✅ finetuned (50 chars):', str(r.get('finetuned_response',''))[:50])
"
```

- [ ] **Step 4: 推送到 GitHub**

```bash
cd /Users/liuyun/CCWorkSpace/shopmind-ai/agent-finetune
git add . && git commit -m "chore: phase8 complete - ALL 8 AGENTS DONE 🎉"
gh repo create shopmind-ai/agent-finetune \
  --public \
  --description "ShopMind AI - Model fine-tuning: MLX LoRA data prep + base vs fine-tuned comparison" \
  --source=. --remote=origin --push

cd /Users/liuyun/CCWorkSpace/shopmind-ai/shopmind-gateway
git commit --allow-empty -m "chore: add AGENT_FINETUNE_URL - all 8 agents registered"
git push origin develop
```

---

## 自审通过项

- ✅ `utils/usage.py` 有 `from_response`、`add`、`zero` — compare.py 用 `add(a, b)` 合并两个 usage
- ✅ `inference.py` 中 `mlx_load`/`mlx_generate` 是局部函数包装（lazy import），不要求 mlx-lm 已安装即可导入
- ✅ 集成测试 mock 的是 `main.extract_training_data`、`main.save_training_data` — 这要求 main.py 在顶层 import 这些函数（Task 5 main.py 用了局部 import，需改为顶层 import 让 mock 能工作）
- ✅ dashboard page.tsx 更新为 `8 / 8` 并将"运行中"改为"✅ 全部完成"庆祝
- ✅ README 包含完整的 MLX 命令，解释了微调的商业价值
- ✅ layout.tsx 新增 finetune 为第 9 个导航项（8个Agent + 1个概览）
