# ShopMind AI Phase 3 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 构建 agent-media-library（多模态素材库 Agent），支持商品图片 URL 上传 → Claude Vision 自动打标签/生成描述/分类 → pgvector 语义索引，以及文本语义搜索返回匹配素材。

**Architecture:** 不使用 LangGraph（无状态流，操作独立），两个核心模块：`tagger.py`（Claude Vision 分析图片）和 `indexer.py`（pgvector 存储 + 语义检索）。元数据存入 `media_assets` PostgreSQL 表，文本 embedding 存入 pgvector（collection: `media_assets_embeddings`），搜索时先查 pgvector 得到 asset_id，再从 `media_assets` 取完整数据。

**Tech Stack:** Python 3.11 · FastAPI · Claude Vision API (claude-sonnet-4-6) · langchain-anthropic · langchain-postgres · pgvector · sentence-transformers · psycopg2

---

## 参考资料

- 设计文档：`shopmind-ai/docs/superpowers/specs/2026-05-31-shopmind-ai-design.md`
- 复用模式来源：`/Users/liuyun/CCWorkSpace/shopmind-ai/agent-customer-service/`
- Gateway 已配置：`AGENT_MEDIA_URL=http://localhost:8003`（需在 gateway `.env` 中新增）

---

## 文件结构总览

```
agent-media-library/
├── main.py                    # FastAPI: POST /invoke, GET /health
├── config.py                  # env vars, get_llm(), get_embeddings()
├── tagger.py                  # Claude Vision 图片分析 → tags + description + category
├── indexer.py                 # pgvector 存储 + 语义检索
├── utils/
│   ├── __init__.py
│   ├── startup.py             # 建 media_assets 表 + pgvector collection 初始化
│   └── usage.py               # token 统计（复用 Phase 1/2 模式）
├── data/
│   └── seed_assets.py         # 写入 5 张演示商品图片（预标注，跳过 Vision API）
├── tests/
│   ├── __init__.py
│   ├── test_tagger.py         # mock Claude Vision，验证标签提取逻辑
│   ├── test_indexer.py        # mock pgvector，验证存储 + 检索接口
│   └── test_integration.py    # 验证 /invoke 端点 upload + search action
├── requirements.txt
├── .env.example
└── .gitignore
```

---

## /invoke 端点契约

```
POST /invoke

# 上传图片
{
  "action": "upload",
  "image_url": "https://example.com/product.jpg",
  "product_hint": "跑步鞋"   # 可选，帮助 Claude 更准确分类
}

Response:
{
  "asset_id": "uuid",
  "image_url": "...",
  "category": "跑步装备",
  "tags": ["红色", "轻量", "碳纤维"],
  "description": "专业马拉松跑步鞋，碳纤维中底...",
  "usage": {"input_tokens": 512, "output_tokens": 80}
}

# 语义搜索
{
  "action": "search",
  "query": "红色跑鞋",
  "limit": 6,
  "category": "跑步装备"    # 可选过滤
}

Response:
{
  "results": [
    {
      "asset_id": "...",
      "image_url": "...",
      "category": "跑步装备",
      "tags": ["红色", "跑鞋"],
      "description": "...",
      "score": 0.93
    }
  ],
  "usage": {"input_tokens": 0, "output_tokens": 0}
}
```

---

## Task 1：项目初始化

**Files:**
- Create: `agent-media-library/requirements.txt`
- Create: `agent-media-library/.env.example`
- Create: `agent-media-library/.gitignore`
- Create: `agent-media-library/config.py`

- [ ] **Step 1: 创建目录结构**

```bash
cd /Users/liuyun/CCWorkSpace/shopmind-ai
mkdir -p agent-media-library/{utils,data,tests}
touch agent-media-library/utils/__init__.py
touch agent-media-library/tests/__init__.py
cd agent-media-library && git init && git checkout -b develop
git commit --allow-empty -m "chore: init agent-media-library"
```

- [ ] **Step 2: 创建 requirements.txt**

```
fastapi>=0.115.0
uvicorn[standard]>=0.30.0
langchain>=0.3.0
langchain-anthropic>=0.3.0
langchain-postgres>=0.0.13
langchain-huggingface>=0.1.0
langchain-community>=0.3.0
langchain-text-splitters>=0.3.0
sentence-transformers>=3.0.0
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

VECTOR_COLLECTION = "media_assets_embeddings"


def get_llm():
    from langchain_anthropic import ChatAnthropic
    return ChatAnthropic(model=CLAUDE_MODEL, api_key=ANTHROPIC_API_KEY)


def get_embeddings():
    from langchain_huggingface import HuggingFaceEmbeddings
    return HuggingFaceEmbeddings(
        model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True},
    )
```

- [ ] **Step 6: 安装依赖并验证**

```bash
pip install -r requirements.txt -q
python -c "import fastapi, langchain_anthropic, langchain_postgres; print('imports OK')"
```

Expected: `imports OK`

- [ ] **Step 7: commit**

```bash
git add . && git commit -m "feat: project setup, config, requirements"
```

---

## Task 2：utils/startup.py + utils/usage.py

**Files:**
- Create: `utils/startup.py`
- Create: `utils/usage.py`

- [ ] **Step 1: 创建 utils/usage.py**

```python
from typing import TypedDict

_INPUT_COST_PER_M = 3.0
_OUTPUT_COST_PER_M = 15.0


class TokenUsage(TypedDict):
    input_tokens: int
    output_tokens: int


def accumulate(prev: TokenUsage, response) -> TokenUsage:
    meta = getattr(response, "usage_metadata", None) or {}
    return {
        "input_tokens":  prev["input_tokens"]  + meta.get("input_tokens", 0),
        "output_tokens": prev["output_tokens"] + meta.get("output_tokens", 0),
    }


def cost_usd(usage: TokenUsage) -> float:
    return (
        usage["input_tokens"]  / 1_000_000 * _INPUT_COST_PER_M +
        usage["output_tokens"] / 1_000_000 * _OUTPUT_COST_PER_M
    )


def zero() -> TokenUsage:
    return {"input_tokens": 0, "output_tokens": 0}
```

- [ ] **Step 2: 创建 utils/startup.py**

```python
import psycopg2
from config import DATABASE_URL, VECTOR_COLLECTION

_DDL = """
CREATE TABLE IF NOT EXISTS media_assets (
    id           SERIAL PRIMARY KEY,
    asset_id     VARCHAR(36) UNIQUE NOT NULL DEFAULT gen_random_uuid()::text,
    image_url    TEXT        NOT NULL,
    category     VARCHAR(50),
    tags         TEXT[]      NOT NULL DEFAULT '{}',
    description  TEXT,
    input_tokens  INTEGER    DEFAULT 0,
    output_tokens INTEGER    DEFAULT 0,
    created_at   TIMESTAMPTZ DEFAULT NOW()
);
"""


def _table_exists(conn) -> bool:
    with conn.cursor() as cur:
        cur.execute(
            "SELECT EXISTS(SELECT 1 FROM information_schema.tables "
            "WHERE table_schema='public' AND table_name='media_assets')"
        )
        return cur.fetchone()[0]


def _vectors_exist(conn) -> bool:
    with conn.cursor() as cur:
        cur.execute(
            "SELECT EXISTS(SELECT 1 FROM information_schema.tables "
            "WHERE table_schema='public' AND table_name='langchain_pg_collection')"
        )
        if not cur.fetchone()[0]:
            return False
        cur.execute(
            "SELECT COUNT(*) FROM langchain_pg_embedding e "
            "JOIN langchain_pg_collection c ON e.collection_id = c.uuid "
            "WHERE c.name = %s",
            (VECTOR_COLLECTION,),
        )
        return cur.fetchone()[0] > 0


def check_and_init():
    if not DATABASE_URL:
        raise RuntimeError("DATABASE_URL is not set.")

    conn = psycopg2.connect(DATABASE_URL)
    try:
        needs_table = not _table_exists(conn)
        needs_seed = not _vectors_exist(conn)
    finally:
        conn.close()

    if needs_table:
        conn2 = psycopg2.connect(DATABASE_URL)
        try:
            with conn2.cursor() as cur:
                cur.execute(_DDL)
            conn2.commit()
            print("[startup] media_assets table created.")
        finally:
            conn2.close()

    if needs_seed:
        print("[startup] Seeding sample assets...")
        from data.seed_assets import seed
        seed()

    if not needs_table and not needs_seed:
        print("[startup] Media library already initialized, skipping.")
```

- [ ] **Step 3: commit**

```bash
git add utils/
git commit -m "feat: startup auto-init (media_assets table) + token usage utils"
```

---

## Task 3：tagger.py（Claude Vision 图片分析）

**Files:**
- Create: `tagger.py`
- Create: `tests/test_tagger.py`

- [ ] **Step 1: 写失败测试**

创建 `tests/test_tagger.py`：

```python
from unittest.mock import patch, MagicMock
import json


def test_tag_image_returns_structured_result():
    from tagger import tag_image
    mock_resp = MagicMock()
    mock_resp.content = json.dumps({
        "category": "跑步装备",
        "tags": ["红色", "轻量", "碳纤维中底"],
        "description": "一双专业马拉松跑步鞋，红色配色，碳纤维中底，适合长距离路跑。",
        "colors": ["红色", "白色"]
    })
    mock_resp.usage_metadata = {"input_tokens": 512, "output_tokens": 80}

    with patch("tagger.get_llm") as mock_llm:
        mock_llm.return_value.invoke.return_value = mock_resp
        result = tag_image("https://example.com/shoe.jpg")

    assert result["category"] == "跑步装备"
    assert "红色" in result["tags"]
    assert len(result["description"]) > 0
    assert result["usage"]["input_tokens"] == 512


def test_tag_image_with_hint():
    from tagger import tag_image
    mock_resp = MagicMock()
    mock_resp.content = json.dumps({
        "category": "健身器材",
        "tags": ["哑铃", "可调节", "铸铁"],
        "description": "可调节哑铃套装，5-30kg，铸铁材质。",
        "colors": ["黑色", "银色"]
    })
    mock_resp.usage_metadata = {"input_tokens": 520, "output_tokens": 70}

    with patch("tagger.get_llm") as mock_llm:
        mock_llm.return_value.invoke.return_value = mock_resp
        result = tag_image("https://example.com/dumbbell.jpg", product_hint="哑铃")

    assert result["category"] == "健身器材"
    assert result["usage"]["output_tokens"] == 70


def test_tag_image_handles_invalid_json_gracefully():
    from tagger import tag_image
    mock_resp = MagicMock()
    mock_resp.content = "This is not JSON"
    mock_resp.usage_metadata = {"input_tokens": 100, "output_tokens": 10}

    with patch("tagger.get_llm") as mock_llm:
        mock_llm.return_value.invoke.return_value = mock_resp
        result = tag_image("https://example.com/img.jpg")

    assert result["category"] == "其他"
    assert result["tags"] == []
    assert result["description"] == "This is not JSON"


def test_tag_image_handles_partial_json():
    from tagger import tag_image
    mock_resp = MagicMock()
    mock_resp.content = json.dumps({"category": "户外露营"})  # missing tags/description
    mock_resp.usage_metadata = {"input_tokens": 200, "output_tokens": 20}

    with patch("tagger.get_llm") as mock_llm:
        mock_llm.return_value.invoke.return_value = mock_resp
        result = tag_image("https://example.com/tent.jpg")

    assert result["category"] == "户外露营"
    assert result["tags"] == []
    assert result["description"] == ""
```

- [ ] **Step 2: 运行，确认失败**

```bash
pytest tests/test_tagger.py -v 2>&1 | tail -5
```

Expected: FAILED（ImportError: cannot import name 'tag_image'）

- [ ] **Step 3: 创建 tagger.py**

```python
import json
from langchain_core.messages import HumanMessage
from config import get_llm
from utils.usage import TokenUsage, accumulate, zero

_PROMPT = """分析这张电商产品图片，提取以下信息并以JSON格式返回（只返回JSON，不要解释）：
{{
  "category": "跑步装备|健身器材|户外露营|其他",
  "tags": ["标签1", "标签2", "标签3"],
  "description": "产品描述（30字以内）",
  "colors": ["主色调1", "主色调2"]
}}"""

_PROMPT_WITH_HINT = """分析这张电商产品图片（商品名称提示：{hint}），提取以下信息并以JSON格式返回（只返回JSON，不要解释）：
{{
  "category": "跑步装备|健身器材|户外露营|其他",
  "tags": ["标签1", "标签2", "标签3"],
  "description": "产品描述（30字以内）",
  "colors": ["主色调1", "主色调2"]
}}"""


def tag_image(image_url: str, product_hint: str = "") -> dict:
    """Call Claude Vision to analyze an image URL. Returns parsed tags + usage."""
    llm = get_llm()
    prompt_text = _PROMPT_WITH_HINT.format(hint=product_hint) if product_hint else _PROMPT

    message = HumanMessage(content=[
        {"type": "image_url", "image_url": {"url": image_url}},
        {"type": "text", "text": prompt_text},
    ])
    response = llm.invoke([message])
    usage = accumulate(zero(), response)

    try:
        raw = response.content.strip().strip("```json").strip("```").strip()
        data = json.loads(raw)
    except (json.JSONDecodeError, ValueError):
        return {
            "category": "其他",
            "tags": [],
            "description": response.content,
            "colors": [],
            "usage": usage,
        }

    return {
        "category": data.get("category", "其他"),
        "tags": data.get("tags", []),
        "description": data.get("description", ""),
        "colors": data.get("colors", []),
        "usage": usage,
    }
```

- [ ] **Step 4: 运行，确认通过**

```bash
pytest tests/test_tagger.py -v 2>&1 | tail -7
```

Expected: 4 PASSED

- [ ] **Step 5: commit**

```bash
git add tagger.py tests/test_tagger.py
git commit -m "feat: Claude Vision tagger with JSON extraction and graceful fallback"
```

---

## Task 4：indexer.py（pgvector 存储 + 语义检索）

**Files:**
- Create: `indexer.py`
- Create: `tests/test_indexer.py`

- [ ] **Step 1: 写失败测试**

创建 `tests/test_indexer.py`：

```python
from unittest.mock import patch, MagicMock


def _make_asset(asset_id="test-001", category="跑步装备",
                tags=None, description="红色专业跑鞋") -> dict:
    return {
        "asset_id": asset_id,
        "image_url": "https://example.com/shoe.jpg",
        "category": category,
        "tags": tags or ["红色", "跑鞋"],
        "description": description,
        "input_tokens": 100,
        "output_tokens": 30,
    }


def test_store_asset_inserts_to_db_and_pgvector():
    from indexer import store_asset
    with patch("indexer.psycopg2.connect") as mock_conn_cls, \
         patch("indexer.PGVector") as mock_vs, \
         patch("indexer.get_embeddings"):
        mock_conn = MagicMock()
        mock_conn.__enter__ = MagicMock(return_value=mock_conn)
        mock_conn.__exit__ = MagicMock(return_value=False)
        mock_conn_cls.return_value = mock_conn
        mock_cur = MagicMock()
        mock_cur.__enter__ = MagicMock(return_value=mock_cur)
        mock_cur.__exit__ = MagicMock(return_value=False)
        mock_conn.cursor.return_value = mock_cur

        store_asset(_make_asset())

    assert mock_conn_cls.called
    assert mock_vs.return_value.add_documents.called


def test_search_assets_returns_results():
    from indexer import search_assets
    mock_doc = MagicMock()
    mock_doc.metadata = {
        "asset_id": "test-001",
        "image_url": "https://example.com/shoe.jpg",
        "category": "跑步装备",
        "tags": ["红色", "跑鞋"],
        "description": "红色专业跑鞋",
    }
    with patch("indexer.PGVector") as mock_vs, \
         patch("indexer.get_embeddings"), \
         patch("indexer.psycopg2.connect") as mock_conn_cls:
        mock_vs.return_value.similarity_search_with_score.return_value = [
            (mock_doc, 0.93)
        ]
        mock_conn = MagicMock()
        mock_conn_cls.return_value.__enter__ = MagicMock(return_value=mock_conn)
        mock_conn_cls.return_value.__exit__ = MagicMock(return_value=False)

        results = search_assets("红色跑鞋", limit=3)

    assert len(results) == 1
    assert results[0]["asset_id"] == "test-001"
    assert results[0]["score"] == 0.93


def test_search_assets_with_category_filter():
    from indexer import search_assets
    mock_doc = MagicMock()
    mock_doc.metadata = {
        "asset_id": "test-002",
        "image_url": "https://example.com/tent.jpg",
        "category": "户外露营",
        "tags": ["帐篷", "防水"],
        "description": "四季防水露营帐篷",
    }
    with patch("indexer.PGVector") as mock_vs, \
         patch("indexer.get_embeddings"), \
         patch("indexer.psycopg2.connect") as mock_conn_cls:
        mock_vs.return_value.similarity_search_with_score.return_value = [
            (mock_doc, 0.88)
        ]
        mock_conn = MagicMock()
        mock_conn_cls.return_value.__enter__ = MagicMock(return_value=mock_conn)
        mock_conn_cls.return_value.__exit__ = MagicMock(return_value=False)

        results = search_assets("防水帐篷", category="户外露营", limit=3)

    assert len(results) == 1
    assert results[0]["category"] == "户外露营"
```

- [ ] **Step 2: 运行，确认失败**

```bash
pytest tests/test_indexer.py -v 2>&1 | tail -5
```

Expected: FAILED（ImportError）

- [ ] **Step 3: 创建 indexer.py**

```python
import json
import psycopg2
import psycopg2.extras
from langchain_core.documents import Document
from langchain_postgres import PGVector
from config import get_embeddings, DATABASE_URL, VECTOR_COLLECTION


def _asset_to_doc(asset: dict) -> Document:
    tags_str = " ".join(asset.get("tags", []))
    content = f"{asset.get('category','')} {tags_str} {asset.get('description','')}".strip()
    return Document(
        page_content=content,
        metadata={
            "asset_id":   asset["asset_id"],
            "image_url":  asset["image_url"],
            "category":   asset.get("category", ""),
            "tags":       json.dumps(asset.get("tags", []), ensure_ascii=False),
            "description": asset.get("description", ""),
        },
    )


def store_asset(asset: dict, database_url: str = None) -> None:
    """Insert asset metadata into media_assets and add embedding to pgvector."""
    url = database_url or DATABASE_URL
    conn = psycopg2.connect(url)
    try:
        with conn.cursor() as cur:
            cur.execute(
                """
                INSERT INTO media_assets
                  (asset_id, image_url, category, tags, description, input_tokens, output_tokens)
                VALUES (%s, %s, %s, %s, %s, %s, %s)
                ON CONFLICT (asset_id) DO UPDATE SET
                  category=EXCLUDED.category, tags=EXCLUDED.tags,
                  description=EXCLUDED.description
                """,
                (
                    asset["asset_id"], asset["image_url"],
                    asset.get("category"), asset.get("tags", []),
                    asset.get("description"), asset.get("input_tokens", 0),
                    asset.get("output_tokens", 0),
                ),
            )
        conn.commit()
    finally:
        conn.close()

    vs = PGVector(
        embeddings=get_embeddings(),
        collection_name=VECTOR_COLLECTION,
        connection=url,
        use_jsonb=True,
    )
    vs.add_documents([_asset_to_doc(asset)])


def search_assets(query: str, category: str = "", limit: int = 6,
                  database_url: str = None) -> list[dict]:
    """Semantic search over indexed assets. Returns list of asset dicts with score."""
    url = database_url or DATABASE_URL
    vs = PGVector(
        embeddings=get_embeddings(),
        collection_name=VECTOR_COLLECTION,
        connection=url,
        use_jsonb=True,
    )

    filter_dict = {}
    if category:
        filter_dict["category"] = category

    results = vs.similarity_search_with_score(
        query, k=limit,
        filter=filter_dict if filter_dict else None,
    )

    return [
        {
            **{k: doc.metadata[k] for k in ("asset_id", "image_url", "category", "description")},
            "tags": json.loads(doc.metadata.get("tags", "[]")),
            "score": round(float(score), 4),
        }
        for doc, score in results
    ]


def list_assets(limit: int = 20, database_url: str = None) -> list[dict]:
    """Return recent assets from media_assets table (no vector search)."""
    url = database_url or DATABASE_URL
    conn = psycopg2.connect(url)
    try:
        with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
            cur.execute(
                "SELECT asset_id, image_url, category, tags, description, created_at "
                "FROM media_assets ORDER BY created_at DESC LIMIT %s",
                (limit,),
            )
            return [dict(r) for r in cur.fetchall()]
    finally:
        conn.close()
```

- [ ] **Step 4: 运行，确认通过**

```bash
pytest tests/test_indexer.py -v 2>&1 | tail -7
```

Expected: 3 PASSED

- [ ] **Step 5: commit**

```bash
git add indexer.py tests/test_indexer.py
git commit -m "feat: pgvector indexer for asset storage and semantic search"
```

---

## Task 5：data/seed_assets.py（演示种子数据）

**Files:**
- Create: `data/seed_assets.py`

- [ ] **Step 1: 创建 data/seed_assets.py**

```python
"""
Pre-tagged seed assets using public Unsplash product images.
Bypasses Claude Vision to avoid API cost during initialization.
"""
import os
import sys
import uuid

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from indexer import store_asset

SAMPLE_ASSETS = [
    {
        "asset_id": str(uuid.uuid5(uuid.NAMESPACE_URL, "shoe-red")),
        "image_url": "https://images.unsplash.com/photo-1542291026-7eec264c27ff?w=400",
        "category": "跑步装备",
        "tags": ["红色", "跑鞋", "专业", "轻量"],
        "description": "专业马拉松跑步鞋，红色配色，碳纤维中底，适合长距离路跑",
        "input_tokens": 0, "output_tokens": 0,
    },
    {
        "asset_id": str(uuid.uuid5(uuid.NAMESPACE_URL, "yoga-mat")),
        "image_url": "https://images.unsplash.com/photo-1601925260368-ae2f83cf8b7f?w=400",
        "category": "健身器材",
        "tags": ["瑜伽垫", "防滑", "TPE", "蓝色"],
        "description": "专业瑜伽垫，TPE材质，双面防滑，蓝色，6mm厚度",
        "input_tokens": 0, "output_tokens": 0,
    },
    {
        "asset_id": str(uuid.uuid5(uuid.NAMESPACE_URL, "camping-tent")),
        "image_url": "https://images.unsplash.com/photo-1504280390367-361c6d9f38f4?w=400",
        "category": "户外露营",
        "tags": ["帐篷", "露营", "防水", "绿色", "四季"],
        "description": "四季通用露营帐篷，防水指数3000mm，绿色，适合3-4人使用",
        "input_tokens": 0, "output_tokens": 0,
    },
    {
        "asset_id": str(uuid.uuid5(uuid.NAMESPACE_URL, "dumbbell")),
        "image_url": "https://images.unsplash.com/photo-1534438327276-14e5300c3a48?w=400",
        "category": "健身器材",
        "tags": ["哑铃", "可调节", "铸铁", "黑色"],
        "description": "可调节哑铃套装，5-30kg，铸铁材质，黑色，附收纳架",
        "input_tokens": 0, "output_tokens": 0,
    },
    {
        "asset_id": str(uuid.uuid5(uuid.NAMESPACE_URL, "backpack")),
        "image_url": "https://images.unsplash.com/photo-1553062407-98eeb64c6a62?w=400",
        "category": "户外露营",
        "tags": ["登山包", "背包", "防水", "橙色", "轻量"],
        "description": "轻量登山背包45L，防水材质，橙色，人体工学背负，重量仅1.2kg",
        "input_tokens": 0, "output_tokens": 0,
    },
]


def seed():
    print(f"Seeding {len(SAMPLE_ASSETS)} sample assets...")
    for asset in SAMPLE_ASSETS:
        store_asset(asset)
        print(f"  ✓ {asset['category']}: {asset['description'][:30]}...")
    print("Asset seeding complete.")


if __name__ == "__main__":
    seed()
```

- [ ] **Step 2: commit**

```bash
git add data/seed_assets.py
git commit -m "feat: pre-tagged seed assets (5 product images, bypasses Vision API on init)"
```

---

## Task 6：main.py（FastAPI /invoke + /health）

**Files:**
- Create: `main.py`
- Create: `tests/test_integration.py`

- [ ] **Step 1: 创建 main.py**

```python
import uuid
from contextlib import asynccontextmanager
from fastapi import FastAPI
from pydantic import BaseModel
from utils.startup import check_and_init
from utils.usage import zero, accumulate

_ready = False


@asynccontextmanager
async def lifespan(app: FastAPI):
    global _ready
    check_and_init()
    _ready = True
    yield


app = FastAPI(title="ShopMind Media Library Agent", version="1.0.0", lifespan=lifespan)


class InvokeRequest(BaseModel):
    action: str                    # "upload" | "search" | "list"
    image_url: str = ""
    product_hint: str = ""
    query: str = ""
    limit: int = 6
    category: str = ""


@app.post("/invoke")
def invoke(req: InvokeRequest):
    if req.action == "upload":
        if not req.image_url:
            return {"error": "image_url is required for upload action"}

        from tagger import tag_image
        result = tag_image(req.image_url, req.product_hint)
        asset_id = str(uuid.uuid4())

        from indexer import store_asset
        asset = {
            "asset_id": asset_id,
            "image_url": req.image_url,
            "category": result["category"],
            "tags": result["tags"],
            "description": result["description"],
            "input_tokens": result["usage"]["input_tokens"],
            "output_tokens": result["usage"]["output_tokens"],
        }
        store_asset(asset)

        return {
            "asset_id": asset_id,
            "image_url": req.image_url,
            "category": result["category"],
            "tags": result["tags"],
            "description": result["description"],
            "usage": result["usage"],
        }

    elif req.action == "search":
        if not req.query:
            return {"error": "query is required for search action"}

        from indexer import search_assets
        results = search_assets(req.query, category=req.category, limit=req.limit)
        return {
            "results": results,
            "usage": zero(),
        }

    elif req.action == "list":
        from indexer import list_assets
        assets = list_assets(limit=req.limit or 20)
        return {"assets": assets, "usage": zero()}

    else:
        return {"error": f"Unknown action: {req.action}. Use 'upload', 'search', or 'list'."}


@app.get("/health")
def health():
    return {"status": "ok", "service": "agent-media-library"}
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


def test_upload_action(client):
    mock_tag_result = {
        "category": "跑步装备",
        "tags": ["红色", "跑鞋"],
        "description": "红色专业跑鞋",
        "colors": ["红色"],
        "usage": {"input_tokens": 500, "output_tokens": 80},
    }
    with patch("main.tag_image", return_value=mock_tag_result), \
         patch("main.store_asset"):
        resp = client.post("/invoke", json={
            "action": "upload",
            "image_url": "https://example.com/shoe.jpg",
        })
    assert resp.status_code == 200
    body = resp.json()
    assert "asset_id" in body
    assert body["category"] == "跑步装备"
    assert body["usage"]["input_tokens"] == 500


def test_upload_missing_url_returns_error(client):
    resp = client.post("/invoke", json={"action": "upload"})
    assert resp.status_code == 200
    assert "error" in resp.json()


def test_search_action(client):
    mock_results = [{"asset_id": "abc", "image_url": "https://x.com/a.jpg",
                     "category": "跑步装备", "tags": ["红色"], "description": "跑鞋", "score": 0.92}]
    with patch("main.search_assets", return_value=mock_results):
        resp = client.post("/invoke", json={"action": "search", "query": "红色跑鞋"})
    assert resp.status_code == 200
    assert len(resp.json()["results"]) == 1
    assert resp.json()["results"][0]["score"] == 0.92


def test_list_action(client):
    mock_assets = [{"asset_id": "abc", "image_url": "https://x.com/a.jpg",
                    "category": "健身器材", "tags": ["哑铃"], "description": "哑铃套装"}]
    with patch("main.list_assets", return_value=mock_assets):
        resp = client.post("/invoke", json={"action": "list"})
    assert resp.status_code == 200
    assert len(resp.json()["assets"]) == 1


def test_unknown_action_returns_error(client):
    resp = client.post("/invoke", json={"action": "invalid"})
    assert "error" in resp.json()


def test_health_endpoint(client):
    resp = client.get("/health")
    assert resp.status_code == 200
    assert resp.json()["service"] == "agent-media-library"
```

- [ ] **Step 3: 运行全部测试**

```bash
pytest tests/ -v 2>&1 | tail -15
```

Expected: 全部 PASSED（约 12 个）

- [ ] **Step 4: 手动验证（需要 .env）**

```bash
# 从 insurance-agent 复制 .env
cp /Users/liuyun/CCWorkSpace/insurance-agent/.env .env
python -m uvicorn main:app --port 8003 --reload
```

```bash
curl -s http://localhost:8003/health
# Expected: {"status":"ok","service":"agent-media-library"}

curl -s -X POST http://localhost:8003/invoke \
  -H "Content-Type: application/json" \
  -d '{"action":"list"}'
# Expected: {"assets":[...5条种子数据...],"usage":{...}}
```

- [ ] **Step 5: commit**

```bash
git add main.py tests/test_integration.py
git commit -m "feat: FastAPI /invoke (upload/search/list) + /health + 12 tests passing"
```

---

## Task 7：更新 shopmind-web（素材库页面）

**Files:**
- Modify: `shopmind-web/app/dashboard/layout.tsx` — 加素材库入口
- Create: `shopmind-web/app/dashboard/media/page.tsx` — 上传 + 搜索 + 网格展示
- Modify: `shopmind-web/app/dashboard/page.tsx` — 已接入 Agent 改为 3/8

- [ ] **Step 1: 切换到 shopmind-web develop 分支**

```bash
cd /Users/liuyun/CCWorkSpace/shopmind-ai/shopmind-web
git checkout develop
mkdir -p app/dashboard/media
```

- [ ] **Step 2: 更新侧边栏 app/dashboard/layout.tsx**

将 `navItems` 从：

```typescript
const navItems = [
  { href: "/dashboard",          label: "概览",   emoji: "📊" },
  { href: "/dashboard/cleaning", label: "数据清洗", emoji: "🧹" },
  { href: "/dashboard/customer", label: "智能客服", emoji: "💬" },
];
```

改为：

```typescript
const navItems = [
  { href: "/dashboard",          label: "概览",   emoji: "📊" },
  { href: "/dashboard/cleaning", label: "数据清洗", emoji: "🧹" },
  { href: "/dashboard/customer", label: "智能客服", emoji: "💬" },
  { href: "/dashboard/media",    label: "素材库",   emoji: "🖼️" },
];
```

- [ ] **Step 3: 更新 app/dashboard/page.tsx（3/8）**

将 `2 / 8` 改为 `3 / 8`：

```typescript
          <p className="text-3xl font-bold mt-1">3 / 8</p>
```

- [ ] **Step 4: 创建 app/dashboard/media/page.tsx**

```typescript
"use client";
import { useState, useEffect } from "react";
import { api } from "@/lib/api";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Card, CardContent } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";

type Asset = {
  asset_id: string;
  image_url: string;
  category: string;
  tags: string[];
  description: string;
  score?: number;
};

const CATEGORIES = ["全部", "跑步装备", "健身器材", "户外露营"];

export default function MediaPage() {
  const [assets, setAssets] = useState<Asset[]>([]);
  const [uploadUrl, setUploadUrl] = useState("");
  const [productHint, setProductHint] = useState("");
  const [searchQuery, setSearchQuery] = useState("");
  const [categoryFilter, setCategoryFilter] = useState("全部");
  const [loading, setLoading] = useState(false);
  const [uploading, setUploading] = useState(false);
  const [lastUsage, setLastUsage] = useState<{ input_tokens: number; output_tokens: number } | null>(null);
  const [error, setError] = useState("");

  useEffect(() => { loadAssets(); }, []);

  async function loadAssets() {
    setLoading(true);
    try {
      const res = await api.invokeAgent("media", { action: "list", limit: 20 }) as { assets: Asset[] };
      setAssets(res.assets || []);
    } catch { /* ignore */ } finally {
      setLoading(false);
    }
  }

  async function handleUpload() {
    if (!uploadUrl.trim()) return;
    setUploading(true);
    setError("");
    try {
      const res = await api.invokeAgent("media", {
        action: "upload",
        image_url: uploadUrl.trim(),
        product_hint: productHint.trim(),
      }) as Asset & { usage: { input_tokens: number; output_tokens: number } };
      if ("error" in res) { setError(String((res as { error: string }).error)); return; }
      setLastUsage((res as { usage: { input_tokens: number; output_tokens: number } }).usage);
      setUploadUrl("");
      setProductHint("");
      await loadAssets();
    } catch (e) {
      setError(e instanceof Error ? e.message : "Upload failed");
    } finally {
      setUploading(false);
    }
  }

  async function handleSearch() {
    if (!searchQuery.trim()) { loadAssets(); return; }
    setLoading(true);
    setError("");
    try {
      const res = await api.invokeAgent("media", {
        action: "search",
        query: searchQuery.trim(),
        category: categoryFilter === "全部" ? "" : categoryFilter,
        limit: 12,
      }) as { results: Asset[]; usage: { input_tokens: number; output_tokens: number } };
      setAssets(res.results || []);
      setLastUsage(res.usage);
    } catch (e) {
      setError(e instanceof Error ? e.message : "Search failed");
    } finally {
      setLoading(false);
    }
  }

  const CATEGORY_COLOR: Record<string, string> = {
    "跑步装备": "bg-blue-100 text-blue-700",
    "健身器材": "bg-green-100 text-green-700",
    "户外露营": "bg-amber-100 text-amber-700",
  };

  return (
    <div className="space-y-6">
      <div>
        <h1 className="text-2xl font-bold">🖼️ 多模态素材库</h1>
        <p className="text-slate-500 text-sm mt-1">上传商品图片 URL，AI 自动打标签 · 语义搜索素材</p>
      </div>

      {/* 上传区 */}
      <Card>
        <CardContent className="p-4 space-y-3">
          <p className="font-medium text-sm">上传商品图片</p>
          <div className="flex gap-2">
            <Input
              value={uploadUrl}
              onChange={e => setUploadUrl(e.target.value)}
              placeholder="粘贴图片 URL（支持 HTTPS）"
              className="flex-1"
            />
            <Input
              value={productHint}
              onChange={e => setProductHint(e.target.value)}
              placeholder="商品名称提示（可选）"
              className="w-40"
            />
            <Button onClick={handleUpload} disabled={uploading || !uploadUrl.trim()}>
              {uploading ? "分析中..." : "上传"}
            </Button>
          </div>
          {error && <p className="text-red-500 text-sm">{error}</p>}
          {lastUsage && (
            <p className="text-xs text-slate-400">
              Tokens: {lastUsage.input_tokens + lastUsage.output_tokens}
              （${((lastUsage.input_tokens / 1e6 * 3) + (lastUsage.output_tokens / 1e6 * 15)).toFixed(4)}）
            </p>
          )}
        </CardContent>
      </Card>

      {/* 搜索区 */}
      <div className="flex gap-2">
        <Input
          value={searchQuery}
          onChange={e => setSearchQuery(e.target.value)}
          onKeyDown={e => e.key === "Enter" && handleSearch()}
          placeholder="语义搜索：红色跑鞋、防水装备..."
          className="flex-1"
        />
        <select
          value={categoryFilter}
          onChange={e => setCategoryFilter(e.target.value)}
          className="border rounded px-2 text-sm"
        >
          {CATEGORIES.map(c => <option key={c}>{c}</option>)}
        </select>
        <Button onClick={handleSearch} disabled={loading} variant="outline">搜索</Button>
        <Button onClick={loadAssets} disabled={loading} variant="ghost">全部</Button>
      </div>

      {/* 素材网格 */}
      {loading ? (
        <p className="text-slate-400 text-sm">加载中...</p>
      ) : assets.length === 0 ? (
        <p className="text-slate-400 text-sm">暂无素材，上传第一张商品图片吧</p>
      ) : (
        <div className="grid grid-cols-2 md:grid-cols-3 gap-4">
          {assets.map(asset => (
            <Card key={asset.asset_id} className="overflow-hidden">
              <img
                src={asset.image_url}
                alt={asset.description}
                className="w-full h-40 object-cover bg-slate-100"
                onError={e => { (e.target as HTMLImageElement).src = "https://placehold.co/400x300?text=Image"; }}
              />
              <CardContent className="p-3 space-y-2">
                <div className="flex items-center justify-between">
                  <span className={`text-xs px-2 py-0.5 rounded-full font-medium ${CATEGORY_COLOR[asset.category] ?? "bg-slate-100 text-slate-600"}`}>
                    {asset.category}
                  </span>
                  {asset.score !== undefined && (
                    <span className="text-xs text-slate-400">{(asset.score * 100).toFixed(0)}%</span>
                  )}
                </div>
                <p className="text-xs text-slate-600 line-clamp-2">{asset.description}</p>
                <div className="flex flex-wrap gap-1">
                  {(asset.tags || []).slice(0, 4).map(tag => (
                    <Badge key={tag} variant="secondary" className="text-xs px-1.5 py-0">{tag}</Badge>
                  ))}
                </div>
              </CardContent>
            </Card>
          ))}
        </div>
      )}
    </div>
  );
}
```

- [ ] **Step 5: 构建验证**

```bash
npm run build 2>&1 | tail -8
```

Expected: 构建成功，新增 `/dashboard/media` 路由。

- [ ] **Step 6: commit**

```bash
git add app/dashboard/layout.tsx app/dashboard/media/ app/dashboard/page.tsx
git commit -m "feat: media library page with upload/search/grid + update agent count to 3/8"
```

---

## Task 8：注册 Gateway + 端到端测试 + 推送

**Files:**
- Modify: `shopmind-gateway/.env` — 新增 AGENT_MEDIA_URL

- [ ] **Step 1: 更新 gateway .env**

```bash
echo "AGENT_MEDIA_URL=http://localhost:8003" >> /Users/liuyun/CCWorkSpace/shopmind-ai/shopmind-gateway/.env
```

- [ ] **Step 2: 启动三个服务**

终端 1：
```bash
cd /Users/liuyun/CCWorkSpace/shopmind-ai/agent-media-library
python -m uvicorn main:app --port 8003 --log-level warning
```

终端 2（gateway 已运行，重启以加载新 .env）：
```bash
pkill -f "uvicorn main:app --port 8000"; sleep 1
cd /Users/liuyun/CCWorkSpace/shopmind-ai/shopmind-gateway
python -m uvicorn main:app --port 8000 --log-level warning
```

终端 3：
```bash
cd /Users/liuyun/CCWorkSpace/shopmind-ai/shopmind-web && npm run dev
```

- [ ] **Step 3: 端到端 curl 测试**

```bash
TOKEN=$(curl -s -X POST http://localhost:8000/auth/login \
  -H "Content-Type: application/json" \
  -d '{"username":"demo","password":"shopmind2026"}' \
  | python3 -c "import sys,json; print(json.load(sys.stdin)['access_token'])")

# 列出素材（应有5条种子数据）
curl -s -X POST http://localhost:8000/api/v1/media/invoke \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer $TOKEN" \
  -d '{"action":"list"}' \
  | python3 -c "import sys,json; r=json.load(sys.stdin); print('assets:', len(r['assets']))"
```

Expected: `assets: 5`

```bash
# 语义搜索测试
curl -s -X POST http://localhost:8000/api/v1/media/invoke \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer $TOKEN" \
  -d '{"action":"search","query":"户外防水装备"}' \
  | python3 -c "import sys,json; r=json.load(sys.stdin); [print(x['category'], x['score']) for x in r['results'][:3]]"
```

Expected: 返回相关素材（category + score）

- [ ] **Step 4: 浏览器验证**

访问 http://localhost:3000/dashboard/media：
- 侧边栏出现"🖼️ 素材库"
- 自动加载 5 条种子素材（网格展示）
- 搜索"红色跑鞋"返回匹配结果并显示相似度分数

- [ ] **Step 5: 推送到 GitHub**

```bash
cd /Users/liuyun/CCWorkSpace/shopmind-ai/agent-media-library
git add . && git commit -m "chore: phase3 complete - media library agent integrated with gateway"

gh repo create shopmind-ai/agent-media-library \
  --public \
  --description "ShopMind AI - Multimodal asset library: Claude Vision auto-tagging + pgvector semantic search" \
  --source=. --remote=origin --push

cd /Users/liuyun/CCWorkSpace/shopmind-ai/shopmind-web
git push origin develop

cd /Users/liuyun/CCWorkSpace/shopmind-ai/shopmind-gateway
git add .env.example routers/proxy.py
git commit -m "fix: add AGENT_MEDIA_URL to .env.example"
git push origin develop
```

---

## 自审通过项

- ✅ `tagger.py` 的 `tag_image()` 返回 `usage` 字段，与 `main.py` 的用法一致
- ✅ `indexer.py` 的 `store_asset()` 参数与 `main.py` 传入的 asset dict 字段完全匹配
- ✅ `seed_assets.py` 使用 `store_asset()` 而非直接写 SQL，保证与 pgvector 同步
- ✅ `list_assets()` 返回的字段与 `Asset` TypeScript 类型匹配（asset_id, image_url, category, tags, description）
- ✅ 搜索结果包含 `score` 字段，前端用于显示相似度百分比
- ✅ `utils/usage.py` 的 `accumulate()` 签名：`(prev: TokenUsage, response) -> TokenUsage`（不依赖 state 对象，更通用）
- ✅ gateway `.env` 需手动加 `AGENT_MEDIA_URL=http://localhost:8003`（Step 8）
- ✅ shopmind-web dashboard page.tsx 已更新为 3/8
