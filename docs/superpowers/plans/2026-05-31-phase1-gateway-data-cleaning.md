# ShopMind AI Phase 1 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 搭建 shopmind-gateway（FastAPI API 网关，JWT 认证 + 代理路由 + Token 统计）、agent-data-cleaning（电商数据清洗 Agent，规则引擎 + LLM 双层）、shopmind-web（Next.js 前端基础框架 + 登录 + 清洗页面），三者联通可端到端演示。

**Architecture:** shopmind-gateway 作为唯一入口接受前端请求，验证 JWT 后代理转发到 agent-data-cleaning，收集 token 用量写入 usage_logs 表。data-cleaning 内部走 LangGraph 管道：rule_clean → detect → clean → validate。

**Tech Stack:** Python 3.11 · FastAPI · LangGraph · LangChain · Claude API · Ollama · psycopg2 · python-jose · passlib · httpx · Next.js 14 · TypeScript · shadcn/ui · Tailwind CSS

---

## 文件结构总览

```
shopmind-gateway/
├── main.py                  # FastAPI 入口，挂载所有路由
├── config.py                # 环境变量（SECRET_KEY, AGENT_*_URL, DATABASE_URL）
├── database.py              # psycopg2 连接工厂
├── routers/
│   ├── auth.py              # POST /auth/login
│   ├── proxy.py             # POST /api/v1/{agent}/invoke  GET /api/v1/{agent}/health
│   └── usage.py             # GET /api/v1/usage/summary
├── middleware/
│   └── auth.py              # JWT Bearer 验证依赖
├── utils/
│   ├── jwt.py               # create_token / verify_token
│   ├── password.py          # hash_password / verify_password
│   └── startup.py           # 建表 users + usage_logs，seed demo 用户
├── tests/
│   ├── test_jwt.py
│   ├── test_auth.py
│   └── test_proxy.py
├── requirements.txt
└── .env.example

agent-data-cleaning/
├── main.py                  # FastAPI 入口，POST /invoke  GET /health
├── config.py                # DATABASE_URL, ANTHROPIC_API_KEY, LLM_PROVIDER
├── agent/
│   ├── state.py             # CleaningState TypedDict
│   ├── rule_cleaner.py      # 规则引擎（无 LLM）
│   ├── detector.py          # LLM 检测语义问题
│   ├── cleaner.py           # LLM 修复问题
│   ├── validator.py         # 规则验证输出
│   └── graph.py             # LangGraph 图组装
├── utils/
│   ├── startup.py           # 建表 cleaning_jobs
│   └── usage.py             # Token 统计（复用 insurance-agent 模式）
├── data/
│   └── sample_dirty.json    # 脏数据演示样本
├── tests/
│   ├── test_rule_cleaner.py
│   ├── test_detector.py
│   ├── test_cleaner.py
│   ├── test_validator.py
│   └── test_integration.py
├── requirements.txt
└── .env.example

shopmind-web/
├── app/
│   ├── layout.tsx           # 根布局
│   ├── page.tsx             # 首页（重定向到 /dashboard）
│   ├── login/page.tsx       # 登录页
│   └── dashboard/
│       ├── layout.tsx       # 侧边栏布局
│       ├── page.tsx         # 仪表盘首页
│       └── cleaning/page.tsx # 数据清洗页
├── components/
│   ├── login-form.tsx
│   └── cleaning-panel.tsx
├── lib/
│   ├── api.ts               # Gateway HTTP 客户端
│   └── auth.ts              # token 存储 + 读取
├── middleware.ts             # 保护 /dashboard 路由
├── package.json
└── .env.local.example
```

---

## Task 1：shopmind-gateway 项目初始化 + 数据库

**Files:**
- Create: `shopmind-gateway/requirements.txt`
- Create: `shopmind-gateway/.env.example`
- Create: `shopmind-gateway/config.py`
- Create: `shopmind-gateway/database.py`
- Create: `shopmind-gateway/utils/startup.py`

- [ ] **Step 1: 创建目录结构**

```bash
mkdir -p shopmind-gateway/{routers,middleware,utils,tests}
touch shopmind-gateway/routers/__init__.py
touch shopmind-gateway/middleware/__init__.py
touch shopmind-gateway/utils/__init__.py
touch shopmind-gateway/tests/__init__.py
cd shopmind-gateway && git init && git commit --allow-empty -m "chore: init shopmind-gateway"
```

- [ ] **Step 2: 创建 requirements.txt**

```
fastapi>=0.115.0
uvicorn[standard]>=0.30.0
python-jose[cryptography]>=3.3.0
passlib[bcrypt]>=1.7.4
httpx>=0.27.0
psycopg2-binary>=2.9.9
python-dotenv>=1.0.0
pytest>=8.0.0
pytest-asyncio>=0.24.0
httpx>=0.27.0
```

- [ ] **Step 3: 创建 .env.example**

```
# JWT
SECRET_KEY=change-me-in-production-use-openssl-rand-hex-32
ACCESS_TOKEN_EXPIRE_MINUTES=60
REFRESH_TOKEN_EXPIRE_DAYS=7

# Database
DATABASE_URL=postgresql://user:pass@host/dbname?sslmode=require

# Agent service URLs (used by proxy router)
AGENT_CLEANING_URL=http://localhost:8001
AGENT_CUSTOMER_URL=http://localhost:8002
AGENT_MEDIA_URL=http://localhost:8003
AGENT_SOCIAL_URL=http://localhost:8004
AGENT_CLIP_URL=http://localhost:8005
AGENT_TRAINING_URL=http://localhost:8006
AGENT_CLUSTER_URL=http://localhost:8007
AGENT_FINETUNE_URL=http://localhost:8008
```

- [ ] **Step 4: 创建 config.py**

```python
import os
from dotenv import load_dotenv

load_dotenv()

SECRET_KEY = os.getenv("SECRET_KEY", "dev-secret-change-in-prod")
ACCESS_TOKEN_EXPIRE_MINUTES = int(os.getenv("ACCESS_TOKEN_EXPIRE_MINUTES", "60"))
REFRESH_TOKEN_EXPIRE_DAYS = int(os.getenv("REFRESH_TOKEN_EXPIRE_DAYS", "7"))
DATABASE_URL = os.getenv("DATABASE_URL", "")

AGENT_URLS: dict[str, str] = {
    "cleaning":  os.getenv("AGENT_CLEANING_URL",  "http://localhost:8001"),
    "customer":  os.getenv("AGENT_CUSTOMER_URL",  "http://localhost:8002"),
    "media":     os.getenv("AGENT_MEDIA_URL",     "http://localhost:8003"),
    "social":    os.getenv("AGENT_SOCIAL_URL",    "http://localhost:8004"),
    "clip":      os.getenv("AGENT_CLIP_URL",      "http://localhost:8005"),
    "training":  os.getenv("AGENT_TRAINING_URL",  "http://localhost:8006"),
    "cluster":   os.getenv("AGENT_CLUSTER_URL",   "http://localhost:8007"),
    "finetune":  os.getenv("AGENT_FINETUNE_URL",  "http://localhost:8008"),
}
```

- [ ] **Step 5: 创建 database.py**

```python
import psycopg2
import psycopg2.extras
from config import DATABASE_URL


def get_conn():
    return psycopg2.connect(DATABASE_URL)
```

- [ ] **Step 6: 创建 utils/startup.py**

```python
import psycopg2
from config import DATABASE_URL
from utils.password import hash_password

DDL = """
CREATE TABLE IF NOT EXISTS users (
    id            SERIAL PRIMARY KEY,
    username      VARCHAR(50)  UNIQUE NOT NULL,
    email         VARCHAR(100) UNIQUE NOT NULL,
    hashed_password VARCHAR(255) NOT NULL,
    created_at    TIMESTAMPTZ DEFAULT NOW()
);

CREATE TABLE IF NOT EXISTS usage_logs (
    id            SERIAL PRIMARY KEY,
    user_id       INTEGER REFERENCES users(id),
    agent_name    VARCHAR(50) NOT NULL,
    input_tokens  INTEGER DEFAULT 0,
    output_tokens INTEGER DEFAULT 0,
    created_at    TIMESTAMPTZ DEFAULT NOW()
);
"""


def check_and_init():
    conn = psycopg2.connect(DATABASE_URL)
    try:
        with conn.cursor() as cur:
            cur.execute(DDL)
            cur.execute("SELECT COUNT(*) FROM users")
            if cur.fetchone()[0] == 0:
                cur.execute(
                    "INSERT INTO users (username, email, hashed_password) VALUES (%s, %s, %s)",
                    ("demo", "demo@shopmind.ai", hash_password("shopmind2026")),
                )
                print("[startup] Demo user created: demo / shopmind2026")
        conn.commit()
        print("[startup] Gateway DB initialized.")
    finally:
        conn.close()
```

- [ ] **Step 7: 安装依赖**

```bash
pip install -r requirements.txt
```

Expected: 无报错。

- [ ] **Step 8: commit**

```bash
git add . && git commit -m "feat: gateway project setup and DB init"
```

---

## Task 2：JWT 工具 + 密码工具

**Files:**
- Create: `shopmind-gateway/utils/jwt.py`
- Create: `shopmind-gateway/utils/password.py`
- Create: `shopmind-gateway/tests/test_jwt.py`

- [ ] **Step 1: 写失败测试**

创建 `tests/test_jwt.py`：

```python
def test_create_and_verify_access_token():
    from utils.jwt import create_access_token, verify_token
    token = create_access_token("demo")
    payload = verify_token(token)
    assert payload["sub"] == "demo"
    assert payload["type"] == "access"


def test_verify_invalid_token_returns_none():
    from utils.jwt import verify_token
    result = verify_token("not.a.valid.token")
    assert result is None


def test_create_and_verify_refresh_token():
    from utils.jwt import create_refresh_token, verify_token
    token = create_refresh_token("demo")
    payload = verify_token(token)
    assert payload["sub"] == "demo"
    assert payload["type"] == "refresh"
```

- [ ] **Step 2: 运行，确认失败**

```bash
pytest tests/test_jwt.py -v
```

Expected: FAILED（ImportError）

- [ ] **Step 3: 创建 utils/password.py**

```python
from passlib.context import CryptContext

_ctx = CryptContext(schemes=["bcrypt"], deprecated="auto")


def hash_password(plain: str) -> str:
    return _ctx.hash(plain)


def verify_password(plain: str, hashed: str) -> bool:
    return _ctx.verify(plain, hashed)
```

- [ ] **Step 4: 创建 utils/jwt.py**

```python
from datetime import datetime, timedelta, timezone
from jose import jwt, JWTError
from config import SECRET_KEY, ACCESS_TOKEN_EXPIRE_MINUTES, REFRESH_TOKEN_EXPIRE_DAYS

_ALGORITHM = "HS256"


def create_access_token(username: str) -> str:
    expire = datetime.now(timezone.utc) + timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    return jwt.encode(
        {"sub": username, "type": "access", "exp": expire},
        SECRET_KEY, algorithm=_ALGORITHM,
    )


def create_refresh_token(username: str) -> str:
    expire = datetime.now(timezone.utc) + timedelta(days=REFRESH_TOKEN_EXPIRE_DAYS)
    return jwt.encode(
        {"sub": username, "type": "refresh", "exp": expire},
        SECRET_KEY, algorithm=_ALGORITHM,
    )


def verify_token(token: str) -> dict | None:
    try:
        return jwt.decode(token, SECRET_KEY, algorithms=[_ALGORITHM])
    except JWTError:
        return None
```

- [ ] **Step 5: 运行，确认通过**

```bash
pytest tests/test_jwt.py -v
```

Expected: 3 PASSED

- [ ] **Step 6: commit**

```bash
git add utils/jwt.py utils/password.py tests/test_jwt.py
git commit -m "feat: JWT token creation/verification + bcrypt password utils"
```

---

## Task 3：认证路由 + 认证中间件

**Files:**
- Create: `shopmind-gateway/routers/auth.py`
- Create: `shopmind-gateway/middleware/auth.py`
- Create: `shopmind-gateway/tests/test_auth.py`

- [ ] **Step 1: 写失败测试**

创建 `tests/test_auth.py`：

```python
import pytest
from fastapi.testclient import TestClient
from unittest.mock import patch, MagicMock


@pytest.fixture
def client():
    from main import app
    return TestClient(app)


def test_login_success(client):
    mock_user = {
        "id": 1,
        "username": "demo",
        "hashed_password": "$2b$12$placeholder"
    }
    with patch("routers.auth.get_user_by_username", return_value=mock_user), \
         patch("routers.auth.verify_password", return_value=True):
        resp = client.post("/auth/login", json={"username": "demo", "password": "shopmind2026"})
    assert resp.status_code == 200
    assert "access_token" in resp.json()
    assert "refresh_token" in resp.json()


def test_login_wrong_password(client):
    mock_user = {"id": 1, "username": "demo", "hashed_password": "$2b$12$placeholder"}
    with patch("routers.auth.get_user_by_username", return_value=mock_user), \
         patch("routers.auth.verify_password", return_value=False):
        resp = client.post("/auth/login", json={"username": "demo", "password": "wrong"})
    assert resp.status_code == 401


def test_login_user_not_found(client):
    with patch("routers.auth.get_user_by_username", return_value=None):
        resp = client.post("/auth/login", json={"username": "noone", "password": "x"})
    assert resp.status_code == 401


def test_protected_route_without_token(client):
    resp = client.get("/api/v1/usage/summary")
    assert resp.status_code == 401


def test_protected_route_with_valid_token(client):
    from utils.jwt import create_access_token
    token = create_access_token("demo")
    with patch("routers.usage.get_usage_summary", return_value={"total_input": 0, "total_output": 0}):
        resp = client.get("/api/v1/usage/summary", headers={"Authorization": f"Bearer {token}"})
    assert resp.status_code == 200
```

- [ ] **Step 2: 运行，确认失败**

```bash
pytest tests/test_auth.py -v
```

Expected: FAILED（ImportError — main.py 不存在）

- [ ] **Step 3: 创建 routers/auth.py**

```python
from fastapi import APIRouter, HTTPException, status
from pydantic import BaseModel
import psycopg2.extras
from database import get_conn
from utils.jwt import create_access_token, create_refresh_token
from utils.password import verify_password

router = APIRouter()


class LoginRequest(BaseModel):
    username: str
    password: str


def get_user_by_username(username: str) -> dict | None:
    conn = get_conn()
    try:
        with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
            cur.execute("SELECT id, username, hashed_password FROM users WHERE username = %s", (username,))
            row = cur.fetchone()
            return dict(row) if row else None
    finally:
        conn.close()


@router.post("/auth/login")
def login(req: LoginRequest):
    user = get_user_by_username(req.username)
    if not user or not verify_password(req.password, user["hashed_password"]):
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid credentials")
    return {
        "access_token": create_access_token(user["username"]),
        "refresh_token": create_refresh_token(user["username"]),
        "token_type": "bearer",
    }
```

- [ ] **Step 4: 创建 middleware/auth.py**

```python
from fastapi import Depends, HTTPException, status
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from utils.jwt import verify_token

_bearer = HTTPBearer()


def require_auth(credentials: HTTPAuthorizationCredentials = Depends(_bearer)) -> str:
    payload = verify_token(credentials.credentials)
    if not payload or payload.get("type") != "access":
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or expired token",
        )
    return payload["sub"]
```

- [ ] **Step 5: 创建 routers/usage.py（stub，供测试用）**

```python
from fastapi import APIRouter, Depends
import psycopg2.extras
from database import get_conn
from middleware.auth import require_auth

router = APIRouter()


def get_usage_summary() -> dict:
    conn = get_conn()
    try:
        with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
            cur.execute("""
                SELECT
                    agent_name,
                    SUM(input_tokens)  AS total_input,
                    SUM(output_tokens) AS total_output,
                    COUNT(*)           AS call_count
                FROM usage_logs
                GROUP BY agent_name
                ORDER BY agent_name
            """)
            rows = cur.fetchall()
            return {"by_agent": [dict(r) for r in rows]}
    finally:
        conn.close()


@router.get("/api/v1/usage/summary")
def usage_summary(_: str = Depends(require_auth)):
    return get_usage_summary()
```

- [ ] **Step 6: 创建 main.py**

```python
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from utils.startup import check_and_init
from routers import auth, proxy, usage

app = FastAPI(title="ShopMind Gateway", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.on_event("startup")
def on_startup():
    check_and_init()


app.include_router(auth.router)
app.include_router(usage.router)
```

（proxy router 在 Task 4 添加）

- [ ] **Step 7: 运行测试**

```bash
pytest tests/test_auth.py -v
```

Expected: 5 PASSED

- [ ] **Step 8: commit**

```bash
git add routers/auth.py routers/usage.py middleware/auth.py main.py tests/test_auth.py
git commit -m "feat: JWT auth login endpoint + Bearer token middleware"
```

---

## Task 4：代理路由 + usage 写入

**Files:**
- Create: `shopmind-gateway/routers/proxy.py`
- Create: `shopmind-gateway/tests/test_proxy.py`
- Modify: `shopmind-gateway/main.py`

- [ ] **Step 1: 写失败测试**

创建 `tests/test_proxy.py`：

```python
import pytest
from fastapi.testclient import TestClient
from unittest.mock import patch, AsyncMock
from utils.jwt import create_access_token


@pytest.fixture
def client():
    from main import app
    return TestClient(app)


@pytest.fixture
def auth_headers():
    return {"Authorization": f"Bearer {create_access_token('demo')}"}


def test_invoke_known_agent_success(client, auth_headers):
    mock_response = AsyncMock()
    mock_response.status_code = 200
    mock_response.json.return_value = {
        "cleaned_data": [],
        "report": {},
        "usage": {"input_tokens": 100, "output_tokens": 50},
    }
    with patch("routers.proxy.httpx.AsyncClient") as mock_client_cls:
        mock_client_cls.return_value.__aenter__.return_value.post = AsyncMock(return_value=mock_response)
        with patch("routers.proxy.log_usage"):
            resp = client.post(
                "/api/v1/cleaning/invoke",
                json={"data": [], "data_type": "products"},
                headers=auth_headers,
            )
    assert resp.status_code == 200


def test_invoke_unknown_agent_returns_404(client, auth_headers):
    resp = client.post("/api/v1/unknown/invoke", json={}, headers=auth_headers)
    assert resp.status_code == 404


def test_invoke_without_auth_returns_401(client):
    resp = client.post("/api/v1/cleaning/invoke", json={})
    assert resp.status_code == 401


def test_health_check_proxied(client, auth_headers):
    mock_response = AsyncMock()
    mock_response.status_code = 200
    mock_response.json.return_value = {"status": "ok"}
    with patch("routers.proxy.httpx.AsyncClient") as mock_client_cls:
        mock_client_cls.return_value.__aenter__.return_value.get = AsyncMock(return_value=mock_response)
        resp = client.get("/api/v1/cleaning/health", headers=auth_headers)
    assert resp.status_code == 200
```

- [ ] **Step 2: 运行，确认失败**

```bash
pytest tests/test_proxy.py -v
```

Expected: FAILED（proxy router 不存在）

- [ ] **Step 3: 创建 routers/proxy.py**

```python
import httpx
from fastapi import APIRouter, Depends, HTTPException, Request
from fastapi.responses import JSONResponse
import psycopg2
from config import AGENT_URLS
from database import get_conn
from middleware.auth import require_auth

router = APIRouter()
_TIMEOUT = 10.0


def log_usage(username: str, agent_name: str, usage: dict):
    if not usage:
        return
    conn = get_conn()
    try:
        with conn.cursor() as cur:
            cur.execute("SELECT id FROM users WHERE username = %s", (username,))
            row = cur.fetchone()
            user_id = row[0] if row else None
            cur.execute(
                "INSERT INTO usage_logs (user_id, agent_name, input_tokens, output_tokens) VALUES (%s,%s,%s,%s)",
                (user_id, agent_name, usage.get("input_tokens", 0), usage.get("output_tokens", 0)),
            )
        conn.commit()
    finally:
        conn.close()


@router.post("/api/v1/{agent}/invoke")
async def invoke(agent: str, request: Request, username: str = Depends(require_auth)):
    if agent not in AGENT_URLS:
        raise HTTPException(status_code=404, detail=f"Unknown agent: {agent}")

    body = await request.json()
    url = f"{AGENT_URLS[agent]}/invoke"

    try:
        async with httpx.AsyncClient(timeout=_TIMEOUT) as client:
            resp = await client.post(url, json=body)
            data = resp.json()
            log_usage(username, agent, data.get("usage", {}))
            return JSONResponse(status_code=resp.status_code, content=data)
    except httpx.TimeoutException:
        return JSONResponse(
            status_code=503,
            content={"status": "degraded", "message": "服务暂时不可用，请稍后重试"},
        )


@router.get("/api/v1/{agent}/health")
async def health(agent: str, _: str = Depends(require_auth)):
    if agent not in AGENT_URLS:
        raise HTTPException(status_code=404, detail=f"Unknown agent: {agent}")

    url = f"{AGENT_URLS[agent]}/health"
    try:
        async with httpx.AsyncClient(timeout=5.0) as client:
            resp = await client.get(url)
            return JSONResponse(status_code=resp.status_code, content=resp.json())
    except httpx.TimeoutException:
        return JSONResponse(status_code=503, content={"status": "unreachable"})
```

- [ ] **Step 4: 更新 main.py，加入 proxy router**

```python
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from utils.startup import check_and_init
from routers import auth, proxy, usage

app = FastAPI(title="ShopMind Gateway", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.on_event("startup")
def on_startup():
    check_and_init()


app.include_router(auth.router)
app.include_router(proxy.router)
app.include_router(usage.router)


@app.get("/health")
def health():
    return {"status": "ok", "service": "shopmind-gateway"}
```

- [ ] **Step 5: 运行全部测试**

```bash
pytest tests/ -v
```

Expected: 全部 PASSED（12 个左右）

- [ ] **Step 6: 手动验证（需要有 .env 并连接数据库）**

```bash
uvicorn main:app --reload --port 8000
curl -X POST http://localhost:8000/auth/login \
  -H "Content-Type: application/json" \
  -d '{"username":"demo","password":"shopmind2026"}'
```

Expected: 返回包含 `access_token` 的 JSON。

- [ ] **Step 7: commit**

```bash
git add routers/proxy.py tests/test_proxy.py main.py
git commit -m "feat: proxy router with JWT auth, timeout fallback, usage logging"
```

---

## Task 5：agent-data-cleaning 项目初始化 + State + 样本数据

**Files:**
- Create: `agent-data-cleaning/requirements.txt`
- Create: `agent-data-cleaning/.env.example`
- Create: `agent-data-cleaning/config.py`
- Create: `agent-data-cleaning/agent/state.py`
- Create: `agent-data-cleaning/utils/usage.py`
- Create: `agent-data-cleaning/data/sample_dirty.json`

- [ ] **Step 1: 创建目录结构**

```bash
mkdir -p agent-data-cleaning/{agent,utils,data,tests}
touch agent-data-cleaning/agent/__init__.py
touch agent-data-cleaning/utils/__init__.py
touch agent-data-cleaning/tests/__init__.py
cd agent-data-cleaning && git init && git commit --allow-empty -m "chore: init agent-data-cleaning"
```

- [ ] **Step 2: 创建 requirements.txt**

```
fastapi>=0.115.0
uvicorn[standard]>=0.30.0
langchain>=0.3.0
langchain-anthropic>=0.3.0
langchain-ollama>=0.2.0
langgraph>=0.2.0
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

- [ ] **Step 4: 创建 config.py**

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

- [ ] **Step 5: 创建 agent/state.py**

```python
from typing import TypedDict


class TokenUsage(TypedDict):
    input_tokens: int
    output_tokens: int


class CleaningState(TypedDict):
    raw_data: list[dict]       # 原始脏数据
    data_type: str             # "products" | "orders" | "customers"
    rule_cleaned: list[dict]   # 规则引擎清洗后
    issues: list[dict]         # LLM 检测到的语义问题
    ai_cleaned: list[dict]     # LLM 修复后
    final_data: list[dict]     # 验证通过的最终数据
    report: dict               # 清洗报告
    usage: TokenUsage          # token 累计
```

- [ ] **Step 6: 创建 utils/usage.py**

```python
from agent.state import CleaningState, TokenUsage

_INPUT_COST_PER_M = 3.0
_OUTPUT_COST_PER_M = 15.0


def accumulate(state: CleaningState, response) -> TokenUsage:
    meta = getattr(response, "usage_metadata", None) or {}
    prev = state.get("usage") or {"input_tokens": 0, "output_tokens": 0}
    return {
        "input_tokens":  prev["input_tokens"]  + meta.get("input_tokens", 0),
        "output_tokens": prev["output_tokens"] + meta.get("output_tokens", 0),
    }


def cost_usd(usage: TokenUsage) -> float:
    return (
        usage["input_tokens"]  / 1_000_000 * _INPUT_COST_PER_M +
        usage["output_tokens"] / 1_000_000 * _OUTPUT_COST_PER_M
    )
```

- [ ] **Step 7: 创建 data/sample_dirty.json**

```json
[
  {"name": "  跑步鞋 Nike Air Max  ", "price": "￥299", "category": "鞋子", "stock": "true", "sku": "NK-001"},
  {"name": "跑步鞋NikeAirMax", "price": "299.00", "category": "运动鞋", "stock": true, "sku": "NK-001B"},
  {"name": "Camping Tent Pro", "price": "$89.99", "category": "outdoor gear", "stock": "50", "sku": "CT-010"},
  {"name": "露营帐篷专业版", "price": "589", "category": "户外/露营", "stock": null, "sku": "CT-010-CN"},
  {"name": "健身手套", "price": "free", "category": "健身", "stock": "20", "sku": "GV-005"},
  {"name": "", "price": "199", "category": "运动", "stock": "10", "sku": "XX-000"},
  {"name": "瑜伽垫加厚", "price": "￥158.00", "category": "瑜伽", "stock": "35", "sku": "YG-002"},
  {"name": "Yoga Mat Premium", "price": "22.99", "category": "yoga & fitness", "stock": "35", "sku": "YG-002-EN"}
]
```

（含问题：重复商品、价格格式不统一、空名称、null 库存、分类不统一）

- [ ] **Step 8: commit**

```bash
git add . && git commit -m "feat: project setup, CleaningState, sample dirty data"
```

---

## Task 6：规则引擎清洗（无 LLM）

**Files:**
- Create: `agent-data-cleaning/agent/rule_cleaner.py`
- Create: `agent-data-cleaning/tests/test_rule_cleaner.py`

- [ ] **Step 1: 写失败测试**

创建 `tests/test_rule_cleaner.py`：

```python
def test_strips_whitespace_from_name():
    from agent.rule_cleaner import rule_clean_node
    state = {"raw_data": [{"name": "  跑步鞋  ", "price": "299", "category": "鞋", "stock": "10"}],
             "data_type": "products", "rule_cleaned": [], "issues": [],
             "ai_cleaned": [], "final_data": [], "report": {},
             "usage": {"input_tokens": 0, "output_tokens": 0}}
    result = rule_clean_node(state)
    assert result["rule_cleaned"][0]["name"] == "跑步鞋"


def test_normalizes_price_with_rmb_symbol():
    from agent.rule_cleaner import rule_clean_node
    state = {"raw_data": [{"name": "x", "price": "￥299.00", "category": "鞋", "stock": "5"}],
             "data_type": "products", "rule_cleaned": [], "issues": [],
             "ai_cleaned": [], "final_data": [], "report": {},
             "usage": {"input_tokens": 0, "output_tokens": 0}}
    result = rule_clean_node(state)
    assert result["rule_cleaned"][0]["price"] == 299.0


def test_normalizes_price_with_dollar_symbol():
    from agent.rule_cleaner import rule_clean_node
    state = {"raw_data": [{"name": "x", "price": "$89.99", "category": "outdoor", "stock": "10"}],
             "data_type": "products", "rule_cleaned": [], "issues": [],
             "ai_cleaned": [], "final_data": [], "report": {},
             "usage": {"input_tokens": 0, "output_tokens": 0}}
    result = rule_clean_node(state)
    assert result["rule_cleaned"][0]["price"] == 89.99


def test_normalizes_stock_boolean_string():
    from agent.rule_cleaner import rule_clean_node
    state = {"raw_data": [{"name": "x", "price": "10", "category": "y", "stock": "true"}],
             "data_type": "products", "rule_cleaned": [], "issues": [],
             "ai_cleaned": [], "final_data": [], "report": {},
             "usage": {"input_tokens": 0, "output_tokens": 0}}
    result = rule_clean_node(state)
    assert result["rule_cleaned"][0]["stock"] is True


def test_converts_null_stock_to_none():
    from agent.rule_cleaner import rule_clean_node
    state = {"raw_data": [{"name": "x", "price": "10", "category": "y", "stock": None}],
             "data_type": "products", "rule_cleaned": [], "issues": [],
             "ai_cleaned": [], "final_data": [], "report": {},
             "usage": {"input_tokens": 0, "output_tokens": 0}}
    result = rule_clean_node(state)
    assert result["rule_cleaned"][0]["stock"] is None


def test_non_numeric_price_kept_as_string():
    from agent.rule_cleaner import rule_clean_node
    state = {"raw_data": [{"name": "x", "price": "free", "category": "y", "stock": "5"}],
             "data_type": "products", "rule_cleaned": [], "issues": [],
             "ai_cleaned": [], "final_data": [], "report": {},
             "usage": {"input_tokens": 0, "output_tokens": 0}}
    result = rule_clean_node(state)
    assert result["rule_cleaned"][0]["price"] == "free"
```

- [ ] **Step 2: 运行，确认失败**

```bash
pytest tests/test_rule_cleaner.py -v
```

Expected: FAILED（ImportError）

- [ ] **Step 3: 创建 agent/rule_cleaner.py**

```python
import re
from agent.state import CleaningState


def _parse_price(value) -> float | str | None:
    if value is None:
        return None
    s = str(value).strip().replace("￥", "").replace("$", "").replace(",", "")
    try:
        return float(s)
    except ValueError:
        return s


def _parse_stock(value):
    if isinstance(value, bool):
        return value
    if value is None:
        return None
    s = str(value).strip().lower()
    if s in ("true", "yes", "1"):
        return True
    if s in ("false", "no", "0"):
        return False
    try:
        return int(s)
    except ValueError:
        return value


def _clean_record(record: dict) -> dict:
    cleaned = {}
    for k, v in record.items():
        if isinstance(v, str):
            v = v.strip()
            if v == "":
                v = None
        if k == "price":
            v = _parse_price(v)
        elif k == "stock":
            v = _parse_stock(v)
        cleaned[k] = v
    return cleaned


def rule_clean_node(state: CleaningState) -> CleaningState:
    cleaned = [_clean_record(r) for r in state["raw_data"]]
    return {**state, "rule_cleaned": cleaned}
```

- [ ] **Step 4: 运行，确认通过**

```bash
pytest tests/test_rule_cleaner.py -v
```

Expected: 6 PASSED

- [ ] **Step 5: commit**

```bash
git add agent/rule_cleaner.py tests/test_rule_cleaner.py
git commit -m "feat: rule-based cleaning node (whitespace, price, stock normalization)"
```

---

## Task 7：Detector Agent（LLM 语义检测）

**Files:**
- Create: `agent-data-cleaning/agent/detector.py`
- Create: `agent-data-cleaning/tests/test_detector.py`

- [ ] **Step 1: 写失败测试**

创建 `tests/test_detector.py`：

```python
from unittest.mock import patch, MagicMock
from agent.state import CleaningState


def make_state(rule_cleaned: list) -> CleaningState:
    return {
        "raw_data": rule_cleaned,
        "data_type": "products",
        "rule_cleaned": rule_cleaned,
        "issues": [],
        "ai_cleaned": [],
        "final_data": [],
        "report": {},
        "usage": {"input_tokens": 0, "output_tokens": 0},
    }


def test_detector_returns_issues_list():
    from agent.detector import detector_node
    mock_resp = MagicMock()
    mock_resp.content = '[{"record_index": 0, "field": "category", "issue": "inconsistent", "severity": "medium"}]'
    mock_resp.usage_metadata = {"input_tokens": 100, "output_tokens": 30}

    with patch("agent.detector.get_llm") as mock_llm:
        mock_llm.return_value.invoke.return_value = mock_resp
        state = make_state([{"name": "跑步鞋", "price": 299.0, "category": "鞋子", "stock": True}])
        result = detector_node(state)

    assert isinstance(result["issues"], list)
    assert len(result["issues"]) == 1
    assert result["issues"][0]["field"] == "category"


def test_detector_handles_invalid_json_gracefully():
    from agent.detector import detector_node
    mock_resp = MagicMock()
    mock_resp.content = "sorry I cannot help with that"
    mock_resp.usage_metadata = {"input_tokens": 50, "output_tokens": 10}

    with patch("agent.detector.get_llm") as mock_llm:
        mock_llm.return_value.invoke.return_value = mock_resp
        state = make_state([{"name": "test", "price": 10.0, "category": "test", "stock": 5}])
        result = detector_node(state)

    assert result["issues"] == []


def test_detector_accumulates_token_usage():
    from agent.detector import detector_node
    mock_resp = MagicMock()
    mock_resp.content = "[]"
    mock_resp.usage_metadata = {"input_tokens": 200, "output_tokens": 50}

    with patch("agent.detector.get_llm") as mock_llm:
        mock_llm.return_value.invoke.return_value = mock_resp
        state = make_state([])
        state["usage"] = {"input_tokens": 100, "output_tokens": 20}
        result = detector_node(state)

    assert result["usage"]["input_tokens"] == 300
    assert result["usage"]["output_tokens"] == 70
```

- [ ] **Step 2: 运行，确认失败**

```bash
pytest tests/test_detector.py -v
```

Expected: FAILED（ImportError）

- [ ] **Step 3: 创建 agent/detector.py**

```python
import json
from agent.state import CleaningState
from config import get_llm
from utils.usage import accumulate

_PROMPT = """你是一名电商平台的数据质量专家。
分析以下商品记录，识别数据质量问题。

重点检查：
1. 重复商品（名称相似但 SKU 不同）
2. 分类名称不统一（如"鞋子"和"运动鞋"可能是同一类）
3. 价格异常（为0、负数、或文字描述）
4. 必填字段缺失（name、price、category）
5. 中英文混用导致的重复（同一商品的中英文版本）

数据：
{data}

只返回 JSON 数组，不要有任何解释或 markdown：
[{{"record_index": 0, "field": "category", "issue": "具体问题描述", "severity": "high|medium|low"}}]

如果没有问题，返回空数组 []。"""


def detector_node(state: CleaningState) -> CleaningState:
    llm = get_llm()
    data_str = json.dumps(state["rule_cleaned"], ensure_ascii=False, indent=2)
    resp = llm.invoke(_PROMPT.format(data=data_str))
    usage = accumulate(state, resp)

    try:
        raw = resp.content.strip().strip("```json").strip("```").strip()
        issues = json.loads(raw)
        if not isinstance(issues, list):
            issues = []
    except (json.JSONDecodeError, ValueError):
        issues = []

    return {**state, "issues": issues, "usage": usage}
```

- [ ] **Step 4: 运行，确认通过**

```bash
pytest tests/test_detector.py -v
```

Expected: 3 PASSED

- [ ] **Step 5: commit**

```bash
git add agent/detector.py tests/test_detector.py
git commit -m "feat: LLM detector node for semantic issue detection"
```

---

## Task 8：Cleaner Agent + Validator

**Files:**
- Create: `agent-data-cleaning/agent/cleaner.py`
- Create: `agent-data-cleaning/agent/validator.py`
- Create: `agent-data-cleaning/tests/test_cleaner.py`
- Create: `agent-data-cleaning/tests/test_validator.py`

- [ ] **Step 1: 写失败测试 (cleaner)**

创建 `tests/test_cleaner.py`：

```python
from unittest.mock import patch, MagicMock
import json
from agent.state import CleaningState


def make_state(rule_cleaned: list, issues: list) -> CleaningState:
    return {
        "raw_data": rule_cleaned, "data_type": "products",
        "rule_cleaned": rule_cleaned, "issues": issues,
        "ai_cleaned": [], "final_data": [], "report": {},
        "usage": {"input_tokens": 0, "output_tokens": 0},
    }


def test_cleaner_skips_when_no_issues():
    from agent.cleaner import cleaner_node
    state = make_state([{"name": "跑步鞋", "price": 299.0, "category": "运动鞋", "stock": 10}], [])
    result = cleaner_node(state)
    assert result["ai_cleaned"] == result["rule_cleaned"]


def test_cleaner_returns_fixed_data():
    from agent.cleaner import cleaner_node
    cleaned = [{"name": "跑步鞋", "price": 299.0, "category": "鞋子", "stock": 10}]
    issues = [{"record_index": 0, "field": "category", "issue": "should be 运动鞋", "severity": "medium"}]
    fixed = [{"name": "跑步鞋", "price": 299.0, "category": "运动鞋", "stock": 10}]

    mock_resp = MagicMock()
    mock_resp.content = json.dumps(fixed, ensure_ascii=False)
    mock_resp.usage_metadata = {"input_tokens": 200, "output_tokens": 80}

    with patch("agent.cleaner.get_llm") as mock_llm:
        mock_llm.return_value.invoke.return_value = mock_resp
        result = cleaner_node(make_state(cleaned, issues))

    assert result["ai_cleaned"][0]["category"] == "运动鞋"
```

创建 `tests/test_validator.py`：

```python
from agent.state import CleaningState


def make_state(ai_cleaned: list) -> CleaningState:
    return {
        "raw_data": ai_cleaned, "data_type": "products",
        "rule_cleaned": ai_cleaned, "issues": [],
        "ai_cleaned": ai_cleaned, "final_data": [], "report": {},
        "usage": {"input_tokens": 0, "output_tokens": 0},
    }


def test_valid_record_passes():
    from agent.validator import validator_node
    state = make_state([{"name": "跑步鞋", "price": 299.0, "category": "运动鞋", "stock": 10}])
    result = validator_node(state)
    assert len(result["final_data"]) == 1
    assert result["report"]["valid_count"] == 1
    assert result["report"]["invalid_count"] == 0


def test_record_with_empty_name_is_invalid():
    from agent.validator import validator_node
    state = make_state([{"name": None, "price": 199.0, "category": "运动", "stock": 10}])
    result = validator_node(state)
    assert result["report"]["invalid_count"] == 1
    assert len(result["final_data"]) == 0


def test_record_with_zero_price_is_invalid():
    from agent.validator import validator_node
    state = make_state([{"name": "商品A", "price": 0.0, "category": "test", "stock": 5}])
    result = validator_node(state)
    assert result["report"]["invalid_count"] == 1


def test_report_contains_change_summary():
    from agent.validator import validator_node
    state = make_state([
        {"name": "商品A", "price": 100.0, "category": "运动", "stock": 5},
        {"name": None, "price": 50.0, "category": "运动", "stock": 3},
    ])
    result = validator_node(state)
    assert "valid_count" in result["report"]
    assert "invalid_count" in result["report"]
    assert "total_input" in result["report"]
```

- [ ] **Step 2: 运行，确认失败**

```bash
pytest tests/test_cleaner.py tests/test_validator.py -v
```

Expected: FAILED（ImportError）

- [ ] **Step 3: 创建 agent/cleaner.py**

```python
import json
from agent.state import CleaningState
from config import get_llm
from utils.usage import accumulate

_PROMPT = """你是一名电商数据清洗专家。
根据以下问题列表，修复商品数据中的错误。

原始数据：
{data}

发现的问题：
{issues}

要求：
1. 只修复问题列表中明确指出的问题
2. 保留所有原始字段
3. 将重复商品合并为一条记录（保留信息更完整的那条）
4. 统一分类名称（使用中文标准分类名）
5. 只返回 JSON 数组，不要有任何解释"""


def cleaner_node(state: CleaningState) -> CleaningState:
    if not state["issues"]:
        return {**state, "ai_cleaned": state["rule_cleaned"]}

    llm = get_llm()
    data_str = json.dumps(state["rule_cleaned"], ensure_ascii=False, indent=2)
    issues_str = json.dumps(state["issues"], ensure_ascii=False, indent=2)
    resp = llm.invoke(_PROMPT.format(data=data_str, issues=issues_str))
    usage = accumulate(state, resp)

    try:
        raw = resp.content.strip().strip("```json").strip("```").strip()
        ai_cleaned = json.loads(raw)
        if not isinstance(ai_cleaned, list):
            ai_cleaned = state["rule_cleaned"]
    except (json.JSONDecodeError, ValueError):
        ai_cleaned = state["rule_cleaned"]

    return {**state, "ai_cleaned": ai_cleaned, "usage": usage}
```

- [ ] **Step 4: 创建 agent/validator.py**

```python
from agent.state import CleaningState


def _is_valid(record: dict) -> tuple[bool, str]:
    name = record.get("name")
    if not name or str(name).strip() == "":
        return False, "name is empty"
    price = record.get("price")
    try:
        if float(price) <= 0:
            return False, f"price {price} is not positive"
    except (TypeError, ValueError):
        return False, f"price '{price}' is not numeric"
    if not record.get("category"):
        return False, "category is empty"
    return True, ""


def validator_node(state: CleaningState) -> CleaningState:
    source = state["ai_cleaned"] if state["ai_cleaned"] else state["rule_cleaned"]
    valid, invalid = [], []
    errors = []

    for i, record in enumerate(source):
        ok, reason = _is_valid(record)
        if ok:
            valid.append(record)
        else:
            invalid.append(record)
            errors.append({"record_index": i, "reason": reason})

    report = {
        "total_input": len(state["raw_data"]),
        "after_rule_clean": len(state["rule_cleaned"]),
        "issues_detected": len(state["issues"]),
        "valid_count": len(valid),
        "invalid_count": len(invalid),
        "validation_errors": errors,
    }
    return {**state, "final_data": valid, "report": report}
```

- [ ] **Step 5: 运行，确认通过**

```bash
pytest tests/test_cleaner.py tests/test_validator.py -v
```

Expected: 全部 PASSED（5 个）

- [ ] **Step 6: commit**

```bash
git add agent/cleaner.py agent/validator.py tests/test_cleaner.py tests/test_validator.py
git commit -m "feat: LLM cleaner node + rule-based validator node"
```

---

## Task 9：LangGraph 图组装 + FastAPI 入口

**Files:**
- Create: `agent-data-cleaning/agent/graph.py`
- Create: `agent-data-cleaning/utils/startup.py`
- Create: `agent-data-cleaning/main.py`
- Create: `agent-data-cleaning/tests/test_integration.py`

- [ ] **Step 1: 创建 agent/graph.py**

```python
from langgraph.graph import StateGraph, START, END
from agent.state import CleaningState
from agent.rule_cleaner import rule_clean_node
from agent.detector import detector_node
from agent.cleaner import cleaner_node
from agent.validator import validator_node


def build_graph():
    g = StateGraph(CleaningState)
    g.add_node("rule_clean", rule_clean_node)
    g.add_node("detect",     detector_node)
    g.add_node("clean",      cleaner_node)
    g.add_node("validate",   validator_node)

    g.add_edge(START,        "rule_clean")
    g.add_edge("rule_clean", "detect")
    g.add_edge("detect",     "clean")
    g.add_edge("clean",      "validate")
    g.add_edge("validate",   END)

    return g.compile()
```

- [ ] **Step 2: 创建 utils/startup.py**

```python
import psycopg2
from config import DATABASE_URL

_DDL = """
CREATE TABLE IF NOT EXISTS cleaning_jobs (
    id            SERIAL PRIMARY KEY,
    job_id        VARCHAR(36) UNIQUE NOT NULL DEFAULT gen_random_uuid()::text,
    data_type     VARCHAR(20) NOT NULL,
    input_count   INTEGER NOT NULL,
    output_count  INTEGER NOT NULL,
    issues_found  INTEGER DEFAULT 0,
    input_tokens  INTEGER DEFAULT 0,
    output_tokens INTEGER DEFAULT 0,
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
        print("[startup] agent-data-cleaning DB initialized.")
    finally:
        conn.close()


def log_job(data_type: str, input_count: int, output_count: int,
            issues_found: int, usage: dict):
    conn = psycopg2.connect(DATABASE_URL)
    try:
        with conn.cursor() as cur:
            cur.execute(
                "INSERT INTO cleaning_jobs "
                "(data_type, input_count, output_count, issues_found, input_tokens, output_tokens) "
                "VALUES (%s,%s,%s,%s,%s,%s)",
                (data_type, input_count, output_count, issues_found,
                 usage.get("input_tokens", 0), usage.get("output_tokens", 0)),
            )
        conn.commit()
    finally:
        conn.close()
```

- [ ] **Step 3: 创建 main.py**

```python
from contextlib import asynccontextmanager
from fastapi import FastAPI
from pydantic import BaseModel
from agent.graph import build_graph
from agent.state import CleaningState
from utils.startup import check_and_init, log_job

_graph = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global _graph
    check_and_init()
    _graph = build_graph()
    yield


app = FastAPI(title="ShopMind Data Cleaning Agent", version="1.0.0", lifespan=lifespan)


class InvokeRequest(BaseModel):
    data: list[dict]
    data_type: str = "products"


@app.post("/invoke")
def invoke(req: InvokeRequest):
    state: CleaningState = {
        "raw_data": req.data,
        "data_type": req.data_type,
        "rule_cleaned": [],
        "issues": [],
        "ai_cleaned": [],
        "final_data": [],
        "report": {},
        "usage": {"input_tokens": 0, "output_tokens": 0},
    }
    result = _graph.invoke(state)
    usage = result.get("usage", {"input_tokens": 0, "output_tokens": 0})
    log_job(
        data_type=req.data_type,
        input_count=len(req.data),
        output_count=len(result["final_data"]),
        issues_found=len(result.get("issues", [])),
        usage=usage,
    )
    return {
        "cleaned_data": result["final_data"],
        "report": result["report"],
        "usage": usage,
    }


@app.get("/health")
def health():
    return {"status": "ok", "service": "agent-data-cleaning"}
```

- [ ] **Step 4: 写集成测试**

创建 `tests/test_integration.py`：

```python
import json
import os
import pytest
from unittest.mock import patch, MagicMock
from fastapi.testclient import TestClient


@pytest.fixture
def client():
    with patch("utils.startup.check_and_init"), \
         patch("utils.startup.log_job"), \
         patch("agent.graph.build_graph") as mock_build:
        from agent.state import CleaningState
        from agent.rule_cleaner import rule_clean_node
        from agent.validator import validator_node

        def fake_graph_invoke(state):
            state = rule_clean_node(state)
            state = {**state, "issues": [], "ai_cleaned": state["rule_cleaned"]}
            state = validator_node(state)
            return state

        mock_graph = MagicMock()
        mock_graph.invoke.side_effect = fake_graph_invoke
        mock_build.return_value = mock_graph

        import importlib
        import main as m
        importlib.reload(m)
        yield TestClient(m.app)


def test_invoke_returns_cleaned_data_and_report(client):
    resp = client.post("/invoke", json={
        "data": [
            {"name": "  跑步鞋  ", "price": "￥299", "category": "运动鞋", "stock": "10"},
            {"name": "健身手套", "price": "59.0", "category": "健身", "stock": "20"},
        ],
        "data_type": "products",
    })
    assert resp.status_code == 200
    body = resp.json()
    assert "cleaned_data" in body
    assert "report" in body
    assert "usage" in body
    assert body["report"]["total_input"] == 2


def test_health_endpoint(client):
    resp = client.get("/health")
    assert resp.status_code == 200
    assert resp.json()["status"] == "ok"
```

- [ ] **Step 5: 运行全部测试**

```bash
pytest tests/ -v
```

Expected: 全部 PASSED（约 17 个）

- [ ] **Step 6: 手动验证**

```bash
uvicorn main:app --reload --port 8001
curl -X POST http://localhost:8001/invoke \
  -H "Content-Type: application/json" \
  -d @data/sample_dirty.json  # 需要包装一层：{"data": [...], "data_type": "products"}
```

- [ ] **Step 7: commit**

```bash
git add agent/graph.py utils/startup.py main.py tests/test_integration.py
git commit -m "feat: LangGraph graph + FastAPI /invoke endpoint + job logging"
```

---

## Task 10：shopmind-web Next.js 基础框架

**Files:**
- Create: `shopmind-web/` 整个目录（通过 create-next-app）

- [ ] **Step 1: 初始化 Next.js 项目**

```bash
npx create-next-app@latest shopmind-web \
  --typescript \
  --tailwind \
  --eslint \
  --app \
  --no-src-dir \
  --import-alias "@/*"
cd shopmind-web
```

- [ ] **Step 2: 安装 shadcn/ui**

```bash
npx shadcn@latest init
```

选择：
- Style: Default
- Base color: Slate
- CSS variables: Yes

然后安装需要的组件：

```bash
npx shadcn@latest add button input label card table badge textarea
```

- [ ] **Step 3: 创建环境变量文件**

创建 `.env.local.example`：

```
NEXT_PUBLIC_GATEWAY_URL=http://localhost:8000
```

复制为 `.env.local`：

```bash
cp .env.local.example .env.local
```

- [ ] **Step 4: 创建 lib/auth.ts**

```typescript
const TOKEN_KEY = "shopmind_access_token";

export function saveToken(token: string) {
  if (typeof window !== "undefined") {
    localStorage.setItem(TOKEN_KEY, token);
  }
}

export function getToken(): string | null {
  if (typeof window === "undefined") return null;
  return localStorage.getItem(TOKEN_KEY);
}

export function clearToken() {
  if (typeof window !== "undefined") {
    localStorage.removeItem(TOKEN_KEY);
  }
}

export function isLoggedIn(): boolean {
  return !!getToken();
}
```

- [ ] **Step 5: 创建 lib/api.ts**

```typescript
import { getToken } from "./auth";

const BASE = process.env.NEXT_PUBLIC_GATEWAY_URL ?? "http://localhost:8000";

async function request<T>(path: string, options: RequestInit = {}): Promise<T> {
  const token = getToken();
  const headers: Record<string, string> = {
    "Content-Type": "application/json",
    ...(options.headers as Record<string, string>),
  };
  if (token) headers["Authorization"] = `Bearer ${token}`;

  const res = await fetch(`${BASE}${path}`, { ...options, headers });
  if (!res.ok) {
    const err = await res.json().catch(() => ({ detail: res.statusText }));
    throw new Error(err.detail ?? "Request failed");
  }
  return res.json();
}

export const api = {
  login: (username: string, password: string) =>
    request<{ access_token: string; refresh_token: string }>("/auth/login", {
      method: "POST",
      body: JSON.stringify({ username, password }),
    }),

  invokeAgent: (agent: string, body: unknown) =>
    request(`/api/v1/${agent}/invoke`, {
      method: "POST",
      body: JSON.stringify(body),
    }),

  usageSummary: () =>
    request<{ by_agent: Array<{ agent_name: string; total_input: number; total_output: number }> }>(
      "/api/v1/usage/summary"
    ),
};
```

- [ ] **Step 6: 创建 middleware.ts（保护 /dashboard）**

```typescript
import { NextResponse } from "next/server";
import type { NextRequest } from "next/server";

export function middleware(request: NextRequest) {
  const token = request.cookies.get("shopmind_token")?.value;
  const isProtected = request.nextUrl.pathname.startsWith("/dashboard");

  if (isProtected && !token) {
    return NextResponse.redirect(new URL("/login", request.url));
  }
  return NextResponse.next();
}

export const config = {
  matcher: ["/dashboard/:path*"],
};
```

- [ ] **Step 7: 创建登录页 app/login/page.tsx**

```typescript
"use client";
import { useState } from "react";
import { useRouter } from "next/navigation";
import { api } from "@/lib/api";
import { saveToken } from "@/lib/auth";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";

export default function LoginPage() {
  const router = useRouter();
  const [username, setUsername] = useState("demo");
  const [password, setPassword] = useState("shopmind2026");
  const [error, setError] = useState("");
  const [loading, setLoading] = useState(false);

  async function handleSubmit(e: React.FormEvent) {
    e.preventDefault();
    setLoading(true);
    setError("");
    try {
      const { access_token } = await api.login(username, password);
      saveToken(access_token);
      document.cookie = `shopmind_token=${access_token}; path=/`;
      router.push("/dashboard");
    } catch (err: unknown) {
      setError(err instanceof Error ? err.message : "Login failed");
    } finally {
      setLoading(false);
    }
  }

  return (
    <div className="min-h-screen flex items-center justify-center bg-slate-50">
      <Card className="w-[380px]">
        <CardHeader>
          <CardTitle className="text-2xl text-center">ShopMind AI</CardTitle>
          <p className="text-center text-slate-500 text-sm">电商智能运营平台</p>
        </CardHeader>
        <CardContent>
          <form onSubmit={handleSubmit} className="space-y-4">
            <div className="space-y-2">
              <Label htmlFor="username">用户名</Label>
              <Input id="username" value={username} onChange={e => setUsername(e.target.value)} />
            </div>
            <div className="space-y-2">
              <Label htmlFor="password">密码</Label>
              <Input id="password" type="password" value={password} onChange={e => setPassword(e.target.value)} />
            </div>
            {error && <p className="text-red-500 text-sm">{error}</p>}
            <Button type="submit" className="w-full" disabled={loading}>
              {loading ? "登录中..." : "登录"}
            </Button>
          </form>
        </CardContent>
      </Card>
    </div>
  );
}
```

- [ ] **Step 8: 创建 dashboard layout app/dashboard/layout.tsx**

```typescript
"use client";
import Link from "next/link";
import { usePathname } from "next/navigation";
import { cn } from "@/lib/utils";

const navItems = [
  { href: "/dashboard", label: "概览", emoji: "📊" },
  { href: "/dashboard/cleaning", label: "数据清洗", emoji: "🧹" },
];

export default function DashboardLayout({ children }: { children: React.ReactNode }) {
  const pathname = usePathname();
  return (
    <div className="flex min-h-screen">
      <aside className="w-56 bg-slate-900 text-white flex flex-col">
        <div className="p-4 border-b border-slate-700">
          <h1 className="font-bold text-lg">ShopMind AI</h1>
          <p className="text-slate-400 text-xs">电商智能运营平台</p>
        </div>
        <nav className="flex-1 p-3 space-y-1">
          {navItems.map(item => (
            <Link
              key={item.href}
              href={item.href}
              className={cn(
                "flex items-center gap-2 px-3 py-2 rounded-md text-sm transition-colors",
                pathname === item.href
                  ? "bg-slate-700 text-white"
                  : "text-slate-300 hover:bg-slate-800"
              )}
            >
              <span>{item.emoji}</span>
              <span>{item.label}</span>
            </Link>
          ))}
        </nav>
      </aside>
      <main className="flex-1 bg-slate-50 p-6">{children}</main>
    </div>
  );
}
```

- [ ] **Step 9: 创建数据清洗页 app/dashboard/cleaning/page.tsx**

```typescript
"use client";
import { useState } from "react";
import { api } from "@/lib/api";
import { Button } from "@/components/ui/button";
import { Textarea } from "@/components/ui/textarea";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";

const SAMPLE = JSON.stringify([
  { name: "  跑步鞋 Nike  ", price: "￥299", category: "鞋子", stock: "true", sku: "NK-001" },
  { name: "Camping Tent", price: "$89.99", category: "outdoor gear", stock: "50", sku: "CT-010" },
  { name: "", price: "199", category: "运动", stock: "10", sku: "XX-000" },
], null, 2);

export default function CleaningPage() {
  const [input, setInput] = useState(SAMPLE);
  const [result, setResult] = useState<null | { cleaned_data: unknown[]; report: Record<string, unknown>; usage: { input_tokens: number; output_tokens: number } }>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");

  async function handleClean() {
    setLoading(true);
    setError("");
    try {
      const data = JSON.parse(input);
      const res = await api.invokeAgent("cleaning", { data, data_type: "products" }) as typeof result;
      setResult(res);
    } catch (e: unknown) {
      setError(e instanceof Error ? e.message : "Failed");
    } finally {
      setLoading(false);
    }
  }

  return (
    <div className="max-w-4xl space-y-6">
      <div>
        <h1 className="text-2xl font-bold">🧹 数据清洗中心</h1>
        <p className="text-slate-500 text-sm mt-1">粘贴原始商品数据（JSON 数组），AI 自动识别并修复质量问题</p>
      </div>

      <Card>
        <CardHeader><CardTitle>输入数据</CardTitle></CardHeader>
        <CardContent className="space-y-3">
          <Textarea
            value={input}
            onChange={e => setInput(e.target.value)}
            rows={12}
            className="font-mono text-sm"
          />
          <Button onClick={handleClean} disabled={loading}>
            {loading ? "清洗中..." : "开始清洗"}
          </Button>
          {error && <p className="text-red-500 text-sm">{error}</p>}
        </CardContent>
      </Card>

      {result && (
        <>
          <Card>
            <CardHeader><CardTitle>清洗报告</CardTitle></CardHeader>
            <CardContent>
              <div className="grid grid-cols-3 gap-4 text-center">
                <div><p className="text-2xl font-bold">{String(result.report.total_input)}</p><p className="text-slate-500 text-sm">原始记录</p></div>
                <div><p className="text-2xl font-bold text-green-600">{String(result.report.valid_count)}</p><p className="text-slate-500 text-sm">清洗通过</p></div>
                <div><p className="text-2xl font-bold text-red-500">{String(result.report.invalid_count)}</p><p className="text-slate-500 text-sm">无法修复</p></div>
              </div>
              <div className="mt-4 text-xs text-slate-400 text-right">
                Token 消耗：{result.usage.input_tokens + result.usage.output_tokens}
                （${((result.usage.input_tokens / 1e6 * 3) + (result.usage.output_tokens / 1e6 * 15)).toFixed(4)}）
              </div>
            </CardContent>
          </Card>

          <Card>
            <CardHeader><CardTitle>清洗后数据（{result.cleaned_data.length} 条）</CardTitle></CardHeader>
            <CardContent>
              <pre className="text-xs bg-slate-100 p-4 rounded overflow-auto max-h-64">
                {JSON.stringify(result.cleaned_data, null, 2)}
              </pre>
            </CardContent>
          </Card>
        </>
      )}
    </div>
  );
}
```

- [ ] **Step 10: 创建 app/page.tsx（重定向）**

```typescript
import { redirect } from "next/navigation";

export default function Home() {
  redirect("/dashboard");
}
```

- [ ] **Step 11: 创建 app/dashboard/page.tsx**

```typescript
export default function DashboardPage() {
  return (
    <div>
      <h1 className="text-2xl font-bold">概览</h1>
      <p className="text-slate-500 mt-2">欢迎使用 ShopMind AI 电商智能运营平台</p>
      <div className="mt-6 grid grid-cols-2 gap-4">
        <div className="border rounded-lg p-4 bg-white">
          <p className="text-slate-500 text-sm">已接入 Agent</p>
          <p className="text-3xl font-bold mt-1">1 / 8</p>
        </div>
        <div className="border rounded-lg p-4 bg-white">
          <p className="text-slate-500 text-sm">平台状态</p>
          <p className="text-3xl font-bold mt-1 text-green-600">运行中</p>
        </div>
      </div>
    </div>
  );
}
```

- [ ] **Step 12: 启动验证**

```bash
npm run dev
```

访问 http://localhost:3000，应跳转到 /login，登录后进入 dashboard。

- [ ] **Step 13: commit**

```bash
git add . && git commit -m "feat: Next.js frontend with login, dashboard layout, cleaning page"
```

---

## Task 11：端到端集成测试

**Files:**
- 无新文件，验证三服务联通

- [ ] **Step 1: 启动三个服务**

终端 1：
```bash
cd agent-data-cleaning && uvicorn main:app --port 8001
```

终端 2：
```bash
cd shopmind-gateway && uvicorn main:app --port 8000
```

终端 3：
```bash
cd shopmind-web && npm run dev
```

- [ ] **Step 2: 验证登录**

访问 http://localhost:3000/login，用 `demo` / `shopmind2026` 登录，确认跳转到 /dashboard。

- [ ] **Step 3: 验证数据清洗端到端**

1. 进入 /dashboard/cleaning
2. 保留默认样本数据，点击"开始清洗"
3. 确认返回：清洗报告 + 清洗后数据 + Token 消耗

- [ ] **Step 4: 验证 Gateway 记录了 usage**

```bash
curl -X GET http://localhost:8000/api/v1/usage/summary \
  -H "Authorization: Bearer <your-token>"
```

Expected: 返回包含 `cleaning` 的用量数据。

- [ ] **Step 5: 检查 PostgreSQL usage_logs 表**

在 Neon SQL Editor 执行：
```sql
SELECT * FROM usage_logs ORDER BY created_at DESC LIMIT 5;
```

Expected: 看到 `agent_name = 'cleaning'` 的记录。

- [ ] **Step 6: 各仓库 final commit**

```bash
# gateway
cd shopmind-gateway
git add . && git commit -m "chore: phase1 complete - gateway serving data-cleaning agent"

# data-cleaning
cd ../agent-data-cleaning
git add . && git commit -m "chore: phase1 complete - data cleaning agent fully integrated"

# web
cd ../shopmind-web
git add . && git commit -m "chore: phase1 complete - frontend integrated with gateway"
```

---

## 自审通过项

- ✅ 所有 spec 中的 Gateway 路由规则均已实现（`/api/v1/{agent}/invoke` + `/health`）
- ✅ JWT 认证覆盖所有 protected 路由
- ✅ 超时熔断：10s → degraded response
- ✅ Token 用量写入 `usage_logs` 表
- ✅ 数据清洗管道：rule_clean → detect → clean → validate
- ✅ LangGraph 状态流转与 state.py 定义一致
- ✅ 所有函数名在测试与实现中保持一致
- ✅ Next.js 前端覆盖：登录 + 仪表盘 + 清洗页
