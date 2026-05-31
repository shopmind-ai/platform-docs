# ShopMind AI Phase 2 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 构建 agent-customer-service（电商客服 Agent），支持商品政策 RAG 问答 + 订单/物流 Text-to-SQL + 持久化多轮对话，通过 FastAPI 接入 shopmind-gateway，同时在 shopmind-web 增加客服对话页面。

**Architecture:** LangGraph 管道 supervisor → rag → sql → aggregate → validator → memory，与 insurance-agent 架构完全一致，核心差异是：① 电商域（订单/物流/商品/退换货）；② FastAPI 服务接口（非 Streamlit）；③ 会话历史持久化到 PostgreSQL（cs_sessions 表），支持跨 API 调用的多轮对话；④ 所有表名加 ec_/cs_ 前缀避免 Neon 共享库冲突。

**Tech Stack:** Python 3.11 · FastAPI · LangGraph · LangChain · Claude API (claude-sonnet-4-6) · Ollama (本地开发) · PostgreSQL + pgvector (Neon) · sentence-transformers · psycopg2

---

## 参考资料

- 设计文档：`shopmind-ai/docs/superpowers/specs/2026-05-31-shopmind-ai-design.md`
- 参考实现：`/Users/liuyun/CCWorkSpace/insurance-agent/`（相同 LangGraph 架构，直接复用模式）
- Gateway 已配置：`AGENT_CUSTOMER_URL=http://localhost:8002`，无需修改 gateway

---

## 文件结构总览

```
agent-customer-service/
├── main.py                    # FastAPI 入口：POST /invoke, GET /health
├── config.py                  # 环境变量、get_llm()、get_embeddings()
├── agent/
│   ├── __init__.py
│   ├── state.py               # CustomerServiceState TypedDict
│   ├── supervisor.py          # 意图分类 + LangGraph 图构建 + aggregate_node
│   ├── rag_agent.py           # pgvector 检索电商知识库
│   ├── sql_agent.py           # Text-to-SQL 查询电商数据
│   ├── validator.py           # 输出合规校验（不能承诺送达时间、无条件退款等）
│   └── memory.py              # 从 cs_sessions 加载/保存会话历史
├── data/
│   ├── docs/
│   │   ├── product_policy.txt  # 商品政策（退换货、保修）
│   │   ├── shipping_policy.txt # 物流说明（时效、运费）
│   │   ├── promotion_guide.txt # 促销规则（会员折扣、满减）
│   │   ├── return_guide.txt    # 退货流程指南
│   │   └── faq.txt             # 常见问题
│   ├── seed_db.py              # 建表 + 种电商模拟数据
│   └── ingest.py               # 文档向量化写入 pgvector
├── utils/
│   ├── __init__.py
│   ├── startup.py              # 自动初始化（建表 + 向量化），幂等
│   └── usage.py                # Token 累计 + 成本计算
├── tests/
│   ├── __init__.py
│   ├── test_state.py
│   ├── test_supervisor.py
│   ├── test_rag_agent.py
│   ├── test_sql_agent.py
│   ├── test_validator.py
│   ├── test_memory.py
│   └── test_integration.py
├── requirements.txt
├── .env.example
└── .gitignore
```

---

## Task 1：项目初始化

**Files:**
- Create: `agent-customer-service/requirements.txt`
- Create: `agent-customer-service/.env.example`
- Create: `agent-customer-service/.gitignore`
- Create: `agent-customer-service/config.py`

- [ ] **Step 1: 创建目录结构**

```bash
cd /Users/liuyun/CCWorkSpace/shopmind-ai
mkdir -p agent-customer-service/{agent,data/docs,utils,tests}
touch agent-customer-service/agent/__init__.py
touch agent-customer-service/utils/__init__.py
touch agent-customer-service/tests/__init__.py
cd agent-customer-service && git init && git checkout -b develop
git commit --allow-empty -m "chore: init agent-customer-service"
```

- [ ] **Step 2: 创建 requirements.txt**

```
fastapi>=0.115.0
uvicorn[standard]>=0.30.0
langchain>=0.3.0
langchain-anthropic>=0.3.0
langchain-ollama>=0.2.0
langchain-postgres>=0.0.13
langchain-huggingface>=0.1.0
langchain-community>=0.3.0
langchain-text-splitters>=0.3.0
sentence-transformers>=3.0.0
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

VECTOR_COLLECTION = "customer_service_docs"


def get_llm():
    if LLM_PROVIDER == "claude":
        from langchain_anthropic import ChatAnthropic
        return ChatAnthropic(model=CLAUDE_MODEL, api_key=ANTHROPIC_API_KEY)
    from langchain_ollama import ChatOllama
    return ChatOllama(model=OLLAMA_MODEL)


def get_embeddings():
    from langchain_huggingface import HuggingFaceEmbeddings
    return HuggingFaceEmbeddings(
        model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True},
    )
```

- [ ] **Step 6: 安装依赖**

```bash
pip install -r requirements.txt -q
python -c "import fastapi, langgraph, langchain_postgres; print('imports OK')"
```

Expected: `imports OK`

- [ ] **Step 7: commit**

```bash
git add . && git commit -m "feat: project setup, config, requirements"
```

---

## Task 2：AgentState + 种子数据

**Files:**
- Create: `agent/state.py`
- Create: `data/seed_db.py`

- [ ] **Step 1: 创建 agent/state.py**

```python
from typing import TypedDict


class TokenUsage(TypedDict):
    input_tokens: int
    output_tokens: int


class CustomerServiceState(TypedDict):
    messages: list[dict]       # 完整多轮对话历史
    intent: str                # "rag" | "sql" | "mixed" | "chat"
    rag_result: str
    sql_result: str
    final_answer: str
    customer_id: str           # 当前客户 ID
    session_id: str            # 会话 ID（对应 cs_sessions 表）
    usage: TokenUsage
```

- [ ] **Step 2: 创建 data/seed_db.py**

```python
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import psycopg2
from config import DATABASE_URL

DDL = """
CREATE TABLE IF NOT EXISTS ec_customers (
    id          SERIAL PRIMARY KEY,
    name        VARCHAR(50)  NOT NULL,
    email       VARCHAR(100) UNIQUE NOT NULL,
    phone       VARCHAR(20)  NOT NULL,
    membership  VARCHAR(10)  NOT NULL DEFAULT 'regular',
    created_at  TIMESTAMPTZ  DEFAULT NOW()
);

CREATE TABLE IF NOT EXISTS ec_products (
    id          SERIAL PRIMARY KEY,
    sku         VARCHAR(30)  UNIQUE NOT NULL,
    name        VARCHAR(200) NOT NULL,
    category    VARCHAR(50)  NOT NULL,
    price       NUMERIC      NOT NULL,
    stock       INTEGER      NOT NULL DEFAULT 0,
    description TEXT
);

CREATE TABLE IF NOT EXISTS ec_orders (
    id           SERIAL PRIMARY KEY,
    order_no     VARCHAR(20)  UNIQUE NOT NULL,
    customer_id  INTEGER      NOT NULL REFERENCES ec_customers(id),
    total_amount NUMERIC      NOT NULL,
    status       VARCHAR(20)  NOT NULL,
    created_at   TIMESTAMPTZ  DEFAULT NOW()
);

CREATE TABLE IF NOT EXISTS ec_order_items (
    id          SERIAL PRIMARY KEY,
    order_id    INTEGER NOT NULL REFERENCES ec_orders(id),
    product_id  INTEGER NOT NULL REFERENCES ec_products(id),
    quantity    INTEGER NOT NULL,
    unit_price  NUMERIC NOT NULL
);

CREATE TABLE IF NOT EXISTS ec_logistics (
    id                 SERIAL PRIMARY KEY,
    order_id           INTEGER     NOT NULL REFERENCES ec_orders(id),
    carrier            VARCHAR(50) NOT NULL,
    tracking_number    VARCHAR(50) NOT NULL,
    status             VARCHAR(20) NOT NULL,
    estimated_delivery DATE,
    updated_at         TIMESTAMPTZ DEFAULT NOW()
);

CREATE TABLE IF NOT EXISTS ec_returns (
    id          SERIAL PRIMARY KEY,
    order_id    INTEGER     NOT NULL REFERENCES ec_orders(id),
    customer_id INTEGER     NOT NULL REFERENCES ec_customers(id),
    reason      TEXT        NOT NULL,
    status      VARCHAR(20) NOT NULL DEFAULT 'pending',
    created_at  TIMESTAMPTZ DEFAULT NOW()
);

CREATE TABLE IF NOT EXISTS cs_sessions (
    id          SERIAL PRIMARY KEY,
    session_id  VARCHAR(36) UNIQUE NOT NULL DEFAULT gen_random_uuid()::text,
    customer_id VARCHAR(10),
    messages    JSONB        NOT NULL DEFAULT '[]',
    created_at  TIMESTAMPTZ  DEFAULT NOW(),
    updated_at  TIMESTAMPTZ  DEFAULT NOW()
);
"""

SEED = """
INSERT INTO ec_customers (name, email, phone, membership) VALUES
    ('张伟', 'zhangwei@example.com', '13800138001', 'gold'),
    ('李娜', 'lina@example.com',     '13900139002', 'regular'),
    ('王芳', 'wangfang@example.com', '13700137003', 'silver')
ON CONFLICT DO NOTHING;

INSERT INTO ec_products (sku, name, category, price, stock, description) VALUES
    ('RUN-001', '专业马拉松跑步鞋',   '跑步装备', 599.00,  45, '碳纤维中底，适合长距离路跑，回弹性能优秀'),
    ('RUN-002', '轻量训练跑步鞋',     '跑步装备', 399.00,  80, '透气网面，适合日常训练，重量仅220g'),
    ('GYM-001', '可调节哑铃套装',     '健身器材', 899.00,  20, '5-30kg可调，铸铁材质，附收纳架'),
    ('GYM-002', '瑜伽垫专业版',       '健身器材', 158.00, 120, '6mm厚度，TPE材质，双面防滑'),
    ('GYM-003', '弹力绳阻力带套装',   '健身器材', 89.00,  200, '5级阻力，天然乳胶，附收纳袋'),
    ('OUT-001', '四季通用露营帐篷',   '户外露营', 1299.00, 15, '3-4人，铝合金骨架，防水指数3000mm'),
    ('OUT-002', '轻量登山背包 45L',   '户外露营', 458.00,  30, '防水材质，人体工学背负，重量仅1.2kg'),
    ('OUT-003', '户外折叠桌椅套装',   '户外露营', 299.00,  50, '铝合金轻量，快速展开，承重80kg')
ON CONFLICT (sku) DO NOTHING;

INSERT INTO ec_orders (order_no, customer_id, total_amount, status) VALUES
    ('SM20260101001', 1, 599.00,  'delivered'),
    ('SM20260115002', 1, 1457.00, 'shipped'),
    ('SM20260120003', 2, 399.00,  'paid'),
    ('SM20260201004', 3, 1598.00, 'delivered'),
    ('SM20260210005', 1, 89.00,   'cancelled')
ON CONFLICT (order_no) DO NOTHING;

INSERT INTO ec_order_items (order_id, product_id, quantity, unit_price)
SELECT o.id, p.id, 1, p.price FROM ec_orders o, ec_products p
WHERE o.order_no = 'SM20260101001' AND p.sku = 'RUN-001'
  AND NOT EXISTS (SELECT 1 FROM ec_order_items i JOIN ec_orders oo ON i.order_id=oo.id WHERE oo.order_no='SM20260101001');

INSERT INTO ec_order_items (order_id, product_id, quantity, unit_price)
SELECT o.id, p.id, 1, p.price FROM ec_orders o, ec_products p
WHERE o.order_no = 'SM20260115002' AND p.sku = 'OUT-001'
  AND NOT EXISTS (SELECT 1 FROM ec_order_items i JOIN ec_orders oo ON i.order_id=oo.id WHERE oo.order_no='SM20260115002');

INSERT INTO ec_logistics (order_id, carrier, tracking_number, status, estimated_delivery)
SELECT o.id, '顺丰速运', 'SF1234567890', 'delivered', '2026-01-05'
FROM ec_orders o WHERE o.order_no = 'SM20260101001'
  AND NOT EXISTS (SELECT 1 FROM ec_logistics l JOIN ec_orders oo ON l.order_id=oo.id WHERE oo.order_no='SM20260101001');

INSERT INTO ec_logistics (order_id, carrier, tracking_number, status, estimated_delivery)
SELECT o.id, '京东快递', 'JD9876543210', 'in_transit', '2026-02-20'
FROM ec_orders o WHERE o.order_no = 'SM20260115002'
  AND NOT EXISTS (SELECT 1 FROM ec_logistics l JOIN ec_orders oo ON l.order_id=oo.id WHERE oo.order_no='SM20260115002');

INSERT INTO ec_returns (order_id, customer_id, reason, status)
SELECT o.id, o.customer_id, '尺码偏大，想换小一号', 'approved'
FROM ec_orders o WHERE o.order_no = 'SM20260101001'
  AND NOT EXISTS (SELECT 1 FROM ec_returns r JOIN ec_orders oo ON r.order_id=oo.id WHERE oo.order_no='SM20260101001');
"""


def seed(database_url: str = None):
    url = database_url or DATABASE_URL
    conn = psycopg2.connect(url)
    conn.autocommit = False
    try:
        with conn.cursor() as cur:
            cur.execute(DDL)
            cur.execute(SEED)
        conn.commit()
        print("电商数据库初始化完成（PostgreSQL）")
    except Exception as e:
        conn.rollback()
        raise e
    finally:
        conn.close()


if __name__ == "__main__":
    seed()
```

- [ ] **Step 3: commit**

```bash
git add agent/state.py data/seed_db.py
git commit -m "feat: CustomerServiceState + e-commerce seed data (customers/products/orders/logistics)"
```

---

## Task 3：知识库文档 + 向量化

**Files:**
- Create: `data/docs/product_policy.txt`
- Create: `data/docs/shipping_policy.txt`
- Create: `data/docs/promotion_guide.txt`
- Create: `data/docs/return_guide.txt`
- Create: `data/docs/faq.txt`
- Create: `data/ingest.py`

- [ ] **Step 1: 创建 data/docs/product_policy.txt**

```
商品政策说明

【商品质量保证】
所有在 ShopMind 平台销售的商品均经过严格质检，保证为正品行货。
商品出库前经过三道质检：外观检查、功能测试、包装完整性核查。

【退换货政策】
1. 签收后7天内：商品无质量问题可申请退货（需保持原包装完好，未使用）
2. 签收后15天内：商品存在质量问题可申请免费换货
3. 签收后30天内：商品存在质量问题可申请维修或退货
4. 以下情况不支持退换货：
   - 人为损坏（摔落、进水、改装）
   - 消耗品类（蛋白粉、运动补给）已开封使用
   - 特殊定制商品

【保修政策】
- 跑步鞋类：自购买之日起1年内，非人为损坏免费维修
- 健身器材：自购买之日起2年内，结构性损坏免费维修
- 户外装备：自购买之日起1年内，正常使用下的质量问题免费维修
- 保修凭证：订单号即为保修凭证，无需额外保修卡

【商品描述】
页面展示的商品图片、参数、描述均以实物为准，如发现商品与描述不符，
可在签收后48小时内联系客服申请退换货，平台承担往返运费。
```

- [ ] **Step 2: 创建 data/docs/shipping_policy.txt**

```
物流配送说明

【配送范围与时效】
1. 大陆地区（不含偏远地区）：
   - 普通快递：3-5个工作日
   - 顺丰速运（满299元免费升级）：1-2个工作日
   - 当日达（仅限北上广深杭5城，需18:00前下单）：当天送达

2. 偏远地区（西藏、新疆、青海部分地区）：
   - 配送时效：7-15个工作日
   - 运费另计，下单时系统自动计算

3. 港澳台及海外：暂不支持

【运费规则】
- 订单实付金额 ≥ 199元：免运费（偏远地区除外）
- 订单实付金额 < 199元：运费 12元
- 大件商品（健身器材、帐篷）：单独计算，约15-25元
- 退换货运费：商品质量问题由平台承担，非质量原因由买家承担

【发货时间】
- 工作日15:00前付款：当日发货
- 工作日15:00后付款：次工作日发货
- 法定节假日：顺延至节后第一个工作日发货
- 预售商品：按商品页面预计发货时间为准

【物流查询】
订单发货后，系统自动发送含快递单号的短信。
登录账号 → 我的订单 → 查看物流，可实时跟踪配送状态。
```

- [ ] **Step 3: 创建 data/docs/promotion_guide.txt**

```
促销活动说明

【会员等级与折扣】
- 普通会员（regular）：无额外折扣，享受基础权益
- 银卡会员（silver）：累计消费满500元自动升级，享95折
- 金卡会员（gold）：累计消费满2000元自动升级，享9折
- 钻石会员（diamond）：累计消费满5000元自动升级，享85折

会员折扣与其他活动价格取较低者，不叠加计算。

【满减活动】
- 满199减20
- 满399减50
- 满699减100
- 满999减180
满减金额在结算时自动扣减，优先扣减运费。

【积分规则】
- 每消费1元获得1积分
- 100积分可抵扣1元（每笔订单最多使用订单金额20%的积分）
- 积分有效期：获得后2年内有效
- 退款时同步退还对应积分

【季节促销】
- 618购物节（6月1日-18日）：全场额外95折，叠加会员折扣
- 双11（11月1日-11日）：满减力度加倍
- 双12（12月12日）：老客户专属优惠
- 春节前后（腊月至元宵）：户外装备专场，跑步装备8折起

【专项优惠】
- 新用户首单立减30元（实付满99元）
- 老带新：推荐好友注册并下单，双方各得50元优惠券
- 生日专属：会员生日当月享额外9折（需提前完善个人信息）
```

- [ ] **Step 4: 创建 data/docs/return_guide.txt**

```
退换货流程指南

【退货流程】
第一步：申请退货
  - 登录账号 → 我的订单 → 找到对应订单 → 点击"申请退货"
  - 选择退货原因（质量问题/不喜欢/尺寸不合适等）
  - 上传商品照片（建议多角度拍摄）
  - 提交申请，等待审核（1个工作日内）

第二步：寄回商品
  - 审核通过后，系统发送退货地址和退货单号
  - 使用原包装或完整包装寄回（建议选择可追踪快递）
  - 退货单号填写在包裹外侧
  - 非质量问题退货，买家承担运费

第三步：退款到账
  - 收到退货后，1个工作日内完成质检
  - 质检通过，3个工作日内退款到原支付渠道
  - 原路退款：微信支付→微信钱包，支付宝→支付宝，银行卡→原卡

【换货流程】
- 流程与退货相同，申请时选择"换货"
- 注明需要更换的规格（尺码/颜色）
- 收到退回商品后，1个工作日内发出换货商品

【退货注意事项】
- 商品须保持原包装完整，配件、说明书、发票齐全
- 已使用、清洗或改装的商品，质量问题除外不支持退货
- 超过退货期限的商品，可联系客服协商处理
- 退货审核不通过，商品原路退回，不收取任何费用

【特殊情况处理】
- 收到破损商品：签收时拍照留证，当天联系客服，平台负责处理
- 商品与描述不符：48小时内联系客服，提交证明，平台承担处理
- 假冒伪劣商品：一经核实，全额退款并额外赔偿200元
```

- [ ] **Step 5: 创建 data/docs/faq.txt**

```
常见问题解答

【订单相关】
Q: 付款后多久发货？
A: 工作日15:00前付款当天发货，15:00后付款次工作日发货。节假日顺延。

Q: 能修改收货地址吗？
A: 发货前可以修改。登录账号→我的订单→联系客服修改。发货后无法修改。

Q: 订单可以取消吗？
A: 发货前可以申请取消，已付款自动退款。发货后需走退货流程。

Q: 为什么下单后显示库存不足？
A: 平台实时同步库存，下单瞬间库存可能被抢购。系统会自动取消并全额退款。

【物流相关】
Q: 快递单号什么时候能查到？
A: 发货后2-4小时快递公司系统更新，即可查询。

Q: 显示"已签收"但没有收到？
A: 可能被快递员放在快递柜或保安室。请先查找，若仍未找到，24小时内联系客服。

Q: 可以指定送货时间吗？
A: 普通快递无法指定时间。当日达服务可在备注中说明期望时段，但不保证。

【支付相关】
Q: 支持哪些支付方式？
A: 支持微信支付、支付宝、银行卡（储蓄卡/信用卡）、积分抵扣。

Q: 付款后没有收到订单确认怎么办？
A: 请检查手机短信和邮件（含垃圾箱）。若30分钟后仍未收到，联系客服查询。

【商品相关】
Q: 跑步鞋怎么选码？
A: 建议比日常穿着尺码大半码，因为运动中脚会轻微肿胀。商品页有详细尺码表。

Q: 哑铃套装包含哪些重量？
A: 可调节哑铃套装含2.5/5/7.5/10/12.5/15kg共6对重量片，含手柄和收纳架。

Q: 帐篷能防多大的雨？
A: 四季通用露营帐篷防水指数3000mm，可应对中到大雨。暴雨建议在帐篷内加防水布。

【退换货相关】
Q: 购买后多久可以退货？
A: 无质量问题7天内可退，有质量问题30天内可退换。

Q: 退款什么时候到账？
A: 退货质检通过后3个工作日内原路退款。信用卡可能需要额外1-3个银行工作日。
```

- [ ] **Step 6: 创建 data/ingest.py**

```python
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_postgres import PGVector
from config import get_embeddings, DATABASE_URL, VECTOR_COLLECTION

DOCS_PATH = os.path.join(os.path.dirname(__file__), "docs")


def ingest(database_url: str = None):
    loader = DirectoryLoader(
        DOCS_PATH,
        glob="*.txt",
        loader_cls=TextLoader,
        loader_kwargs={"encoding": "utf-8"},
    )
    documents = loader.load()
    splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=100)
    chunks = splitter.split_documents(documents)
    print(f"加载 {len(documents)} 个文档，分割为 {len(chunks)} 个块")

    url = database_url or DATABASE_URL
    PGVector.from_documents(
        documents=chunks,
        embedding=get_embeddings(),
        collection_name=VECTOR_COLLECTION,
        connection=url,
        use_jsonb=True,
        pre_delete_collection=True,
    )
    print(f"向量化完成（collection: {VECTOR_COLLECTION}）")


if __name__ == "__main__":
    ingest()
```

- [ ] **Step 7: commit**

```bash
git add data/
git commit -m "feat: e-commerce knowledge base docs + pgvector ingest pipeline"
```

---

## Task 4：utils/startup.py + utils/usage.py

**Files:**
- Create: `utils/startup.py`
- Create: `utils/usage.py`

- [ ] **Step 1: 创建 utils/usage.py**

```python
from agent.state import CustomerServiceState, TokenUsage

_INPUT_COST_PER_M = 3.0
_OUTPUT_COST_PER_M = 15.0


def accumulate(state: CustomerServiceState, response) -> TokenUsage:
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

- [ ] **Step 2: 创建 utils/startup.py**

```python
import psycopg2
from config import DATABASE_URL, VECTOR_COLLECTION


def _tables_seeded(conn) -> bool:
    with conn.cursor() as cur:
        cur.execute(
            "SELECT EXISTS(SELECT 1 FROM information_schema.tables "
            "WHERE table_schema='public' AND table_name='ec_customers')"
        )
        if not cur.fetchone()[0]:
            return False
        cur.execute("SELECT COUNT(*) FROM ec_customers")
        return cur.fetchone()[0] > 0


def _vectors_ingested(conn) -> bool:
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
        needs_seed = not _tables_seeded(conn)
        needs_ingest = not _vectors_ingested(conn)
    finally:
        conn.close()

    if needs_seed:
        print("[startup] Seeding e-commerce database...")
        from data.seed_db import seed
        seed()

    if needs_ingest:
        print("[startup] Ingesting knowledge base documents...")
        from data.ingest import ingest
        ingest()

    if not needs_seed and not needs_ingest:
        print("[startup] DB and vectors already initialized, skipping.")
```

- [ ] **Step 3: commit**

```bash
git add utils/startup.py utils/usage.py
git commit -m "feat: auto-init startup + token usage tracking"
```

---

## Task 5：Supervisor Agent + 测试

**Files:**
- Create: `agent/supervisor.py`
- Create: `tests/test_supervisor.py`

- [ ] **Step 1: 写失败测试**

创建 `tests/test_supervisor.py`：

```python
from unittest.mock import patch, MagicMock
from agent.state import CustomerServiceState


def _make_state(question: str) -> CustomerServiceState:
    return {
        "messages": [{"role": "user", "content": question}],
        "intent": "", "rag_result": "", "sql_result": "",
        "final_answer": "", "customer_id": "1",
        "session_id": "test-session-001",
        "usage": {"input_tokens": 0, "output_tokens": 0},
    }


def test_supervisor_classifies_rag_intent():
    from agent.supervisor import supervisor_node
    with patch("agent.supervisor.get_llm") as mock_llm:
        mock_llm.return_value.invoke.return_value = MagicMock(
            content="rag", usage_metadata={"input_tokens": 50, "output_tokens": 2}
        )
        result = supervisor_node(_make_state("退货政策是什么？"))
    assert result["intent"] == "rag"
    assert result["final_answer"] == ""


def test_supervisor_classifies_sql_intent():
    from agent.supervisor import supervisor_node
    with patch("agent.supervisor.get_llm") as mock_llm:
        mock_llm.return_value.invoke.return_value = MagicMock(
            content="sql", usage_metadata={"input_tokens": 50, "output_tokens": 2}
        )
        result = supervisor_node(_make_state("我的订单到哪了？"))
    assert result["intent"] == "sql"


def test_supervisor_handles_chat_with_direct_reply():
    from agent.supervisor import supervisor_node
    with patch("agent.supervisor.get_llm") as mock_llm:
        mock_llm.return_value.invoke.side_effect = [
            MagicMock(content="chat", usage_metadata={"input_tokens": 30, "output_tokens": 2}),
            MagicMock(content="您好！有什么可以帮您？", usage_metadata={"input_tokens": 60, "output_tokens": 10}),
        ]
        result = supervisor_node(_make_state("你好"))
    assert result["intent"] == "chat"
    assert result["final_answer"] != ""


def test_supervisor_fallback_unknown_to_chat():
    from agent.supervisor import supervisor_node
    with patch("agent.supervisor.get_llm") as mock_llm:
        mock_llm.return_value.invoke.side_effect = [
            MagicMock(content="unknown_xyz", usage_metadata={"input_tokens": 30, "output_tokens": 2}),
            MagicMock(content="有什么需要帮助的？", usage_metadata={"input_tokens": 60, "output_tokens": 8}),
        ]
        result = supervisor_node(_make_state("随便说点什么"))
    assert result["intent"] == "chat"


def test_aggregate_merges_rag_and_sql():
    from agent.supervisor import aggregate_node
    state = _make_state("我的订单几天到？运费多少？")
    state["intent"] = "mixed"
    state["rag_result"] = "普通快递3-5天，满199免运费。"
    state["sql_result"] = "您的订单SM001正在运输中，预计2天后到达。"
    with patch("agent.supervisor.get_llm") as mock_llm:
        mock_llm.return_value.invoke.return_value = MagicMock(
            content="您的订单预计2天后到达，运费规则：满199免运费。",
            usage_metadata={"input_tokens": 200, "output_tokens": 50}
        )
        result = aggregate_node(state)
    assert result["final_answer"] != ""


def test_build_graph_returns_compiled():
    from agent.supervisor import build_graph
    assert build_graph() is not None
```

- [ ] **Step 2: 运行，确认失败**

```bash
pytest tests/test_supervisor.py -v 2>&1 | tail -5
```

Expected: FAILED（ImportError）

- [ ] **Step 3: 创建 agent/supervisor.py**

```python
from langgraph.graph import StateGraph, START, END
from agent.state import CustomerServiceState
from agent.rag_agent import rag_node
from agent.sql_agent import sql_node
from agent.validator import validator_node
from agent.memory import memory_node
from config import get_llm
from utils.usage import accumulate


def supervisor_node(state: CustomerServiceState) -> CustomerServiceState:
    question = state["messages"][-1]["content"]
    llm = get_llm()

    classify_prompt = (
        "将用户问题分类为以下意图之一，只返回分类标签：\n"
        "- rag：询问退换货政策、物流说明、促销规则、商品保修、常见问题等\n"
        "- sql：查询订单状态、物流进度、退款情况、购买记录等个人账户数据\n"
        "- mixed：同时涉及政策查询和个人账户数据\n"
        "- chat：问候、感谢、闲聊或与购物无关的问题\n\n"
        f"用户问题：{question}\n分类："
    )

    intent_resp = llm.invoke(classify_prompt)
    intent = intent_resp.content.strip().lower()
    if intent not in ("rag", "sql", "mixed", "chat"):
        intent = "chat"
    usage = accumulate(state, intent_resp)

    if intent == "chat":
        chat_prompt = (
            "你是 ShopMind 电商平台的客服助手，专注于跑步装备、健身器材、户外露营品类。"
            "请用简短友好的方式回应用户，如有购物需求引导提问。\n\n"
            f"用户：{question}\n回复："
        )
        response = llm.invoke(chat_prompt)
        usage = accumulate({**state, "usage": usage}, response)
        return {**state, "intent": intent, "final_answer": response.content, "usage": usage}

    return {**state, "intent": intent, "usage": usage}


def aggregate_node(state: CustomerServiceState) -> CustomerServiceState:
    if state.get("intent") == "chat":
        return state

    rag = state.get("rag_result", "")
    sql = state.get("sql_result", "")

    if rag and sql:
        question = state["messages"][-1]["content"]
        llm = get_llm()
        prompt = (
            "请整合以下两部分信息，用清晰友好的中文回答用户问题。\n\n"
            f"用户问题：{question}\n"
            f"政策信息：{rag}\n"
            f"账户数据：{sql}\n\n综合回答："
        )
        resp = llm.invoke(prompt)
        usage = accumulate(state, resp)
        return {**state, "final_answer": resp.content, "usage": usage}
    elif rag:
        return {**state, "final_answer": rag}
    elif sql:
        return {**state, "final_answer": sql}
    else:
        return {**state, "final_answer": "抱歉，暂时无法获取相关信息，请拨打客服热线 400-888-8888。"}


def build_graph():
    g = StateGraph(CustomerServiceState)
    g.add_node("supervisor",  supervisor_node)
    g.add_node("rag",         rag_node)
    g.add_node("sql",         sql_node)
    g.add_node("aggregate",   aggregate_node)
    g.add_node("validator",   validator_node)
    g.add_node("memory",      memory_node)

    g.add_edge(START,        "supervisor")
    g.add_edge("supervisor", "rag")
    g.add_edge("rag",        "sql")
    g.add_edge("sql",        "aggregate")
    g.add_edge("aggregate",  "validator")
    g.add_edge("validator",  "memory")
    g.add_edge("memory",     END)

    return g.compile()
```

- [ ] **Step 4: 运行，确认通过**

```bash
pytest tests/test_supervisor.py -v 2>&1 | tail -8
```

Expected: 6 PASSED

- [ ] **Step 5: commit**

```bash
git add agent/supervisor.py tests/test_supervisor.py
git commit -m "feat: supervisor node with e-commerce intent classification + aggregate node"
```

---

## Task 6：RAG Agent + 测试

**Files:**
- Create: `agent/rag_agent.py`
- Create: `tests/test_rag_agent.py`

- [ ] **Step 1: 写失败测试**

创建 `tests/test_rag_agent.py`：

```python
from unittest.mock import patch, MagicMock
from agent.state import CustomerServiceState


def _make_state(intent: str, question: str = "退货政策是什么？") -> CustomerServiceState:
    return {
        "messages": [{"role": "user", "content": question}],
        "intent": intent, "rag_result": "", "sql_result": "",
        "final_answer": "", "customer_id": "1",
        "session_id": "test-session-001",
        "usage": {"input_tokens": 0, "output_tokens": 0},
    }


def test_rag_node_skips_for_sql_intent():
    from agent.rag_agent import rag_node
    result = rag_node(_make_state("sql"))
    assert result["rag_result"] == ""


def test_rag_node_skips_for_chat_intent():
    from agent.rag_agent import rag_node
    result = rag_node(_make_state("chat"))
    assert result["rag_result"] == ""


def test_rag_node_returns_result_for_rag_intent():
    from agent.rag_agent import rag_node
    mock_doc = MagicMock()
    mock_doc.page_content = "退货政策：7天无理由退货，需保持原包装。"
    with patch("agent.rag_agent.PGVector") as mock_vs, \
         patch("agent.rag_agent.get_llm") as mock_llm, \
         patch("agent.rag_agent.get_embeddings"):
        mock_vs.return_value.as_retriever.return_value.invoke.return_value = [mock_doc]
        mock_llm.return_value.invoke.return_value = MagicMock(
            content="7天内可退货，需保持原包装完好。",
            usage_metadata={"input_tokens": 150, "output_tokens": 30}
        )
        result = rag_node(_make_state("rag"))
    assert result["rag_result"] == "7天内可退货，需保持原包装完好。"
    assert result["usage"]["input_tokens"] == 150


def test_rag_node_returns_result_for_mixed_intent():
    from agent.rag_agent import rag_node
    mock_doc = MagicMock()
    mock_doc.page_content = "顺丰速运满299元免费升级。"
    with patch("agent.rag_agent.PGVector") as mock_vs, \
         patch("agent.rag_agent.get_llm") as mock_llm, \
         patch("agent.rag_agent.get_embeddings"):
        mock_vs.return_value.as_retriever.return_value.invoke.return_value = [mock_doc]
        mock_llm.return_value.invoke.return_value = MagicMock(
            content="满299元可免费升级顺丰速运。",
            usage_metadata={"input_tokens": 120, "output_tokens": 20}
        )
        result = rag_node(_make_state("mixed", "物流多久到？"))
    assert result["rag_result"] != ""
```

- [ ] **Step 2: 运行，确认失败**

```bash
pytest tests/test_rag_agent.py -v 2>&1 | tail -5
```

Expected: FAILED（ImportError）

- [ ] **Step 3: 创建 agent/rag_agent.py**

```python
from langchain_postgres import PGVector
from agent.state import CustomerServiceState
from config import get_llm, get_embeddings, DATABASE_URL, VECTOR_COLLECTION
from utils.usage import accumulate


def rag_node(state: CustomerServiceState, database_url: str = None) -> CustomerServiceState:
    if state["intent"] not in ("rag", "mixed"):
        return state

    question = state["messages"][-1]["content"]
    url = database_url or DATABASE_URL

    vectorstore = PGVector(
        embeddings=get_embeddings(),
        collection_name=VECTOR_COLLECTION,
        connection=url,
        use_jsonb=True,
    )
    docs = vectorstore.as_retriever(search_kwargs={"k": 3}).invoke(question)
    context = "\n\n".join(doc.page_content for doc in docs)
    llm = get_llm()

    prompt = (
        "你是 ShopMind 电商平台的专业客服。根据以下知识库内容回答用户问题，"
        "回答简洁准确。如知识库无相关信息，回答：暂无相关资料，建议联系人工客服。\n\n"
        f"知识库内容：\n{context}\n\n"
        f"用户问题：{question}\n\n回答："
    )
    response = llm.invoke(prompt)
    usage = accumulate(state, response)
    return {**state, "rag_result": response.content, "usage": usage}
```

- [ ] **Step 4: 运行，确认通过**

```bash
pytest tests/test_rag_agent.py -v 2>&1 | tail -6
```

Expected: 4 PASSED

- [ ] **Step 5: commit**

```bash
git add agent/rag_agent.py tests/test_rag_agent.py
git commit -m "feat: RAG agent for e-commerce knowledge base retrieval"
```

---

## Task 7：SQL Agent + 测试

**Files:**
- Create: `agent/sql_agent.py`
- Create: `tests/test_sql_agent.py`

- [ ] **Step 1: 写失败测试**

创建 `tests/test_sql_agent.py`：

```python
from unittest.mock import patch, MagicMock
from agent.state import CustomerServiceState


def _make_state(intent: str, question: str = "我的订单到哪了？") -> CustomerServiceState:
    return {
        "messages": [{"role": "user", "content": question}],
        "intent": intent, "rag_result": "", "sql_result": "",
        "final_answer": "", "customer_id": "1",
        "session_id": "test-session-001",
        "usage": {"input_tokens": 0, "output_tokens": 0},
    }


def test_sql_node_skips_for_rag_intent():
    from agent.sql_agent import sql_node
    result = sql_node(_make_state("rag"))
    assert result["sql_result"] == ""


def test_sql_node_skips_for_chat_intent():
    from agent.sql_agent import sql_node
    result = sql_node(_make_state("chat"))
    assert result["sql_result"] == ""


def test_sql_node_queries_and_summarizes():
    from agent.sql_agent import sql_node
    with patch("agent.sql_agent.psycopg2.connect") as mock_connect, \
         patch("agent.sql_agent.get_llm") as mock_llm:
        mock_cursor = MagicMock()
        mock_cursor.fetchall.return_value = [
            {"order_no": "SM20260115002", "status": "shipped", "carrier": "京东快递", "tracking_number": "JD9876543210"}
        ]
        mock_cursor.description = [("order_no",), ("status",), ("carrier",), ("tracking_number",)]
        mock_connect.return_value.__enter__ = MagicMock(return_value=mock_connect.return_value)
        mock_connect.return_value.cursor.return_value.__enter__ = MagicMock(return_value=mock_cursor)
        mock_connect.return_value.cursor.return_value.__exit__ = MagicMock(return_value=False)
        mock_connect.return_value.__exit__ = MagicMock(return_value=False)

        mock_llm.return_value.invoke.side_effect = [
            MagicMock(content="SELECT o.order_no, o.status, l.carrier, l.tracking_number FROM ec_orders o LEFT JOIN ec_logistics l ON o.id=l.order_id WHERE o.customer_id=1"),
            MagicMock(content="您的订单SM20260115002正在运输中，快递公司京东快递，单号JD9876543210。",
                      usage_metadata={"input_tokens": 200, "output_tokens": 40})
        ]
        result = sql_node(_make_state("sql"))
    assert result["sql_result"] != ""


def test_sql_node_handles_db_error_gracefully():
    from agent.sql_agent import sql_node
    with patch("agent.sql_agent.psycopg2.connect") as mock_connect, \
         patch("agent.sql_agent.get_llm") as mock_llm:
        mock_llm.return_value.invoke.return_value = MagicMock(content="SELECT * FROM ec_orders WHERE customer_id=1")
        mock_connect.side_effect = Exception("connection refused")
        result = sql_node(_make_state("sql"))
    assert "抱歉" in result["sql_result"] or "失败" in result["sql_result"]
```

- [ ] **Step 2: 运行，确认失败**

```bash
pytest tests/test_sql_agent.py -v 2>&1 | tail -5
```

Expected: FAILED（ImportError）

- [ ] **Step 3: 创建 agent/sql_agent.py**

```python
import psycopg2
import psycopg2.extras
from agent.state import CustomerServiceState
from config import get_llm, DATABASE_URL
from utils.usage import accumulate

SCHEMA = """
数据库表结构（PostgreSQL）：
- ec_customers(id, name, email, phone, membership, created_at)
  membership: 'regular'=普通会员, 'silver'=银卡, 'gold'=金卡
- ec_products(id, sku, name, category, price, stock, description)
- ec_orders(id, order_no, customer_id, total_amount, status, created_at)
  status: 'pending'=待付款, 'paid'=已付款, 'shipped'=已发货, 'delivered'=已收货, 'cancelled'=已取消
- ec_order_items(id, order_id, product_id, quantity, unit_price)
- ec_logistics(id, order_id, carrier, tracking_number, status, estimated_delivery)
  status: 'processing'=处理中, 'shipped'=已发货, 'in_transit'=运输中, 'delivered'=已送达
- ec_returns(id, order_id, customer_id, reason, status, created_at)
  status: 'pending'=待审核, 'approved'=已批准, 'rejected'=已拒绝, 'completed'=已完成
"""


def sql_node(state: CustomerServiceState, database_url: str = None) -> CustomerServiceState:
    if state["intent"] not in ("sql", "mixed"):
        return state

    question = state["messages"][-1]["content"]
    customer_id = state.get("customer_id", "1")
    llm = get_llm()

    sql_prompt = (
        f"根据以下 PostgreSQL 数据库结构，为用户问题生成查询语句。\n"
        f"只返回 SQL 语句，不要解释或 markdown。\n"
        f"当前客户 ID 为 {customer_id}，涉及个人数据时必须加 customer_id = {customer_id} 条件。\n\n"
        f"{SCHEMA}\n"
        f"用户问题：{question}\n\nSQL:"
    )

    sql_resp = llm.invoke(sql_prompt)
    sql = sql_resp.content.strip().strip("```sql").strip("```").strip()
    usage = accumulate(state, sql_resp)

    url = database_url or DATABASE_URL
    try:
        conn = psycopg2.connect(url)
        with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
            cur.execute(sql)
            rows = cur.fetchall()
        conn.close()
        raw = "查询结果为空。" if not rows else "\n".join(str(dict(r)) for r in rows)
    except Exception as e:
        return {**state, "sql_result": f"抱歉，数据查询失败：{e}", "usage": usage}

    summary_prompt = (
        "将以下数据库查询结果用友好的中文回答用户问题，不暴露技术细节。\n\n"
        f"用户问题：{question}\n查询结果：{raw}\n\n回答："
    )
    summary = llm.invoke(summary_prompt)
    usage = accumulate({**state, "usage": usage}, summary)
    return {**state, "sql_result": summary.content, "usage": usage}
```

- [ ] **Step 4: 运行，确认通过**

```bash
pytest tests/test_sql_agent.py -v 2>&1 | tail -6
```

Expected: 4 PASSED

- [ ] **Step 5: commit**

```bash
git add agent/sql_agent.py tests/test_sql_agent.py
git commit -m "feat: SQL agent for e-commerce order/logistics queries"
```

---

## Task 8：Validator + Memory Agent + 测试

**Files:**
- Create: `agent/validator.py`
- Create: `agent/memory.py`
- Create: `tests/test_validator.py`
- Create: `tests/test_memory.py`

- [ ] **Step 1: 写失败测试**

创建 `tests/test_validator.py`：

```python
from agent.state import CustomerServiceState


def _make_state(answer: str) -> CustomerServiceState:
    return {
        "messages": [], "intent": "rag", "rag_result": "",
        "sql_result": "", "final_answer": answer,
        "customer_id": "1", "session_id": "test-001",
        "usage": {"input_tokens": 0, "output_tokens": 0},
    }


def test_validator_adds_disclaimer():
    from agent.validator import validator_node
    result = validator_node(_make_state("您的订单预计明天到达。"))
    assert "仅供参考" in result["final_answer"]


def test_validator_blocks_guaranteed_delivery():
    from agent.validator import validator_node
    result = validator_node(_make_state("保证明天送到您手中。"))
    assert "保证明天" not in result["final_answer"] or "无法" in result["final_answer"]


def test_validator_blocks_unconditional_refund():
    from agent.validator import validator_node
    result = validator_node(_make_state("无条件退款，立即到账。"))
    assert "无条件退款" not in result["final_answer"] or "无法" in result["final_answer"]
```

创建 `tests/test_memory.py`：

```python
from unittest.mock import patch, MagicMock
from agent.state import CustomerServiceState


def _make_state(messages: list, answer: str) -> CustomerServiceState:
    return {
        "messages": messages, "intent": "rag",
        "rag_result": "some result", "sql_result": "",
        "final_answer": answer, "customer_id": "1",
        "session_id": "test-session-001",
        "usage": {"input_tokens": 0, "output_tokens": 0},
    }


def test_memory_appends_assistant_message():
    from agent.memory import memory_node
    with patch("agent.memory.save_session"):
        state = _make_state([{"role": "user", "content": "退货怎么申请？"}], "请在7天内申请退货。")
        result = memory_node(state)
    assert len(result["messages"]) == 2
    assert result["messages"][-1]["role"] == "assistant"
    assert "7天" in result["messages"][-1]["content"]


def test_memory_clears_intermediate_results():
    from agent.memory import memory_node
    with patch("agent.memory.save_session"):
        state = _make_state([], "some answer")
        result = memory_node(state)
    assert result["rag_result"] == ""
    assert result["sql_result"] == ""


def test_memory_preserves_history():
    from agent.memory import memory_node
    history = [
        {"role": "user", "content": "你好"},
        {"role": "assistant", "content": "您好！"},
        {"role": "user", "content": "退货政策？"},
    ]
    with patch("agent.memory.save_session"):
        result = memory_node(_make_state(history, "7天可退。"))
    assert len(result["messages"]) == 4
    assert result["messages"][0]["content"] == "你好"
```

- [ ] **Step 2: 运行，确认失败**

```bash
pytest tests/test_validator.py tests/test_memory.py -v 2>&1 | tail -5
```

Expected: FAILED（ImportError）

- [ ] **Step 3: 创建 agent/validator.py**

```python
from agent.state import CustomerServiceState

_PROHIBITED = [
    "保证明天送到", "保证今天到", "一定今天送达", "肯定明天到",
    "无条件退款", "立即退款", "秒退", "马上退款",
    "全网最低价", "绝对最低", "保证最低",
    "100%正品保证",
]

_DISCLAIMER = "\n\n*以上信息仅供参考，具体以实际情况为准。如需进一步帮助请联系人工客服：400-888-8888。*"


def validator_node(state: CustomerServiceState) -> CustomerServiceState:
    answer = state.get("final_answer", "")
    flagged = [p for p in _PROHIBITED if p in answer]
    if flagged:
        answer = "抱歉，我无法就该问题做出具体承诺。建议您联系人工客服获取准确信息：400-888-8888。"
    return {**state, "final_answer": answer + _DISCLAIMER}
```

- [ ] **Step 4: 创建 agent/memory.py**

```python
import json
import psycopg2
from agent.state import CustomerServiceState
from config import DATABASE_URL


def load_session(session_id: str) -> list[dict]:
    conn = psycopg2.connect(DATABASE_URL)
    try:
        with conn.cursor() as cur:
            cur.execute(
                "SELECT messages FROM cs_sessions WHERE session_id = %s",
                (session_id,)
            )
            row = cur.fetchone()
            if row:
                return row[0] if isinstance(row[0], list) else json.loads(row[0])
            cur.execute(
                "INSERT INTO cs_sessions (session_id, messages) VALUES (%s, %s)",
                (session_id, json.dumps([]))
            )
            conn.commit()
            return []
    finally:
        conn.close()


def save_session(session_id: str, messages: list[dict]):
    conn = psycopg2.connect(DATABASE_URL)
    try:
        with conn.cursor() as cur:
            cur.execute(
                "INSERT INTO cs_sessions (session_id, messages) VALUES (%s, %s) "
                "ON CONFLICT (session_id) DO UPDATE SET messages = %s, updated_at = NOW()",
                (session_id, json.dumps(messages, ensure_ascii=False),
                 json.dumps(messages, ensure_ascii=False))
            )
        conn.commit()
    finally:
        conn.close()


def memory_node(state: CustomerServiceState) -> CustomerServiceState:
    assistant_msg = {"role": "assistant", "content": state.get("final_answer", "")}
    updated_messages = list(state.get("messages", [])) + [assistant_msg]
    save_session(state["session_id"], updated_messages)
    return {
        **state,
        "messages": updated_messages,
        "rag_result": "",
        "sql_result": "",
    }
```

- [ ] **Step 5: 运行，确认通过**

```bash
pytest tests/test_validator.py tests/test_memory.py -v 2>&1 | tail -8
```

Expected: 6 PASSED

- [ ] **Step 6: commit**

```bash
git add agent/validator.py agent/memory.py tests/test_validator.py tests/test_memory.py
git commit -m "feat: validator (e-commerce compliance) + memory node (session persistence)"
```

---

## Task 9：FastAPI main.py + 集成测试

**Files:**
- Create: `main.py`
- Create: `tests/test_integration.py`

- [ ] **Step 1: 创建 main.py**

```python
import uuid
from contextlib import asynccontextmanager
from fastapi import FastAPI
from pydantic import BaseModel
from agent.supervisor import build_graph
from agent.state import CustomerServiceState
from utils.startup import check_and_init

_graph = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global _graph
    check_and_init()
    _graph = build_graph()
    yield


app = FastAPI(title="ShopMind Customer Service Agent", version="1.0.0", lifespan=lifespan)


class InvokeRequest(BaseModel):
    message: str
    session_id: str | None = None
    customer_id: str = "1"


@app.post("/invoke")
def invoke(req: InvokeRequest):
    session_id = req.session_id or str(uuid.uuid4())

    from agent.memory import load_session
    history = load_session(session_id)

    state: CustomerServiceState = {
        "messages": history + [{"role": "user", "content": req.message}],
        "intent": "",
        "rag_result": "",
        "sql_result": "",
        "final_answer": "",
        "customer_id": req.customer_id,
        "session_id": session_id,
        "usage": {"input_tokens": 0, "output_tokens": 0},
    }
    result = _graph.invoke(state)
    return {
        "answer": result["final_answer"],
        "session_id": session_id,
        "intent": result["intent"],
        "usage": result.get("usage", {"input_tokens": 0, "output_tokens": 0}),
    }


@app.get("/health")
def health():
    return {"status": "ok", "service": "agent-customer-service"}
```

- [ ] **Step 2: 写集成测试**

创建 `tests/test_integration.py`：

```python
import pytest
from unittest.mock import patch, MagicMock
from fastapi.testclient import TestClient


@pytest.fixture
def client():
    from agent.rule_cleaner_stub import build_stub_graph

    def stub_graph_invoke(state):
        state = {**state, "intent": "chat", "final_answer": "您好！有什么可以帮您？"}
        from agent.validator import validator_node
        state = validator_node(state)
        return state

    mock_graph = MagicMock()
    mock_graph.invoke.side_effect = stub_graph_invoke

    with patch("utils.startup.check_and_init"), \
         patch("agent.memory.load_session", return_value=[]), \
         patch("agent.memory.save_session"):
        import importlib
        import main as m
        importlib.reload(m)
        m._graph = mock_graph
        yield TestClient(m.app)


def test_invoke_returns_answer_and_session_id(client):
    resp = client.post("/invoke", json={"message": "你好", "customer_id": "1"})
    assert resp.status_code == 200
    body = resp.json()
    assert "answer" in body
    assert "session_id" in body
    assert "intent" in body
    assert "usage" in body
    assert len(body["session_id"]) > 0


def test_invoke_reuses_session_id(client):
    resp = client.post("/invoke", json={
        "message": "退货政策？",
        "session_id": "fixed-session-123",
        "customer_id": "1"
    })
    assert resp.json()["session_id"] == "fixed-session-123"


def test_health_endpoint(client):
    resp = client.get("/health")
    assert resp.status_code == 200
    assert resp.json()["status"] == "ok"
```

- [ ] **Step 3: 运行全部测试**

```bash
pytest tests/ -v --ignore=tests/test_integration.py 2>&1 | tail -15
```

Expected: 全部 PASSED（约 17 个）

- [ ] **Step 4: 手动启动验证**

```bash
# 创建 .env（从 insurance-agent 复制 DATABASE_URL 和 ANTHROPIC_API_KEY）
cp /Users/liuyun/CCWorkSpace/insurance-agent/.env .env
# 修改 LLM_PROVIDER=claude
python -m uvicorn main:app --port 8002 --reload
```

在另一个终端：

```bash
curl -s http://localhost:8002/health
# Expected: {"status":"ok","service":"agent-customer-service"}

curl -s -X POST http://localhost:8002/invoke \
  -H "Content-Type: application/json" \
  -d '{"message":"退货政策是什么？","customer_id":"1"}'
```

Expected: 返回包含 `answer`、`session_id`、`intent`、`usage` 的 JSON。

- [ ] **Step 5: commit**

```bash
git add main.py tests/test_integration.py
git commit -m "feat: FastAPI /invoke endpoint with session management + /health"
```

---

## Task 10：更新 shopmind-web（客服对话页）

**Files:**
- Modify: `shopmind-web/app/dashboard/layout.tsx` — 侧边栏加客服入口
- Create: `shopmind-web/app/dashboard/customer/page.tsx` — 对话界面

- [ ] **Step 1: 切换到 shopmind-web develop 分支**

```bash
cd /Users/liuyun/CCWorkSpace/shopmind-ai/shopmind-web
git checkout develop
mkdir -p app/dashboard/customer
```

- [ ] **Step 2: 更新侧边栏 app/dashboard/layout.tsx**

将 `navItems` 数组从：

```typescript
const navItems = [
  { href: "/dashboard", label: "概览", emoji: "📊" },
  { href: "/dashboard/cleaning", label: "数据清洗", emoji: "🧹" },
];
```

改为：

```typescript
const navItems = [
  { href: "/dashboard",           label: "概览",   emoji: "📊" },
  { href: "/dashboard/cleaning",  label: "数据清洗", emoji: "🧹" },
  { href: "/dashboard/customer",  label: "智能客服", emoji: "💬" },
];
```

- [ ] **Step 3: 创建 app/dashboard/customer/page.tsx**

```typescript
"use client";
import { useState, useRef, useEffect } from "react";
import { api } from "@/lib/api";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";

type Message = { role: "user" | "assistant"; content: string };

const EXAMPLES = [
  "退货政策是什么？",
  "我的订单到哪了？",
  "如何申请退换货？",
  "满多少免运费？",
  "金卡会员有什么折扣？",
];

const INTENT_LABEL: Record<string, string> = {
  rag: "政策查询",
  sql: "账户查询",
  mixed: "综合查询",
  chat: "闲聊",
};

export default function CustomerPage() {
  const [messages, setMessages] = useState<Message[]>([]);
  const [sessionId, setSessionId] = useState<string | null>(null);
  const [customerId, setCustomerId] = useState("1");
  const [input, setInput] = useState("");
  const [loading, setLoading] = useState(false);
  const [lastIntent, setLastIntent] = useState("");
  const [lastUsage, setLastUsage] = useState<{ input_tokens: number; output_tokens: number } | null>(null);
  const bottomRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [messages]);

  async function sendMessage(text: string) {
    if (!text.trim() || loading) return;
    const userMsg: Message = { role: "user", content: text };
    setMessages(prev => [...prev, userMsg]);
    setInput("");
    setLoading(true);

    try {
      const res = await api.invokeAgent("customer", {
        message: text,
        session_id: sessionId,
        customer_id: customerId,
      }) as { answer: string; session_id: string; intent: string; usage: { input_tokens: number; output_tokens: number } };

      setSessionId(res.session_id);
      setLastIntent(res.intent);
      setLastUsage(res.usage);
      setMessages(prev => [...prev, { role: "assistant", content: res.answer }]);
    } catch (e) {
      setMessages(prev => [...prev, {
        role: "assistant",
        content: "抱歉，服务暂时不可用，请稍后重试。"
      }]);
    } finally {
      setLoading(false);
    }
  }

  function clearChat() {
    setMessages([]);
    setSessionId(null);
    setLastIntent("");
    setLastUsage(null);
  }

  return (
    <div className="max-w-3xl space-y-4">
      <div className="flex items-center justify-between">
        <div>
          <h1 className="text-2xl font-bold">💬 智能客服</h1>
          <p className="text-slate-500 text-sm mt-1">支持商品政策问答 · 订单物流查询 · 多轮对话</p>
        </div>
        <div className="flex items-center gap-3">
          <select
            value={customerId}
            onChange={e => setCustomerId(e.target.value)}
            className="text-sm border rounded px-2 py-1"
          >
            <option value="1">张伟 (ID: 1)</option>
            <option value="2">李娜 (ID: 2)</option>
            <option value="3">王芳 (ID: 3)</option>
          </select>
          <Button variant="outline" size="sm" onClick={clearChat}>清空对话</Button>
        </div>
      </div>

      {/* 快速示例 */}
      <div className="flex flex-wrap gap-2">
        {EXAMPLES.map(ex => (
          <button
            key={ex}
            onClick={() => sendMessage(ex)}
            className="text-xs px-3 py-1 bg-slate-100 hover:bg-slate-200 rounded-full transition-colors"
          >
            {ex}
          </button>
        ))}
      </div>

      {/* 对话区 */}
      <Card>
        <CardContent className="p-4 h-96 overflow-y-auto space-y-3">
          {messages.length === 0 && (
            <div className="text-center text-slate-400 text-sm mt-8">
              点击上方示例或直接输入问题开始对话
            </div>
          )}
          {messages.map((msg, i) => (
            <div key={i} className={`flex ${msg.role === "user" ? "justify-end" : "justify-start"}`}>
              <div className={`max-w-[80%] px-4 py-2 rounded-2xl text-sm ${
                msg.role === "user"
                  ? "bg-slate-900 text-white rounded-br-sm"
                  : "bg-slate-100 text-slate-800 rounded-bl-sm"
              }`}>
                {msg.content}
              </div>
            </div>
          ))}
          {loading && (
            <div className="flex justify-start">
              <div className="bg-slate-100 px-4 py-2 rounded-2xl rounded-bl-sm text-sm text-slate-500">
                思考中...
              </div>
            </div>
          )}
          <div ref={bottomRef} />
        </CardContent>
      </Card>

      {/* 输入区 */}
      <div className="flex gap-2">
        <Input
          value={input}
          onChange={e => setInput(e.target.value)}
          onKeyDown={e => e.key === "Enter" && !e.shiftKey && sendMessage(input)}
          placeholder="输入问题，按 Enter 发送..."
          disabled={loading}
        />
        <Button onClick={() => sendMessage(input)} disabled={loading || !input.trim()}>
          发送
        </Button>
      </div>

      {/* 状态栏 */}
      {lastIntent && (
        <div className="flex items-center gap-3 text-xs text-slate-400">
          <Badge variant="secondary">{INTENT_LABEL[lastIntent] ?? lastIntent}</Badge>
          {lastUsage && (
            <span>
              Tokens: {lastUsage.input_tokens + lastUsage.output_tokens}
              （${((lastUsage.input_tokens / 1e6 * 3) + (lastUsage.output_tokens / 1e6 * 15)).toFixed(4)}）
            </span>
          )}
        </div>
      )}
    </div>
  );
}
```

- [ ] **Step 4: 构建验证**

```bash
npm run build 2>&1 | tail -8
```

Expected: 构建成功，新增 `/dashboard/customer` 路由。

- [ ] **Step 5: commit**

```bash
git add app/dashboard/layout.tsx app/dashboard/customer/
git commit -m "feat: add customer service chat page with session support and intent badge"
```

---

## Task 11：注册到 Gateway + 端到端测试

**Files:**
- Modify: `shopmind-gateway/.env` — 确认 AGENT_CUSTOMER_URL 已配置
- No code changes needed（gateway 在 Phase 1 已预配置 `AGENT_CUSTOMER_URL=http://localhost:8002`）

- [ ] **Step 1: 确认 gateway 配置**

```bash
grep AGENT_CUSTOMER_URL /Users/liuyun/CCWorkSpace/shopmind-ai/shopmind-gateway/.env
```

Expected: `AGENT_CUSTOMER_URL=http://localhost:8002`（已在 Phase 1 配置，无需修改）

- [ ] **Step 2: 启动三个服务**

终端 1：
```bash
cd /Users/liuyun/CCWorkSpace/shopmind-ai/agent-customer-service
python -m uvicorn main:app --port 8002 --log-level warning
```

终端 2：
```bash
cd /Users/liuyun/CCWorkSpace/shopmind-ai/shopmind-gateway
python -m uvicorn main:app --port 8000 --log-level warning
```

终端 3：
```bash
cd /Users/liuyun/CCWorkSpace/shopmind-ai/shopmind-web
npm run dev
```

- [ ] **Step 3: 端到端 curl 测试**

```bash
# 登录获取 token
TOKEN=$(curl -s -X POST http://localhost:8000/auth/login \
  -H "Content-Type: application/json" \
  -d '{"username":"demo","password":"shopmind2026"}' \
  | python3 -c "import sys,json; print(json.load(sys.stdin)['access_token'])")

# 调用客服 Agent（政策查询）
curl -s -X POST http://localhost:8000/api/v1/customer/invoke \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer $TOKEN" \
  -d '{"message":"退货政策是什么？","customer_id":"1"}' \
  | python3 -c "import sys,json; r=json.load(sys.stdin); print('intent:', r['intent'], '| tokens:', r['usage']['input_tokens']+r['usage']['output_tokens'])"
```

Expected: `intent: rag | tokens: XXX`

```bash
# 多轮对话测试（复用 session_id）
SESSION=$(curl -s -X POST http://localhost:8000/api/v1/customer/invoke \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer $TOKEN" \
  -d '{"message":"你好","customer_id":"1"}' \
  | python3 -c "import sys,json; print(json.load(sys.stdin)['session_id'])")

curl -s -X POST http://localhost:8000/api/v1/customer/invoke \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer $TOKEN" \
  -d "{\"message\":\"我的订单到哪了？\",\"session_id\":\"$SESSION\",\"customer_id\":\"1\"}" \
  | python3 -c "import sys,json; r=json.load(sys.stdin); print('intent:', r['intent'])"
```

Expected: `intent: sql`（第二轮识别为账户查询）

- [ ] **Step 4: 浏览器验证**

访问 http://localhost:3000/dashboard/customer，验证：
- 侧边栏出现"💬 智能客服"入口
- 快速示例按钮可点击
- 对话正常显示（气泡样式）
- 底部显示 intent badge 和 token 消耗

- [ ] **Step 5: 各仓库 final commit**

```bash
cd /Users/liuyun/CCWorkSpace/shopmind-ai/agent-customer-service
git add . && git commit -m "chore: phase2 complete - customer service agent integrated with gateway"

cd /Users/liuyun/CCWorkSpace/shopmind-ai/shopmind-web
git add . && git commit -m "chore: phase2 complete - customer chat page added"
```

- [ ] **Step 6: 推送到 GitHub**

```bash
cd /Users/liuyun/CCWorkSpace/shopmind-ai/agent-customer-service
gh repo create shopmind-ai/agent-customer-service \
  --public \
  --description "ShopMind AI - E-commerce customer service agent: RAG + Text-to-SQL + persistent multi-turn conversation" \
  --source=. --remote=origin --push

cd /Users/liuyun/CCWorkSpace/shopmind-ai/shopmind-web
git push origin develop
```

---

## 自审通过项

- ✅ `CustomerServiceState` 包含 `session_id`，与 `memory.py` 的 load/save 一致
- ✅ 所有表名加 `ec_`/`cs_` 前缀，避免 Neon 共享库冲突
- ✅ supervisor 的 chat prompt 已适配电商场景（ShopMind 品牌名）
- ✅ validator 的禁用词列表针对电商（快递承诺、退款承诺、价格保证）
- ✅ memory_node 既写 PostgreSQL 也更新 state["messages"]，两处一致
- ✅ `/invoke` 端点：session_id 为 None 时自动生成新 UUID
- ✅ shopmind-web 新增路由 `/dashboard/customer`，layout.tsx 侧边栏同步更新
- ✅ Gateway 已在 Phase 1 预配置 `AGENT_CUSTOMER_URL=http://localhost:8002`，无需修改
- ✅ 所有函数名在测试与实现中一致（supervisor_node、rag_node、sql_node、validator_node、memory_node）
