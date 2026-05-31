# ShopMind AI — 电商智能运营平台

面向中小电商团队的 AI 运营平台，涵盖数据治理、用户服务、内容生产、销售培训、系统集成、模型优化全链路。8 个独立 Agent 围绕同一套电商业务场景递进构建，每个 Agent 对应一类真实业务需求，可单独使用也可通过集群 Agent 组合调用。

---

## 项目概览

| 仓库 | Agent | 端口 | 业务功能 |
|------|-------|------|---------|
| [shopmind-gateway](https://github.com/shopmind-ai/shopmind-gateway) | API 网关 | 8000 | JWT 认证、路由转发、用量统计 |
| [agent-data-cleaning](https://github.com/shopmind-ai/agent-data-cleaning) | 数据清洗 | 8001 | 规则引擎 + LLM 双层清洗商品数据 |
| [agent-customer-service](https://github.com/shopmind-ai/agent-customer-service) | 智能客服 | 8002 | 政策问答 + 订单查询 + 多轮对话 |
| [agent-media-library](https://github.com/shopmind-ai/agent-media-library) | 素材库 | 8003 | Claude Vision 标注 + 语义图片检索 |
| [agent-social-media](https://github.com/shopmind-ai/agent-social-media) | 内容创作 | 8004 | 小红书 / 抖音 / 公众号文案生成 |
| [agent-clip](https://github.com/shopmind-ai/agent-clip) | 直播切片 | 8005 | Whisper 转录 + Claude 高光识别 + FFmpeg 切片 |
| [agent-sales-training](https://github.com/shopmind-ai/agent-sales-training) | 销售培训 | 8006 | AI 扮演客户陪练 + 4 维度评分 |
| [agent-cluster](https://github.com/shopmind-ai/agent-cluster) | 集群指挥 | 8007 | Claude Tool Use 编排多 Agent |
| [agent-finetune](https://github.com/shopmind-ai/agent-finetune) | 模型微调 | 8008 | MLX LoRA 训练 + 微调前后对比 |
| [shopmind-web](https://github.com/shopmind-ai/shopmind-web) | 前端 | 3000 | Next.js 14 统一操作界面 |

---

## 技术架构

```
浏览器 (Next.js 14)
    │  HTTPS + JWT Bearer Token
    ▼
shopmind-gateway (FastAPI)
  · JWT 认证（HS256，60 分钟有效期）
  · 路由到对应 Agent 服务
  · 统一记录 Token 用量（shopmind_usage_logs 表）
  · 超时熔断：60s 无响应 → 返回降级提示
    │
    ├── agent-data-cleaning  :8001
    ├── agent-customer-service :8002
    ├── agent-media-library  :8003
    ├── agent-social-media   :8004
    ├── agent-clip           :8005
    ├── agent-sales-training :8006
    ├── agent-cluster        :8007
    └── agent-finetune       :8008

共享基础设施
  · PostgreSQL + pgvector（Neon serverless，单实例）
  · 各 Agent 表名加前缀避免冲突（ec_、cs_、shopmind_ 等）
  · LLM：Claude API（生产）/ Ollama（本地开发）
  · Embedding：sentence-transformers（本地运行，免费）
```

---

## 认证机制

平台采用 JWT（JSON Web Token）无状态认证，流程如下：

```
1. 前端 POST /auth/login  { username, password }
2. Gateway 查询 shopmind_users 表，用 bcrypt 验证密码
3. 验证通过 → 用 SECRET_KEY 签名生成 access_token（HS256，60 min）
4. 前端将 token 存入 localStorage 和 cookie
5. 后续请求携带：Authorization: Bearer <token>
6. Gateway middleware 验证签名和有效期（无需查库）
7. 验证失败（过期/篡改）→ 返回 401，前端自动跳转到登录页
```

Token 是**无状态**的——服务端不存储 session，只需验证签名。适合微服务架构，每个 Agent 服务均受 Gateway 保护。

> 注：当前为内部工具模式，账号由管理员预创建（demo / shopmind2026）。如需支持自注册，在 Gateway 添加 `POST /auth/register` 端点即可。

---

## 数据库说明

所有 Agent 共用同一个 **Neon PostgreSQL** 实例（含 pgvector 扩展），通过表名前缀隔离：

| 前缀 | 归属 | 表举例 |
|------|------|--------|
| `shopmind_` | Gateway | `shopmind_users`、`shopmind_usage_logs` |
| `ec_` | 客服 / 数据清洗 | `ec_customers`、`ec_orders`、`ec_products` |
| `cs_` | 客服多轮对话 | `cs_sessions` |
| `media_` | 素材库 | `media_assets` |
| `training_` | 销售培训 | `training_sessions` |
| `cluster_` | 集群 | `cluster_jobs` |
| `finetune_` | 微调 | `finetune_jobs` |
| `content_` | 内容创作 | `content_history` |
| `clip_` | 切片 | `clip_jobs`、`video_clips` |
| pgvector collection | RAG 检索 | `customer_service_docs`、`media_assets_embeddings` |

单实例对作品集项目完全够用（Neon 免费 tier 0.5GB）。生产环境建议按业务域拆分为独立数据库。

---

## 各 Agent 使用场景

### 数据清洗 Agent
**场景**：从 ERP、Excel 导出的商品数据格式混乱（价格带 ¥ 符号、分类名称不统一、有空行），上传前需先规范化。  
**目标用户**：运营专员、数据专员。  
**技术亮点**：规则引擎先处理明显问题（0 API 成本），LLM 处理语义层面问题（重复商品识别、分类标准化）。

### 智能客服 Agent
**场景**：买家咨询退货政策、物流进度、促销规则，全天候自动回复；也可查询具体订单状态。  
**目标用户**：客服主管（负责配置知识库）、最终消费者（使用）。  
**技术亮点**：Supervisor Agent 意图分类 → RAG（政策知识库）+ Text-to-SQL（订单数据）双路并行 → 会话历史持久化到 PostgreSQL。

### 素材库 Agent
**场景**：商品图片上传后自动生成标签和描述，运营可以用自然语言（"红色跑鞋"）搜索素材，无需手动整理文件夹。  
**目标用户**：视觉设计师、内容运营。  
**技术亮点**：Claude Vision 分析图片 → 描述文本 Embedding → pgvector 语义检索。

### 内容创作 Agent
**场景**：输入商品名称和卖点，一键生成符合各平台调性的文案（小红书种草、抖音口播、公众号测评），替代人工写稿。  
**目标用户**：新媒体编辑、内容运营。  
**技术亮点**：LangGraph 5 节点管道（规划→写作→格式化→合规检查→素材推荐），支持联动素材库获取配图。

### 直播切片 Agent
**场景**：直播结束后上传录像 URL，自动提取精彩片段（产品展示、促销时刻、互动高峰），生成可直接复用的短视频。  
**目标用户**：直播运营、短视频剪辑。  
**技术亮点**：Whisper 本地语音转录（免费）→ Claude 识别高光时刻 → FFmpeg 按时间戳切片。

### 销售培训 Agent
**场景**：新销售入职后，与 AI 扮演的挑剔/犹豫/价格敏感型客户反复练习应对话术，完成后获得 4 维度评分和改进建议，无需占用老员工陪练时间。  
**目标用户**：销售总监（配置训练场景）、销售新人（日常练习）。  
**技术亮点**：会话历史持久化、3 场景 × 3 客户类型 = 9 种训练组合、评分以 JSON 结构化返回。

### 集群指挥 Agent
**场景**：运营人员用自然语言描述复杂任务（"把这批数据清洗后生成小红书文案"），系统自动决定调用哪些 Agent、按什么顺序执行。  
**目标用户**：高级运营、产品经理。  
**技术亮点**：Claude Tool Use 取代硬编码 LangGraph 流程，LLM 自主编排 6 个工具，体现 MCP 协议核心思想。

### 模型微调 Agent
**场景**：将平台积累的真实销售对话和客服记录导出为训练集，在本地 M1 Max 用 MLX LoRA 微调小模型，让模型更懂电商业务场景，对比微调前后的回答质量差异。  
**目标用户**：AI 工程师、技术负责人。  
**技术亮点**：从 PostgreSQL 自动提取 JSONL 格式训练数据，支持本地 mlx_lm 推理，无本地模型时自动 fallback 到 Claude。

---

## 快速启动

### 前提

- Python 3.11+
- Node.js 18+
- [Neon](https://neon.tech) 账号（免费）
- Anthropic API Key

### 启动所有后端服务

```bash
# 每个服务在对应目录创建 .env（从 .env.example 复制）
# DATABASE_URL 和 ANTHROPIC_API_KEY 所有服务共用同一个值

# 依次启动（或并行）
cd shopmind-gateway      && uvicorn main:app --port 8000
cd agent-data-cleaning   && uvicorn main:app --port 8001
cd agent-customer-service&& uvicorn main:app --port 8002
cd agent-media-library   && uvicorn main:app --port 8003
cd agent-social-media    && uvicorn main:app --port 8004
cd agent-clip            && uvicorn main:app --port 8005
cd agent-sales-training  && uvicorn main:app --port 8006
cd agent-cluster         && uvicorn main:app --port 8007
cd agent-finetune        && uvicorn main:app --port 8008
```

> 每个服务首次启动时自动初始化数据库表。

### 启动前端

```bash
cd shopmind-web
cp .env.local.example .env.local   # 填入 NEXT_PUBLIC_GATEWAY_URL
npm install && npm run dev
# 访问 http://localhost:3000
# 账号：demo / shopmind2026
```

---

## 技术栈

| 层级 | 技术 |
|------|------|
| 前端 | Next.js 14 · TypeScript · shadcn/ui · Tailwind CSS |
| API 网关 | FastAPI · python-jose（JWT）· bcrypt · httpx |
| Agent 框架 | LangGraph · LangChain · Claude API |
| 视觉理解 | Claude Vision API |
| 语音转录 | OpenAI Whisper（本地运行）|
| 视频处理 | FFmpeg |
| 向量检索 | PostgreSQL + pgvector · sentence-transformers |
| 模型微调 | MLX（Apple Silicon）· LoRA |
| 数据库 | PostgreSQL 16（Neon serverless）|
| 部署 | Vercel（前端）· Railway（后端）|

---

## 目录结构

```
shopmind-ai/
├── docs/
│   └── superpowers/
│       ├── specs/    # 架构设计文档
│       └── plans/    # 各 Phase 实现计划
├── shopmind-gateway/
├── agent-data-cleaning/
├── agent-customer-service/
├── agent-media-library/
├── agent-social-media/
├── agent-clip/
├── agent-sales-training/
├── agent-cluster/
├── agent-finetune/
└── shopmind-web/
```
