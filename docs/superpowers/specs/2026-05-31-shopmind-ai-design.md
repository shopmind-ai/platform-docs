# ShopMind AI — 电商智能运营平台 设计文档

**日期：** 2026-05-31  
**作者：** Liu Yun  
**状态：** 已确认，待实现

---

## 1. 平台定位

**名称：** ShopMind AI  
**副标题：** AI-Powered E-commerce Operations Platform（电商智能运营平台）

**定位：** 面向中小电商团队的 AI 运营平台，涵盖数据治理、用户服务、内容生产、销售培训、系统集成、模型优化全链路。8 个 Agent 对应电商业务的 8 个核心痛点，随业务增长逐期上线，模拟真实的需求演进路径。

**演示场景：** 一家经营运动户外品类的中小电商（跑步装备、健身器材、户外露营），中英文双语界面。

---

## 2. 仓库结构（Multi-Repo）

所有仓库统一放在 GitHub Organization `shopmind-ai` 下：

```
shopmind-ai/                          ← GitHub Organization
│
├── shopmind-web                      # Next.js 14 前端（统一入口）
├── shopmind-gateway                  # FastAPI API Gateway
│
├── agent-data-cleaning               # ① 数据清洗中心
├── agent-customer-service            # ② 电商客服 Agent
├── agent-media-library               # ③ 多模态素材库
├── agent-social-media                # ④ 自媒体 Agent
├── agent-clip                        # ⑤ 直播切片 Agent
├── agent-sales-training              # ⑥ 销售培训系统
├── agent-cluster                     # ⑦ MCP + Agent 集群
└── agent-finetune                    # ⑧ 模型微调
```

### 每个 Agent 仓库的统一结构

```
agent-xxx/
├── main.py              # FastAPI 入口（统一暴露 /invoke、/health）
├── agent/               # LangGraph Agent 逻辑
│   ├── state.py
│   ├── supervisor.py
│   └── ...
├── config.py            # 环境变量（DATABASE_URL、Claude API Key 等）
├── utils/
│   ├── startup.py       # 自动初始化（建表、向量化）
│   └── usage.py         # Token 消耗统计
├── data/                # 种子数据和文档
├── tests/
├── requirements.txt
└── README.md
```

---

## 3. 系统架构

### 整体架构

```
┌─────────────────────────────────────────────────────────┐
│                    shopmind-web                          │
│         Next.js 14 · TypeScript · shadcn/ui             │
│              中英文 i18n · Tailwind CSS                  │
└────────────────────────┬────────────────────────────────┘
                         │ HTTPS
                         ▼
┌─────────────────────────────────────────────────────────┐
│                  shopmind-gateway                        │
│    FastAPI · JWT 认证 · 路由 · 限流 · Token 统计          │
│    超时熔断：10s 无响应 → 降级结果                         │
└──┬──────┬──────┬──────┬──────┬──────┬──────┬───────────┘
   │      │      │      │      │      │      │      │
   ▼      ▼      ▼      ▼      ▼      ▼      ▼      ▼
 ①清洗  ②客服  ③素材  ④社媒  ⑤切片  ⑥培训  ⑦集群  ⑧微调
```

### 共享基础设施

| 服务 | 工具 | 用途 |
|------|------|------|
| 关系型数据库 | PostgreSQL（Neon） | 业务数据持久化 |
| 向量数据库 | pgvector（同 Neon 实例） | RAG 检索 |
| 缓存 | Redis（Upstash） | 任务状态、会话、限流 |
| LLM（生产） | Claude API（claude-sonnet-4-6） | Agent 推理 |
| LLM（本地开发） | Ollama | 零成本本地调试 |
| Embedding | sentence-transformers（多语言） | 向量化，免费本地运行 |

### 跨 Agent 数据流

```
① 数据清洗 → 输出结构化商品/订单数据
                ↓
② 客服 Agent ← 查询订单数据
③ 素材库 ← 存储并标注商品图片/视频
                ↓
④ 自媒体 Agent ← 调用素材库 + 清洗后数据生成内容
⑤ 直播切片 → 切片结果自动入素材库（反哺④）
                ↓
⑥ 销售培训 ← 使用客服对话数据生成培训案例
                ↓
⑦ MCP 集群 → 统一编排 ①–⑥ 所有 Agent 能力
                ↓
⑧ 模型微调 → 用积累的电商数据优化专属模型
```

---

## 4. 技术栈

| 层级 | 技术 | 说明 |
|------|------|------|
| 前端 | Next.js 14 · TypeScript · shadcn/ui · Tailwind CSS | 双语 i18n |
| API 网关 | FastAPI · Python 3.11 · python-jose（JWT） | 统一入口 |
| Agent 框架 | LangGraph · LangChain | 有状态多 Agent 编排 |
| LLM | Claude API（生产）· Ollama（本地）| 环境变量切换 |
| 向量存储 | langchain-postgres + pgvector | 与业务库合并 |
| Embedding | sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2 | 免费多语言 |
| 视频处理 | FFmpeg · Whisper（openai-whisper） | 切片 Agent 专用 |
| 图像理解 | Claude Vision API | 素材库标注 |
| 模型微调 | MLX（Apple Silicon）· LoRA | 本地 M1 Max |
| 数据库 | PostgreSQL 16（Neon serverless） | |
| 缓存 | Redis（Upstash serverless） | |
| 前端部署 | Vercel | 自动 CI/CD |
| 后端部署 | Railway | Docker 容器化 |
| CI/CD | GitHub Actions | 每个 repo 独立 workflow |

---

## 5. 分期交付计划

每期产出一个**可独立部署、有公开 URL** 的子产品。`shopmind-web` 和 `shopmind-gateway` 在 Phase 1 建立，后续每期新 Agent 上线时同步接入。

| 期次 | Agent | 核心技术亮点 | 预计时长 |
|------|-------|------------|---------|
| Phase 1 | ① 数据清洗中心 | LLM 智能识别 + 规则引擎 + 批量处理 | 2 周 |
| Phase 2 | ② 电商客服 Agent | RAG + Text-to-SQL + 多轮对话 | 2 周 |
| Phase 3 | ③ 多模态素材库 | Claude Vision 标注 + pgvector 检索 | 2 周 |
| Phase 4 | ④ 自媒体 Agent | 多步骤 LangGraph + 素材库联动 | 2 周 |
| Phase 5 | ⑤ 直播切片 Agent | FFmpeg + Whisper + 高光识别 | 3 周 |
| Phase 6 | ⑥ 销售培训系统 | 角色扮演对话 + AI 评分 Agent | 2 周 |
| Phase 7 | ⑦ MCP + Agent 集群 | MCP 协议 + 跨服务编排 + Gateway 升级 | 3 周 |
| Phase 8 | ⑧ 模型微调 | MLX LoRA + 微调前后效果对比评估 | 3 周 |

**总计：约 19 周（~5 个月），每天 2 小时节奏**

### MVP 优先级

时间紧时，按此顺序交付：
1. **P0**：Gateway + 客服 Agent（② 最能体现 RAG + SQL 能力）
2. **P1**：自媒体 Agent + 直播切片 Agent（内容生产闭环）
3. **P2**：数据清洗 + 素材库（基础设施完善）
4. **P3**：销售培训 + MCP 集群 + 模型微调

---

## 6. Gateway 设计要点

```python
# 路由规则
POST /api/v1/cleaning/invoke      → agent-data-cleaning:8001
POST /api/v1/customer/invoke      → agent-customer-service:8002
POST /api/v1/media/invoke         → agent-media-library:8003
POST /api/v1/social/invoke        → agent-social-media:8004
POST /api/v1/clip/invoke          → agent-clip:8005
POST /api/v1/training/invoke      → agent-sales-training:8006
POST /api/v1/cluster/invoke       → agent-cluster:8007
POST /api/v1/finetune/invoke      → agent-finetune:8008

# 所有 Agent 统一健康检查
GET  /api/v1/{agent}/health

# 统一 Token 统计
POST /api/v1/usage/summary        → 汇总所有 Agent 消耗
```

**熔断规则：** Agent 服务 10s 无响应 → 返回 `{"status": "degraded", "message": "服务暂时不可用，请稍后重试"}` + 记录告警日志。

---

## 7. 面试话术定位

这套项目在面试中回答的核心问题：

> "你能描述一个你主导的、有一定复杂度的 AI 系统吗？"

回答框架：
1. **业务背景**：中小电商 AI 化的全链路需求
2. **架构决策**：API Gateway 模式，8 个独立 Agent 服务，统一前端
3. **技术亮点**：LangGraph 编排 + pgvector 检索 + FFmpeg 视频处理 + MLX 微调
4. **工程化**：统一 Token 统计、熔断降级、多语言支持、CI/CD
5. **演进逻辑**：从数据治理开始，逐步扩展到内容生产、模型优化，符合真实业务节奏
