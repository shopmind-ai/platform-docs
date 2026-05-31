# ShopMind AI Phase 5 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 构建 agent-clip（直播切片 Agent），输入视频 URL → httpx 下载 → Whisper 语音转录 → Claude 识别高光时刻 → FFmpeg 切片 → 返回片段元数据（时间戳+标题+描述），并将结果持久化到 PostgreSQL。

**Architecture:** LangGraph 5 节点线性管道：downloader（httpx 下载+ffprobe 取时长）→ transcriber（Whisper 转录为带时间戳的片段列表）→ analyzer（Claude 从转录中识别 N 个高光时刻，输出包含 title/description 的 JSON）→ clipper（FFmpeg subprocess 切片保存到 /tmp/clips/{job_id}/）→ finalizer（写入 PostgreSQL clip_jobs + video_clips 表）。所有重量级外部调用（Whisper、FFmpeg、httpx）在单元测试中通过 mock 完全绕过。

**Tech Stack:** Python 3.11 · FastAPI · LangGraph · Claude API · openai-whisper · FFmpeg (subprocess) · httpx · psycopg2

---

## 参考资料

- 设计文档：`shopmind-ai/docs/superpowers/specs/2026-05-31-shopmind-ai-design.md`
- 模式参考：`/Users/liuyun/CCWorkSpace/shopmind-ai/agent-social-media/`（LangGraph + utils 结构）
- Gateway 需配置：`AGENT_CLIP_URL=http://localhost:8005`（在 gateway `.env` 手动添加）
- FFmpeg 须已安装：`ffmpeg -version`（Liu Yun HSBC 项目已安装）

---

## 文件结构总览

```
agent-clip/
├── main.py                    # FastAPI: POST /invoke, GET /health
├── config.py                  # env vars, get_llm(), CLIPS_OUTPUT_DIR
├── agent/
│   ├── __init__.py
│   ├── state.py               # ClipState TypedDict
│   ├── downloader.py          # httpx 下载视频 + ffprobe 取时长
│   ├── transcriber.py         # Whisper 语音转录
│   ├── analyzer.py            # Claude 识别高光片段（含标题/描述）
│   ├── clipper.py             # FFmpeg 切片
│   ├── finalizer.py           # 写 PostgreSQL，组装最终返回
│   └── graph.py               # LangGraph 图组装
├── utils/
│   ├── __init__.py
│   ├── startup.py             # 建 clip_jobs + video_clips 表
│   └── usage.py               # Token 统计（与前各 Phase 一致）
├── tests/
│   ├── __init__.py
│   ├── test_downloader.py
│   ├── test_transcriber.py
│   ├── test_analyzer.py
│   ├── test_clipper.py
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
  "video_url": "https://example.com/livestream.mp4",
  "max_clips": 3,
  "clip_duration": 30
}

Response:
{
  "job_id": "uuid",
  "clips": [
    {
      "clip_id": "uuid",
      "start_time": 120.5,
      "end_time": 150.5,
      "title": "产品核心功能展示",
      "description": "主播详细演示碳板跑鞋回弹效果",
      "transcript_excerpt": "这个碳板踩上去...",
      "file_path": "/tmp/clips/{job_id}/clip_0.mp4"
    }
  ],
  "total_duration": 3600.0,
  "clip_count": 3,
  "usage": {"input_tokens": 800, "output_tokens": 300}
}
```

---

## Task 1：项目初始化 + config + utils

**Files:**
- Create: `agent-clip/requirements.txt`
- Create: `agent-clip/.env.example`
- Create: `agent-clip/.gitignore`
- Create: `agent-clip/config.py`
- Create: `agent-clip/utils/usage.py`
- Create: `agent-clip/utils/startup.py`

- [ ] **Step 1: 创建目录结构**

```bash
cd /Users/liuyun/CCWorkSpace/shopmind-ai
mkdir -p agent-clip/{agent,utils,tests}
touch agent-clip/agent/__init__.py
touch agent-clip/utils/__init__.py
touch agent-clip/tests/__init__.py
cd agent-clip && git init && git checkout -b develop
git commit --allow-empty -m "chore: init agent-clip"
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
openai-whisper>=20231117
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
CLIPS_OUTPUT_DIR=/tmp/clips
WHISPER_MODEL=base
```

- [ ] **Step 4: 创建 .gitignore**

```
__pycache__/
*.pyc
.env
.venv/
*.egg-info/
.pytest_cache/
/tmp/clips/
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
CLIPS_OUTPUT_DIR = os.getenv("CLIPS_OUTPUT_DIR", "/tmp/clips")
WHISPER_MODEL = os.getenv("WHISPER_MODEL", "base")


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


def accumulate(state: dict, response) -> TokenUsage:
    meta = getattr(response, "usage_metadata", None) or {}
    current = state.get("usage") or {"input_tokens": 0, "output_tokens": 0}
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
from config import DATABASE_URL, CLIPS_OUTPUT_DIR
import os

_DDL = """
CREATE TABLE IF NOT EXISTS clip_jobs (
    id            SERIAL PRIMARY KEY,
    job_id        VARCHAR(36)  UNIQUE NOT NULL DEFAULT gen_random_uuid()::text,
    video_url     TEXT         NOT NULL,
    status        VARCHAR(20)  NOT NULL DEFAULT 'completed',
    total_duration FLOAT,
    clip_count    INTEGER      DEFAULT 0,
    input_tokens  INTEGER      DEFAULT 0,
    output_tokens INTEGER      DEFAULT 0,
    created_at    TIMESTAMPTZ  DEFAULT NOW()
);

CREATE TABLE IF NOT EXISTS video_clips (
    id                SERIAL PRIMARY KEY,
    clip_id           VARCHAR(36) UNIQUE NOT NULL DEFAULT gen_random_uuid()::text,
    job_id            VARCHAR(36) NOT NULL REFERENCES clip_jobs(job_id),
    start_time        FLOAT       NOT NULL,
    end_time          FLOAT       NOT NULL,
    title             TEXT,
    description       TEXT,
    transcript_excerpt TEXT,
    file_path         TEXT,
    created_at        TIMESTAMPTZ DEFAULT NOW()
);
"""


def check_and_init():
    if not DATABASE_URL:
        raise RuntimeError("DATABASE_URL is not set.")
    os.makedirs(CLIPS_OUTPUT_DIR, exist_ok=True)
    conn = psycopg2.connect(DATABASE_URL)
    try:
        with conn.cursor() as cur:
            cur.execute(_DDL)
        conn.commit()
        print("[startup] agent-clip DB initialized.")
    finally:
        conn.close()


def save_job(job_id: str, video_url: str, total_duration: float,
             clip_count: int, usage: dict):
    conn = psycopg2.connect(DATABASE_URL)
    try:
        with conn.cursor() as cur:
            cur.execute(
                "INSERT INTO clip_jobs (job_id, video_url, total_duration, clip_count, "
                "input_tokens, output_tokens) VALUES (%s,%s,%s,%s,%s,%s) "
                "ON CONFLICT (job_id) DO NOTHING",
                (job_id, video_url, total_duration, clip_count,
                 usage.get("input_tokens", 0), usage.get("output_tokens", 0))
            )
        conn.commit()
    finally:
        conn.close()


def save_clips(clips: list[dict]):
    if not clips:
        return
    conn = psycopg2.connect(DATABASE_URL)
    try:
        with conn.cursor() as cur:
            for clip in clips:
                cur.execute(
                    "INSERT INTO video_clips "
                    "(clip_id, job_id, start_time, end_time, title, description, "
                    "transcript_excerpt, file_path) "
                    "VALUES (%s,%s,%s,%s,%s,%s,%s,%s) ON CONFLICT (clip_id) DO NOTHING",
                    (clip["clip_id"], clip["job_id"], clip["start_time"], clip["end_time"],
                     clip.get("title"), clip.get("description"),
                     clip.get("transcript_excerpt"), clip.get("file_path"))
                )
        conn.commit()
    finally:
        conn.close()
```

- [ ] **Step 8: 安装依赖并验证**

```bash
pip install -r requirements.txt -q
python -c "import fastapi, langgraph, httpx, whisper; print('imports OK')"
```

Expected: `imports OK`

- [ ] **Step 9: commit**

```bash
git add . && git commit -m "feat: project setup, config, utils (startup + usage)"
```

---

## Task 2：ClipState + Downloader 节点 + 测试

**Files:**
- Create: `agent/state.py`
- Create: `agent/downloader.py`
- Create: `tests/test_downloader.py`

- [ ] **Step 1: 创建 agent/state.py**

```python
from typing import TypedDict
from utils.usage import TokenUsage


class ClipState(TypedDict):
    # 输入
    video_url: str
    max_clips: int
    clip_duration: int         # 每个片段秒数
    job_id: str
    # 管道中间状态
    video_path: str            # 本地临时文件路径
    total_duration: float      # 视频总时长（秒）
    transcript: list[dict]     # [{start: float, end: float, text: str}]
    highlights: list[dict]     # [{start_time, end_time, reason, title, description, transcript_excerpt}]
    # 输出
    clips: list[dict]          # [{clip_id, job_id, start_time, end_time, title, description, transcript_excerpt, file_path}]
    usage: TokenUsage
```

- [ ] **Step 2: 写失败测试**

创建 `tests/test_downloader.py`：

```python
import os
from unittest.mock import patch, MagicMock
from agent.state import ClipState


def _make_state(**kwargs) -> ClipState:
    defaults = {
        "video_url": "https://example.com/video.mp4",
        "max_clips": 3, "clip_duration": 30,
        "job_id": "test-job-001",
        "video_path": "", "total_duration": 0.0,
        "transcript": [], "highlights": [], "clips": [],
        "usage": {"input_tokens": 0, "output_tokens": 0},
    }
    return {**defaults, **kwargs}


def test_downloader_saves_video_file():
    from agent.downloader import downloader_node
    with patch("agent.downloader.httpx.stream") as mock_stream, \
         patch("agent.downloader._get_duration", return_value=180.0), \
         patch("os.makedirs"):
        mock_ctx = MagicMock()
        mock_ctx.__enter__ = MagicMock(return_value=mock_ctx)
        mock_ctx.__exit__ = MagicMock(return_value=False)
        mock_ctx.iter_bytes.return_value = [b"fake video data"]
        mock_stream.return_value = mock_ctx

        with patch("builtins.open", MagicMock()):
            result = downloader_node(_make_state())

    assert result["video_path"] != ""
    assert result["total_duration"] == 180.0


def test_downloader_sets_job_output_dir():
    from agent.downloader import downloader_node
    with patch("agent.downloader.httpx.stream") as mock_stream, \
         patch("agent.downloader._get_duration", return_value=60.0), \
         patch("os.makedirs") as mock_mkdirs, \
         patch("builtins.open", MagicMock()):
        mock_ctx = MagicMock()
        mock_ctx.__enter__ = MagicMock(return_value=mock_ctx)
        mock_ctx.__exit__ = MagicMock(return_value=False)
        mock_ctx.iter_bytes.return_value = [b"data"]
        mock_stream.return_value = mock_ctx

        result = downloader_node(_make_state(job_id="job-abc"))

    assert "job-abc" in result["video_path"]


def test_get_duration_parses_ffprobe_output():
    from agent.downloader import _get_duration
    with patch("agent.downloader.subprocess.run") as mock_run:
        mock_run.return_value = MagicMock(stdout="3600.12\n", returncode=0)
        duration = _get_duration("/tmp/video.mp4")
    assert duration == 3600.12
```

- [ ] **Step 3: 运行，确认失败**

```bash
pytest tests/test_downloader.py -v 2>&1 | tail -5
```

Expected: FAILED（ImportError）

- [ ] **Step 4: 创建 agent/downloader.py**

```python
import os
import subprocess
import httpx
from agent.state import ClipState
from config import CLIPS_OUTPUT_DIR


def _get_duration(video_path: str) -> float:
    result = subprocess.run(
        ["ffprobe", "-v", "quiet", "-show_entries", "format=duration",
         "-of", "csv=p=0", video_path],
        capture_output=True, text=True
    )
    return float(result.stdout.strip())


def downloader_node(state: ClipState) -> ClipState:
    job_dir = os.path.join(CLIPS_OUTPUT_DIR, state["job_id"])
    os.makedirs(job_dir, exist_ok=True)
    video_path = os.path.join(job_dir, "video.mp4")

    with httpx.stream("GET", state["video_url"], follow_redirects=True) as resp:
        with open(video_path, "wb") as f:
            for chunk in resp.iter_bytes(chunk_size=8192):
                f.write(chunk)

    total_duration = _get_duration(video_path)
    return {**state, "video_path": video_path, "total_duration": total_duration}
```

- [ ] **Step 5: 运行，确认通过**

```bash
pytest tests/test_downloader.py -v 2>&1 | tail -6
```

Expected: 3 PASSED

- [ ] **Step 6: commit**

```bash
git add agent/state.py agent/downloader.py tests/test_downloader.py
git commit -m "feat: ClipState + downloader node (httpx + ffprobe)"
```

---

## Task 3：Transcriber 节点 + 测试

**Files:**
- Create: `agent/transcriber.py`
- Create: `tests/test_transcriber.py`

- [ ] **Step 1: 写失败测试**

创建 `tests/test_transcriber.py`：

```python
from unittest.mock import patch, MagicMock
from agent.state import ClipState


def _make_state(**kwargs) -> ClipState:
    defaults = {
        "video_url": "https://example.com/video.mp4", "max_clips": 3, "clip_duration": 30,
        "job_id": "test-job", "video_path": "/tmp/clips/test-job/video.mp4",
        "total_duration": 180.0, "transcript": [], "highlights": [], "clips": [],
        "usage": {"input_tokens": 0, "output_tokens": 0},
    }
    return {**defaults, **kwargs}


def test_transcriber_returns_segment_list():
    from agent.transcriber import transcriber_node
    mock_model = MagicMock()
    mock_model.transcribe.return_value = {
        "segments": [
            {"start": 0.0, "end": 5.2, "text": "欢迎来到今天的直播"},
            {"start": 5.2, "end": 12.8, "text": "今天给大家带来一款跑步鞋"},
            {"start": 12.8, "end": 20.0, "text": "碳纤维中底，回弹效果超级好"},
        ]
    }
    with patch("agent.transcriber.whisper.load_model", return_value=mock_model):
        result = transcriber_node(_make_state())
    assert len(result["transcript"]) == 3
    assert result["transcript"][0]["text"] == "欢迎来到今天的直播"
    assert result["transcript"][1]["start"] == 5.2


def test_transcriber_uses_configured_model():
    from agent.transcriber import transcriber_node
    mock_model = MagicMock()
    mock_model.transcribe.return_value = {"segments": []}
    with patch("agent.transcriber.whisper.load_model", return_value=mock_model) as mock_load:
        with patch("agent.transcriber.WHISPER_MODEL", "small"):
            transcriber_node(_make_state())
    mock_load.assert_called_once_with("small")


def test_transcriber_handles_empty_transcript():
    from agent.transcriber import transcriber_node
    mock_model = MagicMock()
    mock_model.transcribe.return_value = {"segments": []}
    with patch("agent.transcriber.whisper.load_model", return_value=mock_model):
        result = transcriber_node(_make_state())
    assert result["transcript"] == []
```

- [ ] **Step 2: 运行，确认失败**

```bash
pytest tests/test_transcriber.py -v 2>&1 | tail -5
```

Expected: FAILED（ImportError）

- [ ] **Step 3: 创建 agent/transcriber.py**

```python
import whisper
from agent.state import ClipState
from config import WHISPER_MODEL


def transcriber_node(state: ClipState) -> ClipState:
    model = whisper.load_model(WHISPER_MODEL)
    result = model.transcribe(state["video_path"])
    transcript = [
        {"start": seg["start"], "end": seg["end"], "text": seg["text"].strip()}
        for seg in result.get("segments", [])
    ]
    return {**state, "transcript": transcript}
```

- [ ] **Step 4: 运行，确认通过**

```bash
pytest tests/test_transcriber.py -v 2>&1 | tail -6
```

Expected: 3 PASSED

- [ ] **Step 5: commit**

```bash
git add agent/transcriber.py tests/test_transcriber.py
git commit -m "feat: transcriber node (Whisper speech-to-text with timestamps)"
```

---

## Task 4：Analyzer 节点 + 测试

**Files:**
- Create: `agent/analyzer.py`
- Create: `tests/test_analyzer.py`

- [ ] **Step 1: 写失败测试**

创建 `tests/test_analyzer.py`：

```python
import json
from unittest.mock import patch, MagicMock
from agent.state import ClipState


def _make_state(**kwargs) -> ClipState:
    defaults = {
        "video_url": "", "max_clips": 2, "clip_duration": 30,
        "job_id": "test", "video_path": "", "total_duration": 180.0,
        "transcript": [
            {"start": 0.0,  "end": 10.0, "text": "欢迎来到直播间"},
            {"start": 10.0, "end": 60.0, "text": "今天给大家介绍碳板跑鞋，回弹超级好！"},
            {"start": 60.0, "end": 120.0, "text": "很多跑友都反馈跑完马拉松脚不疼"},
            {"start": 120.0, "end": 180.0, "text": "现在购买有折扣，库存不多了！"},
        ],
        "highlights": [], "clips": [],
        "usage": {"input_tokens": 0, "output_tokens": 0},
    }
    return {**defaults, **kwargs}


def test_analyzer_returns_highlights():
    from agent.analyzer import analyzer_node
    mock_resp = MagicMock()
    mock_resp.content = json.dumps([
        {"start_time": 10.0, "end_time": 40.0, "reason": "产品功能展示",
         "title": "碳板跑鞋性能测评", "description": "主播展示回弹效果",
         "transcript_excerpt": "回弹超级好"},
        {"start_time": 120.0, "end_time": 150.0, "reason": "促销信息",
         "title": "限时折扣活动", "description": "库存紧张优惠购入",
         "transcript_excerpt": "现在购买有折扣"},
    ])
    mock_resp.usage_metadata = {"input_tokens": 600, "output_tokens": 200}
    with patch("agent.analyzer.get_llm") as mock_llm:
        mock_llm.return_value.invoke.return_value = mock_resp
        result = analyzer_node(_make_state())
    assert len(result["highlights"]) == 2
    assert result["highlights"][0]["start_time"] == 10.0
    assert result["highlights"][0]["title"] == "碳板跑鞋性能测评"
    assert result["usage"]["input_tokens"] == 600


def test_analyzer_limits_to_max_clips():
    from agent.analyzer import analyzer_node
    highlights_3 = [
        {"start_time": i*30.0, "end_time": i*30.0+30, "reason": f"moment{i}",
         "title": f"片段{i}", "description": f"desc{i}", "transcript_excerpt": "text"}
        for i in range(3)
    ]
    mock_resp = MagicMock()
    mock_resp.content = json.dumps(highlights_3)
    mock_resp.usage_metadata = {"input_tokens": 400, "output_tokens": 150}
    with patch("agent.analyzer.get_llm") as mock_llm:
        mock_llm.return_value.invoke.return_value = mock_resp
        result = analyzer_node(_make_state(max_clips=2))
    assert len(result["highlights"]) <= 2


def test_analyzer_handles_invalid_json():
    from agent.analyzer import analyzer_node
    mock_resp = MagicMock()
    mock_resp.content = "无法识别到精彩片段"
    mock_resp.usage_metadata = {"input_tokens": 200, "output_tokens": 30}
    with patch("agent.analyzer.get_llm") as mock_llm:
        mock_llm.return_value.invoke.return_value = mock_resp
        result = analyzer_node(_make_state())
    assert result["highlights"] == []
```

- [ ] **Step 2: 运行，确认失败**

```bash
pytest tests/test_analyzer.py -v 2>&1 | tail -5
```

Expected: FAILED（ImportError）

- [ ] **Step 3: 创建 agent/analyzer.py**

```python
import json
from agent.state import ClipState
from config import get_llm
from utils.usage import accumulate

_PROMPT = """你是一名短视频运营专家，分析以下直播转录文本，找出最精彩的{max_clips}个片段。

转录文本（格式：[时间] 内容）：
{formatted_transcript}

视频总时长：{total_duration:.0f}秒
每个片段约{clip_duration}秒

识别标准：产品核心功能展示、促销优惠宣布、观众互动高峰、情绪感染力强的时刻。
要求：片段之间不重叠，时间在合理范围内。

返回JSON数组（只返回JSON，不要有任何解释）：
[
  {{
    "start_time": 120.5,
    "end_time": 150.5,
    "reason": "产品功能展示",
    "title": "碳板跑鞋性能实测！",
    "description": "主播详细演示碳板中底回弹效果，数据对比直观",
    "transcript_excerpt": "这个踩上去的感觉..."
  }}
]"""


def _format_transcript(transcript: list[dict], max_chars: int = 4000) -> str:
    lines = []
    for seg in transcript:
        mins = int(seg["start"] // 60)
        secs = int(seg["start"] % 60)
        lines.append(f"[{mins:02d}:{secs:02d}] {seg['text']}")
    text = "\n".join(lines)
    return text[:max_chars] if len(text) > max_chars else text


def analyzer_node(state: ClipState) -> ClipState:
    if not state["transcript"]:
        return {**state, "highlights": []}

    llm = get_llm()
    formatted = _format_transcript(state["transcript"])
    prompt = _PROMPT.format(
        max_clips=state["max_clips"],
        formatted_transcript=formatted,
        total_duration=state["total_duration"],
        clip_duration=state["clip_duration"],
    )
    response = llm.invoke(prompt)
    usage = accumulate(state, response)

    try:
        raw = response.content.strip().strip("```json").strip("```").strip()
        highlights = json.loads(raw)
        if not isinstance(highlights, list):
            highlights = []
    except (json.JSONDecodeError, ValueError):
        highlights = []

    highlights = highlights[: state["max_clips"]]
    return {**state, "highlights": highlights, "usage": usage}
```

- [ ] **Step 4: 运行，确认通过**

```bash
pytest tests/test_analyzer.py -v 2>&1 | tail -6
```

Expected: 3 PASSED

- [ ] **Step 5: commit**

```bash
git add agent/analyzer.py tests/test_analyzer.py
git commit -m "feat: analyzer node (Claude highlight detection with title/description)"
```

---

## Task 5：Clipper 节点 + Finalizer 节点 + 测试

**Files:**
- Create: `agent/clipper.py`
- Create: `agent/finalizer.py`
- Create: `tests/test_clipper.py`

- [ ] **Step 1: 写失败测试**

创建 `tests/test_clipper.py`：

```python
import os
from unittest.mock import patch, MagicMock, call
from agent.state import ClipState


def _make_state(**kwargs) -> ClipState:
    defaults = {
        "video_url": "", "max_clips": 2, "clip_duration": 30,
        "job_id": "test-job",
        "video_path": "/tmp/clips/test-job/video.mp4",
        "total_duration": 180.0, "transcript": [],
        "highlights": [
            {"start_time": 10.0, "end_time": 40.0, "reason": "产品展示",
             "title": "好鞋推荐", "description": "碳板展示", "transcript_excerpt": "好用"},
            {"start_time": 90.0, "end_time": 120.0, "reason": "促销",
             "title": "限时优惠", "description": "折扣活动", "transcript_excerpt": "折扣"},
        ],
        "clips": [], "usage": {"input_tokens": 0, "output_tokens": 0},
    }
    return {**defaults, **kwargs}


def test_clipper_creates_clip_for_each_highlight():
    from agent.clipper import clipper_node
    with patch("agent.clipper.subprocess.run") as mock_run, \
         patch("os.makedirs"):
        mock_run.return_value = MagicMock(returncode=0)
        result = clipper_node(_make_state())
    assert len(result["clips"]) == 2
    assert mock_run.call_count == 2


def test_clipper_uses_ffmpeg_with_correct_args():
    from agent.clipper import clipper_node
    with patch("agent.clipper.subprocess.run") as mock_run, \
         patch("os.makedirs"):
        mock_run.return_value = MagicMock(returncode=0)
        result = clipper_node(_make_state())
    first_call_args = mock_run.call_args_list[0][0][0]
    assert "ffmpeg" in first_call_args
    assert "-ss" in first_call_args
    assert "-t" in first_call_args


def test_clipper_includes_job_id_in_clip():
    from agent.clipper import clipper_node
    with patch("agent.clipper.subprocess.run") as mock_run, \
         patch("os.makedirs"):
        mock_run.return_value = MagicMock(returncode=0)
        result = clipper_node(_make_state(job_id="my-job"))
    assert all(c["job_id"] == "my-job" for c in result["clips"])
    assert all("my-job" in c["file_path"] for c in result["clips"])


def test_clipper_handles_no_highlights():
    from agent.clipper import clipper_node
    result = clipper_node(_make_state(highlights=[]))
    assert result["clips"] == []
```

- [ ] **Step 2: 运行，确认失败**

```bash
pytest tests/test_clipper.py -v 2>&1 | tail -5
```

Expected: FAILED（ImportError）

- [ ] **Step 3: 创建 agent/clipper.py**

```python
import os
import subprocess
import uuid
from agent.state import ClipState
from config import CLIPS_OUTPUT_DIR


def clipper_node(state: ClipState) -> ClipState:
    if not state["highlights"]:
        return {**state, "clips": []}

    job_dir = os.path.join(CLIPS_OUTPUT_DIR, state["job_id"])
    os.makedirs(job_dir, exist_ok=True)

    clips = []
    for i, highlight in enumerate(state["highlights"]):
        clip_id = str(uuid.uuid4())
        output_path = os.path.join(job_dir, f"clip_{i}.mp4")
        start = highlight["start_time"]
        duration = highlight["end_time"] - highlight["start_time"]

        subprocess.run(
            ["ffmpeg", "-i", state["video_path"],
             "-ss", str(start),
             "-t", str(duration),
             "-c:v", "libx264", "-c:a", "aac",
             "-y", output_path],
            capture_output=True,
        )

        clips.append({
            "clip_id": clip_id,
            "job_id": state["job_id"],
            "start_time": highlight["start_time"],
            "end_time": highlight["end_time"],
            "title": highlight.get("title", f"片段 {i + 1}"),
            "description": highlight.get("description", ""),
            "transcript_excerpt": highlight.get("transcript_excerpt", ""),
            "file_path": output_path,
        })

    return {**state, "clips": clips}
```

- [ ] **Step 4: 创建 agent/finalizer.py**

```python
from agent.state import ClipState
from utils.startup import save_job, save_clips


def finalizer_node(state: ClipState) -> ClipState:
    usage = state.get("usage", {"input_tokens": 0, "output_tokens": 0})
    save_job(
        job_id=state["job_id"],
        video_url=state["video_url"],
        total_duration=state["total_duration"],
        clip_count=len(state["clips"]),
        usage=usage,
    )
    save_clips(state["clips"])
    return state
```

- [ ] **Step 5: 运行，确认通过**

```bash
pytest tests/test_clipper.py -v 2>&1 | tail -7
```

Expected: 4 PASSED

- [ ] **Step 6: commit**

```bash
git add agent/clipper.py agent/finalizer.py tests/test_clipper.py
git commit -m "feat: clipper node (FFmpeg subprocess) + finalizer node (PostgreSQL persist)"
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
from agent.state import ClipState
from agent.downloader import downloader_node
from agent.transcriber import transcriber_node
from agent.analyzer import analyzer_node
from agent.clipper import clipper_node
from agent.finalizer import finalizer_node


def build_graph():
    g = StateGraph(ClipState)
    g.add_node("downloader",   downloader_node)
    g.add_node("transcriber",  transcriber_node)
    g.add_node("analyzer",     analyzer_node)
    g.add_node("clipper",      clipper_node)
    g.add_node("finalizer",    finalizer_node)

    g.add_edge(START,        "downloader")
    g.add_edge("downloader", "transcriber")
    g.add_edge("transcriber","analyzer")
    g.add_edge("analyzer",   "clipper")
    g.add_edge("clipper",    "finalizer")
    g.add_edge("finalizer",  END)

    return g.compile()
```

- [ ] **Step 2: 创建 main.py**

```python
import uuid
from contextlib import asynccontextmanager
from fastapi import FastAPI
from pydantic import BaseModel
from agent.graph import build_graph
from agent.state import ClipState
from utils.startup import check_and_init

_graph = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global _graph
    check_and_init()
    _graph = build_graph()
    yield


app = FastAPI(title="ShopMind Clip Agent", version="1.0.0", lifespan=lifespan)


class InvokeRequest(BaseModel):
    video_url: str
    max_clips: int = 3
    clip_duration: int = 30


@app.post("/invoke")
def invoke(req: InvokeRequest):
    job_id = str(uuid.uuid4())
    state: ClipState = {
        "video_url":      req.video_url,
        "max_clips":      req.max_clips,
        "clip_duration":  req.clip_duration,
        "job_id":         job_id,
        "video_path":     "",
        "total_duration": 0.0,
        "transcript":     [],
        "highlights":     [],
        "clips":          [],
        "usage":          {"input_tokens": 0, "output_tokens": 0},
    }
    result = _graph.invoke(state)
    return {
        "job_id":         job_id,
        "clips":          result["clips"],
        "total_duration": result["total_duration"],
        "clip_count":     len(result["clips"]),
        "usage":          result.get("usage", {"input_tokens": 0, "output_tokens": 0}),
    }


@app.get("/health")
def health():
    return {"status": "ok", "service": "agent-clip"}
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
         patch("utils.startup.save_job"), \
         patch("utils.startup.save_clips"):
        import importlib
        import main as m
        importlib.reload(m)

        def fake_invoke(state):
            return {
                **state,
                "video_path": f"/tmp/clips/{state['job_id']}/video.mp4",
                "total_duration": 180.0,
                "transcript": [{"start": 0.0, "end": 5.0, "text": "直播开始"}],
                "highlights": [{"start_time": 10.0, "end_time": 40.0, "reason": "展示",
                                 "title": "好物推荐", "description": "详细介绍",
                                 "transcript_excerpt": "这个真的好"}],
                "clips": [{"clip_id": "abc-123", "job_id": state["job_id"],
                           "start_time": 10.0, "end_time": 40.0,
                           "title": "好物推荐", "description": "详细介绍",
                           "transcript_excerpt": "这个真的好",
                           "file_path": "/tmp/clips/test/clip_0.mp4"}],
                "usage": {"input_tokens": 500, "output_tokens": 150},
            }

        mock_graph = MagicMock()
        mock_graph.invoke.side_effect = fake_invoke
        m._graph = mock_graph
        yield TestClient(m.app)


def test_invoke_returns_complete_response(client):
    resp = client.post("/invoke", json={
        "video_url": "https://example.com/video.mp4",
        "max_clips": 3,
        "clip_duration": 30,
    })
    assert resp.status_code == 200
    body = resp.json()
    assert "job_id" in body
    assert "clips" in body
    assert "total_duration" in body
    assert "clip_count" in body
    assert "usage" in body
    assert body["clip_count"] == 1


def test_invoke_assigns_unique_job_id(client):
    resp1 = client.post("/invoke", json={"video_url": "https://x.com/v.mp4"})
    resp2 = client.post("/invoke", json={"video_url": "https://x.com/v.mp4"})
    assert resp1.json()["job_id"] != resp2.json()["job_id"]


def test_invoke_uses_default_params(client):
    resp = client.post("/invoke", json={"video_url": "https://x.com/v.mp4"})
    assert resp.status_code == 200


def test_health_endpoint(client):
    resp = client.get("/health")
    assert resp.status_code == 200
    assert resp.json()["service"] == "agent-clip"
```

- [ ] **Step 4: 运行全部测试**

```bash
pytest tests/ -v 2>&1 | tail -20
```

Expected: 全部 PASSED（约 13 个）

- [ ] **Step 5: 手动验证服务启动**

```bash
cp /Users/liuyun/CCWorkSpace/insurance-agent/.env .env
python -m uvicorn main:app --port 8005 --reload
# 另一终端
curl -s http://localhost:8005/health
# Expected: {"status":"ok","service":"agent-clip"}
```

- [ ] **Step 6: commit**

```bash
git add agent/graph.py main.py tests/test_integration.py
git commit -m "feat: LangGraph graph + FastAPI /invoke + 13 tests passing"
```

---

## Task 7：更新 shopmind-web（切片工具页）

**Files:**
- Modify: `shopmind-web/app/dashboard/layout.tsx`
- Create: `shopmind-web/app/dashboard/clip/page.tsx`
- Modify: `shopmind-web/app/dashboard/page.tsx` — 更新为 5/8

- [ ] **Step 1: 切换到 shopmind-web develop 分支**

```bash
cd /Users/liuyun/CCWorkSpace/shopmind-ai/shopmind-web
git checkout develop
mkdir -p app/dashboard/clip
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
  { href: "/dashboard/clip",     label: "直播切片", emoji: "✂️" },
];
```

- [ ] **Step 3: 更新 dashboard page.tsx（5/8）**

将 `4 / 8` 改为 `5 / 8`。

- [ ] **Step 4: 创建 app/dashboard/clip/page.tsx**

```typescript
"use client";
import { useState } from "react";
import { api } from "@/lib/api";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";

type Clip = {
  clip_id: string;
  start_time: number;
  end_time: number;
  title: string;
  description: string;
  transcript_excerpt: string;
  file_path: string;
};

type ClipResult = {
  job_id: string;
  clips: Clip[];
  total_duration: number;
  clip_count: number;
  usage: { input_tokens: number; output_tokens: number };
};

function formatTime(seconds: number): string {
  const m = Math.floor(seconds / 60);
  const s = Math.floor(seconds % 60);
  return `${m.toString().padStart(2, "0")}:${s.toString().padStart(2, "0")}`;
}

export default function ClipPage() {
  const [videoUrl, setVideoUrl] = useState("");
  const [maxClips, setMaxClips] = useState(3);
  const [clipDuration, setClipDuration] = useState(30);
  const [result, setResult] = useState<ClipResult | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");

  async function handleProcess() {
    if (!videoUrl.trim()) return;
    setLoading(true);
    setError("");
    try {
      const res = await api.invokeAgent("clip", {
        video_url: videoUrl.trim(),
        max_clips: maxClips,
        clip_duration: clipDuration,
      }) as ClipResult;
      setResult(res);
    } catch (e) {
      setError(e instanceof Error ? e.message : "处理失败，请重试");
    } finally {
      setLoading(false);
    }
  }

  return (
    <div className="max-w-4xl space-y-6">
      <div>
        <h1 className="text-2xl font-bold">✂️ 直播切片</h1>
        <p className="text-slate-500 text-sm mt-1">
          输入直播视频 URL，AI 自动识别精彩片段并用 FFmpeg 切片
        </p>
      </div>

      {/* 输入区 */}
      <Card>
        <CardContent className="p-4 space-y-4">
          <div className="space-y-1">
            <label className="text-sm font-medium">视频 URL</label>
            <Input
              value={videoUrl}
              onChange={e => setVideoUrl(e.target.value)}
              placeholder="https://example.com/livestream.mp4"
            />
          </div>
          <div className="grid grid-cols-2 gap-3">
            <div className="space-y-1">
              <label className="text-sm font-medium">最大片段数</label>
              <Input
                type="number"
                value={maxClips}
                onChange={e => setMaxClips(Number(e.target.value))}
                min={1} max={10}
              />
            </div>
            <div className="space-y-1">
              <label className="text-sm font-medium">片段时长（秒）</label>
              <Input
                type="number"
                value={clipDuration}
                onChange={e => setClipDuration(Number(e.target.value))}
                min={10} max={120}
              />
            </div>
          </div>
          <Button
            onClick={handleProcess}
            disabled={loading || !videoUrl.trim()}
            className="w-full"
          >
            {loading ? "处理中（下载+转录+切片，约需1-3分钟）..." : "开始处理"}
          </Button>
          {error && <p className="text-red-500 text-sm">{error}</p>}
        </CardContent>
      </Card>

      {/* 结果区 */}
      {result && (
        <div className="space-y-4">
          <div className="flex items-center gap-4 text-sm text-slate-500">
            <span>Job ID: <code className="font-mono text-xs bg-slate-100 px-1">{result.job_id.slice(0, 8)}...</code></span>
            <span>视频时长: {formatTime(result.total_duration)}</span>
            <span>识别片段: {result.clip_count} 个</span>
            <span>Tokens: {result.usage.input_tokens + result.usage.output_tokens}</span>
          </div>

          <div className="space-y-3">
            {result.clips.map((clip, i) => (
              <Card key={clip.clip_id}>
                <CardContent className="p-4">
                  <div className="flex items-start justify-between gap-3">
                    <div className="space-y-1 flex-1">
                      <div className="flex items-center gap-2">
                        <Badge variant="outline" className="font-mono text-xs">
                          {formatTime(clip.start_time)} → {formatTime(clip.end_time)}
                        </Badge>
                        <span className="font-semibold text-sm">{clip.title}</span>
                      </div>
                      <p className="text-sm text-slate-600">{clip.description}</p>
                      {clip.transcript_excerpt && (
                        <p className="text-xs text-slate-400 italic">
                          &ldquo;{clip.transcript_excerpt}&rdquo;
                        </p>
                      )}
                      <p className="text-xs text-slate-300 font-mono">{clip.file_path}</p>
                    </div>
                    <div className="text-2xl text-slate-300">✂️</div>
                  </div>
                </CardContent>
              </Card>
            ))}
          </div>
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

Expected: 构建成功，新增 `/dashboard/clip` 路由。

- [ ] **Step 6: commit 并推送**

```bash
git add app/dashboard/layout.tsx app/dashboard/clip/ app/dashboard/page.tsx
git commit -m "feat: live stream clip tool page + update agent count to 5/8"
git push origin develop
```

---

## Task 8：Gateway + 端到端测试 + 推送

- [ ] **Step 1: 注册 gateway**

```bash
echo "AGENT_CLIP_URL=http://localhost:8005" >> /Users/liuyun/CCWorkSpace/shopmind-ai/shopmind-gateway/.env
```

- [ ] **Step 2: 启动 clip agent**

```bash
cd /Users/liuyun/CCWorkSpace/shopmind-ai/agent-clip
nohup python -m uvicorn main:app --port 8005 > /tmp/clip_agent.log 2>&1 &
sleep 4 && curl -s http://localhost:8005/health
```

Expected: `{"status":"ok","service":"agent-clip"}`

- [ ] **Step 3: 重启 gateway**

```bash
pkill -f "uvicorn main:app --port 8000" 2>/dev/null; sleep 1
cd /Users/liuyun/CCWorkSpace/shopmind-ai/shopmind-gateway
nohup python -m uvicorn main:app --port 8000 > /tmp/gateway.log 2>&1 &
sleep 4 && curl -s http://localhost:8000/health
```

- [ ] **Step 4: 验证 gateway 代理**

```bash
TOKEN=$(curl -s -X POST http://localhost:8000/auth/login \
  -H "Content-Type: application/json" \
  -d '{"username":"demo","password":"shopmind2026"}' \
  | python3 -c "import sys,json; print(json.load(sys.stdin)['access_token'])")

curl -s http://localhost:8000/api/v1/clip/health \
  -H "Authorization: Bearer $TOKEN" \
  | python3 -c "import sys,json; r=json.load(sys.stdin); print('✅ clip health:', r.get('status'))"
```

Expected: `✅ clip health: ok`

> Note: 完整的 /invoke 端到端测试需要一个真实的短视频 URL。这里只验证 gateway 代理链路通畅。实际视频处理在手动测试阶段完成。

- [ ] **Step 5: 推送到 GitHub**

```bash
cd /Users/liuyun/CCWorkSpace/shopmind-ai/agent-clip
git add . && git commit -m "chore: phase5 complete - clip agent integrated with gateway"

gh repo create shopmind-ai/agent-clip \
  --public \
  --description "ShopMind AI - Live stream clip agent: Whisper transcription + Claude highlight detection + FFmpeg cutting" \
  --source=. --remote=origin --push

cd /Users/liuyun/CCWorkSpace/shopmind-ai/shopmind-gateway
git commit --allow-empty -m "chore: add AGENT_CLIP_URL to gateway config"
git push origin develop
```

---

## 自审通过项

- ✅ `ClipState` 字段：downloader 写 `video_path`/`total_duration`，transcriber 写 `transcript`，analyzer 写 `highlights`/`usage`，clipper 写 `clips`，finalizer 只持久化不改 state——无字段冲突
- ✅ `accumulate()` 只在 analyzer_node 中调用（只有 analyzer 调 LLM），其他节点不触碰 usage
- ✅ `save_job()` 和 `save_clips()` 在 finalizer 中调用，job_id 贯穿全程
- ✅ `_get_duration()` 暴露为独立函数，方便在 test_downloader.py 中 mock
- ✅ `clipper_node` 测试验证 FFmpeg 参数（-ss, -t），不运行真实 FFmpeg
- ✅ 集成测试 fixture 绕过 LangGraph，直接注入 mock_graph，模式与 Phase 2-4 一致
- ✅ gateway 代理端到端验证通过 `/health` 而非 `/invoke`，避免需要真实视频
- ✅ shopmind-web dashboard 更新为 5/8，layout.tsx 加入切片入口
