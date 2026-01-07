# TaskFriend LLM 工程师工具包

## 概览

TaskFriend 是一套围绕检索增强生成（RAG）和大语言模型（LLM）工作流的实践型学习环境。项目结合了可执行的 Jupyter Notebook、可复用的 Python 模块、可视化工具和精心整理的数据集，对应 “LLM Engineer Course”（原 “LMP-C01 LLM Engineer (Professional)”）课程，主角 TaskFriend 专注于任务管理与效率提升场景，并支持替换式 LLM 后端与文档知识库。

## 仓库结构

- `LLM-Engineer-Course/taskfriend/`：TaskFriend 核心模块，涵盖聊天循环、上下文管理、RAG 索引编排与评估工具。
- `LLM-Engineer-Course/functions/`：配套 Notebook 的辅助函数，提供绘图、HTML 表格、嵌入可视化、打分和流式 LLM 客户端等能力。
- `LLM-Engineer-Course/docs/taskfriend/`：TaskFriend 使用的 Markdown 知识库，构成 RAG 的原始语料。
- `LLM-Engineer-Course/resources/`：训练集、基准题库、示例脚本等数据资产。
- `LLM-Engineer-Course/config/`：DashScope API Key 加载脚本。
- `LLM-Engineer-Course/*.ipynb`：12 篇课程 Notebook，串联整套实验路径。
- `requirements.txt`：项目依赖列表。
- `version-20251016.txt`：课程内容的版本标识。

## 运行前置条件

- Python 3.10 及以上版本。
- DashScope 账号与 API Key，用于调用 Qwen 模型与嵌入服务（OpenAI 兼容接口）。
- 推荐：Jupyter 或 VS Code（安装 Python 与 Jupyter 插件）以运行 Notebook。

## 环境搭建步骤

1. 创建虚拟环境（建议使用 Python 3.12，以满足 `ms-swift` 对 `numpy<2` 的依赖）：

      - Windows（PowerShell）：在 PowerShell 终端执行
         ```powershell
         py -3.12 -m venv taskfriend
         # 若提示 “running scripts is disabled”，请先执行：
         Set-ExecutionPolicy -Scope Process -ExecutionPolicy RemoteSigned
         & .\taskfriend\Scripts\Activate.ps1
         ```

      - macOS / Linux：在 Bash/zsh 终端执行
         ```bash
         python3.12 -m venv taskfriend
         source taskfriend/bin/activate
         ```

2. 在仓库根目录安装依赖，优先使用 CPU 版 torch（避免下载 CUDA 包）：

    - Windows（PowerShell）：
       ```powershell
       python -m pip install --upgrade pip
       pip install --no-cache-dir --extra-index-url https://download.pytorch.org/whl/cpu torch==2.3.1
       pip install --no-cache-dir -r requirements.txt
       ```

    - macOS / Linux：
       ```bash
       python -m pip install --upgrade pip
       pip install --no-cache-dir --extra-index-url https://download.pytorch.org/whl/cpu torch==2.3.1
       pip install --no-cache-dir -r requirements.txt
       ```

    - 国内源（可选，需信任主机）：
       ```bash
       PIP_INDEX_URL=https://pypi.tuna.tsinghua.edu.cn/simple \
       pip install --no-cache-dir --extra-index-url https://download.pytorch.org/whl/cpu torch==2.3.1
       PIP_INDEX_URL=https://pypi.tuna.tsinghua.edu.cn/simple \
       pip install --no-cache-dir -r requirements.txt
       # 或添加 --trusted-host pypi.tuna.tsinghua.edu.cn 以避免证书拦截
       ```

3. 进入课程目录以确保本地包可解析：

      - Windows（PowerShell）：在 PowerShell 终端执行
         ```powershell
         Set-Location "LLM-Engineer-Course"
         ```

      - macOS / Linux：在 Bash/zsh 终端执行
         ```bash
         cd LLM-Engineer-Course
         ```

   > 说明：后续命令默认在 `LLM-Engineer-Course/` 目录内执行。

4. 验证关键依赖（`ms-swift` 安装后导入名为 `swift` 的模块）：

      - Windows（PowerShell）：
         ```powershell
         python -c "import swift, ragas, torch"
         ```

      - macOS / Linux：
         ```bash
         python -c "import swift, ragas, torch"
         ```

   > 如因手动升级导致 `numpy` 升至 2.x 出现 ImportError，可回退：`pip install "numpy<2"`。

5. 配置 DashScope API Key（首次或需要轮换时执行）：

      - Windows（PowerShell）：
         ```powershell
         python config\load_key.py
         ```

      - macOS / Linux：
         ```bash
         python config/load_key.py
         ```

   脚本会写入 `Key.json` 并导出 `DASHSCOPE_API_KEY` 环境变量。

   也可直接手工设置：
   ```powershell
   setx DASHSCOPE_API_KEY "<your_key>"
   ```
   ```bash
   export DASHSCOPE_API_KEY="<your_key>"
   ```

### 在 Jupyter/VS Code 选择内核

- 启动 Notebook 时，在内核列表中选择虚拟环境 `taskfriend`（VS Code: 右上角内核选择器；Jupyter: Kernel → Change Kernel）。
- 若未显示，先激活 venv 后运行 `python -m ipykernel install --user --name taskfriend` 再重启 IDE。

### DashScope Base URL 提醒

- 国内版 DashScope 账号默认使用 `https://dashscope.aliyuncs.com/compatible-mode/v1`，项目在 `functions/llm_utils.py` 已按此配置。
- 国际版 DashScope 账号需要切换到 `https://dashscope-intl.aliyuncs.com/compatible-mode/v1`，请在同一文件中调整 `OpenAI(..., base_url=...)` 的取值后再运行，否则会出现鉴权失败或连接错误。

### DashScope 常见故障排查

- 401 InvalidApiKey：确认 Key 正确且与所用域名（国内/国际）匹配；重新 `export/setx` 后重启终端/内核。
- 429 限流：降低并发，增加重试或 sleep；批量评估时加节流。
- 连接/超时：检查代理/防火墙；必要时在同网络用 `curl https://dashscope.aliyuncs.com` 验证连通。

## 构建与读取 TaskFriend RAG 索引

TaskFriend 依赖 LlamaIndex + DashScope Embedding。默认从 `docs/taskfriend` 读取文档，并将索引持久化在 `knowledge_base/taskfriend`（首次使用时自动创建）。

```python
# 在 "LLM-Engineer-Course" 目录中执行
from taskfriend import rag

# 更新知识库后可重建索引
rag.reindex()

# 读取已持久化的索引并构建查询引擎
index = rag.load_index()
query_engine = rag.query_engine(index)
```

## 启动聊天界面

聊天循环支持直接调用 LLM 或通过 RAG 包装。`functions/llm_utils.py` 已封装 DashScope 的 Qwen 流式接口，可直接套用。

```python
from taskfriend import chat, rag
from functions.llm_utils import get_qwen_stream_response

index = rag.load_index()
engine = rag.query_engine(index)

# 将流式 LLM 函数包装成聊天接口需要的签名
call_llm = chat.wrap_streaming_for_chat(get_qwen_stream_response)
conversation = []

chat.chat_interface(
    full_conversation=conversation,
    query_engine=engine,
    call_llm_fn=call_llm,
    use_context_window=True,
    context_window=2000,
    show_context_preview=False
)
```

可以通过 `use_context_window`、`show_truncated`、`summarize_dropped` 等参数观察 TaskFriend 如何裁剪长对话，上下文默认配置在 `taskfriend/config.py`。

## RAG 质量评估

`taskfriend/evaluation.py` 提供 `Evaluator` 类，集成 Ragas 与 DashScope LLM/Embedding，可对单轮问答或多模型对比打分。

```python
from taskfriend.evaluation import Evaluator
from taskfriend import rag

index = rag.load_index()
engine = rag.query_engine(index, streaming=False)
response = engine.query("TaskFriend 如何帮助缓解任务过载？")

scorecard = Evaluator().evaluate_result(
    question="TaskFriend 如何帮助缓解任务过载？",
    response=response,
    ground_truth="TaskFriend 会引导你重新评估优先级并制定行动计划。"
)
print(scorecard)
```

也可以传入自定义模型/嵌入配置，使用 `compare_models` 与 `compare_embeddings` 对比不同组合的表现。

## 课程 Notebook 实验路线图

12 篇 Notebook 构成由浅入深的学习旅程，相互衔接如下：

- **00 Setting Up the Environment**：完成本地/云端环境配置、依赖安装与 DashScope Key 管理，为后续实验打下基础。
- **01 The LLM Architecture**：回顾 LLM 基础原理、推理流程与部署形态，明确工具链为何物。
- **02 Creating Basic LLM Applications**：基于 00 配好的环境，实现最小可用的问答/对话应用，验证 API 调用链路。
- **03 Bridging Knowledge with RAG**：在 02 的应用上增加文档检索与向量索引，实现依赖知识库的问答。
- **04 Prompt Engineering for Success**：继续使用同一应用场景，系统梳理提示词设计与调优技巧，与 02/03 的代码形成闭环。
- **05 Evaluating RAG Performance**：引入 Ragas 等评估方法，量化 03 中 RAG 流水线的效果，并为 06 的优化提供基线。
- **06 RAG Optimization Techniques**：在 05 的指标基础上，尝试检索召回、重排、缓存等优化手段，对 RAG 作进一步提升。
- **07 Building Agentic AI Applications**：将 RAG 能力扩展到多工具/多阶段 Agent，复用前面构建的任务与评估模块。
- **08 Improving LLM Performance Through Fine-tuning**：探索微调途径，将 02/03 的应用接入调优后的模型，与未调优版本对比。
- **09 Deploying & Serving Your LLM App**：基于前面完成的应用，讨论部署、监控与扩缩容策略，与 07/08 的成果结合。
- **10 Best Practices for Developing & Deploying LLM Apps**：总结端到端流程的工程实践，包括团队协作、CI/CD、指标治理。
- **11 Building a Secure, Resilient AI Assistant**：在 09 的部署方案之上，强化安全、防护与高可用设计，形成全流程闭环。

整体来看，00–02 关注基础搭建，03–06 聚焦 RAG 构建与评估，07–09 延伸至 Agent 与上线阶段，10–11 则将工程规范与安全韧性纳入体系，确保学生能够贯通从原型到生产的完整链路。

## 常用工具模块

- `embedding_viz.py`、`vector_visualization.py`、`visualize_attention.py`：嵌入降维、相似度分析与注意力可视化。
- `metric_barchart.py`、`grader_plot.py`、`rag_eval_table.py`：评估指标可视化与对比图表。
- `html_table.py`：在 Notebook 中快速输出美观的 HTML 表格。
- `token_table.py`、`safe_token_str.py`：估算提示词 token 消耗、规避序列拼接错误。
- `clean_json.py`、`eval_embeddings.py`、`llm_utils.py`：清洗 LLM JSON 输出、批量对比嵌入模型、封装 DashScope 流式客户端。

## 数据资产

- `docs/taskfriend`：TaskFriend 助手的核心知识源。
- `resources/training_dataset_*.jsonl`：用于实验的问答/指令式样本。
- `resources/benchmark_exam.jsonl`：基准测试、考试或自动打分的题库。
- `resources/post.lua`：课程中用于拓展的 Lua 示例脚本。

## 开发提示

- 项目中的 RAG 与聊天模块都依赖 `DASHSCOPE_API_KEY` 环境变量，CI/CD 环境需要手动注入。
- 当启用上下文裁剪时，控制台会输出 ANSI 颜色日志；若终端不支持，可在 `taskfriend/utils.py` 中关闭彩色输出。
- 每次更新 `docs/taskfriend` 后请执行 `rag.reindex()` 刷新向量库。
- DashScope 在高并发时可能返回 429 限速，可在自动化评估时加上重试逻辑。
- 与 Notebook 配套的核心依赖（LlamaIndex、LangChain、Ragas 等）迭代频繁，建议保持 `requirements.txt` 指定的版本以避免 API 变更。
 - Windows 下 `python` 与 `py -3.12` 均可；macOS/Linux 请使用 `python3.12`/`python3`。
