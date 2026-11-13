# 多智能体 Turtlesim

[English](README.md)

本项目包含一条 LangGraph 多智能体流水线,它使用可配置的 LLM/VLM provider(Gemini、OpenAI、Anthropic 或 Ollama),在 ROS 2 turtlesim 上绘制图形,由**一个 supervisor agent** 编排少数几个专职 agent。

## 环境准备

### 可选:安装 pixi

如果你在 **session 1** 里已经装过 pixi,跳过这一步——不需要重装,直接看下面的 `pixi install`。

否则按你的平台安装 [pixi](https://pixi.sh):

**macOS / Linux**

```bash
curl -fsSL https://pixi.sh/install.sh | bash
```

**Windows**(PowerShell)

```powershell
irm -useb https://pixi.sh/install.ps1 | iex
```

然后重启 shell(或执行 `source ~/.bashrc` / `source ~/.zshrc`),让 `pixi` 进入 `PATH`。用这个命令验证:

```bash
pixi --version
```

### 安装与配置

```bash
cd session_2
pixi install
cp .env.example .env          # API 密钥 + 可选的 LangSmith 追踪(见 docs/LANGSMITH.zh.md)
```

## LLM provider

**provider 与模型:** 编辑 `config.toml` 里的 `[llm]`。
**API 密钥:** 把对应的变量加到 `.env`(参考 `.env.example`)。

```toml
[llm]
provider = "gemini"   # gemini | openai | anthropic | ollama | codex
model = "gemini-3.1-flash-lite"
```

## 运行

**终端 1** —— 仿真器:

```bash
pixi run sim
```

**终端 2** —— agent:

```bash
pixi run agents
```

试试这些指令:

- `Draw a blue square with side length 2.`
- `Draw a house`
- `Move forward 0.1`
- `Set pen to red`
- `How big is the canvas?`

---

## 在本容器实际跑法（ROS 2 Jazzy + conda base + codex）

> 本仓库的容器里**没有 pixi**，且系统 ROS 是 **Jazzy**（而非 `pixi.toml` 写的 Humble），
> 另外统一走 **codex** router（`gpt-6-astra`，复用 `~/.codex` 配置）。因此上面的 `pixi run`
> 流程在本容器用不了，改用下面这套。仿真器由脚本自动启动/关闭，**一个终端即可**。

**一次性准备**（只需做一次）:

```bash
# 依赖（conda base 里补装；若已装过可跳过）
eval "$(/data/miniforge3/bin/conda shell.bash hook)"; conda activate base
pip install "langgraph>=0.2.0" "langchain>=0.3.0"

# provider 已在 config.toml 配好：provider = "codex"、model = "gpt-6-astra"
# codex 的 API key / base_url 自动读取 ~/.codex/auth.json 与 ~/.codex/config.toml，
# 无需 .env（也不需要 gemini/openai 等 key）。
```

**单个终端即可运行**（仿真器由脚本自动管理，无需另开终端）:

```bash
cd /home/robot/github/llm-misc/ReasonPlanAct2026/session_2
source /opt/ros/jazzy/setup.bash                       # 先 source ROS
eval "$(/data/miniforge3/bin/conda shell.bash hook)"   # 再激活 conda base
conda activate base
export PYTHONPATH=".:$PYTHONPATH"                        # 关键：追加，不要覆盖 ROS 路径
python -m src.main                                      # 交互模式
# 或一次性：python -m src.main --request "Draw a blue square with side length 2."
```

**仿真器自动管理：** `src/main.py` 启动时会通过 `sim_launcher.ensure_turtlesim()
**先杀掉所有已存在的 turtlesim 进程，再启动一个全新实例**，并在程序退出时自动关闭它。
因此：

- 不需要再单独开一个终端跑 `ros2 run turtlesim turtlesim_node`（`pixi run sim`）。
- 不会出现「复用到上次残留的坏实例」导致绘制阶段 `0/N steps`、只画出半张图的问题。
- 运行期间**不要手动关闭 turtlesim 窗口**；关掉会让仿真器中途消失，导致剩余步骤画不出来。

**两个容易踩的坑:**

- `PYTHONPATH` 必须用 `".:$PYTHONPATH"` **追加**。若写成 `PYTHONPATH=.`（pixi.toml 里的写法）
  会覆盖掉 `source setup.bash` 设置的 ROS 路径，导致 `ModuleNotFoundError: No module named 'rclpy'`。
- 必须**先** `source /opt/ros/jazzy/setup.bash` **再** `conda activate base`，并在 `session_2` 目录下运行。

生成的试验图像落在 `session_2/captures/`。

> 备注：复杂物体（如 house、flower，4+ 部件、数十步、多次视觉重规划）单次绘制较耗时，
> 用交互模式前台运行即可；**不要用 `timeout` 之类会向进程发信号的方式包裹运行**，
> 信号会被 rclpy 捕获并提前关闭 ROS context，导致中途出现 `ExternalShutdownException`。



## 工作原理

```
User ↔ SUPERVISOR(唯一的编排者,也是唯一与你对话的 agent)
          │  简单指令 / 提问 → 自己处理完就结束(1 次 LLM 调用)
          │  绘图请求 → 确定性地跑下面这条流水线:
          │
          ├─→ planner_blueprint   部件 + 尺寸            (每任务 1 次 LLM 调用)
          ├─→ planner_part        当前部件的步骤          (每试验 1 次 LLM 调用)
          ├─→ plan_reviewer       试验 PNG + 视觉检查      (每试验 ≤1 次 LLM 调用)
          └─→ navigator           在 turtle 上绘制        (不用 LLM)
```

每个专职 agent 只向 supervisor 汇报——它们之间从不互相调用。第一次 supervisor 回合之后,流水线的路由就是一个确定性状态机,因此**编排本身不消耗任何 token**。

### 渐进式试验图像

在任何墨迹落到真实画布之前,reviewer 会把计划**叠加在已经画好的全部内容之上**渲染出来,保存为:

```
captures/{shape}_{part}_trial{N}.png     例如 house_body_trial1.png
```

- 试验被驳回 → 图像 + 修改指示回传给 planner,下一次尝试变成 `..._trial2.png`。
- 试验通过 → navigator 把它画出来,下一个部件的第 1 次试验图像就叠加在它之上。

保存的图像**仅此一类**,所以 captures 目录读起来就是这幅画的成长记录:哪里出了问题、怎么修好的、图案怎么长起来的。

免费的确定性检查(空计划、越出 11×11 米画布的墨迹)会在视觉模型被调用之前就直接驳回。

### 审查相关配置(`config.toml`)

```toml
[review]
enabled = true        # REVIEW_ENABLED=0 可让单次运行跳过审查
max_trials = 5        # 每个部件的规划尝试次数(MAX_TRIALS)
fail_action = "skip"  # 次数用尽时的行为,或改成 "draw_anyway"(FAIL_ACTION)
```

任务结束后 turtle 会停靠到一个空白角落(`park.park_turtle_after_draw = true`)。

## 项目结构

```
src/
  main.py              CLI 入口
  app_settings.py      config.toml 加载器(按 agent 的 LLM 配置)
  llm.py               get_llm("supervisor"|"blueprint"|"planner"|"reviewer")
  pydantic_models.py   结构化 LLM 输出的 schema
  agents/              各 agent(supervisor、两个 planner、reviewer、navigator)
  workflow/            LangGraph 装配 + run_mission()
  tools/               供 supervisor 使用的 turtle 运动 @tool
  utils/               ROS 桥接、墨迹追踪、仿真、渲染、执行
docs/
  ARCHITECTURE.zh.md   系统设计与数据流
  TOOLS.zh.md          工具如何运作 / 如何新增一个
  LANGSMITH.zh.md      可选的追踪:在面板里查看每一次 LLM 调用
```
