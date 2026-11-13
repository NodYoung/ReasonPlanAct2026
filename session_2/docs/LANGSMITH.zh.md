# LangSmith —— 看清 agent 到底在做什么

## 它是什么?

当你输入「Draw a house」时,幕后会发生好几次 LLM 调用:supervisor 分析你的请求,blueprint planner 把房子拆成部件,part planner 写出运动步骤,reviewer 查看试验图像。这些在终端里全都看不见。

[LangSmith](https://smith.langchain.com) 是一个免费的网页面板,它把每一次调用都记录下来——发给模型的确切 prompt、附带的图像、返回的响应、耗时多久、花了多少 token。可以把它当成 agent 的黑匣子。它不改变应用的任何行为,只是汇报发生了什么。

## 配置(5 分钟,无需改代码)

1. **注册免费账号**:到 [smith.langchain.com](https://smith.langchain.com)(用 Google 或 GitHub 登录都行)。免费额度对这个项目来说远远够用。
2. **创建 API key**:在 LangSmith 里点 **Settings(齿轮图标)→ API Keys → Create API Key**。复制这个 key,它以 `lsv2_` 开头。
3. **写进 `.env`**(和存放 LLM 密钥的是同一个文件):

   ```bash
   LANGSMITH_TRACING=true
   LANGSMITH_API_KEY=lsv2_...你的key...
   LANGSMITH_PROJECT=multi-agent-turtlesim
   ```

4. **照常运行应用**(`pixi run agents`)。就这样——LangChain 会自动检测这些变量并开始上传追踪数据。

要关闭追踪,把 `LANGSMITH_TRACING` 设成 `false`(或者把这几行删掉)。

## 怎么读一条追踪

打开 [smith.langchain.com](https://smith.langchain.com),选择 **multi-agent-turtlesim** 项目(首次运行时自动创建)。你会看到每次 LLM 交互对应一条**追踪(trace)**,点开任意一条即可。

一条追踪是一棵树——外层步骤包含内层步骤。点任意一行都能看到它完整的输入和输出。值得重点看的:

- **supervisor 的回合** —— 系统 prompt、你的消息,以及它决定调用哪个工具(比如 `start_drawing` 或 `move_forward_tool`)。
- **planner 的调用** —— 注入到 prompt 里的确切上下文(实时 turtle 位姿、已完成的部件、reviewer 反馈),以及模型返回的结构化 `MovementPlanOutput`。
- **reviewer 的调用** —— 视觉模型实际看到的那张试验 PNG,以及它通过/驳回的裁决。要搞清楚一个部件*为什么*被驳回,这里是最佳位置。
- **延迟与 token** —— 按每次调用显示,并按追踪汇总(一条**追踪**就是一整个任务回合,例如「draw a house」)。可以看出时间和钱花在哪儿了(通常是 planner 和 reviewer)。

## 为什么值得折腾

- **调试** —— 画歪了的时候,追踪能显示是 blueprint、plan 还是 review 做出了错误决策(日志也能帮上忙)。
- **调 prompt** —— 看清模型到底收到了什么,改一下 `src/agents/` 里的系统 prompt,再对比改动前后的追踪。
- **成本意识** —— 看一个任务真正消耗了多少 token,以及哪个 agent 最贵。
