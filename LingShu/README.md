# 灵枢 LingShu 2.1.0

基于 `LingShu_审核稳定性修复版_20261111.zip` 完成的代码修订，日期：2026-11-11。保持原有工业运维、多 Agent 协作、独立审核和人工确认流程。新增严格响应校验、角色工具权限与任务内记忆、可执行的本地检查点恢复及回退。

本项目是可运行的比赛原型。
默认的 3 条知识均为模拟资料，不是行业标准。
初赛采用本地 JSON 知识和关键词检索，ChromaDB 可作为决赛扩展。

## 先按这几步启动（Windows / Anaconda Prompt）

**完整解压**，进入压缩包里的 `lingshu` 文件夹；不要只复制 `app.py`。文件夹名字可以更改，程序使用相对项目位置定位配置。关闭旧服务（在旧终端按 Ctrl+C），再启动本版。

在进入之前，我们先配置API，打开streamlit/secrets.tom

- 填入自己的 DeepSeek API Key。

第一次建议使用独立环境，避免 `(base)` 中其他包冲突：

```bat
conda env list
conda create -n lingshu_py312 python=3.12 -y
conda activate lingshu_py312
python --verison
conda activate lingshu_v2
cd /d C:\Users\lenovo\Desktop\LingShu
python knowledge_api.py
```

第二次建议再次创建一个新的Anaconda Prompt（终端）页面
在资源管理器打开包含 `app.py` 的文件夹，复制地址栏的完整路径。下面命令中的路径是示例，必须换成你自己的实际路径：

```bat
conda activate lingshu_py312
python --verison
conda activate agent-contets
cd /d "C:\Users\lenovo\Desktop\lingshu"
python -m pip install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple --prefer-binary # 此代码为清华镜像 方便下载速度更快 不耽误使用
```

然后运行：

```bat
streamlit run app.py
```

排错：

- `dir app.py` 找不到文件：当前目录不对，继续进入解压后的内层 `lingshu` 文件夹。
- Python 环境错误：运行 `python --version`确认已激活 `python=3.12.16`版本需 3.11以上推荐3.12.16。
- 提示无密钥：核对文件位置和名称，关闭服务重新启动；也支持 `DEEPSEEK_API_KEY` 环境变量。
- 旧页面不变：关闭旧终端服务，在新文件夹启动，再刷新页面。不要同时启动多个版本。
  本次在 Python 3.12.16  验证；实际执行 Windows 或 Anaconda 安装。

# 第三方依赖、模型及项目贡献

| 依赖                             | 验证版本    | 元数据声明的许可     | 用途                         |
| ------------------------------ | ------- | ------------ | -------------------------- |
| Streamlit                      | 1.66.1  | Apache-2.0   | 页面与 AppTest                |
| AutoGen AgentChat / Ext / Core | 0.7.5   | MIT          | Agent 与模型客户端、Replay 测试     |
| openai Python SDK              | 1.109.1 | Apache-2.0   | OpenAI 兼容接口客户端，连接 DeepSeek |
| pandas                         | 2.2.3   | BSD 3-Clause | 表格与日志                      |
| Plotly                         | 6.9.0   | MIT          | 事件时间线                      |
| nest-asyncio                   | 1.6.0   | BSD          | 页面异步调用适配                   |
| pydantic                       | 2.14.2  | MIT          | 依赖的数据模型与校验                 |
| protobuf                       | 5.29.6  | BSD 3-Clause | Streamlit 相关序列化依赖          |
| pyarrow                        | 26.0.0  | Apache-2.0   | 表格传输依赖                     |

这不是全部传递依赖清单；实际环境可运行 `python -m pip list` 检查。对外分发额外第三方代码或数据时应保留原许可、来源与版权说明。此文件不为团队原创代码擅自指定开源许可证。

## 使用与状态说明

输入任务后点击“启动灵枢分析”。默认动态编排：LLM 规划领域和是否需要专项分析，平台按依赖、优先级、领域能力与当前任务内负载分派。规划允许时，知识分析与专项分析并行，再诊断、审核和按问题修订。7 个配置角色不代表每轮调用 7 个模型；常见任务调用 4～5 个角色，恢复任务可能更少。默认共享同一个 DeepSeek 模型。

- `PENDING_HUMAN`：文本审核和平台规则检查通过，等待阅读确认。
- `DEMO_CONFIRMED`：仅记录本轮演示确认，不构成现场操作授权。
- `REJECTED`：高风险、未知风险、修订上限、无进展等条件阻断；不能人工确认放行。
- `ERROR`：模型格式、请求或执行异常；不得当成有效风险结论。

同一页面会话、同一任务与配置的重复提交保留原 `run_id` 和结果，不再次调用模型。需要主动新运行时勾选“重新生成”；会产生新的调用，清除旧确认。任务或相关配置变化会新运行。此保护不跨浏览器会话；外部 HTTP 知识内容变化时应主动重新生成。

“Self-loop 失败注入测试”只用于演示闭环：首轮附加明确标注的缺项，后续仍由模型修订、审核，不保证通过；正常案例和性能实测应关闭该开关。不要反复生成直至“通过”来代替问题核实。

## 本地知识、角色配置和工具

编辑 `knowledge_base/documents.json` 可加入有来源、已脱敏的资料，字段为 `id/title/source/text/demo`。编号需唯一；`demo: true` 表示模拟资料。默认检索使用中英文词项匹配，返回来源与 `[KB:编号]`，不匹配时承认无依据，不编造来源。单文件不超过 2MB、1000 条，每条正文不超过 12000 字符。

`config/agents.json` 配置角色名称、版本、能力、领域、提示词、容量、工具及记忆策略。知识角色获准使用 `search_knowledge`；调用前实际检查权限，并记录 `tool_audit`。诊断角色默认使用 `task_window`，保留本次任务该角色最近两条输出，每条最多 1200 字符；当前审核问题优先。它不是跨任务长期记忆，不用于把过去“通过”作为本次放行依据。

可选 HTTP 插件演示：在同一环境另开终端、进入项目目录，运行 `python knowledge_api.py`，再勾选页面“通过本地 HTTP API 检索知识”。服务仅监听回环地址，默认端口 8765；不需要额外数据库。不启用此选项也能完成初赛主流程。

## 恢复和回退

页面“恢复中断任务 / 回退后重新执行”可选择已保存的 2.1.0 版运行记录。支持继续或回退到规划、检索、分析、诊断。每次建立新 `run_id`，只复用连续已完成的主流程节点；回退丢弃目标节点及下游结果。**审核总会重新执行，人工确认总会清除。** 页面标出“已从检查点复用”，本次调用统计不包含父任务的历史费用。

进程被关闭后，检查点在 `.lingshu_checkpoints`，可用命令行显式恢复。将下列文件名替换为本机实际编号

```bat
python recover_run.py --from ".lingshu_checkpoints\实际运行编号.json" --live
python recover_run.py --from ".lingshu_runs\实际运行编号.json" --rollback-to diagnosis --live
```

只支持本地知识模式、同任务、同代码、同角色与知识版本、同模型和运行选项的记录；不同版本或 HTTP 知识模式拒绝复用。离线记录必须用 `--offline`，不能转成真实模型记录。自有可信记录的哈希用于检出不一致，不能证明外部文件可信；不要恢复来历不明的记录。恢复的是计算流程，不撤销设备操作，也不继承旧反馈轮次、长期记忆或安全批准。

整体超时会返回 `ERROR` 并保留内存中已完成的修订和审核历史。文件检查点保存已完成主节点；进程崩溃后的恢复会从这些节点重新进行反馈，不保证复原崩溃前每个未完成 token 或修订轮次。

先跑普通任务、Self-loop 测试、危险请求等案例并导出 JSON，核对输出后录屏。
`evaluation/acceptance_cases.json` 给出 6 条待人工验收用例。需要真实对比数据时显式执行（会产生费用）：

```bat
python benchmark.py --live --dataset evaluation/acceptance_cases.json --repeats 1
```

输出目录含源代码哈希、运行 JSON、CSV 和汇总。填写 CSV 的人工质量与说明列；文本通过率不等于诊断准确率，拒绝也可能是正确表现。此次未调用真实 DeepSeek，未填写人工质量分数，不应把包内离线数字当作参赛效果数据。

## 文件与提交

- `app.py`：页面、当前任务、人工确认、历史预览与恢复入口。
- `platform_runtime.py / registry.py / scheduler.py`：AutoGen 模型适配、角色、编排与调度。
- `lingshu_backend.py / json_contract.py`：审核、Self-loop、严格 JSON 与停止条件。
- `agent_memory.py / knowledge.py / knowledge_api.py`：任务内记忆、知识检索与本地 HTTP 工具。
- `recovery.py / recover_run.py / run_store.py`：可验证检查点、显式恢复和本地记录。
- `tests/ / evaluation/ / verification/`：自动测试、案例与本次真实产生的离线证据。

代码包含真实密钥（但不是上传的使用的文件中，默认使用我们自己的）、旧运行目录、旧版验证截图、虚拟环境或数据库。`SHA256.json` 校验包内文件内容。PPT/PDF/视频是独立提交材料，本包不替代它们。后续接入 ChromaDB、真实资料治理、跨任务记忆、分布式调度和更大规模评测可作为决赛升级，不在当前版本中宣称完成。
