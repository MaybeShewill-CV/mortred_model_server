# 状态机记录

编排方是当前对话里的主代理。subagent 看不到这段对话，只看提示词和流程记录里的文件。只读角色的正文由主代理原样保存。模块调用不使用 resume；每次提示词都带齐要读的绝对路径。

基线分支是 `sub-agent-co-work`。问题分支是 `cowork/r<轮次>-<问题id>-<短名>`。实现和测试发生在该分支的独立 worktree 里，并提交在该分支上。QA 报告经编排方在 worktree 重跑确认后，由编排方把该分支 `--no-ff` 合入基线。合入、阻塞或该问题停止后，删除该 worktree，并删除没有进入合入提交的临时脚本。

评分标准只在 `cowork/ledger/rubric.md`。加权分以 `cowork/ledger/round-NNN/01-score-check.md` 和 `07-score-check.md` 为准。

新记录直接写在 `cowork/ledger/round-NNN/`。第 1 轮是 `cowork/ledger/round-001/`。不要写入 `cowork/ledger/active/`。不要把 git 历史里的旧轮次检回工作区，也不从中继承队列、已关闭 id、分数或阻塞项。

## 目录

`round-001` 代表 `cowork/ledger/round-NNN`，轮次用三位数字。记录都在 `cowork/ledger/` 下，不再套一层目录。

```text
cowork/ledger/rubric.md
cowork/ledger/index.md
cowork/ledger/round-NNN/01-modules/<module-id>.md
cowork/ledger/round-NNN/01-score.md
cowork/ledger/round-NNN/01-score-check.md
cowork/ledger/round-NNN/02-discussion/vN-{pm,tech-lead}.md
cowork/ledger/round-NNN/02-discussion/vN-cto-list.md
cowork/ledger/round-NNN/02-discussion/vN-sign-{cto,pm,tech-lead}.md
cowork/ledger/round-NNN/02-p0-p1-final.md
cowork/ledger/round-NNN/discovered.md
cowork/ledger/round-NNN/issues/<issue-id>/03-discussion/vN-tech-lead.md
cowork/ledger/round-NNN/issues/<issue-id>/03-discussion/vN-sign-{pm,rd,tech-lead}.md
cowork/ledger/round-NNN/issues/<issue-id>/03-design-final.md
cowork/ledger/round-NNN/issues/<issue-id>/04-rd.md
cowork/ledger/round-NNN/issues/<issue-id>/04-deviation-sign.md
cowork/ledger/round-NNN/issues/<issue-id>/05-qa.md
cowork/ledger/round-NNN/issues/<issue-id>/05-merge.md
cowork/ledger/round-NNN/issues/<issue-id>/05-suspend.md
cowork/ledger/round-NNN/07-modules/<module-id>.md
cowork/ledger/round-NNN/07-score.md
cowork/ledger/round-NNN/07-score-check.md
cowork/ledger/round-NNN/08-writeoff.md
```

`index.md` 里的加权分使用评分复核的加权分：

```markdown
# Ledger

- 基线分支: sub-agent-co-work
- 轮次上限: 20
- 当前轮次:
- 当前步骤:
- 当前问题:
- 阻塞:
- 已关闭:
- 本轮起点加权分:
- 状态: running | stopped | done | awaiting-env

## 轮次
| 轮次 | 复核加权分 | 未关闭 P0 | 未关闭 P1 | 未关闭 P2 | 结果 |
```

## 提示词

每次启动都带上：本步编号、轮次、要返回的模板名、`cowork/ledger/rubric.md` 的绝对路径、要读的流程记录绝对路径。第 4、5 步带上 worktree 绝对路径，并写明只修改这个 worktree；构建、自测和测试在本机 WSL Ubuntu 里执行，仓库路径是 `/mnt/g/Codex/mortred_model_server`。第 3 步写明不要修改仓库。

第 1 步的每个模块调用都带上 `cv-cto` 里该模块的必读路径。加权分调用附上九份模块记录的路径。第 7 步只要求 `关闭记录` 模板，附上已合入 id、各自绑定的问题陈述、阻塞项、`discovered.md` 和 diff 路径摘要。已有锁定队列的轮次不重新评分，从第一条未关闭且未阻塞的问题继续。

第 2 步的评审、列表修订、确认分成三次调用。第 3 步的方案正文和确认分成不同调用。

## 评分复核

由编排方书写：

```markdown
# 评分复核

- 轮次:
- 原始分:
- 加权分:
- 上限规则: 无 | P0 上限 7.9 | P1 上限 8.9
- 相对本轮起点:
- 终态: 继续 | 结束 | 待环境
- 开放项: 相对本轮起点减少 | 没有减少
- 阻塞:
- 与角色自评: 一致 | 不一致

## 完成标准
| 编号 | 状态: 已满足 / 未满足 / 阻塞 | 证据或队列 id |
```

`done` 还要求上表每一行都是「已满足」。P0、P1、P2 都是「无」且阻塞项为空，只说明旧队列清空，不单独构成终态。

## 锁定的交付范围

三方确认通过后，编排方把列表正文和确认抄成 `02-p0-p1-final.md`：

```markdown
# 交付队列

- 轮次:
- 版本:
- 加权分上限:

## P0
| id | 问题 | 场景 | 现在的行为 | 改完后能观察到的变化 | 依据 |
## P1
| id | 问题 | 场景 | 现在的行为 | 改完后能观察到的变化 | 依据 |
## P2
| id | 问题 | 场景 | 现在的行为 | 改完后能观察到的变化 | 依据 |
## 附录
## 确认
| 角色 | 结论 | 针对版本 | 记录 |
```

文件名仍是 `02-p0-p1-final.md`。P0、P1、P2 三节都是「无」且阻塞项为空时，不进入实现。完成标准仍有「未满足」时，不得把这三节写成「无」。已确认且未关闭、未阻塞的 id 在下一轮原样保留，不再评审。

## 通过的方案

三方对同一版本确认通过后，或用户在两版仍未统一时从决策选项里选定一版后，编排方把方案正文抄成 `03-design-final.md`，并附上：

```markdown
## 确认
| 角色 | 结论 | 条件 | 方案版本 | 记录 |
```

用户从决策选项选定一版时，确认表保留各方对那一版的原结论，并增加一行「用户」。文末再写：

```markdown
## 用户决定
- 日期:
- 选用版本:
- 决定原文:
- 未选用:
```

## 本轮新增问题

由编排方写入 `discovered.md`，不放进产品 worktree：

```markdown
# 本轮新增问题

| 来源 | 问题 | 依据 |
```

没有新增问题时写「无」。

## 合入记录

由编排方书写。`重跑` 记录在 worktree 里重跑 QA 命令的退出码：

```markdown
# 合入

- 问题:
- 分支:
- worktree:
- 问题分支提交:
- QA 结论:
- 重跑:
- 命令:
- merge commit:
```

## 阻塞记录

QA 结论为 `blocked` 时由编排方书写。不合入，继续下一条。

```markdown
# 阻塞

- 问题:
- 方案版本:
- 分支:
- worktree:
- 问题分支提交:
- QA 结论: blocked
- 缺的环境:
- 恢复条件:
- 未跑的验收:
- 重跑:
```

## 关闭记录

由编排方按 `cv-cto` 的关闭记录模板核对后书写。已关闭 id 的问题陈述不得留在「仍开放的问题」里。

```markdown
# 关闭记录

- 轮次:
- 已关闭:
- 阻塞:
- 未阻塞的开放项: 相对本轮起点减少 | 没有减少

## 已关闭的问题陈述
| id | 模块 | 问题陈述 |
## 仍开放的问题
| id | 模块 | 问题陈述 | 状态 |
## 附录
## 新入列
```

`状态` 取 `未做` 或 `阻塞`。没有仍开放的问题时，该节写「无」。
