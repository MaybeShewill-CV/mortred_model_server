# 状态机记录

调度者是当前对话里的主代理。subagent 看不到这段对话，只看提示词和账本里的文件。只读角色的正文由主代理原样保存。模块调用不使用 resume；每次提示词都带齐要读的绝对路径。

基线分支是 `sub-agent-co-work`。问题分支是 `cowork/r<轮次>-<问题id>-<短名>`。实现和测试发生在该分支的独立 worktree 里，并提交在该分支上。QA 报告经调度者在 worktree 重跑确认后，由调度者把该分支 `--no-ff` 合入基线。

打分口径只在 `cowork/ledger/rubric.md`。综合分以 `01-score-check.md` 和 `07-score-check.md` 为准。

## 目录

`round-001` 代表 `round-NNN`，轮次用三位数字。

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

`index.md` 里的综合分使用核算文件的加权综合分：

```markdown
# Ledger

- 基线分支: sub-agent-co-work
- 轮次上限: 20
- 当前轮次:
- 当前步骤:
- 当前问题:
- 挂起:
- 已销账:
- 本轮起点综合分:
- 状态: running | stopped | done | awaiting-env

## 轮次
| 轮次 | 核算综合分 | 未关闭 P0 | 未关闭 P1 | 未关闭劣势 | 结果 |
```

## 提示词

每次启动都带上：本步编号、轮次、要返回的模板名、`cowork/ledger/rubric.md` 的绝对路径、要读的账本文件绝对路径。第 4、5 步带上工作树绝对路径，并写明只修改这个工作树。第 3 步写明不要修改仓库。

第 1 步的每个模块调用都带上 `cv-cto` 里该模块的必读路径。综合分调用附上九份模块记录的路径。第 7 步只要求 `销账` 模板，附上已合入 id、各自绑定的劣势原句、挂起项、`discovered.md` 和 diff 路径摘要。已有冻结队列的轮次不重打分，从第一条未关闭且未挂起的问题继续。

第 2 步的表决、列表修订、签字分成三次调用。第 3 步的方案正文和签字分成不同调用。

## 核算文件

由调度者书写：

```markdown
# 分数核算

- 轮次:
- 原始综合分:
- 加权综合分:
- 上限规则: 无 | P0 上限 7.9 | P1 上限 8.9
- 相对本轮起点:
- 出口: 继续 | 结束 | 待环境
- 劣势集合: 相对本轮起点减少 | 没有减少
- 挂起:
- 与角色自算: 一致 | 不一致

## 双目标
| 编号 | 状态: 已满足 / 未满足 / 挂起 | 证据或队列 id |
```

`done` 还要求上表每一行都是「已满足」。P0、P1、劣势都是「无」且挂起集为空，只说明旧队列清空，不单独构成出口。

## 冻结列表

三方签字通过后，调度者把列表正文和签字抄成 `02-p0-p1-final.md`：

```markdown
# 交付队列

- 轮次:
- 版本:
- 综合分上限:

## P0
| id | 问题 | 场景 | 现在的行为 | 改完后能观察到的变化 | 依据 |
## P1
| id | 问题 | 场景 | 现在的行为 | 改完后能观察到的变化 | 依据 |
## 劣势
| id | 问题 | 场景 | 现在的行为 | 改完后能观察到的变化 | 依据 |
## 附录
## 签字
| 角色 | 结论 | 针对版本 | 记录 |
```

文件名仍是 `02-p0-p1-final.md`。P0、P1、劣势三节都是「无」且挂起集为空时，不进入实现。双目标仍有「未满足」时，不得把这三节写成「无」。已签字且未关闭、未挂起的 id 在下一轮原样保留，不再表决。

## 通过的方案

三方对同一版本签字通过后，调度者把方案正文抄成 `03-design-final.md`，并附上：

```markdown
## 签字
| 角色 | 结论 | 条件 | 方案版本 | 记录 |
```

## 本轮新发现

由调度者写入 `discovered.md`，不放进产品工作树：

```markdown
# 本轮新发现

| 来源 | 问题 | 依据 |
```

没有新发现时写「无」。

## 合入记录

由调度者书写。`重跑` 记录在 worktree 里重跑 QA 命令的退出码：

```markdown
# 合入

- 问题:
- 分支:
- 工作树:
- 问题分支提交:
- QA 结论:
- 重跑:
- 命令:
- merge commit:
```

## 挂起记录

QA 结论为 `blocked` 时由调度者书写。不合入，继续下一条。

```markdown
# 挂起

- 问题:
- 方案版本:
- 分支:
- 工作树:
- 问题分支提交:
- QA 结论: blocked
- 缺的环境:
- 恢复条件:
- 未跑的验收:
- 重跑:
```

## 销账记录

由调度者按 `cv-cto` 的销账模板核对后书写。已关闭 id 的原句不得留在「留下的劣势」里。

```markdown
# 销账

- 轮次:
- 已关闭:
- 挂起:
- 非挂起劣势: 相对本轮起点减少 | 没有减少

## 删除的原句
| id | 模块 | 原句 |
## 留下的劣势
| id | 模块 | 原句 | 状态 |
## 附录
## 新准入
```

`状态` 取 `未做` 或 `挂起`。没有留下的劣势时，该节写「无」。

