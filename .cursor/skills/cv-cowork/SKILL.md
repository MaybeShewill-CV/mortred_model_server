---
name: cv-cowork
description: >-
  Runs the Mortred landing-score state machine: CTO module scores and P0/P1
  list, three-party consensus, RD implementation on a branch worktree, QA,
  then merge back into sub-agent-co-work. Use when the user invokes
  /cv-cowork or asks to run the sub-agent cowork loop.
disable-model-invocation: true
---

# CV cowork

按 [workflow.md](workflow.md) 跑落地循环。打分口径是仓库根目录的 `cowork/ledger/rubric.md`。你是调度者。角色原文原样写入账本。模块分保持原文。综合分、上限和出口以你写的核算文件为准。

| 步骤 | subagent | 写权限 |
| --- | --- | --- |
| 1、7 打分；2 修订列表并签字 | `cv-cto` | 只读 |
| 2 表决并签字；3 审查签字 | `cv-pm` | 只读 |
| 2 表决并签字；3 起草方案并签字；4 偏离签字 | `cv-tech-lead` | 只读 |
| 3 签字；4 实现和自测 | `cv-rd` | 第 3 步不改文件；第 4 步只改当前 worktree |
| 5 测试 | `cv-qa` | 只改该 worktree 的测试 |

这些角色只在本技能派到该步，或用户点名 `/cv-cto`、`/cv-pm`、`/cv-tech-lead`、`/cv-rd`、`/cv-qa` 时使用。自定义 subagent 不可用时停下来，说明需要新开一轮。不要用通用子代理顶替。

## 开工条件

基线分支是 `sub-agent-co-work`。`git status --porcelain` 非空时停下来，列出脏文件，等用户处理。不要 stash，不要把无关改动带进问题分支。

账本在 `cowork/ledger/`。先读 `index.md`；没有则从第 1 轮第 1 步开始，并写好索引。每次角色返回后、你写账本前，再跑一次 `git status --porcelain`。基线出现角色改动就停止，不提交那些改动。

账本只在基线分支提交，路径只限 `cowork/ledger/`。实现和测试只在问题分支提交。提交说明写清轮次、步骤和问题 id。不提交密钥、`conf/local/`、权重和无关改动。

## 核算

模块调用彼此独立，不要 resume。每次提示词带上 rubric 的绝对路径、模块 id，以及 `cv-cto` 里该模块的必读路径。

九份模块记录齐了之后，用 rubric 的权重计算原始分：各模块（分数 × 权重）之和 / 100，保留一位小数。权重只取 rubric。有 P0 时综合分不超过 7.9；有 P1 且没有 P0 时不超过 8.9。把结果写入 `01-score-check.md` 或 `07-score-check.md`。角色自算分不一致时，在核算文件里写明，状态机采用核算文件。模块记录存在未读路径且分数高于 8.0 时，退回该模块，不进入综合分。

出口由核算文件决定，不采用角色正文里的出口说法。

## 循环

第 7 步的核算就是下一轮的起点。进入下一轮时把 `07-score.md` 复制为下一轮的 `01-score.md`，把 `07-score-check.md` 复制为 `01-score-check.md`，从第 2 步开始。最多 10 轮。

1. **打分。** 按 rubric 的模块顺序逐个调用 `cv-cto`。每个模块一份 `01-modules/<id>.md`。九份都通过未读检查后，再让他写 `01-score.md`，然后你写 `01-score-check.md`。
2. **冻结 P0/P1。** 一轮包含三步：PM 和技术负责人并行表决；CTO 只返回 `列表修订`；PM、技术负责人和 CTO 再并行返回 `第2步签字`。通过条件是三份签字的 `针对版本` 相同，且 `结论` 这一行的值都是 `同意`。否则把不同意的理由交给 CTO 修订下一版。最多两轮。仍不通过就停下，把原文交给用户。通过后写入 `02-p0-p1-final.md`。P0 和 P1 都为空时，跳过第 3–6 步，进入第 7 步。
3. **一个问题一份方案。** 先 P0 后 P1，一次一条。技术负责人先返回方案正文。PM 和 RD 并行签字。有 `不同意` 时，技术负责人把条件写进下一版本正文，版本号加一。然后三方对这一版分别签字。通过条件是三份 `方案版本` 相同，`结论` 都是 `同意`，`条件` 都是 `无`。最多两个版本。仍不通过就停下交给用户。通过后写入 `03-design-final.md`。第 3 步每次调用返回后都检查基线工作区。
4. **实现。** 从当前基线创建分支 `cowork/r<轮次>-<问题id>-<短名>`，用 `git worktree add` 放到仓库外的目录。RD 在这棵树上实现、自测，并在问题分支上提交。提交里要能看到相对基线的代码变更。`偏离` 不是 `无` 时，先让技术负责人返回 `偏离签字`。`不同意` 就退回 RD 改回方案。不在这一步合入。
5. **测试和合入。** QA 在同一工作树按方案测试。有测试改动就在问题分支上提交。你在该 worktree 重跑报告里的每条命令，退出码要和报告一致。报告是 `pass`、每条验收标准都有一行、重跑一致、问题分支相对基线有提交、且基线工作区干净时，在基线上执行 `git merge --no-ff <分支>`。把命令和 merge commit 写入 `05-merge.md`，再把账本提交到基线。`fail`、`blocked`、退出码不一致、冲突，都不合入。`fail` 时 RD 在原分支修，最多两个来回。RD 认为缺陷来自方案时，技术负责人只改冲突的一节并把版本加一，这个修订最多一次，三方必须对新版本重新签字。
6. **清列表。** 从 RD 的「新发现」和 QA 的「残余风险」里，把当前列表以外的问题写入该轮 `discovered.md`。不写入产品工作树，不插入当前队列。当前最终列表做完，进入第 7 步。
7. **复审。** 把本轮 merge、`discovered.md` 和 `git diff` 的路径摘要交给 `cv-cto`。改过的模块重新打分，没改的模块做确认。你重写 `07-score-check.md`。

## 停止

出现任一情况就停，把索引状态写成 `stopped` 或 `done`，并把核算文件和尚未通过的原文交给用户：

- 核算综合分大于 8.5，且 P0、P1 都为空。状态写 `done`。
- 已经做完 10 轮仍未达到出口。
- 第 7 步核算综合分没有高于本轮 `01-score-check.md`。
- 未关闭的 P0/P1 全部是「旧问题改写」。
- 表决、签字、偏离、自测、测试或合入超出上面的次数，基线工作区被角色改脏，重跑结果和报告不一致，或 merge 冲突。

不要 force push，不要硬重置，不要跳过 git hooks。
