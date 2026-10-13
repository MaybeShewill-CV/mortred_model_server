# 技术方案

- 问题: P2-10
- 版本: v1
- 对应列表: `cowork/ledger/round-001/02-p0-p1-final.md` 第 1 轮 v3 的 P2-10

## 边界

只让 `README.zh-cn.md` 里标题为 `### 第一小时：mortredctl` 的这一节，在其 bash 命令块中出现 `mortredctl next`，并且按阅读顺序在这之后仍能读到 `mortredctl doctor`。这一节到下一个 `### ` 标题 `### 从源码构建` 为止。

对照后的现状：该命令块从 `mortredctl init` 列到 `mortredctl doctor`、`mortredctl doctor --strict` 和 `mortredctl status | catalog`，整份 `README.zh-cn.md` 没有 `mortredctl next`。`README.md` 的首小时说明和 First-hour 命令块、`docs/deployment.md` 的首小时主航道，都已从 `mortredctl next` 走到 `doctor --strict`。`docs/deployment.zh-cn.md` 开头同样写了反复执行 `mortredctl next` 直到 `doctor --strict`。`scripts/mortredctl_next.sh` 最后打印的一条是 `mortredctl doctor --strict`。UX2 已由英文 README 与 `docs/deployment.md` 满足，本条不把 UX2 写成未满足。

中文这一节里已经有 `mortredctl doctor`。补上 `mortredctl next` 之后，只看这一节就能从 next 走到 doctor，不必与英文使用同一组注释，也不必改英文文档或两份部署文档。上一版审查意见无。

P2-11 不在本方案内。不改产品代码、配置和脚本行为。不启动 gateway、supervisor 或模型进程。不生成 engine，不跑 calibrate，不跑 HTTP 压测。GPU、显存、worker 数和 TensorRT execution context 不作为本条通过。

## 做法

### 推荐

只改 `README.zh-cn.md`。在「第一小时：mortredctl」这一节的 bash 命令块内增加一行，去掉行尾 `#` 注释并去掉首尾空白后等于 `mortredctl next`。这一行放在该块已有的 `mortredctl doctor` 之前。块内现有的 `mortredctl init`、`init-trust`、`init-edge`、`prepare`、`calibrate`、`mortredctl doctor`、`mortredctl doctor --strict`、`status | catalog` 都保留。行尾可以加一句中文或英文注释，注释不能代替这条命令。

代价：只动一份中文 README 的一个命令块。读者只打开这一节就能看到 `mortredctl next`，并在其后读到 `mortredctl doctor`。不改已满足 UX2 的英文路径，不改脚本每次只打印一条下一条命令的行为。

### 否决

做法二：只在该代码块外面的说明句里写 `mortredctl next`，代码块仍从 `mortredctl init` 列到 `doctor`。代价更小，但打开命令块仍然看不到 `mortredctl next`。锁定句要求命令块里出现这条命令。否决。

做法三：重写英文 README、`docs/deployment.md`、`docs/deployment.zh-cn.md`，或改 `scripts/mortredctl_next.sh`，让四处命令逐字相同。英文 README 与 `docs/deployment.md` 已经能从 `mortredctl next` 走到 `doctor --strict`，UX2 已满足。中文块补上 `mortredctl next` 并保留节内已有的 `mortredctl doctor` 就能走到 doctor，不必改英文。`docs/deployment.zh-cn.md` 开头已经写了这条主航道。改脚本会改变现有下一条命令。否决。

## 改动面

只改 `README.zh-cn.md` 中「第一小时：mortredctl」这一节的 bash 命令块：在已有 `mortredctl doctor` 行之前增加命令 `mortredctl next`。

## 明确不改

`README.md`、`docs/deployment.md`、`docs/deployment.zh-cn.md`、`scripts/mortredctl_next.sh` 以及其余脚本。产品 toml、pack、gateway、supervisor、模型进程、engine、static shape、batch，以及每个 worker 各自的 TensorRT execution context。该命令块里已有的各行，包括 `mortredctl doctor` 与 `mortredctl doctor --strict`。该节代码块之后关于 `mortredctl prepare` 的那句，以及「从源码构建」及以后的章节。P2-11 的 calibrate 与 `--raw` 报告清单。

## 契约

配置、HTTP 和 tensor 契约不变。`mortredctl next` 仍是仓库里已有的子命令，由 `scripts/mortredctl_next.sh` 每次打印一条下一条命令，最后一条仍是 `mortredctl doctor --strict`。本条不改变这些输出，也不新增或删除子命令。

## 性能与资源

不拉起常驻进程，不占显存，不生成或加载 engine。worker 数、decode 路径和 TensorRT execution context 的归属保持现状。本条只改变中文 README 这一节能读到的命令，没有吞吐或占用变化。

## 验收标准

1. 打开 `README.zh-cn.md`，从 `### 第一小时：mortredctl` 到下一个 `### ` 标题 `### 从源码构建` 之前，其中的 bash 代码块有一行，去掉行尾 `#` 注释并去掉首尾空白后等于 `mortredctl next`。
2. 同一节内，这一行之后仍能读到 `mortredctl doctor`。该代码块里现有的 `mortredctl doctor` 与 `mortredctl doctor --strict` 都算读到。
3. 第 1 条和第 2 条只打开这一节即可判定。不打开 `README.md` 或 `docs/deployment.md`，也不执行 `mortredctl`。
4. 本条产品 diff 只有 `README.zh-cn.md`。`README.md`、`docs/deployment.md`、`docs/deployment.zh-cn.md`、`scripts/mortredctl_next.sh` 与改动前内容相同。

## 测试义务

QA 做白盒，只读 `README.zh-cn.md` 的「第一小时：mortredctl」这一节，并核对第 4 条的 diff 范围。不拉起 gateway、supervisor 或模型进程，不跑 calibrate，不跑压测。UX2 保持已满足，不把英文文档没有改动记成失败。仓库里原有未跟踪的 ledger 不计入本条。执行下面两条自测命令，期望退出码都是 0。

## 自测命令

在 WSL Ubuntu 中执行。两条都期望退出码 0。

```bash
cd /mnt/g/Codex/mortred_model_server
python3 - <<'PY'
from pathlib import Path
import sys
text = Path("README.zh-cn.md").read_text(encoding="utf-8")
start = text.find("### 第一小时：mortredctl")
if start < 0:
 sys.exit(1)
rest = text[start:]
nxt = rest.find("\n### ", 1)
section = rest if nxt < 0 else rest[:nxt]
fence = section.find("```bash")
if fence < 0:
 sys.exit(1)
close = section.find("```", fence + len("```bash"))
if close < 0:
 sys.exit(1)
block = section[fence + len("```bash"):close]
cmds = [ln.split("#", 1)[0].strip() for ln in block.splitlines()]
if "mortredctl next" not in cmds:
 sys.exit(1)
at = section.find("mortredctl next")
if "mortredctl doctor" not in section[at + len("mortredctl next"):]:
 sys.exit(1)
sys.exit(0)
PY
```

期望退出码：0

```bash
cd /mnt/g/Codex/mortred_model_server
git diff --exit-code -- README.md docs/deployment.md docs/deployment.zh-cn.md scripts/mortredctl_next.sh
```

期望退出码：0

## RD 不得自行决定

- 不得修改 `README.md`、`docs/deployment.md`、`docs/deployment.zh-cn.md`、`scripts/mortredctl_next.sh`。
- 不得把 `mortredctl next` 只写在 bash 代码块之外。
- 不得从该代码块删除已有的 `mortredctl doctor` 或 `mortredctl doctor --strict`，也不得改写块内其余命令来替换它们。
- 不得修改 `mortredctl` 的子命令实现、产品 toml、pack、gateway、supervisor、模型进程或 engine。
- 不得执行 `mortredctl next`、`doctor`、`prepare` 或 `calibrate` 来当作本条通过。
- 不得把 UX2 改记为未满足，不得把本条扩到 P2-11，不得改「从源码构建」及以后的章节。
