# 技术方案

- 问题: D-5 deployment.md §15 FAQ 结构损坏
- 版本: v1
- 对应列表: `cowork/ledger/round-010/02-p0-p1-final.md` 版本 v2 的劣势 D-5

## 边界

轮次 10。上一版审查意见：无。

只修 `docs/deployment.md` §15 里已经能观察到的结构损坏。`## 15. FAQ` 到 `## 16. Acceptance Gates` 之间，前六条 `<details>` 各自有 `<summary>` 并且已经闭合。`Where do I set worker_nums?` 这条 `</details>` 之后、节末 `---` 之前，残留一段没有 `<summary>` 的段落，并以一个多余的 `</details>` 结束。残留原文是：在有 python 的机器上 fetch，把整个 `weights/` 目录拷走，在目标机运行 `fetch_weights.py --check`；完全没有 python 时，先校验再拷贝。

读者会在 FAQ 里碰到这段残句。冻结后能观察到的变化是 §15 可折叠结构完整：开闭标签成对，每条有 summary，节内没有落在 details 外面的段落，也没有多余的结束标签。完整离线流程仍以 §9.4 现文为准。本条不新增 FAQ 条目，不改 §9.4 的措辞。

不涉及 P0-1、P1-1 至 P1-6、D-1 至 D-4、D-6。

## 做法

### 推荐

只删 §15 里这段无 summary 的残留段落，以及紧跟它的那个多余 `</details>`。六条既有 FAQ 的 summary 和正文保持原文，节末通往 §16 的 `---` 保留。

代价：FAQ 里不再出现这句离线拷贝残文，其中包括「无 python 时先校验再拷贝」。读者按 §9.4 现有的两句做离线拷贝。本条不把这句写进 §9.4，也不为它补 summary。改动是删除 §15 末尾残留，不重排其余章节。

### 否决

给残留段落补上 `<details>` 和 `<summary>`，保留那段正文和现有的 `</details>`，使之成为第七条 FAQ。代价是要新写一条冻结文本里不存在的 summary，并且 FAQ 会与 §9.4 并列两套离线拷贝说法；残留里的 python-less 句比 §9.4 多出一层条件。冻结后的可观察变化是现有 FAQ 可折叠结构完整。模块记录已写明完整离线流程在 §9.4，这段是残留。

把残留并进「Must weights live in /opt/mortred/weights?」的正文，或改写 §9.4 以吸收 python-less 句，或新增一条只指向 §9.4 的 FAQ。代价是改动已经闭合的条目，或新写文案，或改动 §9.4。冻结句要去掉的是无 summary 段落和多余的 details 结束标签。

## 改动面

- `docs/deployment.md`：仅 §15 中 `Where do I set worker_nums?` 条目闭合之后、`---` 之前的无 summary 段落，以及紧随的多余 `</details>`。

## 明确不改

- §15 已有六条 FAQ 的 summary 与正文。
- `### 9.4 Offline environments` 的正文，包括联网机 fetch、拷贝到目标机、`fetch_weights.py --check`，以及依赖用 `--offline DIR`。
- `### 14.3 Deep dives` 的两个 details。
- §16 及文档其余章节、目录链接。
- `README.md`、`README.zh-cn.md`。D-6 的路径 B 不在本条。
- `src/`、`conf/`、`scripts/`、`deploy/`、HTTP 路由、tensor 形状、状态码。
- gateway :8080、supervisor :8787、gpu profile、`mortredctl prepare`、engine 文件。
- cowork 账本。

## 契约

HTTP 路径、请求体、响应字段和 tensor 布局不变。不新增配置键，不新增状态码。

文档结构只变 §15：该节 `<details>`、`<summary>`、`</summary>`、`</details>` 各 6 个，顺序为 details 开、summary 开、summary 关、details 关，重复六次，不嵌套。六条 summary 原文与顺序不变。六条正文在去掉首尾空行和每行行尾空白后与下列原文一致。节内这六块之外只允许标题行、空行和单独成行的 `---`。

六条 summary 原文，按出现顺序：

1. `How do I verify long-task POST /jobs returns 202 immediately?`
2. `Will the cpu profile support more models over time?`
3. `Can I run cpu and gpu side by side?`
4. `Any functional difference between the Docker and tarball tracks?`
5. `Must weights live in /opt/mortred/weights?`
6. `Where do I set worker_nums?`

六条正文原文：

```text
Follow the step-by-step customer procedure in
[async-jobs-customer-test.md](async-jobs-customer-test.md). `POST /v1/models/{id}/jobs`
must return in milliseconds while the same payload on `/infer` still takes the
full sampling time. Immediate `/result` must be 409.
```

```text
Yes, version by version via the formal process in §8.2 - but it stays a curated
set: every addition carries CPU-performance and acceptance ownership. See
CHANGELOG for the extension history.
```

```text
One runtime profile per machine (`MORTRED_PROFILE`). To serve both load classes,
deploy two instances with distinct ports.
```

```text
None. Same binaries, same configs, same acceptance script. Choose by ops habit.
```

```text
Model configs resolve paths relative to the install tree. The tarball track
defaults to `/opt/mortred/weights`; the Docker track mounts into that container
path - the host-side location is up to you.
```

```text
Git `conf/server` stays `worker_nums=1`. Put concurrency on the machine pack
(`[pack.<ID>] worker_nums`) and inject via `MORTRED_PACK`. Calibrate with
`mortredctl calibrate`; persist with `--write-pack` on a machine-local pack
file, then restart the supervisor (§10.3).
```

`docs/deployment.md` 全文不再含 `Fetch on any machine with python`，也不再含 `python-less`。

§9.4 仍含 `Fetch `weights/` on a networked machine → copy to the target → run`，以及 `` `fetch_weights.py --check`. Dependencies work the same way (`--offline DIR`). ``。

## 性能与资源

一个 catalog id 仍是一个 OS 进程。gateway 仍是 :8080，supervisor 仍是 :8787。本条不增加进程，不加载权重，不重建 engine，不改变显存。cpu profile 不因此链接 TensorRT。gpu profile 仍是本机本卡用 `mortredctl prepare` 生成 engine。

## 验收标准

比较正文时，去掉首尾空行，并去掉每行行尾空白。换行和用词参与比较。

1. `## 15. FAQ` 到 `## 16. Acceptance Gates` 之前，标签序列恰好是 `<details>`、`<summary>`、`</summary>`、`</details>` 重复 6 次。没有嵌套，没有第 7 组，没有落在这个序列之外的上述标签。
2. 从该切片去掉这 6 个完整 details 块和标题 `## 15. FAQ` 之后，剩下的非空行只能是单独成行的 `---`。
3. 该切片的 6 条 summary 原文和顺序与「契约」中的六条一致。
4. 该切片的 6 条正文按上述空白规则与「契约」中的六段原文一致，且顺序与 summary 一一对应。
5. `docs/deployment.md` 全文不含 `Fetch on any machine with python`，也不含 `python-less`。
6. `### 9.4 Offline environments` 到 `## 10. Machine pack and TensorRT` 之前仍含 `Fetch `weights/` on a networked machine → copy to the target → run` 和 `` `fetch_weights.py --check`. Dependencies work the same way (`--offline DIR`). ``，且该切片不含第 5 条的两句残留用语。
7. `### 14.3 Deep dives` 到 `## 15. FAQ` 之前，标签序列恰好是 `<details>`、`<summary>`、`</summary>`、`</details>` 重复 2 次。两条 summary 仍是 `Why does the supervisor bind model servers to loopback?` 与 `A model stopped working after an upgrade - now what?`。

## 测试义务

黑盒：在仓库根目录执行「自测命令」。通过时期望退出码 0，任一断言失败时期望退出码 1。该命令覆盖验收第 1 条至第 7 条。不启动 gateway 或 supervisor，不要求权重、GPU 或 engine。

白盒：diff 只删除 §15 中 worker_nums 条目闭合之后的无 summary 段落和多余 `</details>`。六条既有 FAQ 的 summary 与正文无改动。§9.4、§14.3、`README.zh-cn.md` 无改动。本条通过不作为 D-6 或其他 id 的关闭证据。QA 不把中文 README 路径 B 是否仍写 `setup_full_deps.sh` 当作本条通过条件。

## 自测命令

在仓库根目录、能运行 `python3` 的 shell 中执行。通过时期望退出码 0。任一断言失败时期望退出码 1。

```bash
python3 - <<'PY'
import pathlib, re, sys

root = pathlib.Path(".")
text = (root / "docs/deployment.md").read_text(encoding="utf-8").replace("\r\n", "\n")

def fail(msg):
    print(msg)
    raise SystemExit(1)

def between(src, start, end):
    s = src.find(start)
    if s < 0:
        fail("missing " + start.strip())
    e = src.find(end, s + len(start))
    if e < 0:
        fail("missing " + end.strip())
    return src[s:e]

def norm(s):
    return "\n".join(ln.rstrip() for ln in s.strip().splitlines())

def tags(src):
    return re.findall(r"</?details>|</?summary>", src)

sec15 = between(text, "## 15. FAQ\n", "## 16. Acceptance Gates")
sec94 = between(text, "### 9.4 Offline environments\n", "## 10. Machine pack and TensorRT")
sec143 = between(text, "### 14.3 Deep dives\n", "## 15. FAQ")

cycle = ["<details>", "<summary>", "</summary>", "</details>"]
if tags(sec15) != cycle * 6:
    fail("ac1")

stripped = re.sub(r"<details>\s*<summary>.*?</summary>.*?</details>", "", sec15, flags=re.S)
stripped = stripped.replace("## 15. FAQ", "", 1)
for line in stripped.splitlines():
    if line.strip() not in ("", "---"):
        fail("ac2 " + line)

summaries = [
    "How do I verify long-task POST /jobs returns 202 immediately?",
    "Will the cpu profile support more models over time?",
    "Can I run cpu and gpu side by side?",
    "Any functional difference between the Docker and tarball tracks?",
    "Must weights live in /opt/mortred/weights?",
    "Where do I set worker_nums?",
]
bodies = [
    "Follow the step-by-step customer procedure in\n"
    "[async-jobs-customer-test.md](async-jobs-customer-test.md). `POST /v1/models/{id}/jobs`\n"
    "must return in milliseconds while the same payload on `/infer` still takes the\n"
    "full sampling time. Immediate `/result` must be 409.",
    "Yes, version by version via the formal process in §8.2 - but it stays a curated\n"
    "set: every addition carries CPU-performance and acceptance ownership. See\n"
    "CHANGELOG for the extension history.",
    "One runtime profile per machine (`MORTRED_PROFILE`). To serve both load classes,\n"
    "deploy two instances with distinct ports.",
    "None. Same binaries, same configs, same acceptance script. Choose by ops habit.",
    "Model configs resolve paths relative to the install tree. The tarball track\n"
    "defaults to `/opt/mortred/weights`; the Docker track mounts into that container\n"
    "path - the host-side location is up to you.",
    "Git `conf/server` stays `worker_nums=1`. Put concurrency on the machine pack\n"
    "(`[pack.<ID>] worker_nums`) and inject via `MORTRED_PACK`. Calibrate with\n"
    "`mortredctl calibrate`; persist with `--write-pack` on a machine-local pack\n"
    "file, then restart the supervisor (§10.3).",
]
found = re.findall(r"<summary>(.*?)</summary>(.*?)</details>", sec15, flags=re.S)
if [norm(s) for s, _ in found] != summaries:
    fail("ac3")
if [norm(b) for _, b in found] != [norm(b) for b in bodies]:
    fail("ac4")

for phrase in ("Fetch on any machine with python", "python-less"):
    if phrase in text:
        fail("ac5")
    if phrase in sec94:
        fail("ac6")

if "Fetch `weights/` on a networked machine → copy to the target → run" not in sec94:
    fail("ac6 fetch")
if "`fetch_weights.py --check`. Dependencies work the same way (`--offline DIR`)." not in sec94:
    fail("ac6 check")

if tags(sec143) != cycle * 2:
    fail("ac7 tags")
sum143 = re.findall(r"<summary>(.*?)</summary>", sec143, flags=re.S)
expect143 = [
    "Why does the supervisor bind model servers to loopback?",
    "A model stopped working after an upgrade - now what?",
]
if [norm(s) for s in sum143] != expect143:
    fail("ac7 summary")
PY
```

## RD 不得自行决定

- 不得新写 summary，不得把残留收成第七条 FAQ，不得把残留并进已有六条。
- 不得改六条既有 FAQ 的用词、顺序或换行。
- 不得把 python-less 或「先校验再拷贝」写进 §9.4，不得改 §9.4 现有两句。
- 不得改 §14.3、§16、目录，以及其他章节的 details。
- 不得改 `README.md` 或 `README.zh-cn.md`，不得顺手修 D-6。
- 不得改推理代码、配置、脚本、HTTP 契约、tensor、进程拓扑、端口或 engine。
- 不得用格式化工具重排 `docs/deployment.md` 的其余部分。
