# 技术方案

- 问题: D-6 中文 README 路径 B 与现行 full 构建入口不一致
- 版本: v1
- 对应列表: `cowork/ledger/round-010/02-p0-p1-final.md` 版本 v2 的劣势 D-6

## 边界

轮次 10。上一版审查意见：无。

只改 `README.zh-cn.md` 的「路径 B：full build」，使按这一节从源码构建的人看到的依赖命令和编译命令与现行入口相同。现行入口以这三处为准：`README.md` 的 Building from source，`docs/deployment.zh-cn.md` §7.1 与 §7.2 里名为 full、full-cpu 的命令，`CMakePresets.json` 里名为 `full` 与 `full-cpu` 的 configure preset 及其同名 build preset。

冻结后能观察到的变化是：路径 B 使用 `install_deps.sh` 与 `cmake --preset full` / `full-cpu`，切片里不再出现 `setup_full_deps.sh` 和手写 `cmake ..`。gpu 与 cpu 仍是两档，写在同一个代码块里，说明要求只执行其中一档。

不涉及 P0-1、P1-1 至 P1-6、D-1 至 D-5。不重开 D-5，不把 `docs/deployment.md` §15 当作本条通过条件。

## 做法

### 推荐

只改路径 B 切片：从 `#### 路径 B：full build` 这一行起，到 `项目提供了 CMake Presets（见 `CMakePresets.json`）：` 之前为止。标题行保留。切片内只留一个 bash 代码块。

代码块的可执行行共 7 行，顺序写在「契约」里。其中 §7.1 的四条依赖命令和 §7.2 的两条 `cmake --preset` 命令按原行抄入。第 5 行是 `./scripts/install_deps.sh --cpu --check`。这一行在 `README.md` 的 Building from source 和本文件路径 B 之前的 Cpu 源码构建导语里已经有；`docs/deployment.zh-cn.md` §7.1 的代码块没有这一行，所以由路径 B 补上。

代码块外写「契约」里的五句说明。删掉 `setup_full_deps.sh`、`*_ROOT_DIR`、`mkdir build`、`cmake ..`、`make -j10`、vcpkg toolchain，以及「常用 CMake 选项」表。

代价：路径 B 不再写用环境变量补第三方树，也不再写用 `-D` 改输出目录。`scripts/setup_full_deps.sh` 仍留在仓库里，路径 B 不再引用它。说明句要求只执行其中一档。

### 否决

删掉「路径 B：full build」整节，只留文件前部指向 `docs/deployment.zh-cn.md` §7 的那一句。代价是没有路径 B 可以对照。

改 `CMakePresets.json` 或 `docs/deployment.zh-cn.md` §7 与 `README.md`，使它们改回 `setup_full_deps.sh` 和手写 cmake。代价是现行 full 入口被退回。

## 改动面

- `README.zh-cn.md`：仅路径 B 切片内、标题行之下的正文。标题行 `#### 路径 B：full build` 保留。

## 明确不改

- 切片之前的全部正文，包括「从源码构建」导语、路径 A、vcpkg 与 `cmake -B build -DMORTRED_BUILD_FULL=OFF`。
- `项目提供了 CMake Presets（见 `CMakePresets.json`）：` 及其后的 tests-only 代码块，以及该文件其余部分。
- `README.md`、`docs/deployment.zh-cn.md`、`docs/deployment.md`、`CMakePresets.json`。
- `scripts/install_deps.sh`、`scripts/setup_full_deps.sh`、`docs/repository-layout.md`。
- `src/`、`conf/`、HTTP、tensor、状态码、产品 toml。
- gateway `:8080`、supervisor `:8787`、engine。

## 契约

HTTP 与 tensor 布局不变。configure preset 保持现状：

- `full`：`binaryDir` 为 `${sourceDir}/build/full`；`MORTRED_BUILD_FULL=ON`，`MORTRED_ENABLE_WERROR=OFF`，`CMAKE_BUILD_TYPE=Release`；cache 里没有 `MORTRED_BUILD_PROFILE`。
- `full-cpu`：`binaryDir` 为 `${sourceDir}/build/full-cpu`；另有 `MORTRED_BUILD_PROFILE=cpu`。

路径 B 代码块的可执行行恰好 7 行，行首不缩进：

1. 至 4. `docs/deployment.zh-cn.md` 中 `### 7.1 ` 与 `### 7.2 ` 之间那个 bash 代码块的四条可执行行，原样。今日这四行分别是 `./scripts/install_deps.sh --check`、`./scripts/install_deps.sh --all`、`./scripts/install_deps.sh --cpu --all`、`sudo ./scripts/install_deps.sh --nvidia`。
2. `./scripts/install_deps.sh --cpu --check`。同一行以空白后的 `#` 注释结束。注释里有 `cpu`，没有 `TensorRT`，也没有 `CUDA`。
3. 至 7. `### 7.2 ` 与 `### 7.3 ` 之间那个 bash 代码块的前两条可执行行，原样。今日这两行分别以 `cmake --preset full &&` 与 `cmake --preset full-cpu &&` 开头。第三条 `cmake --preset tests-only` 不进入路径 B。

代码块之外、仍在切片之内，按原文出现下面五句：

```text
只执行 gpu 或 cpu 其中一档
gpu 档使用 --all、--nvidia、--check 与 preset full。cpu 档使用 --cpu --all、--cpu --check 与 preset full-cpu。
preset full 写到 build/full。preset full-cpu 写到 build/full-cpu。
默认可执行文件输出到 `$PROJECT_ROOT_DIR/_bin`，动态库输出到 `$PROJECT_ROOT_DIR/_lib`。
configure preset full-cpu 已设 MORTRED_BUILD_PROFILE=cpu
```

## 性能与资源

文档改动不改变常驻进程、显存或 engine。不执行 `install_deps.sh`，不执行 `cmake`。

## 验收标准

1. 在 `README.zh-cn.md` 全文检索下列每一个字符串，每个检索的期望退出码都是 1：`setup_full_deps`、`mkdir build`、`cmake ..`、`make -j`、`MORTRED_BIN_OUTPUT_DIR`、`MORTRED_LIB_OUTPUT_DIR`、`MORTRED_ENABLE_WERROR`、`常用 CMake 选项`、`*_ROOT_DIR`。
2. 在路径 B 切片中检索下列每一个字符串，每个检索的期望退出码都是 1：`-D`、`vcpkg`、`CMAKE_TOOLCHAIN_FILE`、`MORTRED_BUILD_FULL`、`verify_deployment`、`cmake --preset tests-only`、`cmake -B`。
3. 切片内恰好有一个 bash 代码块。可执行行恰好是「契约」的 7 行，顺序相同。
4. 「契约」五句都出现在切片内、代码块外。
5. 切片之前仍有导语 `Cpu 源码构建：`install_deps.sh --cpu --all && --cpu --check`` 与下一行的 `cmake --preset full-cpu`，仍有路径 A 与 vcpkg 说明。全文 `cmake -B build -DMORTRED_BUILD_FULL=OFF` 仍是 2 处，且都在切片之前。紧跟切片的 tests-only 代码块不变。
6. 未提交时，切片之前的前缀、以及从 `项目提供了 CMake Presets` 起的后缀，与 `git show HEAD:README.zh-cn.md` 的对应部分逐字相同。`git diff --name-only HEAD` 里除 `README.zh-cn.md` 外没有其他已跟踪文件。
7. `CMakePresets.json` 里 `full` 与 `full-cpu` 的 configure preset、build preset 仍符合「契约」。
8. `README.md` 仍同时含有 `./scripts/install_deps.sh --all`、`./scripts/install_deps.sh --cpu --all`、`./scripts/install_deps.sh --check`、`./scripts/install_deps.sh --cpu --check`、`cmake --preset full && cmake --build --preset full`、`cmake --preset full-cpu && cmake --build --preset full-cpu`。
9. `test -f scripts/setup_full_deps.sh` 的期望退出码是 0。

自测命令是仓库根目录的 python3 脚本，覆盖第 1 至 9 条。通过退出码 0，失败退出码 1，最后一行 `ok`。脚本按 [v1 正文](v1-tech-lead.md) 的验收实现，比较规则与上面九条一致。实现时把脚本放在问题说明里执行，不提交进产品树。

## 测试义务

黑盒执行自测命令。QA 白盒必须看引入本条的 diff：只允许 `README.zh-cn.md`，增删只落在路径 B 切片内。不把 `docs/deployment.md` §15 或 `docs/repository-layout.md` 是否仍写 `setup_full_deps.sh` 当作本条通过或失败条件。

## 自测命令

在仓库根目录的 bash 中执行。通过时期望退出码 0，且最后一行是 `ok`。任一断言失败时期望退出码 1。

```bash
python3 - <<'PY'
import json, pathlib, subprocess, sys

def fail(msg):
    print(msg, file=sys.stderr)
    raise SystemExit(1)

def load(path):
    data = pathlib.Path(path).read_bytes().replace(b"\r\n", b"\n").replace(b"\r", b"\n")
    return data.decode("utf-8")

bt = chr(96) * 3
zh = load("README.zh-cn.md")
dep = load("docs/deployment.zh-cn.md")
en = load("README.md")
start = "#### 路径 B：full build\n"
end = "项目提供了 CMake Presets（见 `CMakePresets.json`）："
s = zh.find(start)
e = zh.find(end, s + len(start) if s >= 0 else 0)
if s < 0 or e < 0 or zh.find(start, s + len(start)) != -1 or zh.count(end) != 1:
    fail("ac-slice")
chunk = zh[s:e]

for item in (
    "setup_full_deps", "mkdir build", "cmake ..", "make -j",
    "MORTRED_BIN_OUTPUT_DIR", "MORTRED_LIB_OUTPUT_DIR",
    "MORTRED_ENABLE_WERROR", "常用 CMake 选项", "*_ROOT_DIR",
):
    if item in zh:
        fail("ac1 " + item)

for item in (
    "-D", "vcpkg", "CMAKE_TOOLCHAIN_FILE", "MORTRED_BUILD_FULL",
    "verify_deployment", "cmake --preset tests-only", "cmake -B",
):
    if item in chunk:
        fail("ac2 " + item)

bash = bt + "bash\n"
if chunk.count(bash) != 1 or chunk.count(bt) != 2:
    fail("ac3-fence")
b = chunk.find(bash)
body_end = chunk.find("\n" + bt, b + len(bash))
if body_end < 0:
    fail("ac3-fence-close")
outside = chunk[:b] + chunk[body_end:]
sentences = (
    "只执行 gpu 或 cpu 其中一档",
    "gpu 档使用 --all、--nvidia、--check 与 preset full。cpu 档使用 --cpu --all、--cpu --check 与 preset full-cpu。",
    "preset full 写到 build/full。preset full-cpu 写到 build/full-cpu。",
    "默认可执行文件输出到 `$PROJECT_ROOT_DIR/_bin`，动态库输出到 `$PROJECT_ROOT_DIR/_lib`。",
    "configure preset full-cpu 已设 MORTRED_BUILD_PROFILE=cpu",
)
for sent in sentences:
    if sent not in outside:
        fail("ac4 " + sent)

def executable(text):
    out = []
    for ln in text.splitlines():
        stripped = ln.strip()
        if not stripped or stripped.startswith("#"):
            continue
        if ln[:1].isspace():
            fail("ac3-indent")
        out.append(ln)
    return out

def first_bash(text, heading, nxt):
    h = text.find(heading)
    n = text.find(nxt, h + len(heading) if h >= 0 else 0)
    if h < 0 or n < 0:
        fail("ac3-section " + heading)
    part = text[h:n]
    if part.count(bash) != 1:
        fail("ac3-dep-fence " + heading)
    b0 = part.find(bash)
    b1 = part.find("\n" + bt, b0 + len(bash))
    if b1 < 0:
        fail("ac3-dep-close " + heading)
    return part[b0 + len(bash):b1]

if dep.count("### 7.1 ") != 1 or dep.count("### 7.2 ") != 1 or dep.count("### 7.3 ") != 1:
    fail("ac3-dep-headings")
src71 = executable(first_bash(dep, "### 7.1 ", "### 7.2 "))
src72 = executable(first_bash(dep, "### 7.2 ", "### 7.3 "))
prefs = (
    "./scripts/install_deps.sh --check",
    "./scripts/install_deps.sh --all",
    "./scripts/install_deps.sh --cpu --all",
    "sudo ./scripts/install_deps.sh --nvidia",
)
if len(src71) != 4 or any(not src71[i].startswith(prefs[i]) for i in range(4)):
    fail("ac3-src71")
if (
    len(src72) != 3
    or not src72[0].startswith("cmake --preset full &&")
    or not src72[1].startswith("cmake --preset full-cpu &&")
    or not src72[2].startswith("cmake --preset tests-only")
):
    fail("ac3-src72")
got = executable(chunk[b + len(bash):body_end])
if len(got) != 7 or got[0:4] != src71 or got[5] != src72[0] or got[6] != src72[1]:
    fail("ac3-lines")
cpu_line = got[4]
cpu_cmd = "./scripts/install_deps.sh --cpu --check"
if not cpu_line.startswith(cpu_cmd):
    fail("ac3-cpu-check")
rest = cpu_line[len(cpu_cmd):].strip()
if not rest.startswith("#") or "cpu" not in rest[1:] or "TensorRT" in rest or "CUDA" in rest:
    fail("ac3-cpu-comment")

pre = (
    "Cpu 源码构建：`install_deps.sh --cpu --all && --cpu --check`，再\n"
    "`cmake --preset full-cpu`（详见 [docs/deployment.zh-cn.md](docs/deployment.zh-cn.md) §7）。\n"
)
if pre not in zh[:s]:
    fail("ac5-preamble")
if "#### 路径 A：tests-only" not in zh[:s]:
    fail("ac5-path-a")
if "方案 A2 - vcpkg（可选；仅本地开发用，CI 不使用）：" not in zh[:s]:
    fail("ac5-vcpkg")
marker = "cmake -B build -DMORTRED_BUILD_FULL=OFF"
if zh.count(marker) != 2 or zh.find(marker) > s:
    fail("ac5-cmake-b")
tail = (
    "项目提供了 CMake Presets（见 `CMakePresets.json`）：\n\n"
    + bash
    + "cmake --preset tests-only\n"
    + "cmake --build --preset tests-only\n"
    + "# 该 buildPreset 默认 target 为 check（编译 EXCLUDE_FROM_ALL 测试并跑 ctest）\n"
    + bt
    + "\n"
)
if not zh[e:].startswith(tail):
    fail("ac5-tests-only")

head = subprocess.check_output(["git", "show", "HEAD:README.zh-cn.md"])
head = head.replace(b"\r\n", b"\n").replace(b"\r", b"\n").decode("utf-8")
hs = head.find(start)
he = head.find(end, hs + len(start) if hs >= 0 else 0)
if hs < 0 or he < 0 or head.count(end) != 1 or zh[:s] != head[:hs] or zh[e:] != head[he:]:
    fail("ac6-unchanged")
diff = subprocess.check_output(["git", "diff", "--name-only", "HEAD"])
diff = diff.replace(b"\r\n", b"\n").replace(b"\r", b"\n").decode("utf-8")
extra = [name.strip().replace("\\", "/") for name in diff.splitlines() if name.strip() and name.strip().replace("\\", "/") != "README.zh-cn.md"]
if extra:
    fail("ac6-diff " + ",".join(extra))

presets = json.loads(load("CMakePresets.json"))
cfg = {p["name"]: p for p in presets["configurePresets"]}
full, cpu_p = cfg["full"], cfg["full-cpu"]
if full["binaryDir"] != "${sourceDir}/build/full" or cpu_p["binaryDir"] != "${sourceDir}/build/full-cpu":
    fail("ac7-binary")
if full["cacheVariables"].get("MORTRED_BUILD_FULL") != "ON" or cpu_p["cacheVariables"].get("MORTRED_BUILD_FULL") != "ON":
    fail("ac7-full")
if "MORTRED_BUILD_PROFILE" in full["cacheVariables"] or cpu_p["cacheVariables"].get("MORTRED_BUILD_PROFILE") != "cpu":
    fail("ac7-profile")
builds = {p["name"]: p for p in presets["buildPresets"]}
if builds["full"]["configurePreset"] != "full" or builds["full-cpu"]["configurePreset"] != "full-cpu":
    fail("ac7-build")

for needle in (
    "./scripts/install_deps.sh --all",
    "./scripts/install_deps.sh --cpu --all",
    "./scripts/install_deps.sh --check",
    "./scripts/install_deps.sh --cpu --check",
    "cmake --preset full && cmake --build --preset full",
    "cmake --preset full-cpu && cmake --build --preset full-cpu",
):
    if needle not in en:
        fail("ac8 " + needle)

if not pathlib.Path("scripts/setup_full_deps.sh").is_file():
    fail("ac9")
print("ok")
PY
```

## RD 不得自行决定

- 不得重排上述 7 行，不得把 `tests-only` 行抄进路径 B。
- 不得把五句只写进代码块注释，不得写「两档都要装」。
- 不得改路径 A、tests-only 段，以及路径 B 之前的导语。
- 不得改 `README.md`、`docs/deployment.zh-cn.md`、`docs/deployment.md`、`CMakePresets.json`。
- 不得删除 `scripts/setup_full_deps.sh`，不得改 `docs/repository-layout.md`。
- 不得重开 D-5，不得改推理代码。

## 签字
| 角色 | 结论 | 条件 | 方案版本 | 记录 |
| --- | --- | --- | --- | --- |
| cv-pm | 同意 | 无 | v1 | `cowork/ledger/round-010/issues/d-6/03-discussion/v1-sign-pm.md` |
| cv-rd | 同意 | 无 | v1 | `cowork/ledger/round-010/issues/d-6/03-discussion/v1-sign-rd.md` |
| cv-tech-lead | 同意 | 无 | v1 | `cowork/ledger/round-010/issues/d-6/03-discussion/v1-sign-tech-lead.md` |
