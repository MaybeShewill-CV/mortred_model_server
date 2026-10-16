# 技术方案

- 问题: P0-1
- 版本: v1
- 对应列表: v1

## 边界

停跟踪 / gitignore / 仅经 init-edge 本地生成 TLS 材料。不改正文 markdown。磁盘上已有的 conf/local/edge/tls 文件可保留为本地产物，但不得出现在 git 索引。

## 方案

### 推荐

1. .gitignore：保留 conf/local/；补充明确的 edge TLS 私钥/证书模式（*.key、tls/*.pem、tls/*.crt 在 conf/local 下）。
2. scripts/mortredctl_init-edge.sh：在生成 CA/server key 处增加注释——密钥仅写本地 OUT/tls，勿提交。
3. scripts/check_consistency.py：增加断言——git ls-files 不得包含 conf/local/ 下 .key/.pem/.crt（或路径含 /tls/ 的私钥）。
4. 若索引中已有此类文件则 git rm --cached（当前基线无跟踪项则跳过）。

### 否决

- 删除用户本机已生成的密钥文件：超出仓库卫生范围。
- 新写 markdown 文档：用户禁止除非验收强制。

## 改动面

.gitignore；scripts/mortredctl_init-edge.sh；scripts/check_consistency.py（及必要时其测试路径）。

## 明确不改

docs/**；产品代码；已存在的本地密钥文件内容。

## 契约

无 HTTP/tensor 变化。

## 验收标准

1. git ls-files 输出不含 conf/local/edge/tls/ 下任何 .key/.pem/.crt。
2. .gitignore 含 conf/local/ 且含对 TLS 私钥的明确模式或注释块。
3. mortredctl_init-edge.sh 在密钥生成附近有「仅本地生成、勿提交」注释。
4. python3 scripts/check_consistency.py（或其中 secrets 子检查）在干净树上退出 0；若人为 git add 一个假 .key 则检查失败。

## 自测命令

工作目录为 worktree 根。期望退出码 0：

```
git ls-files 'conf/local/**' | grep -E '\\.(key|pem|crt)$' && exit 1 || true
grep -q 'conf/local/' .gitignore
grep -q 'do not commit' scripts/mortredctl_init-edge.sh || grep -qi 'not commit\\|勿提交\\|local only\\|do not track' scripts/mortredctl_init-edge.sh
python3 -c "import subprocess,sys; t=subprocess.check_output(['git','ls-files']).decode();
bad=[l for l in t.splitlines() if l.startswith('conf/local/') and l.endswith(('.key','.pem','.crt'))];
sys.exit(1 if bad else 0)"
```

打印 p0-1-tls-gitignore ok。

## 确认
| 角色 | 结论 | 条件 | 方案版本 | 记录 |
| --- | --- | --- | --- | --- |
| cv-pm | 同意 | 无 | v1 | 用户锁定队列；编排方代行 |
| cv-rd | 同意 | 无 | v1 | 用户锁定队列；编排方代行 |
| cv-tech-lead | 同意 | 无 | v1 | 用户锁定队列；编排方代行 |