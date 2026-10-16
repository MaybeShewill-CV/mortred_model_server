# 技术方案
- 问题: P0-1
- 版本: v1
## 目标
check_consistency 与薄 openapi_doc.h loader 对齐：去掉 embed 同步；断言 loader 形状；对齐 gen_openapi --check；可选 anti-embed 守卫。
## 边界
- 不改 packing/Docker。
- 不把 docs/openapi.json 再写回 header。
- 不改 /openapi.json 运行时加载语义。
## 实现要点
1. 重写 `check_openapi_doc_header_sync`：要求 `load_openapi_json` + `k_openapi_doc_json_ref`；拒绝 `R"MORTRED_OPENAPI(` 嵌入。
2. 新增 `check_openapi_gen_check`：subprocess 跑 `python3 scripts/gen_openapi.py --check`。
3. `gen_openapi.py` 生成路径只写 `docs/openapi.json`，不再假装写 header。
## 确认
| 角色 | 结论 | 条件 | 方案版本 | 记录 |
| --- | --- | --- | --- | --- |
| 用户 | 同意 | 无 | v1 | Round 8 队列锁定文案即方案 |
| cv-pm | 同意 | 无 | v1 | 用户锁定 |
| cv-tech-lead | 同意 | 无 | v1 | 用户锁定 |
| cv-rd | 同意 | 无 | v1 | 用户锁定 |
