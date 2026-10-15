# 第2步确认

- 轮次: 2
- 针对版本: v2
- 结论: 不同意
- 理由: P2-16 要回到 P2。使用者从 `README.zh-cn.md` 的「快速添加新的 DL 服务」打开 `docs/how_to_add_new_server.zh-cn.md`，按示例往分类家族加模型。现在示例是 `Entry{"DENSENET", ...}`、`--model DENSENET`、`model = "DENSENET"`，且没有 family `"densenet"`。一次打开这一份，就能核对 `model_section`、`--model`、`model=` 和 family 都是 `densenet`。这条有出处，也有这一次阅读能确认的变化，优先级已经是 P2。catalog id 在覆盖集外，只说明不为 `densenet` 另做完成线；中文添加入口本身仍要留在队列。已关闭的 P1-2 只含两份英文文件，留在附录。
