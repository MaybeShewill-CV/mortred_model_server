<b><font color='black' size='8' face='Helvetica'><b><font color='black' size='8' face='Helvetica'> 模型服务器配置参数说明 </font></b> </font></b>

所有的模型服务器参数配置均存放在 `$PROJECT_ROOT_DIR/conf/server` 文件夹。

<b><font color='GrayB' size='6' face='Helvetica'> 常规配置参数 </font></b>

以 `mobilenetv2` 图像分类服务器为例
![common_server_config](../resources/images/common_model_server_config_example.png)

**host:** 服务host地址

**port:** 服务端口号

**max connections:** 服务支持的最大连接数。旧链接会被踢出如果超过最大连接数。在没有链接可用的情况下新的链接请求会被拒绝。在并发量大的情况下可以增大这个参数. 你可以在以下issue和tutorial中找到一些有用信息 [#issue463](https://github.com/sogou/workflow/issues/463), [#issue906](https://github.com/sogou/workflow/issues/906) and [tutorial-05-http_proxy](https://github.com/sogou/workflow/blob/516da621aea136c4c25c048b89875f62c9d20af6/docs/en/tutorial-05-http_proxy.md)

**peer_resp_timeout:** 服务读取和发送一段数据的超时设置，默认15秒。

**compute_threads:** 计算线程池的线程数， -1 代表使用默认值即cpu的核心数。

**handler_threads:** 处理网络任务、回调函数的线程个数

**model_run_timeout:** 模型inference的超时设置，超时的任务会被中断，-1代表该值无限大。

**server_url:** 服务的url地址

**model_config_file_path:** 服务使用的DL模型配置。关于DL模型参数配置说明可参考 [about_model_configuration](../docs/about_model_configuration.md)

<b><font color='GrayB' size='6' face='Helvetica'> 其他一些网络服务参数配置 </font></b>

其余一些有关网络服务的全局配置可以参考 [workflow_docs_about_global_configuration](https://github.com/sogou/workflow/blob/f7979e46f3b1f9c0052adb9e2ffa959730dcda6e/docs/about-config.md)

And if you want to adjust some of those configuration params you may do it by modifying your server's `init` function. For example global configuration for mobilenetv2 classification server can be modified at [../src/server/classification/mobilenetv2_server.cpp#L197-L202](../src/server/classification/mobilenetv2_server.cpp)

如果你想修改一些网络服务的全局参数，你可以在server的 `init` 初始化函数中进行修改. 例如 [../src/server/classification/mobilenetv2_server.cpp#L197-L202](../src/server/classification/mobilenetv2_server.cpp)

`workflow全局配置参数代码段`
![benchmakr_code_snappit](../resources/images/workflow_global_config.png)
