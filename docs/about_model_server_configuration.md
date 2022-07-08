<b><font color='black' size='8' face='Helvetica'><b><font color='black' size='8' face='Helvetica'> About Model Sercver Configuration </font></b> </font></b>

All model server's configurations are stored in $PROJECT_ROOT_DIR/conf/server folder.

<b><font color='GrayB' size='6' face='Helvetica'> Common Configuration </font></b>

Use mobilenetv2's model server configuration for example
![common_server_config](../resources/images/common_model_server_config_example.png)

**host:** server's host address

**port:** server's port

**max connections:** server's max connections. old connection will be kicked off if no spare connection left. Connections will be refused if no extra connections can be kicked. Enlarge this param when the amount of concurrency is large. You may find some useful disscussion on [#issue463](https://github.com/sogou/workflow/issues/463) and [#issue906](https://github.com/sogou/workflow/issues/906)