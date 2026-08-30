#include "apps/common/model_server_main.h"
#include "factory/sam_task.h"

int main(int argc, char **argv) {
    return jinq::apps::run_model_server_main(argc, argv, "SAM_AMG_SERVER",
                                             [](const std::string &server_name) -> std::unique_ptr<jinq::server::BaseAiServer> {
                                                 return jinq::factory::segment_anything::create_amg_server(server_name);
                                             });
}
