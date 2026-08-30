#include "apps/common/model_server_main.h"
#include "factory/diffusion_task.h"

int main(int argc, char **argv) {
    return jinq::apps::run_model_server_main(argc, argv, "DDPM_SERVER",
                                             [](const std::string &server_name) -> std::unique_ptr<jinq::server::BaseAiServer> {
                                                 return jinq::factory::diffusion::create_server("DDPM", server_name);
                                             });
}
