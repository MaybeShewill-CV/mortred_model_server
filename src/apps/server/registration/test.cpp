#include <string>
#include <map>

#include "glog/logging.h"

#include "common/status_code.h"
#include "registration/mysql/mysql_helper.h"

using jinq::registration::mysql::MySqlDBConfig;
using jinq::registration::mysql::MySqlHelper;

int main(int argc, char** argv) {

    MySqlDBConfig db_cfg;
    std::string cfg_file_path = "../conf/server/registration_center/registration_config.ini";
    db_cfg.init(cfg_file_path);
//    db_cfg.set_db_name("mortred_ai_serer");
//    db_cfg.set_host("localhost");
//    db_cfg.set_user_name("root");
//    db_cfg.set_user_pw("327205");

    MySqlHelper helper;
    helper.init(db_cfg);
    std::map<std::string, std::string> conds;
    auto res = helper.select("mmai_service_instances", {"ip_address", "service_id"}, conds);

    LOG(INFO) << "select result: " << res;
    return 0;

}