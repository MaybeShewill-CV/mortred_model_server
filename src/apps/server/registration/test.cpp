#include <string>
#include <map>

#include "glog/logging.h"

#include "common/status_code.h"
#include "registration/mysql/mysql_helper.h"
#include "registration/mysql/sql_query_builder.hpp"

using jinq::registration::mysql::MySqlDBConfig;
using jinq::registration::mysql::MySqlHelper;
using jinq::registration::mysql::SelectBuilder;

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
    std::string query_result;
    SelectBuilder builder;
    auto sql_query = builder.select("*").from("mmai_projects").get_query();
    auto res = helper.select(sql_query, query_result);

    LOG(INFO) << "select status: " << res;
    LOG(INFO) << "select result: " << query_result;
    return 0;

}