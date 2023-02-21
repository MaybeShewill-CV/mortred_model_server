#include <string>
#include <map>

#include "glog/logging.h"

#include "common/status_code.h"
#include "registration/mysql/mysql_data_type.h"
#include "registration/mysql/mysql_helper.h"
#include "registration/mysql/sql_query_builder.hpp"

using jinq::registration::mysql::MySqlDBConfig;
using jinq::registration::mysql::MySqlHelper;
using jinq::registration::mysql::KVData;
using jinq::registration::mysql::SelectBuilder;
using jinq::registration::mysql::InsertBuilder;

int main(int argc, char** argv) {

    MySqlDBConfig db_cfg;
    std::string cfg_file_path = "../conf/server/registration_center/registration_config.ini";
    db_cfg.init(cfg_file_path);

    MySqlHelper helper;
    helper.init(db_cfg);

    InsertBuilder insert_builder;
    KVData kvs = {
        {"project_id", 5},
        {"project_name", std::string("test")},
        {"create_user", std::string("luoyao")},
    };
    auto insert_query= insert_builder.insert("mmai_projects", kvs).get_query();
    auto insert_res = helper.insert(insert_query);
    LOG(INFO) << "insert query: " << insert_query;
    LOG(INFO) << "insert status: " << insert_res;


    std::string query_result;
    SelectBuilder select_builder;
    auto select_query = select_builder.select("*").from("mmai_projects").get_query();
    auto select_res = helper.select(select_query, query_result);

    LOG(INFO) << "select status: " << select_res;
    LOG(INFO) << "select result: " << query_result;

    return 0;

}