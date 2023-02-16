#include <string>
#include <map>

#include "glog/logging.h"

#include "common/status_code.h"
#include "registration/mysql/mysql_helper.h"

using jinq::registration::mysql::MySqlDBConfig;
using jinq::registration::mysql::MySqlHelper;

int main(int argc, char** argv) {

    MySqlDBConfig db_cfg;
    std::string cfg_path = "../conf/server/registration_center/registration_config.ini";
    db_cfg.init(cfg_path);

    MySqlHelper helper;
    helper.init(db_cfg);
    std::map<std::string, std::string> conds;
    auto res = helper.select("mmai_projects", {"id", "name"}, conds);

    LOG(INFO) << "select result: " << res;
    return 0;

}