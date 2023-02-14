#include <string>
#include <map>

#include "glog/logging.h"

#include "common/status_code.h"
#include "registration/mysql/mysql_helper.h"

using jinq::registration::mysql::MySqlHelper;

int main(int argc, char** argv) {

    MySqlHelper helper;
    std::map<std::string, std::string> conds;
    auto res = helper.select("", {"", ""}, conds);

    LOG(INFO) << "select info length: " << res.size();

    for (auto& row : res) {
        LOG(INFO) << "key: " << row.begin()->first << ", value: " << std::get<int>(row.begin()->second);
        LOG(INFO) << "key: " << row.end()->first << ", value: " << std::get<std::string>(row.end()->second);
    }

    return 0;

}