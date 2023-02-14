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
        LOG(INFO) << "key: " << row[0].first << ", value: " << std::get<std::int>(row[0].second);
        LOG(INFO) << "key: " << row[1].first << ", value: " << std::get<std::string>(row[1].second);
    }

    return 0;

}