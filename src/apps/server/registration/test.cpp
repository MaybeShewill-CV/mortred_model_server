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
        for (auto& col : row) {
            LOG(INFO) << "key: " << col.first << ", value: " << std::get<std::string>(col.second);
        }
    }

    return 0;

}