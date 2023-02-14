#include <string>

#include "glog/logging.h"

#include "common/status_code.h"
#include "registration/mysql/mysql_helper.h"

using jinq::registration::mysql::MySqlHelper;

int main(int argc, char** argv) {

    MySqlHelper helper;
    auto res = helper.select("", "", "");

    for (auto& row : res) {
        for (auto& col : row) {
            LOG(INFO) << "key: " << col.first << ", value: " << col.second;
        }
    }

    return 0;

}