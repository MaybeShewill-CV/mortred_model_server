/************************************************
 * Copyright MaybeShewill-CV. All Rights Reserved.
 * Author: MaybeShewill-CV
 * File: MySqlHelper.cpp
 * Date: 23-2-14
 ************************************************/

#include "mysql_helper.h"

#include "glog/logging.h"
#include "workflow/WFFacilities.h"
#include "workflow/WFTaskFactory.h"
#include "workflow/MySQLResult.h"
#include "workflow/Workflow.h"

#include "common/file_path_util.h"

namespace jinq {
namespace registration {
namespace mysql {

using jinq::common::FilePathUtil;
using jinq::common::StatusCode;
using jinq::registration::mysql::MySqlDBConfig;
using jinq::registration::mysql::QueryResult;

namespace internal_impl {

void select_callback(WFMySQLTask* task) {

    auto* query_result = (QueryResult*)task->user_data;

    protocol::MySQLResponse* resp = task->get_resp();
    protocol::MySQLResultCursor cursor(resp);
    const protocol::MySQLField* const* fields;
    std::vector<protocol::MySQLCell> arr;

    if (task->get_state() != WFT_STATE_SUCCESS) {
        fprintf(stderr, "error msg: %s\n",
                WFGlobal::get_error_string(task->get_state(),
                                           task->get_error()));
        char log_str[256];
        sprintf(log_str, "error msg: %s\n",
                WFGlobal::get_error_string(task->get_state(),
                                           task->get_error()));
        LOG(ERROR) << log_str;
        return;
    }

    if (cursor.get_cursor_status() == MYSQL_STATUS_GET_RESULT) {
        fprintf(stderr, "cursor_status=%d field_count=%u rows_count=%u\n",
                cursor.get_cursor_status(), cursor.get_field_count(), cursor.get_rows_count());

        do {
            fprintf(stderr, "-------- RESULT SET --------\n");
            //nocopy api
            fields = cursor.fetch_fields();

            for (int i = 0; i < cursor.get_field_count(); i++) {
                if (i == 0) {
                    fprintf(stderr, "db=%s table=%s\n",
                            fields[i]->get_db().c_str(), fields[i]->get_table().c_str());
                    fprintf(stderr, "-------- COLUMNS --------\n");
                }

                fprintf(stderr, "name[%s] type[%s]\n",
                        fields[i]->get_name().c_str(),
                        datatype2str(fields[i]->get_data_type()));
            }

            fprintf(stderr, "------- COLUMNS END ------\n");

            while (cursor.fetch_row(arr)) {
                fprintf(stderr, "---------- ROW ----------\n");

                for (size_t i = 0; i < arr.size(); i++) {
                    fprintf(stderr, "[%s][%s]", fields[i]->get_name().c_str(),
                            datatype2str(arr[i].get_data_type()));

                    if (arr[i].is_string()) {
                        std::string res = arr[i].as_string();

                        if (res.length() == 0) {
                            fprintf(stderr, "[\"\"]\n");
                        } else {
                            fprintf(stderr, "[%s]\n", res.c_str());
                        }
                    } else if (arr[i].is_int()) {
                        fprintf(stderr, "[%d]\n", arr[i].as_int());
                    } else if (arr[i].is_ulonglong()) {
                        fprintf(stderr, "[%llu]\n", arr[i].as_ulonglong());
                    } else if (arr[i].is_float()) {
                        const void* ptr = nullptr;
                        size_t len = 0;
                        int data_type = 0;
                        arr[i].get_cell_nocopy(&ptr, &len, &data_type);
                        size_t pos = 0;

                        for (pos = 0; pos < len; pos++) {
                            if (*((const char*) ptr + pos) == '.') {
                                break;
                            }
                        }

                        if (pos != len) {
                            pos = len - pos - 1;
                        } else {
                            pos = 0;
                        }

                        fprintf(stderr, "[%.*f]\n", (int) pos, arr[i].as_float());
                    } else if (arr[i].is_double()) {
                        const void* ptr = nullptr;
                        size_t len = 0;
                        int data_type = 0;
                        arr[i].get_cell_nocopy(&ptr, &len, &data_type);
                        size_t pos = 0;

                        for (pos = 0; pos < len; pos++) {
                            if (*((const char*) ptr + pos) == '.') {
                                break;
                            }
                        }

                        if (pos != len) {
                            pos = len - pos - 1;
                        } else {
                            pos = 0;
                        }

                        fprintf(stderr, "[%.*lf]\n", (int) pos, arr[i].as_double());
                    } else if (arr[i].is_date()) {
                        fprintf(stderr, "[%s]\n", arr[i].as_string().c_str());
                    } else if (arr[i].is_time()) {
                        fprintf(stderr, "[%s]\n", arr[i].as_string().c_str());
                    } else if (arr[i].is_datetime()) {
                        fprintf(stderr, "[%s]\n", arr[i].as_string().c_str());
                    } else if (arr[i].is_null()) {
                        fprintf(stderr, "[NULL]\n");
                    } else {
                        std::string res = arr[i].as_binary_string();

                        if (res.length() == 0) {
                            fprintf(stderr, "[\"\"]\n");
                        } else {
                            fprintf(stderr, "[%s]\n", res.c_str());
                        }
                    }
                }

                fprintf(stderr, "-------- ROW END --------\n");
            }

            fprintf(stderr, "-------- RESULT SET END --------\n");
        } while (cursor.next_result_set());

    } else if (resp->get_packet_type() == MYSQL_PACKET_OK) {
        fprintf(stderr, "OK. %llu ", task->get_resp()->get_affected_rows());

        if (task->get_resp()->get_affected_rows() == 1) {
            fprintf(stderr, "row ");
        } else {
            fprintf(stderr, "rows ");
        }

        fprintf(stderr, "affected. %d warnings. insert_id=%llu. %s\n",
                task->get_resp()->get_warnings(),
                task->get_resp()->get_last_insert_id(),
                task->get_resp()->get_info().c_str());
    } else if (resp->get_packet_type() == MYSQL_PACKET_ERROR) {
        fprintf(stderr, "ERROR. error_code=%d %s\n",
                task->get_resp()->get_error_code(),
                task->get_resp()->get_error_msg().c_str());
        char log_str[256];
        sprintf(log_str, "ERROR. error_code=%d %s\n",
                task->get_resp()->get_error_code(),
                task->get_resp()->get_error_msg().c_str());
        LOG(ERROR) << log_str;
        LOG(ERROR) << task->get_req()->get_query();
    } else if (resp->get_packet_type() == MYSQL_PACKET_EOF) {
        fprintf(stderr, "EOF packet without any ResultSets\n");
        char log_str[256];
        sprintf(log_str, "EOF packet without any ResultSets\n");
        LOG(ERROR) << log_str;
    } else {
        fprintf(stderr, "Abnormal packet_type=%d\n", resp->get_packet_type());
        char log_str[256];
        sprintf(log_str, "Abnormal packet_type=%d\n", resp->get_packet_type());
        LOG(ERROR) << log_str;
    }
}

}

/***************** Impl Function Sets ******************/

class MySqlHelper::Impl {
  public:
    /***
     *
     */
    Impl() = default;

    /***
     *
     */
    ~Impl() = default;

    /***
    *
    * @param transformer
     */
    Impl(const Impl& transformer) = delete;

    /***
     *
     * @param transformer
     * @return
     */
    Impl& operator=(const Impl& transformer) = delete;

    /***
     *
     * @param db_cfg
     * @return
     */
    StatusCode init(const MySqlDBConfig& db_cfg);

    /***
     *
     * @param table
     * @param columns
     * @param conditions
     * @return
     */
    QueryResult select(
        const std::string& table,
        const std::vector<std::string>& columns,
        const std::map<std::string, std::string>& conditions);

    /***
     *
     * @return
     */
    bool is_successfully_initialized() const {
        return _m_successfully_initialized;
    };

  private:
    // mysql db config
    MySqlDBConfig _m_db_cfg;
    // init success flag
    bool _m_successfully_initialized = false;
};

/***
*
* @param cfg_file_path
* @return
 */
StatusCode MySqlHelper::Impl::init(const MySqlDBConfig& db_cfg) {
    // copy db cfg
    _m_db_cfg = db_cfg;

    _m_successfully_initialized = true;
    return StatusCode::OJBK;
}

/***
 *
 * @param table
 * @param columns
 * @param conditions
 * @return
 */
QueryResult MySqlHelper::Impl::select(
    const std::string &table,
    const std::vector<std::string> &columns,
    const std::map<std::string, std::string> &conditions) {

    WFFacilities::WaitGroup wg(1);


}


/************* Export Function Sets *************/

/***
 *
 */
MySqlHelper::MySqlHelper() {
    _m_pimpl = std::make_unique<Impl>();
}

/***
 *
 * @param config_file_path
 * @return
 */
StatusCode MySqlHelper::init(const jinq::registration::mysql::MySqlDBConfig &db_cfg){
    return _m_pimpl->init(db_cfg);
}

/***
 *
 * @param table
 * @param columns
 * @param conditions
 * @return
 */
QueryResult MySqlHelper::select(
    const std::string &table,
    const std::vector<std::string> &columns,
    const std::map<std::string, std::string> &conditions) {
    return _m_pimpl->select(table, columns, conditions);
}

/***
 *
 * @return
 */
bool MySqlHelper::is_successfully_initialized() const {
    return _m_pimpl->is_successfully_initialized();
}

}
}
}
