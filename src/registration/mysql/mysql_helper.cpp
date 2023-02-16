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
#include "rapidjson/stringbuffer.h"
#include "rapidjson/writer.h"
#include "fmt/core.h"
#include "fmt/ranges.h"

#include "common/file_path_util.h"

namespace jinq {
namespace registration {
namespace mysql {

using jinq::common::FilePathUtil;
using jinq::common::StatusCode;
using jinq::registration::mysql::MySqlDBConfig;
using jinq::registration::mysql::ColumnValue;
using jinq::registration::mysql::ColumnKey;
using jinq::registration::mysql::RowData;
using jinq::registration::mysql::QueryResult;

namespace internal_impl {

void select_callback(WFMySQLTask* task) {

    auto* query_str = (std::string*)task->user_data;
    QueryResult query_result;

    protocol::MySQLResponse* resp = task->get_resp();
    protocol::MySQLResultCursor cursor(resp);
    const protocol::MySQLField* const* fields;
    std::vector<protocol::MySQLCell> arr;

    rapidjson::StringBuffer buf;
    rapidjson::Writer<rapidjson::StringBuffer> writer(buf);
    writer.StartObject();
    int status = 0;
    std::string msg;
    std::string db_name;
    std::string table_name;
    std::vector<std::string> field_names;
    std::vector<std::string> field_types;
    std::vector<std::vector<std::string> > multiple_rows;

    if (task->get_state() != WFT_STATE_SUCCESS) {
        char log_str[256];
        sprintf(log_str, "error msg: %s\n",
                WFGlobal::get_error_string(task->get_state(),
                                           task->get_error()));
        LOG(ERROR) << log_str;

        writer.Key("status");
        writer.Int(task->get_state());
        writer.Key("msg");
        writer.String(log_str);
        writer.Key("data");
        writer.StartObject();
        writer.EndObject();
        *query_str = buf.GetString();
        return;
    }

    if (cursor.get_cursor_status() == MYSQL_STATUS_GET_RESULT) {
//        fprintf(stderr, "cursor_status=%d field_count=%u rows_count=%u\n",
//                cursor.get_cursor_status(), cursor.get_field_count(), cursor.get_rows_count());

        do {
//            fprintf(stderr, "-------- RESULT SET --------\n");
            //nocopy api
            fields = cursor.fetch_fields();
            for (int i = 0; i < cursor.get_field_count(); i++) {
                if (i == 0) {
                    db_name = fields[i]->get_db();
                    table_name = fields[i]->get_table();
//                    fprintf(stderr, "db=%s table=%s\n",
//                            fields[i]->get_db().c_str(), fields[i]->get_table().c_str());
//                    fprintf(stderr, "-------- COLUMNS --------\n");
                }

//                fprintf(stderr, "name[%s] type[%s]\n",
//                        fields[i]->get_name().c_str(),
//                        datatype2str(fields[i]->get_data_type()));
                field_names.push_back(fields[i]->get_name());
                field_types.emplace_back(datatype2str(fields[i]->get_data_type()));
            }

//            fprintf(stderr, "------- COLUMNS END ------\n");
            
            // fetch every row data
            while (cursor.fetch_row(arr)) {
//                fprintf(stderr, "---------- ROW ----------\n");
                std::vector<std::string> single_row;
                for (size_t i = 0; i < arr.size(); i++) {
//                    fprintf(stderr, "[%s][%s]", fields[i]->get_name().c_str(),
//                            datatype2str(arr[i].get_data_type()));

                    if (arr[i].is_string()) {
                        std::string res = arr[i].as_string();
                        single_row.push_back(res);

//                        if (res.length() == 0) {
//                            fprintf(stderr, "[\"\"]\n");
//                        } else {
//                            fprintf(stderr, "[%s]\n", res.c_str());
//                        }
                    } else if (arr[i].is_int()) {
//                        fprintf(stderr, "[%d]\n", arr[i].as_int());
                        int res = arr[i].as_int();
                        single_row.push_back(std::to_string(res));
                    } else if (arr[i].is_ulonglong()) {
//                        fprintf(stderr, "[%llu]\n", arr[i].as_ulonglong());
                        auto res = arr[i].as_ulonglong();
                        single_row.push_back(std::to_string(res));
                    } else if (arr[i].is_float()) {
//                        const void* ptr = nullptr;
//                        size_t len = 0;
//                        int data_type = 0;
//                        arr[i].get_cell_nocopy(&ptr, &len, &data_type);
//                        size_t pos = 0;
//
//                        for (pos = 0; pos < len; pos++) {
//                            if (*((const char*) ptr + pos) == '.') {
//                                break;
//                            }
//                        }
//
//                        if (pos != len) {
//                            pos = len - pos - 1;
//                        } else {
//                            pos = 0;
//                        }
//                        fprintf(stderr, "[%.*f]\n", (int) pos, arr[i].as_float());

                        auto res = arr[i].as_float();
                        single_row.push_back(std::to_string(res));
                    } else if (arr[i].is_double()) {
//                        const void* ptr = nullptr;
//                        size_t len = 0;
//                        int data_type = 0;
//                        arr[i].get_cell_nocopy(&ptr, &len, &data_type);
//                        size_t pos = 0;
//
//                        for (pos = 0; pos < len; pos++) {
//                            if (*((const char*) ptr + pos) == '.') {
//                                break;
//                            }
//                        }
//
//                        if (pos != len) {
//                            pos = len - pos - 1;
//                        } else {
//                            pos = 0;
//                        }
//
//                        fprintf(stderr, "[%.*lf]\n", (int) pos, arr[i].as_double());

                        auto res = arr[i].as_double();
                        single_row.push_back(std::to_string(res));
                    } else if (arr[i].is_date()) {
//                        fprintf(stderr, "[%s]\n", arr[i].as_string().c_str());
                        auto res = arr[i].as_date();
                        single_row.push_back(res);
                    } else if (arr[i].is_time()) {
//                        fprintf(stderr, "[%s]\n", arr[i].as_string().c_str());
                        auto res = arr[i].as_time();
                        single_row.push_back(res);
                    } else if (arr[i].is_datetime()) {
//                        fprintf(stderr, "[%s]\n", arr[i].as_string().c_str());
                        auto res = arr[i].as_datetime();
                        single_row.push_back(res);
                    } else if (arr[i].is_null()) {
//                        fprintf(stderr, "[NULL]\n");
                        single_row.emplace_back("");
                    } else {
                        std::string res = arr[i].as_binary_string();

//                        if (res.length() == 0) {
//                            fprintf(stderr, "[\"\"]\n");
//                        } else {
//                            fprintf(stderr, "[%s]\n", res.c_str());
//                        }
                        single_row.push_back(res);
                    }
                }

//                fprintf(stderr, "-------- ROW END --------\n");
                multiple_rows.push_back(single_row);
            }

//            fprintf(stderr, "-------- RESULT SET END --------\n");
        } while (cursor.next_result_set());
        status = 0;
        msg = "ok";
    } else if (resp->get_packet_type() == MYSQL_PACKET_ERROR) {
        char log_str[256];
        sprintf(log_str, "ERROR. error_code=%d %s\n",
                task->get_resp()->get_error_code(),
                task->get_resp()->get_error_msg().c_str());
        LOG(ERROR) << log_str;
        LOG(ERROR) << task->get_req()->get_query();
        status = -1;
        msg = log_str;
    } else if (resp->get_packet_type() == MYSQL_PACKET_EOF) {
        char log_str[256];
        sprintf(log_str, "EOF packet without any ResultSets\n");
        LOG(ERROR) << log_str;
        status = -1;
        msg = log_str;
    } else {
        char log_str[256];
        sprintf(log_str, "Abnormal packet_type=%d\n", resp->get_packet_type());
        LOG(ERROR) << log_str;
        status = -1;
        msg = log_str;
    }

    writer.Key("status");
    writer.Int(status);
    writer.Key("msg");
    writer.String(msg.c_str());
    writer.Key("data");
    writer.StartObject();
    if (multiple_rows.empty()) {
        writer.EndObject();
    } else {
        writer.Key("db");
        writer.String(db_name.c_str());
        writer.Key("table");
        writer.String(db_name.c_str());
        writer.Key("fields_names");
        auto f_names = fmt::to_string(fmt::join(field_names, ", "));
        writer.String(f_names.c_str());
        writer.Key("fields_types");
        auto f_types = fmt::to_string(fmt::join(field_types, ", "));
        writer.String(f_types.c_str());
        writer.Key("select_objs");
        writer.StartArray();
        for (auto& single_row : multiple_rows) {
            auto row_str = fmt::to_string(fmt::join(single_row, ", "));
            writer.String(row_str.c_str());
        }
        writer.EndArray();
        writer.EndObject();
    }

    *query_str = buf.GetString();
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
    std::string select(
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
std::string MySqlHelper::Impl::select(
    const std::string &table,
    const std::vector<std::string> &columns,
    const std::map<std::string, std::string> &conditions) {

    // prepare mysql url
    char mysql_url_chars[128];
    sprintf(mysql_url_chars, "mysql://%s:%s@%s/%s",
            _m_db_cfg.get_user_name().c_str(), _m_db_cfg.get_user_pw().c_str(),
            _m_db_cfg.get_host().c_str(), _m_db_cfg.get_db_name().c_str());
    std::string mysql_url = std::string(mysql_url_chars);

    // construct sql query
    std::stringstream col_ss;
    for (auto index = 0; index < columns.size(); ++index) {
        if (index == columns.size() - 1) {
            col_ss << columns[index];
        } else {
            col_ss << columns[index] << ", ";
        }
    }
    std::stringstream cond_ss;
    for (auto& iter : conditions) {
        cond_ss << " " << iter.first << iter.second;
    }
    std::string sql_query;
    if (conditions.empty()) {
        sql_query = fmt::format("SELECT {0} FROM {1};", col_ss.str(), table);
    } else {
        sql_query = fmt::format("SELECT {0} FROM {1} WHERE{2};", col_ss.str(), table, cond_ss.str());
    }
    LOG(INFO) << "query: " << sql_query;

    auto* task = WFTaskFactory::create_mysql_task(mysql_url, 5, internal_impl::select_callback);
    task->get_req()->set_query(sql_query);
    std::string result;
    task->user_data = &result;

    WFFacilities::WaitGroup wait_group(1);
    SeriesWork *series = Workflow::create_series_work(task,
            [&wait_group](const SeriesWork *series) {
                    wait_group.done();
            });
    series->set_context(&mysql_url);
    series->start();
    wait_group.wait();

    return result;
}


/************* Export Function Sets *************/

/***
 *
 * @param cfg
 */
MySqlHelper::MySqlHelper(const jinq::registration::mysql::MySqlDBConfig &cfg) {
    _m_pimpl = std::make_unique<Impl>();
    _m_pimpl->init(cfg);
}

/***
 *
 */
MySqlHelper::MySqlHelper() {
    _m_pimpl = std::make_unique<Impl>();
}

/***
 *
 */
MySqlHelper::~MySqlHelper() = default;

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
std::string MySqlHelper::select(
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
