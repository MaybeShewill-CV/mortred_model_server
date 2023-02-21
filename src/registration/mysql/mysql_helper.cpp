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
using jinq::registration::mysql::QueryResult;

namespace internal_impl {

void select_callback(WFMySQLTask* task) {

    struct query_status {
        std::string query_result;
        StatusCode query_status;
    };
    auto* q_status = (query_status*)task->user_data;

    protocol::MySQLResponse* resp = task->get_resp();
    protocol::MySQLResultCursor cursor(resp);
    const protocol::MySQLField* const* fields;
    std::vector<protocol::MySQLCell> arr;

    rapidjson::StringBuffer buf;
    rapidjson::Writer<rapidjson::StringBuffer> writer(buf);
    writer.StartObject();
    int status;
    std::string msg;
    std::string db_name;
    std::string table_name;
    std::vector<std::string> field_names;
    std::vector<std::string> field_types;
    std::vector<std::vector<std::string> > multiple_rows;

    if (task->get_state() != WFT_STATE_SUCCESS) {
        std::string log_str = fmt::format(
            "error msg: {}", WFGlobal::get_error_string(task->get_state(), task->get_error()));
        writer.Key("status");
        writer.Int(task->get_state());
        writer.Key("msg");
        writer.String(log_str.c_str());
        writer.Key("data");
        writer.StartObject();
        writer.EndObject();
        writer.EndObject();
        q_status->query_result = buf.GetString();
        q_status->query_status = StatusCode::MYSQL_SELECT_FAILED;
        return;
    }

    if (cursor.get_cursor_status() == MYSQL_STATUS_GET_RESULT) {
        do {
            fields = cursor.fetch_fields();
            for (int i = 0; i < cursor.get_field_count(); i++) {
                if (i == 0) {
                    db_name = fields[i]->get_db();
                    table_name = fields[i]->get_table();
                }
                field_names.push_back(fields[i]->get_name());
                field_types.emplace_back(datatype2str(fields[i]->get_data_type()));
            }
            // fetch every row data
            while (cursor.fetch_row(arr)) {
                std::vector<std::string> single_row;
                for (auto& each_arr : arr) {
                    if (each_arr.is_string()) {
                        std::string res = each_arr.as_string();
                        single_row.push_back(res);
                    } else if (each_arr.is_int()) {
                        int res = each_arr.as_int();
                        single_row.push_back(std::to_string(res));
                    } else if (each_arr.is_ulonglong()) {
                        auto res = each_arr.as_ulonglong();
                        single_row.push_back(std::to_string(res));
                    } else if (each_arr.is_float()) {
                        auto res = each_arr.as_float();
                        single_row.push_back(std::to_string(res));
                    } else if (each_arr.is_double()) {
                        auto res = each_arr.as_double();
                        single_row.push_back(std::to_string(res));
                    } else if (each_arr.is_date()) {
                        auto res = each_arr.as_date();
                        single_row.push_back(res);
                    } else if (each_arr.is_time()) {
                        auto res = each_arr.as_time();
                        single_row.push_back(res);
                    } else if (each_arr.is_datetime()) {
                        auto res = each_arr.as_datetime();
                        single_row.push_back(res);
                    } else if (each_arr.is_null()) {
                        single_row.emplace_back("");
                    } else {
                        std::string res = each_arr.as_binary_string();
                        single_row.push_back(res);
                    }
                }
                multiple_rows.push_back(single_row);
            }
        } while (cursor.next_result_set());
        status = 0;
        msg = "ok";
    } else if (resp->get_packet_type() == MYSQL_PACKET_ERROR) {
        std::string log_str = fmt::format(
            "ERROR, error_code={} {}", task->get_resp()->get_error_code(), task->get_resp()->get_error_msg());
        LOG(ERROR) << log_str;
        LOG(ERROR) << task->get_req()->get_query();
        status = -1;
        msg = log_str;
    } else if (resp->get_packet_type() == MYSQL_PACKET_EOF) {
        std::string log_str = "EOF packet without any ResultSets";
        LOG(ERROR) << log_str;
        status = -1;
        msg = log_str;
    } else {
        std::string log_str = fmt::format("Abnormal packet_type={}", resp->get_packet_type());
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
        writer.String(table_name.c_str());
        writer.Key("fields_names");
        auto f_names = fmt::to_string(fmt::join(field_names, ", "));
        writer.String(f_names.c_str());
        writer.Key("fields_types");
        auto f_types = fmt::to_string(fmt::join(field_types, ", "));
        writer.String(f_types.c_str());
        writer.Key("select_objs");
        writer.StartArray();
        for (auto& single_row : multiple_rows) {
            writer.StartObject();
            for (auto idx = 0; idx < single_row.size(); ++idx) {
                writer.Key(field_names[idx].c_str());
                writer.StartObject();
                writer.Key("type");
                writer.String(field_types[idx].c_str());
                writer.Key("value");
                writer.String(single_row[idx].c_str());
                writer.EndObject();
            }
            writer.EndObject();
        }
        writer.EndArray();
        writer.EndObject();
    }
    writer.EndObject();
    q_status->query_result = buf.GetString();
    if (status != 0) {
        q_status->query_status = StatusCode::MYSQL_SELECT_FAILED;
    } else {
        q_status->query_status = StatusCode::OK;
    }
}

void insert_callback(WFMySQLTask* task) {

    auto* q_status = (StatusCode*)task->user_data;

    protocol::MySQLResponse* resp = task->get_resp();
    protocol::MySQLResultCursor cursor(resp);

    if (task->get_state() != WFT_STATE_SUCCESS) {
        std::string log_str = fmt::format(
            "error msg: {}", WFGlobal::get_error_string(task->get_state(), task->get_error()));
        *q_status = StatusCode::MYSQL_INSERT_FAILED;
        return;
    }

    if (resp->get_error_code() != 0) {
        std::string log_str = fmt::format(
            "ERROR, error_code={} {}", resp->get_error_code(), resp->get_error_msg());
        LOG(ERROR) << log_str;
        *q_status = StatusCode::MYSQL_INSERT_FAILED;
    } else if (resp->get_packet_type() != MYSQL_PACKET_OK) {
        std::string log_str = fmt::format("Abnormal packet_type={}", resp->get_packet_type());
        LOG(ERROR) << log_str;
        *q_status = StatusCode::MYSQL_INSERT_FAILED;
    } else {
        *q_status = StatusCode::OJBK;
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
     * @param query
     * @return
     */
    StatusCode select(const std::string& query, std::string& query_result);

    /***
     *
     * @param query
     * @return
     */
    StatusCode insert(const std::string& query);

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
    // check if db config is complete
    if (_m_db_cfg.get_user_name().empty()) {
        _m_successfully_initialized = false;
        return StatusCode::MYSQL_INIT_DB_CONFIG_ERROR;
    }

    _m_successfully_initialized = true;
    return StatusCode::OJBK;
}

/***
 *
 * @param query
 * @param query_result
 * @return
 */
StatusCode MySqlHelper::Impl::select(const std::string &query, std::string &query_result) {
    // prepare mysql url
    std::string mysql_url = fmt::format(
        "mysql://{}:{}@{}/{}", _m_db_cfg.get_user_name(), _m_db_cfg.get_user_pw(), _m_db_cfg.get_host(), _m_db_cfg.get_db_name());

    auto* task = WFTaskFactory::create_mysql_task(mysql_url, 5, internal_impl::select_callback);
    task->get_req()->set_query(query);
    struct query_status {
        std::string query_result;
        StatusCode query_status = StatusCode::OK;
    } q_status;
    task->user_data = &q_status;

    WFFacilities::WaitGroup wait_group(1);
    SeriesWork *series = Workflow::create_series_work(
        task,
        [&wait_group](const SeriesWork *series) {
            wait_group.done();
        });
    series->set_context(&mysql_url);
    series->start();
    wait_group.wait();

    query_result = q_status.query_result;
    return q_status.query_status;
}

/***
 *
 * @param query
 * @param query_result
 * @return
 */
StatusCode MySqlHelper::Impl::insert(const std::string &query) {
    // prepare mysql url
    std::string mysql_url = fmt::format(
        "mysql://{}:{}@{}/{}", _m_db_cfg.get_user_name(), _m_db_cfg.get_user_pw(), _m_db_cfg.get_host(), _m_db_cfg.get_db_name());

    auto* task = WFTaskFactory::create_mysql_task(mysql_url, 5, internal_impl::insert_callback);
    task->get_req()->set_query(query);
    StatusCode query_status;
    task->user_data = &query_status;

    WFFacilities::WaitGroup wait_group(1);
    SeriesWork *series = Workflow::create_series_work(
        task,
        [&wait_group](const SeriesWork *series) {
            wait_group.done();
        });
    series->set_context(&mysql_url);
    series->start();
    wait_group.wait();

    return query_status;
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
StatusCode MySqlHelper::select(const std::string &query, std::string &query_result) {
    return _m_pimpl->select(query, query_result);
}

/***
 *
 * @param query
 * @return
 */
StatusCode MySqlHelper::insert(const std::string &query) {
    return _m_pimpl->insert(query);
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
