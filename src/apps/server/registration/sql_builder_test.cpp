/************************************************
* Copyright MaybeShewill-CV. All Rights Reserved.
* Author: MaybeShewill-CV
* File: sql_builder_test.cpp
* Date: 23-02-15
************************************************/

#include <string>

#include "glog/logging.h"

#include "registration/mysql/mysql_data_type.h"
#include "registration/mysql/sql_query_builder.hpp"

using jinq::registration::mysql::KVData;
using jinq::registration::mysql::SelectBuilder;
using jinq::registration::mysql::InsertBuilder;
using jinq::registration::mysql::UpdateBuilder;
using jinq::registration::mysql::DeleteBuilder;

int main(int argc, char** argv) {
    // test select query builder
    SelectBuilder select_builder;
    auto select_query = select_builder.select("name, age")
                         .from("user")
                         .where("age > 18")
                         .order_by("age", true)
                         .get_query();
    LOG(INFO) << "select query: " << select_query;

    // test insert query builder
    InsertBuilder insert_query_builder;
    KVData values = {{"name", std::string("John")}, {"age", 21}};
    std::string insert_query = insert_query_builder.insert("user", values).get_query();
    LOG(INFO) << "insert query: " << insert_query;

    // test update query builder
    UpdateBuilder update_query_builder;
    KVData update_values = {{"name", std::string("Mike")}, {"age", "21"}};
    std::string conditions = "name = 'John'";
    std::string update_query = update_query_builder.update("user", update_values).where(conditions).get_query();
    LOG(INFO) << "update query: " << update_query;

    // test delete query builder
    DeleteBuilder delete_query_builder;
    conditions = "name = 'Mike'";
    std::string delete_query = delete_query_builder.remove("user").where(conditions).get_query();
    LOG(INFO) << "delete query: " << delete_query << std::endl;

}

