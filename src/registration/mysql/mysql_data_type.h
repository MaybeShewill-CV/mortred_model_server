/************************************************
 * Copyright MaybeShewill-CV. All Rights Reserved.
 * Author: MaybeShewill-CV
 * File: mysql_data_type.h
 * Date: 23-2-14
 ************************************************/

#ifndef MORTRED_MODEL_SERVER_MYSQL_DATA_TYPE_H
#define MORTRED_MODEL_SERVER_MYSQL_DATA_TYPE_H

#include <vector>
#include <map>
#include <unordered_map>
#include <variant>

namespace jinq {
namespace registration {
namespace mysql {

using ColumnValue = std::variant<std::string, char*, int, float, double, bool>;
using ColumnKey = std::string;
using KVData = std::map<ColumnKey, ColumnValue>;
using QueryResult = std::vector<KVData>;

}
}
}

#endif // MORTRED_MODEL_SERVER_MYSQL_DATA_TYPE_H
