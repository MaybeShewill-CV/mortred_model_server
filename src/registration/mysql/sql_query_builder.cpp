/************************************************
 * Copyright MaybeShewill-CV. All Rights Reserved.
 * Author: MaybeShewill-CV
 * File: SqlQueryBuilder.cpp
 * Date: 23-2-15
 ************************************************/

#include "sql_query_builder.h"

#include <unordered_map>

namespace jinq {
namespace registration {
namespace mysql {

/***
 *
 * @param columns
 * @return
 */
SqlQueryBuilder &SqlQueryBuilder::select(const std::string &columns) {
    _m_query = "SELECT " + columns;
    return *this;
}

}
}
}
