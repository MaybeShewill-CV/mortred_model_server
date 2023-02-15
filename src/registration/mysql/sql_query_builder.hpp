/************************************************
 * Copyright MaybeShewill-CV. All Rights Reserved.
 * Author: MaybeShewill-CV
 * File: SqlQueryBuilder.h
 * Date: 23-2-15
 ************************************************/

#ifndef MORTRED_MODEL_SERVER_SQLQUERYBUILDER_H
#define MORTRED_MODEL_SERVER_SQLQUERYBUILDER_H

#include <string>
#include <sstream>
#include <memory>

#include "registration/mysql/mysql_data_type.h"

namespace jinq {
namespace registration {
namespace mysql {

class SqlBuilder {
public:
    /***
     *
    */
    virtual ~SqlBuilder() = 0;

    /***
     *
     * @param columns
     * @return
     */
    virtual SqlBuilder& select(const std::string& columns) = 0;

    /***
     *
     * @param table
     * @return
     */
    virtual SqlBuilder& from(const std::string& table) = 0;

    /***
     *
     * @param condition
     * @return
     */
    virtual SqlBuilder& where(const std::string& condition) = 0;

    /***
     *
     * @param column
     * @param ascending
     * @return
     */
    virtual SqlBuilder& order_by(const std::string& column, bool ascending) = 0;

    /***
     *
     * @param table
     * @param condition
     * @return
     */
    virtual SqlBuilder& join(const std::string& table, const std::string& condition) = 0;

    /***
     *
     * @param table
     * @param values
     * @return
     */
    virtual SqlBuilder& insert(const std::string& table, const std::unordered_map<std::string, std::string>& values) = 0;

    /***
     *
     * @param table
     * @param values
     * @param condition
     * @return
     */
    virtual SqlBuilder& update(const std::string& table, const std::unordered_map<std::string, std::string>& values, const std::string& condition) = 0;

    /***
     *
     * @param table
     * @param condition
     * @return
     */
    virtual SqlBuilder& remove(const std::string& table, const std::string& condition) = 0;

    /***
     *
     * @return
     */
    virtual std::string get_query() = 0;
};

/***
 * select query sql builder
 */
class SelectBuilder : public SqlBuilder {
public:

    ~SelectBuilder() override = default;

    /***
     *
     * @param columns
     * @return
     */
    SqlBuilder& select(const std::string& columns) override {
        _m_query = "SELECT " + columns;
        return *this;
    }

    /***
     *
     * @param table
     * @return
     */
    SqlBuilder& from(const std::string& table) override {
        _m_query += " FROM " + table;
        return *this;
    }

    /***
     *
     * @param condition
     * @return
     */
    SqlBuilder& where(const std::string& condition) override {
        _m_query += " WHERE " + condition;
        return *this;
    }

    /***
     *
     * @param column
     * @param ascending
     * @return
     */
    SqlBuilder& order_by(const std::string& column, bool ascending) override {
        _m_query += " ORDER BY " + column + (ascending ? " ASC" : " DESC");
        return *this;
    }

    /***
     *
     * @param table
     * @param condition
     * @return
     */
    SqlBuilder& join(const std::string& table, const std::string& condition) override {
        _m_query += " JOIN " + table + " ON " + condition;
        return *this;
    }

    /***
     *
     * @return
     */
    std::string get_query() override {
        if (!_m_query.empty() && _m_query.back() != ';') {
            _m_query += ";";
        }
        return _m_query;
    }

  private:
    /***
     *
     * @param table
     * @param values
     * @return
     */
    SqlBuilder& insert(const std::string& table, const std::unordered_map<std::string, std::string>& values) override {
        return *this;
    };

    /***
     *
     * @param table
     * @param values
     * @param condition
     * @return
     */
    SqlBuilder& update(
        const std::string& table,
        const std::unordered_map<std::string, std::string>& values,
        const std::string& condition) override {
        return *this;
    };

    /***
     *
     * @param table
     * @param condition
     * @return
     */
    SqlBuilder& remove(const std::string& table, const std::string& condition) override {
        return *this;
    }

private:
    std::string _m_query;
};

/***
 * insert query sql builder
 */
class InsertBuilder : public SqlBuilder {
  public:

    ~InsertBuilder() override = default;

    /***
     * 
     * @param table 
     * @param values 
     * @return 
     */
    SqlBuilder& insert(const std::string& table, const std::unordered_map<std::string, std::string>& values) override {
        std::string columns;
        std::string vals;
        for (const auto& [key, value] : values) {
            columns += key + ", ";
            vals += "'" + value + "', ";
        }
        columns = columns.substr(0, columns.length() - 2);
        vals = vals.substr(0, vals.length() - 2);
        _m_query = "INSERT INTO " + table + " (" + columns + ") VALUES (" + vals + ")";
        return *this;
    }

    /***
     *
     * @return
     */
    std::string get_query() override {
        if (!_m_query.empty() && _m_query.back() != ';') {
            _m_query += ";";
        }
        return _m_query;
    }
    
  private:
    SqlBuilder& select(const std::string& columns) override {
        return *this;
    }

    SqlBuilder& from(const std::string& table) override {
        return *this;
    }

    SqlBuilder& where(const std::string& condition) override {
        return *this;
    }

    SqlBuilder& order_by(const std::string& column, bool ascending) override {
        return *this;
    }

    SqlBuilder& join(const std::string& table, const std::string& condition) override {
        return *this;
    }

    SqlBuilder& update(const std::string& table, const std::unordered_map<std::string, std::string>& values, const std::string& condition) override {
        return *this;
    }

    SqlBuilder& remove(const std::string& table, const std::string& condition) override {
        return *this;
    }

  private:
    std::string _m_query;
};

/***
 * update query sql builder
 */
class UpdateBuilder : public SqlBuilder {
  public:

    ~UpdateBuilder() override = default;

    /***
     *
     * @param table
     * @param values
     * @param condition
     * @return
     */
    SqlBuilder& update(
        const std::string& table,
        const std::unordered_map<std::string, std::string>& values,
        const std::string& condition) override {
        std::stringstream query;
        query << "UPDATE " << table << " SET ";
        bool first = true;
        for (const auto& [key, value] : values) {
            if (!first) {
                query << ", ";
            }
            first = false;
            query << key << "='" << value << "'";
        }
        if (!condition.empty()) {
            query << " WHERE " << condition;
        }
        _m_query = query.str();
        return *this;
    }

    /***
     *
     * @return
     */
    std::string get_query() override {
        if (!_m_query.empty() && _m_query.back() != ';') {
            _m_query += ";";
        }
        return _m_query;
    }
    
  private:
    SqlBuilder& select(const std::string& columns) override { return *this; }
    SqlBuilder& from(const std::string& table) override { return *this; }
    SqlBuilder& where(const std::string& condition) override { return *this; }
    SqlBuilder& order_by(const std::string& column, bool ascending) override { return *this; }
    SqlBuilder& join(const std::string& table, const std::string& condition) override { return *this; }
    SqlBuilder& insert(const std::string& table, const std::unordered_map<std::string, std::string>& values) override { return *this; }
    SqlBuilder& remove(const std::string& table, const std::string& condition) override { return *this; }

    std::string _m_query;
};

class DeleteBuilder : public SqlBuilder {
  public:

    ~DeleteBuilder() override = default;

    SqlBuilder& remove(const std::string& table, const std::string& condition) override {
        _m_query = "DELETE FROM " + table;
        if (!condition.empty()) {
            _m_query += " WHERE " + condition;
        }
        return *this;
    }

    std::string get_query() override {
        if (!_m_query.empty() && _m_query.back() != ';') {
            _m_query += ";";
        }
        return _m_query;
    }

  private:
    SqlBuilder& select(const std::string& columns) override { return *this; }
    SqlBuilder& from(const std::string& table) override { return *this; }
    SqlBuilder& where(const std::string& condition) override { return *this; }
    SqlBuilder& order_by(const std::string& column, bool ascending) override { return *this; }
    SqlBuilder& join(const std::string& table, const std::string& condition) override { return *this; }
    SqlBuilder& insert(const std::string& table, const std::unordered_map<std::string, std::string>& values) override { return *this; }
    SqlBuilder& update(
        const std::string& table,
        const std::unordered_map<std::string, std::string>& values,
        const std::string& condition) override { return *this; }

    std::string _m_query;
};

}
}
}

#endif // MORTRED_MODEL_SERVER_SQLQUERYBUILDER_H
