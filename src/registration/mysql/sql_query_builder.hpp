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
    virtual ~SqlBuilder() = default;

    /***
     *
     * @param table
     * @return
     */
    virtual SqlBuilder& from(const std::string& table) {
        _m_query += " FROM " + table;
        return *this;
    }

    /***
     *
     * @param condition
     * @return
     */
    virtual SqlBuilder& where(const std::string& condition) {
        _m_query += " WHERE " + condition;
        return *this;
    }

    /***
     *
     * @param column
     * @param ascending
     * @return
     */
    virtual SqlBuilder& order_by(const std::string& column, bool ascending) {
        _m_query += " ORDER BY " + column + (ascending ? " ASC" : " DESC");
        return *this;
    }

    /***
     *
     * @param table
     * @param condition
     * @return
     */
    virtual SqlBuilder& join(const std::string& table, const std::string& condition) {
        _m_query += " JOIN " + table + " ON " + condition;
        return *this;
    }

    /***
     *
     * @return
     */
    virtual std::string get_query() {
        if (!_m_query.empty() && _m_query.back() != ';') {
            _m_query += ";";
        }
        return _m_query;
    }

    /***
     *
     * @param columns
     * @return
     */
    virtual SqlBuilder& select(const std::string& columns) = 0;

    /***
     *
     * @param table
     * @param values
     * @return
     */
    virtual SqlBuilder& insert(const std::string& table, const mysql::KVData& values) = 0;

    /***
     *
     * @param table
     * @param values
     * @param condition
     * @return
     */
    virtual SqlBuilder& update(const std::string& table, const mysql::KVData& values) = 0;

    /***
     *
     * @param table
     * @param values
     * @param condition
     * @return
     */
    virtual SqlBuilder& update(const std::string& table, const mysql::KVData& values, const std::string& condition) = 0;

    /***
     *
     * @param table
     * @param condition
     * @return
     */
    virtual SqlBuilder& remove(const std::string& table) = 0;

    /***
     *
     * @param table
     * @param condition
     * @return
     */
    virtual SqlBuilder& remove(const std::string& table, const std::string& condition) = 0;

protected:
    std::string _m_query;
};

/***
 * select query sql builder
 */
class SelectBuilder : public SqlBuilder {
public:
    /***
     *
     */
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

  private:
    SqlBuilder& insert(const std::string& table, const mysql::KVData& values) override { return *this; };
    SqlBuilder& update(const std::string& table, const mysql::KVData& values) override { return *this; };
    SqlBuilder& update(const std::string& table, const mysql::KVData& values, const std::string& condition) override { return *this; };
    SqlBuilder& remove(const std::string& table) override { return *this; }
    SqlBuilder& remove(const std::string& table, const std::string& condition) override { return *this; }
};

/***
 * insert query sql builder
 */
class InsertBuilder : public SqlBuilder {
  public:
    /***
     *
     */
    ~InsertBuilder() override = default;

    /***
     * 
     * @param table 
     * @param values 
     * @return 
     */
    SqlBuilder& insert(const std::string& table, const mysql::KVData& values) override {
        std::string columns;
        std::string vals;
        for (const auto& [key, value] : values) {
            columns += key + ", ";
            if (std::holds_alternative<std::string>(value)) {
                vals += "'" + std::get<std::string>(value) + "', ";
            } else if (std::holds_alternative<int>(value)) {
                vals += std::to_string(std::get<int>(value)) + ", ";
            } else if (std::holds_alternative<float>(value)) {
                vals += std::to_string(std::get<float>(value)) + ", ";
            } else if (std::holds_alternative<double>(value)) {
                vals += std::to_string(std::get<double>(value)) + ", ";
            } else if (std::holds_alternative<char*>(value)) {
                vals += "'" + std::string(std::get<char*>(value)) + "', ";
            } else if (std::holds_alternative<bool>(value)) {
                if (std::get<bool>(value)) {
                    vals += std::string("true") + ", ";
                } else {
                    vals += std::string("false") + ", ";
                }
            } else {
                throw std::runtime_error("Unsupported value type.");
            }
        }
        columns = columns.substr(0, columns.length() - 2);
        vals = vals.substr(0, vals.length() - 2);
        _m_query = "INSERT INTO " + table + " (" + columns + ") VALUES (" + vals + ")";
        return *this;
    }
    
  private:
    SqlBuilder& select(const std::string& columns) override { return *this; }
    SqlBuilder& update(const std::string& table, const mysql::KVData& values) override { return *this; };
    SqlBuilder& update(const std::string& table, const mysql::KVData& values, const std::string& condition) override { return *this; }
    SqlBuilder& remove(const std::string& table) override { return *this; }
    SqlBuilder& remove(const std::string& table, const std::string& condition) override { return *this; }
};

/***
 * update query sql builder
 */
class UpdateBuilder : public SqlBuilder {
  public:
    /***
     *
     */
    ~UpdateBuilder() override = default;

    /***
     *
     * @param table
     * @param values
     * @return
     */
    SqlBuilder& update(const std::string& table, const mysql::KVData& values) override {
        std::string update_query = "UPDATE " + table + " SET ";
        for (const auto& [key, value] : values) {
            update_query += key + "=";
            if (std::holds_alternative<std::string>(value)) {
                update_query += "'" + std::get<std::string>(value) + "', ";
            } else if (std::holds_alternative<int>(value)) {
                update_query += std::to_string(std::get<int>(value)) + ", ";
            } else if (std::holds_alternative<float>(value)) {
                update_query += std::to_string(std::get<float>(value)) + ", ";
            } else if (std::holds_alternative<double>(value)) {
                update_query += std::to_string(std::get<double>(value)) + ", ";
            } else if (std::holds_alternative<bool>(value)) {
                if (std::get<bool>(value)) {
                    update_query += std::string("true") + ", ";
                } else {
                    update_query += std::string("false") + ", ";
                }
            } else {
                throw std::runtime_error("Unsupported value type.");
            }
        }
        _m_query = update_query.substr(0, update_query.length() - 2);
        return *this;
    };

    /***
     *
     * @param table
     * @param values
     * @param condition
     * @return
     */
    SqlBuilder& update(const std::string& table, const mysql::KVData& values, const std::string& condition) override {
        std::string update_query = "UPDATE " + table + " SET ";
        for (const auto& [key, value] : values) {
            update_query += key + "=";
            if (std::holds_alternative<std::string>(value)) {
                update_query += "'" + std::get<std::string>(value) + "', ";
            } else if (std::holds_alternative<int>(value)) {
                update_query += std::to_string(std::get<int>(value)) + ", ";
            } else if (std::holds_alternative<float>(value)) {
                update_query += std::to_string(std::get<float>(value)) + ", ";
            } else if (std::holds_alternative<double>(value)) {
                update_query += std::to_string(std::get<double>(value)) + ", ";
            } else if (std::holds_alternative<bool>(value)) {
                if (std::get<bool>(value)) {
                    update_query += std::string("true") + ", ";
                } else {
                    update_query += std::string("false") + ", ";
                }
            } else {
                throw std::runtime_error("Unsupported value type.");
            }
        }
        if (!condition.empty()) {
            update_query = update_query.substr(0, update_query.length() - 2) + " WHERE " + condition;
        }
        _m_query = update_query;
        return *this;
    }

  private:
    SqlBuilder& select(const std::string& columns) override { return *this; }
    SqlBuilder& insert(const std::string& table, const mysql::KVData& values) override { return *this; }
    SqlBuilder& remove(const std::string& table) override { return *this; }
    SqlBuilder& remove(const std::string& table, const std::string& condition) override { return *this; }
};

class DeleteBuilder : public SqlBuilder {
  public:
    /***
     *
     */
    ~DeleteBuilder() override = default;

    /***
     *
     * @param table
     * @param condition
     * @return
     */
    SqlBuilder& remove(const std::string& table) override {
        _m_query = "DELETE FROM " + table;
        return *this;
    }

    /***
     *
     * @param table
     * @param condition
     * @return
     */
    SqlBuilder& remove(const std::string& table, const std::string& condition) override {
        _m_query = "DELETE FROM " + table;
        if (!condition.empty()) {
            _m_query += " WHERE " + condition;
        }
        return *this;
    }

  private:
    SqlBuilder& select(const std::string& columns) override { return *this; }
    SqlBuilder& insert(const std::string& table, const mysql::KVData& values) override { return *this; }
    SqlBuilder& update(const std::string& table, const mysql::KVData& values) override { return *this; };
    SqlBuilder& update(const std::string& table, const mysql::KVData& values, const std::string& condition) override { return *this; }
};

}
}
}

#endif // MORTRED_MODEL_SERVER_SQLQUERYBUILDER_H
