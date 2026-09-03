/************************************************
 * Copyright MaybeShewill-CV. All Rights Reserved.
 * Author: MaybeShewill-CV
 * File: product_index.h
 * Date: 26-9-3
 ************************************************/

#ifndef MORTRED_APPS_PRODUCT_INDEX_H
#define MORTRED_APPS_PRODUCT_INDEX_H

#include <cstdio>
#include <functional>
#include <memory>
#include <string>
#include <vector>

#include "server/abstract_server.h"

namespace jinq {
namespace apps {

/***
 * Runtime projection of the factory `*_task.h::catalog()` tables. There is no
 * handwritten product list here: every row is copied from a catalog entry at
 * process start. `id` is the catalog `model_section` (`YOLOV8`, `MOBILENETV2`).
 */
struct ProductEntry {
    std::string id;
    std::string family;
    std::string display_name;
    std::string server_section;
    bool http = false;
    bool benchmark = false;
    std::function<std::unique_ptr<jinq::server::BaseAiServer>(const std::string &server_name)> make_server;
    std::function<int(int argc, char **argv)> run_benchmark;
};

class ProductIndex {
  public:
    static const std::vector<ProductEntry> &all();
    static const ProductEntry *find(const std::string &id);
    static void print_list(std::FILE *out);
};

} // namespace apps
} // namespace jinq

#endif // MORTRED_APPS_PRODUCT_INDEX_H
