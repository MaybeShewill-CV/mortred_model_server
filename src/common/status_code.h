/************************************************
* Copyright MaybeShewill-CV. All Rights Reserved.
* Author: MaybeShewill-CV
* File: status_code.h
* Date: 22-6-2
************************************************/

#ifndef MORTRED_MODEL_SERVER_STATUSCODE_H
#define MORTRED_MODEL_SERVER_STATUSCODE_H

#include <string>

namespace jinq {
namespace common {

/***
 * 状态码单一事实来源：枚举定义与错误文案必须同时维护在这里。
 * 新增错误码时，只需在列表中增加一行，枚举、error_code_to_str、单元测试会自动同步。
 */
#define MORTRED_STATUS_CODE_LIST(X) \
    X(OK, 0, "OK") \
    X(MODEL_INIT_FAILED, 1, "model init failed") \
    X(MODEL_RUN_SESSION_FAILED, 2, "model run session failed") \
    X(MODEL_EMPTY_INPUT_IMAGE, 3, "model input empty") \
    X(MODEL_RUN_TIMEOUT, 4, "model run timeout") \
    X(SERVER_INIT_FAILED, 11, "server init failed") \
    X(SERVER_RUN_FAILED, 12, "server run failed") \
    X(FILE_READ_ERROR, 30, "file read error") \
    X(FILE_WRITE_ERROR, 31, "file write error") \
    X(FILE_NOT_EXIST_ERROR, 32, "file not exist error") \
    X(COMPRESS_ERROR, 40, "compress not support") \
    X(UNCOMPRESS_ERROR, 41, "uncompress error") \
    X(JSON_DECODE_ERROR, 50, "decode json error") \
    X(JSON_ENCODE_ERROR, 51, "encode json error") \
    X(MYSQL_INIT_DB_CONFIG_ERROR, 60, "init mysql connection failed") \
    X(MYSQL_SELECT_FAILED, 61, "exec select sql failed") \
    X(MYSQL_INSERT_FAILED, 62, "exec insert sql failed") \
    X(MYSQL_UPDATE_FAILED, 63, "exec update sql failed") \
    X(MYSQL_DELETE_FAILED, 64, "exec delete sql failed") \
    X(ROUTER_ADD_HANDLER_FAILED, 70, "add handler to router table failed") \
    X(ROUTER_GET_HANDLER_FAILED, 71, "get handler from router table failed") \
    X(ROUTER_GET_PROJECT_NAMES_FAILED, 72, "get project names from router table failed") \
    X(ROUTER_GET_SERVICE_NAMES_FAILED, 73, "get service names from router table failed") \
    X(ROUTER_GET_URI_NAMES_FAILED, 74, "get uri names from router table failed") \
    X(TOKENIZE_UNKNOWN_TOKEN, 80, "unknown token") \
    X(TOKENIZE_FAILED, 81, "tokenize failed") \
    X(ENCODE_TOKEN_FAILED, 82, "encode token failed") \
    X(TRT_CUDA_ERROR, 90, "tensorrt cuda error") \
    X(TRT_ALLOC_MEMO_FAILED, 91, "tensorrt allocate memory failed") \
    X(TRT_CONVERT_ONNX_MODEL_FAILED, 92, "convert onnx model to trt failed") \
    X(TRT_ALLOC_DYNAMIC_SHAPE_MEMO, 93, "tensorrt allocate dynamic shape memory failed") \
    X(LLM_CONTEXT_SIZE_EXCEEDED, 100, "llm context size exceeded") \
    X(LLM_APPLY_CHAT_TEMPLATE_FAILED, 101, "llm apply chat template failed") \
    X(LLM_LLAMA_DECODE_FAILED, 102, "llm llama decode failed") \
    X(LLM_LLAMA_SAMPLE_NEW_TOKEN_FAILED, 103, "llm llama sample new token failed") \
    X(LLM_SHIFT_KV_CACHE_FAILED, 104, "llm shift kv cache failed") \
    X(RAG_PARSE_WIKI_STR_FAILED, 120, "rag parse wiki string failed") \
    X(RAG_BUILD_CORPUS_INDEX_FAILED, 121, "rag build corpus index failed") \
    X(RAG_LOAD_INDEX_FAILED, 122, "rag load index failed") \
    X(RAG_LOAD_SEGMENT_CORPUS_FAILED, 123, "rag load segment corpus failed") \
    X(RAG_SEARCH_SEGMENT_CORPUS_FAILED, 124, "rag search segment corpus failed") \
    X(VLM_QWEN_ENCODE_IMAGE_FAILED, 140, "vlm qwen encode image failed") \
    X(VLM_QWEN_DECODE_IMAGE_EMBEDDING_FAILED, 141, "vlm qwen decode image embedding failed") \
    X(VLM_QWEN_PARSE_IMAGE_URL_FAILED, 142, "vlm qwen parse image url failed")

enum StatusCode {
#define MORTRED_STATUS_CODE_DEFINE(name, value, desc) name = value,
    MORTRED_STATUS_CODE_LIST(MORTRED_STATUS_CODE_DEFINE)
#undef MORTRED_STATUS_CODE_DEFINE
    // 别名：与 OK 等价，不进入宏列表（避免 switch 重复 case）
    OJBK = OK,
};

/***
 *
 * @param error_code
 * @return
 */
std::string error_code_to_str(int error_code);

}
}

#endif //MORTRED_MODEL_SERVER_STATUSCODE_H
