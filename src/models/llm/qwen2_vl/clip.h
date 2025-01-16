#ifndef CLIP_H
#define CLIP_H

#include <stddef.h>
#include <stdint.h>
#include <vector>
#include <map>
#include <string>

#include "ggml/ggml.h"
#include "ggml/ggml-cpu.h"
#include "ggml/ggml-alloc.h"
#include "ggml/ggml-backend.h"
#include "ggml/gguf.h"

#ifdef LLAMA_SHARED
#    if defined(_WIN32) && !defined(__MINGW32__)
#        ifdef LLAMA_BUILD
#            define CLIP_API __declspec(dllexport)
#        else
#            define CLIP_API __declspec(dllimport)
#        endif
#    else
#        define CLIP_API __attribute__ ((visibility ("default")))
#    endif
#else
#    define CLIP_API
#endif

#ifdef __cplusplus
extern "C" {
#endif

struct clip_ctx;

// RGB uint8 image
struct clip_image_u8 {
    int nx = 0;
    int ny = 0;

    std::vector<uint8_t> buf;
};

// RGB float32 image (NHWC)
// Memory layout: RGBRGBRGB...
struct clip_image_f32 {
    int nx = 0;
    int ny = 0;

    std::vector<float> buf;
};

struct clip_image_size {
    int width;
    int height;
};

struct clip_image_u8_batch {
    struct clip_image_u8 * data;
    size_t size;
};

struct clip_image_f32_batch {
    struct clip_image_f32 * data;
    size_t size;
};

enum projector_type {
    PROJECTOR_TYPE_MLP,
    PROJECTOR_TYPE_MLP_NORM,
    PROJECTOR_TYPE_LDP,
    PROJECTOR_TYPE_LDPV2,
    PROJECTOR_TYPE_RESAMPLER,
    PROJECTOR_TYPE_MERGER,
    PROJECTOR_TYPE_UNKNOWN,
};

static std::map<projector_type, std::string> PROJECTOR_TYPE_NAMES = {
    { PROJECTOR_TYPE_MLP, "mlp" },
    { PROJECTOR_TYPE_LDP, "ldp" },
    { PROJECTOR_TYPE_LDPV2, "ldpv2"},
    { PROJECTOR_TYPE_RESAMPLER, "resampler"},
    { PROJECTOR_TYPE_MERGER, "qwen2vl_merger"},
};

//
// clip layers
//

struct clip_hparams {
    int32_t image_size;
    int32_t patch_size;
    int32_t hidden_size;
    int32_t n_intermediate;
    int32_t projection_dim;
    int32_t n_head;
    int32_t n_layer;

    float eps;

    char mm_patch_merge_type[32] = "flat"; // spatial_unpad or flat (default)

    int32_t image_grid_pinpoints[32];
    int32_t image_crop_resolution;
};

struct clip_layer {
    // attention
    struct ggml_tensor * k_w;
    struct ggml_tensor * k_b;
    struct ggml_tensor * q_w;
    struct ggml_tensor * q_b;
    struct ggml_tensor * v_w;
    struct ggml_tensor * v_b;

    struct ggml_tensor * o_w;
    struct ggml_tensor * o_b;

    // layernorm 1
    struct ggml_tensor * ln_1_w;
    struct ggml_tensor * ln_1_b;

    // ff
    struct ggml_tensor * ff_i_w;
    struct ggml_tensor * ff_i_b;

    struct ggml_tensor * ff_o_w;
    struct ggml_tensor * ff_o_b;

    // layernorm 2
    struct ggml_tensor * ln_2_w;
    struct ggml_tensor * ln_2_b;
};

struct clip_vision_model {
    struct clip_hparams hparams;

    // embeddings
    struct ggml_tensor * class_embedding;
    struct ggml_tensor * patch_embeddings_0;
    struct ggml_tensor * patch_embeddings_1;  // second Conv2D kernel when we decouple Conv3D along temproal dimension (Qwen2VL)
    struct ggml_tensor * patch_bias;
    struct ggml_tensor * position_embeddings;

    struct ggml_tensor * pre_ln_w;
    struct ggml_tensor * pre_ln_b;

    std::vector<clip_layer> layers;

    struct ggml_tensor * post_ln_w;
    struct ggml_tensor * post_ln_b;

    struct ggml_tensor * projection;

    // LLaVA projection
    struct ggml_tensor * mm_0_w = NULL;
    struct ggml_tensor * mm_0_b = NULL;
    struct ggml_tensor * mm_2_w = NULL;
    struct ggml_tensor * mm_2_b = NULL;

    struct ggml_tensor * image_newline = NULL;

    // Yi type models with mlp+normalization projection
    struct ggml_tensor * mm_1_w = NULL; // Yi type models have 0, 1, 3, 4
    struct ggml_tensor * mm_1_b = NULL;
    struct ggml_tensor * mm_3_w = NULL;
    struct ggml_tensor * mm_3_b = NULL;
    struct ggml_tensor * mm_4_w = NULL;
    struct ggml_tensor * mm_4_b = NULL;

    // MobileVLM projection
    struct ggml_tensor * mm_model_mlp_1_w;
    struct ggml_tensor * mm_model_mlp_1_b;
    struct ggml_tensor * mm_model_mlp_3_w;
    struct ggml_tensor * mm_model_mlp_3_b;
    struct ggml_tensor * mm_model_block_1_block_0_0_w;
    struct ggml_tensor * mm_model_block_1_block_0_1_w;
    struct ggml_tensor * mm_model_block_1_block_0_1_b;
    struct ggml_tensor * mm_model_block_1_block_1_fc1_w;
    struct ggml_tensor * mm_model_block_1_block_1_fc1_b;
    struct ggml_tensor * mm_model_block_1_block_1_fc2_w;
    struct ggml_tensor * mm_model_block_1_block_1_fc2_b;
    struct ggml_tensor * mm_model_block_1_block_2_0_w;
    struct ggml_tensor * mm_model_block_1_block_2_1_w;
    struct ggml_tensor * mm_model_block_1_block_2_1_b;
    struct ggml_tensor * mm_model_block_2_block_0_0_w;
    struct ggml_tensor * mm_model_block_2_block_0_1_w;
    struct ggml_tensor * mm_model_block_2_block_0_1_b;
    struct ggml_tensor * mm_model_block_2_block_1_fc1_w;
    struct ggml_tensor * mm_model_block_2_block_1_fc1_b;
    struct ggml_tensor * mm_model_block_2_block_1_fc2_w;
    struct ggml_tensor * mm_model_block_2_block_1_fc2_b;
    struct ggml_tensor * mm_model_block_2_block_2_0_w;
    struct ggml_tensor * mm_model_block_2_block_2_1_w;
    struct ggml_tensor * mm_model_block_2_block_2_1_b;

    // MobileVLM_V2 projection
    struct ggml_tensor * mm_model_mlp_0_w;
    struct ggml_tensor * mm_model_mlp_0_b;
    struct ggml_tensor * mm_model_mlp_2_w;
    struct ggml_tensor * mm_model_mlp_2_b;
    struct ggml_tensor * mm_model_peg_0_w;
    struct ggml_tensor * mm_model_peg_0_b;

    // MINICPMV projection
    struct ggml_tensor * mm_model_pos_embed_k;
    struct ggml_tensor * mm_model_query;
    struct ggml_tensor * mm_model_proj;
    struct ggml_tensor * mm_model_kv_proj;
    struct ggml_tensor * mm_model_attn_q_w;
    struct ggml_tensor * mm_model_attn_q_b;
    struct ggml_tensor * mm_model_attn_k_w;
    struct ggml_tensor * mm_model_attn_k_b;
    struct ggml_tensor * mm_model_attn_v_w;
    struct ggml_tensor * mm_model_attn_v_b;
    struct ggml_tensor * mm_model_attn_o_w;
    struct ggml_tensor * mm_model_attn_o_b;
    struct ggml_tensor * mm_model_ln_q_w;
    struct ggml_tensor * mm_model_ln_q_b;
    struct ggml_tensor * mm_model_ln_kv_w;
    struct ggml_tensor * mm_model_ln_kv_b;
    struct ggml_tensor * mm_model_ln_post_w;
    struct ggml_tensor * mm_model_ln_post_b;
};

struct clip_ctx {
    bool has_text_encoder    = false;
    bool has_vision_encoder  = false;
    bool has_llava_projector = false;
    bool has_minicpmv_projector = false;
    bool has_qwen2vl_merger = false;
    int minicpmv_version = 2;

    struct clip_vision_model vision_model;
    projector_type proj_type = PROJECTOR_TYPE_MLP;

    float image_mean[3];
    float image_std[3];
    bool use_gelu = false;
    bool use_silu = false;
    int32_t ftype = 1;

    bool has_class_embedding = true;
    bool has_pre_norm = true;
    bool has_post_norm = false;
    bool has_patch_bias = false;

    struct gguf_context * ctx_gguf;
    struct ggml_context * ctx_data;

    std::vector<uint8_t> buf_compute_meta;

    // memory buffers to evaluate the model
    ggml_backend_buffer_t params_buffer  = NULL;

    ggml_backend_t backend       = NULL;
    ggml_gallocr_t compute_alloc = NULL;

    struct clip_image_size * load_image_size;
};

CLIP_API struct clip_ctx * clip_model_load    (const char * fname, int verbosity);
CLIP_API struct clip_ctx * clip_model_load_cpu(const char * fname, int verbosity);
CLIP_API struct clip_ctx * clip_model_load_cuda(const char * fname, int device_id, int verbosity);

CLIP_API void clip_free(struct clip_ctx * ctx);

CLIP_API size_t clip_embd_nbytes(const struct clip_ctx * ctx);
CLIP_API size_t clip_embd_nbytes_by_img(const struct clip_ctx * ctx, int img_h, int img_w);

CLIP_API int32_t clip_image_size (const struct clip_ctx * ctx);
CLIP_API int32_t clip_patch_size (const struct clip_ctx * ctx);
CLIP_API int32_t clip_hidden_size(const struct clip_ctx * ctx);

// TODO: should be enum, not string
CLIP_API const char * clip_patch_merge_type(const struct clip_ctx * ctx);

CLIP_API const int32_t * clip_image_grid(const struct clip_ctx * ctx);

CLIP_API int clip_n_patches        (const struct clip_ctx * ctx);
CLIP_API int clip_n_patches_by_img (const struct clip_ctx * ctx, struct clip_image_f32 * img);
CLIP_API int clip_n_mmproj_embd    (const struct clip_ctx * ctx);

CLIP_API int clip_uhd_num_image_embeds_col(struct clip_ctx * ctx_clip);
CLIP_API void clip_add_load_image_size(struct clip_ctx * ctx_clip, struct clip_image_size * load_image_size);
CLIP_API struct clip_image_size * clip_get_load_image_size(struct clip_ctx * ctx_clip);

CLIP_API struct clip_image_size * clip_image_size_init();
CLIP_API struct clip_image_u8  * clip_image_u8_init ();
CLIP_API struct clip_image_f32 * clip_image_f32_init();

CLIP_API void clip_image_u8_free (struct clip_image_u8  * img);
CLIP_API void clip_image_f32_free(struct clip_image_f32 * img);
CLIP_API void clip_image_u8_batch_free (struct clip_image_u8_batch  * batch);
CLIP_API void clip_image_f32_batch_free(struct clip_image_f32_batch * batch);

CLIP_API bool clip_image_load_from_file(const char * fname, struct clip_image_u8 * img);

/** interpret bytes as an image file with length bytes_length, and use the result to populate img */
CLIP_API bool clip_image_load_from_bytes(const unsigned char * bytes, size_t bytes_length, struct clip_image_u8 * img);

/** preprocess img and store the result in res_imgs, pad_to_square may be overridden to false depending on model configuration */
CLIP_API bool clip_image_preprocess(struct clip_ctx * ctx, const struct clip_image_u8 * img, struct clip_image_f32_batch * res_imgs );

CLIP_API struct ggml_tensor * clip_get_newline_tensor(const struct clip_ctx * ctx);

CLIP_API bool clip_image_encode      (struct clip_ctx * ctx, int n_threads, struct clip_image_f32 * img, float * vec);
CLIP_API bool clip_image_batch_encode(struct clip_ctx * ctx, int n_threads, const struct clip_image_f32_batch * imgs, float * vec);

CLIP_API bool clip_model_quantize(const char * fname_inp, const char * fname_out, int itype);

CLIP_API int clip_is_minicpmv(const struct clip_ctx * ctx);
CLIP_API bool clip_is_qwen2vl(const struct clip_ctx * ctx);

CLIP_API bool clip_encode_float_image (struct clip_ctx * ctx, int n_threads, float * img, int h, int w, float * vec);

#ifdef __cplusplus
}
#endif

#endif // CLIP_H
