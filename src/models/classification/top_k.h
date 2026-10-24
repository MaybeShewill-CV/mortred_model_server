/************************************************
* Copyright MaybeShewill-CV. All Rights Reserved.
* Author: MaybeShewill-CV
* File: top_k.h
* Date: 26-10-24
************************************************/

// Request-level top_k for the classification family: keep the k highest scores
// in descending order. Three models carried this loop verbatim.

#ifndef MORTRED_MODELS_CLASSIFICATION_TOP_K_H
#define MORTRED_MODELS_CLASSIFICATION_TOP_K_H

#include <algorithm>
#include <cstddef>
#include <numeric>
#include <utility>
#include <vector>

namespace mortred {
namespace models {
namespace classification {

/*** In place: when keep < scores->size(), replace the scores with the keep
 * highest values in descending order. keep <= 0 yields an empty vector (the
 * caller's parameter is validated to be >= 1; this keeps direct callers safe),
 * keep >= size is a no-op. Order among equal scores is partial_sort's, i.e.
 * unchanged from the loop this replaced. */
inline void apply_top_k(std::vector<float>* scores, int keep) {
    if (scores == nullptr || keep >= static_cast<int>(scores->size())) {
        return;
    }
    if (keep <= 0) {
        scores->clear();
        return;
    }
    std::vector<int> order(scores->size());
    std::iota(order.begin(), order.end(), 0);
    std::partial_sort(order.begin(), order.begin() + keep, order.end(),
                      [scores](int lhs, int rhs) {
                          return (*scores)[static_cast<std::size_t>(lhs)] >
                                 (*scores)[static_cast<std::size_t>(rhs)];
                      });
    std::vector<float> top;
    top.reserve(static_cast<std::size_t>(keep));
    for (int idx = 0; idx < keep; ++idx) {
        top.push_back((*scores)[static_cast<std::size_t>(order[static_cast<std::size_t>(idx)])]);
    }
    *scores = std::move(top);
}

} // namespace classification
} // namespace models
} // namespace mortred

#endif // MORTRED_MODELS_CLASSIFICATION_TOP_K_H
