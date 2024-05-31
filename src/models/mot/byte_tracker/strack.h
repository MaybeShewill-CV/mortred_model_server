/************************************************
 * Copyright MaybeShewill-CV. All Rights Reserved.
 * Author: MaybeShewill-CV
 * File: strack.h
 * Date: 24-5-30
 ************************************************/

#ifndef MORTRED_MODEL_SERVER_STRACK_H
#define MORTRED_MODEL_SERVER_STRACK_H

#include <opencv2/opencv.hpp>

#include "kalman_filter.h"
#include "data_type.h"

namespace jinq {
namespace models {
namespace mot {
namespace byte_tracker {

class STrack {
  public:
    /***
     *
     * @param tlwh_
     * @param score
     */
    STrack(std::vector<float> tlwh_, float score);

    /***
     *
     */
    ~STrack() = default;

    /***
     *
     * @param tlbr
     * @return
     */
    std::vector<float> static tlbr_to_tlwh(std::vector<float> &tlbr);

    /***
     *
     * @param stracks
     * @param kalman_filter
     */
    void static multi_predict(std::vector<STrack*> &stracks, KalmanFilter &kalman_filter);

    /***
     *
     */
    void static_tlwh();

    /***
     *
     */
    void static_tlbr();

    /***
     *
     * @param tlwh_tmp
     * @return
     */
    static std::vector<float> tlwh_to_xyah(const std::vector<float>& tlwh_tmp);

    /***
     *
     * @return
     */
    std::vector<float> to_xyah() const;

    /***
     *
     */
    void mark_lost();

    /***
     *
     */
    void mark_removed();

    /***
     *
     * @return
     */
    static int next_id();

    /***
     *
     * @return
     */
    int end_frame() const;

    /***
     *
     * @param kalman_filter
     * @param frame_id
     */
    void activate(KalmanFilter &kalman_filter, int frame_id);

    /***
     *
     * @param new_track
     * @param frame_id
     * @param new_id
     */
    void re_activate(STrack &new_track, int frame_id, bool new_id = false);

    /***
     *
     * @param new_track
     * @param frame_id
     */
    void update(STrack &new_track, int frame_id);

  public:
    bool is_activated;
    int track_id;
    int state;

    std::vector<float> _tlwh;
    std::vector<float> tlwh;
    std::vector<float> tlbr;
    int frame_id;
    int tracklet_len;
    int start_frame;

    KAL_MEAN mean;
    KAL_COVA covariance;
    float score;

  private:
    KalmanFilter _m_kalman_filter;
};

}
}
}
}

#endif // MORTRED_MODEL_SERVER_STRACK_H
