/************************************************
 * Copyright MaybeShewill-CV. All Rights Reserved.
 * Author: MaybeShewill-CV
 * File: byte_tracker.cpp
 * Date: 24-5-31
 ************************************************/

#include "byte_tracker.h"

#include <glog/logging.h>

#include "common/status_code.h"
#include "lapjv.h"
#include "kalman_filter.h"

namespace jinq {
namespace models {
namespace mot {
namespace byte_tracker {

using jinq::common::StatusCode;

/***************** Impl Function Sets ******************/

class ByteTracker::Impl {
  public:
    /***
     *
     */
    Impl() = default;

    /***
     *
     */
    ~Impl() = default;

    /***
    *
    * @param transformer
     */
    Impl(const Impl& transformer) = delete;

    /***
     *
     * @param transformer
     * @return
     */
    Impl& operator=(const Impl& transformer) = delete;

    /***
     *
     * @param cfg_file_path
     * @return
     */
    StatusCode init(const decltype(toml::parse(""))& config);

    /***
     *
     * @param objects
     * @return
     */
    std::vector<STrack> update(const std::vector<Object>& objects);

    /***
     *
     * @param idx
     * @return
     */
    static cv::Scalar get_color(int idx);

    /***
     *
     * @return
     */
    bool is_successfully_initialized() const {
        return _m_successfully_initialized;
    };

  private:
    // byte tracker threshold
    float _m_track_thresh = 0.0f;
    float _m_high_thresh = 0.0f;
    float _m_match_thresh = 0.0f;
    int _m_frame_id = 0;
    int _m_max_time_lost = -1;
    int _m_frame_rate = -1;

    // track obj cls info
    std::unordered_map<int, std::string> _m_tracked_cls_ids;

    // byte track containers
    std::vector<STrack> _m_tracked_stracks;
    std::vector<STrack> _m_lost_stracks;
    std::vector<STrack> _m_removed_stracks;
    KalmanFilter _m_kalman_filter;

    // init flag
    bool _m_successfully_initialized = false;

  private:
    /***
     *
     * @param tlista
     * @param tlistb
     * @return
     */
    static std::vector<STrack*> joint_stracks(std::vector<STrack*> &tlista, std::vector<STrack> &tlistb);

    /***
     *
     * @param tlista
     * @param tlistb
     * @return
     */
    static std::vector<STrack> joint_stracks(std::vector<STrack> &tlista, std::vector<STrack> &tlistb);

    /***
     *
     * @param tlista
     * @param tlistb
     * @return
     */
    static std::vector<STrack> sub_stracks(std::vector<STrack> &tlista, std::vector<STrack> &tlistb);

    /***
     *
     * @param resa
     * @param resb
     * @param stracksa
     * @param stracksb
     */
    static void remove_duplicate_stracks(
        std::vector<STrack> &resa, std::vector<STrack> &resb,
        std::vector<STrack> &stracksa, std::vector<STrack> &stracksb);

    /***
     *
     * @param cost_matrix
     * @param cost_matrix_size
     * @param cost_matrix_size_size
     * @param thresh
     * @param matches
     * @param unmatched_a
     * @param unmatched_b
     */
    static void linear_assignment(
        std::vector<std::vector<float> > &cost_matrix, int cost_matrix_size, int cost_matrix_size_size, float thresh,
        std::vector<std::vector<int> > &matches, std::vector<int> &unmatched_a, std::vector<int> &unmatched_b);

    /***
     *
     * @param atracks
     * @param btracks
     * @param dist_size
     * @param dist_size_size
     * @return
     */
    static std::vector<std::vector<float> > iou_distance(
        std::vector<STrack*> &atracks, std::vector<STrack> &btracks, int &dist_size, int &dist_size_size);

    /***
     *
     * @param atracks
     * @param btracks
     * @return
     */
    static std::vector<std::vector<float> > iou_distance(std::vector<STrack> &atracks, std::vector<STrack> &btracks);

    /***
     *
     * @param atlbrs
     * @param btlbrs
     * @return
     */
    static std::vector<std::vector<float> > ious(std::vector<std::vector<float> > &atlbrs, std::vector<std::vector<float> > &btlbrs);

    /***
     *
     * @param cost
     * @param rowsol
     * @param colsol
     * @param extend_cost
     * @param cost_limit
     * @param return_cost
     * @return
     */
    static double lapjv(const std::vector<std::vector<float> > &cost, std::vector<int> &rowsol, std::vector<int> &colsol,
                 bool extend_cost = false, float cost_limit = static_cast<float>(LONG_MAX), bool return_cost = true);

};

/***
 *
 * @param config
 * @return
 */
StatusCode ByteTracker::Impl::init(const decltype(toml::parse("")) &config) {
    if (!config.contains("BYTE_TRACK")) {
        LOG(ERROR) << "Config file does not contain BYTE_TRACK section";
        _m_successfully_initialized = false;
        return StatusCode::MODEL_INIT_FAILED;
    }

    toml::value cfg_content = config.at("BYTE_TRACK");

    _m_track_thresh = static_cast<float>(cfg_content.at("tracker_thresh").as_floating());
    _m_high_thresh = static_cast<float>(cfg_content.at("tracker_high_thresh").as_floating());
    _m_match_thresh = static_cast<float>(cfg_content.at("tracker_match_thresh").as_floating());
    _m_frame_rate = static_cast<int>(cfg_content.at("frame_rate").as_integer());
    auto track_buffer = static_cast<float>(cfg_content.at("track_buffer").as_integer());
    _m_max_time_lost = static_cast<int>(static_cast<float>(_m_frame_rate) / 30.0f * track_buffer);  // seconds
    _m_frame_id = 0;

    std::vector<int> tracked_cls_ids;
    for (auto& value : cfg_content.at("tracked_cls_ids").as_array()) {
        tracked_cls_ids.push_back(static_cast<int>(value.as_integer()));
    }
    std::vector<std::string> tracked_cls_names;
    for (auto& value : cfg_content.at("tracked_cls_names").as_array()) {
        tracked_cls_names.push_back(value.as_string());
    }
    assert(tracked_cls_ids.size() == tracked_cls_names.size());
    for (auto idx = 0; idx < tracked_cls_ids.size(); ++idx) {
        _m_tracked_cls_ids.insert(std::make_pair(tracked_cls_ids[idx], tracked_cls_names[idx]));
    }
    _m_successfully_initialized = true;

    return StatusCode::OK;
}

/***
 *
 * @param objects
 * @return
 */
std::vector<STrack> ByteTracker::Impl::update(const std::vector<Object> &objects) {
    ////////////////// Step 1: Get detections //////////////////
    _m_frame_id++;
    std::vector<STrack> activated_stracks;
    std::vector<STrack> refind_stracks;
    std::vector<STrack> removed_stracks;
    std::vector<STrack> lost_stracks;
    std::vector<STrack> detections;
    std::vector<STrack> detections_low;

    std::vector<STrack> detections_cp;
    std::vector<STrack> tracked_stracks_swap;
    std::vector<STrack> resa, resb;
    std::vector<STrack> output_stracks;

    std::vector<STrack*> unconfirmed;
    std::vector<STrack*> tracked_stracks;
    std::vector<STrack*> strack_pool;
    std::vector<STrack*> r_tracked_stracks;

    if (!objects.empty()) {
        for (const auto & object : objects) {
            // only track specific objs
            if (_m_tracked_cls_ids.find(object.class_id) == _m_tracked_cls_ids.end()) {
                continue;
            }

            // split objects
            std::vector<float> tlbr_;
            tlbr_.resize(4);
            tlbr_[0] = object.bbox.x;
            tlbr_[1] = object.bbox.y;
            tlbr_[2] = object.bbox.x + object.bbox.width;
            tlbr_[3] = object.bbox.y + object.bbox.height;

            float score = object.score;

            STrack strack(STrack::tlbr_to_tlwh(tlbr_), score);
            if (score >= _m_track_thresh) {
                detections.push_back(strack);
            } else {
                detections_low.push_back(strack);
            }
        }
    }

    // Add newly detected tracklets to tracked_stracks
    for (auto & _m_tracked_strack : _m_tracked_stracks) {
        if (!_m_tracked_strack.is_activated) {
            unconfirmed.push_back(&_m_tracked_strack);
        } else {
            tracked_stracks.push_back(&_m_tracked_strack);
        }
    }

    ////////////////// Step 2: First association, with IoU //////////////////
    strack_pool = joint_stracks(tracked_stracks, _m_lost_stracks);
    STrack::multi_predict(strack_pool, _m_kalman_filter);

    std::vector<std::vector<float> > dists;
    int dist_size = 0, dist_size_size = 0;
    dists = iou_distance(strack_pool, detections, dist_size, dist_size_size);

    std::vector<std::vector<int> > matches;
    std::vector<int> u_track, u_detection;
    linear_assignment(dists, dist_size, dist_size_size, _m_match_thresh, matches, u_track, u_detection);

    for (auto & matche : matches) {
        STrack *track = strack_pool[matche[0]];
        STrack *det = &detections[matche[1]];
        if (track->state == TrackState::Tracked) {
            track->update(*det, _m_frame_id);
            activated_stracks.push_back(*track);
        } else {
            track->re_activate(*det, _m_frame_id, false);
            refind_stracks.push_back(*track);
        }
    }

    ////////////////// Step 3: Second association, using low score dets //////////////////
    for (int i : u_detection) {
        detections_cp.push_back(detections[i]);
    }
    detections.clear();
    detections.assign(detections_low.begin(), detections_low.end());

    for (int i : u_track) {
        if (strack_pool[i]->state == TrackState::Tracked) {
            r_tracked_stracks.push_back(strack_pool[i]);
        }
    }

    dists.clear();
    dists = iou_distance(r_tracked_stracks, detections, dist_size, dist_size_size);

    matches.clear();
    u_track.clear();
    u_detection.clear();
    linear_assignment(dists, dist_size, dist_size_size, 0.5, matches, u_track, u_detection);

    for (auto & matche : matches) {
        STrack *track = r_tracked_stracks[matche[0]];
        STrack *det = &detections[matche[1]];
        if (track->state == TrackState::Tracked) {
            track->update(*det, _m_frame_id);
            activated_stracks.push_back(*track);
        } else {
            track->re_activate(*det, _m_frame_id, false);
            refind_stracks.push_back(*track);
        }
    }

    for (int i : u_track) {
        STrack *track = r_tracked_stracks[i];
        if (track->state != TrackState::Lost) {
            track->mark_lost();
            lost_stracks.push_back(*track);
        }
    }

    // Deal with unconfirmed tracks, usually tracks with only one beginning frame
    detections.clear();
    detections.assign(detections_cp.begin(), detections_cp.end());

    dists.clear();
    dists = iou_distance(unconfirmed, detections, dist_size, dist_size_size);

    matches.clear();
    std::vector<int> u_unconfirmed;
    u_detection.clear();
    linear_assignment(dists, dist_size, dist_size_size, 0.7, matches, u_unconfirmed, u_detection);

    for (auto & matche : matches) {
        unconfirmed[matche[0]]->update(detections[matche[1]], _m_frame_id);
        activated_stracks.push_back(*unconfirmed[matche[0]]);
    }

    for (int i : u_unconfirmed) {
        STrack *track = unconfirmed[i];
        track->mark_removed();
        removed_stracks.push_back(*track);
    }

    ////////////////// Step 4: Init new stracks //////////////////
    for (int i : u_detection) {
        STrack *track = &detections[i];
        if (track->score < _m_high_thresh) {
            continue;
        }
        track->activate(_m_kalman_filter, _m_frame_id);
        activated_stracks.push_back(*track);
    }

    ////////////////// Step 5: Update state //////////////////
    for (auto & _m_lost_strack : _m_lost_stracks) {
        if (_m_frame_id - _m_lost_strack.end_frame() > _m_max_time_lost) {
            _m_lost_strack.mark_removed();
            removed_stracks.push_back(_m_lost_strack);
        }
    }

    for (auto & _m_tracked_strack : _m_tracked_stracks) {
        if (_m_tracked_strack.state == TrackState::Tracked) {
            tracked_stracks_swap.push_back(_m_tracked_strack);
        }
    }
    _m_tracked_stracks.clear();
    _m_tracked_stracks.assign(tracked_stracks_swap.begin(), tracked_stracks_swap.end());

    _m_tracked_stracks = joint_stracks(_m_tracked_stracks, activated_stracks);
    _m_tracked_stracks = joint_stracks(_m_tracked_stracks, refind_stracks);

    _m_lost_stracks = sub_stracks(_m_lost_stracks, _m_tracked_stracks);
    for (const auto & lost_strack : lost_stracks) {
        _m_lost_stracks.push_back(lost_strack);
    }

    _m_lost_stracks = sub_stracks(_m_lost_stracks, _m_removed_stracks);
    for (const auto & removed_strack : removed_stracks) {
        _m_removed_stracks.push_back(removed_strack);
    }

    remove_duplicate_stracks(resa, resb, _m_tracked_stracks, _m_lost_stracks);

    _m_tracked_stracks.clear();
    _m_tracked_stracks.assign(resa.begin(), resa.end());
    _m_lost_stracks.clear();
    _m_lost_stracks.assign(resb.begin(), resb.end());

    for (auto & _m_tracked_strack : _m_tracked_stracks) {
        if (_m_tracked_strack.is_activated) {
            output_stracks.push_back(_m_tracked_strack);
        }
    }
    return output_stracks;
}

/***
 *
 * @param idx
 * @return
 */
cv::Scalar ByteTracker::Impl::get_color(int idx) {
    idx += 3;
    auto out = cv::Scalar(37 * idx % 255, 17 * idx % 255, 29 * idx % 255);
    return out;
}

/************ Private Func Sets ***********************/

/***
 *
 * @param tlista
 * @param tlistb
 * @return
 */
std::vector<STrack*> ByteTracker::Impl::joint_stracks(std::vector<STrack*> &tlista, std::vector<STrack> &tlistb) {
    std::map<int, int> exists;
    std::vector<STrack*> res;
    for (auto & i : tlista) {
        exists.insert(std::pair<int, int>(i->track_id, 1));
        res.push_back(i);
    }
    for (auto & i : tlistb) {
        int tid = i.track_id;
        if (!exists[tid] || exists.count(tid) == 0) {
            exists[tid] = 1;
            res.push_back(&i);
        }
    }
    return res;
}

/***
 *
 * @param tlista
 * @param tlistb
 * @return
 */
std::vector<STrack> ByteTracker::Impl::joint_stracks(std::vector<STrack> &tlista, std::vector<STrack> &tlistb) {
    std::map<int, int> exists;
    std::vector<STrack> res;
    for (auto & i : tlista) {
        exists.insert(std::pair<int, int>(i.track_id, 1));
        res.push_back(i);
    }
    for (auto & i : tlistb) {
        int tid = i.track_id;
        if (!exists[tid] || exists.count(tid) == 0)
        {
            exists[tid] = 1;
            res.push_back(i);
        }
    }
    return res;
}

/***
 *
 * @param tlista
 * @param tlistb
 * @return
 */
std::vector<STrack> ByteTracker::Impl::sub_stracks(std::vector<STrack> &tlista, std::vector<STrack> &tlistb) {
    std::map<int, STrack> stracks;
    for (auto & i : tlista) {
        stracks.insert(std::pair<int, STrack>(i.track_id, i));
    }
    for (auto & i : tlistb) {
        int tid = i.track_id;
        if (stracks.count(tid) != 0) {
            stracks.erase(tid);
        }
    }

    std::vector<STrack> res;
    std::map<int, STrack>::iterator  it;
    for (it = stracks.begin(); it != stracks.end(); ++it) {
        res.push_back(it->second);
    }

    return res;
}

/***
 *
 * @param resa
 * @param resb
 * @param stracksa
 * @param stracksb
 */
void ByteTracker::Impl::remove_duplicate_stracks(
    std::vector<STrack> &resa, std::vector<STrack> &resb,
    std::vector<STrack> &stracksa, std::vector<STrack> &stracksb) {
    std::vector<std::vector<float> > pdist = iou_distance(stracksa, stracksb);
    std::vector<std::pair<int, int> > pairs;
    for (int i = 0; i < pdist.size(); i++) {
        for (int j = 0; j < pdist[i].size(); j++) {
            if (pdist[i][j] < 0.15) {
                pairs.emplace_back(i, j);
            }
        }
    }

    std::vector<int> dupa;
    std::vector<int> dupb;
    for (auto& pair : pairs) {
        int timep = stracksa[pair.first].frame_id - stracksa[pair.first].start_frame;
        int timeq = stracksb[pair.second].frame_id - stracksb[pair.second].start_frame;
        if (timep > timeq) {
            dupb.push_back(pair.second);
        } else {
            dupa.push_back(pair.first);
        }
    }

    for (int i = 0; i < stracksa.size(); i++) {
        auto iter = find(dupa.begin(), dupa.end(), i);
        if (iter == dupa.end()) {
            resa.push_back(stracksa[i]);
        }
    }

    for (int i = 0; i < stracksb.size(); i++) {
        auto iter = find(dupb.begin(), dupb.end(), i);
        if (iter == dupb.end()) {
            resb.push_back(stracksb[i]);
        }
    }
}

/***
 *
 * @param cost_matrix
 * @param cost_matrix_size
 * @param cost_matrix_size_size
 * @param thresh
 * @param matches
 * @param unmatched_a
 * @param unmatched_b
 */
void ByteTracker::Impl::linear_assignment(
    std::vector<std::vector<float> > &cost_matrix, int cost_matrix_size, int cost_matrix_size_size, float thresh,
    std::vector<std::vector<int> > &matches, std::vector<int> &unmatched_a, std::vector<int> &unmatched_b) {

    if (cost_matrix.empty()) {
        for (int i = 0; i < cost_matrix_size; i++) {
            unmatched_a.push_back(i);
        }
        for (int i = 0; i < cost_matrix_size_size; i++) {
            unmatched_b.push_back(i);
        }
        return;
    }

    std::vector<int> rowsol; std::vector<int> colsol;
    lapjv(cost_matrix, rowsol, colsol, true, thresh);
    for (int i = 0; i < rowsol.size(); i++) {
        if (rowsol[i] >= 0) {
            std::vector<int> match;
            match.push_back(i);
            match.push_back(rowsol[i]);
            matches.push_back(match);
        } else {
            unmatched_a.push_back(i);
        }
    }

    for (int i = 0; i < colsol.size(); i++) {
        if (colsol[i] < 0) {
            unmatched_b.push_back(i);
        }
    }
}

/***
 *
 * @param atlbrs
 * @param btlbrs
 * @return
 */
std::vector<std::vector<float> > ByteTracker::Impl::ious(
    std::vector<std::vector<float> > &atlbrs,
    std::vector<std::vector<float> > &btlbrs) {

    std::vector<std::vector<float> > ious;
    if (atlbrs.size()*btlbrs.size() == 0) {
        return ious;
    }

    ious.resize(atlbrs.size());
    for (auto & iou : ious) {
        iou.resize(btlbrs.size());
    }

    //bbox_ious
    for (int k = 0; k < btlbrs.size(); k++) {
        std::vector<float> ious_tmp;
        float box_area = (btlbrs[k][2] - btlbrs[k][0] + 1)*(btlbrs[k][3] - btlbrs[k][1] + 1);
        for (int n = 0; n < atlbrs.size(); n++) {
            float iw = std::min(atlbrs[n][2], btlbrs[k][2]) - std::max(atlbrs[n][0], btlbrs[k][0]) + 1;
            if (iw > 0) {
                float ih = std::min(atlbrs[n][3], btlbrs[k][3]) - std::max(atlbrs[n][1], btlbrs[k][1]) + 1;
                if(ih > 0) {
                    float ua = (atlbrs[n][2] - atlbrs[n][0] + 1)*(atlbrs[n][3] - atlbrs[n][1] + 1) + box_area - iw * ih;
                    ious[n][k] = iw * ih / ua;
                } else {
                    ious[n][k] = 0.0;
                }
            } else {
                ious[n][k] = 0.0;
            }
        }
    }

    return ious;
}

/***
 *
 * @param atracks
 * @param btracks
 * @param dist_size
 * @param dist_size_size
 * @return
 */
std::vector<std::vector<float> > ByteTracker::Impl::iou_distance(
    std::vector<STrack*> &atracks, std::vector<STrack> &btracks, int &dist_size, int &dist_size_size) {

    std::vector<std::vector<float> > cost_matrix;
    if (atracks.size() * btracks.size() == 0) {
        dist_size = static_cast<int>(atracks.size());
        dist_size_size = static_cast<int>(btracks.size());
        return cost_matrix;
    }
    std::vector<std::vector<float> > atlbrs, btlbrs;
    for (auto & atrack : atracks) {
        atlbrs.push_back(atrack->tlbr);
    }
    for (auto & btrack : btracks) {
        btlbrs.push_back(btrack.tlbr);
    }

    dist_size = static_cast<int>(atracks.size());
    dist_size_size = static_cast<int>(btracks.size());

    std::vector<std::vector<float> > _ious = ious(atlbrs, btlbrs);
    for (auto & i : _ious) {
        std::vector<float> _iou;
        for (float j : i) {
            _iou.push_back(1.0f - j);
        }
        cost_matrix.push_back(_iou);
    }

    return cost_matrix;
}

/***
 *
 * @param atracks
 * @param btracks
 * @return
 */
std::vector<std::vector<float> > ByteTracker::Impl::iou_distance(std::vector<STrack> &atracks, std::vector<STrack> &btracks) {
    std::vector<std::vector<float> > atlbrs;
    std::vector<std::vector<float> > btlbrs;
    for (auto & atrack : atracks) {
        atlbrs.push_back(atrack.tlbr);
    }
    for (auto & btrack : btracks) {
        btlbrs.push_back(btrack.tlbr);
    }

    std::vector<std::vector<float> > _ious = ious(atlbrs, btlbrs);
    std::vector<std::vector<float> > cost_matrix;
    for (auto & i : _ious) {
        std::vector<float> _iou;
        for (float j : i) {
            _iou.push_back(1 - j);
        }
        cost_matrix.push_back(_iou);
    }

    return cost_matrix;
}

/***
 *
 * @param cost
 * @param rowsol
 * @param colsol
 * @param extend_cost
 * @param cost_limit
 * @param return_cost
 * @return
 */
double ByteTracker::Impl::lapjv(
    const std::vector<std::vector<float> > &cost,
    std::vector<int> &rowsol, std::vector<int> &colsol,
    bool extend_cost, float cost_limit, bool return_cost) {

    std::vector<std::vector<float> > cost_c;
    cost_c.assign(cost.begin(), cost.end());

    std::vector<std::vector<float> > cost_c_extended;

    auto n_rows = cost.size();
    auto n_cols = cost[0].size();
    rowsol.resize(n_rows);
    colsol.resize(n_cols);

    size_t n;
    if (n_rows == n_cols) {
        n = n_rows;
    } else {
        if (!extend_cost) {
            LOG(ERROR) << "set extend_cost=True";
            return 0.0;
        }
    }

    if (extend_cost || cost_limit < static_cast<float>(LONG_MAX)) {
        n = n_rows + n_cols;
        cost_c_extended.resize(n);
        for (auto & i : cost_c_extended) {
            i.resize(n);
        }

        if (cost_limit < static_cast<float>(LONG_MAX)) {
            for (auto & i : cost_c_extended) {
                for (float & j : i) {
                    j = static_cast<float>(cost_limit / 2.0f);
                }
            }
        } else {
            float cost_max = -1.0f;
            for (auto & i : cost_c) {
                for (float j : i) {
                    if (j > cost_max) {
                        cost_max = j;
                    }
                }
            }
            for (auto & i : cost_c_extended) {
                for (float & j : i) {
                    j = cost_max + 1.0f;
                }
            }
        }

        for (size_t i = n_rows; i < cost_c_extended.size(); i++) {
            for (size_t j = n_cols; j < cost_c_extended[i].size(); j++) {
                cost_c_extended[i][j] = 0;
            }
        }
        for (int i = 0; i < n_rows; i++) {
            for (int j = 0; j < n_cols; j++) {
                cost_c_extended[i][j] = cost_c[i][j];
            }
        }

        cost_c.clear();
        cost_c.assign(cost_c_extended.begin(), cost_c_extended.end());
    }

    double **cost_ptr;
    cost_ptr = new double *[sizeof(double *) * n];
    for (int i = 0; i < n; i++) {
        cost_ptr[i] = new double[sizeof(double) * n];
    }
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            cost_ptr[i][j] = cost_c[i][j];
        }
    }

    int* x_c = new int[sizeof(int) * n];
    int *y_c = new int[sizeof(int) * n];

    int ret = lapjv_internal(n, cost_ptr, x_c, y_c);
    if (ret != 0) {
        LOG(ERROR) << "Calculate Wrong!";
        return 0.0;
    }

    double opt = 0.0;

    if (n != n_rows) {
        for (int i = 0; i < n; i++) {
            if (x_c[i] >= n_cols) {
                x_c[i] = -1;
            }
            if (y_c[i] >= n_rows) {
                y_c[i] = -1;
            }
        }
        for (int i = 0; i < n_rows; i++) {
            rowsol[i] = x_c[i];
        }
        for (int i = 0; i < n_cols; i++) {
            colsol[i] = y_c[i];
        }

        if (return_cost) {
            for (int i = 0; i < rowsol.size(); i++) {
                if (rowsol[i] != -1) {
                    opt += cost_ptr[i][rowsol[i]];
                }
            }
        }
    } else if (return_cost) {
        for (int i = 0; i < rowsol.size(); i++) {
            opt += cost_ptr[i][rowsol[i]];
        }
    }

    for (int i = 0; i < n; i++) {
        delete[]cost_ptr[i];
    }
    delete[]cost_ptr;
    delete[]x_c;
    delete[]y_c;

    return opt;
}


/************* Export Function Sets *************/

/***
 *
 */
ByteTracker::ByteTracker() {
    _m_pimpl = std::make_unique<Impl>();
}

/***
 *
 */
ByteTracker::~ByteTracker() = default;

/***
 *
 * @param cfg
 * @return
 */
StatusCode ByteTracker::init(const decltype(toml::parse("")) &cfg) {
   return _m_pimpl-> init(cfg);
}

/***
 * 
 * @param objects 
 * @return 
 */
std::vector<STrack> ByteTracker::update(const std::vector<Object>& objects) {
    return _m_pimpl->update(objects);
}

/***
 * 
 * @param idx 
 * @return 
 */
cv::Scalar ByteTracker::get_color(int idx) {
    return _m_pimpl->get_color(idx);
}

/***
 *
 * @return
 */
bool ByteTracker::is_successfully_initialized() const {
    return _m_pimpl->is_successfully_initialized();
}

}
}
}
}