/************************************************
 * Copyright MaybeShewill-CV. All Rights Reserved.
 * Author: MaybeShewill-CV
 * File: byte_tracker.h
 * Date: 24-5-30
 ************************************************/

#ifndef MORTRED_MODEL_SERVER_BYTE_TRACKER_H
#define MORTRED_MODEL_SERVER_BYTE_TRACKER_H

#include <vector>

#include <opencv2/opencv.hpp>
#include "toml/parser.hpp"

#include "common/status_code.h"
#include "strack.h"
#include "data_type.h"

namespace jinq {
namespace models {
namespace mot {
namespace byte_tracker {

class ByteTracker {
  public:

    /***
    * constructor
    * @param config
     */
    ByteTracker();

    /***
     *
     */
    ~ByteTracker();

    /***
    * constructor
    * @param transformer
     */
    ByteTracker(const ByteTracker& transformer) = delete;

    /***
     * constructor
     * @param transformer
     * @return
     */
    ByteTracker& operator=(const ByteTracker& transformer) = delete;

    /***
     *
     * @param toml
     * @return
     */
    jinq::common::StatusCode init(const decltype(toml::parse(""))& cfg);

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
    cv::Scalar get_color(int idx);

    /***
     * if bytetracker successfully initialized
     * @return
     */
    bool is_successfully_initialized() const;

  private:
    class Impl;
    std::unique_ptr<Impl> _m_pimpl;
};

}
}
}
}

#endif // MORTRED_MODEL_SERVER_BYTE_TRACKER_H
