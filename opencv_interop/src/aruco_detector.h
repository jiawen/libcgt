#ifndef ARUCO_DETECTOR_H
#define ARUCO_DETECTOR_H

#define _USE_MATH_DEFINES
#include <cmath>

#include <opencv2/aruco.hpp>

#include <core/common/Array2D.h>
#include <core/vecmath/EuclideanTransform.h>

class ArucoDetector {
public:

    struct DetectionResult {
        std::vector< int > ids;
        std::vector< std::vector< cv::Point2f > > corners;
        std::vector< std::vector< cv::Point2f > > rejected;
    };

    // Pose estimate.
    //
    // If valid = true:
    //
    // camera_from_board will be a rigid transform mapping camera <-- board.
    //
    // camera_from_board_rotation and camera_from_board_translation represents
    // the same transformation but as as rotation and translation vectors.
    //
    // Coordinate conventions:
    //
    // Aruco stores the board's coordinates using OpenGL conventions.
    //
    // camera_from_board is a fully-OpenGL transformation, mapping a GL board
    // to a GL camera (y up, z out of screen for both).
    //
    // camera_from_board_rotation and camera_from_board_translation maps from
    // a GL board (y up, z out of screen) to a CV camera
    // (y down, z into screen).
    //
    // Therefore, there is a 180 degree rotation about the x axis between the
    // two representations.
    struct PoseEstimate {
        bool valid = false;

        libcgt::core::vecmath::EuclideanTransform camera_from_board;

        cv::Vec3d camera_from_board_rotation;
        cv::Vec3d camera_from_board_translation;
    };

    // TODO(jiawen): also pass in a cv::aruco::DetectorParameters object.
    ArucoDetector(const std::string& detector_params_filename);

    // Generate the board image. width and height are in pixels.
    // Returns a cv::Mat(height, width, CV_8U).
    cv::Mat BoardImage(int width, int height);

    // Generate the board image, flipped up/down as a Array2D<uint8_t> so that
    // it is suitable for OpenGL.
    // "width" and "height" are in pixels.
    Array2D<uint8_t> GLBoardImage(int width, int height);

    DetectionResult Detect(cv::Mat image) const;

    // Refine the markers returned by Detect().
    // image must be the same as the one used in Detect().
    // result is modified in place.
    void Refine(cv::Mat image, cv::Mat camera_intrinsics, const std::vector<float>& camera_dist_coeffs,
        DetectionResult* result) const;

    // Estimate the camera pose from detected and optionally refined markers.
    // The result may be invalid if there are not enough corners.
    PoseEstimate EstimatePose(const DetectionResult& detection_result,
        cv::Mat camera_intrinsics,
        const std::vector<float>& camera_dist_coeffs) const;

    static void VisualizeDetections(const DetectionResult& detection_result,
        bool show_rejected, cv::Mat* vis_image);

    void VisualizePoseEstimate(const PoseEstimate& estimate_result,
        cv::Mat camera_intrinsics,
        const std::vector<float>& camera_dist_coeffs,
        cv::Mat* vis_image);

private:

    cv::aruco::Dictionary dictionary_;
    cv::aruco::GridBoard board_;
    cv::aruco::DetectorParameters detector_params_;

    // Number of markers in x and y.
    int markers_x_;
    int markers_y_;

    // Side length of one marker, in meters.
    float marker_length_;

    // Length of the empty space between markers:
    // the end of one marker and the start of the next.
    float marker_separation_;

    // Computed just for visualizing the axis.
    float axis_length_;

    // Rotation matrix about the x-axis by pi radians.
    const Matrix4f rot_x_pi_;
};

#endif // ARUCO_DETECTOR_H
