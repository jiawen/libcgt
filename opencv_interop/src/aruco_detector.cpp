#include "aruco_detector.h"

#include <opencv2/calib3d.hpp>

#include <core/common/ArrayUtils.h>
#include <opencv_interop/ArrayUtils.h>
#include <opencv_interop/VecmathUtils.h>

using libcgt::core::arrayutils::copy;
using libcgt::core::arrayutils::flipY;
using libcgt::core::vecmath::EuclideanTransform;
using libcgt::opencv_interop::cvMatAsArray2DView;
using libcgt::opencv_interop::fromCV3x3;

bool readDetectorParameters(const std::string& filename, cv::aruco::DetectorParameters& params) {
    cv::FileStorage fs(filename, cv::FileStorage::READ);
    if (!fs.isOpened()) {
        return false;
    }
    fs["adaptiveThreshWinSizeMin"] >> params.adaptiveThreshWinSizeMin;
    fs["adaptiveThreshWinSizeMax"] >> params.adaptiveThreshWinSizeMax;
    fs["adaptiveThreshWinSizeStep"] >> params.adaptiveThreshWinSizeStep;
    fs["adaptiveThreshConstant"] >> params.adaptiveThreshConstant;
    fs["minMarkerPerimeterRate"] >> params.minMarkerPerimeterRate;
    fs["maxMarkerPerimeterRate"] >> params.maxMarkerPerimeterRate;
    fs["polygonalApproxAccuracyRate"] >> params.polygonalApproxAccuracyRate;
    fs["minCornerDistanceRate"] >> params.minCornerDistanceRate;
    fs["minDistanceToBorder"] >> params.minDistanceToBorder;
    fs["minMarkerDistanceRate"] >> params.minMarkerDistanceRate;
    fs["doCornerRefinement"] >> params.doCornerRefinement;
    fs["cornerRefinementWinSize"] >> params.cornerRefinementWinSize;
    fs["cornerRefinementMaxIterations"] >> params.cornerRefinementMaxIterations;
    fs["cornerRefinementMinAccuracy"] >> params.cornerRefinementMinAccuracy;
    fs["markerBorderBits"] >> params.markerBorderBits;
    fs["perspectiveRemovePixelPerCell"] >> params.perspectiveRemovePixelPerCell;
    fs["perspectiveRemoveIgnoredMarginPerCell"] >> params.perspectiveRemoveIgnoredMarginPerCell;
    fs["maxErroneousBitsInBorderRate"] >> params.maxErroneousBitsInBorderRate;
    fs["minOtsuStdDev"] >> params.minOtsuStdDev;
    fs["errorCorrectionRate"] >> params.errorCorrectionRate;
    return true;
}

// 4x4_50: 4x4 bits (4 bits in each direction), 50 markers.
ArucoDetector::ArucoDetector(const std::string& detector_params_filename) :
    dictionary_(cv::aruco::getPredefinedDictionary(cv::aruco::DICT_4X4_50)),
    markers_x_(2),
    markers_y_(2),
    marker_length_(0.064f),
    marker_separation_(0.064f),
    rot_x_pi_(Matrix4f::rotateX(static_cast<float>(M_PI))) {

    // Detailed documentation here:
    // http://docs.opencv.org/3.1.0/d9/d6d/tutorial_table_of_content_aruco.html#gsc.tab=0
    // Predefined dictionaries have "names" which are actually enum instances
    // of the form: DICT_<num_bits_x>X<num_bits_y>_<num_markers>
    // DICT_4X4_50 means: each marker is 4 bits by 4 bits. The dictionary
    // consists of 50 markers.

    board_ = cv::aruco::GridBoard::create(
        markers_x_, markers_y_,
        marker_length_, marker_separation_,
        dictionary_);

    axis_length_ = 0.5f * (marker_separation_ +
        (float)std::min(markers_x_, markers_y_) *
        (marker_length_ + marker_separation_));

    // TODO: bake this as a static default option. And let them pass one in.
    bool read_ok = readDetectorParameters(detector_params_filename, detector_params_);
}

cv::Mat ArucoDetector::BoardImage(int width, int height) {
    cv::Mat board_image(height, width, CV_8U);
    board_.draw(board_image.size(), board_image);
    return board_image;
}

Array2D<uint8_t> ArucoDetector::GLBoardImage(int width, int height)
{
    Array2D<uint8_t> gl_board_image({width, height});
    cv::Mat cv_board_image = BoardImage(width, height);
    Array2DView<const uint8_t> src = flipY(
      cvMatAsArray2DView<uint8_t>(cv_board_image));
    copy(src, gl_board_image.writeView());
    return gl_board_image;
}

ArucoDetector::DetectionResult ArucoDetector::Detect(cv::Mat image) const {
    ArucoDetector::DetectionResult result;
    cv::aruco::detectMarkers(image, dictionary_, result.corners, result.ids, detector_params_, result.rejected);
    return result;
}

void ArucoDetector::Refine(cv::Mat image, cv::Mat camera_intrinsics,
    const std::vector<float>& camera_dist_coeffs,
    ArucoDetector::DetectionResult* result) const {
    cv::aruco::refineDetectedMarkers(image, board_,
        result->corners, result->ids, result->rejected,
        camera_intrinsics, camera_dist_coeffs);
}

ArucoDetector::PoseEstimate ArucoDetector::EstimatePose(
    const DetectionResult& detection_result,
    cv::Mat camera_intrinsics,
    const std::vector<float>& camera_dist_coeffs) const {
    ArucoDetector::PoseEstimate pose;
    pose.valid = false;

    int num_markers_detected = 0;
    if (detection_result.ids.size() > 0) {
      num_markers_detected =
          cv::aruco::estimatePoseBoard(
              detection_result.corners, detection_result.ids,
              board_,
              camera_intrinsics, camera_dist_coeffs,
              pose.camera_from_board_rotation,
              pose.camera_from_board_translation);
    }

    if (num_markers_detected > 0) {
      pose.valid = true;

      // Convert to GL.
      cv::Mat cv_rotation_matrix;
      cv::Rodrigues(pose.camera_from_board_rotation, cv_rotation_matrix);
      Matrix4f rt;
      rt.setSubmatrix3x3(0, 0, fromCV3x3(cv_rotation_matrix));
      rt(0, 3) = static_cast<float>(pose.camera_from_board_translation(0));
      rt(1, 3) = static_cast<float>(pose.camera_from_board_translation(1));
      rt(2, 3) = static_cast<float>(pose.camera_from_board_translation(2));
      rt(3, 3) = 1.0f;

      // We only rotation by pi on one side and don't conjugate on both sides
      // because aruco's coordinate system is such that:
      // the board is in GL conventions, but the camera is in CV conventions.
      // Therefore, their computed transformation is cv_camera <-- gl_board.
      pose.camera_from_board = EuclideanTransform::fromMatrix(rot_x_pi_ * rt);
    }
    return pose;
}

// static
void ArucoDetector::VisualizeDetections(
    const ArucoDetector::DetectionResult& detection_result,
    bool show_rejected, cv::Mat* vis_image) {
    if (detection_result.ids.size() > 0) {
        cv::aruco::drawDetectedMarkers(*vis_image,
            detection_result.corners, detection_result.ids);
    }
    if (show_rejected && detection_result.rejected.size() > 0) {
        cv::aruco::drawDetectedMarkers(*vis_image, detection_result.rejected,
            cv::noArray(), cv::Scalar(100, 0, 255));
    }
}

void ArucoDetector::VisualizePoseEstimate(
    const ArucoDetector::PoseEstimate& estimate_result,
    cv::Mat camera_intrinsics,
    const std::vector<float>& camera_dist_coeffs,
    cv::Mat* vis_image) {
    if (estimate_result.valid) {
        cv::aruco::drawAxis(*vis_image,
            camera_intrinsics, camera_dist_coeffs,
            estimate_result.camera_from_board_rotation,
            estimate_result.camera_from_board_translation,
            axis_length_);
    }
}
