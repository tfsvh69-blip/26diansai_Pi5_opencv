// USB摄像头 C++ 版 - 固定曝光/增益/白平衡 + 可切换去畸变
// 对应 Python 的 ready_code/v1.2.py，行为一致，做 C++ 工程骨架。
//
// 编译: 见同目录 build.sh 或 CMakeLists.txt
// 运行: ./undistort_view
//
// 说明:
//   - 固定参数(曝光/增益/白平衡/亮度)与 Python 端 camera_common.py 保持一致。
//     那边是唯一事实来源，改参数请两边同步，或重新跑 tune_camera.py。
//   - 用 v4l2-ctl 命令下发 V4L2 控制项(和 Python 端同理，OpenCV 属性映射不可靠)。
//   - 标定数据从 camera_calib.yaml 读取(由 Python 端 npz 导出，跨 OpenCV 版本通用)。
//   - 去畸变用预计算映射 + cv::remap()，比每帧 cv::undistort() 快。
//
// 按键:
//   u        : 切换 原始 / 去畸变 画面
//   q / ESC  : 退出

#include <opencv2/opencv.hpp>
#include <cstdlib>
#include <string>
#include <iostream>

// 与 camera_common.py 一致的固定参数（2026-07-17 现场调出）
static const char* DEVICE = "/dev/video0";
static const int   FRAME_W = 640, FRAME_H = 480, FRAME_FPS = 30;
static const int   EXPOSURE_TIME_ABSOLUTE = 13;
static const int   GAIN = 2;
static const int   WHITE_BALANCE_TEMPERATURE = 4532;
static const int   BRIGHTNESS = -11;

static void set_ctrl(const std::string& name, int value) {
    std::string cmd = "v4l2-ctl -d " + std::string(DEVICE) +
                      " --set-ctrl=" + name + "=" + std::to_string(value) +
                      " >/dev/null 2>&1";
    if (std::system(cmd.c_str()) != 0) {
        // v4l2-ctl 下发失败就忽略，和 Python 端一样宽松处理(不因单个控制项失败中断)
    }
}

static void apply_fixed_params() {
    set_ctrl("auto_exposure", 1);              // 1 = Manual Mode
    set_ctrl("white_balance_automatic", 0);    // 关闭自动白平衡
    set_ctrl("exposure_time_absolute", EXPOSURE_TIME_ABSOLUTE);
    set_ctrl("gain", GAIN);
    set_ctrl("white_balance_temperature", WHITE_BALANCE_TEMPERATURE);
    set_ctrl("brightness", BRIGHTNESS);
}

int main() {
    apply_fixed_params();

    cv::VideoCapture cap(0, cv::CAP_V4L2);
    if (!cap.isOpened()) {
        std::cerr << "❌ 无法打开摄像头 " << DEVICE << std::endl;
        return 1;
    }
    cap.set(cv::CAP_PROP_FOURCC, cv::VideoWriter::fourcc('M', 'J', 'P', 'G'));
    cap.set(cv::CAP_PROP_FRAME_WIDTH, FRAME_W);
    cap.set(cv::CAP_PROP_FRAME_HEIGHT, FRAME_H);
    cap.set(cv::CAP_PROP_FPS, FRAME_FPS);

    int w = (int)cap.get(cv::CAP_PROP_FRAME_WIDTH);
    int h = (int)cap.get(cv::CAP_PROP_FRAME_HEIGHT);
    std::cout << "✅ 摄像头已打开: " << w << "x" << h
              << " (MJPG, 固定曝光/增益/白平衡)" << std::endl;

    // 载入标定并预计算去畸变映射
    cv::Mat camera_matrix, dist_coeffs, map1, map2;
    bool has_calib = false;
    cv::FileStorage fs("camera_calib.yaml", cv::FileStorage::READ);
    if (fs.isOpened()) {
        fs["camera_matrix"] >> camera_matrix;
        fs["dist_coeffs"] >> dist_coeffs;
        double rms = 0.0; fs["rms"] >> rms;
        fs.release();
        // alpha=0：裁掉去畸变黑边，画面略放大但干净
        cv::Mat newK = cv::getOptimalNewCameraMatrix(
            camera_matrix, dist_coeffs, cv::Size(w, h), 0.0, cv::Size(w, h));
        cv::initUndistortRectifyMap(
            camera_matrix, dist_coeffs, cv::Mat(), newK,
            cv::Size(w, h), CV_16SC2, map1, map2);
        has_calib = true;
        std::cout << "   已载入标定 (RMS=" << rms << "px)，按 [u] 切换 原始/去畸变。"
                  << std::endl;
    } else {
        std::cout << "   ⚠️ 未找到 camera_calib.yaml，只能显示原始画面。" << std::endl;
    }

    bool undistort_on = has_calib;  // 有标定就默认去畸变
    std::cout << "   [u] 切换去畸变   [q/ESC] 退出" << std::endl;

    const std::string win = "USB Camera (C++)";
    cv::namedWindow(win, cv::WINDOW_NORMAL);
    cv::resizeWindow(win, w, h);

    cv::Mat frame, view;
    while (true) {
        if (!cap.read(frame) || frame.empty()) {
            std::cerr << "❌ 读取帧失败" << std::endl;
            break;
        }

        std::string label;
        if (undistort_on && has_calib) {
            cv::remap(frame, view, map1, map2, cv::INTER_LINEAR);
            label = "UNDISTORTED (u to toggle)";
        } else {
            view = frame;
            label = "ORIGINAL (u to toggle)";
        }

        cv::putText(view, label, cv::Point(10, 25),
                    cv::FONT_HERSHEY_SIMPLEX, 0.6, cv::Scalar(0, 255, 0), 2);
        cv::imshow(win, view);

        int key = cv::waitKey(1) & 0xFF;
        if (key == 'q' || key == 27) break;
        if (key == 'u') {
            if (!has_calib) {
                std::cout << "   还没标定，无法切换。" << std::endl;
            } else {
                undistort_on = !undistort_on;
                std::cout << "   切换为: "
                          << (undistort_on ? "去畸变" : "原始") << " 画面" << std::endl;
            }
        }
    }

    cap.release();
    cv::destroyAllWindows();
    std::cout << "已退出" << std::endl;
    return 0;
}
