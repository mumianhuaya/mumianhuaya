#include <iostream>
#include <fstream>
#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/videoio.hpp>
#include <opencv2/video.hpp>
#include <opencv2/dnn.hpp>
#include <vector>
#include <cmath>

using namespace cv;
using namespace std;
using namespace cv::dnn;

// 定义 flow_t 结构
struct flow_t {
    Point2f pos;
    float flow_x;
    float flow_y;
};

// 定义 get_size_divergence 函数，计算光流场的大小发散
float get_size_divergence(const flow_t* flow_vectors, int count, int n_samples) {
    float divs_sum = 0.f;
    uint32_t used_samples = 0;

    if (count < 2) {
        return 0.f;
    }

    if (n_samples == 1) {
        // 遍历所有可能的线段：
        for (int i = 0; i < count; i++) {
            for (int j = i + 1; j < count; j++) {
                float dx = flow_vectors[i].pos.x - flow_vectors[j].pos.x;
                float dy = flow_vectors[i].pos.y - flow_vectors[j].pos.y;
                float distance_1 = sqrtf(dx * dx + dy * dy);

                if (distance_1 < 1E-5) {
                    continue;
                }

                float dx2 = flow_vectors[i].pos.x + flow_vectors[i].flow_x - flow_vectors[j].pos.x - flow_vectors[j].flow_x;
                float dy2 = flow_vectors[i].pos.y + flow_vectors[i].flow_y - flow_vectors[j].pos.y - flow_vectors[j].flow_y;
                float distance_2 = sqrtf(dx2 * dx2 + dy2 * dy2);

                divs_sum += (distance_2 - distance_1) / distance_1;
                used_samples++;
            }
        }
    } else {
        // 随机抽样：
        for (int sample = 0; sample < n_samples; sample++) {
            int i = rand() % count;
            int j = rand() % count;
            while (i == j) {
                j = rand() % count;
            }

            float dx = flow_vectors[i].pos.x - flow_vectors[j].pos.x;
            float dy = flow_vectors[i].pos.y - flow_vectors[j].pos.y;
            float distance_1 = sqrtf(dx * dx + dy * dy);

            if (distance_1 < 1E-5) {
                continue;
            }

            float dx2 = flow_vectors[i].pos.x + flow_vectors[i].flow_x - flow_vectors[j].pos.x - flow_vectors[j].flow_x;
            float dy2 = flow_vectors[i].pos.y + flow_vectors[i].flow_y - flow_vectors[j].pos.y - flow_vectors[j].flow_y;
            float distance_2 = sqrtf(dx2 * dx2 + dy2 * dy2);

            divs_sum += (distance_2 - distance_1) / distance_1;
            used_samples++;
        }
    }

    if (used_samples < 1) {
        return 0.f;
    }

    // 返回计算得到的平均大小发散
    return divs_sum / used_samples;
}

// 加载 YOLO 模型
Net load_yolo(vector<string> &classes, vector<Scalar> &colors, vector<string> &output_layers) {
    Net net = readNet("yolov3-tiny.weights", "yolov3-tiny.cfg");
    ifstream ifs("coco.names");
    string line;
    while (getline(ifs, line)) {
        classes.push_back(line);
    }

    for (const auto& layerName : net.getUnconnectedOutLayersNames()) {
        output_layers.push_back(layerName);
    }
    colors = vector<Scalar>(classes.size());
    for (size_t i = 0; i < classes.size(); ++i) {
        colors[i] = Scalar(rand() % 256, rand() % 256, rand() % 256);
    }

    return net;
}

// 检测对象
void detect_objects(Mat &img, Net &net, const vector<string> &outputLayers, vector<Rect> &boxes, vector<float> &confs, vector<int> &class_ids, const float confThreshold = 0.5, const float nmsThreshold = 0.4) {
    Mat blob;
    blobFromImage(img, blob, 1 / 255.0, Size(320, 320), Scalar(0, 0, 0), true, false);
    net.setInput(blob);
    vector<Mat> outs;
    net.forward(outs, outputLayers);

    for (const auto& out : outs) {
        const auto* data = (float*)out.data;
        for (int j = 0; j < out.rows; ++j, data += out.cols) {
            Mat scores = out.row(j).colRange(5, out.cols);
            Point classIdPoint;
            double confidence;
            minMaxLoc(scores, 0, &confidence, 0, &classIdPoint);
            if (confidence > confThreshold) {
                int centerX = (int)(data[0] * img.cols);
                int centerY = (int)(data[1] * img.rows);
                int width = (int)(data[2] * img.cols);
                int height = (int)(data[3] * img.rows);
                int left = centerX - width / 2;
                int top = centerY - height / 2;

                class_ids.push_back(classIdPoint.x);
                confs.push_back((float)confidence);
                boxes.push_back(Rect(left, top, width, height));
            }
        }
    }

    // 应用非最大抑制
    vector<int> indices;
    NMSBoxes(boxes, confs, confThreshold, nmsThreshold, indices);
    vector<Rect> nms_boxes;
    vector<float> nms_confs;
    vector<int> nms_class_ids;
    for (int idx : indices) {
        nms_boxes.push_back(boxes[idx]);
        nms_confs.push_back(confs[idx]);
        nms_class_ids.push_back(class_ids[idx]);
    }
    boxes = nms_boxes;
    confs = nms_confs;
    class_ids = nms_class_ids;
}

// 绘制标签
void draw_labels(Mat &img, const vector<Rect> &boxes, const vector<float> &confs, const vector<int> &class_ids, const vector<string> &classes, const vector<Scalar> &colors) {
    for (size_t i = 0; i < boxes.size(); ++i) {
        rectangle(img, boxes[i], colors[class_ids[i]], 2);
        string label = format("%s: %.2f", classes[class_ids[i]].c_str(), confs[i]);
        int baseLine;
        Size labelSize = getTextSize(label, FONT_HERSHEY_SIMPLEX, 0.5, 1, &baseLine);
        rectangle(img, Point(boxes[i].x, boxes[i].y - labelSize.height),
                  Point(boxes[i].x + labelSize.width, boxes[i].y + baseLine), colors[class_ids[i]], FILLED);
        putText(img, label, Point(boxes[i].x, boxes[i].y), FONT_HERSHEY_SIMPLEX, 0.5, Scalar(0, 0, 0), 1);
    }
}

int main() {
    VideoCapture capture(0); // 打开默认摄像头

    if (!capture.isOpened()) {
        cerr << "Unable to open camera!" << endl;
        return -1;
    }

    Mat old_frame, old_gray;
    vector<Point2f> p0, p1;
    vector<string> classes;
    vector<Scalar> colors;
    vector<string> output_layers;
    Net yolo_net = load_yolo(classes, colors, output_layers);
    float divergence_threshold = 1.0;
    vector<float> divergence_values;
    int num_frames_to_average = 5;

    // 在第一帧中检测角点
    capture >> old_frame;
    if (old_frame.empty()) {
        cerr << "No frame captured!" << endl;
        return -1;
    }

    // 高斯滤波处理图像，以降低噪声
    GaussianBlur(old_frame, old_frame, Size(5, 5), 0);

    cvtColor(old_frame, old_gray, COLOR_BGR2GRAY);
    goodFeaturesToTrack(old_gray, p0, 80, 0.15, 7, Mat(), 7, false, 0.04);

    while (true) {
        Mat frame, frame_gray;
        capture >> frame;
        if (frame.empty())
            break;

        // 高斯滤波处理图像，以降低噪声
        GaussianBlur(frame, frame, Size(5, 5), 0);
        cvtColor(frame, frame_gray, COLOR_BGR2GRAY);

        // 计算光流
        vector<uchar> status;
        vector<float> err;
        TermCriteria criteria = TermCriteria((TermCriteria::COUNT) + (TermCriteria::EPS), 10, 0.03);
        calcOpticalFlowPyrLK(old_gray, frame_gray, p0, p1, status, err, Size(15, 15), 2, criteria);

        // 转换 p0 和 p1 到 flow_t 数组
        vector<flow_t> flow_array;
        for (size_t i = 0; i < p0.size(); ++i) {
            if (status[i]) {
                flow_t flow;
                flow.pos = p0[i];
                flow.flow_x = p1[i].x - p0[i].x;
                flow.flow_y = p1[i].y - p0[i].y;
                flow_array.push_back(flow);
            }
        }

        // 调用函数计算光流场的大小发散
        float divergence = get_size_divergence(flow_array.data(), static_cast<int>(flow_array.size()), 1); // n_samples 设为 1，计算所有线段

        // 将当前帧的divergence值添加到数组中
        divergence_values.push_back(divergence);
        if (divergence_values.size() > num_frames_to_average) {
            divergence_values.erase(divergence_values.begin()); // 移除最早的值
        }

        // 计算平均divergence
        float average_divergence = 0.f;
        for (float d : divergence_values) {
            average_divergence += d;
        }
        average_divergence /= divergence_values.size();

        // 输出特征点的数量和光流向量的数量
        cout << "Number of feature points: " << p0.size() << endl;
        cout << "Number of optical flow vectors: " << flow_array.size() << endl;
        cout << "Divergence: " << divergence << endl;

        if (average_divergence > divergence_threshold) {
            // 使用 YOLOv3 进行目标检测
            vector<Rect> boxes;
            vector<float> confs;
            vector<int> class_ids;
            detect_objects(frame, yolo_net, output_layers, boxes, confs, class_ids);
            draw_labels(frame, boxes, confs, class_ids, classes, colors);
        }

        // 显示当前帧
        imshow("Frame", frame);

        // 更新旧帧和角点
        old_gray = frame_gray.clone();
        p0 = p1;

        // 检查用户输入
        int key = waitKey(30);
        if (key == 'q' || key == 27)
            break;

        // 确保只保留有效的特征点
        vector<Point2f> p0_new;
        for (size_t i = 0; i < status.size(); i++) {
            if (status[i]) {
                p0_new.push_back(p1[i]);
            }
        }
        p0 = p0_new;
    }

    capture.release();
    destroyAllWindows();

    return 0;
}
