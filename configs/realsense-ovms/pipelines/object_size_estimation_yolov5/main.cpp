//*****************************************************************************
// Copyright 2023 Intel Corporation
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
//*****************************************************************************
#include <algorithm>
#include <array>
#include <chrono>
#include <cstring>
#include <iostream>
#include <numeric>
#include <sstream>
#include <thread>
#include <vector>
#include <iomanip>
#include <regex>
#include <atomic>
#include <mutex>
#include <condition_variable>

#include <signal.h>
#include <stdio.h>

// Realsense
#include <librealsense2/rs.hpp> // Include RealSense Cross Platform API
#include "example.hpp"
#define _USE_MATH_DEFINES
#include <math.h>
#include <queue>
#include <unordered_set>
#include <map>


// Utilized for OpenCV based Rendering only
#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/videoio.hpp>

// Utilized for infernece output layer post-processing
#include <cmath>

#include "ovms.h"  // NOLINT

using namespace std;
using namespace cv;
using pixel = std::pair<int, int>;

int _color_frame_width;
int _color_frame__height;
int _color_frame_rate;
int _depth_frame_width;
int _depth_frame_height;
int _depth_frame_rate;
int _frame_align;
int _enable_filters;
bool _disable_realsense_gpu_acceleration;

std::mutex _mtx;
std::mutex _infMtx;
std::mutex _drawingMtx;
std::condition_variable _cvAllDecodersInitd;
bool _allDecodersInitd = false;

typedef struct DetectedResult {
	int frameId;
	float x;
	float y;
	float width;
	float height;
	float confidence;
	int classId;
    int estimatedDepth;
    int actual_width;
    int actual_height;
    int point_x0_0_actual_measured;
    int point_y0_0_actual_measured;
    int point_x1_0_actual_measured;
    int point_y1_0_actual_measured;
    int point_x0_1_actual_measured;
    int point_y0_1_actual_measured;
    int point_x1_1_actual_measured;
    int point_y1_1_actual_measured;
	char classText[1024];
} DetectedResult;

OVMS_Server* _srv;
OVMS_ServerSettings* _serverSettings = 0;
OVMS_ModelsSettings* _modelsSettings = 0;
int _server_grpc_port;
int _server_http_port;

std::string _videoStreamPipeline;
int _detectorModel = 0;
bool _render = 0;
bool _use_onevpl = 0;
bool _renderPortrait = 0;
cv::Mat _presentationImg;
int _video_input_width = 0;  // Get from media _img
int _video_input_height = 0; // Get from media _img
std::vector<cv::VideoCapture> _vidcaps;
int _window_width = 1280;
int _window_height = 720;

class ObjectDetectionInterface {
public:
    const static size_t MODEL_DIM_COUNT = 4;
    int64_t model_input_shape[MODEL_DIM_COUNT] = { 0 };

    virtual ~ObjectDetectionInterface() {}
    virtual const char* getModelName() = 0;
    virtual const uint64_t getModelVersion() = 0;
    virtual const char* getModelInputName() = 0;
    virtual const  size_t getModelDimCount() = 0;
    virtual const std::vector<int> getModelInputShape() = 0;
    virtual const std::string getClassLabelText(int classIndex) = 0;    

    static inline float sigmoid(float x) {
        return 1.f / (1.f + std::exp(-x));
    }

    static inline float linear(float x) {
        return x;
    }

    double intersectionOverUnion(const DetectedResult& o1, const DetectedResult& o2) {
        double overlappingWidth = std::fmin(o1.x + o1.width, o2.x + o2.width) - std::fmax(o1.x, o2.x);
        double overlappingHeight = std::fmin(o1.y + o1.height, o2.y + o2.height) - std::fmax(o1.y, o2.y);
        double intersectionArea = (overlappingWidth < 0 || overlappingHeight < 0) ? 0 : overlappingHeight * overlappingWidth;
        double unionArea = o1.width * o1.height + o2.width * o2.height - intersectionArea;
        return intersectionArea / unionArea;
    }

    virtual void postprocess(const int64_t* output_shape, const void* voutputData, const size_t bytesize, const uint32_t dimCount, std::vector<DetectedResult> &detectedResults)
    {
        // derived to implement
    }

    // Geti detection/inference postprocess
    virtual void postprocess(
        const int64_t* output_shape_boxes, const void* voutputData_boxes, const size_t bytesize_boxes, const uint32_t dimCount_boxes,
        const int64_t* output_shape_labels, const void* voutputData_labels, const size_t bytesize_labels, const uint32_t dimCount_labels, 
        std::vector<DetectedResult> &detectedResults)
    {
        // derived to implement
    }

    // IOU postproc filter
    void postprocess(std::vector<DetectedResult> &detectedResults, std::vector<DetectedResult> &outDetectedResults)
    {
        if (useAdvancedPostprocessing) {
            // Advanced postprocessing
            // Checking IOU threshold conformance
            // For every i-th object we're finding all objects it intersects with, and comparing confidence
            // If i-th object has greater confidence than all others, we include it into result
            for (const auto& obj1 : detectedResults) {
                bool isGoodResult = true;
                for (const auto& obj2 : detectedResults) {
                    if (obj1.classId == obj2.classId && obj1.confidence < obj2.confidence &&
                        intersectionOverUnion(obj1, obj2) >= boxiou_threshold) {  // if obj1 is the same as obj2, condition
                                                                                // expression will evaluate to false anyway
                        isGoodResult = false;
                        break;
                    }
                }
                if (isGoodResult) {
                    outDetectedResults.push_back(obj1);
                }
            }
        } else {
            // Classic postprocessing
            std::sort(detectedResults.begin(), detectedResults.end(), [](const DetectedResult& x, const DetectedResult& y) {
                return x.confidence > y.confidence;
            });
            for (size_t i = 0; i < detectedResults.size(); ++i) {
                if (detectedResults[i].confidence == 0)
                    continue;
                for (size_t j = i + 1; j < detectedResults.size(); ++j)
                    if (intersectionOverUnion(detectedResults[i], detectedResults[j]) >= boxiou_threshold)
                        detectedResults[j].confidence = 0;
                outDetectedResults.push_back(detectedResults[i]);
            } //end for
        } // end if
    } // end postprocess IOU filter


protected:
    float confidence_threshold = .9;
    float boxiou_threshold = .4;
    float iou_threshold = 0.4;
    int classes =  80;
    bool useAdvancedPostprocessing = false;

};
   
class Midasnet  {
public:

    const static size_t MODEL_DIM_COUNT = 4;
    int64_t model_input_shape[MODEL_DIM_COUNT] = { 0 };

    Midasnet() {
        //confidence_threshold = .7;
        //classes = 1;
        std::vector<int> vmodel_input_shape = getModelInputShape();
        std::copy(vmodel_input_shape.begin(), vmodel_input_shape.end(), model_input_shape);
    }

    const char* getModelName() {
        return MODEL_NAME;
    }

    const uint64_t getModelVersion() {
        return MODEL_VERSION;
    }

    const char* getModelInputName() {
        return INPUT_NAME;
    }

    const size_t getModelDimCount() {
        return MODEL_DIM_COUNT;
    }

    const std::vector<int> getModelInputShape() {
        std::vector<int> shape{1, 3, 384, 384};
        return shape;
    }

    const std::string getClassLabelText(int classIndex) {
        return "N/A";
    }

    float postprocess(const void* voutputData, const int64_t* output_shape, size_t bytesize, pixel depth_pixel)
    {
        if (!voutputData || !output_shape) {
            // nothing to do
            return 0;
        }

        return 0;
        
        // Output Info
        // Inverse depth map, name - inverse_depth, shape - 1, 384, 384, format is B, H, W
        const float* outData = reinterpret_cast<const float*>(voutputData);
        cv::Mat tmp(output_shape[3], output_shape[2], CV_32F, (void*) voutputData);
        
        resize(tmp, tmp, cv::Size(_video_input_width, _video_input_height), 0, 0, cv::INTER_LINEAR);
        cv::normalize(tmp, tmp, 0, 1, cv::NORM_MINMAX, CV_32F);

        int x0 = (static_cast<float>(depth_pixel.first));
        int y0 = (static_cast<float>(depth_pixel.second) );
        int idxResized = y0 *  _video_input_width + x0;
        int idx = y0 *  output_shape[2] + x0;
        //printf("idx %i and value %f \n", idx, tmp.at<float>(idxResized));
        return outData[idx];
        
    } // End of Post-Processing

private:
    /* https://github.com/openvinotoolkit/open_model_zoo/tree/master/models/public/midasnet */
    // Image, name - image, shape - 1, 3, 384, 384, format is B, C, H, W
    const char* MODEL_NAME = "midasnet";
    const uint64_t MODEL_VERSION = 0;
    const char* INPUT_NAME = "image";
};

class GetiYoloX : public ObjectDetectionInterface {
public:

    GetiYoloX() {
        confidence_threshold = .7;
        classes = 1;
        std::vector<int> vmodel_input_shape = getModelInputShape();
        std::copy(vmodel_input_shape.begin(), vmodel_input_shape.end(), model_input_shape);
    }

    const char* getModelName() {
        return MODEL_NAME;
    }

    const uint64_t getModelVersion() {
        return MODEL_VERSION;
    }

    const char* getModelInputName() {
        return INPUT_NAME;
    }

    const size_t getModelDimCount() {
        return MODEL_DIM_COUNT;
    }

    const std::vector<int> getModelInputShape() {
        std::vector<int> shape{1, 3, 416, 416};
        return shape;
    }

    const std::string getClassLabelText(int classIndex) {
        return "FoundItem";
    }

    void postprocess(
        const int64_t* output_shape_boxes, const void* voutputData_boxes, const size_t bytesize_boxes, const uint32_t dimCount_boxes,
        const int64_t* output_shape_labels, const void* voutputData_labels, const size_t bytesize_labels, const uint32_t dimCount_labels, 
        std::vector<DetectedResult> &detectedResults)
    {
        if (!voutputData_boxes || !output_shape_boxes || !voutputData_labels || !output_shape_labels) {
            // nothing to do
            return;
        }

        if (dimCount_boxes != 3 || dimCount_labels != 2)
        {
            printf("Unknown Geti detection model.\n");
            return;
        }

        // Input Info
        // image  - 1,3,H,W

        // Output Info
        // boxes -  1, 100, 5
        // labels - 1,100
        // [  x_min, y_min, x_max, y_max, conf]
        const int numberOfDetections = output_shape_boxes[1];
        const int boxesSize = output_shape_boxes[2];
        const float* outData = reinterpret_cast<const float*>(voutputData_boxes);
        const double* outDataLabels = reinterpret_cast<const double*>(voutputData_labels);

        std::vector<int> input_shape = getModelInputShape();
        int network_h =  input_shape[2];
        int network_w =  input_shape[3];

        
        // printf("Network %f %f numDets %d outputDims: %d imageId: %f label: %f  \n",
        //     network_h, network_w, numberOfDetections, objectSize, outData[0 * objectSize + 0], outData[0 * objectSize + 1]);

        for (int i = 0; i < numberOfDetections; i++)
        {
            float confidence = outData[i * boxesSize + 4];
            double classId = outDataLabels[i];

            //printf("Confidence found: %f\n", confidence);

            if (confidence > confidence_threshold ) {
                DetectedResult obj;
                obj.x = std::clamp(
                    static_cast<int>((outData[i * boxesSize + 0] / ((float)network_w / (float)_video_input_width))),
                     0, 
                     _video_input_width); 
                obj.y = std::clamp(
                    static_cast<int>((outData[i * boxesSize + 1] / ((float)network_h/(float)_video_input_height))), 
                    0, 
                    _video_input_height);
                obj.width = std::clamp(
                    static_cast<int>((outData[i * boxesSize + 2] / ((float)network_w/(float)_video_input_width)  - obj.x)), 
                    0, 
                    _video_input_width); 
                obj.height = std::clamp(
                    static_cast<int>((outData[i * boxesSize + 3] / ((float)network_h/(float)_video_input_height) - obj.y)), 
                    0, 
                    _video_input_height); 
                obj.confidence = confidence;
                obj.classId = (int) classId;
                strncpy(obj.classText, getClassLabelText(obj.classId).c_str(), sizeof(obj.classText));

                // printf("Actual found: %f...%f,%f,%f,%f...%ix%i \n", 
                //     confidence,
                //     obj.x,
                //     obj.y,
                //     obj.width,
                //     obj.height,
                //     _video_input_width,
                //     _video_input_height);

                
                if (obj.classId != 0)
                    printf("SHOULDN'T OCCUR:---------found: %s\n", obj.classText);
                detectedResults.push_back(obj);
            } // end if confidence
        } // end for
    }

private:
    /* Model Serving Info for https://github.com/openvinotoolkit/open_model_zoo/tree/master/models/intel/face-detection-retail-0005 */
    // geti yolox - ?x3x416x416 NCHW
    const char* MODEL_NAME = "geti_yolox";
    const uint64_t MODEL_VERSION = 0;
    const char* INPUT_NAME = "image";
};

class FaceDetection0005 : public ObjectDetectionInterface {
public:

    FaceDetection0005() {
        confidence_threshold = .5;
        classes = 1;
        std::vector<int> vmodel_input_shape = getModelInputShape();
        std::copy(vmodel_input_shape.begin(), vmodel_input_shape.end(), model_input_shape);
    }

    const char* getModelName() {
        return MODEL_NAME;
    }

    const uint64_t getModelVersion() {
        return MODEL_VERSION;
    }

    const char* getModelInputName() {
        return INPUT_NAME;
    }

    const size_t getModelDimCount() {
        return MODEL_DIM_COUNT;
    }

    const std::vector<int> getModelInputShape() {
        std::vector<int> shape{1, 3, 800, 800};
        return shape;
    }

    const std::string getClassLabelText(int classIndex) {
        return (classIndex == 1 ? "Face" : "Unknown");
    }

    /*
    * Reference: FaceDetection
    * TODO: Move a shared lib.
    */
    void postprocess(const int64_t* output_shape, const void* voutputData, const size_t bytesize, const uint32_t dimCount, std::vector<DetectedResult> &detectedResults)
    {
        if (!voutputData || !output_shape) {
            // nothing to do
            return;
        }
        // Input Info
        // input.1  - 1,3,H,W

        // Output Info
        // data - 1, 1, 200, 7
        // [image_id, label, conf, x_min, y_min, x_max, y_max],
        const int numberOfDetections = output_shape[2];
        const int objectSize = output_shape[3];
        const float* outData = reinterpret_cast<const float*>(voutputData);
        std::vector<int> input_shape = getModelInputShape();
        int network_h =  input_shape[2];
        int network_w =  input_shape[3];
        // printf("Network %f %f numDets %d outputDims: %d imageId: %f label: %f  \n",
        //     network_h, network_w, numberOfDetections, objectSize, outData[0 * objectSize + 0], outData[0 * objectSize + 1]);

        for (int i = 0; i < numberOfDetections; i++)
        {
            float image_id = outData[i * objectSize + 0];
            if (image_id < 0)
                break;

            float confidence = outData[i * objectSize + 2];

            //printf("Confidence found: %f\n", confidence);

            if (confidence > confidence_threshold ) {
                //printf("Confidence found: %f\n", confidence);
                DetectedResult obj;
                obj.x = std::clamp(static_cast<int>(outData[i * objectSize + 3] * _video_input_width), 0, _video_input_width); 
                obj.y = std::clamp(static_cast<int>(outData[i * objectSize + 4] * _video_input_height), 0, _video_input_height); 
                obj.width = std::clamp(static_cast<int>(outData[i * objectSize + 5] * _video_input_width - obj.x), 0, _video_input_width); 
                obj.height = std::clamp(static_cast<int>(outData[i * objectSize + 6] * _video_input_height - obj.y), 0, _video_input_height); 
                obj.confidence = confidence;
                obj.classId = outData[i * objectSize + 1];
                strncpy(obj.classText, getClassLabelText(obj.classId).c_str(), sizeof(obj.classText));
                
                if (obj.classId != 1)
                    printf("SHOULDN'T OCCUR:---------found: %s\n", obj.classText);
                detectedResults.push_back(obj);
            } // end if confidence
        } // end for
    } // End of FaceDetect Post-Processing


private:
    /* Model Serving Info for https://github.com/openvinotoolkit/open_model_zoo/tree/master/models/intel/face-detection-retail-0005 */
    // FaceDet - 1x3x300x300 NCHW
    const char* MODEL_NAME = "face_detection";
    const uint64_t MODEL_VERSION = 0;
    const char* INPUT_NAME = "input.1";
};

class SSD : public ObjectDetectionInterface {
public:

    SSD() {
        confidence_threshold = .9;
        classes = 2;
        std::vector<int> vmodel_input_shape = getModelInputShape();
        std::copy(vmodel_input_shape.begin(), vmodel_input_shape.end(), model_input_shape);

        //std::cout << "Using object detection type person-detection-retail-0013" << std::endl;
    }

    const char* getModelName() {
        return MODEL_NAME;
    }

    const uint64_t getModelVersion() {
        return MODEL_VERSION;
    }

    const char* getModelInputName() {
        return INPUT_NAME;
    }

    const size_t getModelDimCount() {
        return MODEL_DIM_COUNT;
    }

    const std::vector<int> getModelInputShape() {
        std::vector<int> shape{1, 320, 544, 3};
        return shape;
    }

    const std::string getClassLabelText(int classIndex) {
        return (classIndex == 1 ? "Person" : "Unknown");
    }

    /*
    * Reference: SSD
    * TODO: Move a shared lib.
    */
    void postprocess(const int64_t* output_shape, const void* voutputData, const size_t bytesize, const uint32_t dimCount, std::vector<DetectedResult> &detectedResults)
    {
        if (!voutputData || !output_shape) {
            // nothing to do
            return;
        }
            // detection_out 4 1 1 200 7 5600 1
        const int numberOfDetections = output_shape[2];
        const int objectSize = output_shape[3];
        const float* outData = reinterpret_cast<const float*>(voutputData);
        std::vector<int> input_shape = getModelInputShape();
        float network_h = (float) input_shape[1];
        float network_w = (float) input_shape[2];
        //printf("Network %f %f numDets %d \n", network_h, network_w, numberOfDetections);

        for (int i = 0; i < numberOfDetections; i++)
        {
            float image_id = outData[i * objectSize + 0];
            if (image_id < 0)
                break;

            float confidence = outData[i * objectSize + 2];

            if (confidence > confidence_threshold ) {
                DetectedResult obj;
                        obj.x = std::clamp(outData[i * objectSize + 3] * network_w, 0.f, static_cast<float>(network_w)); // std::clamp(outData[i * objectSize +3], 0.f,network_w);
                        obj.y = std::clamp(outData[i * objectSize + 4] * network_h, 0.f, static_cast<float>(network_h)); //std::clamp(outData[i * objectSize +4], 0.f,network_h);
                        obj.width = std::clamp(outData[i * objectSize + 5] * network_w, 0.f, static_cast<float>(network_w)) - obj.x; // std::clamp(outData[i*objectSize+5],0.f,network_w-obj.x);
                        obj.height = std::clamp(outData[i * objectSize + 6] * network_h, 0.f, static_cast<float>(network_h)) - obj.y; // std::clamp(outData[i*objectSize+6],0.f, network_h-obj.y);
                obj.confidence = confidence;
                            obj.classId = outData[i * objectSize + 1];
                            strncpy(obj.classText, getClassLabelText(obj.classId).c_str(), sizeof(obj.classText));
                //if (strncmp(obj.classText, "person", sizeof("person") != 0 ))
                //	continue;
                if (obj.classId != 1)
                printf("---------found: %s\n", obj.classText);

                            detectedResults.push_back(obj);

            } // end if confidence
        } // end for
    } // End of SSD Person Detection Post-Processing


private:
    /* Model Serving Info for https://github.com/openvinotoolkit/open_model_zoo/tree/master/models/intel/person-detection-retail-0013 */
    // SSD - 1x3x320x544 NCHW
    const char* MODEL_NAME = "people-detection-retail-0013";
    const uint64_t MODEL_VERSION = 0;
    const char* INPUT_NAME = "data";
};

class Yolov5 : public ObjectDetectionInterface
{
public:

    Yolov5()
    {
        confidence_threshold = .5;
        classes = 80;
        std::vector<int> vmodel_input_shape = getModelInputShape();
        std::copy(vmodel_input_shape.begin(), vmodel_input_shape.end(), model_input_shape);

        //std::cout << "Using object detection type Yolov5" << std::endl;
    }

    const char* getModelName() {
        return MODEL_NAME;
    }

    const uint64_t getModelVersion() {
        return MODEL_VERSION;
    }

    const char* getModelInputName() {
        return INPUT_NAME;
    }

    const size_t getModelDimCount() {
        return MODEL_DIM_COUNT;
    }

    const std::vector<int> getModelInputShape() {
        std::vector<int> shape{1, 416, 416, 3};
        return shape;
    }

    const std::string getClassLabelText(int classIndex) {
        if (classIndex > 80)
            return "Unknown";
        return labels[classIndex];
    }

    int calculateEntryIndex(int totalCells, int lcoords, size_t lclasses, int location, int entry) {
        int n = location / totalCells;
        int loc = location % totalCells;
        return (n * (lcoords + lclasses) + entry) * totalCells + loc;
    }

    // Yolov5
    void postprocess(const int64_t* output_shape, const void* voutputData, const size_t bytesize, const uint32_t dimCount, std::vector<DetectedResult> &detectedResults)
    {
        if (!voutputData || !output_shape) {
            // nothing to do
            return;
        }

        const int regionCoordsCount  = dimCount;
        const int sideH = output_shape[2]; // NCHW
        const int sideW = output_shape[3]; // NCHW
        const int regionNum = 3;
        std::vector<int> input_shape = getModelInputShape();
        const int scaleH = input_shape[1]; // NHWC
        const int scaleW = input_shape[2]; // NHWC

        auto entriesNum = sideW * sideH;
        const float* outData = reinterpret_cast<const float*>(voutputData);
        int original_im_w = _video_input_width;
        int original_im_h = _video_input_height;
        //printf("original im: %ix%i\n", original_im_w, original_im_h);

        auto postprocessRawData = sigmoid; //sigmoid or linear

        for (int i = 0; i < entriesNum; ++i) {
            int row = i / sideW;
            int col = i % sideW;

            for (int n = 0; n < regionNum; ++n) {

                int obj_index = calculateEntryIndex(entriesNum,  regionCoordsCount, classes + 1 /* + confidence byte */, n * entriesNum + i,regionCoordsCount);
                int box_index = calculateEntryIndex(entriesNum, regionCoordsCount, classes + 1, n * entriesNum + i, 0);
                float outdata = outData[obj_index];
                float scale = postprocessRawData(outData[obj_index]);

                //printf("scale found: %f\n", scale);

                if (scale >= confidence_threshold) {
                    float x, y,height,width;
                    x = static_cast<float>((col + postprocessRawData(outData[box_index + 0 * entriesNum])) / sideW * original_im_w);
                    y = static_cast<float>((row + postprocessRawData(outData[box_index + 1 * entriesNum])) / sideH * original_im_h);
                    width = static_cast<float>(std::pow(2*postprocessRawData(outData[box_index + 2 * entriesNum]),2) * anchors_13[2 * n] *   original_im_w / scaleW);
                    height = static_cast<float>(std::pow(2*postprocessRawData(outData[box_index + 3 * entriesNum]),2) * anchors_13[2 * n + 1] *   original_im_h / scaleH);

                    DetectedResult obj;
                    obj.x = std::clamp(x - width / 2, 0.f, static_cast<float>(original_im_w));
                    obj.y = std::clamp(y - height / 2, 0.f, static_cast<float>(original_im_h));
                    obj.width = std::clamp(width, 0.f, static_cast<float>(original_im_w - obj.x));
                    obj.height = std::clamp(height, 0.f, static_cast<float>(original_im_h - obj.y));

                    for (size_t j = 0; j < classes; ++j) {
                        int class_index = calculateEntryIndex(entriesNum, regionCoordsCount, classes + 1, n * entriesNum + i, regionCoordsCount + 1 + j);
                        float prob = scale * postprocessRawData(outData[class_index]);

                        if (prob >= confidence_threshold) {
                            obj.confidence = prob;
                            obj.classId = j;
                            strncpy(obj.classText, getClassLabelText(j).c_str(), sizeof(obj.classText));
                            // if (j != 0)   // debug for person
                            //     continue;
                            // if ( j != 28) // debug for suitecase
                            //     continue;
                            // if ( j != 39) // debug for bottle
                            //     continue;
                            //  printf("found %s %f, %f, %f, %f ---> %f, %f, %f, %f %ix%i %ix%i %ix%i \n", 
                            //      obj.classText, x,y,width,height, obj.x, obj.y, obj.width, obj.height, sideW, sideH, scaleW, scaleH, original_im_w, original_im_h);
                            detectedResults.push_back(obj);
                        }
                    }
                } // end else
            } // end for
        } // end for
    }
// End of Yolov5 Post-Processing

private:
    /* Yolov5s Model Serving Info */
    // YOLOV5 - 1x3x416x416 NCHW
    const char* MODEL_NAME = "yolov5s";
    const uint64_t MODEL_VERSION = 0;
    const char* INPUT_NAME = "images";

    // Anchors by region/output layer
    const float anchors_52[6] = {
        10.0,
        13.0,
        16.0,
        30.0,
        33.0,
        23.0
    };

    const float anchors_26[6] = {
        30.0,
        61.0,
        62.0,
        45.0,
        59.0,
        119.0
    };

    const float anchors_13[6] = {
        116.0,
        90.0,
        156.0,
        198.0,
        373.0,
        326.0
    };

    const std::string labels[80] = {
        "person",
        "bicycle",
        "car",
        "motorbike",
        "aeroplane",
        "bus",
        "train",
        "truck",
        "boat",
        "traffic light",
        "fire hydrant",
        "stop sign",
        "parking meter",
        "bench",
        "bird",
        "cat",
        "dog",
        "horse",
        "sheep",
        "cow",
        "elephant",
        "bear",
        "zebra",
        "giraffe",
        "backpack",
        "umbrella",
        "handbag",
        "tie",
        "suitcase",
        "frisbee",
        "skis",
        "snowboard",
        "sports ball",
        "kite",
        "baseball bat",
        "baseball glove",
        "skateboard",
        "surfboard",
        "tennis racket",
        "bottle",
        "wine glass",
        "cup",
        "fork",
        "knife",
        "spoon",
        "bowl",
        "banana",
        "apple",
        "sandwich",
        "orange",
        "broccoli",
        "carrot",
        "hot dog",
        "pizza",
        "donut",
        "cake",
        "chair",
        "sofa",
        "pottedplant",
        "bed",
        "diningtable",
        "toilet",
        "tvmonitor",
        "laptop",
        "mouse",
        "remote",
        "keyboard",
        "cell phone",
        "microwave",
        "oven",
        "toaster",
        "sink",
        "refrigerator",
        "book",
        "clock",
        "vase",
        "scissors",
        "teddy bear",
        "hair drier",
        "toothbrush"
    };

};

class TextDetection : public ObjectDetectionInterface {
public:

    TextDetection() {
        confidence_threshold = .2;
        classes = 1;
        std::vector<int> vmodel_input_shape = getModelInputShape();
        std::copy(vmodel_input_shape.begin(), vmodel_input_shape.end(), model_input_shape);

        //std::cout << "Using object detection type text-detection-00012" << std::endl;
    }

    const char* getModelName() {
        return MODEL_NAME;
    }

    const uint64_t getModelVersion() {
        return MODEL_VERSION;
    }

    const char* getModelInputName() {
        return INPUT_NAME;
    }

    const size_t getModelDimCount() {
        return MODEL_DIM_COUNT;
    }

    const std::vector<int> getModelInputShape() {
        std::vector<int> shape{1, 704, 704, 3};
        return shape;
    }

    const std::string getClassLabelText(int classIndex) {
        return "text";
    }

    /*
    * Reference: https://github.com/openvinotoolkit/model_server/blob/4d4c067baec66f01b1f17795406dd01e18d8cf6a/demos/horizontal_text_detection/python/horizontal_text_detection.py
    * TODO: Move a shared lib.
    */
    void postprocess(const int64_t* output_shape, const void* voutputData, const size_t bytesize, const uint32_t dimCount, std::vector<DetectedResult> &detectedResults)
    {
        if (!voutputData || !output_shape) {
            // nothing to do
            return;
        }
        // boxes shape - N,5 or 100,5
        const int numberOfDetections = output_shape[1];
        const int objectSize = output_shape[2];
        const float* outData = reinterpret_cast<const float*>(voutputData);
        std::vector<int> input_shape = getModelInputShape();
        float network_h = (float) input_shape[1];
        float network_w = (float) input_shape[2];
        float scaleW = 1.0;
        float scaleH = 1.0;

        if (_render) {
            scaleW = (float)_window_width / network_w;
            scaleH = (float)_window_height / network_w;
        }

        //printf("----->Network %f %f numDets %d objsize %d \n", network_h, network_w, numberOfDetections, objectSize);

        for (int i = 0; i < numberOfDetections; i++)
        {
            float confidence = outData[i * objectSize + 4];
            //printf("------>text conf: %f\n", outData[i * objectSize + 4]);

            if (confidence > confidence_threshold ) {
                DetectedResult obj;
                obj.x = outData[i * objectSize + 0] * scaleW;
                obj.y = outData[i * objectSize + 1] * scaleH;
                // Yolo/SSD is not bottom-left/bottom-right so make consistent by subtracking
                obj.width = outData[i * objectSize + 2] * scaleW - obj.x;
                obj.height = outData[i * objectSize + 3] * scaleH - obj.y;
                obj.confidence = confidence;
                obj.classId = 0; // only text can be detected
                strncpy(obj.classText, getClassLabelText(obj.classId).c_str(), sizeof(obj.classText));
                //printf("Adding obj %f %f %f %f with label %s\n",obj.x, obj.y, obj.width, obj.height, obj.classText);
                detectedResults.push_back(obj);

            } // end if confidence
        } // end for
    } // End of Text-Det Post-Processing


private:
    /* Model Serving Info :
      https://github.com/dlstreamer/pipeline-zoo-models/blob/main/storage/horizontal-text-detection-0002/
      https://github.com/openvinotoolkit/model_server/blob/4d4c067baec66f01b1f17795406dd01e18d8cf6a/demos/horizontal_text_detection/python/horizontal_text_detection.py
    */
    const char* MODEL_NAME = "text-detect-0002";
    const uint64_t MODEL_VERSION = 0;
    const char* INPUT_NAME = "input";
};

namespace {
volatile sig_atomic_t shutdown_request = 0;
}

Midasnet* _objMidas;

bool stringIsInteger(std::string strInput) {
    std::string::const_iterator it = strInput.begin();
    while (it != strInput.end() && std::isdigit(*it)) ++it;
    return !strInput.empty() && it == strInput.end();
}

bool setActiveModel(int detectionType, ObjectDetectionInterface** objDet)
{
    if (objDet == NULL)
        return false;

    _objMidas = new Midasnet();

    if (detectionType == 0) {
        *objDet = new Yolov5();
    }
    else if(detectionType == 1) {
        *objDet = new SSD();
    }
    else if(detectionType == 2) {
        *objDet = new FaceDetection0005();
    }
    else if(detectionType == 3) {
        *objDet = new GetiYoloX();
    }    
    else
        std::cout << "ERROR: detectionType option must be 0 (yolov5) or 1 (people-detection-retail-0013) or 3 face detection" << std::endl;
    return true;
}

static void onInterrupt(int status) {
    shutdown_request = 1;
}

static void onTerminate(int status) {
    shutdown_request = 1;
}

static void onIllegal(int status) {
    shutdown_request = 2;
}

static void installSignalHandlers() {
    static struct sigaction sigIntHandler;
    sigIntHandler.sa_handler = onInterrupt;
    sigemptyset(&sigIntHandler.sa_mask);
    sigIntHandler.sa_flags = 0;
    sigaction(SIGINT, &sigIntHandler, NULL);

    static struct sigaction sigTermHandler;
    sigTermHandler.sa_handler = onTerminate;
    sigemptyset(&sigTermHandler.sa_mask);
    sigTermHandler.sa_flags = 0;
    sigaction(SIGTERM, &sigTermHandler, NULL);

    static struct sigaction sigIllHandler;
    sigIllHandler.sa_handler = onIllegal;
    sigemptyset(&sigIllHandler.sa_mask);
    sigIllHandler.sa_flags = 0;
    sigaction(SIGILL, &sigIllHandler, NULL);
}

void printInferenceResults(std::vector<DetectedResult> &results)
{
	for (auto & obj : results) {
	  std::cout << "Rect: [ " << obj.x << " , " << obj.y << " " << obj.width << ", " << obj.height << "] Class: " << obj.classText << "(" << obj.classId << ") Conf: " << obj.confidence << std::endl;
	}
}

// TODO: Multiple references state that imshow can't be used in any other thread than main!
void displayGUIInferenceResults(cv::Mat analytics_frame, std::vector<DetectedResult> &results, int latency, int througput)
{
    auto ttid = std::this_thread::get_id();
    std::stringstream ss;
    ss << ttid;
    std::string tid = ss.str();

    for (auto & obj : results) {
        
        if (obj.confidence == 0)
            continue;

	    const float x0 = obj.x;
        const float y0 = obj.y;
        const float x1 = obj.x + obj.width;
        const float y1 = obj.y + obj.height;

        //printf("--------->coords: %f %f %f %f\n", x0, y0, x1, y1);
        cv::rectangle( analytics_frame,
            cv::Point( (int)(x0),(int)(y0) ),
            cv::Point( (int)x1, (int)y1 ),
            cv::Scalar(255, 0, 0),
            2, cv::LINE_8 );

        if (obj.actual_width && obj.actual_height)
        {            
            string message = "Object Size: " + to_string(obj.actual_width) + "cm X " + to_string(obj.actual_height) + " cm";
            cv::Size textsize = cv::getTextSize(message.c_str(), cv::FONT_HERSHEY_PLAIN, 1, 0,0);
            cv::rectangle(analytics_frame, 
                cv::Point( (int)(x0),(int)(y0-20) ), 
                cv::Point((int)x0 + textsize.width, (int)y0 + textsize.height), 
                CV_RGB(0, 0, 0), 
                -1);
            cv::putText(analytics_frame, message.c_str(), cv::Size((int)x0, (int)y0), cv::FONT_HERSHEY_PLAIN, 1, CV_RGB(255, 255, 255), 1);

            cv::line(analytics_frame, 
                cv::Point(obj.point_x0_0_actual_measured, obj.point_y0_0_actual_measured), 
                cv::Point(obj.point_x1_0_actual_measured, obj.point_y1_0_actual_measured), 
                CV_RGB(0, 255, 0),
                1, LINE_4);

            cv::line(analytics_frame, 
                cv::Point(obj.point_x0_1_actual_measured, obj.point_y0_1_actual_measured), 
                cv::Point(obj.point_x1_1_actual_measured, obj.point_y1_1_actual_measured), 
                CV_RGB(0, 255, 0),
                1, LINE_4);
        }
    } // end for

    // std::string fps_msg = (througput == 0) ? "..." : std::to_string(througput) + "fps";
    // std::string latency_msg = (latency == 0) ? "..." :  std::to_string(latency) + "ms";
    // std::string roiCount_msg = std::to_string(results.size());
    // std::string message = "E2E Pipeline Performance: " + latency_msg + " and " + fps_msg + " with ROIs#" + roiCount_msg;
    // cv::putText(analytics_frame, message.c_str(), cv::Size(0, 20), cv::FONT_HERSHEY_PLAIN, 1, (255, 0, 0), 1, cv::LINE_4);
    // cv::putText(analytics_frame, tid, cv::Size(0, 40), cv::FONT_HERSHEY_PLAIN, 1, (255, 0, 0), 1, cv::LINE_4);

    //printf("%ix%i\n", analytics_frame.cols, analytics_frame.rows);

    cv::Mat presenter;

    {
        std::lock_guard<std::mutex> lock(_drawingMtx);
        cv::imshow("OpenVINO Results " + tid, analytics_frame);
        cv::waitKey(1);
    }
}

void saveInferenceResultsAsVideo(cv::Mat &presenter, std::vector<DetectedResult> &results)
{
    for (auto & obj : results) {

        const float scaler_w = 416.0f/_video_input_width;
        const float scaler_h = 416.0f/_video_input_height;
        //std::cout << " Scalers " << scaler_w << " " << scaler_h << std::endl;
        //std::cout << "xDrawing at " << (int)obj.x*scaler_w << "," << (int)obj.y*scaler_h << " " << (int)(obj.x+obj.width)*scaler_w << " " << (int) (obj.y+obj.height)* scaler_h << std::endl;

        cv::rectangle( presenter,
         cv::Point( (int)(obj.x*scaler_w),(int)(obj.y*scaler_h) ),
         cv::Point( (int)((obj.x+obj.width) * scaler_w), (int)((obj.y+obj.height)*scaler_h) ),
         cv::Scalar(255, 0, 0),
         4, cv::LINE_8 );
  } // end for
  cv::imwrite("result.jpg", presenter);
}

// This function is responsible for generating a GST pipeline that
// decodes and resizes the video stream based on the desired window size or
// the largest analytics frame size needed if running headless
std::string getVideoPipeline(std::string serial, rs2::pipeline** pipe, rs2::config** cfg, rs2::align** align_to, ObjectDetectionInterface* objDet, ObjectDetectionInterface* textDet)
{

    *pipe = new rs2::pipeline();
    *cfg = new rs2::config();

    string pipelineText = "";
    std::vector<int> modelFrameShape = objDet->getModelInputShape();
    if (textDet) {
        modelFrameShape = textDet->getModelInputShape();
    }

    int frame_width = modelFrameShape[1];
    int frame_height = modelFrameShape[2];

    if (_render)
    {
        frame_width = _window_width;
        frame_height = _window_height;
    }

    (*cfg)->enable_device(serial);
    pipelineText += "Camera: ";
    pipelineText += serial;

    // (*cfg)->enable_stream(RS2_STREAM_ACCEL, RS2_FORMAT_MOTION_XYZ32F);      
    // (*cfg)->enable_stream(RS2_STREAM_GYRO, RS2_FORMAT_MOTION_XYZ32F);
    if (_color_frame__height && _color_frame_width  && _color_frame_rate )
    {
        //(*cfg)->enable_stream(RS2_STREAM_INFRARED, _color_frame_width, _color_frame__height, RS2_FORMAT_Y8, _color_frame_rate);
        (*cfg)->enable_stream(RS2_STREAM_COLOR, _color_frame_width, _color_frame__height, RS2_FORMAT_RGB8, _color_frame_rate);
        
        pipelineText += ", Color Settings : " + to_string(_color_frame_width)  + " x " + to_string(_color_frame__height);
        pipelineText += " " + to_string(_color_frame_rate) + " Hz";
    }
    else
    {
        (*cfg)->enable_stream(RS2_STREAM_COLOR, RS2_FORMAT_RGB8);
        pipelineText += ", Color Settings: Default";
    }

    if (_depth_frame_height && _depth_frame_rate && _depth_frame_width)
    {
        (*cfg)->enable_stream(RS2_STREAM_DEPTH, _depth_frame_width, _depth_frame_height, RS2_FORMAT_Z16, _depth_frame_rate);
        pipelineText += ", Depth Settings : " + to_string(_depth_frame_width)  + " x " + to_string(_depth_frame_height);
        pipelineText += " " + to_string(_depth_frame_rate) + " Hz";
    }
    else
    {
        pipelineText += ", Depth Settings: Default";
        (*cfg)->enable_stream(RS2_STREAM_DEPTH, RS2_FORMAT_Z16);
    }

    *align_to = nullptr;
    switch(_frame_align)
    {
      case 0:
        pipelineText += ", Frame align: None";        
        break;
      case 1:
        *align_to = new rs2::align(RS2_STREAM_COLOR); 
        pipelineText += ", Frame align: to color";
        break;
      case 2:
        *align_to = new rs2::align(RS2_STREAM_DEPTH); 
        pipelineText += ", Frame align: to depth";
        break;
      default:
        pipelineText += ", Frame align: Unknown opttion specified";
    }

    return pipelineText;
}

bool createModelServer()
{
    if (_srv == NULL)
        return false;

    OVMS_Status* res = OVMS_ServerStartFromConfigurationFile(_srv, _serverSettings, _modelsSettings);

    if (res) {
        uint32_t code = 0;
        const char* details = nullptr;

        OVMS_StatusGetCode(res, &code);
        OVMS_StatusGetDetails(res, &details);
        std::cerr << "ERROR: during start: code:" << code << "; details:" << details
                  << "; grpc_port: " << _server_grpc_port
                  << "; http_port: " << _server_http_port
                  << ";" << std::endl;

        OVMS_StatusDelete(res);

        if (_srv)
            OVMS_ServerDelete(_srv);

        if (_modelsSettings)
            OVMS_ModelsSettingsDelete(_modelsSettings);

        if (_serverSettings)
            OVMS_ServerSettingsDelete(_serverSettings);

        return false;
    }

    return true;
}

bool loadRealsense(std::string serial_number, rs2::pipeline** pipe, rs2::config** cfg, rs2::align** align_to, ObjectDetectionInterface** objDet)
{
    static int threadCnt = 0;    

    std::string videoPipelineText = getVideoPipeline(serial_number, pipe, cfg, align_to, *objDet, NULL);
    std::cout << "--------------------------------------------------------------" << std::endl;
    std::cout << "Opening Media Pipeline: " << videoPipelineText << std::endl;
    std::cout << "--------------------------------------------------------------" << std::endl;    

    auto profile = (*pipe)->start(**cfg);
    auto sensor = profile.get_device().first<rs2::depth_sensor>();

    // Set the device to High Accuracy preset of the D400 stereoscopic cameras
    if (sensor && sensor.is<rs2::depth_stereo_sensor>())
    {
        // Below throws an error when using 1280x720 resolution for depth and color streams
        // Not needed unless diff rates/res'z
        //sensor.set_option(RS2_OPTION_VISUAL_PRESET, RS2_RS400_VISUAL_PRESET_HIGH_ACCURACY);
        //RS2_OPTION_EMITTER_ON_OFF
    }

    auto stream = profile.get_stream(RS2_STREAM_COLOR).as<rs2::video_stream_profile>();
    
    if (_render)
    {
        _window_width = stream.width();
        _window_height = stream.height();
        _video_input_height = _window_height;
        _video_input_width = _window_width;
    }

    return true;
}

// OVMS C-API is a global process (singleton design) wide server so can't create many of them
bool loadOVMS()
{
     OVMS_Status* res = NULL;     

     OVMS_ServerSettingsNew(&_serverSettings);
     OVMS_ModelsSettingsNew(&_modelsSettings);
     OVMS_ServerNew(&_srv);
     OVMS_ServerSettingsSetGrpcPort(_serverSettings, _server_grpc_port);
     OVMS_ServerSettingsSetRestPort(_serverSettings, _server_http_port);
     OVMS_ServerSettingsSetLogLevel(_serverSettings, OVMS_LOG_ERROR);

     OVMS_ModelsSettingsSetConfigPath(_modelsSettings, "./models/config_active.json");

     if (!createModelServer()) {
         std::cout << "Failed to create model server\n" << std::endl;
         return false;
     }
     else {
         std::cout << "--------------------------------------------------------------" << std::endl;
         std::cout << "Server ready for inference C-API ports " << _server_grpc_port << " " << _server_http_port << std::endl;
         std::cout << "--------------------------------------------------------------" << std::endl;
         _server_http_port+=1;
         _server_grpc_port+=1;
     }
     return true;
}

bool getMAPipeline(string serial_number, rs2::pipeline** pipe, rs2::config** cfg, rs2::align** align_to, ObjectDetectionInterface** objDet)
{
    if (!setActiveModel(_detectorModel, objDet)) {
        std::cout << "Unable to set active detection model" << std::endl;
        return false;
    }

    return loadRealsense(serial_number, pipe, cfg, align_to, objDet);
}

void hwc_to_chw(cv::InputArray src, cv::OutputArray dst) {
  std::vector<cv::Mat> channels;
  cv::split(src, channels);

  for (auto &img : channels) {
    img = img.reshape(1, 1);
  }

  // Concatenate three vectors to one
  cv::hconcat( channels, dst );
}

// Convert rs2::frame to cv::Mat
cv::Mat frame_to_mat(const rs2::frame& f)
{
    using namespace cv;
    using namespace rs2;

    auto vf = f.as<video_frame>();
    const int w = vf.get_width();
    const int h = vf.get_height();

    if (f.get_profile().format() == RS2_FORMAT_BGR8)
    {
        return Mat(Size(w, h), CV_8UC3, (void*)f.get_data(), Mat::AUTO_STEP);
    }
    else if (f.get_profile().format() == RS2_FORMAT_RGB8)
    {
        auto r_rgb = Mat(Size(w, h), CV_8UC3, (void*)f.get_data(), Mat::AUTO_STEP);
        Mat r_bgr;
        cvtColor(r_rgb, r_bgr, COLOR_RGB2BGR);
        return r_bgr;
    }
    else if (f.get_profile().format() == RS2_FORMAT_Z16)
    {
        return Mat(Size(w, h), CV_16UC1, (void*)f.get_data(), Mat::AUTO_STEP);
    }
    else if (f.get_profile().format() == RS2_FORMAT_Y8)
    {
        return Mat(Size(w, h), CV_8UC1, (void*)f.get_data(), Mat::AUTO_STEP);
    }
    else if (f.get_profile().format() == RS2_FORMAT_DISPARITY32)
    {
        return Mat(Size(w, h), CV_32FC1, (void*)f.get_data(), Mat::AUTO_STEP);
    }

    throw std::runtime_error("Frame format is not supported yet!");
}



float dist_3d(const rs2::depth_frame& frame, pixel u, pixel v)
{
    float upixel[2]; // From pixel
    float upoint[3]; // From point (in 3D)

    float vpixel[2]; // To pixel
    float vpoint[3]; // To point (in 3D)

    // Copy pixels into the arrays (to match rsutil signatures)
    upixel[0] = static_cast<float>(u.first);
    upixel[1] = static_cast<float>(u.second);
    vpixel[0] = static_cast<float>(v.first);
    vpixel[1] = static_cast<float>(v.second);

    // Too much void
    // auto udist = frame.get_distance(static_cast<int>(upixel[0]), static_cast<int>(upixel[1]));
    // auto vdist = frame.get_distance(static_cast<int>(vpixel[0]), static_cast<int>(vpixel[1]));
    // printf("getting distances calc: %fx%f and %fx%f == %fm and %fm\n", upixel[0], upixel[1], vpixel[0], vpixel[1], udist, vdist);

    // Calc distance from center of the BB/item
    float x0 = upixel[0];
    float y0 = upixel[1];
    float x1 = vpixel[0];
    float y1 = vpixel[1];
    x0 = x0 + ((x1 - x0)/2);
    y0 = y0 + ((y1 - y0)/2);

    auto udist = frame.get_distance(static_cast<int>(x0), static_cast<int>(y0));
    auto vdist = udist;
    //printf("orig: %fx%f->%fx%f and getting distances mid calc: %fx%f == %fm and %fm\n", upixel[0], upixel[1], vpixel[0], vpixel[1], x0, y0, udist, vdist);

    // Deproject from pixel to point in 3D
    rs2_intrinsics intr = frame.get_profile().as<rs2::video_stream_profile>().get_intrinsics(); // Calibration data
    rs2_deproject_pixel_to_point(upoint, &intr, upixel, udist);
    rs2_deproject_pixel_to_point(vpoint, &intr, vpixel, vdist);  

    // Calculate euclidean distance between the two points
    return sqrt(pow(upoint[0] - vpoint[0], 2.f) +
                pow(upoint[1] - vpoint[1], 2.f) +
                pow(upoint[2] - vpoint[2], 2.f));
}


void estimateObjectSizes(std::vector<DetectedResult> &detectedResults, const rs2::depth_frame& depth, const void* voutputData, const int64_t* outputShape, size_t bytesize)
{
    for (auto & obj : detectedResults) 
    {
        obj.actual_height = 0;
        obj.actual_width  = 0;

        if (obj.confidence == 0)
            continue;

        int BOUNDING_BOX_OVER_X0 = 7;
        int BOUNDING_BOX_OVER_X1 = 7;
        int BOUNDING_BOX_OVER_Y0 = 0;
        int BOUNDING_BOX_OVER_Y1 = 0;
        float air_dist = 0;

        // tighten bounding box manually
        int x0 = static_cast<int>(obj.x + BOUNDING_BOX_OVER_X0);
        int y0 = static_cast<int>(obj.y + BOUNDING_BOX_OVER_Y0);
        int x1 = static_cast<int>(obj.x + obj.width - BOUNDING_BOX_OVER_X1);
        int y1 = static_cast<int>(obj.y + obj.height - BOUNDING_BOX_OVER_Y1);

        if (x1 >= _window_width)
            x1 = _window_width -1;
        if (y1 >= _window_height)
            y1 = _window_height -1;
        if (x0 <= 0)
            x0 = 1;
        if (y0 <= 0)
            y0 = 1;
        
        // measure horizontally from the middle of the BB
        const int py = static_cast<int>(y1 - ((y1 - y0) / 2));

        if (py <= 0)
            continue;

        pair<int,int> from_pixel_w ( x0, py );
        pair<int,int> to_pixel_w   ( x1, py );
        // pair<int,int> from_pixel ( 451, 389 );
        // pair<int,int> to_pixel   ( 621, 382 );

        air_dist = dist_3d(depth, from_pixel_w, to_pixel_w);

        if (air_dist > 0)
        {
            //pixel depth_pixel ( x0 + ((x1 - x0)/2), py );
            //float estimatedDepth = _objMidas->postprocess(voutputData, outputShape, bytesize, depth_pixel);
            //printf("Estimated Depth: %f vs. actual depth\n", estimatedDepth, air_dist);                
            //printf("Measuring: %ix%i, %ix%i == %f\n", x0, py, x1, py, air_dist);
            
            obj.point_x0_0_actual_measured = x0;
            obj.point_y0_0_actual_measured = py;
            obj.point_x1_0_actual_measured = x1;
            obj.point_y1_0_actual_measured = py;
            obj.actual_width = int(air_dist * 100);
        }

        // start measuring veritical ruler from the middle of the BB
        const int px = static_cast<int>(  x1 - ((x1 - x0) / 2));

        if (px <= 0)
            continue;

        pair<int,int> from_pixel_h ( px, y0 );
        pair<int,int> to_pixel_h   ( px, y1 );
        air_dist = dist_3d(depth, from_pixel_h, to_pixel_h);

        if (air_dist > 0)
        {
            //printf("Measuring: %ix%i, %ix%i == %f\n", px, y0, px, y1, air_dist);
            obj.point_x0_1_actual_measured = px;
            obj.point_y0_1_actual_measured = y0;
            obj.point_x1_1_actual_measured = px;
            obj.point_y1_1_actual_measured = y1;

            obj.actual_height = int(air_dist * 100);
        }
    }
}



void run_stream(std::string camera_serial, rs2::pipeline* pipe, rs2::align* align_to, ObjectDetectionInterface* objDet)
{
    auto ttid = std::this_thread::get_id();
    std::stringstream ss;
    ss << ttid;
    std::string tid = ss.str();    

    // Create a simple OpenGL window for rendering:
    //window app(stream.width(), stream.height(), "RealSense Measure Example");    

    // Wait for all decoder streams to init...otherwise causes a segfault when OVMS loads
    // https://stackoverflow.com/questions/48271230/using-condition-variablenotify-all-to-notify-multiple-threads
    std::unique_lock<std::mutex> lk(_mtx);
    _cvAllDecodersInitd.wait(lk, [] { return _allDecodersInitd;} );
    lk.unlock();

    
    // rs2::frameset data;
    // printf("trying pipe poll\n");
    // if (pipe->poll_for_frames(&data))
    //     printf("Got frame!\n");
    // printf("No crash!!\n");
    // return 0;    

    // After initial post-processing, frames will flow into this queue:
    rs2::frame_queue postprocessed_frames;

    printf("Starting RS processing thread\n");
    // Decimation filter reduces the amount of data (while preserving best samples)
    rs2::decimation_filter dec;
    // If the demo is too slow increase the following parameter
    //  to decimate depth more (reducing quality)
    dec.set_option(RS2_OPTION_FILTER_MAGNITUDE, 2);
    // Define transformations from and to Disparity domain
    rs2::disparity_transform depth2disparity;
    rs2::disparity_transform disparity2depth(false);
    // Define spatial filter (edge-preserving)
    rs2::spatial_filter spat;
    // Enable hole-filling
    // Hole filling is an agressive heuristic and it gets the depth wrong many times
    // However, this demo is not built to handle holes
    // (the shortest-path will always prefer to "cut" through the holes since they have zero 3D distance)
    spat.set_option(RS2_OPTION_HOLES_FILL, 5); // 5 = fill all the zero pixels
    // Define temporal filter
    rs2::temporal_filter temp;

    // Video-processing thread will fetch frames from the camera,
    // apply post-processing and send the result to the main thread for rendering
    // It recieves synchronized (but not spatially aligned) pairs
    // and outputs synchronized and aligned pairs
    std::thread video_processing_thread([&]() {
        while (!shutdown_request)
        {
            // Fetch frames from the pipeline and send them for processing
            rs2::frameset data;
            if (pipe->poll_for_frames(&data))
            {
                // First make the frames spatially aligned
                if (align_to)
                    data = data.apply_filter(*align_to);
                // else
                //     continue;

                // Decimation will reduce the resultion of the depth image,
                // closing small holes and speeding-up the algorithm
                //data = data.apply_filter(dec);

                // To make sure far-away objects are filtered proportionally
                // we try to switch to disparity domain
                //data = data.apply_filter(depth2disparity);

                // Apply spatial filtering
                //data = data.apply_filter(spat);

                // Apply temporal filtering
                //data = data.apply_filter(temp);

                // If we are in disparity domain, switch back to depth
                //data = data.apply_filter(disparity2depth);

                //// Apply color map for visualization of depth
                //data = data.apply_filter(color_map);

                // Send resulting frames for visualization in the main thread
                postprocessed_frames.enqueue(data);
            }
        }
    });

    printf("Starting thread: %s\n", tid.c_str()) ;

    rs2::frameset current_frameset;
    auto initTime = std::chrono::high_resolution_clock::now();
    unsigned long numberOfFrames = 0;
    long long numberOfSkipFrames = 0;
    OVMS_Status* res = NULL;

    while (!shutdown_request) {
        auto startTime = std::chrono::high_resolution_clock::now();

        // Yolov5
        const void* voutputData1;
        size_t bytesize1 = 0;
        OVMS_DataType datatype1 = (OVMS_DataType)42;
        const int64_t* shape1{nullptr};
        size_t dimCount1 = 0;
        OVMS_BufferType bufferType1 = (OVMS_BufferType)42;
        uint32_t deviceId1 = 42;
        const char* outputName1{nullptr};

        // Midasnet
        const void* voutputData2;
        size_t bytesize2 = 0;
        OVMS_DataType datatype2 = (OVMS_DataType)42;
        const int64_t* shape2{nullptr};
        size_t dimCount2 = 0;
        OVMS_BufferType bufferType2 = (OVMS_BufferType)42;
        uint32_t deviceId2 = 42;
        const char* outputName2{nullptr};

        // Common across getoutput API
        uint32_t outputCount = 0;
        uint32_t outputCount2 = 0;
        uint32_t outputId;                

        // Inference results
        std::vector<DetectedResult> detectedResults;
        std::vector<DetectedResult> detectedResultsFiltered;

        // Fetch the latest available post-processed frameset
        postprocessed_frames.poll_for_frame(&current_frameset);

        if (!current_frameset)
        {
            std::this_thread::sleep_for(std::chrono::milliseconds(10));
            continue;
        }

        auto depth = current_frameset.get_depth_frame();
        auto color = current_frameset.get_color_frame();
        //auto colorized_depth = current_frameset.first(RS2_STREAM_DEPTH, RS2_FORMAT_RGB8);

        // glEnable(GL_BLEND);
        // // Use the Alpha channel for blending
        // glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);

        // // First render the colorized depth image
        // depth_image.render(colorized_depth, { 0, 0, app.width(), app.height() });

        // // Render the color frame (since we have selected RGBA format
        // // pixels out of FOV will appear transparent)
        // color_image.render(color, { 0, 0, app.width(), app.height() });

        // Render the simple pythagorean distance
        //render_simple_distance(depth, app_state, app);

        // Render the ruler
        // app_state.ruler_start.render(app);
        // app_state.ruler_end.render(app);

        // glColor3f(1.f, 1.f, 1.f);
        // glDisable(GL_BLEND);

        cv::Mat analytics_frame;
        cv::Mat analytics_frame2;
        cv::Mat floatImage;
        cv:Mat floatImage2;
        std::vector<int> inputShape;
        std::vector<int> inputShape2;

        cv::Mat dt = frame_to_mat(depth);
        cv::Mat img = frame_to_mat(color);

        // cout << "Color size: " << img.cols << "x" << img.rows << endl;
        // cout << "Depth size: " << dt.cols << "x" << dt.rows << endl;

        inputShape = objDet->getModelInputShape();
        inputShape2 = _objMidas->getModelInputShape();
        
        // When rendering is enabled then the input frame is resized to window size and not the needed model input size
        // TODO: Move to model post/pre-processor lib instead
        if (_render) {
            
            if (dynamic_cast<const Midasnet*>(_objMidas) != nullptr)
	        {
                resize(img, analytics_frame2, cv::Size(inputShape2[2], inputShape2[3]), 0, 0, cv::INTER_LINEAR);
                //printf("Midas resize: %ix%i\n",inputShape2[2], inputShape2[3] );
                //cv::imwrite("faceresized.jpg", analytics_frame);
		        //hwc_to_chw(analytics_frame2, analytics_frame2);
	        }
            if ( dynamic_cast<const Yolov5*>(objDet) != nullptr  || dynamic_cast<const SSD*>(objDet) != nullptr  )
            {
                resize(img, analytics_frame, cv::Size(inputShape[1], inputShape[2]), 0, 0, cv::INTER_LINEAR);
            }
            else
	        {
                printf("ERROR: Unknown model type\n");
		        return;
	        }
	        analytics_frame.convertTo(floatImage, CV_32F);
            analytics_frame2.convertTo(floatImage2, CV_32F);
        }
        else {
            //hwc_to_chw(img, analytics_frame);
            analytics_frame.convertTo(floatImage, CV_32F);
            analytics_frame2.convertTo(floatImage2, CV_32F);
        }

        const int DATA_SIZE = floatImage.step[0] * floatImage.rows;
        const int DATA_SIZE2 = floatImage2.step[0] * floatImage2.rows;

	    OVMS_InferenceResponse* response = nullptr;
        OVMS_InferenceRequest* request{nullptr};
        OVMS_InferenceResponse* response2 = nullptr;
        OVMS_InferenceRequest* request2{nullptr};

        // OD Inference
        {
            std::lock_guard<std::mutex> lock(_infMtx);

            OVMS_InferenceRequestNew(&request, _srv, objDet->getModelName(), objDet->getModelVersion());

            OVMS_InferenceRequestAddInput(
                request,
                objDet->getModelInputName(),
                OVMS_DATATYPE_FP32,
                objDet->model_input_shape,
                objDet->getModelDimCount()
            );

            // run sync request
            OVMS_InferenceRequestInputSetData(
                request,
                objDet->getModelInputName(),
                reinterpret_cast<void*>(floatImage.data),
                DATA_SIZE ,
                OVMS_BUFFERTYPE_CPU,
                0
            );

            res = OVMS_Inference(_srv, request, &response);

            if (res != nullptr) {
                std::cout << "OVMS_Inference failed " << std::endl;
                uint32_t code = 0;
                const char* details = 0;
                OVMS_StatusGetCode(res, &code);
                OVMS_StatusGetDetails(res, &details);
                std::cout << "Error occured during inference. Code:" << code
                        << ", details:" << details << std::endl;
                
                OVMS_StatusDelete(res);
                if (request)
                    OVMS_InferenceRequestDelete(request);
                break;
            }

            // Midas Inference
            OVMS_InferenceRequestNew(&request2, _srv, _objMidas->getModelName(), _objMidas->getModelVersion());

            OVMS_InferenceRequestAddInput(
                request2,
                _objMidas->getModelInputName(),
                OVMS_DATATYPE_FP32,
                _objMidas->model_input_shape,
                _objMidas->getModelDimCount()
            );

            // run sync request
            OVMS_InferenceRequestInputSetData(
                request2,
                _objMidas->getModelInputName(),
                reinterpret_cast<void*>(floatImage2.data),
                DATA_SIZE2 ,
                OVMS_BUFFERTYPE_CPU,
                0
            );

            res = OVMS_Inference(_srv, request2, &response2);

            if (res != nullptr) {
                std::cout << "OVMS_Inference Midas failed " << std::endl;
                uint32_t code = 0;
                const char* details = 0;
                OVMS_StatusGetCode(res, &code);
                OVMS_StatusGetDetails(res, &details);
                std::cout << "Error occured during inference. Code:" << code
                        << ", details:" << details << std::endl;
                
                OVMS_StatusDelete(res);
                if (request2)
                    OVMS_InferenceRequestDelete(request2);
                break;
            }
        } // end lock on inference request to server

        OVMS_InferenceResponseGetOutputCount(response, &outputCount);
        OVMS_InferenceResponseGetOutputCount(response2, &outputCount2);
        outputId = outputCount - 1; // hard-coded for first result/smallest

        // Yolo Output
        OVMS_InferenceResponseGetOutput(response, outputId, &outputName1, &datatype1, &shape1, &dimCount1, &voutputData1, &bytesize1, &bufferType1, &deviceId1);
        // std::cout << "------------>" << tid << " : " << "DeviceID " << deviceId1
        //  << ", OutputName " << outputName1
        //  << ", DimCount " << dimCount1
        //  << ", shape " << shape1[0] << " " << shape1[1] << " " << shape1[2] << " " << shape1[3]
        //  << ", byteSize " << bytesize1
        //  << ", OutputCount " << outputCount << std::endl;

        // Midasnet Output
        outputId = 0;
        OVMS_InferenceResponseGetOutput(response2, outputId, &outputName2, &datatype2, &shape2, &dimCount2, &voutputData2, &bytesize2, &bufferType2, &deviceId2);
        // std::cout << "------------>" << tid << " : " << "DeviceID " << deviceId1
        //  << ", OutputName " << outputName2
        //  << ", DimCount " << dimCount2
        //  << ", shape " << shape2[0] << " " << shape2[1] 
        //  << ", byteSize " << bytesize2
        //  << ", OutputCount " << outputCount2 << std::endl;

        objDet->postprocess(shape1, voutputData1, bytesize1, dimCount1, detectedResults);
        objDet->postprocess(detectedResults, detectedResultsFiltered);

        estimateObjectSizes(detectedResultsFiltered, depth, voutputData2, shape2, bytesize2);        

        numberOfSkipFrames++;
        float fps = 0;
        if (numberOfSkipFrames <= 120) // allow warm up for latency/fps measurements
        {
            initTime = std::chrono::high_resolution_clock::now();
            numberOfFrames = 1;

            //printf("Too early...Skipping frames..\n");
        }
        else
        {
            numberOfFrames++;

            auto endTime = std::chrono::high_resolution_clock::now();            
            auto latencyTime = ((std::chrono::duration_cast<std::chrono::milliseconds>(endTime-startTime)).count());
            auto runningLatencyTime = ((std::chrono::duration_cast<std::chrono::milliseconds>(endTime-initTime)).count());
            if (runningLatencyTime > 0) { // skip a few to account for init
                fps = (float)numberOfFrames/(float)(runningLatencyTime/1000); // convert to seconds
            }

            if (_render)
                displayGUIInferenceResults(img, detectedResultsFiltered, latencyTime, fps);                

            if (numberOfFrames % 30 == 0) {
                time_t     currTime = time(0);
                struct tm  tstruct;
                char       bCurrTime[80];
                tstruct = *localtime(&currTime);
                // http://en.cppreference.com/w/cpp/chrono/c/strftime
                strftime(bCurrTime, sizeof(bCurrTime), "%Y-%m-%d.%X", &tstruct);

                cout << detectedResultsFiltered.size() << " object(s) detected at " << bCurrTime  << endl;
                //cout << "Pipeline Throughput FPS: " << fps << endl;
                //cout << "Pipeline Latency (ms): " << chrono::duration_cast<chrono::milliseconds>(endTime - startTime).count() << endl;
            }
            
            //saveInferenceResultsAsVideo(img, detectedResultsFiltered);
        }

        if (request) {
           OVMS_InferenceRequestInputRemoveData(request, objDet->getModelInputName()); // doesn't help
           OVMS_InferenceRequestRemoveInput(request, objDet->getModelInputName());
           OVMS_InferenceRequestDelete(request);
        }

        if (response) {
           OVMS_InferenceResponseDelete(response);
        }
        
        if (shutdown_request > 0)
            break;
    } // end while get frames

    video_processing_thread.join();

    std::cout << "Goodbye..." << std::endl;

    if (res != NULL) {
        OVMS_StatusDelete(res);
        res = NULL;
    }

    if (_objMidas)
    {
        delete _objMidas;
        _objMidas = NULL;
    }

    if (objDet) {
        delete objDet;
        objDet = NULL;
    }
}

void print_usage(const char* programName) {
    // # 1 - camera serial number
    // # 2 - color_width
    // # 3 - color_height
    // # 4 - color_rate
    // # 5 - depth_width
    // # 6 - depth_height
    // # 7 - depth_rate
    // # 8 - frame_align ( 0-none 1-depth_to_color 2-color_to_depth  )
    // # 9 - enable_filters ( 0-none | 1-temporal | 2-hole_filling | 3-temporal and hole_filling )
    // # 10 - enable rendering 
    // # 11 - disable_rs_gpu_accel
    std::cout << "Usage: ./" << programName << " \n\n"
        << "camera serial number of the RealSense device\n"
        << "color_frame_width\n"
        << "color_frame_height\n"
        << "color_frame_rate\n"
        << "depth_frame_width\n"
        << "depth_frame_height\n"
        << "depth_frame_rate\n"
        << "frame_align ( 0-color_to_depth | 1-depth_to_color ) \n"
        << "enable_filters ( 0-none | 1-temporal | 2-hole_filling | 3-temporal and hole_filling )\n"
        << "disable_realsense_gpu_acceleration 0-default 1-disabled gpu acceleration\n";
}

int main(int argc, char** argv) {
    std::cout << std::setprecision(2) << std::fixed;

    _server_grpc_port = 9178;
    _server_http_port = 11338;    

    if (argc < 11) {
        print_usage(argv[0]);
        return 1;
    }

    if (!stringIsInteger(argv[2]) || !stringIsInteger(argv[3]) || !stringIsInteger(argv[4])
        || !stringIsInteger(argv[5]) || !stringIsInteger(argv[6]) || !stringIsInteger(argv[7]) 
        || !stringIsInteger(argv[8]) || !stringIsInteger(argv[9]) || !stringIsInteger(argv[10]) 
        || !stringIsInteger(argv[11])
       ) {
        printf("not a num??\n");
        print_usage(argv[0]);
        return 1;
    } else {
        std::string serial;
        if (strncmp(argv[1], "AUTO", sizeof("AUTO")) == 0 ) {
            if (!device_with_streams({ RS2_STREAM_COLOR,RS2_STREAM_DEPTH }, serial))
                return 1;
            _videoStreamPipeline = serial; //auto detected serial number
        }
        else {
            _videoStreamPipeline = argv[1]; //camera serial # from user
        }
        _color_frame_width = std::stoi(argv[2]);
        _color_frame__height = std::stoi(argv[3]);
        _color_frame_rate = std::stoi(argv[4]);
        _depth_frame_width = std::stoi(argv[5]);
        _depth_frame_height = std::stoi(argv[6]);
        _depth_frame_rate = std::stoi(argv[7]);
        _frame_align = std::stoi(argv[8]);
        _enable_filters = std::stoi(argv[9]);
        _render = std::stoi(argv[10]);
        _disable_realsense_gpu_acceleration = std::stoi(argv[11]);
        _detectorModel = 0; // use yolov5 model
    }

    std::vector<std::thread> running_streams;
    _allDecodersInitd = false;
    
    ObjectDetectionInterface* objDet;
    rs2::pipeline* pipe;
    rs2::align* align_to;
    rs2::config* cfg;

    getMAPipeline(_videoStreamPipeline, &pipe, &cfg, &align_to, &objDet);
    running_streams.emplace_back(run_stream, _videoStreamPipeline, pipe, align_to, objDet);
    //run_stream(_videoStreamPipeline, objDet);

    printf("loading ovms\n");
    if (!loadOVMS())
        return -1;

    _allDecodersInitd = true;
    _cvAllDecodersInitd.notify_all();;

    
   for(auto& running_stream : running_streams)
       running_stream.join();

    if (_srv)
        OVMS_ServerDelete(_srv);
    if (_modelsSettings)
        OVMS_ModelsSettingsDelete(_modelsSettings);
    if (_serverSettings)
        OVMS_ServerSettingsDelete(_serverSettings);

    if (pipe)
    {
        delete pipe;
        pipe = NULL;
    }

    if (cfg)
    {
        delete cfg;
        cfg = NULL;
    }

    if (align_to)
    {
        delete align_to;
        align_to = NULL;
    }

    return 0;
}
