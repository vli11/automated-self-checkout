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

// Utilized for GStramer hardware accelerated decode and pre-preprocessing
#include <gst/gst.h>
#include <gst/app/gstappsrc.h>
#include <gst/app/gstappsink.h>

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
	const char *classText;
} DetectedResult;

class MediaPipelineServiceInterface {
public:
    enum VIDEO_TYPE {
        H265,
        H264
    };

    virtual ~MediaPipelineServiceInterface() {}
    virtual const std::string getVideoDecodedPreProcessedPipeline(std::string mediaLocation, VIDEO_TYPE videoType, int video_width, int video_height, bool use_onevpl) = 0;
    virtual const std::string getBroadcastPipeline() = 0;
    virtual const std::string getRecordingPipeline() = 0;

    const std::string updateVideoDecodedPreProcessedPipeline(int video_width, int video_height, bool use_onevpl)
    {
        return getVideoDecodedPreProcessedPipeline(m_mediaLocation, m_videoType, video_width, video_height, use_onevpl);
    }

protected:
    std::string m_mediaLocation;
    VIDEO_TYPE m_videoType;
    int m_videoWidth;
    int m_videoHeight;
};

OVMS_Server* _srv;
OVMS_ServerSettings* _serverSettings = 0;
OVMS_ModelsSettings* _modelsSettings = 0;
int _server_grpc_port;
int _server_http_port;

std::string _videoStreamPipeline;
MediaPipelineServiceInterface::VIDEO_TYPE _videoType = MediaPipelineServiceInterface::VIDEO_TYPE::H264;
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

class GStreamerMediaPipelineService : public MediaPipelineServiceInterface {
public:
    const std::string getVideoDecodedPreProcessedPipeline(std::string mediaLocation, VIDEO_TYPE videoType, int video_width, int video_height, bool use_onevpl) {
        m_mediaLocation = mediaLocation;
        m_videoType = videoType;
        m_videoWidth = video_width;
        m_videoHeight = video_height;

        if (mediaLocation.find("rtsp") != std::string::npos ) {
        // video/x-raw(memory:VASurface),format=NV12
            switch (videoType)
            {
                case H264:
                if (use_onevpl)
                    return "rtspsrc location=" + mediaLocation + " ! rtph264depay ! h264parse ! " +
                    "msdkh264dec ! msdkvpp scaling-mode=lowpower ! " +
                    "video/x-raw, width=" + std::to_string(video_width) +
                    ", height=" + std::to_string(video_height) + " ! videoconvert ! video/x-raw,format=BGR ! queue ! appsink drop=1 sync=0";
                else
                    return "rtspsrc location=" + mediaLocation + " ! rtph264depay ! vaapidecodebin ! video/x-raw(memory:VASurface),format=NV12 ! vaapipostproc" +
                    " width=" + std::to_string(video_width) +
                    " height=" + std::to_string(video_height) +
                    " scale-method=fast ! videoconvert ! video/x-raw,format=BGR ! queue ! appsink drop=1 sync=0";
                case H265:
                if (use_onevpl)
                    return "rtspsrc location=" + mediaLocation + " ! rtph265depay ! h265parse ! " +
                    "msdkh265dec ! " +
                    "msdkvpp scaling-mode=lowpower ! " +
                    "video/x-raw, width=" + std::to_string(video_width) +
                    ", height=" + std::to_string(video_height) + " ! videoconvert ! video/x-raw,format=BGR ! queue ! appsink drop=1 sync=0";
                else
                    return "rtspsrc location=" + mediaLocation + " ! rtph265depay ! vaapidecodebin ! vaapipostproc" +
                    " width=" + std::to_string(video_width) +
                    " height=" + std::to_string(video_height) +
                    " scale-method=fast ! videoconvert ! video/x-raw,format=BGR ! appsink sync=0 drop=1";
                default:
                    std::cout << "Video type not supported!" << std::endl;
                    return "";
            }
        }
        else if (mediaLocation.find(".mp4") != std::string::npos ) {
            switch (videoType)
            {
                case H264:
                if (use_onevpl)
                    return "filesrc location=" + mediaLocation + " ! qtdemux ! h264parse ! " +
                    "msdkh264dec ! msdkvpp scaling-mode=lowpower ! " +
                    "video/x-raw, width=" + std::to_string(video_width) + ", height=" + std::to_string(video_height) + 
                    " ! videoconvert ! video/x-raw,format=BGR ! queue ! appsink drop=1 sync=0";
                else
                    return "filesrc location=" + mediaLocation + " ! qtdemux ! h264parse ! vaapidecodebin ! vaapipostproc" +
                    " width=" + std::to_string(video_width) +
                    " height=" + std::to_string(video_height) +
                    " scale-method=fast ! videoconvert ! video/x-raw,format=BGR ! appsink drop=1";
                case H265:
                if (use_onevpl)
                    return "filesrc location=" + mediaLocation + " ! qtdemux ! h265parse ! " +
                    "msdkh265dec ! msdkvpp scaling-mode=lowpower ! " +
                    " video/x-raw, width=" + std::to_string(video_width) + ", height=" + std::to_string(video_height) +
                    " ! videoconvert ! video/x-raw,format=BGR ! queue ! appsink drop=1 sync=0";
                else
                    return "filesrc location=" + mediaLocation + " ! qtdemux ! h265parse ! vaapidecodebin ! vaapipostproc" +
                    " width=" + std::to_string(video_width) +
                    " height=" + std::to_string(video_height) +
                    " scale-method=fast ! videoconvert ! video/x-raw,format=BGR ! appsink drop=1";
                default:
                    std::cout << "Video type not supported!" << std::endl;
                    return "";
            }
        }
        else {
            std::cout << "Unknown media source specified " << mediaLocation << " !!" << std::endl;
            return "";
        }
    }

    const std::string getBroadcastPipeline() {
        // TODO: Not implemented
        return "videotestsrc ! videoconvert,format=BGR ! video/x-raw ! appsink drop=1";
    }

    const std::string getRecordingPipeline() {
        // TODO: Not implemented
        return "videotestsrc ! videoconvert,format=BGR ! video/x-raw ! appsink drop=1";
    }
protected:

};

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
    virtual void postprocess(const int64_t* output_shape, const void* voutputData, const size_t bytesize, const uint32_t dimCount, std::vector<DetectedResult> &detectedResults) = 0;

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
    } // end postprocess filter


protected:
    float confidence_threshold = .9;
    float boxiou_threshold = .4;
    float iou_threshold = 0.4;
    int classes =  80;
    bool useAdvancedPostprocessing = false;

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
                obj.classText = getClassLabelText(obj.classId).c_str();
                
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
                            obj.classText = getClassLabelText(obj.classId).c_str();
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

        auto postprocessRawData = sigmoid; //sigmoid or linear

        for (int i = 0; i < entriesNum; ++i) {
            int row = i / sideW;
            int col = i % sideW;

            for (int n = 0; n < regionNum; ++n) {

                int obj_index = calculateEntryIndex(entriesNum,  regionCoordsCount, classes + 1 /* + confidence byte */, n * entriesNum + i,regionCoordsCount);
                int box_index = calculateEntryIndex(entriesNum, regionCoordsCount, classes + 1, n * entriesNum + i, 0);
                float outdata = outData[obj_index];
                float scale = postprocessRawData(outData[obj_index]);

                if (scale >= confidence_threshold) {
                    float x, y,height,width;
                    x = static_cast<float>((col + postprocessRawData(outData[box_index + 0 * entriesNum])) / sideW * original_im_w);
                    y = static_cast<float>((row + postprocessRawData(outData[box_index + 1 * entriesNum])) / sideH * original_im_h);
                    height = static_cast<float>(std::pow(2*postprocessRawData(outData[box_index + 3 * entriesNum]),2) * anchors_13[2 * n + 1] * original_im_h / scaleH  );
                    width = static_cast<float>(std::pow(2*postprocessRawData(outData[box_index + 2 * entriesNum]),2) * anchors_13[2 * n] * original_im_w / scaleW  );

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
                            obj.classText = getClassLabelText(j).c_str();
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
                obj.classText = getClassLabelText(obj.classId).c_str();
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

GStreamerMediaPipelineService* _mediaService = NULL;
std::string _user_request;

namespace {
volatile sig_atomic_t shutdown_request = 0;
}

bool stringIsInteger(std::string strInput) {
    std::string::const_iterator it = strInput.begin();
    while (it != strInput.end() && std::isdigit(*it)) ++it;
    return !strInput.empty() && it == strInput.end();
}

bool setActiveModel(int detectionType, ObjectDetectionInterface** objDet)
{
    if (objDet == NULL)
        return false;

    if (detectionType == 0) {
        *objDet = new Yolov5();
    }
    else if(detectionType == 1) {
        *objDet = new SSD();
    }
    else if(detectionType == 2) {
        *objDet = new FaceDetection0005();
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
    } // end for

    //latency
    std::string fps_msg = (througput == 0) ? "..." : std::to_string(througput) + "fps";
    std::string latency_msg = (latency == 0) ? "..." :  std::to_string(latency) + "ms";
    std::string roiCount_msg = std::to_string(results.size());
    std::string message = "E2E Pipeline Performance: " + latency_msg + " and " + fps_msg + " with ROIs#" + roiCount_msg;
    cv::putText(analytics_frame, message.c_str(), cv::Size(0, 20), cv::FONT_HERSHEY_PLAIN, 1, (255, 0, 0), 1, cv::LINE_4);
    cv::putText(analytics_frame, tid, cv::Size(0, 40), cv::FONT_HERSHEY_PLAIN, 1, (255, 0, 0), 1, cv::LINE_4);

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
std::string getVideoPipelineText(std::string mediaPath, ObjectDetectionInterface* objDet, ObjectDetectionInterface* textDet)
{

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

    return _mediaService->getVideoDecodedPreProcessedPipeline(
        mediaPath,
        _videoType,
        frame_width,
        frame_height,
        _use_onevpl);
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

bool loadGStreamer(GstElement** pipeline,  GstElement** appsink, std::string mediaPath, ObjectDetectionInterface* _objDet)
{
    static int threadCnt = 0;

    std::string videoPipelineText = getVideoPipelineText(mediaPath, _objDet, NULL);
    std::cout << "--------------------------------------------------------------" << std::endl;
    std::cout << "Opening Media Pipeline: " << videoPipelineText << std::endl;
    std::cout << "--------------------------------------------------------------" << std::endl;

    *pipeline = gst_parse_launch (videoPipelineText.c_str(), NULL);
    if (*pipeline == NULL) {
        std::cout << "ERROR: Failed to parse GST pipeline. Quitting." << std::endl;
        return false;
    }

    std::string appsinkName = "appsink" + std::to_string(threadCnt++);

    *appsink = gst_bin_get_by_name (GST_BIN (*pipeline), appsinkName.c_str());

    // Check if all elements were created
    if (!(*appsink))
    {
        printf("ERROR: Failed to initialize GST pipeline (missing %s) Quitting.\n", appsinkName.c_str());
        return false;
    }

    GstStateChangeReturn gst_res;

    // Start pipeline so it could process incoming data
    gst_res = gst_element_set_state(*pipeline, GST_STATE_PLAYING);

    if (gst_res != GST_STATE_CHANGE_SUCCESS && gst_res != GST_STATE_CHANGE_ASYNC  ) {
        printf("ERROR: StateChange not successful. Error Code: %d\n", gst_res);
        return false;
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

bool getMAPipeline(std::string mediaPath, GstElement** pipeline,  GstElement** appsink, ObjectDetectionInterface** _objDet)
{
    if (!setActiveModel(_detectorModel, _objDet)) {
        std::cout << "Unable to set active detection model" << std::endl;
        return false;
    }

    return loadGStreamer(pipeline, appsink, mediaPath, *_objDet);
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

void run_stream(std::string mediaPath, GstElement* pipeline, GstElement* appsink, ObjectDetectionInterface* objDet)
{
    auto ttid = std::this_thread::get_id();
    std::stringstream ss;
    ss << ttid;
    std::string tid = ss.str();

    // Wait for all decoder streams to init...otherwise causes a segfault when OVMS loads
    // https://stackoverflow.com/questions/48271230/using-condition-variablenotify-all-to-notify-multiple-threads
    std::unique_lock<std::mutex> lk(_mtx);
    _cvAllDecodersInitd.wait(lk, [] { return _allDecodersInitd;} );
    lk.unlock();

    printf("Starting thread: %s\n", tid.c_str()) ;

    auto initTime = std::chrono::high_resolution_clock::now();
    unsigned long numberOfFrames = 0;
    long long numberOfSkipFrames = 0;
    OVMS_Status* res = NULL;

    while (!shutdown_request) {
        auto startTime = std::chrono::high_resolution_clock::now();

        const void* voutputData1;
        size_t bytesize1 = 0;
        uint32_t outputCount = 0;
        uint32_t outputId;
        OVMS_DataType datatype1 = (OVMS_DataType)42;
        const int64_t* shape1{nullptr};
        size_t dimCount1 = 0;
        OVMS_BufferType bufferType1 = (OVMS_BufferType)42;
        uint32_t deviceId1 = 42;
        const char* outputName1{nullptr};

        GstSample *sample;
        GstStructure *s;
        GstBuffer *buffer;
        GstMapInfo m;

        std::vector<DetectedResult> detectedResults;
        std::vector<DetectedResult> detectedResultsFiltered;

        if (gst_app_sink_is_eos(GST_APP_SINK(appsink))) {
            std::cout << "INFO: EOS " << std::endl;
            return;
        }

        sample = gst_app_sink_try_pull_sample (GST_APP_SINK(appsink), 5 * GST_SECOND);

        if (sample == nullptr) {
            std::cout << "ERROR: No sample found" << std::endl;
            return;
        }

        GstCaps *caps;
        caps = gst_sample_get_caps(sample);

        if (caps == nullptr) {
            std::cout << "ERROR: No caps found for sample" << std::endl;
            return;
        }

        s = gst_caps_get_structure(caps, 0);
        gst_structure_get_int(s, "width", &_video_input_width);
        gst_structure_get_int(s, "height", &_video_input_height);

        buffer = gst_sample_get_buffer(sample);
        gst_buffer_map(buffer, &m, GST_MAP_READ);

        if (m.size <= 0) {
            std::cout << "ERROR: Invalid buffer size" << std::endl;
            return;
        }

        cv::Mat analytics_frame;
        cv::Mat floatImage;
        std::vector<int> inputShape;

        inputShape = objDet->getModelInputShape();

        cv::Mat img(_video_input_height, _video_input_width, CV_8UC3, (void *) m.data);

        // When rendering is enabled then the input frame is resized to window size and not the needed model input size
        // TODO: Move to model post/pre-processor lib instead
        if (_render) {
            
            if (dynamic_cast<const FaceDetection0005*>(objDet) != nullptr)
	        {
                resize(img, analytics_frame, cv::Size(inputShape[2], inputShape[3]), 0, 0, cv::INTER_LINEAR);
                //cv::imwrite("faceresized.jpg", analytics_frame);
		        hwc_to_chw(analytics_frame, analytics_frame);
	        }
            else
	        {
                printf("ERROR: Unknown model type\n");
		        return;
	        }
	        analytics_frame.convertTo(floatImage, CV_32F);
        }
        else {
            hwc_to_chw(img, analytics_frame);
            analytics_frame.convertTo(floatImage, CV_32F);
        }

        const int DATA_SIZE = floatImage.step[0] * floatImage.rows;

	    OVMS_InferenceResponse* response = nullptr;
        OVMS_InferenceRequest* request{nullptr};

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
        } // end lock on inference request to server

        OVMS_InferenceResponseGetOutputCount(response, &outputCount);
        outputId = outputCount - 1;

        OVMS_InferenceResponseGetOutput(response, outputId, &outputName1, &datatype1, &shape1, &dimCount1, &voutputData1, &bytesize1, &bufferType1, &deviceId1);
        // std::cout << "------------>" << tid << " : " << "DeviceID " << deviceId1
        //  << ", OutputName " << outputName1
        //  << ", DimCount " << dimCount1
        //  << ", shape " << shape1[0] << " " << shape1[1] << " " << shape1[2] << " " << shape1[3]
        //  << ", byteSize " << bytesize1
        //  << ", OutputCount " << outputCount << std::endl;

        objDet->postprocess(shape1, voutputData1, bytesize1, dimCount1, detectedResults);
        objDet->postprocess(detectedResults, detectedResultsFiltered);

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

        gst_buffer_unmap(buffer, &m);
        gst_sample_unref(sample);

        if (shutdown_request > 0)
            break;
    } // end while get frames

    std::cout << "Goodbye..." << std::endl;

    if (res != NULL) {
        OVMS_StatusDelete(res);
        res = NULL;
    }

    if (objDet) {
        delete objDet;
        objDet = NULL;
    }

    gst_element_set_state (pipeline, GST_STATE_NULL);
    if (pipeline)
        gst_object_unref(pipeline);

    if (appsink)
        gst_object_unref(appsink);
}

void print_usage(const char* programName) {
    std::cout << "Usage: ./" << programName << " \n\n"
        << "mediaLocation is an rtsp://127.0.0.1:8554/camera_0 url or a path to an *.mp4 file\n"
        << "use_onevpl is 0 (libva - default) or 1 for onevpl\n"
        << "render is 1 to launch render window or 0 (default) for headless\n"
        << "video_type is 0 for AVC or 1 for HEVC\n";
}


int main(int argc, char** argv) {
    std::cout << std::setprecision(2) << std::fixed;

    // Use GST pipelines for media HWA decode and pre-procesing
    _mediaService = new GStreamerMediaPipelineService();

    _server_grpc_port = 9178;
    _server_http_port = 11338;

    _videoStreamPipeline = "people-detection.mp4";

    if (argc < 3) {
        print_usage(argv[0]);
        return 1;
    }

    if (!stringIsInteger(argv[2]) || !stringIsInteger(argv[3]) || !stringIsInteger(argv[4])
        || !stringIsInteger(argv[5])) {
        print_usage(argv[0]);
        return 1;
    } else {
        _videoStreamPipeline = argv[1];
        _use_onevpl = std::stoi(argv[2]);
        _render = std::stoi(argv[3]);
        _renderPortrait = std::stoi(argv[4]);
        _videoType = (MediaPipelineServiceInterface::VIDEO_TYPE) std::stoi(argv[5]);
        _detectorModel = 2; // use geti-yolox model

        if (_renderPortrait) {
            int tmp = _window_width;
            _window_width = _window_height;
            _window_height = tmp;
        }
    }

    gst_init(NULL, NULL);

    std::vector<std::thread> running_streams;
    _allDecodersInitd = false;
    
    GstElement *pipeline;
    GstElement *appsink;
    ObjectDetectionInterface* objDet;
    getMAPipeline(_videoStreamPipeline, &pipeline,  &appsink, &objDet);
    running_streams.emplace_back(run_stream, _videoStreamPipeline, pipeline, appsink, objDet);

    // GstElement *pipeline2;
    // GstElement *appsink2;
    // ObjectDetectionInterface* objDet2;
    // _videoStreamPipeline = "rtsp://127.0.0.1:8554/camera_2";
    // getMAPipeline(_videoStreamPipeline, &pipeline2,  &appsink2, &objDet2, &textDet2);
    // running_streams.emplace_back(run_stream, _videoStreamPipeline, pipeline2, appsink2, objDet2, textDet2);

    if (!loadOVMS())
        return -1;

    _allDecodersInitd = true;
    _cvAllDecodersInitd.notify_all();;
    

   for(auto& running_stream : running_streams)
       running_stream.join();

    if (_mediaService != NULL) {
        delete _mediaService;
        _mediaService = NULL;
    }

    if (_srv)
        OVMS_ServerDelete(_srv);
    if (_modelsSettings)
        OVMS_ModelsSettingsDelete(_modelsSettings);
    if (_serverSettings)
        OVMS_ServerSettingsDelete(_serverSettings);

    return 0;
}
