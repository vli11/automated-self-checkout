#
# Copyright (c) 2022 Intel Corporation
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#

CV_LIBS = -L/usr/lib/x86_64-linux-gnu/ -L/usr/local/lib/x86_64-linux-gnu/ -L/ovms/lib/
CV_INCLUDES = -I/usr/lib/include/opencv4
RS_INCLUDES = -I/rs/librealsense -I/rs/librealsense/examples/measure/.. -I/rs/librealsense/include -I/rs/librealsense/third-party/rsutils/include
RS_FLAGS = -pedantic -g -Wno-missing-field-initializers -Wno-switch -Wno-multichar -Wsequence-point -Wformat -Wformat-security -mssse3 -pthread -O3 -DNDEBUG -fPIE -std=c++17
build: 
	g++ main.cpp $(RS_INCLUDES) -I/usr/include/glib-2.0 -I/usr/lib/x86_64-linux-gnu/glib-2.0/include -I/ovms/include $(CV_INCLUDES) $(CV_LIBS) -lgobject-2.0 -lglib-2.0 -lopencv_imgcodecs -lopencv_imgproc -lopencv_core -lovms_shared -lopencv_highgui -Wl,-rpath,/rs/librealsense/build: /usr/local/lib/librealsense2.so.2.54.1 /usr/lib/x86_64-linux-gnu/libglfw.so.3.3 -lOpenGL -lGLX -lGLU /usr/local/lib/librsutils.a -pedantic -g -Wno-missing-field-initializers -Wno-switch -Wno-multichar -Wsequence-point -Wformat -Wformat-security -mssse3 -lpthread -fPIC --std=c++17 $(RS_FLAGS) -o object_size_estimation_yolov5

