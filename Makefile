# Copyright © 2023 Intel Corporation. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

.PHONY: build-all build-soc build-dgpu run-camera-simulator clean clean-simulator clean-ovms-client clean-grpc-go clean-model-server clean-ovms clean-all build-grpc-go clean-results

MKDOCS_IMAGE ?= asc-mkdocs
BASE_OS_TAG_UBUNTU ?= 22.04
OVMS_CPP_DOCKER_IMAGE ?= openvino/model_server
OVMS_CPP_IMAGE_TAG ?= latest
BASE_OS ?= ubuntu
DIST_OS ?= $(BASE_OS)

build-all: build-soc build-dgpu

build-soc:
	echo "Building for SOC (e.g. TGL/ADL/Xeon SP/etc) HTTPS_PROXY=${HTTPS_PROXY} HTTP_PROXY=${HTTP_PROXY}"
	docker build --no-cache --build-arg HTTPS_PROXY=${HTTPS_PROXY} --build-arg HTTP_PROXY=${HTTP_PROXY} -t sco-soc:2.0 -f Dockerfile.soc .

build-dgpu:
	echo "Building for dgpu Arc/Flex HTTPS_PROXY=${HTTPS_PROXY} HTTP_PROXY=${HTTP_PROXY}"
	docker build --no-cache --build-arg HTTPS_PROXY=${HTTPS_PROXY} --build-arg HTTP_PROXY=${HTTP_PROXY} -t sco-dgpu:2.0 -f Dockerfile.dgpu .

run-camera-simulator:
	./camera-simulator/camera-simulator.sh

clean:
	./clean-containers.sh automated-self-checkout

clean-simulator:
	./clean-containers.sh camera-simulator

build-ovms-realsense-client:
	echo "Building RealSense OVMS C-API Client HTTPS_PROXY=${HTTPS_PROXY} HTTP_PROXY=${HTTP_PROXY}"
	rm -vrf ovms.tar.gz 
	docker run $(OVMS_CPP_DOCKER_IMAGE)-pkg:$(OVMS_CPP_IMAGE_TAG) bash -c \
      "tar -c -C / ovms.tar.gz ; sleep 2" | tar -x
	
	# # Remove old docker image
	-docker rm -v $$(docker ps -a -q -f status=exited -f ancestor=$(OVMS_CPP_DOCKER_IMAGE)-pkg:$(OVMS_CPP_IMAGE_TAG))
	
	# # Build CAPI docker image
	# cp -vR capi_files/* capi/$(DIST_OS)/
	# cp *.mp4 capi/$(DIST_OS)/demos
	# cp -vR configs capi/$(DIST_OS)/demos
	# cp -vR model_repo capi/$(DIST_OS)/demos
	docker build $(NO_CACHE_OPTION) -f Dockerfile.ovms-capi-realsense . \
		--build-arg http_proxy=$(HTTP_PROXY) \
		--build-arg https_proxy="$(HTTPS_PROXY)" \
		--build-arg no_proxy=$(NO_PROXY) \
		--build-arg BASE_IMAGE=ubuntu:22.04 \
		--progress=plain \
		-t $(OVMS_CPP_DOCKER_IMAGE)-capi-realsense:$(OVMS_CPP_IMAGE_TAG)

build-ovms-gst-client:
	echo "Building GStreamer OVMS C-API Client HTTPS_PROXY=${HTTPS_PROXY} HTTP_PROXY=${HTTP_PROXY}"
	# Build C-API for optimized distributed architecture. Includes GST for HWA media	
	rm -vrf ovms.tar.gz 
	docker run $(OVMS_CPP_DOCKER_IMAGE)-pkg:$(OVMS_CPP_IMAGE_TAG) bash -c \
      "tar -c -C / ovms.tar.gz ; sleep 2" | tar -x
	
	# # Remove old docker image
	-docker rm -v $$(docker ps -a -q -f status=exited -f ancestor=$(OVMS_CPP_DOCKER_IMAGE)-pkg:$(OVMS_CPP_IMAGE_TAG))
	
	# # Build CAPI docker image
	# cp -vR capi_files/* capi/$(DIST_OS)/
	# cp *.mp4 capi/$(DIST_OS)/demos
	# cp -vR configs capi/$(DIST_OS)/demos
	# cp -vR model_repo capi/$(DIST_OS)/demos
	docker build $(NO_CACHE_OPTION) -f Dockerfile.ovms-capi-gst . \
		--build-arg http_proxy=$(HTTP_PROXY) \
		--build-arg https_proxy="$(HTTPS_PROXY)" \
		--build-arg no_proxy=$(NO_PROXY) \
		--build-arg BASE_IMAGE=ubuntu:22.04 \
		--progress=plain \
		-t $(OVMS_CPP_DOCKER_IMAGE)-capi-gst:$(OVMS_CPP_IMAGE_TAG)

build-ovms-grpc-client:
	echo "Building for OVMS gRPC/HTTP Client HTTPS_PROXY=${HTTPS_PROXY} HTTP_PROXY=${HTTP_PROXY}"
	docker build --build-arg HTTPS_PROXY=${HTTPS_PROXY} --build-arg HTTP_PROXY=${HTTP_PROXY} -t ovms-client:latest -f Dockerfile.ovms-client .

build-ovms-server-22.04:
	@echo "Building for OVMS Server Ubuntu 22.04 (Recommended) HTTPS_PROXY=${HTTPS_PROXY} HTTP_PROXY=${HTTP_PROXY}"
	# Pull docker images for grpc/kserv distributed architecture
	# Ubuntu 22.04 support with CPU/iGPU/dGPU 
	docker pull openvino/model_server:2023.0-gpu

build-ovms-server-20.04:
	@echo "Building for OVMS Server Ubuntu 20.04 HTTPS_PROXY=${HTTPS_PROXY} HTTP_PROXY=${HTTP_PROXY}"
	# Pull docker images for grpc/kserv distributed architecture
	# Ubuntu 20.04 support with CPU/iGPU
	docker pull openvino/model_server:2022.3.0.1-gpu

clean-ovms-client: clean-grpc-go
	./clean-containers.sh ovms-client

clean-grpc-go:
	./clean-containers.sh dev

clean-model-server:
	./clean-containers.sh model-server

clean-ovms: clean-ovms-client clean-model-server

clean-all: clean clean-ovms clean-simulator clean-results

docs: clean-docs
	mkdocs build
	mkdocs serve -a localhost:8008

docs-builder-image:
	docker build \
		-f Dockerfile.docs \
		-t $(MKDOCS_IMAGE) \
		.

build-docs: docs-builder-image
	docker run --rm \
		-v $(PWD):/docs \
		-w /docs \
		$(MKDOCS_IMAGE) \
		build

serve-docs: docs-builder-image
	docker run --rm \
		-it \
		-p 8008:8000 \
		-v $(PWD):/docs \
		-w /docs \
		$(MKDOCS_IMAGE)

build-grpc-go: build-ovms-grpc-client
	cd configs/opencv-ovms/grpc_go && make build

clean-docs:
	rm -rf docs/

clean-results:
	sudo rm -rf results/*
