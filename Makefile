# Copyright © 2023 Intel Corporation. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

.PHONY: build-all build-soc build-dgpu build-grpc-python build-grpc-go build-python-apps build-telegraf
.PHONY: run-camera-simulator run-telegraf
.PHONY: clean-grpc-go clean-segmentation clean-ovms-server clean-ovms clean-all clean-results clean-telegraf clean-models
.PHONY: clean clean-simulator clean-object-detection clean-classification clean-gst
.PHONY: list-profiles
.PHONY: unit-test-profile-launcher build-profile-launcher profile-launcher-status clean-profile-launcher
.PHONY: build-ovms-gst-client

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

build-telegraf:
	cd telegraf && $(MAKE) build

run-camera-simulator:
	./camera-simulator/camera-simulator.sh

run-telegraf:
	cd telegraf && $(MAKE) run

clean:
	./clean-containers.sh automated-self-checkout

clean-simulator:
	./clean-containers.sh camera-simulator

build-profile-launcher:
	@cd ./configs/opencv-ovms/cmd_client && $(MAKE) build
	@./create-symbolic-link.sh $(PWD)/configs/opencv-ovms/cmd_client/profile-launcher profile-launcher
	@./create-symbolic-link.sh $(PWD)/configs/opencv-ovms/scripts scripts
	@./create-symbolic-link.sh $(PWD)/configs/opencv-ovms/envs envs
	@./create-symbolic-link.sh $(PWD)/benchmark-scripts/stream_density.sh stream_density.sh

build-ovms-server:
	HTTPS_PROXY=${HTTPS_PROXY} HTTP_PROXY=${HTTP_PROXY} docker pull openvino/model_server:2023.1-gpu
	sudo docker build --build-arg HTTPS_PROXY=${HTTPS_PROXY} --build-arg HTTP_PROXY=${HTTP_PROXY} -f configs/opencv-ovms/models/2022/Dockerfile.updateDevice -t update_config:dev configs/opencv-ovms/models/2022/.

clean-profile-launcher: clean-grpc-python clean-grpc-go clean-segmentation clean-object-detection clean-classification clean-gst
	@echo "containers launched by profile-launcher are cleaned up."
	@pkill profile-launcher || true

profile-launcher-status:
	$(eval profileLauncherPid = $(shell ps -aux | grep ./profile-launcher | grep -v grep))
	$(if $(strip $(profileLauncherPid)), @echo "$@: profile-launcher running: "$(profileLauncherPid), @echo "$@: profile laucnher stopped")

clean-grpc-python:
	./clean-containers.sh grpc_python

clean-grpc-go:
	./clean-containers.sh grpc_go

clean-gst:
	./clean-containers.sh gst

clean-segmentation:
	./clean-containers.sh segmentation

clean-object-detection:
	./clean-containers.sh object-detection

clean-classification:
	./clean-containers.sh classification

clean-ovms-server:
	./clean-containers.sh ovms-server

clean-ovms: clean-profile-launcher clean-ovms-server

clean-telegraf: 
	./clean-containers.sh influxdb2
	./clean-containers.sh telegraf

clean-all: clean clean-ovms clean-simulator clean-results clean-telegraf

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
		-u $(shell id -u):$(shell id -g) \
		-v $(PWD):/docs \
		-w /docs \
		$(MKDOCS_IMAGE) \
		build

serve-docs: docs-builder-image
	docker run --rm \
		-it \
		-u $(shell id -u):$(shell id -g) \
		-p 8008:8000 \
		-v $(PWD):/docs \
		-w /docs \
		$(MKDOCS_IMAGE)

build-grpc-python: build-profile-launcher
	cd configs/opencv-ovms/grpc_python && $(MAKE) build

build-grpc-go: build-profile-launcher
	cd configs/opencv-ovms/grpc_go && $(MAKE) build

build-python-apps: build-profile-launcher
	cd configs/opencv-ovms/demos && make build	

clean-docs:
	rm -rf docs/

clean-results:
	sudo rm -rf results/*

list-profiles:
	@echo "Here is the list of profile names, you may choose to use one of them for pipeline run script:"
	@echo
	@find ./configs/opencv-ovms/cmd_client/res/ -mindepth 1 -maxdepth 1 -type d -exec basename {} \;
	@echo
	@echo "Example: "
	@echo "PIPELINE_PROFILE=\"grpc_python\" sudo -E ./run.sh --workload ovms --platform core --inputsrc rtsp://127.0.0.1:8554/camera_0"

clean-models:
	@find ./configs/opencv-ovms/models/2022/ -mindepth 1 -maxdepth 1 -type d -exec sudo rm -r {} \;

unit-test-profile-launcher:
	@cd ./configs/opencv-ovms/cmd_client && $(MAKE) unit-test

build-ovms-gst-client: build-ovms-server-22.04
	echo "Building GStreamer OVMS C-API Client HTTPS_PROXY=${HTTPS_PROXY} HTTP_PROXY=${HTTP_PROXY}"
	# Build C-API for optimized distributed architecture. Includes GST for HWA media	
	rm -vrf ovms.tar.gz 
	wget -O ovms.tar.gz https://github.com/openvinotoolkit/model_server/releases/download/v2023.0/ovms_ubuntu22.tar.gz

	# # Build CAPI docker image
	docker build $(NO_CACHE_OPTION) -f Dockerfile.ovms-capi-gst . \
		--build-arg http_proxy=$(HTTP_PROXY) \
		--build-arg https_proxy="$(HTTPS_PROXY)" \
		--build-arg no_proxy=$(NO_PROXY) \
		--build-arg BASE_IMAGE=ubuntu:22.04 \
		--progress=plain \
		-t $(OVMS_CPP_DOCKER_IMAGE)-capi-gst:$(OVMS_CPP_IMAGE_TAG)

build-ovms-server-22.04:
	@echo "Building for OVMS Server Ubuntu 22.04 (Recommended) HTTPS_PROXY=${HTTPS_PROXY} HTTP_PROXY=${HTTP_PROXY}"
	# Pull docker images for grpc/kserv distributed architecture
	# Ubuntu 22.04 support with CPU/iGPU/dGPU 
	docker pull openvino/model_server:2023.0-gpu