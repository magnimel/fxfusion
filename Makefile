.PHONY: compile rebuild clean run

# CC := /opt/homebrew/bin/gcc-15
# CXX := /opt/homebrew/bin/g++-15

CC := /usr/bin/clang 
CXX := /usr/bin/clang++

# -DCMAKE_PREFIX_PATH=`python3 -c 'import torch;print(torch.utils.cmake_prefix_path)'` 

BUILD_TYPE ?= Release
BUILD_DIR := src/build

$(BUILD_DIR)/.configured: CMakeLists.txt
	@CC=$(CC) CXX=$(CXX) cmake \
		-S . \
		-B $(BUILD_DIR) \
		-DCMAKE_BUILD_TYPE=$(BUILD_TYPE) \

	@touch $(BUILD_DIR)/.configured

compile: $(BUILD_DIR)/.configured
	@cmake --build $(BUILD_DIR)

rebuild: clean compile

run:
	./$(BUILD_DIR)/engine

gen: 
	./scripts/codegen.bash

clean:
	rm -rf $(BUILD_DIR)

