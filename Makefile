.PHONY: compile rebuild clean gen run

# CC := /opt/homebrew/bin/gcc-15
# CXX := /opt/homebrew/bin/g++-15

CC := /usr/bin/clang 
CXX := /usr/bin/clang++

BUILD_TYPE ?= Release
BUILD_DIR := csrc/build
DATA_DIR := data

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

test:
	@pytest py/tests -v
	
gen: 
	@rm -rf csrc/gen
	@rm -rf py/gen
	@./scripts/codegen.bash
	@echo "⚠️  Run 'make rebuild' to recompile with new schema"

clean:
	rm -rf $(BUILD_DIR) 
	rm -f $(DATA_DIR)/*.bin

