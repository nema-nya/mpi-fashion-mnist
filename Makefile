NAME ?= rvs
BUILD ?= release

CC ?= cc

SRC_DIR := src
BUILD_DIR := build/$(BUILD)

SRCS := $(wildcard $(SRC_DIR)/*.c)
OBJS := $(patsubst $(SRC_DIR)/%.c,$(BUILD_DIR)/%.o,$(SRCS))
DEPS := $(OBJS:.o=.d)

CPPFLAGS ?=
CPPFLAGS += -I$(SRC_DIR)

CFLAGS ?= -std=c23 -Wall -Wextra -Wpedantic
LDFLAGS ?=
LDLIBS ?=
LDLIBS += -lm

ifeq ($(BUILD),debug)
	CFLAGS += -O0 -g3
else ifeq ($(BUILD),release)
	CFLAGS += -O2
else
	$(error Unknown BUILD '$(BUILD)' (use BUILD=debug or BUILD=release))
endif

BIN := $(BUILD_DIR)/$(NAME)

.PHONY: all clean debug release run help

all: $(BIN)

debug:
	$(MAKE) BUILD=debug all

release:
	$(MAKE) BUILD=release all

$(BIN): $(OBJS)
	$(CC) $(LDFLAGS) $^ $(LDLIBS) -o $@

$(BUILD_DIR)/%.o: $(SRC_DIR)/%.c | $(BUILD_DIR)
	$(CC) $(CPPFLAGS) $(CFLAGS) -MMD -MP -c $< -o $@

$(BUILD_DIR):
	mkdir -p $@

-include $(DEPS)

run: all
	./$(BIN)

clean:
	rm -rf build

help:
	@printf '%s\n' \
	  'Targets:' \
	  '  all (default)   Build $(BIN)' \
	  '  debug           Build debug (BUILD=debug)' \
	  '  release         Build release (BUILD=release)' \
	  '  run             Build & run $(BIN)' \
	  '  clean           Remove build artifacts' \
	  '' \
	  'Variables:' \
	  '  NAME=<name>     Output binary name (default: rvs)' \
	  '  CC=<compiler>   C compiler (default: cc)' \
	  '  CFLAGS=...      Extra compiler flags' \
	  '  CPPFLAGS=...    Extra preprocessor flags' \
	  '  LDFLAGS=...     Extra linker flags' \
	  '  LDLIBS=...      Extra linker libs'
