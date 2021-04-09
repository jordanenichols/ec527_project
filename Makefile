TARGET_EXEC ?= a.out

CC = g++
BUILD_DIR ?= ./build
SRC_DIRS ?= ./src
LIBS := -lm 
SRCS := $(shell find $(SRC_DIRS) -name *.cpp -or -name *.c -or -name *.s)
OBJS := $(SRCS:%=$(BUILD_DIR)/%.o)
DEPS := $(OBJS:.o=.d)

INC_DIRS := $(shell find $(SRC_DIRS) -type d)
INC_FLAGS := $(addprefix -I,$(INC_DIRS))

CFLAGS = -g -Wall 


$(BUILD_DIR)/$(TARGET_EXEC): $(OBJS)
		$(CC) $(OBJS) $(LIBS) -o $@ $(LDFLAGS)

$(BUILD_DIR)/%.cpp.o: %.cpp
		$(MKDIR_P) $(dir $@)
		$(CC) $(CFLAGS) -c $< -o $@

.PHONY: clean

clean:
		$(RM) -r $(BUILD_DIR)

-include $(DEPS)

MKDIR_P ?= mkdir -p
