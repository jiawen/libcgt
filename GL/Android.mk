LOCAL_PATH := $(call my-dir)
include $(CLEAR_VARS)

define all-h-files-under
$(patsubst ./%,%, \
  $(shell cd $(LOCAL_PATH) ; \
          find $(1) -name "*.h" -and -not -name ".*") \
 )
endef

define all-inc-files-under
$(patsubst ./%,%, \
  $(shell cd $(LOCAL_PATH) ; \
          find $(1) -name "*.inc" -and -not -name ".*") \
 )
endef

define all-inl-files-under
$(patsubst ./%,%, \
  $(shell cd $(LOCAL_PATH) ; \
          find $(1) -name "*.inl" -and -not -name ".*") \
 )
endef

define all-cpp-files-under
$(patsubst ./%,%, \
  $(shell cd $(LOCAL_PATH) ; \
          find $(1) -name "*.cpp" -and -not -name ".*") \
 )
endef

LOCAL_MODULE := libcgt_gl

LOCAL_CFLAGS += -D GL_PLATFORM_ES_31 -D GL_GLEXT_PROTOTYPES
LOCAL_LDLIBS := -lGLESv3

LOCAL_C_INCLUDES += $(LOCAL_PATH)/../core/src
LOCAL_C_INCLUDES += $(LOCAL_PATH)/src/common
LOCAL_C_INCLUDES += $(LOCAL_PATH)/src/GLES_31
LOCAL_SRC_FILES := $(call all-cpp-files-under,src/common)
LOCAL_SRC_FILES += $(call all-cpp-files-under,src/GLES_31)

LOCAL_SHARED_LIBRARIES += libcgt_core

include $(BUILD_SHARED_LIBRARY)
