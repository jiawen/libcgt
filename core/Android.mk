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

LOCAL_MODULE := libcgt_core

LOCAL_C_INCLUDES += $(LOCAL_PATH)/src
LOCAL_C_INCLUDES += $(LOCAL_PATH)/third_party
LOCAL_SRC_FILES := $(call all-cpp-files-under,src)
LOCAL_SRC_FILES += $(call all-cpp-files-under,third_party)

include $(BUILD_SHARED_LIBRARY)
