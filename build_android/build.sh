#!/bin/bash

ndk-build NDK_PROJECT_PATH=. NDK_APPLICATION_MK=../core/Application.mk APP_BUILD_SCRIPT=../core/Android.mk
