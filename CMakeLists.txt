cmake_minimum_required(VERSION 3.10)


set(CMAKE_C_COMPILER "$ENV{GCC_COMPILER}-gcc")
set(CMAKE_CXX_COMPILER "$ENV{GCC_COMPILER}-g++")

project(YoloBindings)

add_definitions(-DRV1106_1103)
add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/3rdparty/ 3rdparty.out)
add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/utils/ utils.out)

set(CMAKE_INSTALL_RPATH "$ORIGIN/lib")
set(CMAKE_C_FLAGS "-fPIC")
set(CMAKE_CXX_FLAGS "-fPIC")

file(GLOB SRCS ${CMAKE_CURRENT_SOURCE_DIR}/src/*.cc)

#Opencv 4 
set(OpenCV_DIR "${CMAKE_CURRENT_SOURCE_DIR}/opencv_mobile_lib/opencv-mobile-4.11.0-luckfox-pico/lib/cmake/opencv4")
find_package(OpenCV REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})

set(THREADS_PREFER_PTHREAD_FLAG ON)
find_package(Threads REQUIRED)


add_library(${PROJECT_NAME} SHARED ${SRCS})

target_link_libraries(${PROJECT_NAME}
    ${LIBRKNNRT}
    ${OpenCV_LIBS}
    Threads::Threads
    imageutils
    fileutils
)

target_include_directories(${PROJECT_NAME} PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/include
    ${CMAKE_CURRENT_SOURCE_DIR}/3rdparty/rknpu2/include
    ${CMAKE_CURRENT_SOURCE_DIR}/3rdparty/allocator/dma
    ${OpenCV_INCLUDE_DIRS}
)

# Install
set(CMAKE_INSTALL_PREFIX "${CMAKE_CURRENT_SOURCE_DIR}/bundle")
install(TARGETS ${PROJECT_NAME} DESTINATION ${CMAKE_INSTALL_PREFIX})
install(FILES ${CMAKE_CURRENT_SOURCE_DIR}/python_lib/pyolo.py DESTINATION ${CMAKE_INSTALL_PREFIX})
install(FILES ${CMAKE_CURRENT_SOURCE_DIR}/python_lib/od_structs.py DESTINATION ${CMAKE_INSTALL_PREFIX})
