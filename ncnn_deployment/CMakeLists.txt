# 1. 设置 CMake 版本
cmake_minimum_required(VERSION 3.10)

# 2. 定义项目名称和版本
project(MyProject VERSION 1.0 LANGUAGES CXX)

# 3. 设置 C++ 标准
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED True)

# 4. 查找 OpenCV 库
find_package(OpenCV REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})

# 5. 查找 NCNN 库
# 假设你已经安装了 NCNN 并设置了环境变量
# 如果没有设置环境变量，请手动设置路径
#set(NCNN_DIR "/path/to/ncnn")
#find_package(ncnn REQUIRED)  # 搜索 NCNN 库
#include_directories(${ncnn_INCLUDE_DIRS})  # 包含 NCNN 的头文件路径

set(ncnn_DIR "/home/leiwang/workspace/pytorch_learn/kaggle/ncnn/build/install/lib/cmake/ncnn")
find_package(ncnn REQUIRED)
include_directories("/home/leiwang/workspace/pytorch_learn/kaggle/ncnn/build/install/include/ncnn")

# 6. 添加源文件
add_executable(MyExecutable main.cpp)

# 7. 链接 OpenCV 和 NCNN 库
target_link_libraries(MyExecutable ${OpenCV_LIBS} ncnn)

# 8. 设置编译选项 (可选)
target_compile_options(MyExecutable PRIVATE -Wall -Wextra)

# 9. 设置可执行文件输出路径
set(EXECUTABLE_OUTPUT_PATH ${PROJECT_SOURCE_DIR}/bin)

# 10. 安装目标 (可选)
install(TARGETS MyExecutable DESTINATION bin)

