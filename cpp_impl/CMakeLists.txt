cmake_minimum_required(VERSION 3.18 FATAL_ERROR)
project(pcqo_mis)

find_package(Torch REQUIRED)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${TORCH_CXX_FLAGS}")

# Set the output directory for the executable
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY "${CMAKE_SOURCE_DIR}/../external")

add_executable(pcqo_mis pcqo_mis.cpp)
target_link_libraries(pcqo_mis "${TORCH_LIBRARIES}")
set_property(TARGET pcqo_mis PROPERTY CXX_STANDARD 17)

# The following code block is suggested to be used on Windows.
# According to https://github.com/pytorch/pytorch/issues/25457,
# the DLLs need to be copied to avoid memory errors.
if (MSVC)
  file(GLOB TORCH_DLLS "${TORCH_INSTALL_PREFIX}/lib/*.dll")
  add_custom_command(TARGET pcqo_mis
                    POST_BUILD
                    COMMAND ${CMAKE_COMMAND} -E copy_if_different
                    ${TORCH_DLLS}
                    $<TARGET_FILE_DIR:pcqo_mis>)
endif (MSVC)
