#root/CMakeLists.txt

#CMake Version Minimum Required
cmake_minimum_required (VERSION 2.8.11)

###############################################################################
# DEFINITION
###############################################################################
set(PROJECT_ACRONYM_STR "SSIGRes")
set(PROJECT_NAME_STR "Researchs on Smart Surveillance Interest Group")
project(${PROJECT_ACRONYM_STR} CXX)

set(SSIGLIB_PATH "ssiglib" CACHE PATH "Path to SSIGLib Directory")

if(EXISTS "${SSIGLIB_PATH}" AND IS_DIRECTORY "${SSIGLIB_PATH}")
  add_subdirectory (${SSIGLIB_PATH})
  message(STATUS "SSIGLib diretory found")
endif()

file(GLOB RESEARCH_FILES "${PROJECT_SOURCE_DIR}/src/*.cpp" "${PROJECT_SOURCE_DIR}/include/*.hpp")

add_executable(${PROJECT_ACRONYM_STR} ${RESEARCH_FILES})
target_include_directories(${PROJECT_ACRONYM_STR} PUBLIC ${PROJECT_SOURCE_DIR}/include/)

#SELECT THE SSIGLIB MODULES
target_link_libraries(${PROJECT_ACRONYM_STR} core ml)
