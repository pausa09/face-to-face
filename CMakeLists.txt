cmake_minimum_required(VERSION 2.8.11)

project(Face2Face)

# Find includes in corresponding build directories
set(CMAKE_INCLUDE_CURRENT_DIR ON)
# Instruct CMake to run moc automatically when needed.
set(CMAKE_AUTOMOC ON)

SET (CMAKE_LIBRARY_OUTPUT_DIRECTORY
   ${PROJECT_BINARY_DIR}/bin
   CACHE PATH
   "Single Directory for all Libraries"
   )

# --------- Setup the Executable output Directory -------------
SET (CMAKE_RUNTIME_OUTPUT_DIRECTORY
   ${PROJECT_BINARY_DIR}/bin
   CACHE PATH
   "Single Directory for all Executables."
   )

# --------- Setup the Executable output Directory -------------
SET (CMAKE_ARCHIVE_OUTPUT_DIRECTORY
   ${PROJECT_BINARY_DIR}/bin
   CACHE PATH
   "Single Directory for all static libraries."
   )

# Find the QtWidgets library
find_package(Qt5Widgets)
find_package(Qt5Network)

include_directories("C:\\opencv\\build\\include")

set(OpenCV_STATIC OFF)
find_package(OpenCV REQUIRED)

# subprojects
add_subdirectory(aamlib)
add_subdirectory(AAMed)
add_subdirectory(aamrec)
add_subdirectory(F2FI_lib)
add_subdirectory(F2FI_lib_test)
