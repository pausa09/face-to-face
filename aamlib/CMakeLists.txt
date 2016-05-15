cmake_minimum_required(VERSION 2.8.11)

project(aamlib)

set( SOURCE_FILES
	aam.cpp
	cvimagewidget.cpp
	fastsicaam.cpp
	icaam.cpp
	model.cpp
	robustaam.cpp
	sicaam.cpp
	trainingdata.cpp
	wsicaam.cpp
	 )

set( HEADERS
	aam.h
	cvimagewidget.h
	fastsicaam.h
	icaam.h
	model.h
	robustaam.h
	sicaam.h
	trainingdata.h
	wsicaam.h
)

# Tell CMake to create the helloworld executable
add_library(aamlib STATIC ${SOURCE_FILES} ${HEADERS} )

# Use the Widgets module from Qt 5.
target_link_libraries(aamlib Qt5::Widgets)
target_link_libraries(aamlib ${OpenCV_LIBS})