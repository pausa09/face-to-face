cmake_minimum_required(VERSION 2.8.11)

set(CMAKE_AUTOMOC ON)

project(F2FI_lib)

set( SOURCE_FILES
	 ChartWidget.cpp
	 MimRec.cpp
	 PrisonersDilemma.cpp
	 FaceTracking.cpp
	 BarChartModel.cpp
	 BackProp.cpp
)

set( HEADERS
	 ChartWidget.h
	 MimRec.h
	 PrisonersDilemma.h
	 FaceTracking.h
	 BarChartModel.h
	 BackProp.h
)

include_directories("C:\\kdchart-2.5.1-source\\include\\KDChart")

# Tell CMake to create the helloworld executable
add_library(F2FI_lib STATIC ${SOURCE_FILES} ${HEADERS} )

# Use the Widgets module from Qt 5.
target_link_libraries(F2FI_lib ${OpenCV_LIBS})
target_link_libraries(F2FI_lib Qt5::Core)
target_link_libraries(F2FI_lib Qt5::Widgets)
target_link_libraries(F2FI_lib Qt5::Network)
target_link_libraries(F2FI_lib "C:\\kdchart-2.5.1-source\\lib\\kdchartd2.lib")
target_link_libraries(F2FI_lib aamlib)