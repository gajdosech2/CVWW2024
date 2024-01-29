cmake_minimum_required (VERSION 3.8)

project ("3DCamerasParallelCapture")

add_executable(3DCamerasParallelCapture
    main.cpp
    dataManipulation.cpp
    misc.cpp

    KinectV1.cpp

    KinectV2.cpp

    KinectV3Azure.cpp

    MotionCamPhoXi.cpp
)


# change these variables as fit 
set(KinectV1Path "$ENV{HOMEDRIVE}\\Program Files\\Microsoft SDKs\\Kinect\\v1.8")       #https://www.microsoft.com/en-us/download/confirmation.aspx?id=40278
set(KinectV2Path "$ENV{HOMEDRIVE}\\Program Files\\Microsoft SDKs\\Kinect\\v2.0_1409")  #https://www.microsoft.com/en-us/download/confirmation.aspx?id=44561
set(KinectAzurePath "$ENV{HOMEDRIVE}\\Program Files\\Azure Kinect SDK v1.4.1")         #https://download.microsoft.com/download/3/d/6/3d6d9e99-a251-4cf3-8c6a-8e108e960b4b/Azure%20Kinect%20SDK%201.4.1.exe
set(PhoXiControlVersion "1.9.4")
set(PhotoNeoPath "$ENV{HOMEDRIVE}\\Program Files\\Photoneo\\PhoXiControl-${PhoXiControlVersion}")
set(ENV{OUTPUTDIRECTORY} "$ENV{USERPROFILE}\\Documents\\Depth map output folder")


# Kinect v1
target_include_directories(3DCamerasParallelCapture PRIVATE "${KinectV1Path}\\inc\\")
link_directories(3DCamerasParallelCapture PRIVATE "${KinectV1Path}\\lib\\amd64\\")
target_link_libraries(3DCamerasParallelCapture "${KinectV1Path}\\lib\\amd64\\Kinect10.lib")


# Kinect v2
target_include_directories(3DCamerasParallelCapture PRIVATE "${KinectV2Path}\\inc\\")
link_directories(3DCamerasParallelCapture PRIVATE "${KinectV2Path}\\Lib\\x64\\")
target_link_libraries(3DCamerasParallelCapture "${KinectV2Path}\\Lib\\x64\\Kinect20.lib")




# Kinect Azure
target_include_directories(3DCamerasParallelCapture PRIVATE "${KinectAzurePath}\\sdk\\include\\")
link_directories(3DCamerasParallelCapture PRIVATE "${KinectAzurePath}\\sdk\\windows-desktop\\amd64\\release\\lib\\")
target_link_libraries(3DCamerasParallelCapture "${KinectAzurePath}\\sdk\\windows-desktop\\amd64\\release\\lib\\k4a.lib")

# PhoXi
target_include_directories(3DCamerasParallelCapture PRIVATE "${PhotoNeoPath}\\API\\include\\")
link_directories(3DCamerasParallelCapture PRIVATE "${PhotoNeoPath}\\API\\lib\\")
target_link_libraries(3DCamerasParallelCapture "${PhotoNeoPath}\\API\\lib\\PhoXi_API_msvc14_Release_${PhoXiControlVersion}.lib")
file(COPY_FILE "${PhotoNeoPath}\\API\\bin\\PhoXi_API_msvc14_Release_${PhoXiControlVersion}.dll" 
               "${CMAKE_CURRENT_BINARY_DIR}/PhoXi_API_msvc14_Release_${PhoXiControlVersion}.dll"
               ONLY_IF_DIFFERENT)


add_custom_target(mytargetname ALL COMMAND ${CMAKE_COMMAND} -E make_directory $ENV{OUTPUTDIRECTORY})