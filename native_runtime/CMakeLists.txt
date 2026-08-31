cmake_minimum_required(VERSION 3.21)
project(ncrc_physics_runtime LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_MSVC_RUNTIME_LIBRARY "MultiThreaded$<$<CONFIG:Debug>:Debug>")

set(PHYSX_ROOT "${CMAKE_CURRENT_LIST_DIR}/../evidence/official/PhysX-107.3/physx" CACHE PATH "PhysX SDK root")
set(PHYSX_BIN "${PHYSX_ROOT}/bin/win.x86_64.vc143.mt/release")

add_executable(ncrc_physics
    app/main.cpp
    engine/physx_context.cpp
    engine/physics_scene.cpp
    robot/articulation.cpp
)
target_include_directories(ncrc_physics PRIVATE
    ${CMAKE_CURRENT_LIST_DIR}
    ${PHYSX_ROOT}/include
)
target_compile_definitions(ncrc_physics PRIVATE PX_PHYSX_STATIC_LIB=0)
target_link_directories(ncrc_physics PRIVATE ${PHYSX_BIN})
target_link_libraries(ncrc_physics PRIVATE
    PhysX_64 PhysXCommon_64 PhysXFoundation_64 PhysXCooking_64
    PhysXExtensions_static_64
)

foreach(dll PhysX_64.dll PhysXCommon_64.dll PhysXFoundation_64.dll PhysXCooking_64.dll)
    add_custom_command(TARGET ncrc_physics POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy_if_different
            "${PHYSX_BIN}/${dll}" "$<TARGET_FILE_DIR:ncrc_physics>/${dll}")
endforeach()

enable_testing()
add_test(NAME native_self_test COMMAND ncrc_physics test)
