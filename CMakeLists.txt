cmake_minimum_required(VERSION 3.27)

project(gsplat_core LANGUAGES CXX)

# ----------------------------- Setup -----------------------------
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

option(BUILD_SHARED_LIBS "Build shared libraries" ON)
option(GSPLAT_BUILD_PYTHON "Build nanobind Python module" ON)
option(GSPLAT_BUILD_TEST "Build C++ test executable" ON)
option(GSPLAT_BUILD_METAL "Build metallib when MLX helpers are available" ON)
option(GSPLAT_USE_MLX_MODULE_CMAKE_DIR "Resolve MLX root from the Python mlx package" OFF)
option(GSPLAT_XCODE_DEV_MODE "Enable Xcode Metal source visibility" OFF)

# ----------------------------- Dependencies -----------------------------
find_package(
  Python 3.11
  COMPONENTS Interpreter Development.Module
  REQUIRED)

execute_process(
  COMMAND "${Python_EXECUTABLE}" -c "import sysconfig; print(sysconfig.get_paths()['purelib'])"
  OUTPUT_STRIP_TRAILING_WHITESPACE
  OUTPUT_VARIABLE PYTHON_SITE_PACKAGES)

# Prefer robust pip/conda-installed MLX CMake location by default.
set(MLX_ROOT "${PYTHON_SITE_PACKAGES}/mlx/share/cmake/MLX")
if(GSPLAT_USE_MLX_MODULE_CMAKE_DIR)
  execute_process(
    COMMAND "${Python_EXECUTABLE}" -c "import importlib.util, pathlib; spec = importlib.util.find_spec('mlx'); print(pathlib.Path(spec.submodule_search_locations[0]) / 'share' / 'cmake' / 'MLX')"
    RESULT_VARIABLE MLX_CMAKE_DIR_RESULT
    OUTPUT_STRIP_TRAILING_WHITESPACE
    OUTPUT_VARIABLE MLX_ROOT_FROM_MODULE)

  if(MLX_CMAKE_DIR_RESULT EQUAL 0 AND NOT MLX_ROOT_FROM_MODULE STREQUAL "")
    set(MLX_ROOT "${MLX_ROOT_FROM_MODULE}")
  endif()
endif()

# MLX headers bundled in Python package.
set(MLX_PYTHON_INCLUDE_DIR "${PYTHON_SITE_PACKAGES}/mlx/include")

find_package(MLX CONFIG REQUIRED HINTS "${MLX_ROOT}")

if(GSPLAT_BUILD_PYTHON)
  execute_process(
    COMMAND "${Python_EXECUTABLE}" -m nanobind --cmake_dir
    OUTPUT_STRIP_TRAILING_WHITESPACE
    OUTPUT_VARIABLE nanobind_ROOT)
  find_package(nanobind CONFIG REQUIRED)
endif()

# ----------------------------- Sources -----------------------------
file(GLOB GSPLAT_CPP_SOURCES CONFIGURE_DEPENDS
  "${CMAKE_CURRENT_LIST_DIR}/gsplat_core/*.cpp"
)

file(GLOB GSPLAT_METAL_SOURCES CONFIGURE_DEPENDS
  "${CMAKE_CURRENT_LIST_DIR}/gsplat_core/metal/*.metal"
)

# ----------------------------- Core Library -----------------------------
add_library(gsplat_core_lib ${GSPLAT_CPP_SOURCES})

target_include_directories(gsplat_core_lib PUBLIC
  ${CMAKE_CURRENT_LIST_DIR}/gsplat_core/include
  ${MLX_PYTHON_INCLUDE_DIR}
)

target_link_libraries(gsplat_core_lib PUBLIC mlx)

if(GSPLAT_XCODE_DEV_MODE AND CMAKE_GENERATOR STREQUAL "Xcode")
  target_sources(gsplat_core_lib PRIVATE ${GSPLAT_METAL_SOURCES})
  set_source_files_properties(${GSPLAT_METAL_SOURCES} PROPERTIES LANGUAGE METAL)
endif()

# ----------------------------- Metal -----------------------------
if(APPLE AND GSPLAT_BUILD_METAL AND GSPLAT_METAL_SOURCES)
  if(COMMAND mlx_build_metallib)
    mlx_build_metallib(
      TARGET gsplat_core_metallib
      TITLE gsplat_core
      SOURCES ${GSPLAT_METAL_SOURCES}
      INCLUDE_DIRS
        ${CMAKE_CURRENT_LIST_DIR}/gsplat_core
        ${MLX_PYTHON_INCLUDE_DIR}
      OUTPUT_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
    )

    add_dependencies(gsplat_core_lib gsplat_core_metallib)

    add_custom_command(TARGET gsplat_core_metallib POST_BUILD
      COMMAND ${CMAKE_COMMAND} -E copy
        "${CMAKE_CURRENT_BINARY_DIR}/gsplat_core.metallib"
        "$<TARGET_FILE_DIR:gsplat_core_lib>/gsplat_core.metallib"
    )
  else()
    message(WARNING "mlx_build_metallib not available. Skip metallib build.")
  endif()
endif()

# ----------------------------- Python Bindings -----------------------------
if(GSPLAT_BUILD_PYTHON)
  set(GSPLAT_PY_PACKAGE_DIR "${CMAKE_CURRENT_LIST_DIR}/python_package/gsplat_core")

  nanobind_add_module(
    _gsplat_core
    NB_STATIC
    STABLE_ABI
    LTO
    NOMINSIZE
    NB_DOMAIN mlx
    ${CMAKE_CURRENT_LIST_DIR}/gsplat_core/binding/binding.cpp
  )

  target_include_directories(_gsplat_core PRIVATE
    ${CMAKE_CURRENT_LIST_DIR}/gsplat_core/include
    ${MLX_PYTHON_INCLUDE_DIR}
  )

  target_link_libraries(_gsplat_core PRIVATE gsplat_core_lib)

  if(BUILD_SHARED_LIBS)
    target_link_options(_gsplat_core PRIVATE -Wl,-rpath,@loader_path)
  endif()

  add_custom_command(TARGET _gsplat_core POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy
      "$<TARGET_FILE:gsplat_core_lib>"
      "$<TARGET_FILE_DIR:_gsplat_core>/$<TARGET_FILE_NAME:gsplat_core_lib>"
  )

  # Local dev convenience: mirror extension artifacts into python package dir.
  add_custom_command(TARGET _gsplat_core POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E make_directory
      "${GSPLAT_PY_PACKAGE_DIR}"
    COMMAND ${CMAKE_COMMAND} -E copy
      "$<TARGET_FILE:_gsplat_core>"
      "${GSPLAT_PY_PACKAGE_DIR}/$<TARGET_FILE_NAME:_gsplat_core>"
    COMMAND ${CMAKE_COMMAND} -E copy
      "$<TARGET_FILE:gsplat_core_lib>"
      "${GSPLAT_PY_PACKAGE_DIR}/$<TARGET_FILE_NAME:gsplat_core_lib>"
  )

  if(TARGET gsplat_core_metallib)
    add_custom_command(TARGET _gsplat_core POST_BUILD
      COMMAND ${CMAKE_COMMAND} -E copy
        "${CMAKE_CURRENT_BINARY_DIR}/gsplat_core.metallib"
        "$<TARGET_FILE_DIR:_gsplat_core>/gsplat_core.metallib"
    )
    add_custom_command(TARGET _gsplat_core POST_BUILD
      COMMAND ${CMAKE_COMMAND} -E copy
        "${CMAKE_CURRENT_BINARY_DIR}/gsplat_core.metallib"
        "${GSPLAT_PY_PACKAGE_DIR}/gsplat_core.metallib"
    )
  endif()
endif()

# ----------------------------- Tests -----------------------------
if(GSPLAT_BUILD_TEST)
  add_executable(gsplat_core_dummy_test
    ${CMAKE_CURRENT_LIST_DIR}/gsplat_core/test/test.cpp
  )
  target_include_directories(gsplat_core_dummy_test PRIVATE
    ${CMAKE_CURRENT_LIST_DIR}/gsplat_core/include
  )
  target_link_libraries(gsplat_core_dummy_test PRIVATE gsplat_core_lib)
endif()
