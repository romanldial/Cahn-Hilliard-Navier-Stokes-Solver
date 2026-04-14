cmake_minimum_required(VERSION 3.16)
project(CahnHilliardNavierStokesSolver
  VERSION 0.1.0
  LANGUAGES CXX
)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# ── MPI ────────────────────────────────────────────────────────────────────────
find_package(MPI REQUIRED)

# ── Find MFEM ─────────────────────────────────────────────────────────────────
# Default to the in-source build at ~/Packages/mfem. Override via
#   cmake -DMFEM_ROOT=/path/to/mfem ..
# Note: MFEM_DIR is reserved by CMake's config-mode find_package and must not
# be used as a user-facing cache variable.
set(MFEM_ROOT "$ENV{HOME}/Packages/mfem"
    CACHE PATH "Root of the MFEM source/build tree")

# Prefer an installed CMake config; fall back to in-source build layout.
find_package(MFEM CONFIG QUIET HINTS "${MFEM_ROOT}")
if(NOT MFEM_FOUND)
  message(STATUS
    "MFEMConfig.cmake not found — using build-tree layout at ${MFEM_ROOT}")
  find_library(_MFEM_LIB NAMES mfem HINTS "${MFEM_ROOT}" REQUIRED)
  add_library(mfem STATIC IMPORTED GLOBAL)
  set_target_properties(mfem PROPERTIES
    IMPORTED_LOCATION             "${_MFEM_LIB}"
    INTERFACE_INCLUDE_DIRECTORIES "${MFEM_ROOT}"
  )
  # ── HYPRE (required by parallel MFEM) ────────────────────────────────────────
  set(_HYPRE_DEFAULT "$ENV{HOME}/Packages/firedrake/petsc/arch-firedrake-default")
  set(HYPRE_ROOT "${_HYPRE_DEFAULT}" CACHE PATH "Root of the HYPRE installation")
  find_path(_HYPRE_INC  HYPRE_utilities.h HINTS "${HYPRE_ROOT}/include" REQUIRED)
  find_library(_HYPRE_LIB NAMES HYPRE       HINTS "${HYPRE_ROOT}/lib"     REQUIRED)
  set_property(TARGET mfem APPEND PROPERTY
    INTERFACE_INCLUDE_DIRECTORIES "${_HYPRE_INC}")
  set_property(TARGET mfem APPEND PROPERTY
    INTERFACE_LINK_LIBRARIES "${_HYPRE_LIB}")

  # ── METIS 4 (required by parallel MFEM) ───────────────────────────────────
  set(METIS_ROOT "$ENV{HOME}/Packages/metis-4.0" CACHE PATH "Root of METIS 4")
  find_path(_METIS_INC  metis.h  HINTS "${METIS_ROOT}/Lib" REQUIRED)
  find_library(_METIS_LIB NAMES metis HINTS "${METIS_ROOT}"   REQUIRED)
  set_property(TARGET mfem APPEND PROPERTY
    INTERFACE_INCLUDE_DIRECTORIES "${_METIS_INC}")
  set_property(TARGET mfem APPEND PROPERTY
    INTERFACE_LINK_LIBRARIES "${_METIS_LIB}")

  # ── Platform ──────────────────────────────────────────────────────────────
  # On macOS, MFEM depends on the Accelerate framework for BLAS/LAPACK.
  if(APPLE)
    find_library(ACCELERATE_LIB Accelerate REQUIRED)
    set_property(TARGET mfem APPEND PROPERTY
      INTERFACE_LINK_LIBRARIES "${ACCELERATE_LIB}")
  endif()
  set_property(TARGET mfem APPEND PROPERTY
    INTERFACE_LINK_LIBRARIES MPI::MPI_CXX)
endif()

# ── Library ────────────────────────────────────────────────────────────────────
add_library(chns STATIC
  src/CahnHilliard.cpp
  src/ChemicalPotential.cpp
  src/LILS.cpp
)
target_include_directories(chns
  PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:include>
)
target_link_libraries(chns PUBLIC mfem)

# ── Examples ───────────────────────────────────────────────────────────────────
add_subdirectory(examples)
