// Copyright (c) 2019-2024, Lawrence Livermore National Security, LLC and
// other Serac Project Developers. See the top-level LICENSE file for
// details.
//
// SPDX-License-Identifier: (BSD-3-Clause)

/**
 * @file test_finite_element_staet.cpp
 */

#include "serac/physics/state/finite_element_state.hpp"

#include "axom/slic/core/SimpleLogger.hpp"
#include <gtest/gtest.h>

#include "serac/infrastructure/initialize.hpp"
#include "serac/infrastructure/terminator.hpp"
#include "serac/mesh/mesh_utils.hpp"
#include "serac/numerics/functional/tensor.hpp"

namespace serac {

TEST(FiniteElementState, Set)
{
    constexpr int p = 1;
    constexpr int spatial_dim = 3;
    int serial_refinement = 0;
    int parallel_refinement = 0;

    // Construct the appropriate dimension mesh and give it to the data store
    std::string filename = SERAC_REPO_DIR "/data/meshes/beam-hex.mesh";
    auto mesh = mesh::refineAndDistribute(buildMeshFromFile(filename), serial_refinement, parallel_refinement);

    FiniteElementState scalar_state(*mesh, H1<p>{}, "scalar_field");
    double c = 2.0;
    auto scalar_field = [c](tensor<double, spatial_dim> X) -> double { return c*X[0]; };
    scalar_state.setFromField(scalar_field);

    // constexpr int vdim = 3;
    // auto vector_state = serac::StateManager::newState(H1<p, vdim>{}, "vector_field", mesh_tag);
    // auto vector_field = [](tensor<double, spatial_dim> X) { return X; };
    // vector_state.setFromField(vector_field);
}

} // namespace serac

int main(int argc, char* argv[])
{
    ::testing::InitGoogleTest(&argc, argv);
    serac::initialize(argc, argv);
    int result = RUN_ALL_TESTS();
    serac::exitGracefully(result);
}