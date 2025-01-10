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

TEST(FiniteElementState, SetFromFieldFunction)
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

    // Get the nodal positions for the state in grid function form
    mfem::ParGridFunction nodal_coords(const_cast<mfem::ParFiniteElementSpace*>(&scalar_state.space()));
    mesh->GetNodes(nodal_coords);

    for (int node = 0; node < scalar_state.space().GetNDofs(); node++) {
        tensor<double, spatial_dim> Xn;
        for (int i = 0; i < spatial_dim; i++) {
            int dof_index = mfem::Ordering::Map<serac::ordering>(
                nodal_coords.FESpace()->GetNDofs(), nodal_coords.FESpace()->GetVDim(), node, i);
            Xn[i] = nodal_coords(dof_index);
        }
        EXPECT_DOUBLE_EQ(scalar_field(Xn), scalar_state(node));
    }

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