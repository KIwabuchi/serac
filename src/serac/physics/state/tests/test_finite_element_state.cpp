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
#include "mfem.hpp"

#include "serac/infrastructure/initialize.hpp"
#include "serac/infrastructure/terminator.hpp"
#include "serac/mesh/mesh_utils.hpp"
#include "serac/numerics/functional/tensor.hpp"

namespace serac {



// class FiniteElementTest : public testing::Test {
//  protected:

//     FiniteElementTest() : spatial_dim_(3) {}

//     void SetUp override {
//         constexpr int p = 2;
//         constexpr int spatial_dim = 3;
//         int serial_refinement = 0;
//         int parallel_refinement = 0;

//         std::string filename = SERAC_REPO_DIR "/data/meshes/beam-hex.mesh";
//         auto mesh_ptr = mesh::refineAndDistribute(buildMeshFromFile(filename), serial_refinement, parallel_refinement);
//         mesh_ = std::move(mesh_ptr);
//     }

//     void foo(const FiniteElementState& state) const
//     {
//         for (int node = 0; node < state.space().GetNDofs(); node++) {
//         tensor<double, spatial_dim> Xn;
//         for (int i = 0; i < spatial_dim; i++) {
//             int dof_index = mfem::Ordering::Map<serac::ordering>(
//                 nodal_coords.FESpace()->GetNDofs(), nodal_coords.FESpace()->GetVDim(), node, i);
//             Xn[i] = nodal_coords(dof_index);
//         }
//         EXPECT_DOUBLE_EQ(scalar_field(Xn), scalar_state(node));
//     }
//     }

//     const int spatial_dim_;
//     std::unique_ptr<mfem::ParMesh> mesh_;
// };

// TEST_F(FiniteElementTest, SetFromScalarFieldFunction)
// {
//     FiniteElementState scalar_state(*mesh, H1<p>{}, "scalar_field");
//     double c = 2.0;
//     auto scalar_field = [c](tensor<double, spatial_dim> X) -> double { return c*X[0]; };
//     scalar_state.setFromField(scalar_field);

// }

TEST(FiniteElementState, SetScalarStateFromFieldFunction)
{
    constexpr int p = 1;
    constexpr int spatial_dim = 3;
    int serial_refinement = 0;
    int parallel_refinement = 0;

    // Construct the appropriate dimension mesh and give it to the data store
    std::string filename = SERAC_REPO_DIR "/data/meshes/beam-hex.mesh";
    auto mesh = mesh::refineAndDistribute(buildMeshFromFile(filename), serial_refinement, parallel_refinement);

    FiniteElementState scalar_state(*mesh, H1<p>{}, "scalar_field");
    // check that captures work
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
}

TEST(FiniteElementState, SetVectorStateFromFieldFunction)
{
    constexpr int p = 2;
    constexpr int spatial_dim = 3;
    int serial_refinement = 0;
    int parallel_refinement = 0;

    // Construct mesh
    std::string filename = SERAC_REPO_DIR "/data/meshes/beam-hex.mesh";
    auto mesh = mesh::refineAndDistribute(buildMeshFromFile(filename), serial_refinement, parallel_refinement);
    ASSERT_EQ(spatial_dim, mesh->SpaceDimension()) << "Test configured incorrectly. The variable spatial_dim must match the spatial dimension of the mesh.";

    // Choose vector dimension for state field that is different from spatial dimension
    // to test the field indexing more thoroughly.
    constexpr int vdim = 2;
    FiniteElementState state(*mesh, H1<p, vdim>{}, "vector_field");

    // set the field with an arbitrarily chosen field function
    auto vector_field = [](tensor<double, spatial_dim> X) { return tensor<double, vdim>{norm(X), 1.0/(1.0 + norm(X))}; };
    state.setFromField(vector_field);
    
    // Get the nodal positions for the state in a grid function
    auto [coords_fe_space, coords_fe_coll] = serac::generateParFiniteElementSpace<H1<p, spatial_dim>>(mesh.get());
    mfem::ParGridFunction nodal_coords_gf(coords_fe_space.get());
    mesh->GetNodes(nodal_coords_gf);

    // we need the state values and the nodal coordinates in the same kind of container,
    // so we will get the grid function view of the state
    mfem::ParGridFunction& state_gf = state.gridFunction();


    for (int node = 0; node < state_gf.FESpace()->GetNDofs(); node++) {

        // Fill a tensor with the coordinates of the node
        tensor<double, spatial_dim> Xn;
        for (int i = 0; i < spatial_dim; i++) {
            int dof_index = nodal_coords_gf.FESpace()->DofToVDof(node, i);
            Xn[i] = nodal_coords_gf(dof_index);
        }
        
        // apply the field function to the node coords
        auto v = vector_field(Xn);

        // check that value set in the state matches the field function
        for (int j = 0; j < vdim; j++) {
            int dof_index = state_gf.FESpace()->DofToVDof(node, j);
            EXPECT_DOUBLE_EQ(v[j], state_gf(dof_index));
        }
    }
}

TEST(FiniteElementState, ErrorsIfFieldFunctionDimensionMismatchedToState)
{
    constexpr int p = 2;
    constexpr int spatial_dim = 3;
    int serial_refinement = 0;
    int parallel_refinement = 0;

    // Construct mesh
    std::string filename = SERAC_REPO_DIR "/data/meshes/beam-hex.mesh";
    auto mesh = mesh::refineAndDistribute(buildMeshFromFile(filename), serial_refinement, parallel_refinement);
    ASSERT_EQ(spatial_dim, mesh->SpaceDimension()) << "Test configured incorrectly. The variable spatial_dim must match the spatial dimension of the mesh.";

    // Choose vector dimension for state field that is different from spatial dimension
    constexpr int vdim = 2;
    FiniteElementState state(*mesh, H1<p, vdim>{}, "vector_field");

    // Set the field with a field function with the wrong vector dimension.
    // Should return tensor of size vdim!
    auto vector_field = [](tensor<double, spatial_dim> X) { return X; };

    EXPECT_DEATH(state.setFromField(vector_field), "Cannot copy tensor into an MFEM Vector with incompatible size.");
}

} // namespace serac

int main(int argc, char* argv[])
{
    ::testing::InitGoogleTest(&argc, argv);
    serac::initialize(argc, argv);
    int result = RUN_ALL_TESTS();
    serac::exitGracefully(result);
}