// Copyright (c) 2019-2024, Lawrence Livermore National Security, LLC and
// other Serac Project Developers. See the top-level LICENSE file for
// details.
//
// SPDX-License-Identifier: (BSD-3-Clause)

/**
 * @file equation_solver.hpp
 *
 * @brief This file contains the declaration of an equation solver wrapper
 */

#pragma once

#include <memory>
#include <optional>
#include <variant>

#include "serac/physics/state/finite_element_state.hpp"
#include "serac/physics/state/finite_element_dual.hpp"


namespace serac {

int GlobalSize(const mfem::Vector& parallel_v, const MPI_Comm& comm);

std::tuple<FiniteElementState, std::vector<FiniteElementState>, std::vector<double>> 
solveSubspaceProblem(const std::vector<FiniteElementState>& directions,
                     const std::vector<FiniteElementState>& Adirections,
                     const FiniteElementState& b,
                     double delta, int num_leftmost);

}