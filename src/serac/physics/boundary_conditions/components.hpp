// Copyright (c) 2019-2024, Lawrence Livermore National Security, LLC and
// other Serac Project Developers. See the top-level LICENSE file for
// details.
//
// SPDX-License-Identifier: (BSD-3-Clause)

/**
 * @file solid_mechanics.hpp
 *
 * @brief Tools for taging a set of vector components, say for boundary condition enforcement
 */

#pragma once

#include <bitset>

namespace serac {

/// Vector component names
enum class Component : size_t { X = 0b001, Y = 0b010, Z = 0b100, ALL = 0b111 };

/// A set to flag components of a vector field
class Components {
 public:
    Components(Component i) : flags_{size_t(i)} {};

    bool operator[] (size_t i) const {return flags_[i]; };

    friend Components operator+ (Component i, Component j);

    Components operator+ (Component i) {
        flags_ |= size_t(i);
        return *this;
    };

    friend Components operator+ (Component i, Components j);

 private:
    std::bitset<3> flags_;
};

inline Components operator+ (Component i, Component j) {
    return Components(i) + j;
};

inline Components operator+ (Component i, Components c) {
    return c + i;
}

} // namespace serac