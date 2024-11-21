// Copyright (c) 2019-2024, Lawrence Livermore National Security, LLC and
// other Serac Project Developers. See the top-level LICENSE file for
// details.
//
// SPDX-License-Identifier: (BSD-3-Clause)

#pragma once

#include <vector>

#include "mfem.hpp"

#include "serac/numerics/functional/tensor.hpp"
#include "serac/numerics/functional/finite_element.hpp"
#include "serac/numerics/functional/element_restriction.hpp"
#include "serac/numerics/functional/typedefs.hpp"

namespace serac {

struct BlockElementRestriction;

/**
 * @brief a class for representing a geometric region that can be used for integration
 *
 * This region can be an entire mesh or some subset of its elements
 */
struct Domain {

  /// @brief enum describing what kind of elements are included in a Domain
  enum Type
  {
    Elements,
    BoundaryElements,
    InteriorFaces
  };

  static constexpr int num_types = 3;  ///< the number of entries in the Type enum

  /// @brief the underyling mesh for this domain
  const mesh_t& mesh_;

  /// @brief the geometric dimension of the domain
  int dim_;

  /// @brief whether the elements in this domain are on the boundary or not
  Type type_;

  /// note: only lists with appropriate dimension (see dim_) will be populated
  ///       for example, a 2D Domain may have `tri_ids_` and `quad_ids_` non-nonempty,
  ///       but all other lists will be empty
  ///
  /// these lists hold indices into the "E-vector" of the appropriate geometry
  ///
  /// @cond
  std::vector<int> vertex_ids_;
  std::vector<int> edge_ids_;
  std::vector<int> tri_ids_;
  std::vector<int> quad_ids_;
  std::vector<int> tet_ids_;
  std::vector<int> hex_ids_;

  std::vector<int> mfem_edge_ids_;
  std::vector<int> mfem_tri_ids_;
  std::vector<int> mfem_quad_ids_;
  std::vector<int> mfem_tet_ids_;
  std::vector<int> mfem_hex_ids_;
  /// @endcond

  std::map< FunctionSpace, BlockElementRestriction > restriction_operators;

  /**
   * @brief empty Domain constructor, with connectivity info to be populated later
   */
  Domain(const mesh_t& m, int d, Type type = Domain::Type::Elements) : mesh_(m), dim_(d), type_(type) {}

  /**
   * @brief create a domain from some subset of the vertices in an mfem::Mesh
   * @param mesh the entire mesh
   * @param func predicate function for determining which vertices will be
   * included in this domain. The function's argument is the spatial position of the vertex.
   */
  static Domain ofVertices(const mesh_t& mesh, std::function<bool(vec2)> func);

  /// @overload
  static Domain ofVertices(const mesh_t& mesh, std::function<bool(vec3)> func);

  /**
   * @brief create a domain from some subset of the edges in an mfem::Mesh
   * @param mesh the entire mesh
   * @param func predicate function for determining which edges will be
   * included in this domain. The function's arguments are the list of vertex coordinates and
   * an attribute index (if appropriate).
   */
  static Domain ofEdges(const mesh_t& mesh, std::function<bool(std::vector<vec2>, int)> func);

  /// @overload
  static Domain ofEdges(const mesh_t& mesh, std::function<bool(std::vector<vec3>)> func);

  /**
   * @brief create a domain from some subset of the faces in an mfem::Mesh
   * @param mesh the entire mesh
   * @param func predicate function for determining which faces will be
   * included in this domain. The function's arguments are the list of vertex coordinates and
   * an attribute index (if appropriate).
   */
  static Domain ofFaces(const mesh_t& mesh, std::function<bool(std::vector<vec2>, int)> func);

  /// @overload
  static Domain ofFaces(const mesh_t& mesh, std::function<bool(std::vector<vec3>, int)> func);

  /**
   * @brief create a domain from some subset of the elements (spatial dim == geometry dim) in an mfem::Mesh
   * @param mesh the entire mesh
   * @param func predicate function for determining which elements will be
   * included in this domain. The function's arguments are the list of vertex coordinates and
   * an attribute index (if appropriate).
   */
  static Domain ofElements(const mesh_t& mesh, std::function<bool(std::vector<vec2>, int)> func);

  /// @overload
  static Domain ofElements(const mesh_t& mesh, std::function<bool(std::vector<vec3>, int)> func);

  /**
   * @brief create a domain from some subset of the boundary elements (spatial dim == geometry dim + 1) in an mfem::Mesh
   * @param mesh the entire mesh
   * @param func predicate function for determining which boundary elements will be included in this domain
   */
  static Domain ofBoundaryElements(const mesh_t& mesh, std::function<bool(std::vector<vec2>, int)> func);

  /// @overload
  static Domain ofBoundaryElements(const mesh_t& mesh, std::function<bool(std::vector<vec3>, int)> func);

  /// @brief get elements by geometry type
  const std::vector<int>& get(mfem::Geometry::Type geom) const
  {
    if (geom == mfem::Geometry::POINT) return vertex_ids_;
    if (geom == mfem::Geometry::SEGMENT) return edge_ids_;
    if (geom == mfem::Geometry::TRIANGLE) return tri_ids_;
    if (geom == mfem::Geometry::SQUARE) return quad_ids_;
    if (geom == mfem::Geometry::TETRAHEDRON) return tet_ids_;
    if (geom == mfem::Geometry::CUBE) return hex_ids_;

    exit(1);
  }

  /// @brief get elements by geometry type
  const std::vector<int>& get_mfem_ids(mfem::Geometry::Type geom) const
  {
    if (geom == mfem::Geometry::SEGMENT) return mfem_edge_ids_;
    if (geom == mfem::Geometry::TRIANGLE) return mfem_tri_ids_;
    if (geom == mfem::Geometry::SQUARE) return mfem_quad_ids_;
    if (geom == mfem::Geometry::TETRAHEDRON) return mfem_tet_ids_;
    if (geom == mfem::Geometry::CUBE) return mfem_hex_ids_;

    exit(1);
  }

  /**
   * @brief returns how many elements of any type belong to this domain 
   */
  int total_elements() const {
    return int(vertex_ids_.size() + edge_ids_.size() + tri_ids_.size() + quad_ids_.size() + tet_ids_.size() + hex_ids_.size());
  }

  /**
   * @brief returns an array of the prefix sum of element counts belonging to this domain. 
   *        Primarily intended to be used in mfem::BlockVector::Update(double * data, mfem::Array<int> bOffsets);
   */
  mfem::Array<int> bOffsets() const {
    mfem::Array<int> offsets(mfem::Geometry::NUM_GEOMETRIES + 1);

    int total = 0;
    offsets[mfem::Geometry::POINT] = total;       total += vertex_ids_.size();
    offsets[mfem::Geometry::SEGMENT] = total;     total += edge_ids_.size();
    offsets[mfem::Geometry::TRIANGLE] = total;    total += tri_ids_.size();
    offsets[mfem::Geometry::SQUARE] = total;      total += quad_ids_.size();
    offsets[mfem::Geometry::TETRAHEDRON] = total; total += tet_ids_.size();
    offsets[mfem::Geometry::CUBE] = total;        total += hex_ids_.size();
    offsets[mfem::Geometry::PRISM] = total;
    offsets[mfem::Geometry::PYRAMID] = total;
    offsets[mfem::Geometry::NUM_GEOMETRIES] = total;

    return offsets;
  }

  /// @brief get mfem degree of freedom list for a given FiniteElementSpace
  mfem::Array<int> dof_list(const fes_t* fes) const;

  /// @brief TODO
  void insert_restriction(const fes_t * fes, FunctionSpace space);

  /// @brief TODO
  const BlockElementRestriction & get_restriction(FunctionSpace space);

};

/// @brief constructs a domain from all the elements in a mesh
Domain EntireDomain(const mesh_t& mesh);

/// @brief constructs a domain from all the boundary elements in a mesh
Domain EntireBoundary(const mesh_t& mesh);

/// @brief constructs a domain from all the interior face elements in a mesh
Domain InteriorFaces(const mesh_t& mesh);

/// @brief create a new domain that is the union of `a` and `b`
Domain operator|(const Domain& a, const Domain& b);

/// @brief create a new domain that is the intersection of `a` and `b`
Domain operator&(const Domain& a, const Domain& b);

/// @brief create a new domain that is the set difference of `a` and `b`
Domain operator-(const Domain& a, const Domain& b);

/// @brief convenience predicate for creating domains by attribute
template <int dim>
inline auto by_attr(int value)
{
  return [value](std::vector<tensor<double, dim> >, int attr) { return attr == value; };
}

/**
 * @brief count the number of elements of each geometry in a domain
 * @param domain the domain to count
 */
inline std::array<uint32_t, mfem::Geometry::NUM_GEOMETRIES> geometry_counts(const Domain& domain)
{
  std::array<uint32_t, mfem::Geometry::NUM_GEOMETRIES> counts{};

  constexpr std::array<mfem::Geometry::Type, 5> geometries = {mfem::Geometry::SEGMENT, mfem::Geometry::TRIANGLE,
                                                              mfem::Geometry::SQUARE, mfem::Geometry::TETRAHEDRON,
                                                              mfem::Geometry::CUBE};
  for (auto geom : geometries) {
    counts[uint32_t(geom)] = uint32_t(domain.get(geom).size());
  }
  return counts;
}

/**
 * @brief convenience function for computing the arithmetic mean of some list of vectors
 */
template <int dim>
inline tensor<double, dim> average(std::vector<tensor<double, dim> >& positions)
{
  tensor<double, dim> total{};
  for (auto x : positions) {
    total += x;
  }
  return total / double(positions.size());
}

}  // namespace serac
