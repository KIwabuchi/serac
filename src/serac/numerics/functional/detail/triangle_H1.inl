// Copyright (c) 2019-2023, Lawrence Livermore National Security, LLC and
// other Serac Project Developers. See the top-level LICENSE file for
// details.
//
// SPDX-License-Identifier: (BSD-3-Clause)

/**
 * @file quadrilateral_H1.inl
 *
 * @brief Specialization of finite_element for H1 on quadrilateral geometry
 */

// this specialization defines shape functions (and their gradients) that
// interpolate at Gauss-Lobatto nodes for the appropriate polynomial order
//
// note: mfem assumes the parent element domain is [0,1]x[0,1]
// for additional information on the finite_element concept requirements, see finite_element.hpp
/// @cond
template <int p, int c>
struct finite_element<Geometry::Triangle, H1<p, c> > {
  static constexpr auto geometry   = Geometry::Triangle;
  static constexpr auto family     = Family::H1;
  static constexpr int  components = c;
  static constexpr int  dim        = 2;
  static constexpr int  n          = (p + 1);
  static constexpr int  ndof       = (p + 1) * (p + 1);

  static constexpr int VALUE = 0, GRADIENT = 1;
  static constexpr int SOURCE = 0, FLUX = 1;

  using residual_type =
      typename std::conditional<components == 1, tensor<double, ndof>, tensor<double, ndof, components> >::type;


  using value_type = typename std::conditional<components == 1, double, tensor<double, components> >::type;
  using derivative_type =
      typename std::conditional<components == 1, tensor<double, dim>, tensor<double, components, dim> >::type;
  using qf_input_type = tuple<value_type, derivative_type>;

  /*

    interpolation nodes and their associated numbering:

      linear
    2
    | .
    |   .
    |     .
    |       .
    |         . 
    0-----------1

      quadratic
    2
    | .
    |   .
    5     4
    |       .
    |         . 
    0-----3-----1


      cubic
    2
    | .
    7   6
    |     .
    8   9   5
    |         . 
    0---3---4---1

  */

  SERAC_HOST_DEVICE static constexpr double shape_function(tensor<double, dim> xi, int i)
  {

    // linear
    if constexpr (n == 2) {
      switch(i) {
        case 0: return 1-xi[0]-xi[1];
        case 1: return xi[0];
        case 2: return xi[1];
      }
    }

    // quadratic
    if constexpr (n == 3) {
      switch(i) {
        case 0: return (-1+xi[0]+xi[1])*(-1+2*xi[0]+2*xi[1]);
        case 1: return xi[0]*(-1+2*xi[0]);
        case 2: return xi[1]*(-1+2*xi[1]);
        case 3: return -4*xi[0]*(-1+xi[0]+xi[1]);
        case 4: return 4*xi[0]*xi[1];
        case 5: return -4*xi[1]*(-1+xi[0]+xi[1]);
      }
    }

    // cubic
    if constexpr (n == 4) {
      constexpr double sqrt5 = 2.23606797749978981;
      switch(i) {
        case 0: return -((-1+xi[0]+xi[1])*(1+5*xi[0]*xi[0]+5*(-1+xi[1])*xi[1]+xi[0]*(-5+11*xi[1])));
        case 1: return xi[0]*(1+5*xi[0]*xi[0]+xi[1]-xi[1]*xi[1]-xi[0]*(5+xi[1]));
        case 2: return xi[1]*(1+xi[0]-xi[0]*xi[0]-xi[0]*xi[1]+5*(-1+xi[1])*xi[1]);
        case 3: return (5*xi[0]*(-1+xi[0]+xi[1])*(-89-41*sqrt5+2*(60+29*sqrt5)*xi[0]+(147+65*sqrt5)*xi[1]))/(58+24*sqrt5);
        case 4: return (-5*xi[0]*(-1+xi[0]+xi[1])*(-6-4*sqrt5+(25+13*sqrt5)*xi[0]-(7+sqrt5)*xi[1]))/(13+5*sqrt5);
        case 5: return (5*xi[0]*xi[1]*(-3-sqrt5+(7+3*sqrt5)*xi[0]+2*xi[1]))/(3+sqrt5);
        case 6: return (5*xi[0]*xi[1]*(-3-sqrt5+2*xi[0]+(7+3*sqrt5)*xi[1]))/(3+sqrt5);
        case 7: return (5*xi[1]*(-1+xi[0]+xi[1])*(1+sqrt5+2*xi[0]-(5+3*sqrt5)*xi[1]))/(3+sqrt5);
        case 8: return (5*xi[1]*(-1+xi[0]+xi[1])*(-1-sqrt5+(3+sqrt5)*xi[0]+2*sqrt5*xi[1]))/2.;
        case 9: return -27*xi[0]*xi[1]*(-1+xi[0]+xi[1]);
      }
    }

    return 0.0;

  }

  SERAC_HOST_DEVICE static constexpr tensor<double, dim> shape_function_gradient(tensor<double, dim> xi, int i)
  {

    // linear
    if constexpr (n == 2) {
      switch(i) {
        case 0: return {-1.0, -1.0};
        case 1: return { 1.0,  0.0};
        case 2: return { 0.0,  1.0};
      }
    }

    // quadratic
    if constexpr (n == 3) {
      switch(i) {
        case 0: return {-3+4*xi[0]+4*xi[1], -3+4*xi[0]+4*xi[1]};
        case 1: return {-1+4*xi[0], 0.0};
        case 2: return {0.0, -1+4*xi[1]};
        case 3: return {-4*(-1+2*xi[0]+xi[1]), -4*xi[0]};
        case 4: return {4*xi[1], 4*xi[0]};
        case 5: return {-4*xi[1], -4*(-1+xi[0]+2*xi[1])};
      }
    }

    // cubic
    if constexpr (n == 4) {
      constexpr double sqrt5 = 2.23606797749978981;
      switch (i) {
        case 0:
          return {-6 - 15 * xi[0] * xi[0] + 4 * xi[0] * (5 - 8 * xi[1]) + (21 - 16 * xi[1]) * xi[1],
                  -6 - 16 * xi[0] * xi[0] + xi[0] * (21 - 32 * xi[1]) + 5 * (4 - 3 * xi[1]) * xi[1]};
        case 1:
          return {1 + 15 * xi[0] * xi[0] + xi[1] - xi[1] * xi[1] - 2 * xi[0] * (5 + xi[1]),
                  -(xi[0] * (-1 + xi[0] + 2 * xi[1]))};
        case 2:
          return {-(xi[1] * (-1 + 2 * xi[0] + xi[1])),
                  1 + xi[0] - xi[0] * xi[0] - 2 * (5 + xi[0]) * xi[1] + 15 * xi[1] * xi[1]};
        case 3:
          return {15 * sqrt5 * xi[0] * xi[0] + 5 * xi[0] * (-1 - 3 * sqrt5 + 3 * (1 + sqrt5) * xi[1]) +
                      (5 * (-1 + xi[1]) * (-1 - sqrt5 + (3 + sqrt5) * xi[1])) / 2.,
                  (5 * xi[0] * (-2 * (2 + sqrt5) + 3 * (1 + sqrt5) * xi[0] + 2 * (3 + sqrt5) * xi[1])) / 2.};
        case 4:
          return {(5 * (-3 * (25 + 13 * sqrt5) * xi[0] * xi[0] + (-1 + xi[1]) * (6 + 4 * sqrt5 + (7 + sqrt5) * xi[1]) +
                        xi[0] * (62 + 34 * sqrt5 - 12 * (3 + 2 * sqrt5) * xi[1]))) / (13 + 5 * sqrt5),
                  (-5 * xi[0] * (1 - 3 * sqrt5 + 6 * (3 + 2 * sqrt5) * xi[0] - 2 * (7 + sqrt5) * xi[1])) / (13 + 5 * sqrt5)};
        case 5:
          return {(5 * xi[1] * (-3 - sqrt5 + 2 * (7 + 3 * sqrt5) * xi[0] + 2 * xi[1])) / (3 + sqrt5),
                  (5 * xi[0] * (-3 - sqrt5 + (7 + 3 * sqrt5) * xi[0] + 4 * xi[1])) / (3 + sqrt5)};
        case 6:
          return {(5 * xi[1] * (-3 - sqrt5 + 4 * xi[0] + (7 + 3 * sqrt5) * xi[1])) / (3 + sqrt5),
                  (5 * xi[0] * (-3 - sqrt5 + 2 * xi[0] + 2 * (7 + 3 * sqrt5) * xi[1])) / (3 + sqrt5)};
        case 7:
          return {(5 * (-3 + sqrt5) * xi[1] * (1 - sqrt5 - 4 * xi[0] + 3 * (1 + sqrt5) * xi[1])) / 4.0,
                  (-5 * (-1 + sqrt5 + (-3 + sqrt5) * xi[0] * xi[0] + 2 * xi[1] * (1 - 3 * sqrt5 + 3 * sqrt5 * xi[1]) +
                         xi[0] * (4 - 2 * sqrt5 + 6 * (-1 + sqrt5) * xi[1]))) / 2.0};
        case 8:
          return {(5 * xi[1] * (-2 * (2 + sqrt5) + 2 * (3 + sqrt5) * xi[0] + 3 * (1 + sqrt5) * xi[1])) / 2.,
                  (5 * (1 + sqrt5 - 2 * (2 + sqrt5) * xi[0] + (3 + sqrt5) * xi[0] * xi[0] +
                        6 * (1 + sqrt5) * xi[0] * xi[1] + 2 * xi[1] * (-1 - 3 * sqrt5 + 3 * sqrt5 * xi[1]))) / 2.0};
        case 9:
          return {-27 * xi[1] * (-1 + 2 * xi[0] + xi[1]), -27 * xi[0] * (-1 + xi[0] + 2 * xi[1])};
      }
    }

    return {};

  }

  SERAC_HOST_DEVICE static constexpr tensor<double, ndof> shape_functions(tensor<double, dim> xi)
  {
    tensor< double, ndof > output{};
    for (int i = 0; i < ndof; i++) {
      output[i] = shape_function(xi, i);
    }
    return output;
  }

  SERAC_HOST_DEVICE static constexpr tensor<double, ndof, dim> shape_function_gradients(tensor<double, dim> xi)
  {
    tensor< double, ndof, dim > output{};
    for (int i = 0; i < ndof; i++) {
      output[i] = shape_function_gradient(xi, i);
    }
    return output;
  }

  /**
   * @brief B(i,j) is the
   *  jth shape function evaluated at the ith quadrature point
   *
   * @tparam apply_weights optionally multiply the rows of B by the associated quadrature weight
   * @tparam q the number of quadrature points along each dimension
   *
   * @return the matrix B of shape function evaluations
   */
  template <bool apply_weights, int q>
  static constexpr auto calculate_B()
  {
    constexpr auto points1D  = TriangleGaussLegendreNodes<q>();
    constexpr auto weights1D = TriangleGaussLegendreWeights<q>();

    tensor<double, q * (q + 1) / 2, ndof> B{};
    for (int i = 0; i < q * (q + 1) / 2; i++) {
      B[i] = shape_functions(points1D[i]) * ((apply_weights) ? weights1D[i] : 1.0);
    }
    return B;
  }

  /**
   * @brief B(i,j) is the gradient of the 
   *  jth shape function evaluated at the ith quadrature point
   *
   * @tparam apply_weights optionally multiply the rows of G by the associated quadrature weight
   * @tparam q the number of quadrature points along each dimension
   *
   * @return the matrix G of shape function evaluations
   */
  template <bool apply_weights, int q>
  static constexpr auto calculate_G()
  {
    constexpr auto points1D  = TriangleGaussLegendreNodes<q>();
    constexpr auto weights1D = TriangleGaussLegendreWeights<q>();

    tensor<double, q * (q + 1) / 2, ndof, dim > G{};
    for (int i = 0; i < q; i++) {
      G[i] = shape_function_gradients(points1D[i]) * ((apply_weights) ? weights1D[i] : 1.0);
    }
    return G;
  }

  template <typename in_t, int q>
  static auto batch_apply_shape_fn(int j, tensor<in_t, q * (q + 1) / 2> input, const TensorProductQuadratureRule<q>&)
  {
    using source_t = decltype(get<0>(get<0>(in_t{})) + dot(get<1>(get<0>(in_t{})), tensor<double, 2>{}));
    using flux_t   = decltype(get<0>(get<1>(in_t{})) + dot(get<1>(get<1>(in_t{})), tensor<double, 2>{}));

    constexpr auto xi = TriangleGaussLegendreNodes<q>();

    static constexpr int Q = q * (q + 1) / 2;
    tensor<tuple<source_t, flux_t>, Q> output;

    for (int i = 0; i < Q; i++) {
      double              phi_j      = shape_function(xi[i], j);
      tensor<double, dim> dphi_j_dxi = shape_function_gradient(xi[i], j);

      auto& d00 = get<0>(get<0>(input(i)));
      auto& d01 = get<1>(get<0>(input(i)));
      auto& d10 = get<0>(get<1>(input(i)));
      auto& d11 = get<1>(get<1>(input(i)));

      output[q] = {d00 * phi_j + dot(d01, dphi_j_dxi), d10 * phi_j + dot(d11, dphi_j_dxi)};
    }

    return output;
  }

  template <int q>
  static auto interpolate(const tensor< double, c, ndof > & X, const TensorProductQuadratureRule<q>&)
  {
    constexpr auto xi = TriangleGaussLegendreNodes<q>();
    static constexpr int num_quadrature_points = q * (q + 1) / 2;

    tensor< qf_input_type, num_quadrature_points > output{};

    for (int i = 0; i < c; i++) {
      for (int j = 0; j < num_quadrature_points; j++) {
        for (int k = 0; k < ndof; k++) {
          get<VALUE>(output[j])[i] += X(i, k) * shape_function(xi[j], k);
          get<GRADIENT>(output[j])[i] += X(i, k) * shape_function_gradient(xi[j], k);
        }
      }
    }

    return output;
  }

  template <typename source_type, typename flux_type, int q>
  static void integrate(const tensor<tuple<source_type, flux_type>, q * (q + 1) / 2>& qf_output,
                        const TensorProductQuadratureRule<q>&, const tensor< double, c, ndof > * element_residual, int step = 1)
  {
    if constexpr (is_zero<source_type>{} && is_zero<flux_type>{}) {
      return;
    }

    constexpr int ntrial = std::max(size(source_type{}), size(flux_type{}) / dim) / c;

    using s_buffer_type = std::conditional_t<is_zero<source_type>{}, zero, tensor<double, q, q> >;
    using f_buffer_type = std::conditional_t<is_zero<flux_type>{}, zero, tensor<double, dim, q, q> >;

    static constexpr bool apply_weights = true;
    static constexpr auto B             = calculate_B<apply_weights, q>();
    static constexpr auto G             = calculate_G<apply_weights, q>();

    for (int j = 0; j < ntrial; j++) {
      for (int i = 0; i < c; i++) {
        s_buffer_type source;
        f_buffer_type flux;

        for (int qy = 0; qy < q; qy++) {
          for (int qx = 0; qx < q; qx++) {
            int Q          = qy * q + qx;
            source(qy, qx) = reinterpret_cast<const double*>(&get<SOURCE>(qf_output[Q]))[i * ntrial + j];
            for (int k = 0; k < dim; k++) {
              flux(k, qy, qx) = reinterpret_cast<const double*>(&get<FLUX>(qf_output[Q]))[(i * dim + k) * ntrial + j];
            }
          }
        }

        auto A0 = contract<1, 0>(source, B) + contract<1, 0>(flux(0), G);
        auto A1 = contract<1, 0>(flux(1), B);

        element_residual[j * step](i) += contract<0, 0>(A0, B) + contract<0, 0>(A1, G);
      }
    }
  }

};
/// @endcond
