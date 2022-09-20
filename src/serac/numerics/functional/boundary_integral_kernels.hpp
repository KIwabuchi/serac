// Copyright (c) 2019-2022, Lawrence Livermore National Security, LLC and
// other Serac Project Developers. See the top-level LICENSE file for
// details.
//
// SPDX-License-Identifier: (BSD-3-Clause)
#pragma once

#include <array>

#include "serac/numerics/quadrature_data.hpp"
#include "serac/numerics/functional/integral_utilities.hpp"
#include "serac/numerics/functional/evector_view.hpp"

namespace serac {

namespace boundary_integral {

/**
 * @overload
 * @note This specialization of detail::Preprocess is called when doing integrals
 * where the spatial dimension is different from the dimension of the element geometry
 * (i.e. surface integrals in 3D space, line integrals in 2D space, etc)
 *
 * TODO: provide gradients as well (needs some more info from mfem)
 */
template <typename element_type, typename T, typename coord_type>
auto Preprocess(const T& u, const coord_type& xi)
{
  if constexpr (element_type::family == Family::H1 || element_type::family == Family::L2) {
    return serac::tuple{dot(u, element_type::shape_functions(xi)), serac::zero{}};
  }

  // we can't support HCURL until some issues in mfem are fixed
  // if constexpr (element_type::family == Family::HCURL) {
  //  return dot(u, dot(element_type::shape_functions(xi), inv(J)));
  //}
}

template <Geometry geom, typename... trials, typename tuple_type, int dim, int... i>
auto PreprocessHelper(const tuple_type& u, const tensor<double, dim>& xi, std::integer_sequence<int, i...>)
{
  return serac::make_tuple(Preprocess<finite_element<geom, trials>>(get<i>(u), xi)...);
}

template <Geometry geom, typename... trials, typename tuple_type, int dim>
auto Preprocess(const tuple_type& u, const tensor<double, dim>& xi)
{
  return PreprocessHelper<geom, trials...>(u, xi,
                                           std::make_integer_sequence<int, static_cast<int>(sizeof...(trials))>{});
}

/**
 * @overload
 * @note This specialization of detail::Postprocess is called when doing integrals
 * where the spatial dimension is different from the dimension of the element geometry
 * (i.e. surface integrals in 3D space, line integrals in 2D space, etc)
 *
 * In this case, q-function outputs are only integrated against test space shape functions
 */
template <typename element_type, typename T, typename coord_type>
auto Postprocess(const T& f, [[maybe_unused]] const coord_type& xi)
{
  if constexpr (element_type::family == Family::H1 || element_type::family == Family::L2) {
    return outer(element_type::shape_functions(xi), f);
  }

  // we can't support HCURL until fixing some shortcomings in mfem
  // if constexpr (element_type::family == Family::HCURL) {
  //  return outer(element_type::shape_functions(xi), dot(inv(J), f));
  //}

  if constexpr (element_type::family == Family::QOI) {
    return f;
  }
}

/**
 *  @tparam space the user-specified trial space
 *  @tparam dimension describes whether the problem is 1D, 2D, or 3D
 *
 *  @brief a struct used to encode what type of arguments will be passed to a domain integral q-function, for the given
 * trial space
 */
template <typename space, typename dimension>
struct QFunctionArgument;

/// @overload
template <int p, int dim>
struct QFunctionArgument<H1<p, 1>, Dimension<dim>> {
  using type = serac::tuple<double, tensor<double, dim> >;  ///< what will be passed to the q-function
};

/// @overload
template <int p, int c, int dim>
struct QFunctionArgument<H1<p, c>, Dimension<dim>> {
  using type = serac::tuple<tensor<double, c>, tensor<double, c, dim> >;  ///< what will be passed to the q-function
};

template <int i, int dim, typename... trials, typename lambda>
auto get_derivative_type(lambda qf)
{
  using qf_arguments = serac::tuple<typename QFunctionArgument<trials, serac::Dimension<dim>>::type...>;
  return get_gradient(
      detail::apply_qf(qf, tensor<double, dim + 1>{}, tensor<double, dim + 1>{}, make_dual_wrt<i>(qf_arguments{})));
};

template <int i>
struct DerivativeWRT {
};

template <int Q, Geometry g, typename test, typename... trials>
struct KernelConfig {
};

template <typename lambda, int dim, int n, typename ... T, int ... I >
auto batch_apply_qf(lambda qf, const tensor< double, dim, n > & positions, const tensor< double, dim, n > & normals, const tuple < tensor<T, n> ... > & inputs, std::integer_sequence< int, I ... >)
{
  using return_type = decltype(qf(tensor<double,dim>{}, tensor<double,dim>{}, T{} ...));
  tensor<tuple<return_type, zero>, n> outputs{};
  for (int i = 0; i < n; i++) {
    tensor< double, dim > x_q;
    tensor< double, dim > n_q;
    for (int j = 0; j < dim; j++) { 
      x_q[j] = positions(j, i); 
      n_q[j] = normals(j, i); 
    }
    get<0>(outputs[i]) = qf(x_q, n_q, get<I>(inputs)[i] ...);
  }
  return outputs;
}

template <int index_to_differentiate, typename derivative_type, typename lambda, int dim, int n, typename ... T, int ... I >
auto batch_apply_qf_with_AD(derivative_type * qf_derivatives, lambda qf, const tensor< double, dim, n > & positions, const tensor< double, dim, n > & normals, const tuple < tensor<T, n> ... > & inputs, std::integer_sequence< int, I ... >)
{
  using return_type = decltype(qf(tensor<double,dim>{}, tensor<double,dim>{}, T{} ...));
  tensor<tuple< return_type, zero >, n> outputs{};
  for (int i = 0; i < n; i++) {
    tensor< double, dim > x_q;
    tensor< double, dim > n_q;
    for (int j = 0; j < dim; j++) { 
      x_q[j] = positions(j, i); 
      n_q[j] = normals(j, i); 
    }
    auto outputs_and_derivatives = qf(x_q, n_q, promote_to_dual_when< I == index_to_differentiate >(get<I>(inputs)[i]) ...);
    get<0>(outputs[i]) = get_value(outputs_and_derivatives);
    qf_derivatives[i] = get_gradient(outputs_and_derivatives);
  }
  return outputs;
}

/**
 * @tparam S type used to specify which argument to differentiate with respect to.
 *    `void` => evaluation kernel with no differentiation
 *    `DerivativeWRT<i>` => evaluation kernel with AD applied to trial space `i`
 * @tparam T the "function signature" of the form `test(trial0, trial1, ...)`
 * @tparam derivatives_type the type of the derivative of the q-function
 * @tparam lambda the type of the q-function
 *
 * @brief Functor type providing a callback for the evaluation of the user-specified q-function over the domain
 */
template <typename S, typename T, typename derivatives_type, typename lambda>
struct EvaluationKernel;

/**
 * @overload
 * @note evaluation kernel with no differentiation
 */
template <int Q, Geometry geom, typename test, typename... trials, typename lambda>
struct EvaluationKernel<void, KernelConfig<Q, geom, test, trials...>, void, lambda> {
  static constexpr auto exec             = ExecutionSpace::CPU;     ///< this specialization is CPU-specific
  static constexpr int  num_trial_spaces = int(sizeof...(trials));  ///< how many trial spaces are provided
  static constexpr auto Iseq = std::make_integer_sequence<int, sizeof ... (trials)>{};

  using test_element = finite_element<geom, test>;
  static constexpr type_list < finite_element< geom, trials > ... > trial_elements{};

  /**
   * @brief initialize the functor by providing the necessary quadrature point data
   *
   * @param J values of sqrt(det(J^T * J)) at each quadrature point
   * @param X Spatial positions of each quadrature point
   * @param N Unit surface normals at each quadrature point
   * @param num_elements how many elements in the domain
   * @param qf q-function
   */
  EvaluationKernel(KernelConfig<Q, geom, test, trials...>, const mfem::Vector& J, const mfem::Vector& X,
                   const mfem::Vector& N, std::size_t num_elements, lambda qf)
      : J_(J), X_(X), N_(N), num_elements_(num_elements), qf_(qf)
  {
  }

  /**
   * @brief integrate the q-function over the specified domain, at the specified trial space values
   *
   * @param U input E-vectors
   * @param R output E-vector
   */
  void operator()(const std::vector<const mfem::Vector*> U, mfem::Vector& R)
  {
    std::array<const double*, num_trial_spaces> ptrs{};
    if constexpr (num_trial_spaces > 0) {
      for (uint32_t j = 0; j < num_trial_spaces; j++) {
        ptrs[j] = U[j]->Read();
      }
    }
    EVector_t u(ptrs, std::size_t(num_elements_));

    using test_element              = finite_element<geom, test>;
    using element_residual_type     = typename test_element::residual_type;
    static constexpr int  dim       = dimension_of(geom);
    static constexpr int  test_ndof = test_element::ndof;
    static constexpr auto rule      = GaussQuadratureRule<geom, Q>();

    // mfem provides this information in 1D arrays, so we reshape it
    // into strided multidimensional arrays before using
    constexpr int dim = dimension_of(geom);
    constexpr int nqp = num_quadrature_points<geom, Q>();
    auto J = reinterpret_cast<const tensor< double, nqp > *>(J_.Read());
    auto X = reinterpret_cast<const tensor< double, dim, nqp > *>(X_.Read());
    auto N = reinterpret_cast<const tensor< double, dim, nqp > *>(N_.Read());
    auto r = reinterpret_cast<typename test_element::dof_type*>(R.ReadWrite());
    static constexpr TensorProductQuadratureRule<Q> rule{};

    // for each element in the domain
    for (uint32_t e = 0; e < num_elements_; e++) {

      // load the jacobians, positions and normals for each quadrature point in this element
      auto J_e = J[e];
      auto X_e = X[e];
      auto N_e = N[e];

      tuple < 
        decltype(finite_element< geom, trials >::interpolate(typename finite_element< geom, trials >::dof_type{}, rule)) ... 
      > qf_inputs{};

      for_constexpr< num_trial_spaces >([&](auto j){

        using trial_element = decltype(trial_elements[j]);
        
        auto u = reinterpret_cast<const typename trial_element::dof_type*>(U[j].Read());

        // (batch) interpolate each quadrature point's value
        get<j>(qf_inputs) = trial_element::interpolate(u[e], rule);

      });

      // (batch) evalute the q-function at each quadrature point
      auto qf_outputs = batch_apply_qf(qf_, X_e, N_e, qf_inputs, Iseq);

      qf_outputs = elementwise_multiply(qf_outputs, J_e);

      // (batch) integrate the material response against the test-space basis functions
      test_element::integrate(qf_outputs, rule, r[e]);

    }

  }

  const mfem::Vector& J_;             ///< values of sqrt(det(J^T * J)) at each quadrature point
  const mfem::Vector& X_;             ///< Spatial positions of each quadrature point
  const mfem::Vector& N_;             ///< Unit surface normals at each quadrature point
  std::size_t         num_elements_;  ///< how many elements in the domain
  lambda              qf_;            ///< q-function
};

/**
 * @overload
 * @note evaluation kernel that also calculates derivative w.r.t. `I`th trial space
 */
template <int I, int Q, Geometry geom, typename test, typename... trials, typename derivatives_type, typename lambda>
struct EvaluationKernel<DerivativeWRT<I>, KernelConfig<Q, geom, test, trials...>, derivatives_type, lambda> {
  static constexpr auto exec             = ExecutionSpace::CPU;     ///< this specialization is CPU-specific
  static constexpr int  num_trial_spaces = static_cast<int>(sizeof...(trials));  ///< how many trial spaces are provided
  static constexpr auto Iseq = std::make_integer_sequence<int, sizeof ... (trials)>{};

  using test_element = finite_element<geom, test>;
  static constexpr type_list < finite_element< geom, trials > ... > trial_elements{};

  /**
   * @brief initialize the functor by providing the necessary quadrature point data
   *
   * @param qf_derivatives a container for the derivatives of the q-function w.r.t. trial space I
   * @param J values of sqrt(det(J^T * J)) at each quadrature point
   * @param X Spatial positions of each quadrature point
   * @param N Unit surface normals at each quadrature point
   * @param num_elements how many elements in the domain
   * @param qf q-function
   */
  EvaluationKernel(DerivativeWRT<I>, KernelConfig<Q, geom, test, trials...>,
                   CPUArrayView<derivatives_type, 2> qf_derivatives, const mfem::Vector& J, const mfem::Vector& X,
                   const mfem::Vector& N, std::size_t num_elements, lambda qf)
      : qf_derivatives_(qf_derivatives), J_(J), X_(X), N_(N), num_elements_(num_elements), qf_(qf)
  {
  }

  /**
   * @brief integrate the q-function over the specified domain, at the specified trial space values
   *
   * @param U input E-vectors
   * @param R output E-vector
   */
  void operator()(const std::vector<const mfem::Vector*> U, mfem::Vector& R)
  {
    // mfem provides this information in 1D arrays, so we reshape it
    // into strided multidimensional arrays before using
    constexpr int dim = dimension_of(geom);
    constexpr int nqp = num_quadrature_points<geom, Q>();
    auto J = reinterpret_cast<const tensor< double, nqp > *>(J_.Read());
    auto X = reinterpret_cast<const tensor< double, dim, nqp > *>(X_.Read());
    auto N = reinterpret_cast<const tensor< double, dim, nqp > *>(N_.Read());
    auto r = reinterpret_cast<typename test_element::dof_type*>(R.ReadWrite());
    static constexpr TensorProductQuadratureRule<Q> rule{};

        auto u = reinterpret_cast<const typename trial_element::dof_type*>(U[j].Read());

        // (batch) interpolate each quadrature point's value
        get<j>(qf_inputs) = trial_element::interpolate(u[e], rule);

      });

      // (batch) evalute the q-function at each quadrature point
      auto qf_outputs = batch_apply_qf_with_AD< I >(&qf_derivatives_(e, 0), qf_, X_e, N_e, qf_inputs, Iseq);

      qf_outputs = elementwise_multiply(qf_outputs, J_e);

      // (batch) integrate the material response against the test-space basis functions
      test_element::integrate(qf_outputs, rule, r[e]);

    }

  }

  ExecArrayView<derivatives_type, 2, exec> qf_derivatives_;  ///< derivatives of the q-function w.r.t. trial space `I`
  const mfem::Vector&                      J_;               ///< values of sqrt(det(J^T * J)) at each quadrature point
  const mfem::Vector&                      X_;               ///< Spatial positions of each quadrature point
  const mfem::Vector&                      N_;               ///< Unit surface normals at each quadrature point
  std::size_t                              num_elements_;    ///< how many elements in the domain
  lambda                                   qf_;              ///< q-function
};

template <int Q, Geometry geom, typename test, typename... trials, typename lambda>
EvaluationKernel(KernelConfig<Q, geom, test, trials...>, const mfem::Vector&, const mfem::Vector&, const mfem::Vector&,
                 int, lambda) -> EvaluationKernel<void, KernelConfig<Q, geom, test, trials...>, void, lambda>;

template <int i, int Q, Geometry geom, typename test, typename... trials, typename derivatives_type, typename lambda>
EvaluationKernel(DerivativeWRT<i>, KernelConfig<Q, geom, test, trials...>, CPUArrayView<derivatives_type, 2>,
                 const mfem::Vector&, const mfem::Vector&, const mfem::Vector&, int, lambda)
    -> EvaluationKernel<DerivativeWRT<i>, KernelConfig<Q, geom, test, trials...>, derivatives_type, lambda>;

//clang-format off
template <typename S, typename T>
auto chain_rule(const S& dfdx, const T& dx)
{
  return serac::chain_rule(serac::get<0>(dfdx), serac::get<0>(dx)) +
         serac::chain_rule(serac::get<1>(dfdx), serac::get<1>(dx));
}
//clang-format on

template <typename derivative_type, int n, typename T >
auto batch_apply_chain_rule(derivative_type * qf_derivatives, const tensor<T, n> & inputs)
{
  using return_type = decltype(chain_rule(derivative_type{}, T{}));
  tensor< tuple< return_type, zero >, n> outputs{};
  for (int i = 0; i < n; i++) {
    get<0>(outputs[i]) = chain_rule(qf_derivatives[i], inputs[i]);
  }
  return outputs;
}

/**
 * @brief The base kernel template used to create create custom directional derivative
 * kernels associated with finite element calculations
 *
 * @tparam test The type of the test function space
 * @tparam trial The type of the trial function space
 * The above spaces can be any combination of {H1, Hcurl, Hdiv (TODO), L2 (TODO)}
 *
 * Template parameters other than the test and trial spaces are used for customization + optimization
 * and are erased through the @p std::function members of @p BoundaryIntegral
 * @tparam g The shape of the element (only quadrilateral and hexahedron are supported at present)
 * @tparam Q Quadrature parameter describing how many points per dimension
 * @tparam derivatives_type Type representing the derivative of the q-function w.r.t. its input arguments
 *
 * @note lambda does not appear as a template argument, as the directional derivative is
 * inherently just a linear transformation
 *
 * @param[in] dU The full set of per-element DOF values (primary input)
 * @param[inout] dR The full set of per-element residuals (primary output)
 * @param[in] derivatives_ptr The address at which derivatives of the q-function with
 * respect to its arguments are stored
 * @param[in] J_ The Jacobians of the element transformations at all quadrature points
 * @see mfem::GeometricFactors
 * @param[in] num_elements The number of elements in the mesh
 */
template <Geometry g, typename test, typename trial, int Q, typename derivatives_type>
void action_of_gradient_kernel(const mfem::Vector& dU, mfem::Vector& dR,
                               CPUArrayView<derivatives_type, 2> qf_derivatives, const mfem::Vector& J_,
                               std::size_t num_elements)
{
  using test_element  = finite_element<g, test>;
  using trial_element = finite_element<g, trial>;

  // mfem provides this information in 1D arrays, so we reshape it
  // into strided multidimensional arrays before using
  constexpr int nqp = num_quadrature_points<g, Q>();
  auto J = reinterpret_cast<const tensor< double, nqp > *>(J_.Read());
  auto du = reinterpret_cast<const typename trial_element::dof_type*>(dU.Read());
  auto dr = reinterpret_cast<typename test_element::dof_type*>(dR.ReadWrite());
  static constexpr TensorProductQuadratureRule<Q> rule{};

  // for each element in the domain
  for (uint32_t e = 0; e < num_elements; e++) {

    // load the jacobians and positions for each quadrature point in this element
    auto J_e = J[e];

    // (batch) interpolate each quadrature point's value
    auto qf_inputs = trial_element::interpolate(du[e], rule);

    // (batch) evalute the q-function at each quadrature point
    auto qf_outputs = batch_apply_chain_rule(&qf_derivatives(e, 0), qf_inputs);

    qf_outputs = elementwise_multiply(qf_outputs, J_e);

    // (batch) integrate the material response against the test-space basis functions
    test_element::integrate(qf_outputs, rule, dr[e]);
  
  }

}

/**
 * @brief The base kernel template used to create create custom element stiffness matrices
 * associated with finite element calculations
 *
 * @tparam test The type of the test function space
 * @tparam trial The type of the trial function space
 * The above spaces can be any combination of {H1, Hcurl, Hdiv (TODO), L2 (TODO), QOI}
 *
 * Template parameters other than the test and trial spaces are used for customization + optimization
 * and are erased through the @p std::function members of @p BoundaryIntegral
 * @tparam g The shape of the element (only quadrilateral and hexahedron are supported at present)
 * @tparam Q Quadrature parameter describing how many points per dimension
 * @tparam derivatives_type Type representing the derivative of the q-function w.r.t. its input arguments
 *
 * @note lambda does not appear as a template argument, as the stiffness matrix is
 * inherently just a linear transformation
 *
 * @param[in] dk array for storing each element's gradient contributions
 * @param[in] derivatives_ptr The address at which derivatives of the q-function with
 * respect to its arguments are stored
 * @param[in] J_ The Jacobians of the element transformations at all quadrature points
 * @see mfem::GeometricFactors
 * @param[in] num_elements The number of elements in the mesh
 */
template <Geometry g, typename test, typename trial, int Q, typename derivatives_type>
void element_gradient_kernel(CPUArrayView<double, 3> dk, CPUArrayView<derivatives_type, 2> qf_derivatives,
                             const mfem::Vector& J_, std::size_t num_elements)
{
  using test_element               = finite_element<g, test>;
  using trial_element              = finite_element<g, trial>;
  static constexpr int  test_ndof  = test_element::ndof;
  static constexpr int  test_dim   = test_element::components;
  static constexpr int  trial_ndof = trial_element::ndof;
  static constexpr int  trial_dim  = trial_element::components;
  static constexpr auto rule       = GaussQuadratureRule<g, Q>();

  // mfem provides this information in 1D arrays, so we reshape it
  // into strided multidimensional arrays before using
  auto J = mfem::Reshape(J_.Read(), rule.size(), num_elements);

  // for each element in the domain
  for (size_t e = 0; e < num_elements; e++) {
    tensor<double, test_ndof, trial_ndof, test_dim, trial_dim> K_elem{};

    // for each quadrature point in the element
    for (int q = 0; q < static_cast<int>(rule.size()); q++) {
      // get the position of this quadrature point in the parent and physical space,
      // and calculate the measure of that point in physical space.
      auto   xi_q  = rule.points[q];
      auto   dxi_q = rule.weights[q];
      double dx    = J(q, e) * dxi_q;

      // recall the derivative of the q-function w.r.t. its arguments at this quadrature point
      auto dq_darg = qf_derivatives(static_cast<size_t>(e), static_cast<size_t>(q));

      if constexpr (std::is_same<test, QOI>::value) {
        auto N = trial_element::shape_functions(xi_q);
        for (int j = 0; j < trial_ndof; j++) {
          K_elem[0][j][0] += serac::get<0>(dq_darg) * N[j] * dx;
        }
      } else {
        auto M = test_element::shape_functions(xi_q);
        auto N = trial_element::shape_functions(xi_q);
        auto dN = trial_element::shape_function_gradients(xi_q);

        for (int i = 0; i < test_ndof; i++) {
          for (int j = 0; j < trial_ndof; j++) {
            K_elem[i][j] += M[i] * (serac::get<0>(dq_darg) *  N[j] + 
                                    serac::get<1>(dq_darg) * dN[j]) * dx;
          }
        }
      }
    }

    // once we've finished the element integration loop, write our element gradients
    // out to memory, to be later assembled into the global gradient by mfem
    //
    // Note: we "transpose" these values to get them into the layout that mfem expects
    // clang-format off
    for_loop<test_ndof, test_dim, trial_ndof, trial_dim>([e, &dk, &K_elem](int i, int j, int k, int l) {
      dk(static_cast<size_t>(e), static_cast<size_t>(i + test_ndof * j), static_cast<size_t>(k + trial_ndof * l)) += K_elem[i][k][j][l];
    });
    // clang-format on
  }
}

}  // namespace boundary_integral

}  // namespace serac
