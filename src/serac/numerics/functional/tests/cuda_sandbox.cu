#include <mfem.hpp>

#include "serac/numerics/functional/tensor.hpp"

namespace serac {

enum class Family { 
  H1, 
  Hcurl, 
  Hdiv, 
  DG
};

template < mfem::Geometry::Type g, Family f >
struct FiniteElement;

template <>
struct FiniteElement< mfem::Geometry::TRIANGLE, Family::H1 > {

  using source_type = double;
  using flux_type = vec2;

  static constexpr int dim = 2;

  uint32_t num_nodes() const { return ((p + 1) * (p + 2)) / 2; }

  constexpr double shape_function(vec2 xi, uint32_t i) const {
    if (p == 1) {
      if (i == 0) return 1.0 - xi[0] - xi[1];
      if (i == 1) return xi[0];
      if (i == 2) return xi[1];
    }
    if (p == 2) {
      if (i == 0) return (-1+xi[0]+xi[1])*(-1+2*xi[0]+2*xi[1]);
      if (i == 1) return -4*xi[0]*(-1+xi[0]+xi[1]);
      if (i == 2) return xi[0]*(-1+2*xi[0]);
      if (i == 3) return -4*xi[1]*(-1+xi[0]+xi[1]);
      if (i == 4) return 4*xi[0]*xi[1];
      if (i == 5) return xi[1]*(-1+2*xi[1]);
    }
    if (p == 3) {
      double sqrt5 = 2.23606797749978981;
      if (i == 0) return -((-1+xi[0]+xi[1])*(1+5*xi[0]*xi[0]+5*(-1+xi[1])*xi[1]+xi[0]*(-5+11*xi[1])));
      if (i == 1) return (5*xi[0]*(-1+xi[0]+xi[1])*(-1-sqrt5+2*sqrt5*xi[0]+(3+sqrt5)*xi[1]))/2.0;
      if (i == 2) return (-5*xi[0]*(-1+xi[0]+xi[1])*(1-sqrt5+2*sqrt5*xi[0]+(-3+sqrt5)*xi[1]))/2.0;
      if (i == 3) return xi[0]*(1+5*xi[0]*xi[0]+xi[1]-xi[1]*xi[1]-xi[0]*(5+xi[1]));
      if (i == 4) return (5*xi[1]*(-1+xi[0]+xi[1])*(-1-sqrt5+(3+sqrt5)*xi[0]+2*sqrt5*xi[1]))/2.0;
      if (i == 5) return -27*xi[0]*xi[1]*(-1+xi[0]+xi[1]);
      if (i == 6) return (5*xi[0]*xi[1]*(-2+(3+sqrt5)*xi[0]-(-3+sqrt5)*xi[1]))/2.;
      if (i == 7) return (5*xi[1]*(-1+xi[0]+xi[1])*(5-3*sqrt5+2*(-5+2*sqrt5)*xi[0]+5*(-1+sqrt5)*xi[1]))/(-5+sqrt5);
      if (i == 8) return (-5*xi[0]*xi[1]*(2+(-3+sqrt5)*xi[0]-(3+sqrt5)*xi[1]))/2.;
      if (i == 9) return xi[1]*(1+xi[0]-xi[0]*xi[0]-xi[0]*xi[1]+5*(-1+xi[1])*xi[1]);
    }

    return -1.0;
  }

  vec2 shape_function_gradient(vec2 xi, uint32_t i) const {
    // expressions generated symbolically by mathematica
    if (p == 1) {
      if (i == 0) return {-1.0, -1.0};
      if (i == 1) return { 1.0,  0.0};
      if (i == 2) return { 0.0,  1.0};
    }
    if (p == 2) {
      if (i == 0) return {-3+4*xi[0]+4*xi[1], -3+4*xi[0]+4*xi[1]};
      if (i == 1) return {-4*(-1+2*xi[0]+xi[1]), -4*xi[0]};
      if (i == 2) return {-1+4*xi[0], 0};
      if (i == 3) return {-4*xi[1], -4*(-1+xi[0]+2*xi[1])};
      if (i == 4) return {4*xi[1], 4*xi[0]};
      if (i == 5) return {0, -1+4*xi[1]};
    }
    if (p == 3) {
      double sqrt5 = 2.23606797749978981;
      if (i == 0) return {-6-15*xi[0]*xi[0]+4*xi[0]*(5-8*xi[1])+(21-16*xi[1])*xi[1], -6-16*xi[0]*xi[0]+xi[0]*(21-32*xi[1])+5*(4-3*xi[1])*xi[1]};
      if (i == 1) return {(5*(6*sqrt5*xi[0]*xi[0]+xi[0]*(-2-6*sqrt5+6*(1+sqrt5)*xi[1])+(-1+xi[1])*(-1-sqrt5+(3+sqrt5)*xi[1])))/2., (5*xi[0]*(-2*(2+sqrt5)+3*(1+sqrt5)*xi[0]+2*(3+sqrt5)*xi[1]))/2.};
      if (i == 2) return {(-5*(6*sqrt5*xi[0]*xi[0]+(-1+xi[1])*(1-sqrt5+(-3+sqrt5)*xi[1])+xi[0]*(2-6*sqrt5+6*(-1+sqrt5)*xi[1])))/2., (-5*xi[0]*(4-2*sqrt5+3*(-1+sqrt5)*xi[0]+2*(-3+sqrt5)*xi[1]))/2.};
      if (i == 3) return {1+15*xi[0]*xi[0]+xi[1]-xi[1]*xi[1]-2*xi[0]*(5+xi[1]), -(xi[0]*(-1+xi[0]+2*xi[1]))};
      if (i == 4) return {(5*xi[1]*(-2*(2+sqrt5)+2*(3+sqrt5)*xi[0]+3*(1+sqrt5)*xi[1]))/2., (5*(1+sqrt5-2*(2+sqrt5)*xi[0]+(3+sqrt5)*xi[0]*xi[0]+6*(1+sqrt5)*xi[0]*xi[1]+2*xi[1]*(-1-3*sqrt5+3*sqrt5*xi[1])))/2.};
      if (i == 5) return {-27*xi[1]*(-1+2*xi[0]+xi[1]), -27*xi[0]*(-1+xi[0]+2*xi[1])};
      if (i == 6) return {(-5*xi[1]*(2-2*(3+sqrt5)*xi[0]+(-3+sqrt5)*xi[1]))/2., (5*xi[0]*(-2+(3+sqrt5)*xi[0]-2*(-3+sqrt5)*xi[1]))/2.};
      if (i == 7) return {(-5*xi[1]*(4-2*sqrt5+2*(-3+sqrt5)*xi[0]+3*(-1+sqrt5)*xi[1]))/2., (-5*(-1+sqrt5+(-3+sqrt5)*xi[0]*xi[0]+2*xi[1]*(1-3*sqrt5+3*sqrt5*xi[1])+xi[0]*(4-2*sqrt5+6*(-1+sqrt5)*xi[1])))/2.};
      if (i == 8) return {(5*xi[1]*(-2-2*(-3+sqrt5)*xi[0]+(3+sqrt5)*xi[1]))/2., (-5*xi[0]*(2+(-3+sqrt5)*xi[0]-2*(3+sqrt5)*xi[1]))/2.};
      if (i == 9) return {-(xi[1]*(-1+2*xi[0]+xi[1])), 1+xi[0]-xi[0]*xi[0]-2*(5+xi[0])*xi[1]+15*xi[1]*xi[1]};
    }

    return {};
  }

#if 0
  nd::array< double, 2 > evaluate_shape_functions(nd::view<const double,2> xi) const {
    uint32_t q = xi.shape[0];
    nd::array<double, 2> shape_fns({q, num_nodes()});
    for (int i = 0; i < q; i++) {
      GaussLobattoInterpolationTriangle(&xi(i, 0), p, &shape_fns(i, 0));
    }
    return shape_fns;
  }

  nd::array< double, 3 > evaluate_shape_function_gradients(nd::view<const double, 2> xi) const {
    uint32_t q = xi.shape[0];
    nd::array<double, 3> shape_fn_grads({q, num_nodes(), dim});
    for (int i = 0; i < q; i++) {
      GaussLobattoInterpolationDerivativeTriangle(&xi(i, 0), p, &shape_fn_grads(i, 0, 0));
    }
    return shape_fn_grads;
  }

  void interpolate(nd::view<double, 1> values_q, nd::view<const double, 1> values_e, nd::view<const double, 2> shape_fns) const {
    int nnodes = num_nodes();
    int nqpts = values_q.shape[0];

    for (int q = 0; q < nqpts; q++) {
      double sum = 0.0;
      for (int i = 0; i < nnodes; i++) {
        sum += shape_fns(q, i) * values_e(i);
      }
      values_q(q) = sum;
    }
  }

  void gradient(nd::view<double, 2> gradients_q, nd::view<const double, 1> values_e, nd::view<const double, 3> shape_fn_grads) const {
    int nnodes = num_nodes();
    int nqpts = gradients_q.shape[0];
    for (int j = 0; j < nqpts; j++) {
      double sum[2]{};
      for (int i = 0; i < nnodes; i++) {
        sum[0] += values_e(i) * shape_fn_grads(j, i, 0);
        sum[1] += values_e(i) * shape_fn_grads(j, i, 1);
      }
      gradients_q(j, 0) = sum[0];
      gradients_q(j, 1) = sum[1];

    }
  }

  nd::array< double, 2 > evaluate_weighted_shape_functions(nd::view<const double,2> xi,
                                                           nd::view<const double,1> weights) const {
    uint32_t nnodes = num_nodes();
    uint32_t q = xi.shape[0];
    nd::array<double, 2> shape_fns({q, nnodes});
    for (uint32_t i = 0; i < q; i++) {
      GaussLobattoInterpolationTriangle(&xi(i, 0), p, &shape_fns(i, 0));
      for (uint32_t j = 0; j < nnodes; j++) {
        shape_fns(i, j) = shape_fns(i, j) * weights(i);
      }
    }
    return shape_fns;
  }

  void integrate_source(nd::view<double> residual_e, nd::view<const source_type> source_q, nd::view<const double, 2> shape_fn) const {
    int nnodes = num_nodes();
    int nqpts = source_q.shape[0];

    for (int i = 0; i < nnodes; i++) {
      double sum = 0.0;
      for (int q = 0; q < nqpts; q++) {
        sum += shape_fn(q, i) * source_q(q);
      }
      residual_e(i) = sum;
    }
  }

  nd::array< double, 3 > evaluate_weighted_shape_function_gradients(nd::view<const double,2> xi,
                                                                    nd::view<const double,1> weights) const {
    uint32_t nnodes = num_nodes();
    uint32_t q = xi.shape[0];
    nd::array<double, 3> shape_fn_grads({q, nnodes, dim});
    for (uint32_t i = 0; i < q; i++) {
      GaussLobattoInterpolationDerivativeTriangle(&xi(i, 0), p, &shape_fn_grads(i, 0, 0));
      for (uint32_t j = 0; j < nnodes; j++) {
        shape_fn_grads(i, j, 0) = shape_fn_grads(i, j, 0) * weights(i);
        shape_fn_grads(i, j, 1) = shape_fn_grads(i, j, 1) * weights(i);
      }
    }

    return shape_fn_grads;
  }

  void integrate_flux(nd::view<double> residual_e, nd::view<const flux_type> flux_q, nd::view<const double, 3> shape_fn_grads) const {
    int nnodes = num_nodes();
    int nqpts = flux_q.shape[0];

    for (int i = 0; i < nnodes; i++) {
      double sum = 0.0;
      for (int q = 0; q < nqpts; q++) {
        for (int d = 0; d < dim; d++) {
          sum += shape_fn_grads(q, i, d) * flux_q(q)[d];
        }
      }
      residual_e(i) = sum;
    }
  }
#endif

  int p;

};

}

//#include "elements/h1_edge.hpp"
//#include "elements/h1_triangle.hpp"
//#include "elements/h1_quadrilateral.hpp"
//#include "elements/h1_tetrahedron.hpp"
//#include "elements/h1_hexahedron.hpp"

__global__ void kernel() {}

int main() {
    kernel<<<1,1>>>();
}
