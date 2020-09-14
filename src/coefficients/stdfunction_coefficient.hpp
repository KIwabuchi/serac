// Copyright (c) 2019-2020, Lawrence Livermore National Security, LLC and
// other Serac Project Developers. See the top-level LICENSE file for
// details.
//
// SPDX-License-Identifier: (BSD-3-Clause)

/**
 * @file stdfunction_coefficient.hpp
 *
 * @brief MFEM coefficients and helper functions based on std::functions
 */

#ifndef STD_FUNCTION_COEFFICIENT_HPP
#define STD_FUNCTION_COEFFICIENT_HPP

#include <functional>
#include <memory>

#include "mfem.hpp"
#include "common/expr_template_ops.hpp"

namespace serac {

/**
 * @brief StdFunctionCoefficient is an easy way to make an mfem::Coefficient
 *  using a lambda
 *
 * This is a place holder until the coefficient of the same name is merged into
 * mfem.
 */
class StdFunctionCoefficient : public mfem::Coefficient {
public:

  /**
   * @brief Constructor that takes in a time-independent function of space that returns a double
   *
   * @param[in] func: a function of space and time that returns a double
   */
  StdFunctionCoefficient(std::function<double(mfem::Vector&)> func);

  /**
   * @brief Constructor that takes in a time-dependent function of space that returns a double
   *
   * @param[in] func: a function of space and time that returns a double
   */
  StdFunctionCoefficient(std::function<double(mfem::Vector&, double)> func);

  /**
   * @brief Evalate the coefficient at a quadrature point
   *
   * @param[in] Tr The element transformation for the evaluation
   * @param[in] ip The integration point for the evaluation
   * @return The value of the coefficient at the quadrature point
   */
  virtual double Eval(mfem::ElementTransformation& Tr, const mfem::IntegrationPoint& ip);
  
  bool is_time_dependent() const { return is_time_dependent_; } 

  friend StdFunctionCoefficient d_dt(const StdFunctionCoefficient &, const double);
  friend StdFunctionCoefficient d2_dt2(const StdFunctionCoefficient &, const double);

private:
  /**
   * @brief The function to evaluate for the coefficient
   */
  std::function<double(mfem::Vector&, double)> func_;
  bool is_time_dependent_;
};

inline StdFunctionCoefficient d_dt(const StdFunctionCoefficient & y, const double dt = 1.0e-8) {
  return StdFunctionCoefficient([dt, y = y.func_](mfem::Vector& x, double t) {
    return (y(x, t + dt) - y(x, t - dt)) / (2.0 * dt);
  });
}

inline StdFunctionCoefficient d2_dt2(const StdFunctionCoefficient & y, const double dt = 1.0e-4) {
  return StdFunctionCoefficient([dt, y = y.func_](mfem::Vector& x, double t) {
    return (y(x, t + dt) - 2 * y(x, t) + y(x, t - dt)) / (dt * dt);
  });
}

/**
 * @brief StdFunctionVectorCoefficient is an easy way to make an
 * mfem::VectorCoefficient using a lambda
 */
class StdFunctionVectorCoefficient : public mfem::VectorCoefficient {
public:
  /**
   * @brief StdFunctionVectorCoefficient is an easy way to make an
   * mfem::Coefficient using a lambda
   *
   * @param[in] dim The dimension of the VectorCoefficient
   * @param[in] func Is a function that matches the following prototype
   * void(mfem::Vector &, mfem::Vector &). The first argument of the function is
   * the position, and the second argument is the output of the function.
   */
  StdFunctionVectorCoefficient(int dim, std::function<void(mfem::Vector&, mfem::Vector&, double)> func);

  /**
   * @brief Evalate the coefficient at a quadrature point
   *
   * @param[out] V The evaluated coefficient vector at the quadrature point
   * @param[in] T The element transformation for the evaluation
   * @param[in] ip The integration point for the evaluation
   */
  virtual void Eval(mfem::Vector& V, mfem::ElementTransformation& T, const mfem::IntegrationPoint& ip);

  friend StdFunctionVectorCoefficient d_dt(const StdFunctionVectorCoefficient &, const double);
  friend StdFunctionVectorCoefficient d2_dt2(const StdFunctionVectorCoefficient &, const double);

private:
  /**
   * @brief The function to evaluate for the coefficient
   */
  std::function<void(mfem::Vector&, mfem::Vector&, double)> func_;
};

inline StdFunctionVectorCoefficient d_dt(const StdFunctionVectorCoefficient & y, const double dt = 1.0e-8) {
  return StdFunctionVectorCoefficient(y.vdim, [dt, y = y.func_](mfem::Vector& x, mfem::Vector & return_value, double t) {
    // if mfem would use return statements instead of 
    // always using out parameters, we could just write
    // return (y(x, t + dt) - y(x, t - dt)) / (2.0 * dt);

    // instead, we get this:
    mfem::Vector yl, yr;
    y(x, yl, t - dt);
    y(x, yr, t + dt);
    return_value = (yr - yl) * (1.0 / (2.0 * dt));
  });
}

inline StdFunctionVectorCoefficient d2_dt2(const StdFunctionVectorCoefficient & y, const double dt = 1.0e-4) {
  return StdFunctionVectorCoefficient(y.vdim, [dt, y = y.func_](mfem::Vector& x, mfem::Vector& return_value, double t) {
    // return_value = (y(x, t + dt) - 2 * y(x, t) + y(x, t - dt)) / (dt * dt);
    
    mfem::Vector yl, ym, yr;
    y(x, yl, t - dt);
    y(x, ym, t     );
    y(x, yr, t + dt);
    return_value = (yr - 2 * ym + yl) * (1.0 / (dt * dt));
  });
}

/**
 * @brief MakeTrueEssList takes in a FESpace, a vector coefficient, and produces a list
 *  of essential boundary conditions
 *
 * @param[in] pfes A finite element space for the constrained grid function
 * @param[in] c A VectorCoefficient that is projected on to the mesh. All
 * d.o.f's are examined and those that are the condition (> 0.) are appended to
 * the vdof list.
 * @return The list of true dofs that should be part of the essential boundary conditions
 */
mfem::Array<int> makeTrueEssList(mfem::ParFiniteElementSpace& pfes, mfem::VectorCoefficient& c);

/**
 * @brief MakeEssList takes in a FESpace, a vector coefficient, and produces a list
 * of essential boundary conditions
 *
 * @param[in] pfes A finite element space for the constrained grid function
 * @param[in] c A VectorCoefficient that is projected on to the mesh. All
 * d.o.f's are examined and those that are the condition (> 0.) are appended to
 * the vdof list.
 * @return The list of vector dofs that should be
 * part of the essential boundary conditions
 */
mfem::Array<int> makeEssList(mfem::ParFiniteElementSpace& pfes, mfem::VectorCoefficient& c);

/**
 * @brief This method creates an array of size(local_elems), and assigns
 * attributes based on the coefficient c
 *
 * This method is useful for creating lists of attributes that correspond to
 * elements in the mesh
 *
 * @param[in] m The mesh
 * @param[in] c The coefficient provided that will be evaluated on the mesh
 * @param[in] digitize An optional function that can be
 * called to assign attributes based on the value of c at a given projection
 * point. By default, values of c at a given d.o.f that are > 0. are assigned
 * attribute 2, otherwise attribute 1.
 * @return An array holding the attributes that correspond to each element
 */
mfem::Array<int> makeAttributeList(
    mfem::Mesh& m, mfem::Coefficient& c, std::function<int(double)> digitize = [](double v) { return v > 0. ? 2 : 1; });

/**
 * @brief This method creates an array of size(local_bdr_elems), and assigns
 * attributes based on the coefficient c
 *
 * This method is useful for creating lists of attributes that correspond to bdr
 * elements in the mesh
 *
 * @param[in] m The mesh
 * @param[in] c The coefficient provided that will be evaluated on the mesh
 * @param[in] digitize An optional function that can be
 * called to assign attributes based on the value of c at a given projection
 * point. By default, values of c at a given d.o.f that are ==1. are assigned
 * attribute 2, otherwise attribute 1. This means that only if all the d.o.f's
 * of an bdr_element are "tagged" 1, will this bdr element be assigned
 * attribute 2.
 * @return An array holding the attributes that correspond to each element
 */
mfem::Array<int> makeBdrAttributeList(
    mfem::Mesh& m, mfem::Coefficient& c,
    std::function<int(double)> digitize = [](double v) { return v == 1. ? 2 : 1; });

/**
 * @brief AttributemodifierCoefficient class
 *
 * This class temporarily changes the attribute to a given attribute list during
 * evaluation
 */
class AttributeModifierCoefficient : public mfem::Coefficient {
public:
  /**
   * @brief This class temporarily changes the attribute during coefficient
   * evaluation based on a given list.
   *
   * @param[in] attr_list A list of attributes values corresponding to the type
   * of coefficient at each element.
   * @param[in] c The coefficient to "modify" the element attributes
   */
  AttributeModifierCoefficient(const mfem::Array<int>& attr_list, mfem::Coefficient& c)
      : attr_list_(attr_list), coef_(c)
  {
  }

  /**
   * @brief Evalate the coefficient at a quadrature point
   *
   * @param[in] Tr The element transformation for the evaluation
   * @param[in] ip The integration point for the evaluation
   * @return The value of the coefficient at the quadrature point
   */
  virtual double Eval(mfem::ElementTransformation& Tr, const mfem::IntegrationPoint& ip);

protected:
  /**
   * @brief A list of attributes values corresponding to the type
   * of coefficient at each element.
   */
  const mfem::Array<int>& attr_list_;

  /**
   * @brief The coefficient to "modify" the element attributes
   */
  mfem::Coefficient& coef_;
};

/**
 * @brief Applies various operations to modify a
 * VectorCoefficient
 */
class TransformedVectorCoefficient : public mfem::VectorCoefficient {
public:
  /**
   * @brief Apply a vector function, Func, to v1
   *
   * @param[in] v1 A VectorCoefficient to apply Func to
   * @param[in] func A function that takes in an input vector, and returns the
   * output as the second argument.
   */
  TransformedVectorCoefficient(std::shared_ptr<mfem::VectorCoefficient>          v1,
                               std::function<void(mfem::Vector&, mfem::Vector&)> func);

  /**
   * @brief Apply a vector function, Func, to v1 and v2
   *
   * @param[in] v1 A VectorCoefficient to apply Func to
   * @param[in] v2 A VectorCoefficient to apply Func to
   * @param[in] func A function that takes in two input vectors, and returns the
   * output as the third argument.
   */
  TransformedVectorCoefficient(std::shared_ptr<mfem::VectorCoefficient> v1, std::shared_ptr<mfem::VectorCoefficient> v2,
                               std::function<void(mfem::Vector&, mfem::Vector&, mfem::Vector&)> func);

  /**
   * @brief Evalate the coefficient at a quadrature point
   *
   * @param[out] V The evaluated coefficient vector at the quadrature point
   * @param[in] T The element transformation for the evaluation
   * @param[in] ip The integration point for the evaluation
   */
  virtual void Eval(mfem::Vector& V, mfem::ElementTransformation& T, const mfem::IntegrationPoint& ip);

private:
  /**
   * @brief The first vector coefficient in the transformation
   */
  std::shared_ptr<mfem::VectorCoefficient> v1_;

  /**
   * @brief The first vector coefficient in the transformation
   */
  std::shared_ptr<mfem::VectorCoefficient> v2_;

  /**
   * @brief The one argument function for a transformed coefficient
   */
  std::function<void(mfem::Vector&, mfem::Vector&)> mono_function_;

  /**
   * @brief The two argument function for a transformed coefficient
   */
  std::function<void(mfem::Vector&, mfem::Vector&, mfem::Vector&)> bi_function_;
};

/**
 * @brief TransformedScalarCoefficient applies various operations to modify a
 * scalar Coefficient
 */
class TransformedScalarCoefficient : public mfem::Coefficient {
public:
  /**
   * @brief Apply a scalar function, Func, to s1
   *
   * @param[in] s1 A Coefficient to apply Func to
   * @param[in] func A function that takes in an input scalar, and returns the
   * output.
   */
  TransformedScalarCoefficient(std::shared_ptr<mfem::Coefficient> s1, std::function<double(const double)> func);

  /**
   * @brief Apply a scalar function, Func, to s1 and s2
   *
   * @param[in] s1 A scalar Coefficient to apply Func to
   * @param[in] s2 A scalar Coefficient to apply Func to
   * @param[in] func A function that takes in two input scalars, and returns the
   * output.
   */
  TransformedScalarCoefficient(std::shared_ptr<mfem::Coefficient> s1, std::shared_ptr<mfem::Coefficient> s2,
                               std::function<double(const double, const double)> func);

  /**
   * @brief Evaluate the coefficient at a quadrature point
   *
   * @param[in] T The element transformation for the evaluation
   * @param[in] ip The integration point for the evaluation
   * @return The value of the coefficient at the quadrature point
   */
  virtual double Eval(mfem::ElementTransformation& T, const mfem::IntegrationPoint& ip);

private:
  /**
   * @brief The first scalar coefficient in the transformation
   */
  std::shared_ptr<mfem::Coefficient> s1_;

  /**
   * @brief The second scalar coefficient in the transformation
   */
  std::shared_ptr<mfem::Coefficient> s2_;

  /**
   * @brief The one argument transformation function
   */
  std::function<double(const double)> mono_function_;

  /**
   * @brief The two argument transformation function
   */
  std::function<double(const double, const double)> bi_function_;
};



}  // namespace serac

#endif
