// Copyright (c) 2019-2024, Lawrence Livermore National Security, LLC and
// other Serac Project Developers. See the top-level LICENSE file for
// details.
//
// SPDX-License-Identifier: (BSD-3-Clause)

#include "serac/numerics/trust_region_solver.hpp"
#include <iostream>
#include "serac/infrastructure/profiling.hpp"
#include "serac/numerics/dense_petsc.hpp"

namespace serac {

int globalSize(const mfem::Vector& parallel_v, const MPI_Comm& comm)
{
  int local_size = parallel_v.Size();
  int global_size;
  MPI_Allreduce(&local_size, &global_size, 1, MPI_INT, MPI_SUM, comm);
  return global_size;
}

struct BasisVectors
{
  // construct with a representative state to set sizes
  BasisVectors(const mfem::Vector& state) 
    : local_rows(state.Size()), global_rows(globalSize(state, PETSC_COMM_WORLD))
  {
    VecCreateMPI(PETSC_COMM_WORLD, local_rows, global_rows, &v);

    PetscInt iStart, iEnd;
    VecGetOwnershipRange(v, &iStart, &iEnd);

    col_indices.reserve(static_cast<size_t>(local_rows));
    for (int i = iStart; i < iEnd; ++i) {
      col_indices.push_back(i);
    }
  }

  ~BasisVectors()
  {
    VecDestroy(&v);
  }

  BV constructBases(const std::vector<mfem::Vector*>& states) const {
    size_t num_cols = states.size();
    BV Q;
    BVCreate(PETSC_COMM_SELF, &Q);
    BVSetType(Q, BVVECS);
    BVSetSizesFromVec(Q, v, static_cast<int>(num_cols));
    for (size_t c=0; c < num_cols; ++c) {
      VecSetValues(v, local_rows, &col_indices[0], &(*states[c])[0], INSERT_VALUES);
      VecAssemblyBegin(v);
      VecAssemblyEnd(v);
      int c_int = static_cast<int>(c);
      BVInsertVec(Q, c_int, v);
    }
    return Q;
  }

private:

  const int local_rows;
  const int global_rows;
  
  std::vector<int> col_indices;
  Vec v;
};


Vec petscVec(const mfem::Vector& s)
{
  const int local_rows = s.Size();
  const int global_rows = globalSize(s, PETSC_COMM_WORLD);

  Vec v;
  VecCreateMPI(PETSC_COMM_WORLD, local_rows, global_rows, &v);

  PetscInt iStart, iEnd;
  VecGetOwnershipRange(v, &iStart, &iEnd);

  std::vector<int> col_indices;
  col_indices.reserve(static_cast<size_t>(local_rows));
  for (int i = iStart; i < iEnd; ++i) {
    col_indices.push_back(i);
  }

  VecSetValues(v, local_rows, &col_indices[0], &s[0], INSERT_VALUES);

  VecAssemblyBegin(v);
  VecAssemblyEnd(v);

  return v;
}

void copy(const Vec& v, mfem::Vector& s)
{
  const int local_rows = s.Size();
  PetscInt iStart, iEnd;
  VecGetOwnershipRange(v, &iStart, &iEnd);

  SLIC_ERROR_IF(local_rows != iEnd-iStart, "Inconsistency between local t-dof vector size and petsc start and end indices");

  std::vector<int> col_indices;
  col_indices.reserve(static_cast<size_t>(local_rows));
  for (int i = iStart; i < iEnd; ++i) {
    col_indices.push_back(i);
  }

  VecGetValues(v, local_rows, &col_indices[0], &s[0]);
}

/*
auto eigenOrthonormalize(const std::vector<serac::FiniteElementState>& states)
{
  const int local_rows = states[0].Size();
  const int global_rows = states[0].GlobalSize();

  size_t num_cols = states.size();

  SVD svd;
  SVDCreate(PETSC_COMM_WORLD, &svd);

  Mat Bases;
  MatCreate(PETSC_COMM_WORLD, &Bases);
  MatSetSizes(Bases, local_rows, static_cast<int>(num_cols), global_rows, static_cast<int>(num_cols));

  std::vector<Vec> left_vecs(num_cols);
  std::vector<Vec> right_vecs(num_cols);
  for (size_t c=0; c < num_cols; ++c) {
    MatCreateVecs(Bases, &right_vecs[c], &left_vecs[c]);
  }

  PetscInt Istart,Iend;
  MatGetOwnershipRange(Bases,&Istart,&Iend);
  std::vector<int> row_ids;
  row_ids.reserve(static_cast<size_t>(local_rows));
  for (int i=Istart; i < Iend; ++i) {
    row_ids.push_back(i);
  }

  for (size_t col=0; col < num_cols; ++col) {
    int col_int = static_cast<int>(col);
    MatSetValues(Bases, local_rows, &row_ids[0], 1, &col_int, &states[col][0], INSERT_VALUES);
  }

  MatAssemblyBegin(Bases, MAT_FINAL_ASSEMBLY);
  MatAssemblyEnd(Bases, MAT_FINAL_ASSEMBLY);

  SVDSetOperators(svd, Bases, NULL);
  SVDSetDimensions(svd, static_cast<int>(num_cols), PETSC_DETERMINE, PETSC_DETERMINE);

  SVDSetWhichSingularTriplets(svd, SVD_LARGEST);
  SVDSolve(svd);
  int nconv;
  SVDGetConverged(svd, &nconv);
    
  if (static_cast<size_t>(nconv)!=num_cols) {
    SLIC_WARNING("Slepc svd solve was unable to converge");
    // consider trying to return just the converged basis?
  }

  std::vector<double> eigenvals(num_cols);
  for (size_t col=0; col < num_cols; ++col) {
    SVDGetSingularTriplet(svd, static_cast<int>(col), &eigenvals[col], left_vecs[col], right_vecs[col]);
  }

  for (size_t col=0; col < num_cols; ++col) {
    double norm;
    VecNorm( left_vecs[col], NORM_2, &norm);
    VecNorm( right_vecs[col], NORM_2, &norm);
  }

  //std::cout << "nconv = " << nconv1 << std::endl;

  SVDDestroy(&svd);
  MatDestroy(&Bases);
  for (auto& v : left_vecs) {
    VecDestroy(&v);
  }
  for (auto& v : right_vecs) {
    VecDestroy(&v);
  }
}
*/

Mat dot(const std::vector<mfem::Vector*>& s, const std::vector<mfem::Vector*>& As)
{
  SLIC_ERROR_IF(s.size() != As.size(), "Search directions and their linear operator result must have same number of columns");
  size_t num_cols = s.size();
  int num_cols_int = static_cast<int>(num_cols);
  Mat sAs;
  MatCreateSeqDense(PETSC_COMM_SELF, num_cols_int, num_cols_int, NULL, &sAs);
  for (size_t i=0; i < num_cols; ++i) {
    for (size_t j=0; j < num_cols; ++j) {
      MatSetValue(sAs, static_cast<int>(i), static_cast<int>(j), mfem::InnerProduct(PETSC_COMM_WORLD, *s[i], *As[j]), INSERT_VALUES); 
    }
  }
  MatAssemblyBegin(sAs, MAT_FINAL_ASSEMBLY);
  MatAssemblyEnd(sAs, MAT_FINAL_ASSEMBLY);
  return sAs;
}

Vec dot(const std::vector<mfem::Vector*>& s, const mfem::Vector& b)
{
  size_t num_cols = s.size();
  Vec sb;
  VecCreateSeq(PETSC_COMM_SELF, static_cast<int>(num_cols), &sb);
  for (size_t i=0; i < num_cols; ++i) {
    VecSetValue(sb, static_cast<int>(i), mfem::InnerProduct(PETSC_COMM_WORLD, *s[i], b), INSERT_VALUES); 
  }
  return sb;
}

auto qr(const std::vector<mfem::Vector*>& states)
{
  BasisVectors bvs(*states[0]);
  BV Q = bvs.constructBases(states);

  Mat R;
  int num_cols = static_cast<int>(states.size());
  MatCreateSeqDense(PETSC_COMM_SELF, num_cols, num_cols, NULL, &R);
  BVOrthogonalize(Q, R);

  return std::make_pair(Q, DenseMat(R));
}

double quadraticEnergy(const DenseMat& A, const DenseVec& b, const DenseVec& x)
{
  DenseVec Ax = A*x;
  double xAx = dot(x, Ax);
  double xb = dot(x, b);
  return 0.5 * xAx - xb;
}

double pnorm_squared(const DenseVec& bvv, const DenseVec& sig) {
  auto bvv_div_sig_squared = bvv / (sig * sig);
  return sum(bvv_div_sig_squared);
  //return bvv.dot((1.0 / (sig * sig)).matrix());
}

double qnorm_squared(const DenseVec& bvv, const DenseVec& sig) {
  auto bvv_div_sig_cubed = bvv / (sig * sig * sig);
  return sum(bvv_div_sig_cubed);
  //return bvv.dot((1.0 / (sig * sig * sig)).matrix());
}

//  returns:
//    minimum energy solution within delta
//    N leftmost eigenvectors
//    N smallest eigenvalue
//    success status
auto exactTrustRegionSolve(const DenseMat& A, const DenseVec& b, double delta, int num_leftmost)
{
  // minimize 1/2 x^T A x - b^T x, s.t. norm(x) <= delta

  auto [isize,jsize] = A.size();
  auto isize2 = b.size();
  SLIC_ERROR_IF(isize!=jsize, "Exact trust region solver requires square matrices");
  SLIC_ERROR_IF(isize!=isize2, "The right hand size for exact trust region solve must be consistent with the input matrix size");

  auto [sigs, V] = eigh(A);

  std::vector<DenseVec> leftmosts;
  std::vector<double> minsigs;
  for (int i=0; i < num_leftmost; ++i) {
    leftmosts.emplace_back(V[i]);
    minsigs.emplace_back(sigs[i]);
  }

  const auto& leftMost = V[0]; 
  double minSig = sigs[0];

  // bv = V.T b, V has columns which are eigenvectors
  DenseVec bv(isize);
  for (size_t i=0; i < static_cast<size_t>(isize); ++i) {
    bv.setValue(i, dot(V[i], b));
  }

  DenseVec bvOverSigs = bv / sigs;
  double sigScale = sum(abs(sigs)) / isize;
  double eps = 1e-12 * sigScale;

  // Check if solution is inside the trust region
  if ((minSig >= eps) && (norm(bvOverSigs) <= delta)) {
    return std::make_tuple(A.solve(b), leftmosts, minsigs, true);
  }

  // if we get here, the solution must be on the tr boundary
  // consider bounding the initial guess, see More' Sorenson paper
  double lam = minSig < eps ? -minSig + eps : 0.0;

  // try to solve this for lam:
  // (A + lam I)p = b, such that norm(p) = Delta
  DenseVec sigsPlusLam = sigs + lam;

  bvOverSigs = bv / sigsPlusLam;

  // Check for the hard case
  if ((minSig < eps) && (norm(bvOverSigs) < delta)) {
    DenseVec p(isize);  p = 0.0;
    for (int i=0; i < isize; ++i) {
      p.add(bv[i], V[i]);
    }

    const auto& z = leftMost;
    double pz = dot(p, z);
    double pp = dot(p, p);
    double ddmpp = std::max(delta * delta - pp, 0.0);

    double tau1 = -pz + std::sqrt(pz * pz + ddmpp);
    double tau2 = -pz - std::sqrt(pz * pz + ddmpp);

    DenseVec x1(p);
    DenseVec x2(p);
    x1.add(tau1, z);
    x2.add(tau2, z);

    double e1 = quadraticEnergy(A, b, x1);
    double e2 = quadraticEnergy(A, b, x2);

    DenseVec x = e1 < e2 ? x1 : x2;

    return std::make_tuple(x, leftmosts, minsigs, true);
  }

  DenseVec bvbv = bv * bv;
  sigsPlusLam = sigs + lam;

  double pNormSq = pnorm_squared(bvbv, sigsPlusLam);
  double pNorm = std::sqrt(pNormSq);
  double bError = (pNorm - delta) / delta;

  // consider an out if it doesn't converge, or use a better initial guess, or bound the lam from below and above.
  size_t iters = 0;
  size_t maxIters = 30;
  while ((std::abs(bError) > 1e-9) && (iters++ < maxIters)) {
    double qNormSq = qnorm_squared(bvbv, sigsPlusLam);
    lam += (pNormSq / qNormSq) * bError;
    sigsPlusLam = sigs + lam;
    pNormSq = pnorm_squared(bvbv, sigsPlusLam);
    pNorm = std::sqrt(pNormSq);
    bError = (pNorm - delta) / delta;
  }

  bool success = true;
  if (iters >= maxIters) {
    success = false;
  }

  bvOverSigs = bv / sigsPlusLam;

  DenseVec x(isize);  x = 0.0;
  for (int i=0; i < isize; ++i) {
    x.add(bvOverSigs[i], V[i]);
  }

  double e1 = quadraticEnergy(A, b, x);
  double e2 = quadraticEnergy(A, b, -x);

  if (e2 < e1) {
    x *= -delta / norm(x);
  } else {
    x *= delta / norm(x);
  }

  return std::make_tuple(x, leftmosts, minsigs, success);
}


// returns the solution, as well as a list of the N leftmost eigenvectors
std::tuple<mfem::Vector, std::vector<mfem::Vector>, std::vector<double>> 
solveSubspaceProblem(const std::vector<mfem::Vector*>& states, 
                     const std::vector<mfem::Vector*>& Astates, 
                     const mfem::Vector& b, double delta, int num_leftmost)
{
  DenseMat sAs = dot(states, Astates);

  auto [Q_parallel,R] = qr(states);

  auto Rinv = inverse(R);
  DenseMat pAp = sAs.PtAP(Rinv);

  Vec b_parallel = petscVec(b);
  std::vector<double> pb_vec(states.size());
  BVDotVec(Q_parallel, b_parallel, &pb_vec[0]);
  DenseVec pb(pb_vec);

  auto [reduced_x, leftvecs, leftvals, success] = exactTrustRegionSolve(pAp, pb, delta, num_leftmost);

  Vec x_parallel; VecDuplicate(b_parallel, &x_parallel);

  std::vector<double> reduced_x_vec = reduced_x.getValues();
  BVMultVec(Q_parallel, 1.0, 1.0, x_parallel, &reduced_x_vec[0]);
  mfem::Vector sol(b);
  copy(x_parallel, sol);

  std::vector<mfem::Vector> leftmosts;
  for (int i=0; i < num_leftmost; ++i) {
    auto reduced_leftvec = leftvecs[i].getValues();
    BVMultVec(Q_parallel, 1.0, 1.0, x_parallel, &reduced_leftvec[0]);
    leftmosts.emplace_back(b);
    copy(x_parallel, leftmosts[i]);
  }

  BVDestroy(&Q_parallel);
  VecDestroy(&b_parallel);
  VecDestroy(&x_parallel);

  return std::make_tuple(sol, leftmosts, leftvals);
}

}