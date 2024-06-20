// Copyright (c) 2019-2024, Lawrence Livermore National Security, LLC and
// other Serac Project Developers. See the top-level LICENSE file for
// details.
//
// SPDX-License-Identifier: (BSD-3-Clause)

#include "serac/infrastructure/petsc_ext.hpp"
#include "serac/infrastructure/logger.hpp"
#include "petsc_ext.hpp"

#if defined(MFEM_USE_PETSC) && defined(SERAC_USE_PETSC)

#include "petsc/private/matimpl.h"
#include "petscmathypre.h"

namespace serac::mfem_ext {

// Static functions needed to create a shell PC
typedef struct {
  mfem::Solver* solver;
  bool          ownsop;
} __mfem_pc_shell_ctx;

static PetscErrorCode __mfem_pc_shell_view(PC pc, PetscViewer viewer)
{
  __mfem_pc_shell_ctx* ctx;

  PetscFunctionBeginUser;
  auto* void_ctx = static_cast<void*>(&ctx);
  PetscCall(PCShellGetContext(pc, &void_ctx));
  ctx = static_cast<__mfem_pc_shell_ctx*>(void_ctx);
  if (ctx->solver) {
    mfem::PetscPreconditioner* ppc = dynamic_cast<mfem::PetscPreconditioner*>(ctx->solver);
    if (ppc) {
      PetscCall(PCView(*ppc, viewer));
    }
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode __mfem_pc_shell_apply(PC pc, Vec x, Vec y)
{
  Mat                  A;
  PetscBool            is_hypre;
  __mfem_pc_shell_ctx* ctx;

  PetscFunctionBeginUser;
  auto* void_ctx = static_cast<void*>(&ctx);
  PetscCall(PCShellGetContext(pc, &void_ctx));
  ctx = static_cast<__mfem_pc_shell_ctx*>(void_ctx);
  mfem::PetscParVector xx(x, true);
  mfem::PetscParVector yy(y, true);
  // Get the operator from the nonlinear solver and wrap as mfem::PetscParMatrix
  PetscCall(PCGetOperators(pc, nullptr, &A));
  PetscCall(PetscObjectTypeCompare(reinterpret_cast<PetscObject>(A), MATHYPRE, &is_hypre));
  std::unique_ptr<mfem::Operator> mat;
  if (is_hypre) {
    hypre_ParCSRMatrix* hypre_mat;
    PetscCall(MatHYPREGetParCSR(A, &hypre_mat));
    mat = std::make_unique<mfem::HypreParMatrix>(hypre_mat, false);
  } else {
    mat = std::make_unique<mfem::PetscParMatrix>(A, true);
  }
  ctx->solver->SetOperator(*mat);
  if (ctx->solver) {
    ctx->solver->Mult(xx, yy);
    yy.UpdateVecFromFlags();
  } else  // operator is not present, copy x
  {
    yy = xx;
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode __mfem_pc_shell_apply_transpose(PC pc, Vec x, Vec y)
{
  Mat                  A;
  PetscBool            is_hypre;
  __mfem_pc_shell_ctx* ctx;

  PetscFunctionBeginUser;
  auto* void_ctx = static_cast<void*>(&ctx);
  PetscCall(PCShellGetContext(pc, &void_ctx));
  ctx = static_cast<__mfem_pc_shell_ctx*>(void_ctx);
  mfem::PetscParVector xx(x, true);
  mfem::PetscParVector yy(y, true);
  // Get the operator from the nonlinear solver and wrap as mfem::PetscParMatrix
  PetscCall(PCGetOperators(pc, nullptr, &A));
  PetscCall(PetscObjectTypeCompare(reinterpret_cast<PetscObject>(A), MATHYPRE, &is_hypre));
  std::unique_ptr<mfem::Operator> mat;
  if (is_hypre) {
    hypre_ParCSRMatrix* hypre_mat;
    PetscCall(MatHYPREGetParCSR(A, &hypre_mat));
    mat = std::make_unique<mfem::HypreParMatrix>(hypre_mat, false);
  } else {
    mat = std::make_unique<mfem::PetscParMatrix>(A, true);
  }
  ctx->solver->SetOperator(*mat);
  if (ctx->solver) {
    ctx->solver->MultTranspose(xx, yy);
    yy.UpdateVecFromFlags();
  } else  // operator is not present, copy x
  {
    yy = xx;
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode __mfem_pc_shell_setup(PC pc)
{
  __mfem_pc_shell_ctx* ctx;

  PetscFunctionBeginUser;
  auto* void_ctx = static_cast<void*>(&ctx);
  PetscCall(PCShellGetContext(pc, &void_ctx));
  ctx = static_cast<__mfem_pc_shell_ctx*>(void_ctx);
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode __mfem_pc_shell_destroy(PC pc)
{
  __mfem_pc_shell_ctx* ctx;

  PetscFunctionBeginUser;
  auto* void_ctx = static_cast<void*>(&ctx);
  PetscCall(PCShellGetContext(pc, &void_ctx));
  ctx = static_cast<__mfem_pc_shell_ctx*>(void_ctx);
  if (ctx->ownsop) {
    delete ctx->solver;
  }
  PetscCall(PetscFree(void_ctx));
  ctx = NULL;
  PetscFunctionReturn(PETSC_SUCCESS);
}

// Sets the type of PC to PCSHELL and wraps the solver action
// if ownsop is true, ownership of precond is transferred to the PETSc object
static PetscErrorCode MakeShellPC(PC pc, mfem::Solver& precond, bool ownsop)
{
  __mfem_pc_shell_ctx* ctx;

  PetscFunctionBeginUser;
  PetscCall(PetscCalloc1(1, &ctx));

  ctx->solver = &precond;
  ctx->ownsop = ownsop;

  // In case the PC was already of type SHELL, this will destroy any
  // previous user-defined data structure
  // We cannot call PCReset as it will wipe out any operator already set
  PetscCall(PCSetType(pc, PCNONE));

  PetscCall(PCSetType(pc, PCSHELL));
  PetscCall(PCShellSetName(pc, "MFEM Solver"));
  auto* void_ctx = static_cast<void*>(ctx);
  PetscCall(PCShellSetContext(pc, void_ctx));
  PetscCall(PCShellSetApply(pc, __mfem_pc_shell_apply));
  PetscCall(PCShellSetApplyTranspose(pc, __mfem_pc_shell_apply_transpose));
  PetscCall(PCShellSetSetUp(pc, __mfem_pc_shell_setup));
  PetscCall(PCShellSetView(pc, __mfem_pc_shell_view));
  PetscCall(PCShellSetDestroy(pc, __mfem_pc_shell_destroy));
  PetscFunctionReturn(PETSC_SUCCESS);
}

// PetscPCSolver methods

PetscErrorCode convertPCPreSolve(PC pc, [[maybe_unused]] KSP ksp)
{
  PetscPCSolver* solver;
  Mat            A;
  void*          void_solver;

  PetscFunctionBeginUser;
  PetscCall(PCGetApplicationContext(pc, &void_solver));
  solver = static_cast<PetscPCSolver*>(void_solver);
  // If this function is called, we have a PETSc preconditioner
  // That means we have to ensure the matrix is MATAIJ
  if (!solver->checked_for_convert_ || solver->converted_matrix_) {
    PetscCall(PCGetOperators(pc, NULL, &A));
    PetscBool is_aij;
    PetscCall(PetscObjectTypeCompare(reinterpret_cast<PetscObject>(A), MATAIJ, &is_aij));
    if (is_aij) PetscFunctionReturn(PETSC_SUCCESS);
    SLIC_INFO("convertPCPreSolve(...) - Converting operators to MATAIJ format.");
    mfem::PetscParMatrix temp_mat(A, true);
    solver->converted_matrix_ = std::make_unique<mfem::PetscParMatrix>(temp_mat, mfem::Operator::PETSC_MATAIJ);
    PetscCall(PCSetOperators(pc, *solver->converted_matrix_, *solver->converted_matrix_));
  }
  solver->checked_for_convert_ = true;
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscPCSolver::PetscPCSolver(MPI_Comm comm, PCType pc_type, const std::string& prefix)
    : PetscPreconditioner(comm, prefix)
{
  PetscCallAbort(GetComm(), PCSetType(*this, pc_type));
  PetscCallAbort(GetComm(), PCSetApplicationContext(*this, this));
  PetscCallAbort(GetComm(), PCSetPreSolve(*this, convertPCPreSolve));
}

PetscPCSolver::PetscPCSolver(mfem::PetscParMatrix& A, PCType pc_type, const std::string& prefix)
    : PetscPreconditioner(A, prefix)
{
  PetscCallAbort(GetComm(), PCSetType(*this, pc_type));
  PetscCallAbort(GetComm(), PCSetApplicationContext(*this, this));
  PetscCallAbort(GetComm(), PCSetPreSolve(*this, convertPCPreSolve));
}

PetscPCSolver::PetscPCSolver(MPI_Comm comm, Operator& op, PCType pc_type, const std::string& prefix)
    : PetscPreconditioner(comm, op, prefix)
{
  PetscCallAbort(GetComm(), PCSetType(*this, pc_type));
  PetscCallAbort(GetComm(), PCSetApplicationContext(*this, this));
  PetscCallAbort(GetComm(), PCSetPreSolve(*this, convertPCPreSolve));
}

// PetscPreconditionerSpaceDependent methods

void PetscPreconditionerSpaceDependent::SetOperator(const Operator& op)
{
  // Update parent class
  PetscPreconditioner::SetOperator(op);
  SLIC_WARNING_ROOT_IF(
      !fespace_,
      "Finite element space not set with SetFESpace() method, expect performance and/or convergence issues.");
  if (fespace_) {
    Mat pA, ppA;
    PetscCallAbort(GetComm(), PCGetOperators(*this, NULL, &ppA));
    int vdim = fespace_->GetVDim();

    // Ideally, the block size should be set at matrix creation
    // but the MFEM assembly does not allow us to do so
    PetscCallAbort(GetComm(), MatSetBlockSize(ppA, vdim));
    PetscCallAbort(GetComm(), PCGetOperators(*this, &pA, NULL));
    if (ppA != pA) {
      PetscCallAbort(GetComm(), MatSetBlockSize(pA, vdim));
    }
  }
}

// PetscGAMGSolver methods

static PetscErrorCode gamg_pre_solve(PC pc, KSP ksp)
{
  PetscGAMGSolver* solver;
  void*            void_solver;

  PetscFunctionBeginUser;
  PetscCall(convertPCPreSolve(pc, ksp));
  PetscCall(PCGetApplicationContext(pc, &void_solver));
  solver = static_cast<PetscGAMGSolver*>(void_solver);
  solver->SetupNearNullSpace();
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscGAMGSolver::PetscGAMGSolver(MPI_Comm& comm, const std::string& prefix)
    : PetscPreconditionerSpaceDependent(comm, PCGAMG, prefix)
{
  PetscCallAbort(GetComm(), PCSetApplicationContext(*this, this));
  PetscCallAbort(GetComm(), PCSetPreSolve(*this, gamg_pre_solve));
  Customize();
}

PetscGAMGSolver::PetscGAMGSolver(mfem::PetscParMatrix& A, const std::string& prefix)
    : PetscPreconditionerSpaceDependent(A, PCGAMG, prefix)
{
  PetscCallAbort(GetComm(), PCSetApplicationContext(*this, this));
  PetscCallAbort(GetComm(), PCSetPreSolve(*this, gamg_pre_solve));
  Customize();
}

PetscGAMGSolver::PetscGAMGSolver(MPI_Comm comm, Operator& op, const std::string& prefix)
    : PetscPreconditionerSpaceDependent(comm, op, PCGAMG, prefix)
{
  PetscCallAbort(GetComm(), PCSetApplicationContext(*this, this));
  PetscCallAbort(GetComm(), PCSetPreSolve(*this, gamg_pre_solve));
  Customize();
}

static void func_coords(const mfem::Vector& x, mfem::Vector& y) { y = x; }

void PetscGAMGSolver::SetupNearNullSpace()
{
  Mat pA;
  PetscCallAbort(GetComm(), PCGetOperators(*this, NULL, &pA));
  MatNullSpace nnsp;
  PetscCallAbort(GetComm(), MatGetNearNullSpace(pA, &nnsp));
  if (!fespace_ || nnsp) return;

  // get PETSc object
  PC pc = *this;

  PetscBool is_op_set;
  PetscCallAbort(GetComm(), PCGetOperatorsSet(pc, nullptr, &is_op_set));
  if (!is_op_set) return;

  PetscBool ismatis, ismataij;
  bool      has_local_mat;
  PetscCallAbort(GetComm(), PetscObjectTypeCompare(reinterpret_cast<PetscObject>(pA), MATIS, &ismatis));
  PetscCallAbort(GetComm(), PetscObjectTypeCompare(reinterpret_cast<PetscObject>(pA), MATAIJ, &ismataij));
  has_local_mat = ismatis || ismataij;

  PetscInt sdim = fespace_->GetParMesh()->SpaceDimension();
  int      vdim = fespace_->GetVDim();

  // Ideally, the block size should be set at matrix creation
  // but the MFEM assembly does not allow us to do so
  PetscCallAbort(GetComm(), MatSetBlockSize(pA, vdim));

  // coordinates
  const mfem::FiniteElementCollection* fec     = fespace_->FEColl();
  bool                                 h1space = dynamic_cast<const mfem::H1_FECollection*>(fec);
  if (h1space) {
    SLIC_INFO("PetscGAMGSolver::SetupNearNullSpace(...) - Setting up near null space");
    mfem::ParFiniteElementSpace* fespace_coords = fespace_;

    sdim = fespace_->GetParMesh()->SpaceDimension();
    if (vdim != sdim || fespace_->GetOrdering() != mfem::Ordering::byVDIM) {
      fespace_coords = new mfem::ParFiniteElementSpace(fespace_->GetParMesh(), fec, sdim, mfem::Ordering::byVDIM);
    }
    mfem::VectorFunctionCoefficient coeff_coords(sdim, func_coords);
    mfem::ParGridFunction           gf_coords(fespace_coords);
    gf_coords.ProjectCoefficient(coeff_coords);
    int                   num_nodes   = fespace_->GetNDofs();
    mfem::HypreParVector* hvec_coords = gf_coords.ParallelProject();
    auto data_coords = const_cast<PetscScalar*>(mfem::Read(hvec_coords->GetMemory(), hvec_coords->Size(), false));
    PetscCallAbort(GetComm(), PCSetCoordinates(*this, sdim, num_nodes, data_coords));

    Vec pvec_coords;
    PetscCallAbort(GetComm(), VecCreateMPIWithArray(GetComm(), sdim, hvec_coords->Size(), hvec_coords->GlobalSize(),
                                                    data_coords, &pvec_coords));
    PetscCallAbort(GetComm(), MatNullSpaceCreateRigidBody(pvec_coords, &nnsp));
    PetscCallAbort(GetComm(), MatSetNearNullSpace(pA, nnsp));
    PetscCallAbort(GetComm(), MatNullSpaceDestroy(&nnsp));

    // likely elasticity -> we attach rigid-body modes as near-null space information to the local matrices
    if (vdim == sdim) {
      if (has_local_mat) {
        Mat                    lA = nullptr;
        Vec                    lvec_coords;
        ISLocalToGlobalMapping l2g;
        PetscSF                sf;
        PetscLayout            rmap;
        const PetscInt*        gidxs;
        PetscInt               nleaves;

        if (ismatis) {
          PetscCallAbort(GetComm(), MatISGetLocalMat(pA, &lA));
        } else if (ismataij) {
          PetscCallAbort(GetComm(), MatAIJGetLocalMat(pA, &lA));
        } else {
          SLIC_ERROR_ROOT("Unsupported mat type.");
        }
        PetscCallAbort(GetComm(), MatCreateVecs(lA, &lvec_coords, NULL));
        PetscCallAbort(GetComm(), VecSetBlockSize(lvec_coords, sdim));
        PetscCallAbort(GetComm(), MatGetLocalToGlobalMapping(pA, &l2g, NULL));
        PetscCallAbort(GetComm(), MatGetLayouts(pA, &rmap, NULL));
        PetscCallAbort(GetComm(), PetscSFCreate(GetComm(), &sf));
        PetscCallAbort(GetComm(), ISLocalToGlobalMappingGetIndices(l2g, &gidxs));
        PetscCallAbort(GetComm(), ISLocalToGlobalMappingGetSize(l2g, &nleaves));
        PetscCallAbort(GetComm(), PetscSFSetGraphLayout(sf, rmap, nleaves, NULL, PETSC_OWN_POINTER, gidxs));
        PetscCallAbort(GetComm(), ISLocalToGlobalMappingRestoreIndices(l2g, &gidxs));
        {
          PetscReal* garray;
          PetscReal* larray;

          PetscCallAbort(GetComm(), VecGetArray(pvec_coords, &garray));
          PetscCallAbort(GetComm(), VecGetArray(lvec_coords, &larray));
#if PETSC_VERSION_LT(3, 15, 0)
          PetscCallAbort(GetComm(), PetscSFBcastBegin(sf, MPIU_SCALAR, garray, larray));
          PetscCallAbort(GetComm(), PetscSFBcastEnd(sf, MPIU_SCALAR, garray, larray));
#else
          PetscCallAbort(GetComm(), PetscSFBcastBegin(sf, MPIU_SCALAR, garray, larray, MPI_REPLACE));
          PetscCallAbort(GetComm(), PetscSFBcastEnd(sf, MPIU_SCALAR, garray, larray, MPI_REPLACE));
#endif
          PetscCallAbort(GetComm(), VecRestoreArray(pvec_coords, &garray));
          PetscCallAbort(GetComm(), VecRestoreArray(lvec_coords, &larray));
        }
        PetscCallAbort(GetComm(), MatNullSpaceCreateRigidBody(lvec_coords, &nnsp));
        PetscCallAbort(GetComm(), VecDestroy(&lvec_coords));
        PetscCallAbort(GetComm(), MatSetNearNullSpace(lA, nnsp));
        PetscCallAbort(GetComm(), MatNullSpaceDestroy(&nnsp));
        PetscCallAbort(GetComm(), PetscSFDestroy(&sf));
      }
      PetscCallAbort(GetComm(), VecDestroy(&pvec_coords));
    }
    if (fespace_coords != fespace_) {
      delete fespace_coords;
    }
    delete hvec_coords;
  }
  PetscCallAbort(GetComm(), MatGetNearNullSpace(pA, &nnsp));
  SLIC_WARNING_ROOT_IF(!nnsp, "Global near null space was not set successfully, expect slow (or no) convergence.");
}

void PetscGAMGSolver::SetOperator(const Operator& op)
{
  // Update parent class
  PetscPreconditionerSpaceDependent::SetOperator(op);
  // Set rigid body near null space
  SLIC_WARNING_ROOT_IF(
      fespace_ == nullptr,
      "Displacement FE space not set with PetscGAMGSolver::SetFESpace, expect slow (or no) convergence.");
  if (fespace_) {
    SetupNearNullSpace();
  }
}

// PetscKSPSolver methods

PetscErrorCode convertKSPPreSolve(KSP ksp, [[maybe_unused]] Vec rhs, [[maybe_unused]] Vec x, void* ctx)
{
  PetscKSPSolver* solver;
  Mat             A;

  PetscFunctionBeginUser;
  solver                              = static_cast<PetscKSPSolver*>(ctx);
  auto*                      prec     = solver->prec;
  mfem::PetscPreconditioner* petsc_pc = dynamic_cast<mfem::PetscPreconditioner*>(prec);
  if (!solver->operatorset || solver->needs_hypre_wrapping_) {
    PetscCall(KSPGetOperators(ksp, &A, NULL));
    PetscBool is_hypre;
    PetscCall(PetscObjectTypeCompare(reinterpret_cast<PetscObject>(A), MATHYPRE, &is_hypre));
    SLIC_WARNING_IF(
        is_hypre && petsc_pc,
        "convertKSPPreSolve(...) - MATHYPRE is not supported for most PETSc preconditioners, converting to MATAIJ.");
    if (!is_hypre || petsc_pc) PetscFunctionReturn(PETSC_SUCCESS);
    hypre_ParCSRMatrix *hypre_csr = nullptr, *old_hypre_csr = nullptr;
    PetscCall(MatHYPREGetParCSR(A, &hypre_csr));
    if (solver->wrapped_matrix_) {
      old_hypre_csr = *solver->wrapped_matrix_;
    }
    if (old_hypre_csr != hypre_csr || !solver->wrapped_matrix_) {
      SLIC_INFO("convertKSPPreSolve(...) - Rebuilding HypreParMatrix wrapper");
      solver->wrapped_matrix_ = std::make_unique<mfem::HypreParMatrix>(hypre_csr, false);
    }
    if (solver->prec) solver->prec->SetOperator(*solver->wrapped_matrix_);
    solver->needs_hypre_wrapping_ = true;
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscKSPSolver::PetscKSPSolver(MPI_Comm comm, KSPType ksp_type, const std::string& prefix, bool wrap, bool iter_mode)
    : mfem::IterativeSolver(comm), mfem::PetscLinearSolver(comm, prefix, wrap, iter_mode)
{
  abs_tol  = PETSC_DEFAULT;
  rel_tol  = PETSC_DEFAULT;
  max_iter = PETSC_DEFAULT;
  PetscCallAbort(comm, KSPConvergedDefaultSetConvergedMaxits(*this, PETSC_TRUE));
  PetscCallAbort(comm, KSPSetType(*this, ksp_type));
  PetscCallAbort(comm, KSPSetPreSolve(*this, convertKSPPreSolve, this));
  Customize();
}

PetscKSPSolver::PetscKSPSolver(const mfem::PetscParMatrix& A, KSPType ksp_type, const std::string& prefix,
                               bool iter_mode)
    : mfem::IterativeSolver(A.GetComm()), mfem::PetscLinearSolver(A, prefix, iter_mode), wrap_(false)
{
  abs_tol  = PETSC_DEFAULT;
  rel_tol  = PETSC_DEFAULT;
  max_iter = PETSC_DEFAULT;
  PetscCallAbort(GetComm(), KSPConvergedDefaultSetConvergedMaxits(*this, PETSC_TRUE));
  PetscCallAbort(GetComm(), KSPSetType(*this, ksp_type));
  PetscCallAbort(GetComm(), KSPSetPreSolve(*this, convertKSPPreSolve, this));
  Customize();
}

PetscKSPSolver::PetscKSPSolver(const mfem::HypreParMatrix& A, KSPType ksp_type, const std::string& prefix, bool wrap,
                               bool iter_mode)
    : mfem::IterativeSolver(A.GetComm()), mfem::PetscLinearSolver(A, wrap, prefix, iter_mode), wrap_(wrap)
{
  abs_tol  = PETSC_DEFAULT;
  rel_tol  = PETSC_DEFAULT;
  max_iter = PETSC_DEFAULT;
  PetscCallAbort(GetComm(), KSPConvergedDefaultSetConvergedMaxits(*this, PETSC_TRUE));
  PetscCallAbort(GetComm(), KSPSetType(*this, ksp_type));
  PetscCallAbort(GetComm(), KSPSetPreSolve(*this, convertKSPPreSolve, this));
  Customize();
}

void PetscKSPSolver::SetTolerances()
{
  PetscCallAbort(GetComm(), KSPSetTolerances(*this, rel_tol, abs_tol, PETSC_DEFAULT, max_iter));
}

void PetscKSPSolver::Mult(const mfem::Vector& b, mfem::Vector& x) const { mfem::PetscLinearSolver::Mult(b, x); }

void PetscKSPSolver::MultTranspose(const mfem::Vector& b, mfem::Vector& x) const
{
  mfem::PetscLinearSolver::MultTranspose(b, x);
}

void PetscKSPSolver::SetOperator(const mfem::Operator& op)
{
  const mfem::HypreParMatrix* hA = dynamic_cast<const mfem::HypreParMatrix*>(&op);
  mfem::PetscParMatrix*       pA = const_cast<mfem::PetscParMatrix*>(dynamic_cast<const mfem::PetscParMatrix*>(&op));
  const mfem::Operator*       oA = dynamic_cast<const mfem::Operator*>(&op);

  // set tolerances from user
  SetTolerances();

  // Check if preconditioner can use HYPRE matrices
  mfem::PetscPreconditioner* petsc_pc = dynamic_cast<mfem::PetscPreconditioner*>(prec);

  // delete existing matrix, if created
  if (pA_) delete pA_;
  pA_ = nullptr;
  // update base classes: Operator, Solver, PetscLinearSolver
  if (!pA) {
    if (hA) {
      // Create MATSHELL object or convert into a format suitable to construct preconditioners
      if (PETSC_HAVE_HYPRE && !petsc_pc) {
        SLIC_INFO("PetscKSPSolver::SetOperator(...) - Wrapping existing HYPRE matrix");
        pA = new mfem::PetscParMatrix(hA, wrap_ ? PETSC_MATSHELL : PETSC_MATAIJ);
      } else {
        SLIC_WARNING(
            "PetscKSPSolver::SetOperator(...) - Converting operator, consider using PetscParMatrix to avoid conversion "
            "costs");
        pA = new mfem::PetscParMatrix(hA, wrap_ ? PETSC_MATSHELL : PETSC_MATAIJ);
      }
    } else if (oA)  // fallback to general operator
    {
      // Create MATSHELL or MATNEST (if oA is a BlockOperator) object
      // If oA is a BlockOperator, Operator::Type is relevant to the subblocks
      SLIC_WARNING(
          "PetscKSPSolver::SetOperator(...) - Converting operator, consider using PetscParMatrix to avoid conversion "
          "costs");
      pA = new mfem::PetscParMatrix(GetComm(), oA, wrap_ ? PETSC_MATSHELL : PETSC_MATAIJ);
    }
    pA_ = pA;
  }
  MFEM_VERIFY(pA, "PetscKSPSolver::SetOperator(...) - Unsupported operation!");

  // Set operators into PETSc KSP
  KSP ksp = *this;
  Mat A   = *pA;
  if (operatorset) {
    Mat      C;
    PetscInt nheight, nwidth, oheight, owidth;

    PetscCallAbort(GetComm(), KSPGetOperators(ksp, &C, NULL));
    PetscCallAbort(GetComm(), MatGetSize(A, &nheight, &nwidth));
    PetscCallAbort(GetComm(), MatGetSize(C, &oheight, &owidth));
    if (nheight != oheight || nwidth != owidth) {
      // reinit without destroying the KSP
      // communicator remains the same
      SLIC_WARNING("PetscKSPSolver::SetOperator(...) - Rebuilding KSP");
      PetscCallAbort(GetComm(), KSPReset(ksp));
      delete X;
      delete B;
      X = B = NULL;
    }
  }

  mfem::PetscParMatrix op_wrapped(GetComm(), &op);
  PetscCallAbort(GetComm(), KSPSetOperators(ksp, op_wrapped, A));

  // Update PetscSolver
  operatorset = true;

  // Update the Operator fields.
  IterativeSolver::height   = pA->Height();
  PetscLinearSolver::height = pA->Height();
  IterativeSolver::width    = pA->Width();
  PetscLinearSolver::width  = pA->Width();

  if (prec && petsc_pc) {
    prec->SetOperator(*pA);
  } else if (prec) {
    prec->SetOperator(op);
  }
}

void PetscKSPSolver::SetPreconditioner(mfem::Solver& pc)
{
  mfem::PetscLinearSolver::SetPreconditioner(pc);
  prec = &pc;
}

// PetscNewtonSolver methods

PetscNewtonSolver::PetscNewtonSolver(MPI_Comm comm, SNESType snes_type, SNESLineSearchType linesearch_type,
                                     const std::string& prefix)
    : mfem::NewtonSolver(comm),
      mfem::PetscNonlinearSolver(comm, prefix),
      snes_type_(snes_type),
      linesearch_type_(linesearch_type)
{
  rel_tol  = PETSC_DEFAULT;
  abs_tol  = PETSC_DEFAULT;
  max_iter = PETSC_DEFAULT;
  SetJacobianType(ANY_TYPE);
  PetscCallVoid(SNESSetType(*this, snes_type_));
  Customize();
  NewtonSolver::iterative_mode = PetscNonlinearSolver::iterative_mode = true;
}

PetscNewtonSolver::PetscNewtonSolver(MPI_Comm comm, Operator& op, SNESType snes_type,
                                     SNESLineSearchType linesearch_type, const std::string& prefix)
    : mfem::NewtonSolver(comm),
      mfem::PetscNonlinearSolver(comm, op, prefix),
      snes_type_(snes_type),
      linesearch_type_(linesearch_type)
{
  rel_tol  = PETSC_DEFAULT;
  abs_tol  = PETSC_DEFAULT;
  max_iter = PETSC_DEFAULT;
  PetscCallVoid(SNESSetType(*this, snes_type_));
  SetJacobianType(ANY_TYPE);
  Customize();
  NewtonSolver::iterative_mode = PetscNonlinearSolver::iterative_mode = true;
}

void PetscNewtonSolver::SetTolerances()
{
  PetscCallAbort(GetComm(), SNESSetTolerances(*this, abs_tol, rel_tol, step_tol_, max_iter, PETSC_DEFAULT));
  // Fix specifically the absolute tolerance for CP linesearch, since a PETSc bug will erroneously lead to early
  // "convergence". See: https://gitlab.com/petsc/petsc/-/issues/1583
  if (operatorset) {
    PetscBool is_newtonls, is_cp;
    PetscCallAbort(GetComm(), PetscStrcmp(SNESNEWTONLS, snes_type_, &is_newtonls));
    PetscCallAbort(GetComm(), PetscStrcmp(SNESLINESEARCHCP, linesearch_type_, &is_cp));
    if (is_newtonls && is_cp) {
      SNESLineSearch linesearch;
      PetscCallAbort(GetComm(), SNESGetLineSearch(*this, &linesearch));
      PetscCallAbort(GetComm(), SNESLineSearchSetTolerances(linesearch, PETSC_DEFAULT, PETSC_DEFAULT, PETSC_DEFAULT,
                                                            1e-30, PETSC_DEFAULT, PETSC_DEFAULT));
    }
  }
}

void PetscNewtonSolver::SetNonPetscSolver(mfem::Solver& solver)
{
  // Set the KSP object associated with the SNES to be PREONLY so no linear solver is used
  KSP ksp;
  PetscCallAbort(GetComm(), SNESGetKSP(*this, &ksp));
  PetscCallAbort(GetComm(), KSPSetType(ksp, KSPPREONLY));
  // Place the non-PETSc solver into a shell PC
  PC pc_shell;
  PetscCallAbort(GetComm(), KSPGetPC(ksp, &pc_shell));
  PetscCallAbort(GetComm(), MakeShellPC(pc_shell, solver, false));
}

void PetscNewtonSolver::SetLineSearchType(SNESLineSearchType linesearch_type)
{
  linesearch_type_ = linesearch_type;
  if (operatorset) {
    SNESLineSearch linesearch;
    PetscCallAbort(GetComm(), SNESGetLineSearch(*this, &linesearch));
    PetscCallAbort(GetComm(), SNESLineSearchSetType(linesearch, linesearch_type_));
  }
  SetTolerances();
  Customize();
}

void PetscNewtonSolver::SetSNESType(SNESType snes_type)
{
  snes_type_ = snes_type;
  PetscCallAbort(GetComm(), SNESSetType(*this, snes_type_));
  Customize();
}

void PetscNewtonSolver::SetSolver(mfem::Solver& solver)
{
  auto petsc_solver = dynamic_cast<mfem::PetscLinearSolver*>(&solver);
  if (petsc_solver) {
    PetscCallAbort(GetComm(), SNESSetKSP(*this, *petsc_solver));
    prec             = &solver;
    auto* ksp_solver = dynamic_cast<PetscKSPSolver*>(&solver);
    if (ksp_solver) {
      auto* inner_prec       = ksp_solver->GetPreconditioner();
      auto* petsc_inner_prec = dynamic_cast<mfem::PetscPreconditioner*>(inner_prec);
      if (petsc_inner_prec) {
        SLIC_INFO("PetscNewtonSolver::SetSolver(...) - Set Jacobian type to PETSC_MATAIJ");
        SetJacobianType(PETSC_MATAIJ);
      } else {
        SLIC_INFO("PetscNewtonSolver::SetSolver(...) - Set Jacobian type to PETSC_MATHYPRE");
        SetJacobianType(ANY_TYPE);
      }
    }
  } else {
    SetNonPetscSolver(solver);
  }
}

void PetscNewtonSolver::SetOperator(const mfem::Operator& op)
{
  bool first_set = !operatorset;
  mfem::PetscNonlinearSolver::SetOperator(op);
  // mfem::NewtonSolver::SetOperator sets defaults, we need to override them
  if (first_set) {
    SetSNESType(snes_type_);
    SetLineSearchType(linesearch_type_);
  }
  SetTolerances();
  Customize();
}

void PetscNewtonSolver::Mult(const mfem::Vector& b, mfem::Vector& x) const
{
  bool b_nonempty = b.Size();
  auto solver     = dynamic_cast<const PetscNonlinearSolver*>(this);
  if (!B) {
    B = new mfem::PetscParVector(PetscObjectComm(obj), *solver, true, !b_nonempty);
  }
  if (!X) {
    X = new mfem::PetscParVector(PetscObjectComm(obj), *solver, false, false);
  }
  X->PlaceMemory(x.GetMemory(), PetscNonlinearSolver::iterative_mode);
  if (b_nonempty) {
    B->PlaceMemory(b.GetMemory());
  } else {
    *B = 0.0;
  }

  Customize();

  if (!PetscNonlinearSolver::iterative_mode) {
    *X = 0.;
  }

  // Solve the system.
  PetscCallAbort(GetComm(), SNESSolve(*this, *B, *X));
  X->ResetMemory();
  if (b_nonempty) {
    B->ResetMemory();
  }
}

}  // namespace serac::mfem_ext
#endif
