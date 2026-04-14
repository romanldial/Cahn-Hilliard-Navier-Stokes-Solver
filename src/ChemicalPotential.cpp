// Copyright 2025 Esteban Cisneros
//
// Chemical potential for the Cahn-Hilliard equation using the
// Landau-Ginzburg type potential:
//
//   mu = (3sigma/4epsilon) * phi * (phi^2 - 1)
//      - (3/2) * sigma * epsilon * Laplacian(phi)
//
// Weak formulation (surface integral vanishes by conservation of mass):
//
//   int{mu*v} dOmega
//     =   int{(3s/4e) * phi*(phi^2-1) * v} dOmega
//       + int{(3/2)*s*e * Nabla(phi) . Nabla(v)} dOmega

#include "ChemicalPotential.hpp"

#include <iostream>

#include "mfem.hpp"

ChemicalPotentialOperator::ChemicalPotentialOperator(
    mfem::FiniteElementSpace &fespace,
    mfem::Vector &X,
    const mfem::Array<int> &ess_tdof_list,
    const Params &params)
    : mfem::Operator(fespace.GetTrueVSize()),
      fespace_(fespace),
      ess_tdof_list_(ess_tdof_list),
      params_(params) {
  std::cout << "  Setting space..." << std::endl;
  phi_lagged_gf_.SetSpace(&fespace_);
  std::cout << "  Setting From True Dofs..." << std::endl;
  phi_lagged_gf_.SetFromTrueDofs(X);
  std::cout << "  Building Matricies..." << std::endl;
  BuildMatricies();
  std::cout << "  LHS_M size: "
            << LHS_M_.Height() << "x" << LHS_M_.Width() << std::endl;
  std::cout << "  RHS_K size: "
            << RHS_K_.Height() << "x" << RHS_K_.Width() << std::endl;
  std::cout << "  Setting mu FESpace..." << std::endl;
  mu_.SetSpace(&fespace_);
  std::cout << "  Constructor done." << std::endl;
}

ChemicalPotentialOperator::~ChemicalPotentialOperator() {}

void ChemicalPotentialOperator::BuildMatricies() {
  mfem::real_t first_constant =
      (3.0 * params_.sigma) / (4.0 * params_.epsilon);
  mfem::real_t second_constant =
      (3.0 * params_.sigma * params_.epsilon) / 2.0;

  mfem::ConstantCoefficient neg_one_coef(-1.0);
  mfem::ConstantCoefficient first_coef(first_constant);
  mfem::ConstantCoefficient second_coef(second_constant);

  mfem::GridFunctionCoefficient phi_lagged_coef(&phi_lagged_gf_);
  mfem::ProductCoefficient      a_term(first_coef, phi_lagged_coef);
  mfem::ProductCoefficient      phi_squared(phi_lagged_coef, phi_lagged_coef);
  mfem::SumCoefficient          phi_squared_one(phi_squared, neg_one_coef);
  mfem::ProductCoefficient      nonlinear_mass_term(a_term, phi_squared_one);

  mfem::BilinearForm rhs_mass(&fespace_);
  mfem::BilinearForm rhs_stiffness(&fespace_);
  mfem::BilinearForm lhs_mass(&fespace_);

  rhs_mass.AddDomainIntegrator(
      new mfem::MassIntegrator(nonlinear_mass_term));
  rhs_stiffness.AddDomainIntegrator(
      new mfem::DiffusionIntegrator(second_coef));
  lhs_mass.AddDomainIntegrator(new mfem::MassIntegrator());

  rhs_mass.Assemble();
  rhs_stiffness.Assemble();
  lhs_mass.Assemble();

  mfem::OperatorPtr op_rhs_m, op_rhs_k, op_lhs_m;

  rhs_mass.FormSystemMatrix(ess_tdof_list_, op_rhs_m);
  rhs_stiffness.FormSystemMatrix(ess_tdof_list_, op_rhs_k);
  lhs_mass.FormSystemMatrix(ess_tdof_list_, op_lhs_m);

  RHS_M_ = *op_rhs_m.As<mfem::SparseMatrix>();
  RHS_K_ = *op_rhs_k.As<mfem::SparseMatrix>();
  LHS_M_ = *op_lhs_m.As<mfem::SparseMatrix>();
}

void ChemicalPotentialOperator::SolveSystem(mfem::Vector &phi_current) {
  mfem::Vector rhs_nonlinear(phi_current.Size());
  mfem::Vector rhs_stiffness(phi_current.Size());
  mfem::Vector rhs_mu_complete(phi_current.Size());
  mu_.SetSize(phi_current.Size());

  RHS_M_.Mult(phi_current, rhs_nonlinear);
  RHS_K_.Mult(phi_current, rhs_stiffness);
  rhs_mu_complete  = rhs_nonlinear;
  rhs_mu_complete += rhs_stiffness;

  mfem::GSSmoother prec_mu(LHS_M_);
  mfem::CGSolver cg_mu;
  cg_mu.SetOperator(LHS_M_);
  cg_mu.SetPreconditioner(prec_mu);
  cg_mu.SetRelTol(1e-8);
  cg_mu.SetMaxIter(1000);
  cg_mu.SetPrintLevel(0);
  cg_mu.Mult(rhs_mu_complete, mu_);
}

void ChemicalPotentialOperator::UpdatePhi(const mfem::GridFunction &phi) {
  phi_lagged_gf_ = phi;
  BuildMatricies();
}

void ChemicalPotentialOperator::SetEpsilon(mfem::real_t epsilon) {
  std::cout << "  SetEpsilon called..." << std::endl;
  params_.epsilon = epsilon;
  BuildMatricies();
}

void ChemicalPotentialOperator::SetSigma(mfem::real_t sigma) {
  std::cout << "  SetSigma called..." << std::endl;
  params_.sigma = sigma;
  BuildMatricies();
}

const mfem::SparseMatrix &ChemicalPotentialOperator::GetLHS_M() const {
  return LHS_M_;
}

const mfem::SparseMatrix &ChemicalPotentialOperator::GetRHS_M() const {
  return RHS_M_;
}

const mfem::SparseMatrix &ChemicalPotentialOperator::GetRHS_K() const {
  return RHS_K_;
}

mfem::GridFunction &ChemicalPotentialOperator::GetMu() {
  return mu_;
}
