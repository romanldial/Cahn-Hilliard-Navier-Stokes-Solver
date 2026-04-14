// Copyright 2025 Esteban Cisneros

#include "CahnHilliard.hpp"

#include "mfem.hpp"

CahnHilliardOperator::CahnHilliardOperator(
    mfem::FiniteElementSpace &fespace,
    const mfem::Array<int> &ess_tdof_list)
    : mfem::Operator(fespace.GetTrueVSize()),
      fespace_(fespace),
      ess_tdof_list_(ess_tdof_list),
      mobility_(nullptr) {}

CahnHilliardOperator::~CahnHilliardOperator() {
  delete mobility_;
}

void CahnHilliardOperator::SetMobility(
    mfem::real_t (*f)(const mfem::Vector &)) {
  delete mobility_;
  mobility_ = new mfem::FunctionCoefficient(f);
}

void CahnHilliardOperator::BuildMatricies() {
  MFEM_VERIFY(mobility_ != nullptr,
              "Mobility must be set before invoking BuildMatricies()");

  mfem::BilinearForm b_mass(&fespace_);
  mfem::BilinearForm b_mob(&fespace_);

  b_mass.AddDomainIntegrator(new mfem::MassIntegrator());
  b_mob.AddDomainIntegrator(new mfem::DiffusionIntegrator(*mobility_));

  b_mass.Assemble();
  b_mob.Assemble();

  mfem::OperatorPtr op_mass, op_mob;

  b_mass.FormSystemMatrix(ess_tdof_list_, op_mass);
  b_mob.FormSystemMatrix(ess_tdof_list_, op_mob);

  M_phi_ = *op_mass.As<mfem::SparseMatrix>();
  K_mob_ = *op_mob.As<mfem::SparseMatrix>();
}

void CahnHilliardOperator::ComputeSource(const mfem::Vector &mu_vec,
                                         mfem::Vector &source) const {
  source.SetSize(mu_vec.Size());
  K_mob_.Mult(mu_vec, source);
}

void CahnHilliardOperator::Mult(const mfem::Vector &x,
                                mfem::Vector &y) const {
  M_phi_.Mult(x, y);
}

const mfem::SparseMatrix &CahnHilliardOperator::GetMass() const {
  return M_phi_;
}

const mfem::SparseMatrix &CahnHilliardOperator::GetKmob() const {
  return K_mob_;
}
