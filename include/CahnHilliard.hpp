// Copyright 2025 Esteban Cisneros
// Header file for CahnHilliardOperator.

#pragma once

#include "mfem.hpp"

class CahnHilliardOperator : public mfem::Operator {
 public:
  CahnHilliardOperator(mfem::FiniteElementSpace &fespace,
                       const mfem::Array<int> &ess_tdof_list);

  ~CahnHilliardOperator();

  void SetMobility(mfem::real_t (*f)(const mfem::Vector &));
  void BuildMatricies();

  void ComputeSource(const mfem::Vector &mu_vec,
                     mfem::Vector &source) const;

  void Mult(const mfem::Vector &x, mfem::Vector &y) const override;

  const mfem::SparseMatrix &GetMass() const;
  const mfem::SparseMatrix &GetKmob() const;

 private:
  const mfem::Array<int> ess_tdof_list_;
  mfem::FiniteElementSpace &fespace_;
  mfem::FunctionCoefficient *mobility_;
  mfem::SparseMatrix M_phi_;
  mfem::SparseMatrix K_mob_;
};
