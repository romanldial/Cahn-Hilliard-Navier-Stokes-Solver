// Header file for ChemicalPotentialOperator: Landau-Ginzburg 
// chemical potential operator for Cahn-Hilliard phase field 
// simulations.

#pragma once

#include "mfem.hpp"

class ChemicalPotentialOperator : public mfem::Operator
{
public:
    struct Params
    {
        mfem::real_t epsilon;
        mfem::real_t sigma;
        Params() : epsilon(1.0), sigma(1.0) {}
    };

    ChemicalPotentialOperator(mfem::FiniteElementSpace  &fespace,
                              mfem::Vector              &X,
                              const mfem::Array<int>    &ess_tdof_list,
                              const Params              &params = Params());
    
    ~ChemicalPotentialOperator();

    void SolveSystem(mfem::Vector  &phi_current);

    void UpdatePhi(const mfem::GridFunction &phi);

    void SetEpsilon(mfem::real_t epsilon);
    void SetSigma(mfem::real_t sigma);

    const mfem::SparseMatrix &GetLHS_M() const;
    const mfem::SparseMatrix &GetRHS_M() const;
    const mfem::SparseMatrix &GetRHS_K() const;
          mfem::GridFunction &GetMu();

    void Mult(const mfem::Vector &x, mfem::Vector &y) const override {}

private:
    mfem::Vector               X_;

    mfem::FiniteElementSpace   &fespace_;
    const mfem::Array<int>     &ess_tdof_list_;
    Params                     params_;

    mfem::GridFunction         phi_lagged_gf_;

    mfem::SparseMatrix         LHS_M_;
    mfem::SparseMatrix         RHS_M_;
    mfem::SparseMatrix         RHS_K_;
    mfem::GridFunction         mu_;

    void BuildMatricies();
};