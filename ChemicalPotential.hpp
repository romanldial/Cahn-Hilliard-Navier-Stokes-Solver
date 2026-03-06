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
        mfem::real_t epsilon = 1.0;
        mfem::real_t sigma   = 1.0;
    };

    ChemicalPotentialOperator(mfem::FiniteElementSpace  &fespace,
                              mfem::Vector              &X,
                              const mfem::Array<int>    &ess_tdof_list,
                              const Params              &params = Params());

    void SolveSystem(mfem::Vector  &phi_current,
                     mfem::Vector  &phi_next,
                     mfem::real_t   dt);

    void UpdatePhi(const mfem::Vector &X);

    void SetEpsilon(mfem::real_t epsilon);
    void SetSigma(mfem::real_t sigma);

    const mfem::SparseMatrix &GetLHS_M() const;
    const mfem::SparseMatrix &GetRHS_M() const;
    const mfem::SparseMatrix &GetRHS_K() const;
    const mfem::Vector       &GetMu()    const;

private:
    mfem::FiniteElementSpace  &fespace_;
    const mfem::Array<int>    &ess_tdof_list_;
    Params                     params_;

    mfem::ConstantCoefficient  negativeOneCoefficient;
    mfem::ConstantCoefficient  firstIntConstant;
    mfem::ConstantCoefficient  secondIntConstant;

    mfem::GridFunction         phi_lagged_gf_;

    mfem::SparseMatrix         LHS_M_;
    mfem::SparseMatrix         RHS_M_;
    mfem::SparseMatrix         RHS_K_;
    mfem::Vector               mu_;

    void BuildMatricies();
};