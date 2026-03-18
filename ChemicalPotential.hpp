// Header file for ChemicalPotentialOperator: Landau-Ginzburg 
// chemical potential operator for Cahn-Hilliard phase field 
// simulations.

#pragma once

#include "mfem.hpp"
#include "LILS.hpp"

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
                              mfem::real_t              dt,
                              const Params              &params = Params());
    
    ~ChemicalPotentialOperator();

    void SolveSystem(mfem::Vector  &phi_current,
                     mfem::Vector  &phi_next,
                     mfem::real_t   dt);

    void UpdatePhi(const mfem::GridFunction &phi);

    void SetEpsilon(mfem::real_t epsilon);
    void SetSigma(mfem::real_t sigma);

    const mfem::SparseMatrix &GetLHS_M() const;
    const mfem::SparseMatrix &GetRHS_M() const;
    const mfem::SparseMatrix &GetRHS_K() const;
    const mfem::Vector       &GetMu()    const;

    void Mult(const mfem::Vector &x, mfem::Vector &y) const override {}

private:
    LinearImplicitLinearSolve* lils_;
    mfem::real_t               dt_;
    mfem::Vector               X_;

    mfem::FiniteElementSpace   &fespace_;
    const mfem::Array<int>     &ess_tdof_list_;
    Params                     params_;

    mfem::GridFunction         phi_lagged_gf_;

    mfem::SparseMatrix         LHS_M_;
    mfem::SparseMatrix         RHS_M_;
    mfem::SparseMatrix         RHS_K_;
    mfem::Vector               mu_;

    void BuildMatricies();
};