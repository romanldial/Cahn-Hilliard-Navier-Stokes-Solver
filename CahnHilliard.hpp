// Header file for CahnHilliard.cpp

#pragma once

#include "mfem.hpp"
#include "ChemicalPotential.hpp"
#include "LILS.hpp"

class CahnHilliardOperator : public mfem::Operator
{
public:
    CahnHilliardOperator(mfem::FiniteElementSpace  &fespace,
                         const mfem::Array<int>    &ess_tdof_list);

    ~CahnHilliardOperator();

    void SetMobility(mfem::real_t (*f)(const mfem::Vector &));

    void BuildMatricies();

    void SolveSystem();

    mfem::FunctionCoefficient &GetMobility();
private:
        mfem::FiniteElementSpace  &fespace;
        const mfem::Array<int>    &ess_tdof_list;

        mfem::GridFunction        mu_;
        mfem::FunctionCoefficient mobility_;
        mfem::SparseMatrix        PHI_, MU_;
};