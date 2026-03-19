    //     To begin here the chemical potential function is defined as the Lau 
    //     Ginzburg type potential. This means that the chemical potential can 
    //     be expressed as:
    //
    //               mu =   (3sigma/4epsilon)(phi)(phi^2-1) 
    //                    - (3/2)(sigma*epsilon)(Laplacial phi)
    //
    //     This potential can then be expressed in the weak formulation as follows:
    //     
    //      int{mu * v}d\Omega =   int{(3sigma/4epsilon)(phi)(phi^2-1) * v} d\Omega 
    //                           + Int{(3/2)(sigma*epsilon)(Nabla phi dot Nabla v)}d\Omega
    //                           - Int{(3/2)(sigma*epsilon)(Nabla phi dot n_hat) * v} dS
    //
    //      In our case, the surface integral term will vanish due to conservation 
    //      of mass.
#include "mfem.hpp"
#include "LILS.hpp"
#include "ChemicalPotential.hpp"
#include <iostream>

using namespace std;
using namespace mfem;

ChemicalPotentialOperator::ChemicalPotentialOperator(
    mfem::FiniteElementSpace  &fespace,
    mfem::Vector              &X,
    const mfem::Array<int>    &ess_tdof_list,
    mfem::real_t              dt,
    const Params              &params)
    : mfem::Operator(fespace.GetTrueVSize()),
      fespace_(fespace),
      ess_tdof_list_(ess_tdof_list),
      params_(params),
      dt_(dt)
{
    std::cout << "  Setting space..." << std::endl;
    phi_lagged_gf_.SetSpace(&fespace_);
    std::cout << "  Setting From True Dofs..." << std::endl;
    phi_lagged_gf_.SetFromTrueDofs(X);
    std::cout << "  Building Matricies..." << std::endl;
    BuildMatricies();
    std::cout << "  LHS_M size: " << LHS_M_.Height() << "x" << LHS_M_.Width() << std::endl;
    std::cout << "  RHS_K size: " << RHS_K_.Height() << "x" << RHS_K_.Width() << std::endl;
    std::cout << "  Creating LILS..." << std::endl;
    lils_ = new LinearImplicitLinearSolve(LHS_M_, RHS_K_, dt);
    std::cout << "  Constructor done." << std::endl;
}

ChemicalPotentialOperator::~ChemicalPotentialOperator()
{
    delete lils_;
}

void ChemicalPotentialOperator::BuildMatricies()
{
    mfem::real_t firstConstant  = (3.0 * params_.sigma) / (4.0 * params_.epsilon);
    mfem::real_t secondConstant = (3.0 * params_.sigma * params_.epsilon) / 2.0;

    mfem::ConstantCoefficient negOneCoef(-1.0);
    mfem::ConstantCoefficient firstCoef(firstConstant);
    mfem::ConstantCoefficient secondCoef(secondConstant);

    mfem::GridFunctionCoefficient phi_lagged_coef(&phi_lagged_gf_);
    mfem::ProductCoefficient      a_term(firstCoef, phi_lagged_coef);
    mfem::ProductCoefficient      phiSquared(phi_lagged_coef, phi_lagged_coef);
    mfem::SumCoefficient          phiSquared_one(phiSquared, negOneCoef);
    mfem::ProductCoefficient      nonlinear_mass_term(a_term, phiSquared_one);

    mfem::BilinearForm            RHS_mass(&fespace_);
    mfem::BilinearForm            RHS_stiffness(&fespace_);
    mfem::BilinearForm            LHS_mass(&fespace_);

    RHS_mass.AddDomainIntegrator(new mfem::MassIntegrator(nonlinear_mass_term));
    RHS_stiffness.AddDomainIntegrator(new mfem::DiffusionIntegrator(secondCoef));
    LHS_mass.AddDomainIntegrator(new mfem::MassIntegrator());

    RHS_mass.Assemble();
    RHS_stiffness.Assemble();
    LHS_mass.Assemble();

    mfem::OperatorPtr op_RHS_M, op_RHS_K, op_LHS_M;
    mfem::Vector dummy_x, dummy_b;

    RHS_mass.FormSystemMatrix(ess_tdof_list_, op_RHS_M);
    RHS_stiffness.FormSystemMatrix(ess_tdof_list_, op_RHS_K);
    LHS_mass.FormSystemMatrix(ess_tdof_list_, op_LHS_M);

    RHS_M_ = *op_RHS_M.As<mfem::SparseMatrix>();
    RHS_K_ = *op_RHS_K.As<mfem::SparseMatrix>();
    LHS_M_ = *op_LHS_M.As<mfem::SparseMatrix>();
}

void ChemicalPotentialOperator::SolveSystem(mfem::Vector &phi_current,
                                             mfem::Vector &phi_next,
                                             mfem::real_t dt)
{
    lils_->SetTimeStep(dt);

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

    mfem::Vector rhs_phi(phi_current.Size());
    LHS_M_.Mult(phi_current, rhs_phi);
    lils_->StepWithRHS(rhs_phi, phi_next);
    phi_current = phi_next;
}

void ChemicalPotentialOperator::UpdatePhi(const mfem::GridFunction &phi)
{
    phi_lagged_gf_ = phi;
    BuildMatricies();
    lils_->UpdateStiffness(RHS_K_);
}

void ChemicalPotentialOperator::SetEpsilon(mfem::real_t epsilon)
{
    std::cout << "  SetEpsilon called..." << std::endl;
    params_.epsilon = epsilon;
    BuildMatricies();
    lils_->UpdateStiffness(RHS_K_);
}

void ChemicalPotentialOperator::SetSigma(mfem::real_t sigma)
{
    std::cout << "  SetSigma called..." << std::endl;
    params_.sigma = sigma;
    BuildMatricies();
    lils_->UpdateStiffness(RHS_K_);
}

const mfem::SparseMatrix &ChemicalPotentialOperator::GetLHS_M() const { return LHS_M_; }
const mfem::SparseMatrix &ChemicalPotentialOperator::GetRHS_M() const { return RHS_M_; }
const mfem::SparseMatrix &ChemicalPotentialOperator::GetRHS_K() const { return RHS_K_; }
const mfem::Vector       &ChemicalPotentialOperator::GetMu()    const { return mu_;    }