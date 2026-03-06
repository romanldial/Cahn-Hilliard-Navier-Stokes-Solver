#include "mfem.hpp"
#include <iostream>

using namespace std;
using namespace mfem;

class ChemicalPotentialOperator : public mfem::Operator
{
    //     To begin, here the chemical potential function is defined as the Lau 
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
public:
    struct Params
    {
        mfem::real_t epsilon   = 1.0;  // Interface width
        mfem::real_t sigma     = 1.0;  // Surface tension
    };

private:
    LinearImplicitLinearSolve* lils_;
    mfem::FiniteElementSpace   &fespace_;
    mfem::real_t               dt_;
    const mfem::Array<int>     &ess_tdof_list_;
    Params params_;

    mfem::ConstantCoefficient  negativeOneCoefficient;
    mfem::ConstantCoefficient  firstIntConstant;
    mfem::ConstantCoefficient  secondIntConstant;

    mfem::GridFunction         phi_lagged_gf_;

    mfem::SparseMatrix         LHS_M_;
    mfem::SparseMatrix         RHS_M_;
    mfem::SparseMatrix         RHS_K_;
    mfem::Vector               mu_;
    mfem::Vector               X_;
    void BuildMatricies(){
    //      This defines and builds the matricies based on the Lau Ginzburg type potential
    //      inside the chemical potential's weak form. This assembles the complete right 
    //      hand side of the weak form.
      mfem::real_t firstConstant  = (3.0 * params_.sigma) / (4.0 * params_.epsilon);
      mfem::real_t secondConstant = (3.0 * params_.sigma * params_.epsilon) / (2.0);
      mfem::real_t negativeOne    = -1.0;
      mfem::ConstantCoefficient negativeOneCoefficient = ConstantCoefficient(negativeOne);
      mfem::ConstantCoefficient firstIntConstant       = ConstantCoefficient(firstConstant);
      mfem::ConstantCoefficient secondIntConstant      = ConstantCoefficient(secondConstant);

      mfem::GridFunctionCoefficient   phi_lagged_coef(&phi_lagged_gf_);
      mfem::ProductCoefficient        a_term(firstIntConstant, phi_lagged_coef);
      mfem::ProductCoefficient        phiSquared(phi_lagged_coef, phi_lagged_coef);
      mfem::SumCoefficient            phiSquared_one(phiSquared, negativeOneCoefficient);
      mfem::ProductCoefficient        nonlinear_mass_term(a_term, phiSquared_one);
    
      mfem::BilinearForm            RHS_mass(&fespace_);
      mfem::BilinearForm            RHS_stiffness(&fespace_);
      RHS_mass.AddDomainIntegrator(new mfem::MassIntegrator(nonlinear_mass_term));
      RHS_stiffness.AddDomainIntegrator(new mfem::DiffusionIntegrator(secondIntConstant));

      RHS_mass.Assemble();
      RHS_stiffness.Assemble();
      RHS_mass.FormSystemMatrix(ess_tdof_list_, RHS_M_);
      RHS_stiffness.FormSystemMatrix(ess_tdof_list_, RHS_K_);

    //      This builds and assembles the mass matrix for the left hand side of the equation.
      mfem::BilinearForm     LHS_mass(&fespace_);
      LHS_mass.AddDomainIntegrator(new mfem::MassIntegrator());
      LHS_mass.Assemble();
      LHS_mass.FormSystemMatrix(ess_tdof_list_, LHS_M_);
    }

public: 
    ChemicalPotentialOperator(mfem::FiniteElementSpace &fespace,
                              mfem::Vector             &X,
                              const mfem::Array<int>   &ess_tdof_list,
                              mfem::real_t             dt,
                              const Params &params = Params())
                : fespace_(fespace), 
                  ess_tdof_list_(ess_tdof_list), 
                  params_(params),
                  dt_(dt)
    { 
        phi_lagged_gf_.SetFromTrueDofs(X);
        BuildMatricies();
        lils_ = new LinearImplicitLinearSolve(LHS_M_, RHS_K_, dt);
    }

    //      This method uses inherited and home methods to solve the system after
    //      the user builds the matricies.
    void SolveSystem(mfem::Vector &phi_current,
                     mfem::Vector &phi_next,
                     mfem::real_t dt)
    {
    //      We need to match the paramaters for what to feed lils_. We will ultamately 
    //      take the form LHS_M_ * mu = RHS_M_ * ones + RHS_K_ * phi_current. 
        mfem::Vector    ones(phi_current.Size());
        ones            = 1.0;

        X_ = phi_current;

        mfem::Vector    rhs_nonlinear(phi_current.Size());
        mfem::Vector    rhs_stiffness(phi_current.Size());
        mfem::Vector    rhs_mu_complete(phi_current.Size());
                        mu_.SetSize(phi_current.Size());

        RHS_M_.Mult(ones, rhs_nonlinear);
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

    //      This is where we compute (LHS_M_ + dt * RHS_K_) * phi_next = LHS_M_ * phi_current
    //      for this single step. The user must set up a loop for this member to walk the 
    //      time stepper member foreward. 
        lils_->UpdateMass(RHS_M_);
        lils_->UpdateStiffness(RHS_K_);
        lils_->Step(phi_current, phi_next);
        phi_current = phi_next;
    }

    //      This user should use this in the solving loop in order to update the current phi 
    //      gridfunction.
    void UpdatePhi(const mfem::Vector &X)
    {
        phi_lagged_gf_.SetFromTrueDofs(X);
        BuildMatricies();  
    }

    void SetEpsilon(mfem::real_t epsilon){
        params_.epsilon = epsilon;
        BuildMatricies();
    }

    void SetSigma(mfem::real_t sigma){
        params_.sigma = sigma;
        BuildMatricies();
    }

    const mfem::SparseMatrix &GetLHS_M()  const { return LHS_M_; }
    const mfem::SparseMatrix &GetRHS_M()  const { return RHS_M_; }
    const mfem::SparseMatrix &GetRHS_K()  const { return RHS_K_; }
    const mfem::Vector       &GetMu()     const { return mu_; }

    ~ChemicalPotentialOperator()                { delete lils_; }
};