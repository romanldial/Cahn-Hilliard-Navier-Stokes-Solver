#include "LILS.hpp"

//      We want to create a linear implicic solve for linear problems. 
//      For this problem we will use the form:
//
//                             M du/dt = - K u 
//
//      where M is the mass matrix and K is the stiffness matrix. This 
//      solve can be approximated by a backward difference becoming:
//
//                         (M+dt*K)v^n+1 = M*v^n
//
//      We will take the mass and stiffness matricies from the caller 
//      and use a Conjugate Gradient solver with a Gauss-Seidel
//      preconditioner to solve the system.
LinearImplicitLinearSolve::LinearImplicitLinearSolve(mfem::SparseMatrix &M,
                                                     mfem::SparseMatrix &K,
                                                     mfem::real_t dt)
   : M_(M),               // Mass matrix
     K_(K),               // Stiffness matrix
     dt_(dt),             // Time step size
     T_(nullptr),         // System matrix (M + dt*K)
     rhs_(M.Height()),    // Right-hand side vector
     lin_solver_()        // Linear solver
{
   std::cout << "    LILS: Building system matrix..." << std::endl;
   BuildSystemMatrix();
   std::cout << "    LILS: T_ size: " << T_->Height() << "x" << T_->Width() << std::endl;
   std::cout << "    LILS: Configuring solver..." << std::endl;
   ConfigureLinearSolver();
   std::cout << "    LILS: Constructor done." << std::endl;
}

void LinearImplicitLinearSolve::SetTimeStep(const mfem::real_t dt)
{
   dt_ = dt;
   BuildSystemMatrix();
   ConfigureLinearSolver();
}

mfem::real_t LinearImplicitLinearSolve::GetTimeStep() const
{
   return dt_;
}

void LinearImplicitLinearSolve::Step(mfem::Vector &u_current,
                                     mfem::Vector &u_next)
{
   M_.Mult(u_current, rhs_);
   lin_solver_->Mult(rhs_, u_next);

   mfem::Vector Au(rhs_.Size());
   T_->Mult(u_next, Au);
   Au -= rhs_;
   const mfem::real_t rhs_n = rhs_.Norml2();
   const mfem::real_t rel_res = Au.Norml2() / (rhs_n > 0.0 ? rhs_n : 1.0);
   MFEM_VERIFY(rel_res == rel_res, "Linear solve residual is NaN.");
}

void LinearImplicitLinearSolve::Step(mfem::Vector &u_current,
                                     const mfem::Vector &source,
                                     mfem::Vector &u_next)
{
   M_.Mult(u_current, rhs_);
   rhs_.Add(dt_, source);
   lin_solver_->Mult(rhs_, u_next);


   mfem::Vector Au(rhs_.Size());
   T_->Mult(u_next, Au);
   Au -= rhs_;
   const mfem::real_t rhs_n = rhs_.Norml2();
   const mfem::real_t rel_res = Au.Norml2() / (rhs_n > 0.0 ? rhs_n : 1.0);
   MFEM_VERIFY(rel_res == rel_res, "Linear solve residual is NaN.");
}

void LinearImplicitLinearSolve::StepWithRHS(const mfem::Vector &rhs,
                                             mfem::Vector &u_next)
{
   lin_solver_->Mult(rhs, u_next);

   mfem::Vector Au(rhs.Size());
   T_->Mult(u_next, Au);
   Au -= rhs;
   const mfem::real_t rhs_n = rhs.Norml2();
   const mfem::real_t rel_res = Au.Norml2() / (rhs_n > 0.0 ? rhs_n : 1.0);
   MFEM_VERIFY(rel_res == rel_res, "Linear solve residual is NaN.");
}

void LinearImplicitLinearSolve::BuildSystemMatrix()
{
   if (!M_.Finalized()) M_.Finalize();
   if (!K_.Finalized()) K_.Finalize();
   auto newT = std::make_unique<mfem::SparseMatrix>(M_);
   newT->Add(dt_, K_);                      // T = M + dt*K
   newT->Finalize();
   T_ = std::move(newT);
}

void LinearImplicitLinearSolve::ConfigureLinearSolver()
{
   A_prec_ = std::make_unique<mfem::GSSmoother>(*T_);
   lin_solver_ = std::make_unique<mfem::CGSolver>();
   lin_solver_->SetOperator(*T_);
   lin_solver_->SetPreconditioner(*A_prec_);
   lin_solver_->SetRelTol(1e-8);
   lin_solver_->SetAbsTol(0.0);
   lin_solver_->SetMaxIter(1000);
   lin_solver_->SetPrintLevel(0);
}

void LinearImplicitLinearSolve::UpdateStiffness(mfem::SparseMatrix &K)
{
   K_ = *mfem::Add(1.0, K, 0.0, K);
   BuildSystemMatrix();
   ConfigureLinearSolver();
}

void LinearImplicitLinearSolve::UpdateMass(mfem::SparseMatrix &M)
{
    M_ = *mfem::Add(1.0, M, 0.0, M);
    BuildSystemMatrix();
    ConfigureLinearSolver();
}