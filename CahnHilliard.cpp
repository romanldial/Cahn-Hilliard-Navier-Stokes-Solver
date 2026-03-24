#include "mfem.hpp"
#include "ChemicalPotential.hpp"
#include "LILS.hpp"

CahnHilliardOperator::CahnHilliardOperator(
    mfem::FiniteElementSpace  &fespace,
    const mfem::Array<int>    &ess_tdof_list,
    mfem::real_t (*mobility)(const mfem::Vector &))
    : mfem::Operator(fespace.GetTrueVSize()),
      fespace_(fespace),
      ess_tdof_list_(ess_tdof_list)
{
    std::cout << "  Building Matricies..." << std::endl;
    BuildMatricies();
}

CahnHilliardOperator::~CahnHilliardOperator()
{
}

void CahnHilliardOperator::SetMobility(mfem::real_t (*f)(const mfem::Vector &))
{
    mobility_ = mfem::FunctionCoefficient(f);
}

void CahnHilliardOperator::BuildMatricies()
{
    mfem::BilinearForm       b_phi_(&fespace_);
    mfem::BilinearForm       b_mu_(&fespace_);

    b_phi_.AddDomainIntegrator(new mfem::MassIntegrator());
    b_mu_.AddDomainIntegrator(new mfem::DiffusionIntegrator(mobility_));

    b_phi_.Assemble();
    b_mu_.Assemble();

    mfem::OperatorPtr op_phi, op_mu;

    b_phi_.FormSystemMatrix(ess_tdof_list_, op_phi);
    b_mu_.FormSystemMatrix(ess_tdof_list_, op_mu);

    PHI_  =  *op_phi.As<mfem::SparseMatrix>();
    MU_   =  *op_mu.As<mfem::SparseMatrix>();
}

void CahnHilliardOperator::SolveSystem(const mfem::Vector &mu_vec,
                                        mfem::GridFunction &phi_out)
{
    mfem::Vector rhs(mu_vec.Size());
    phi_out.SetSpace(&fespace_);

    // Form RHS: -M(mobility) * mu
    MU_.Mult(mu_vec, rhs);
    rhs.Neg();

    // Solve: PHI_ * phi = rhs
    mfem::GSSmoother prec(PHI_);
    mfem::CGSolver cg;
    cg.SetOperator(PHI_);
    cg.SetPreconditioner(prec);
    cg.SetRelTol(1e-8);
    cg.SetMaxIter(1000);
    cg.SetPrintLevel(0);
    cg.Mult(rhs, phi_out);
}

mfem::FunctionCoefficient &CahnHilliardOperator::GetMobility() { return mobility_; }