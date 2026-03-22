//      This is the first and simplest example of solving the Chemical 
//      Potential system using MFEM.

#include "mfem.hpp"
#include "ChemicalPotential.hpp"
#include "LILS.hpp"
#include <iostream>
#include <fstream>
#include <filesystem>

using namespace std;
using namespace mfem;

int main(int argc, char *argv[])
{
    //      Set mesh file - here it is a 1D segment
    mfem::Mesh   mesh("../data/ref-square.mesh", 1, 1);
    int dim      = mesh.Dimension();

    //      Refine the mesh - 50 elements uniformly refined
    {
        int ref_levels =
                (int)floor(log(500./mesh.GetNE())/log(2.)/dim);
                for (int l = 0; l < ref_levels; l++)
            {
                mesh.UniformRefinement();
            }
    }

    //      Bild the Finite Element Space 
    int order = 1;
    mfem::FiniteElementCollection *fec     = new mfem::H1_FECollection(order, dim);
    mfem::FiniteElementSpace *fespace      = new  mfem::FiniteElementSpace(&mesh, fec);
        cout  << "Number of finite element unknowns: "
              << fespace->GetTrueVSize() << endl;

    //      Initalize the grid functions
    mfem::GridFunction    phi_current(fespace);
    mfem::GridFunction    phi_lagged(fespace);
    srand(42);
    mfem::FunctionCoefficient ic([](const mfem::Vector &x) {
        return 0.5 * sin(2.0 * M_PI * x[0]) * sin(2.0 * M_PI * x[1]);
    });
    phi_current                = 1e-6;
    phi_lagged                 = 1e-6;
    phi_current.ProjectCoefficient(ic);
    phi_lagged                 = phi_current;

    //      Initalize the boundaries to be marked as essensial and the list that 
    //      will hold the location of those boundaries. There are set to a 
    //      Dirichlet Boundary Condition.
    Array<int>    ess_bdr(mesh.bdr_attributes.Max());
    Array<int>    ess_tdof_list;
    ess_bdr       = 1;
    fespace->GetEssentialTrueDofs(ess_bdr, ess_tdof_list);

    //      Initalize and set up the Chemical Potential Operator.
    mfem::real_t dt     = 1e-4;

    std::cout << "Building ChemicalPotentialOperator..." << std::endl;

    ChemicalPotentialOperator::Params params;
    params.epsilon = 0.1;
    params.sigma   = 1.0;

    ChemicalPotentialOperator chemPotOp(*fespace,
                                         phi_current,
                                         ess_tdof_list,
                                         params);

    std::cout << "ChemicalPotentialOperator built." << std::endl;

    //      Set Up Visit Data Collection
    mfem::VisItDataCollection *visit_dc = new mfem::VisItDataCollection("chem_pot_ex1", &mesh);
    visit_dc->SetPrefixPath("/lustre/isaac24/scratch/rdial/mfem/mfem-4.9/examples/output/chem_pot_ex1");
    visit_dc->SetPrecision(8);
    mfem::GridFunction &mu_gf = chemPotOp.GetMu();
    visit_dc->RegisterField("Chemical Potential", &mu_gf);
    visit_dc->SetCycle(0);
    visit_dc->SetTime(0.0);
    visit_dc->Save();
    int vis_steps = 1;

    mfem::real_t epsilon   = 0.1;
    mfem::real_t sigma     = 1.0;
    chemPotOp.SetEpsilon(epsilon);
        std::cout << "Epsilon set." << std::endl;
    chemPotOp.SetSigma(sigma);
        std::cout << "Sigma set." << std::endl;

    //      Set up loop for integration. Dont forget to rebuild the lagged
    //      term after the loop (for postprocessing norms).
    mfem::real_t t_i    = 1e-6;
    mfem::real_t t_f    = 0.1;
    int step = 0;
    mfem::GridFunction phi_next(fespace);
    LinearImplicitLinearSolve lils(chemPotOp.GetLHS_M(), 
                                   chemPotOp.GetRHS_K(), dt);
    while (t_i < t_f) {
        std::cout << "Step " << step << " t=" << t_i << std::endl;
        phi_lagged = phi_current;
        chemPotOp.UpdatePhi(phi_lagged);
        std::cout << "  Phi Updated" << std::endl;
        chemPotOp.SolveSystem(phi_current);
        std::cout << "  System Solved" << std::endl;
        lils.UpdateStiffness(chemPotOp.GetRHS_K());
        lils.Step(phi_current, phi_next);
        t_i += dt;
        step++;
        std::cout << "t = " << t_i << std::endl;

        if (step % vis_steps == 0) {
            visit_dc->SetCycle(step);
            visit_dc->SetTime(t_i);
            visit_dc->Save();
        }

    };
    std::cout << "  Time Integration Completed" << std::endl;

    //      Get L2 norms for verification. 
    //      Residual r = RHS_K*phi - mu + LHS_M*(phi_current - phi_lagged)/dt
    Vector Kphi(phi_current.Size());
    chemPotOp.GetRHS_K().Mult(phi_current, Kphi);

    Vector delta_phi(phi_current.Size());
    delta_phi  = phi_current;
    delta_phi -= phi_lagged;

    Vector Mdelta_phi(phi_current.Size());
    chemPotOp.GetLHS_M().Mult(delta_phi, Mdelta_phi);
    Mdelta_phi /= dt;

    Vector r(phi_current.Size());
    r  = Kphi;
    r -= chemPotOp.GetMu();
    r += Mdelta_phi;

    mfem::real_t r_l2 = r.Norml2();
    cout << "Residual L2 norm: " << r_l2 << endl;

    //      Delete the pointer
    delete visit_dc;

    return 0;
}