// This file is the first example of the Coupled Cahn-Hilliard Operator and the Chemical Potential 
// Operator. 

#include "mfem.hpp"
#include "ChemicalPotential.hpp"
#include "CahnHilliard.hpp"
#include "LILS.hpp"
#include <iostream>
#include <fstream>
#include <filesystem>

int main(int argc, char *argv[])
{
    //      Set mesh file - here it is a 1D segment
    mfem::Mesh   mesh("../data/ref-square.mesh", 1, 1);
    int dim      = mesh.Dimension();

    //      Refine the mesh - 50 elements uniformly refined
    {
        int ref_levels =
                (int)floor(log(5000./mesh.GetNE())/log(2.)/dim);
                for (int l = 0; l < ref_levels; l++)
            {
                mesh.UniformRefinement();
            }
    }

    //      Bild the Finite Element Space 
    int order = 1;
    mfem::FiniteElementCollection *fec     = new  mfem::H1_FECollection(order, dim);
    mfem::FiniteElementSpace *fespace      = new  mfem::FiniteElementSpace(&mesh, fec);
        std::cout  << "Number of finite element unknowns: "
              << fespace->GetTrueVSize() << std::endl;

    //      Initalize the grid functions & Initalize Initial Condition
    mfem::GridFunction phi_lagged(fespace);
    mfem::GridFunction phi_current(fespace);
    mfem::GridFunction mu(fespace);

    mfem::FunctionCoefficient ic([](const mfem::Vector &x) {
        return 0.5 * sin(2.0 * M_PI * x[0]) * sin(2.0 * M_PI * x[1]);
    });

    phi_lagged                 = 1e-6;
    phi_current                = 1e-6;
    mu                         = 1e-6;
    phi_current.ProjectCoefficient(ic);
    
    //      Initalize the boundaries to be marked as essensial and the list that 
    //      will hold the location of those boundaries. There are set to a 
    //      Neumann Boundary Condition.
    mfem::Array<int>    ess_tdof_list;
    

    //      Initalize and set up the Chemical Potential Operator.
    mfem::real_t dt     = 1e-10;

    std::cout << "Building ChemicalPotentialOperator..." << std::endl;
    ChemicalPotentialOperator::Params params;
    params.epsilon = 0.02;
    params.sigma   = 1.0;

    ChemicalPotentialOperator chemPotOp(*fespace,
                                         phi_current,
                                         ess_tdof_list,
                                         params);

    std::cout << "ChemicalPotentialOperator built." << std::endl;

    //     Initalize and set up the Cahn Hilliard Operator with constant Mobility.
    std::cout << "Building CahnHilliardOperator..." << std::endl;
    CahnHilliardOperator cahnHilOp(*fespace, ess_tdof_list);
    std::cout << "CahnHilliardOperator Built." << std::endl;
    std::cout << "Setting Mobility..." << std::endl;
    cahnHilOp.SetMobility([](const mfem::Vector &x) { return 1.0; });
    cahnHilOp.BuildMatricies();

    //      Set Up Visit Data Collection
    mfem::VisItDataCollection *visit_dc = new mfem::VisItDataCollection("ch_couple_ex1", &mesh);
    visit_dc->SetPrefixPath("/lustre/isaac24/scratch/rdial/mfem/mfem-4.9/examples/output/ch_couple_ex1");
    visit_dc->SetPrecision(8);
    visit_dc->RegisterField("Order Paramater", &phi_current);
    visit_dc->SetCycle(0);
    visit_dc->SetTime(0.0);
    visit_dc->Save();
    int vis_steps = 556;

    std::ofstream energy_file("energy.csv");
    energy_file << "step,time,E_bulk,E_interface,E_total,mass\n";

    //      Set up loop for integration. Dont forget to rebuild the lagged
    //      term after the loop. 
    mfem::real_t t_i    = 0.0;
    mfem::real_t t_f    = 1e-2;
    int step = 0;
    LinearImplicitLinearSolve lils(cahnHilOp.GetMass(),
                                  cahnHilOp.GetKmob(),
                                  dt,
                                  ess_tdof_list);

    mfem::GridFunction source(fespace);
    source = 0.0;

    //      Checks before loop
    std::cout << "mu norm: "   << mu.Norml2()    << std::endl;
    std::cout << "phi norm: "  << phi_current.Norml2() << std::endl;


    //      Ensure that the mean of phi stays constant. This is a property of 
    //      the Neumann condition for mass conservation.
        mfem::ConstantCoefficient one_coeff(1.0);

        mfem::LinearForm one(fespace);
        one.AddDomainIntegrator(new mfem::DomainLFIntegrator(one_coeff));
        one.Assemble();

        double volume = one.Sum();
        double initial_mass = one * phi_current;

    //      Use this Bilinear Form to Calculate the Energy Norms
        mfem::BilinearForm grad_squared(fespace);
        grad_squared.AddDomainIntegrator(new mfem::DiffusionIntegrator());
        grad_squared.Assemble();
        mfem::GridFunction bulk(fespace);


    while (t_i < t_f) {
        std::cout << "Step " << step << std::endl;

        phi_lagged = phi_current;

        chemPotOp.UpdatePhi(phi_lagged);

        chemPotOp.SolveSystem(phi_lagged);

        mfem::GridFunction &mu_gf = chemPotOp.GetMu();

        cahnHilOp.ComputeSource(mu_gf, source);

        source.Neg();

        mfem::GridFunction phi_next(fespace);
        phi_next = 0.0;
        lils.Step(phi_current, source, phi_next);

            //  correction for constant phi mean
            double current_mass = one * phi_next;
            double correction = (current_mass - initial_mass) / volume;
                for (int i = 0; i < phi_next.Size(); i++)
                {
                    phi_next(i) -= correction;
                }
    
        phi_current = phi_next; 

            // Calculate Energy Norms
                // Bulk energy: (3/16) * (sigma/epsilon) * (phi^2 - 1)^2
                for (int i = 0; i < phi_current.Size(); i++)
                {
                    double p = phi_current(i);
                    bulk(i) = (p*p - 1.0) * (p*p - 1.0);
                }
                double E_bulk      = (3.0/16.0) * (params.sigma / params.epsilon) * (one * bulk);

                // Interface energy: (3/4) * sigma * epsilon * (grad phi, grad phi)
                double E_interface = (3.0/4.0) * params.sigma * params.epsilon * grad_squared.InnerProduct(phi_current, phi_current);

                double E_total = E_bulk + E_interface;

                energy_file << step << ","
                            << t_i  << ","
                            << E_bulk << ","
                            << E_interface << ","
                            << E_total << ","
                            << (one * phi_current) << "\n";

        t_i += dt;
        step++;
        std::cout << "t = " << t_i << std::endl;

        if (step % vis_steps == 0) {
            visit_dc->SetCycle(step);
            visit_dc->SetTime(t_i);
            visit_dc->Save();
        }

        

    };
    energy_file.close();

    std::cout << " \n "<<"  Time Integration Completed" << std::endl;

    delete visit_dc;

    return 0;
}