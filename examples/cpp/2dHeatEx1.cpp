// Heat Equation Example with Dirichlet and Natural Boundary Conditions

/*
    This file is for solving the 2D Heat equation. The problem can be solved analytically.
    The set up is as follows:

                                du / dt = kappa * nabla^2 u

                                        u(x,0) = f(x)
                                        u(x,t) = 0 on \partial\Omega
                                        du/dn  = 0 on \partial\Omega
    
    Where,  f(x) = sin(2πx) * sin(πy)
*/

#include "mfem.hpp"
#include "LILS.hpp"
#include <fstream>
#include <filesystem>
#include <iostream>

int main(int argc, char *argv[])
{

//  1. Here we will choose the mesh file, build the mesh, and refine the mesh.

    const char *mesh_file = "../data/ref-square.mesh";
    mfem::Mesh mesh(mesh_file, 1, 1);
    int dim = mesh.Dimension();
    int order = 1;
    std::cout << "Number of Attributes: " << mesh.bdr_attributes.Size() << std::endl;

    {
        int ref_levels =
            (int)floor(log(50000./mesh.GetNE())/log(2.)/dim);
            for (int l = 0; l < ref_levels; l++)
                {
                    mesh.UniformRefinement();
                }
        }



//  2. Here we set up the finite element space

    mfem::H1_FECollection          fec(order, dim);
    mfem::FiniteElementSpace       fespace(&mesh, &fec);
    std::cout << "Number of finite element unknowns: " << fespace.GetTrueVSize()
              << std::endl;



//  3. Here we set up the grid function

    mfem::GridFunction             u(&fespace);
    u = 0.0;



//  4. Here we set up the initial condition and project it onto the mesh.

    mfem::FunctionCoefficient ic([](const mfem::Vector &x) {
    return sin(M_PI * x[0]) * sin(M_PI * x[1]);
    });
    u.ProjectCoefficient(ic);



//  5. Here we set up the homogeneous Dirichlet boundary conditions to all boundaries of the domain.
//     This is done by creating a marker array for the boundaries, marking all boundaries as essential
//     for the linear solver, and then setting the list of essential true dofs to zero.   

//     The Natural boundary conditions are automatically applied by the weak formulation of the problem. 
//     This is reflected by the lack of an integral about the boundary in the linear form.

    //      Value for the specific Dirichlet boundary Condition 
    mfem::ConstantCoefficient zero_boundary(0.0);

    //      Explicitly list which boundaries are Dirichlet. In this case all boundaries are Dirichlet.
    mfem::Array<int> dirichlet_attrs;
    for (int i = 1; i <= mesh.bdr_attributes.Max(); i++)
    {
        dirichlet_attrs.Append(i);
    }

    //      Marker Array for the selected boundaries. Everything selected from the dirichlet_attrs will
    //      be included.
    mfem::Array<int> ess_bdr_marker(mesh.bdr_attributes.Max());
    ess_bdr_marker = 0;
    for (int i = 0; i < dirichlet_attrs.Size(); i++)
    {
        ess_bdr_marker[dirichlet_attrs[i] - 1] = 1;  // attributes are 1-indexed
    }

    //      Project the boundary condition values onto the mesh. 
    u.ProjectBdrCoefficient(zero_boundary, ess_bdr_marker);

    //     Convert the boundary marker array to a list of essential true dofs for the linear solver.
    mfem::Array<int> ess_tdof_list;
    fespace.GetEssentialTrueDofs(ess_bdr_marker, ess_tdof_list);



//   6. Here we set up the bilinear form of the problem. The bilinear form is conventionally 
//      the left hand side of the weak formulation of the problem.

    //      du/dt * v                          | mass term
    mfem::BilinearForm          lhs_mass(&fespace);
    mfem::ConstantCoefficient   mass_coeff(1.0);
    lhs_mass.AddDomainIntegrator(new mfem::MassIntegrator(mass_coeff));
    lhs_mass.Assemble();

    //      alpha * (nabla v \dot nabla u)     | stiffness term
    mfem::BilinearForm          rhs_stiffness(&fespace);
    mfem::ConstantCoefficient   diffusion_coeff(1.0);
    rhs_stiffness.AddDomainIntegrator(new mfem::DiffusionIntegrator(diffusion_coeff));
    rhs_stiffness.Assemble();



//   8. Set up the linear solver

    //      Extract the sparse matrix from the bilinear forms and set up the linear
    //      implicit linear solve.
    mfem::SparseMatrix &M = lhs_mass.SpMat();
    mfem::SparseMatrix &K = rhs_stiffness.SpMat();

    //      Set the time step size and initialize the linear solver.
    mfem::real_t       dt = 5e-5;
    LinearImplicitLinearSolve lils(M, K, dt, ess_tdof_list);



//   10. Visualization set up before the loop

    std::ofstream temp_data("temp.csv");
    mfem::VisItDataCollection *visit_dc = new mfem::VisItDataCollection("2dHeatEx1", &mesh);
    visit_dc->SetPrefixPath("/lustre/isaac24/scratch/rdial/mfem/mfem-4.9/examples/output/2dHeatEx1");
    visit_dc->SetPrecision(8);
    visit_dc->RegisterField("temperature", &u);
    visit_dc->SetCycle(0);
    visit_dc->SetTime(0.0);
    visit_dc->Save();
    int step = 0;
    int vis_steps = 1;



//   9. Perform the time stepping integration. 

    mfem::GridFunction u_next(&fespace);
    mfem::real_t t_i = 0.0;
    mfem::real_t t_f = 0.15;

    for (int i = 0; t_i < t_f; i++)
    {
        std::cout << "t = " << t_i << std::endl;

        lils.Step(u, u_next);
        u = u_next;
        t_i += dt;
        step++;

        if (step % vis_steps == 0) 
        {
            visit_dc->SetCycle(step);
            visit_dc->SetTime(t_i);
            visit_dc->Save();
        }
    }
    

delete visit_dc;

return 0;
}