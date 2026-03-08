/*
    This file is for solving the 1D Heat equation. The problem can be solved analytically.
    The set up is as follows:

                                du / dt = kappa * d^2u / dx^2

                                        u(x,0) = f(x)
                                        u(0,t) = 0
                                        u(L,t) = 0
    
    Where,  f(x) = 6 * sin(M_PI * x / L)

    This is over the domain of a long thin segment. In other words the 1D heat equation. The
    analytical solution can be expressed as follows:

                   u(x,t) = 6 * sin(M_PI * x / L) * e^(-kappa (M_PI / L)^2 * t )

*/

#include "mfem.hpp"
#include "LILS.hpp"
#include <fstream>
#include <filesystem>
#include <iostream>

int main(int argc, char *argv[])
{
const char *mesh_file = "../data/ref-segment.mesh";
mfem::Mesh mesh(mesh_file, 1, 1);
int dim = mesh.Dimension();
int order = 1;
std::cout << "Number of Attributes: " << mesh.bdr_attributes.Size() << std::flush;

{
    int ref_levels =
        (int)floor(log(5000./mesh.GetNE())/log(2.)/dim);
        for (int l = 0; l < ref_levels; l++)
        {
            mesh.UniformRefinement();
        }
}

mfem::H1_FECollection fec(order, dim);
mfem::FiniteElementSpace fespace(&mesh, &fec);
std::cout << "Number of finite element unknowns: " << fespace.GetTrueVSize()
     << std::endl << "Assembling: " << std::flush;

mfem::GridFunction x(&fespace);
x = 0.0;
    
mfem::Array<int> ess_tdof_list;
mfem::Array<int> ess_bdr(mesh.bdr_attributes.Max());
ess_bdr = 1;
fespace.GetEssentialTrueDofs(ess_bdr, ess_tdof_list);
mfem::real_t zero = 0.0;
mfem::ConstantCoefficient zeroCoef(zero);

double L = 1.0;
std::function<double(const mfem::Vector&)> f_func = 
    [L](const mfem::Vector &p) {
        return 6.0 * sin(M_PI * p(0) / L);
    };
mfem::FunctionCoefficient fCoef(f_func);
x.ProjectCoefficient(fCoef);
x.ProjectBdrCoefficient(zeroCoef, ess_bdr);

mfem::real_t kappa_val = 0.5;
mfem::ConstantCoefficient kappaCoef(kappa_val);
mfem::BilinearForm lhs_mass(&fespace);
lhs_mass.AddDomainIntegrator(new mfem::MassIntegrator);
mfem::BilinearForm rhs_stiffness(&fespace);
rhs_stiffness.AddDomainIntegrator(new mfem::DiffusionIntegrator(kappaCoef));
lhs_mass.Assemble();
rhs_stiffness.Assemble();

mfem::SparseMatrix M, K;
lhs_mass.FormSystemMatrix(ess_tdof_list, M);
rhs_stiffness.FormSystemMatrix(ess_tdof_list, K);

mfem::real_t dt = 1e-3;
mfem::real_t t = 0.0;
mfem::real_t t_final = 1.0;
LinearImplicitLinearSolve LILS(M, K, dt);
mfem::Vector x_current(fespace.GetTrueVSize());
mfem::Vector x_next(fespace.GetTrueVSize());
x.GetTrueDofs(x_current);

std::function<double(const mfem::Vector&)> exact_func = 
    [L, kappa_val, &t](const mfem::Vector &p) {
        return 6.0 * sin(M_PI * p(0) / L)
               * exp(-kappa_val * pow(M_PI / L, 2) * t);
    };
mfem::FunctionCoefficient exactCoef(exact_func);

std::ofstream l2_out("l2_error.csv");
l2_out << "time,l2_error\n";

while (t < t_final) {
    LILS.Step(x_current, x_next);  
    x_current = x_next;             
    t += dt;
    x.SetFromTrueDofs(x_current);  

    mfem::real_t l2_error = x.ComputeL2Error(exactCoef);
    std::cout << "t = " << t << "  L2 Error = " << l2_error << "\n";
    l2_out << t << "," << l2_error << "\n";
}
x.SetFromTrueDofs(x_current);

mfem::Vector Mx(x.Size());
mfem::Vector Kx(x.Size());
M.Mult(x, Mx);
K.Mult(x, Kx);
mfem::real_t Mx_norm = Mx.Norml2();
mfem::real_t Kx_norm = Kx.Norml2();
std::cout << "Mass Term Norm: " << Mx_norm << "\n";
std::cout << "Stiffness Term Norm: " << Kx_norm << "\n";


return 0;
}