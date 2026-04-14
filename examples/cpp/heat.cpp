#include "mfem.hpp"
#include <fstream>
#include <iostream>

using namespace std;
using namespace mfem;

int main(int argc, char *argv[])
{

    // MFEM Hardcoded options
    const char *mesh_file = "../data/ref-square.mesh";
    int order = 2;
    bool static_cond = false;
    const char *device_config = "cpu";
    bool visualization = true;
    bool algebraic_ceed = false;

    Device device(device_config);
    device.Print();


    //   1. Read Mesh File, take dimension, and print number of attributes.
    Mesh mesh(mesh_file, 1, 1);
    int dim = mesh.Dimension();
    cout << "Number of Attributes: " << mesh.bdr_attributes.Size() << flush;
    

    //   2. Refine the mesh for resolution. We choose 'ref_levels' to be the
    //      largest number that gives a final mesh with no more than 5,000
    //      elements.
    {
    int ref_levels =
        (int)floor(log(5000./mesh.GetNE())/log(2.)/dim);
        for (int l = 0; l < ref_levels; l++)
        {
            mesh.UniformRefinement();
        }
    }


    //   3. Define the finite element space on the mesh. 
    FiniteElementCollection *fec;
    FiniteElementSpace *fespace;
    fec = new H1_FECollection(order, dim);
    fespace = new FiniteElementSpace(&mesh, fec);
    cout << "Number of finite element unknowns: " << fespace->GetTrueVSize()
        << endl << "Assembling: " << flush;


    //   4. Determine the list of essensial boundary dofs.
    //      Tell MFEM that the outer edge of the outer edge
    //      of the mesh is essensial. Atribute 1 is bottom, 
    //      then 2 is right, 3 is top, and 4 is left. 
    Array<int> ess_tdof_list, left_bdr(mesh.bdr_attributes.Max());
    cout << mesh.bdr_attributes.Max() << flush;
    left_bdr = 0;
    left_bdr[4-1] = 1;
    fespace->GetEssentialTrueDofs(left_bdr, ess_tdof_list);
    

    //   5. Set the coefficents used later in assembly 
    real_t alpha_val = 1.0;
    real_t q_flux_val = 373.15;
    real_t zero = 0.0;
    real_t fixed_temp = 273.15;
    ConstantCoefficient alphaCoef(alpha_val); 
    ConstantCoefficient qFluxCoef(q_flux_val);
    ConstantCoefficient zeroCoef(zero);
    ConstantCoefficient fixedTempCoef(fixed_temp);


    //   6. Define the Grid Function & project Dirichlet Boundary
    //      coefficient
    GridFunction x(fespace);
    x = 0.0;
    x.ProjectBdrCoefficient(fixedTempCoef, ess_tdof_list);


    //   7. Set up the Bileniear Form m(.) and a(.). Here the 
    //      mass integratorsets up the integral over space of 
    //      (u * v), while the diffusion integrator sets up the
    //      integral over space of (alpha*Nabla_u dot Nabla_v).
    BilinearForm m(fespace);
    m.AddDomainIntegrator(new MassIntegrator);
    BilinearForm a(fespace);
    a.AddDomainIntegrator(new DiffusionIntegrator(alphaCoef));


    //   8. Set up linear form b(.) & Neumann Boundary, this 
    //      means that there is no forcing term (exept right)  
    //      and everything enters through that right boundary.
    //      This sets up the surface integral over the boundary
    //      of (-alpha Nabla_u dot n).
    LinearForm b(fespace);
    b = 0.0;
    Array<int> right_bdr(mesh.bdr_attributes.Max());
    right_bdr = 0;
    right_bdr[2-1] = 1;
    b.AddBoundaryIntegrator(new BoundaryLFIntegrator(qFluxCoef),
                            right_bdr);
    

    //   9. Assemble the mass matrix M, diffusion matrix K
    //      and the forcing vector B. Then Form constrained 
    //      linear system for the operator 'a' and rhs 'b'
    m.Assemble();
    a.Assemble();
    b.Assemble();
    SparseMatrix M;
    SparseMatrix K;
    Vector B(b.Size());
    m.FormSystemMatrix(ess_tdof_list, M);
    a.FormSystemMatrix(ess_tdof_list, K);

    OperatorPtr A;
    Vector X;
    a.FormLinearSystem(ess_tdof_list, x, b, A, X, B);
    

    //  10. Currently, out solution vector x takes the form:
    //      
    //                     Mx - Kx = B 
    //      
    //      Where afer Spacial Discretizaation, we can want to 
    //      write our equation as:
    //       
    //                    dx/dt = M^-1(B-Kx)
    // 
    //      So, we need to tell MFEM the right hand side (RHS)
    //      varries with time, then we then get to compute dx/dt 
    //      for the Backwards Euler implementation. We use the 
    //      CGSolver to find the inverse of our matrix. We take 
    //      three matricies M, K, and B as inputs and initalise 
    //      the parent class with M_.Height(). We actually 
    //      conpute dx/dt with the Mult() member.
    class TimeDependence : public mfem::TimeDependentOperator
    {
    private:
        SparseMatrix &M;
        SparseMatrix &K;
        Vector &B;

        CGSolver M_solver;
        CGSolver T_solver;

        mutable Vector RHS; 
        // Operator that applies (M + gamma*K) to a vector without forming the matrix
        struct SumOperator : public Operator
        {
            SparseMatrix &Mref;
            SparseMatrix &Kref;
            mutable real_t gamma;
            mutable Vector tmp;

            SumOperator(int n, SparseMatrix &M_, SparseMatrix &K_)
                : Operator(n, n), Mref(M_), Kref(K_), gamma(0.0), tmp(n) { }

            virtual void Mult(const Vector &x, Vector &y) const
            {
                Mref.Mult(x, y);
                Kref.Mult(x, tmp);
                y.Add(gamma, tmp);
            }
        };

        SumOperator *sumOp;
    
    public:
        // Constructor - store local version of matrices
             TimeDependence(SparseMatrix &M_, SparseMatrix &K_, Vector &B_)
            : TimeDependentOperator(M_.Height()),
              M(M_), K(K_), B(B_), M_solver(), T_solver(), RHS(M.Height())
        {
            M_solver.SetOperator(M);
                // create sum operator wrapper and set it on the solver (gamma will be set in ImplicitSolve)
                sumOp = new SumOperator(M_.Height(), M, K);
                T_solver.SetOperator(*sumOp);
        }

        // Compute dx/dt = M^-1(B - Kx)
        virtual void Mult(const Vector &x, Vector &dxdt) const
        {
            K.Mult(x, RHS);
            RHS *= -1.0;
            RHS.Add(1.0, B);
            M_solver.Mult(RHS, dxdt);
        }

        // Solve (M + gamma*K) k = B - K*u  --> used by implicit integrators
        virtual void ImplicitSolve(const real_t gamma, const Vector &u, Vector &k)
        {
            // rhs = B - K*u
            K.Mult(u, RHS);
            RHS *= -1.0;
            RHS.Add(1.0, B);

            sumOp->gamma = gamma;
            T_solver.Mult(RHS, k);
        }

        virtual ~TimeDependence() { delete sumOp; }
    };
    

    //  11. Set up Backwards Euler to aproximate the time 
    //      derravitives. The Backwards Euler Method takes  
    //      the form:
    //            y_n+1 = y_n + h * f(x_n+1,y_n+1)
    //
    //      So, now we need to get it to the form:
    //      
    //                  dx/dt = M^-1(B-Kx) 
    //
    //      When we apply Backwards Euler Method we find:
    //
    //               (M - dtK)x^n+1 = Mx^n + dtB 
    //      Then with the while loop we advance t by dt until t_final.
    TimeDependence TD(M, K, B);
    BackwardEulerSolver beSolver;
    real_t t = 0.0;
    real_t t_final = 10.0;
    real_t dt = 1.e-2;
    beSolver.Init(TD);
    while (t < t_final) {
        beSolver.Step(x, t, dt);
    }


    //  13. Get L2 Norms and Energy Norms. Compute the L2 norm by: 
    //      sqrt(x^T M x) and energy norm: sqrt(x^T K x).
    Vector tmp(x.Size());
    Vector Mx(x.Size());
    Vector Kx(x.Size());
    M.Mult(x, Mx);
    K.Mult(x, Kx);
    real_t energy_norm = sqrt(x * Kx);

    //      Norms of RHS B and of Mx, Kx
    real_t B_norm = B.Norml2();
    real_t Mx_norm = Mx.Norml2();
    real_t Kx_norm = Kx.Norml2();

    //      Residual r = Kx - B
    Vector r(x.Size());
    r = Kx;
    r -= B;
    real_t r_l2 = r.Norml2();
    

    cout << "B L2: " << B_norm << ", Mx L2: " << Mx_norm << ", Kx L2: " << Kx_norm << endl;
    cout << "Residual L2: " << r_l2 << endl;

    delete fespace;
    delete fec;
 
    return 0;
}