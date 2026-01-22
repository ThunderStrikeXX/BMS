#include <vector>
#include <array>
#include <cmath>
#include <iostream>
#include <fstream>
#include <iomanip>
#include <filesystem>
#include <omp.h>
#include <stdexcept>
#include <string>

#include "vapor_sodium.h"
#include "tdma.h"

int main() {

    // Constants
    constexpr double M_PI = 3.1415;             // Pi [-]

    // Mass fluxes parameters
    constexpr double sigma_e = 1.0;             // Evaporation accomodation coefficient [-]
    constexpr double sigma_c = 1.0;             // Condensation accomodation coefficient [-]
    double Omega = 1.0;                         // Omega THROHPUT parameter [-]

    // Physical constants
    constexpr double kB = 1.380649e-23;         // Boltzmann [J/K]
    const double d_v = 3.7e-10;                 // Sodium molecular diameter [m]

    // Mesh and geometrical parameters
    const int       N = 20;                     // Number of elements [-]
    const double    L_pipe = 1.0;               // Heat pipe length [m]
    const double    dz = L_pipe / N;            // Element size [m]
    const double    r_v = 1.075e-2;             // Vapor radius [m]
    const double    D_h = 2.0 * r_v;            // Hydraulic diameter [m]
    const double    A_v = M_PI * r_v * r_v;     // Vapor cross section [m2]

    // Picard parameters
    const int       max_picard = 100;           // Maximum Picard iterations [-]
    int             pic = 0;                    // Actual Picard iterations [-]
    int             picard_error = 0.0;         // Picard error vector [kg/m3, m/s, K, Pa]
    double          picard_toler = 1e-6;        // Picard error tolerance [kg/m3, m/s, K, Pa]

    // Time parameters
    const double    dt_user = 1e-6;             // Default timestep [s]
    const double    simulation_time = 1.0;      // Simulation to be executed for this time [s]
    double          dt;                         // Actual time step [s] 
    double          time_total = 0.0;           // Total time elapsed [s]
    int             halves = 0;                 // Number of times the time step has been halved [-]

    // Printing parameters
    double t_last_print = 0.0;                  // Time from last print [s]
    const double print_interval = 1e-6;         // Time interval for printing [s]

    // Temperature and saturation pressure initialization
    std::vector<double> T(N);
    std::vector<double> p_sat(N);

    const double T_left = 500.0;        // [K]
    const double T_right = 300.0;       // [K]

    for (int i = 0; i < N; ++i) {

        const double x = static_cast<double>(i) / (N - 1); // 0 → 1
        T[i] = T_left + x * (T_right - T_left);
        p_sat[i] = vapor_sodium::P_sat(T[i]);
    }

    // Initial conditions
    std::vector<double> rho(N, 9.22e-12);       // Density [kg/m3]
    std::vector<double> p(N, 1e-6);             // Absolute pressure [Pa]

    // Physical parameters
    const double R_vapor = 361.5;               // Sodium vapor constant [J/kgK] 
    std::vector<double> G_mol(N);               // Molecular conductance [Pa/m]  
    std::vector<double> G_visc(N);              // Viscous conductance [Pa/m] 
    std::vector<double> G_eff(N);               // Effective conductance [Pa/m] 
    std::vector<double> Kn(N);                  // Knudsen number [-]
    const double alpha = 5.0;                   // Viscous conductance correction [-]

    // Old vectors
    std::vector<double> rho_old = rho;
    std::vector<double> p_old = p;

    // Iter vectors
    std::vector<double> rho_iter(N);
    std::vector<double> p_iter(N);

    // TDMA vectors
    std::vector<double> a(N, 0.0);
    std::vector<double> b(N, 0.0);
    std::vector<double> c(N, 0.0);
    std::vector<double> d(N, 0.0);

    // Sources vector
    std::vector<double> Sm(N, 0.0);

    // Create result folder
    int new_case = 0;
    std::string name = "case_0";
    while (true) {
        name = "case_" + std::to_string(new_case);
        if (!std::filesystem::exists(name)) {
            std::filesystem::create_directory(name);
            break;
        }
        new_case++;
    }

    // Output
    std::ofstream mesh_out(name + "/mesh.txt", std::ios::trunc);
    std::ofstream time_out(name + "/time.txt", std::ios::trunc);

    std::ofstream rho_out(name + "/rho.txt", std::ios::trunc);
    std::ofstream p_out(name + "/p.txt", std::ios::trunc);
    std::ofstream T_out(name + "/T.txt", std::ios::trunc);

    const int prec = 6;

    mesh_out << std::setprecision(prec);
    time_out << std::setprecision(prec);

    rho_out << std::setprecision(prec);
    p_out << std::setprecision(prec);
    T_out << std::setprecision(prec);

    // Print mesh cell centers
    for (int i = 0; i < N; ++i) mesh_out << (i + 0.5) * dz << ", ";
    mesh_out.flush();
    mesh_out.close();

    // Print number of working threads
    std::cout << "Threads: " << omp_get_max_threads() << "\n";

    // Start computational time clock
    double start = omp_get_wtime();

    // TDMA solver
    tdma::Solver tdma_solver(N);

    // Time-stepping loop
    while (time_total < simulation_time) {

        // Check timestep validity
        if (dt_user < 1e-10) std::cout << "Simulation did not converge. ";

        // Timestep halving
        dt = dt_user * std::ldexp(1.0, -halves);

        // Picard iteration loop
        for (pic = 0; pic < max_picard; ++pic) {

            // Iter = new
            rho_iter = rho;
            p_iter = p;

            for (int i = 0; i < N; ++i) {

                // Updating temperature from interface HERE

                // Updating saturation pressure
                p_sat[i] = vapor_sodium::P_sat(T[i]);

                // Updating molecular conductance
                G_mol[i] =
                    A_v * std::sqrt(R_vapor * T[i] / (2.0 * M_PI)); // [kgm/sPa]

                // Updating dynamic viscosity
                const double mu = vapor_sodium::mu(T[i]);           // [Pa s]

                // Updating viscous conductance
                const double G_visc_0 =
                    (M_PI * std::pow(r_v, 4)) /
                    (8.0 * mu) * rho[i];                            // base Poiseuille

                // Slip / rarefaction correction
                G_visc[i] = G_visc_0 * (1.0 + alpha * Kn[i]);

                // ---------------- Bosanquet -----------------
                G_eff[i] =
                    1.0 / (1.0 / G_visc[i] + 1.0 / G_mol[i]);
            }

            // Assembly of the matrices
            // Parallelizing here does not save time, even with 1000 nodes
            for (int i = 1; i < N - 1; ++i) {

                // Updating mean free path
                const double lambda =
                    kB * T[i] /
                    (std::sqrt(2.0) * M_PI * d_v * d_v * p[i]);     // [m]

                // Updating Knudsen
                Kn[i] = lambda / D_h;

                // Updating source
                const double beta =
                    1.0 / std::sqrt(2.0 * M_PI * R_vapor * T[i]);  // [s/m]

                const double j_m =
                    beta * (sigma_e * p_sat[i] - sigma_c * Omega * p[i]); // [kg/m2 s]

                // ------------------ Volumetric source -------------
                double S = j_m * (2.0 / r_v);  // [kg/m3 s]

                // Cannot condensate more mass than present in the cell
                if (S < 0.0) {
                    const double S_min = -rho[i] / dt;  // [kg/m3 s]
                    if (S < S_min) S = S_min;
                }

                Sm[i] = S;

                // ------------------ Coefficienti su facce ------------------
                const double G_L = 0.5 * (G_eff[i - 1] + G_eff[i]);     // i-1/2
                const double G_R = 0.5 * (G_eff[i] + G_eff[i + 1]);     // i+1/2

                // ------------------ Termini temporali ----------------------
                const double invRTdt = 1.0 / (R_vapor * T[i] * dt);

                // ------------------ TDMA coefficients ----------------------
                a[i] = -G_L / (A_v * dz * dz);

                b[i] = invRTdt
                    + (G_L + G_R) / (A_v * dz * dz);

                c[i] = -G_R / (A_v * dz * dz);

                d[i] = p[i] * invRTdt
                    + Sm[i];

            }

            b[0] = 1.0;
            c[0] = -1.0;

            b[N - 1] = 1.0;
            a[N - 1] = -1.0;

            tdma_solver.solve(a, b, c, d, p);

            for (int i = 0; i < N; ++i) {

                rho[i] = p[i] / (R_vapor * T[i]);
            }

            double Aold, Anew, denom, eps;

            picard_error = 0.0;

            for (int i = 0; i < N; ++i) {

                Aold = p_iter[i];
                Anew = p[i];
                denom = 0.5 * (std::fabs(Aold) + std::fabs(Anew));
                eps = denom > 1e-12 ? std::fabs((Anew - Aold) / denom) : std::fabs(Anew - Aold);
                picard_error += eps;
            }

            picard_error /= N;

            if (picard_error < picard_toler) {

                halves = 0;             // Reset halves if Picard converged
                break;                  // Picard converged
            }
        }

        // Picard converged or max iterations reached
        if (pic != max_picard) {

            // Update n values (old = new) and update total time
            for (int i = 0; i < N; ++i) {
                rho_old[i] = rho[i];
                p_old[i] = p[i];
            }

            time_total += dt;
        }
        else {

            // Rollback to previous time step (new = old) and halve dt
            for (int i = 0; i < N; ++i) {

                rho[i] = rho_old[i];
                p[i] = p_old[i];
            }

            halves += 1;
        }

        if (time_total >= t_last_print + print_interval) {

            for (int i = 0; i < N; ++i) {
                rho_out << rho[i] << ", ";
                p_out << p[i] << ", ";
            }

            time_out << time_total << ", ";

            rho_out << "\n";
            p_out << "\n";

            rho_out.flush();
            p_out.flush();

            time_out.flush();

            t_last_print += print_interval;
        }
    }

    time_out.close();

    rho_out.close();
    p_out.close();

    double end = omp_get_wtime();

    std::cout << "Execution time: " << end - start;

    system("pause");

    return 0;
}