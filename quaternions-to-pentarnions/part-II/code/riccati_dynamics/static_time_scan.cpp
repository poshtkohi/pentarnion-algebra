// Reproduces the static gamma scan, the two time-dependent omega scans,
// and representative trajectory CSV files used in the report.
// Compile: g++ -O3 -march=native -fopenmp static_time_scan.cpp -o static_time_scan
// Run:     ./static_time_scan ../data

#include <filesystem>
#include <stdexcept>
#include <cstdlib>
#include <algorithm>
#include <array>
#include <cmath>
#include <cstdio>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <string>
#include <vector>
#ifdef _OPENMP
#include <omp.h>
#endif

using Vec = std::array<double, 5>;

inline Vec add(const Vec& a, const Vec& b) {
    Vec c{};
    for (int i = 0; i < 5; ++i) c[i] = a[i] + b[i];
    return c;
}
inline Vec sub(const Vec& a, const Vec& b) {
    Vec c{};
    for (int i = 0; i < 5; ++i) c[i] = a[i] - b[i];
    return c;
}
inline Vec scale(const Vec& a, double s) {
    Vec c{};
    for (int i = 0; i < 5; ++i) c[i] = s * a[i];
    return c;
}
inline double normv(const Vec& a) {
    double s = 0.0;
    for (double x : a) s += x * x;
    return std::sqrt(s);
}
inline bool finitev(const Vec& a) {
    for (double x : a) if (!std::isfinite(x)) return false;
    return true;
}

inline Vec product(const Vec& a, const Vec& b, double gamma) {
    const double kappa = 1.0 / std::sqrt(1.0 + gamma * gamma);
    const double sigma = gamma * kappa;
    const double inv_sqrt2 = 1.0 / std::sqrt(2.0);

    Vec q{0, 0, 0, 0, 0};
    q[0] = a[0] * b[0] - a[1] * b[1] - a[2] * b[2]
         - a[3] * b[3] - a[4] * b[4];
    q[1] = a[0] * b[1] + a[1] * b[0];
    q[2] = a[0] * b[2] + a[2] * b[0];
    q[3] = a[0] * b[3] + a[3] * b[0];
    q[4] = a[0] * b[4] + a[4] * b[0];

    double s = a[1] * b[2] - a[2] * b[1];
    q[3] += kappa * s; q[4] += sigma * s;
    s = a[2] * b[3] - a[3] * b[2];
    q[1] += kappa * s; q[4] += sigma * s;
    s = a[3] * b[1] - a[1] * b[3];
    q[2] += kappa * s; q[4] += sigma * s;

    s = a[4] * b[1] - a[1] * b[4];
    q[2] += inv_sqrt2 * s; q[3] += inv_sqrt2 * s;
    s = a[4] * b[2] - a[2] * b[4];
    q[3] += inv_sqrt2 * s; q[1] += inv_sqrt2 * s;
    s = a[4] * b[3] - a[3] * b[4];
    q[1] += inv_sqrt2 * s; q[2] += inv_sqrt2 * s;
    return q;
}

inline Vec static_field(const Vec& p, double gamma) {
    const Vec A{-0.5, -2.0, 0.5, -1.0, -0.5 * gamma};
    const Vec B{1.0, 0.0, 2.0, 0.0, -0.5 * gamma};
    return add(product(A, p, gamma), product(B, product(p, p, gamma), gamma));
}

template <class RHS>
inline Vec rk4(const Vec& y, double t, double h, const RHS& rhs) {
    const Vec k1 = rhs(t, y);
    const Vec k2 = rhs(t + 0.5 * h, add(y, scale(k1, 0.5 * h)));
    const Vec k3 = rhs(t + 0.5 * h, add(y, scale(k2, 0.5 * h)));
    const Vec k4 = rhs(t + h, add(y, scale(k3, h)));
    Vec out{};
    for (int i = 0; i < 5; ++i)
        out[i] = y[i] + h * (k1[i] + 2 * k2[i] + 2 * k3[i] + k4[i]) / 6.0;
    return out;
}

struct Result {
    double parameter = 0, max_radius = 0, tail_mean = 0, tail_std = 0,
           tail_min = 0, tail_max = 0, final_radius = 0, lle = 0,
           peak_interval_mean = 0, peak_interval_cv = 0;
    int n_peaks = 0, blowup = 0;
};

Result scan_static(double gamma, double T = 800.0, double dt = 0.01,
                   double transient = 300.0) {
    Vec p{0.5, -0.5, -0.5, 0.5, -0.5 * gamma};
    Vec direction{0.37, -0.51, 0.19, 0.62, -0.43};
    const double dn = normv(direction);
    for (double& x : direction) x /= dn;
    constexpr double delta = 1e-8;
    Vec q = add(p, scale(direction, delta));
    auto rhs = [gamma](double, const Vec& y) { return static_field(y, gamma); };

    const long N = std::llround(T / dt);
    const long trans = std::llround(transient / dt);
    const int renorm_steps = 10;
    double sum_log = 0.0;
    long n_log = 0;
    double max_radius = normv(p), sum = 0.0, sum2 = 0.0;
    double tail_min = 1e300, tail_max = 0.0;
    long n_tail = 0;
    std::vector<double> peak_times;
    double z2 = p[3], z1 = p[3];
    int blowup = 0;

    for (long n = 1; n <= N; ++n) {
        const double t = (n - 1) * dt;
        p = rk4(p, t, dt, rhs);
        q = rk4(q, t, dt, rhs);
        const double radius = normv(p);
        if (!finitev(p) || radius > 1e10) { blowup = 1; break; }
        max_radius = std::max(max_radius, radius);
        if (n >= trans) {
            sum += radius; sum2 += radius * radius;
            tail_min = std::min(tail_min, radius);
            tail_max = std::max(tail_max, radius);
            ++n_tail;
        }
        if (n > trans + 2) {
            const double z = p[3];
            if (z1 > z2 && z1 > z) peak_times.push_back((n - 1) * dt);
            z2 = z1; z1 = z;
        } else {
            z2 = z1; z1 = p[3];
        }
        if (n % renorm_steps == 0) {
            Vec dv = sub(q, p);
            double dist = normv(dv);
            if (dist == 0.0 || !std::isfinite(dist)) dist = delta;
            if (n >= trans) { sum_log += std::log(dist / delta); ++n_log; }
            q = add(p, scale(dv, delta / dist));
        }
    }

    const double mean = n_tail ? sum / n_tail : NAN;
    const double variance = n_tail ? std::max(0.0, sum2 / n_tail - mean * mean) : NAN;
    double peak_mean = NAN, peak_cv = NAN;
    if (peak_times.size() > 3) {
        double s = 0.0, s2 = 0.0; int m = 0;
        for (size_t i = 1; i < peak_times.size(); ++i) {
            const double d = peak_times[i] - peak_times[i - 1];
            s += d; s2 += d * d; ++m;
        }
        peak_mean = s / m;
        peak_cv = std::sqrt(std::max(0.0, s2 / m - peak_mean * peak_mean)) / peak_mean;
    }
    return Result{gamma, max_radius, mean, std::sqrt(variance), tail_min,
                  tail_max, normv(p), n_log ? sum_log / (n_log * renorm_steps * dt) : NAN,
                  peak_mean, peak_cv, static_cast<int>(peak_times.size()), blowup};
}

struct Protocol { double center, amplitude; };
inline void protocol_value(double t, double omega, const Protocol& P,
                           double& gamma, double& gamma_dot) {
    gamma = P.center + P.amplitude * std::sin(omega * t);
    gamma_dot = P.amplitude * omega * std::cos(omega * t);
}
inline Vec hidden_field(const Vec& q, double t, double omega, const Protocol& P) {
    double gamma, gamma_dot;
    protocol_value(t, omega, P, gamma, gamma_dot);
    Vec p{q[0], q[1], q[2], q[3], gamma * q[4]};
    const Vec fp = static_field(p, gamma);
    return Vec{fp[0], fp[1], fp[2], fp[3], (fp[4] - gamma_dot * q[4]) / gamma};
}

Result scan_time(double omega, const Protocol& P, double T = 1200.0,
                 double dt = 0.005, double transient = 400.0) {
    Vec qstate{0.5, -0.5, -0.5, 0.5, -0.5};
    Vec direction{0.37, -0.51, 0.19, 0.62, -0.43};
    const double dn = normv(direction);
    for (double& x : direction) x /= dn;
    constexpr double delta = 1e-8;
    Vec qpert = add(qstate, scale(direction, delta));
    auto rhs = [omega, P](double t, const Vec& y) { return hidden_field(y, t, omega, P); };

    const long N = std::llround(T / dt);
    const long trans = std::llround(transient / dt);
    const int renorm_steps = 20;
    double sum_log = 0.0; long n_log = 0;
    double max_radius = 0.0, sum = 0.0, sum2 = 0.0;
    double tail_min = 1e300, tail_max = 0.0;
    long n_tail = 0;
    std::vector<double> peak_times;
    double z2 = qstate[3], z1 = qstate[3];
    int blowup = 0;

    for (long n = 1; n <= N; ++n) {
        const double t = (n - 1) * dt;
        qstate = rk4(qstate, t, dt, rhs);
        qpert = rk4(qpert, t, dt, rhs);
        double gamma, gamma_dot;
        protocol_value(n * dt, omega, P, gamma, gamma_dot);
        Vec actual{qstate[0], qstate[1], qstate[2], qstate[3], gamma * qstate[4]};
        const double radius = normv(actual);
        if (!finitev(qstate) || radius > 1e10) { blowup = 1; break; }
        max_radius = std::max(max_radius, radius);
        if (n >= trans) {
            sum += radius; sum2 += radius * radius;
            tail_min = std::min(tail_min, radius);
            tail_max = std::max(tail_max, radius);
            ++n_tail;
        }
        if (n > trans + 2) {
            const double z = qstate[3];
            if (z1 > z2 && z1 > z) peak_times.push_back((n - 1) * dt);
            z2 = z1; z1 = z;
        } else {
            z2 = z1; z1 = qstate[3];
        }
        if (n % renorm_steps == 0) {
            Vec dv = sub(qpert, qstate);
            double dist = normv(dv);
            if (dist == 0.0 || !std::isfinite(dist)) dist = delta;
            if (n >= trans) { sum_log += std::log(dist / delta); ++n_log; }
            qpert = add(qstate, scale(dv, delta / dist));
        }
    }

    const double mean = n_tail ? sum / n_tail : NAN;
    const double variance = n_tail ? std::max(0.0, sum2 / n_tail - mean * mean) : NAN;
    double peak_mean = NAN, peak_cv = NAN;
    if (peak_times.size() > 3) {
        double s = 0.0, s2 = 0.0; int m = 0;
        for (size_t i = 1; i < peak_times.size(); ++i) {
            const double d = peak_times[i] - peak_times[i - 1];
            s += d; s2 += d * d; ++m;
        }
        peak_mean = s / m;
        peak_cv = std::sqrt(std::max(0.0, s2 / m - peak_mean * peak_mean)) / peak_mean;
    }
    double gamma, gamma_dot;
    protocol_value(T, omega, P, gamma, gamma_dot);
    Vec actual{qstate[0], qstate[1], qstate[2], qstate[3], gamma * qstate[4]};
    return Result{omega, max_radius, mean, std::sqrt(variance), tail_min,
                  tail_max, normv(actual), n_log ? sum_log / (n_log * renorm_steps * dt) : NAN,
                  peak_mean, peak_cv, static_cast<int>(peak_times.size()), blowup};
}

void write_results(const std::string& path, const std::vector<Result>& rows,
                   const std::string& parameter_name) {
    std::filesystem::create_directories(std::filesystem::path(path).parent_path());
    std::ofstream f(path); if(!f) throw std::runtime_error("Cannot open output: "+path);
    if (!f) throw std::runtime_error("cannot open " + path);
    f << parameter_name << ",max_radius,tail_mean,tail_std,tail_min,tail_max,final_radius,"
      << "lle,peak_interval_mean,peak_interval_cv,n_peaks,blowup\n";
    f << std::setprecision(12);
    for (const auto& r : rows) {
        f << r.parameter << ',' << r.max_radius << ',' << r.tail_mean << ',' << r.tail_std << ','
          << r.tail_min << ',' << r.tail_max << ',' << r.final_radius << ',' << (r.blowup ? NAN : r.lle) << ','
          << r.peak_interval_mean << ',' << r.peak_interval_cv << ',' << r.n_peaks << ','
          << r.blowup << '\n';
    }
}

void save_static_trajectory(const std::string& path, double gamma,
                            double T = 1000.0, double dt = 0.01) {
    Vec p{0.5, -0.5, -0.5, 0.5, -0.5 * gamma};
    auto rhs = [gamma](double, const Vec& y) { return static_field(y, gamma); };
    std::filesystem::create_directories(std::filesystem::path(path).parent_path());
    std::ofstream f(path); if(!f) throw std::runtime_error("Cannot open output: "+path);
    f << "t,w,x,y,z,u,radius\n" << std::setprecision(12);
    const long N = std::llround(T / dt);
    for (long n = 0; n <= N; ++n) {
        if (n % 5 == 0)
            f << n * dt << ',' << p[0] << ',' << p[1] << ',' << p[2] << ',' << p[3]
              << ',' << p[4] << ',' << normv(p) << '\n';
        p = rk4(p, n * dt, dt, rhs);
        if (!finitev(p) || normv(p) > 1e10) break;
    }
}

void save_time_trajectory(const std::string& path, double omega, const Protocol& P,
                          double T = 1000.0, double dt = 0.005) {
    Vec qstate{0.5, -0.5, -0.5, 0.5, -0.5};
    auto rhs = [omega, P](double t, const Vec& y) { return hidden_field(y, t, omega, P); };
    std::filesystem::create_directories(std::filesystem::path(path).parent_path());
    std::ofstream f(path); if(!f) throw std::runtime_error("Cannot open output: "+path);
    f << "t,gamma,w,x,y,z,v_hidden,u_active,radius\n" << std::setprecision(12);
    const long N = std::llround(T / dt);
    for (long n = 0; n <= N; ++n) {
        const double t = n * dt;
        double gamma, gamma_dot;
        protocol_value(t, omega, P, gamma, gamma_dot);
        Vec p{qstate[0], qstate[1], qstate[2], qstate[3], gamma * qstate[4]};
        if (n % 10 == 0)
            f << t << ',' << gamma << ',' << qstate[0] << ',' << qstate[1] << ','
              << qstate[2] << ',' << qstate[3] << ',' << qstate[4] << ',' << p[4]
              << ',' << normv(p) << '\n';
        qstate = rk4(qstate, t, dt, rhs);
        if (!finitev(qstate) || normv(qstate) > 1e12) break;
    }
}

int main(int argc, char** argv) {
    if (argc < 2) {
        std::cerr << "Usage: static_time_scan OUTPUT_DATA_DIRECTORY\n";
        return 2;
    }
    const std::string out = argv[1];
    std::filesystem::create_directories(out + "/static_trajectories");
    std::filesystem::create_directories(out + "/time_trajectories");

    const int N_gamma = 201;
    std::vector<Result> static_rows(N_gamma);
    #pragma omp parallel for schedule(dynamic)
    for (int i = 0; i < N_gamma; ++i)
        static_rows[i] = scan_static(i / static_cast<double>(N_gamma - 1));
    write_results(out + "/static_scan.csv", static_rows, "gamma");

    const std::vector<double> omegas{0.01, 0.02, 0.03, 0.05, 0.075, 0.1, 0.15,
        0.2, 0.3, 0.4, 0.6, 0.8, 1.0, 1.5, 2.0};
    const Protocol transition_band{0.68, 0.08};
    const Protocol regularized_full{0.525, 0.475};
    std::vector<Result> band_rows(omegas.size()), full_rows(omegas.size());
    #pragma omp parallel for schedule(dynamic)
    for (int i = 0; i < static_cast<int>(omegas.size()); ++i) {
        band_rows[i] = scan_time(omegas[i], transition_band);
        full_rows[i] = scan_time(omegas[i], regularized_full);
    }
    write_results(out + "/time_scan_transition_band.csv", band_rows, "omega");
    write_results(out + "/time_scan_full_range_regularized.csv", full_rows, "omega");

    for (double g : {0.4, 0.6, 0.7, 0.8, 1.0}) {
        char name[128];
        std::snprintf(name, sizeof(name), "/static_trajectories/static_gamma_%0.3f.csv", g);
        save_static_trajectory(out + name, g);
    }
    for (double omega : {0.2, 0.8}) {
        char name[128];
        std::snprintf(name, sizeof(name), "/time_trajectories/time_band_omega_%0.3f.csv", omega);
        save_time_trajectory(out + name, omega, transition_band);
        std::snprintf(name, sizeof(name), "/time_trajectories/time_full_omega_%0.3f.csv", omega);
        save_time_trajectory(out + name, omega, regularized_full);
    }
    return 0;
}
