#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <mpi.h>

using namespace std;

const int nx = 41;
const int ny = 41;
const int nt = 500;
const int nit = 50;
const double dx = 2. / (nx - 1);
const double dy = 2. / (ny - 1);
const double dt = .01;
const double rho = 1.;
const double nu = .02;

void divide_length(int size, int rank, int* begin, int* end) {
    int llen = (ny + size - 1) / size;
    int b = llen * rank;
    int e = llen * (rank + 1);

    if (b > ny) b = ny;
    if (e > ny) e = ny;

    *begin = b;
    *end = e;
}

float compute_b_entry(float* u, float* v, float* tu, float* tv, float* bu, float* bv, int i, int j, int begin, int end) {
    float uu, ud, ur, ul;
    float vu, vd, vr, vl;
    if (j == begin) {
        uu = tu[i];
        vu = tv[i];
        ud = u[(j + 1) * nx + i];
        vd = v[(j + 1) * nx + i];
    }
    else if (j == end - 1) {
        uu = u[(j - 1) * nx + i];
        vu = v[(j - 1) * nx + i];
        ud = bu[i];
        vd = bv[i];
    }
    else {
        uu = u[(j - 1) * nx + i];
        vu = v[(j - 1) * nx + i];
        ud = u[(j + 1) * nx + i];
        vd = v[(j + 1) * nx + i];
    }
    ur = u[j * nx + i + 1];
    ul = u[j * nx + i - 1];
    vr = v[j * nx + i + 1];
    vl = v[j * nx + i - 1];
    float b1 = 1. / dt * ((ur - ul) / (2. * dx) + (vd - vu) / (2. * dy));
    float b2 = (ur - ul) * (ur - ul) / (4. * dx * dx);
    float b3 = 2. * (ud - uu) / (2. * dy) * (vr - vl) / (2. * dx);
    float b4 = (vd - vu) * (vd - vu) / (4. * dy * dy);
    return rho * dx * dx * dy * dy * (b1 - b2 - b3 - b4);
}

float compute_p_entry(float* pn, float* b, float* tpn, float* bpn, int i, int j, int begin, int end) {
    float pnu, pnd, pnr, pnl;
    if (j == begin) {
        pnu = tpn[i];
        pnd = pn[(j + 1) * nx + i];
    }
    else if (j == end - 1) {
        pnu = pn[(j - 1) * nx + i];
        pnd = bpn[i];
    }
    else {
        pnu = pn[(j - 1) * nx + i];
        pnd = pn[(j + 1) * nx + i];
    }
    pnr = pn[j * nx + i + 1];
    pnl = pn[j * nx + i - 1];
    float p1 = ((pnr + pnl) * dy * dy + (pnd + pnu) * dx * dx);
    float p2 = b[j * nx + i];
    return (p1 - p2) / (2. * (dx * dx + dy * dy));
}

float compute_u_entry(float* un, float* vn, float* p, float* tun, float* bun, int i, int j, int begin, int end) {
    float unc, unu, und, unr, unl;
    float vnc;
    float pr, pl;
    if (j == begin) {
        unu = tun[i];
        und = un[(j + 1) * nx + i];
    }
    else if (j == end - 1) {
        unu = un[(j - 1) * nx + i];
        und = bun[i];
    }
    else {
        unu = un[(j - 1) * nx + i];
        und = un[(j + 1) * nx + i];
    }
    unc = un[j * nx + i];
    unr = un[j * nx + i + 1];
    unl = un[j * nx + i - 1];
    vnc = vn[j * nx + i];
    pr = p[j * nx + i + 1];
    pl = p[j * nx + i - 1];
    float u1 = unc * dt / dx * (unc - unl);
    float u2 = vnc * dt / dy * (unc - unu);
    float u3 = dt / (2. * rho * dx) * (pr - pl);
    float u4 = dt / (dx * dx) * (unr - 2. * unc + unl);
    float u5 = dt / (dy * dy) * (und - 2. * unc + unu);
    return unc - u1 - u2 - u3 + nu * (u4 + u5);
}

float compute_v_entry(float* un, float* vn, float*p, float* tvn, float* bvn, float* tp, float* bp, int i, int j, int begin, int end) {
    float vnc, vnu, vnd, vnr, vnl;
    float unc;
    float pu, pd;
    if (j == begin) {
        vnu = tvn[i];
        vnd = vn[(j + 1) * nx + i];
        pu = tp[i];
        pd = p[(j + 1) * nx + i];
    }
    else if (j == end - 1) {
        vnu = vn[(j - 1) * nx + i];
        vnd = bvn[i];
        pu = p[(j - 1) * nx + i];
        pd = bp[i];
    }
    else {
        vnu = vn[(j - 1) * nx + i];
        vnd = vn[(j + 1) * nx + i];
        pu = p[(j - 1) * nx + i];
        pd = p[(j + 1) * nx + i];
    }
    vnc = vn[j * nx + i];
    vnr = vn[j * nx + i + 1];
    vnl = vn[j * nx + i - 1];
    unc = un[j * nx + i];
    float v1 = unc * dt / dx * (vnc - vnl);
    float v2 = vnc * dt / dy * (vnc - vnu);
    float v3 = dt / (2. * rho * dy) * (pd - pu);
    float v4 = dt / (dx * dx) * (vnr - 2. * vnc + vnl);
    float v5 = dt / (dy * dy) * (vnd - 2. * vnc + vnu);
    return vnc - v1 - v2 - v3 + nu * (v4 + v5);
}

int main(int argc, char** argv) {
    int size, rank;
    int begin, end;

    float u[ny * nx];
    float v[ny * nx];
    float p[ny * nx];
    float b[ny * nx];
    float un[ny * nx];
    float vn[ny * nx];
    float pn[ny * nx];
    float u_res[ny * nx];
    float v_res[ny * nx];
    float p_res[ny * nx];

    MPI_Init(&argc, &argv);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    divide_length(size, rank, &begin, &end);

    for (int j = begin; j < end; j++) {
        for (int i = 0; i < nx; i++) {
            u[j * nx + i] = 0;
            v[j * nx + i] = 0;
            p[j * nx + i] = 0;
            b[j * nx + i] = 0;
        }
    }

    ofstream ufile("u.dat");
    ofstream vfile("v.dat");
    ofstream pfile("p.dat");

    float top_u_buf[nx], bottom_u_buf[nx];
    float top_v_buf[nx], bottom_v_buf[nx];
    float top_p_buf[nx], bottom_p_buf[nx];
    float top_un_buf[nx], bottom_un_buf[nx];
    float top_vn_buf[nx], bottom_vn_buf[nx];
    float top_pn_buf[nx], bottom_pn_buf[nx];
    MPI_Request top_u_req, bottom_u_req;
    MPI_Request top_v_req, bottom_v_req;
    MPI_Request top_p_req, bottom_p_req;
    MPI_Request top_un_req, bottom_un_req;
    MPI_Request top_vn_req, bottom_vn_req;
    MPI_Request top_pn_req, bottom_pn_req;
    MPI_Status stat;
    for (int n = 0; n < nt; n++) {
        // Communication for computation of b
        if (rank > 0) {
            MPI_Irecv(top_u_buf, nx, MPI_FLOAT, rank - 1, 1, MPI_COMM_WORLD, &top_u_req);
            MPI_Irecv(top_v_buf, nx, MPI_FLOAT, rank - 1, 2, MPI_COMM_WORLD, &top_v_req);
        }
        if (rank < size - 1) {
            MPI_Irecv(bottom_u_buf, nx, MPI_FLOAT, rank + 1, 1, MPI_COMM_WORLD, &bottom_u_req);
            MPI_Irecv(bottom_v_buf, nx, MPI_FLOAT, rank + 1, 2, MPI_COMM_WORLD, &bottom_v_req);
        }
        if (rank > 0) {
            MPI_Send(&u[begin * nx], nx, MPI_FLOAT, rank - 1, 1, MPI_COMM_WORLD);
            MPI_Send(&v[begin * nx], nx, MPI_FLOAT, rank - 1, 2, MPI_COMM_WORLD);
        }
        if (rank < size - 1) {
            MPI_Send(&u[(end - 1) * nx], nx, MPI_FLOAT, rank + 1, 1, MPI_COMM_WORLD);
            MPI_Send(&v[(end - 1) * nx], nx, MPI_FLOAT, rank + 1, 2, MPI_COMM_WORLD);
        }
        if (rank > 0) {
            MPI_Wait(&top_u_req, &stat);
            MPI_Wait(&top_v_req, &stat);
        }
        if (rank < size - 1) {
            MPI_Wait(&bottom_u_req, &stat);
            MPI_Wait(&bottom_v_req, &stat);
        }

        // Compute b
        if (rank > 0)
            for (int i = 1; i < nx - 1; i++)
                b[begin * nx + i] = compute_b_entry(u, v, top_u_buf, top_v_buf, bottom_u_buf, bottom_v_buf, i, begin, begin, end);
        if (rank < size - 1)
            for (int i = 1; i < nx - 1; i++)
                b[end * nx + i] = compute_b_entry(u, v, top_u_buf, top_v_buf, bottom_u_buf, bottom_v_buf, i, end - 1, begin, end);
        for (int j = begin + 1; j < end - 1; j++)
            for (int i = 1; i < nx - 1; i++)
                b[j * nx + i] = compute_b_entry(u, v, top_u_buf, top_v_buf, bottom_u_buf, bottom_v_buf, i, j, begin, end);
        
        for (int it = 0; it < nit; it++) {
            for (int j = begin; j < end; j++)
                for (int i = 0; i < nx; i++)
                    pn[j * nx + i] = p[j * nx + i];
            
            // Communication for computation of p
            if (rank > 0) {
                MPI_Irecv(top_pn_buf, nx, MPI_FLOAT, rank - 1, 1, MPI_COMM_WORLD, &top_pn_req);
            }
            if (rank < size - 1) {
                MPI_Irecv(bottom_pn_buf, nx, MPI_FLOAT, rank + 1, 1, MPI_COMM_WORLD, &bottom_pn_req);
            }
            if (rank > 0) {
                MPI_Send(&pn[begin * nx], nx, MPI_FLOAT, rank - 1, 1, MPI_COMM_WORLD);
            }
            if (rank < size - 1) {
                MPI_Send(&pn[(end - 1) * nx], nx, MPI_FLOAT, rank + 1, 1, MPI_COMM_WORLD);
            }
            if (rank > 0) {
                MPI_Wait(&top_pn_req, &stat);
            }
            if (rank < size - 1) {
                MPI_Wait(&bottom_pn_req, &stat);
            }

            // Compute p
            if (rank > 0)
                for (int i = 1; i < nx - 1; i++)
                    p[begin * nx + i] = compute_p_entry(pn, b, top_pn_buf, bottom_pn_buf, i, begin, begin, end);
            if (rank < size - 1)
                for (int i = 1; i < nx - 1; i++) 
                    p[(end - 1) * nx + i] = compute_p_entry(pn, b, top_pn_buf, bottom_pn_buf, i, end - 1, begin, end);
            for (int j = begin + 1; j < end - 1; j++) 
                for (int i = 1; i < nx - 1; i++) 
                    p[j * nx + i] = compute_p_entry(pn, b, top_pn_buf, bottom_pn_buf, i, j, begin, end);
            // Compute boundary of p
            for (int j = begin; j < end; j++) {
                p[j * nx] = p[j * nx + 1];
                p[(j + 1) * nx - 1] = p[(j + 1) * nx - 2];
            }
            if (rank == 0)
                for (int i = 0; i < nx; i++)
                    p[i] = p[nx + i];
            if (rank == end - 1)
                for (int i = 0; i < nx; i++) 
                    p[(ny - 1) * nx + i] = 0;    
        }

        for (int j = begin; j < end; j++) {
            for (int i = 0; i < nx; i++) {
                un[j * nx + i] = u[j * nx + i];
                vn[j * nx + i] = v[j * nx + i];
            }
        }

        // Communication for computation of u and v
        if (rank > 0) {
            MPI_Irecv(top_p_buf, nx, MPI_FLOAT, rank - 1, 1, MPI_COMM_WORLD, &top_p_req);
            MPI_Irecv(top_un_buf, nx, MPI_FLOAT, rank - 1, 2, MPI_COMM_WORLD, &top_un_req);
            MPI_Irecv(top_vn_buf, nx, MPI_FLOAT, rank - 1, 3, MPI_COMM_WORLD, &top_vn_req);
        }
        if (rank < size - 1) {
            MPI_Irecv(bottom_p_buf, nx, MPI_FLOAT, rank + 1, 1, MPI_COMM_WORLD, &bottom_p_req);
            MPI_Irecv(bottom_un_buf, nx, MPI_FLOAT, rank + 1, 2, MPI_COMM_WORLD, &bottom_un_req);
            MPI_Irecv(bottom_vn_buf, nx, MPI_FLOAT, rank + 1, 3, MPI_COMM_WORLD, &bottom_vn_req);
        }
        if (rank > 0) {
            MPI_Send(&p[begin * nx], nx, MPI_FLOAT, rank - 1, 1, MPI_COMM_WORLD);
            MPI_Send(&un[begin * nx], nx, MPI_FLOAT, rank - 1, 2, MPI_COMM_WORLD);
            MPI_Send(&vn[begin * nx], nx, MPI_FLOAT, rank - 1, 3, MPI_COMM_WORLD);
        }
        if (rank < size - 1) {
            MPI_Send(&p[(end - 1) * nx], nx, MPI_FLOAT, rank + 1, 1, MPI_COMM_WORLD);
            MPI_Send(&un[(end - 1) * nx], nx, MPI_FLOAT, rank + 1, 2, MPI_COMM_WORLD);
            MPI_Send(&vn[(end - 1) * nx], nx, MPI_FLOAT, rank + 1, 3, MPI_COMM_WORLD);
        }
        if (rank > 0) {
            MPI_Wait(&top_p_req, &stat);
            MPI_Wait(&top_un_req, &stat);
            MPI_Wait(&top_vn_req, &stat);
        }
        if (rank < size - 1) {
            MPI_Wait(&bottom_p_req, &stat);
            MPI_Wait(&bottom_un_req, &stat);
            MPI_Wait(&bottom_vn_req, &stat);
        }

        // Compute u and v
        if (rank > 0) {
            for (int i = 1; i < nx - 1; i++) {
                u[begin * nx + i] = compute_u_entry(un, vn, p, top_un_buf, bottom_un_buf, i, begin, begin, end);
                v[begin * nx + i] = compute_v_entry(un, vn, p, top_vn_buf, bottom_vn_buf, top_p_buf, bottom_p_buf, i, begin, begin, end);
            }
        }
        if (rank < size - 1) {
            for (int i = 1; i < nx - 1; i++) {
                u[(end - 1) * nx + i] = compute_u_entry(un, vn, p, top_un_buf, bottom_un_buf, i, end - 1, begin, end);
                v[(end - 1) * nx + i] = compute_v_entry(un, vn, p, top_vn_buf, bottom_vn_buf, top_p_buf, bottom_p_buf, i, end - 1, begin, end);
            }
        }
        for (int j = begin + 1; j < end - 1; j++) {
            for (int i = 1; i < nx - 1; i++) {
                u[j * nx + i] = compute_u_entry(un, vn, p, top_un_buf, bottom_un_buf, i, j, begin, end);
                v[j * nx + i] = compute_v_entry(un ,vn, p, top_vn_buf, bottom_vn_buf, top_p_buf, bottom_p_buf, i, j, begin, end);
            }
        }

        // Compute boundary of u and v
        for (int j = begin; j < end; j++) {
            u[j * nx] = 0;
            u[(j + 1) * nx - 1] = 0;
            v[j * nx] = 0;
            v[(j + 1) * nx - 1] = 0;
        }
        if (rank == 0) {
            for (int i = 0; i < nx; i++) {
                u[i] = 0;
                v[i] = 0;
            }
        }
        if (rank == size - 1) {
            for (int i = 0; i < nx; i++) {
                u[(ny - 1) * nx + i] = 1.;
                v[(ny - 1) * nx + i] = 0;
            }
        }

        if (n % 10 == 0) {
            MPI_Gather(&u[begin * nx], (end - begin) * nx, MPI_FLOAT, u_res, (end - begin) * nx, MPI_FLOAT, 0, MPI_COMM_WORLD);
            MPI_Gather(&v[begin * nx], (end - begin) * nx, MPI_FLOAT, v_res, (end - begin) * nx, MPI_FLOAT, 0, MPI_COMM_WORLD);
            MPI_Gather(&p[begin * nx], (end - begin) * nx, MPI_FLOAT, p_res, (end - begin) * nx, MPI_FLOAT, 0, MPI_COMM_WORLD);

            if (rank == 0) {
                for (int i = 0; i < nx * ny; i++) {
                    ufile << u_res[i] << " ";
                    vfile << v_res[i] << " ";
                    pfile << p_res[i] << " ";
                }
                ufile << endl;
                vfile << endl;
                pfile << endl;
            }
        }
    }

    ufile.close();
    vfile.close();
    pfile.close();
    
    MPI_Finalize();
}
