################################################################################
#=

A Semismooth Newton-based Solver for Risk Neutral PDE-Constrained Optimization
with Control Constraints

1. The test problem includes bilateral bounds on the control.
2. The random quantities are treated by a Monte Carlo sampling procedure
3. The deterministic spaces are discretized via P1 FE on a uniform mesh.

This code was used to generate the plots on slides:

1. 44 (l) OptimalControl.pdf           (r) OutOfSampleState.pdf
2. 45 (l) EmpiricalStabilityPDESol.pdf (r) EmpiricalStabilityOptimalVal.pdf

Some function are called once on a small exampl to take advantage of the
JIT compilier

Date: 22 June 2020
Author: Thomas M. Surowiec
Contact: surowiec@mathematik.uni-marburg.de

=#
################################################################################
cd(dirname(@__FILE__()))
using Plots
using SparseArrays
using LinearAlgebra
using StatsBase

################################################################################
# Deterministic Input Parameters
################################################################################
function desired_state(x::Float64)
    return sin(50.0*x/(pi))
end
desired_state(1.0)


################################################################################
# Stochastic Input Parameters
################################################################################
function random_inputs()
    return [10.0^(-1.0 + 2.0*rand() - 2.0),
                 (-1.0 + 2.0*rand())/100.0,
            1.0+(-1.0 + 2.0*rand())/1000.0,
                (1.0 + 2.0*rand())/1000.0]
end
random_inputs()

function u_bar(x::Float64,xi_vec::Vector{Float64})
    return (1.0 - x)*xi_vec[3] +
            xi_vec[4]*x +
            xi_vec[1]^(-1)*xi_vec[2]*(0.5*x^2-0.5*x)
end
xi_vec = random_inputs()
u_bar(0.5,xi_vec)
################################################################################


################################################################################
# Assembly for FE Matrices
################################################################################
# Stiffness Matrix
function stiff_mat(mesh_h::Float64,mesh_N::Int64)

    A_h = (1/mesh_h)*spdiagm(-1 =>  -ones(mesh_N-2),
                              0 => 2*ones(mesh_N-1),
                              1 =>  -ones(mesh_N-2))
    return A_h
end
A_h = stiff_mat(4.0,3)

# Mass Matrix
function mass_mat(mesh_h::Float64,mesh_N::Int64)
    M_h = (mesh_h/6)*spdiagm(-1 =>   ones(mesh_N-2),
                              0 => 4*ones(mesh_N-1),
                              1 =>   ones(mesh_N-2))

    return M_h
end
M_h = mass_mat(4.0,3)

################################################################################
# Solution and Adjoint Operators
################################################################################
function s_bar(z::Vector{Float64},
               A_h::SparseMatrixCSC{Float64,Int64},
               M_h::SparseMatrixCSC{Float64,Int64})
    return A_h\(M_h*z)
end
# Generate a state for a single sample
Sb = s_bar(randn(2),A_h,M_h)

function solution_op(xi_vec::Vector{Float64},
                     s_bar::Vector{Float64},
                     mesh_h)

    z_sol     = [0 ; s_bar ; 0]
    u_bar_vec = zeros(length(z_sol))

    for i in 1:length(u_bar_vec)
        u_bar_vec[i] = u_bar((i-1)*mesh_h,xi_vec)
    end
    return u_bar_vec + xi_vec[1]^(-1)*z_sol
end
Sz = solution_op(random_inputs(),Sb,4.0)


function adjoint_op(Sz::Vector{Float64},
                    yd::Vector{Float64},
                    psi::Vector{Float64},
                    A_h::SparseMatrixCSC{Float64,Int64},
                    M_h::SparseMatrixCSC{Float64,Int64})

    rhs  = M_h*(Sz - yd)

    return A_h\(rhs)
end
# Generate an adjoint state for a single sample
Lam_z = adjoint_op(Sz[2:end-1],ones(2),zeros(2),A_h,M_h)


################################################################################
# Additional Functions (Projection, Characteristic, etc.)
################################################################################
# Pointwise projection on simple bounds
function proj_ab(y::Vector{Float64},a::Vector{Float64},b::Vector{Float64})
    return y - max.(0,y-b) + max.(0,a-y)
end
proj_ab(randn(2),-0.5*ones(2),0.5*ones(2))


# Discrete L^2-norm
function norm_L2(x::Vector{Float64},M_h::SparseMatrixCSC{Float64,Int64})
    return sqrt(x'*M_h*x)
end
norm_L2(rand(2),M_h)


# Characteristic function (vector) for the inactive set
function inactive_bl(z::Vector{Float64},a::Vector{Float64},b::Vector{Float64})
    chi_I = zeros(length(z))
    for i in 1:length(z)
        if z[i] - a[i] >= 0 && z[i] - b[i] <= 0
            chi_I[i] = 1.0
        end
    end

    return chi_I
end
inactive_bl(randn(10),-0.5*ones(10),0.5*ones(10))

# residual of first-order system
function residual_op(x::Vector{Float64},
                     y::Vector{Float64},
                     a::Vector{Float64},
                     b::Vector{Float64},
                     M_h::SparseMatrixCSC{Float64,Int64})

    return x - proj_ab(x - M_h*x - M_h*y,a,b)
end
op_res = residual_op(rand(2),rand(2),-0.025*ones(2),0.1*ones(2),M_h)


# Characteristic function (vector) for the active sets
function active_a(z::Vector{Float64},a::Vector{Float64})
    chi_A = zeros(length(z))
    for i in 1:length(z)
        if z[i] - a[i] < 0
            chi_A[i] = 1.0
        end
    end

    return chi_A
end

function active_b(z::Vector{Float64},b::Vector{Float64})
    chi_B = zeros(length(z))
    for i in 1:length(z)
        if z[i] - b[i] > 0
            chi_B[i] = 1.0
        end
    end

    return chi_B
end
test_z = rand(2)
test_y = rand(2)
Act_a = active_a(test_z - M_h*test_z - M_h*test_y,-0.025*ones(2))
Act_b = active_b(test_z - M_h*test_z - M_h*test_y,0.1*ones(2))
Act_I = inactive_bl(test_z - M_h*test_z - M_h*test_y,
                    -0.025*ones(size(test_z)),0.1*ones(size(test_z)))

Act_I = inactive_bl(test_z - M_h*test_z - M_h*test_y,
                    -0.025*ones(size(test_z)),0.1*ones(size(test_z)))
chi_I = (Act_I.!=0)
test_z[(Act_I.!=0)]
Act_I.*test_z

################################################################################
# Hessian Vector Products (Single and Mean)
################################################################################
# A single Hessian vector product
function hess_vec_prod(dz::Vector{Float64},
                       A_h::SparseMatrixCSC{Float64,Int64},
                       M_h::SparseMatrixCSC{Float64,Int64},
                       Sz::Vector{Float64},
                       psi::Vector{Float64})

    q = A_h\(M_h*dz)

    chi_C = zeros(length(q))
    for i in 1:length(q)
        if -Sz[i] + psi[i] > 0
            chi_C[i] = 1.0
        end
    end

    rhs = M_h*q
    p   = A_h\(rhs)
    return p
end
Gh_dz = hess_vec_prod(randn(size(A_h,1)),A_h,M_h,Sz[2:end-1],zeros(2))


# Expectation of the hessian vector products
function exp_hv_prod(dz::Vector{Float64},
                     Sb::Vector{Float64},
                     xi_mat::Matrix{Float64},
                     psi::Vector{Float64},
                     A_h::SparseMatrixCSC{Float64,Int64},
                     M_h::SparseMatrixCSC{Float64,Int64},
                     nu::Float64,
                     mesh_h::Float64,
                     mesh_N::Int64)

    samp_N  = size(xi_mat,2)
    E_Gh_dz = zeros(size(dz))

    for i in 1:samp_N
        Sz      = solution_op(xi_mat[:,i],Sb,mesh_h)
        Gh_dz   = hess_vec_prod(dz,A_h,M_h,Sz[2:end-1],psi)
        E_Gh_dz = E_Gh_dz + xi_mat[1,i]^(-1)*Gh_dz
    end

    return (nu*samp_N)^(-1)*M_h*E_Gh_dz
end

xi_mat = zeros(4,2)
for i in 1:2
    xi_mat[:,i] = random_inputs()
end
exp_hv_prod(randn(size(A_h,1)),Sb,xi_mat,zeros(2),A_h,M_h,1.0,4.0,2)


# Reduced Matrix for z updates on inactive set
function red_matr_vec_prod(chi_I::BitArray{1},
                           dz_I::Vector{Float64},
                           Sb::Vector{Float64},
                           xi_mat::Matrix{Float64},
                           psi::Vector{Float64},
                           A_h::SparseMatrixCSC{Float64,Int64},
                           M_h::SparseMatrixCSC{Float64,Int64},
                           nu::Float64,
                           mesh_h::Float64,
                           mesh_N::Int64)



    tmp_z        = zeros(size(chi_I))
    tmp_z[chi_I] = dz_I

    M_h_I = M_h[chi_I,chi_I]
    G_I   = exp_hv_prod(tmp_z,Sb,xi_mat,psi,A_h,M_h,nu,mesh_h,mesh_N)

    return M_h_I*dz_I + G_I[chi_I]
end

test_dz = ones(2)
red_matr_vec_prod(chi_I,test_dz[chi_I],Sb,xi_mat,zeros(2),A_h,M_h,1.0,4.0,2)

# A partial function application of the previous function for use in CG method
function pa_red_matr_vec_prod(chi_I::BitArray{1},
                              Sb::Vector{Float64},
                              xi_mat::Matrix{Float64},
                              psi::Vector{Float64},
                              A_h::SparseMatrixCSC{Float64,Int64},
                              M_h::SparseMatrixCSC{Float64,Int64},
                              nu::Float64,
                              mesh_h::Float64,
                              mesh_N::Int64)

    Ap(dz_I::Vector{Float64}) = red_matr_vec_prod(chi_I,dz_I,Sb,xi_mat,
                                                  psi,A_h,M_h,nu,mesh_h,mesh_N)
    return Ap
end

################################################################################
# a "Matrix-free" CG method
################################################################################

function cg_no_matrix(x_0::Array{Float64,1},
                      b::Array{Float64,1},
                      Ap::Function,
                      M_h::SparseMatrixCSC{Float64,Int64},
                      epsilon::Float64)
  n    = length(x_0)
  r    = b - Ap(x_0)
  rtr  = r'*r
  beta = 0.0
  p    = zeros(n)
  x    = copy(x_0)
  # println(["Current residual", norm_L2(r,M_h)])
    for i=0:1:n
        if norm_L2(r,M_h) <= epsilon
            return x , i
        end

        if i > 0
            rtr_o = rtr
            rtr   = r'*r
            beta  = rtr/rtr_o
        end

        p     = r + beta[1]*p
        Atp   = Ap(p)
        ptAp  = p'*Atp
        alph  = rtr/ptAp
        x     = x + alph[1]*p
        r     = r - alph[1]*Atp
        # print(i, ' ')
        # println(["Current residual", norm_L2(r,M_h)])
    end
  return 0
end


################################################################################
# The Semismooth Newton Method
################################################################################
function main(z_k::Vector{Float64},
              a::Vector{Float64},
              b::Vector{Float64},
              yd::Vector{Float64},
              psi::Vector{Float64},
              xi_mat::Matrix{Float64},
              A_h::SparseMatrixCSC{Float64,Int64},
              M_h::SparseMatrixCSC{Float64,Int64},
              samp_N::Int64,
              mesh_h::Float64,
              mesh_N::Int64,
              tikh_a::Float64)

    z_o = z_k

    Sb = s_bar(z_k,A_h,M_h)
    E_Lz = zeros(mesh_N-1)
    for i in 1:samp_N
        Sz   = solution_op(xi_mat[:,i],Sb,mesh_h)
        Lz   = adjoint_op(Sz[2:end-1],yd,zeros(mesh_N-1),A_h,M_h)
        E_Lz = E_Lz + Lz
    end
    E_Lz = (tikh_a*samp_N)^(-1)*E_Lz

    op_res   = residual_op(z_k,E_Lz,a,b,M_h)
    norm_res = norm_L2(op_res,M_h)

    it = 1
    while norm_res > 1e-8 && it < 20
        Act_I = inactive_bl(z_k - M_h*z_k - M_h*E_Lz,a,b)
        chi_I = (Act_I.!=0)

        Act_A = active_a(z_k - M_h*z_k - M_h*E_Lz,a)
        chi_A = (Act_A.!=0)

        Act_B = active_b(z_k - M_h*z_k - M_h*E_Lz,b)
        chi_B = (Act_B.!=0)

        dz        = zeros(mesh_N-1)
        dz[chi_A] = a[chi_A] - z_k[chi_A]
        dz[chi_B] = b[chi_B] - z_k[chi_B]

        Mdz = M_h[:,chi_A]*dz[chi_A] + M_h[:,chi_B]*dz[chi_B]

        dz_ab = Act_A.*dz + Act_B.*dz
        G_ab  = exp_hv_prod(dz_ab,Sb,xi_mat,psi,A_h,M_h,tikh_a,mesh_h,mesh_N)

        Fz = -1.0*(op_res[chi_I] + Mdz[chi_I] + G_ab[chi_I])

        Gz_I = pa_red_matr_vec_prod(chi_I,Sb,xi_mat,psi,
                                    A_h,M_h,tikh_a,mesh_h,mesh_N)
        M_h_I = M_h[chi_I,chi_I]

        dz_I, cg_it = cg_no_matrix(dz[chi_I],Fz,Gz_I,M_h_I,1e-8)

        z_k[chi_A] = z_k[chi_A] + dz[chi_A]
        z_k[chi_B] = z_k[chi_B] + dz[chi_B]
        z_k[chi_I] = z_k[chi_I] + dz_I


        Sb = s_bar(z_k,A_h,M_h)
        E_Lz = zeros(mesh_N-1)
        for i in 1:samp_N
            Sz   = solution_op(xi_mat[:,i],Sb,mesh_h)
            Lz   = adjoint_op(Sz[2:end-1],yd,zeros(mesh_N-1),A_h,M_h)
            E_Lz = E_Lz + Lz
        end
        E_Lz = (tikh_a*samp_N)^(-1)*E_Lz

        op_res   = residual_op(z_k,E_Lz,a,b,M_h)
        norm_res = norm_L2(op_res,M_h)
        println("##############")
        println("norm_res = ", norm_res)
        println("cg_it = ", cg_it)
        println("##############")

        it += 1
        if it == 20
            println("##############")
            println("Warning!")
            println("Newton Iterations = ", it)
            println("norm_res = ", norm_res)
            println("Breaking loop.")
            println("##############")
            return z_0
        end
    end

    return z_k
end


################################################################################
# An Example
################################################################################
samp_N  = 500              # sample sizes
tol_abs = 1e-8             # absolute tolerance
tol_rel = 1e-4             # relative tolerance
mesh_N  = 2^10             # number of nodes
mesh_h  = 1/(mesh_N - 1)   # width of intervals
tikh_a  = 0.001            # Tikhonnov regularization value
################################################################################

A_h = stiff_mat(mesh_h,mesh_N)
M_h = mass_mat(mesh_h,mesh_N)

xi_mat = zeros(4,samp_N)
for i in 1:samp_N
    xi_mat[:,i] = random_inputs()
end

a  = -0.75*ones(mesh_N-1)
b  =  0.75*ones(mesh_N-1)

yd     = zeros(mesh_N-1)
x_grid = collect(0:(1/(mesh_N-2)):1)
for i in 1:(mesh_N-1)
    yd[i] = desired_state(x_grid[i])
end

z_star = main(ones(mesh_N-1),a,b,yd,
              zeros(mesh_N-1),xi_mat,A_h,M_h,samp_N,mesh_h,mesh_N,tikh_a)


################################################################################
#=

The following functions are used to rapidly generate the plots.

=#
################################################################################
zp = plot(collect(0:1/(mesh_N-2):1),z_star, lw = 3, legend = false)
savefig(zp,"OptimalControl.pdf")

# zp = plot(z_star, lw = 2, legend = false)
# for i in 1:(samp_N-1)
#     z_star = main(ones(mesh_N-1),a,b,yd,
#                   zeros(mesh_N-1),xi_mat[:,1:i],
#                   A_h,M_h,samp_N,mesh_h,mesh_N,tikh_a)
#     zp = plot!(z_star, lw = 2, legend = false)
#     display(zp)
# end
# savefig(zp,"PDESolutionEvol.pdf")

################################################################################
# Stability Plots
################################################################################
function plot_diff_zhN(z_k::Vector{Float64},
                       a::Vector{Float64},
                       b::Vector{Float64},
                       yd::Vector{Float64},
                       psi::Vector{Float64},
                       A_h::SparseMatrixCSC{Float64,Int64},
                       M_h::SparseMatrixCSC{Float64,Int64},
                       samp_N::Int64,
                       mesh_h::Float64,
                       mesh_N::Int64,
                       tikh_a::Float64)

    # samp_N  = 50
    xi_mat = zeros(4,samp_N)
    for i in 1:samp_N
        xi_mat[:,i] = random_inputs()
    end

    z_star = main(ones(mesh_N-1),a,b,yd,zeros(mesh_N-1),xi_mat,
                  A_h,M_h,samp_N,mesh_h,mesh_N,tikh_a)

    p = plot()
    display(p)
    for k = 1:100
        z_diff = zeros(samp_N)
        for i in 1:samp_N
            samp_m = i
            cols_m = sample(1:samp_N, i, replace = false)
            z_star_m = main(ones(mesh_N-1),a,b,yd,zeros(mesh_N-1),
                            xi_mat[:,cols_m],A_h,M_h,samp_m,mesh_h,mesh_N,tikh_a)

            z_diff[i] = norm_L2(z_star - z_star_m,M_h)
            # println(z_diff)
        end
        z_diff = z_diff
        p_m = scatter!(z_diff, legend = false)
        display(p_m)
        savefig(p_m,"EmpiricalStabilityPDESol.pdf")
    end
    return p_m
end

plot_diff_zhN(ones(mesh_N-1),a,b,yd,zeros(mesh_N-1),A_h,M_h,50,mesh_h,mesh_N,tikh_a)

function plot_sol_oos(z_star,A_h,M_h,mesh_h,yd,tikh_a)
    Sb = s_bar(z_star,A_h,M_h)
    Sz = solution_op(random_inputs(),Sb,mesh_h)
    Obj = 0.0
    Obj += 0.5*norm_L2(Sz[2:end-1]-yd,M_h)^2
    ES = Sz
    p = plot(collect(0:1/mesh_N:1),Sz,legend=false)
    for i = 1:1000
        Sb = s_bar(z_star,A_h,M_h)
        Sz = solution_op(random_inputs(),Sb,mesh_h)
        Obj += 0.5*norm_L2(Sz[2:end-1]-yd,M_h)^2
        ES = ES + Sz
        # p = plot!(collect(0:1/mesh_N:1),Sz,legend = false)
    end
    # display(p)
    q = plot(collect(0:1/(mesh_N-2):1),
                     [1/(1000)*ES[2:end-1],yd],
                     lw = 3, legend = false)
    display(q)
    savefig(q,"OutOfSampleState.pdf")
    nothing
end
plot_sol_oos(z_star,A_h,M_h,mesh_h,yd,tikh_a)


function optimal_value(z_star,A_h,M_h,xi_mat,mesh_h,yd,tikh_a,samp_N)
    Sb = s_bar(z_star,A_h,M_h)
    Sz = solution_op(xi_mat[:,1],Sb,mesh_h)
    Obj = 0.0
    Obj += 0.5*norm_L2(Sz[2:end-1]-yd,M_h)^2
    for i = 2:samp_N
        Sb = s_bar(z_star,A_h,M_h)
        Sz = solution_op(xi_mat[:,i],Sb,mesh_h)
        Obj += 0.5*norm_L2(Sz[2:end-1]-yd,M_h)^2
    end
    Obj = Obj/samp_N + tikh_a/2*norm_L2(z_star,M_h)
    return Obj
end
optimal_value(z_star,A_h,M_h,xi_mat,mesh_h,yd,tikh_a,samp_N)
samp_m = 2
optimal_value(z_star,A_h,M_h,xi_mat[:,1:samp_m],mesh_h,yd,tikh_a,samp_m)



function stab_optimal_value(z_k::Vector{Float64},
                            a::Vector{Float64},
                            b::Vector{Float64},
                            yd::Vector{Float64},
                            psi::Vector{Float64},
                            A_h::SparseMatrixCSC{Float64,Int64},
                            M_h::SparseMatrixCSC{Float64,Int64},
                            L_h::SparseMatrixCSC{Float64,Int64},
                            samp_N::Int64,
                            mesh_h::Float64,
                            mesh_N::Int64,
                            tikh_a::Float64,
                            gamma::Float64)

    # samp_N  = 50
    xi_mat = zeros(4,samp_N)
    for i in 1:samp_N
        xi_mat[:,i] = random_inputs()
    end

    z_star = main(ones(mesh_N-1),a,b,yd,
                  zeros(mesh_N-1),xi_mat,
                  A_h,M_h,L_h,
                  samp_N,mesh_h,mesh_N,
                  tikh_a,gamma)

    obj_vec = zeros(samp_N)

    Sb = s_bar(z_star,A_h,M_h)
    Sz = solution_op(xi_mat[:,1],Sb,mesh_h)
    Obj = 0.0
    Obj += 0.5*norm_L2(Sz[2:end-1]-yd,M_h)^2
    for i = 2:samp_N
        Sb = s_bar(z_star,A_h,M_h)
        Sz = solution_op(xi_mat[:,i],Sb,mesh_h)
        Obj += 0.5*norm_L2(Sz[2:end-1]-yd,M_h)^2
    end
    Obj = Obj/samp_N + tikh_a/2*norm_L2(z_star,M_h)
    Opt = Obj

    p = plot()
    display(p)
    for k = 1:100
        obj_vec = zeros(samp_N)
        for i in 1:samp_N
            samp_m = i
            cols_m = sample(1:samp_N, i, replace = false)
            z_star_m = main(ones(mesh_N-1),
                            a,b,yd,
                            zeros(mesh_N-1),
                            xi_mat[:,cols_m],
                            A_h,M_h,L_h,
                            samp_m,mesh_h,mesh_N,
                            tikh_a,gamma)

            Sb = s_bar(z_star_m,A_h,M_h)
            Sz = solution_op(xi_mat[:,cols_m[1]],Sb,mesh_h)
            Obj = 0.0
            Obj += 0.5*norm_L2(Sz[2:end-1]-yd,M_h)^2
            for i = 2:samp_m
                Sb = s_bar(z_star_m,A_h,M_h)
                Sz = solution_op(xi_mat[:,cols_m[i]],Sb,mesh_h)
                Obj += 0.5*norm_L2(Sz[2:end-1]-yd,M_h)^2
            end
            Obj = Obj/samp_m + tikh_a/2*norm_L2(z_star,M_h)
            obj_vec[i] = abs(Obj-Opt)
            println(obj_vec[i])
            # z_diff[i] = norm_L2(z_star - z_star_m,M_h)
            # println(z_diff)
        end
        # z_diff = z_diff .+ 1e-16
        p_m = scatter!(obj_vec, legend = false)
        display(p_m)
        savefig(p_m,"EmpiricalStabilityOptimalVal.pdf")
    end
    return 0.0#p_m
end
stab_optimal_value(ones(mesh_N-1),a,b,yd,
                   zeros(mesh_N-1),
                   A_h,M_h,L_h,
                   50,mesh_h,mesh_N,
                   tikh_a,gamma)
