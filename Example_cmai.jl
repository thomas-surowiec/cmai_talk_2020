cd(dirname(@__FILE__()))
using Plots

################################################################################
# Abstract Types
################################################################################
mutable struct Objective
     Obj::Function
    dObj::Function
end#

mutable struct Projection
    Proj::Function
end#

mutable struct StepSize
     # StepRule::Function
    FixedStep::Float64
end

################################################################################
# DEFINE a projection onto multidimensional bilateral constraints
################################################################################
function projection_a(x::Vector{Float64},a::Vector{Float64})
    proj = zeros(length(x))

    for i in 1:length(x)
        if x[i] <= a[i]
            proj[i] = a[i]
        else
            proj[i] = x[i]
        end
    end

    return proj
end
################################################################################

################################################################################
# This is a partial function application of the projection_ab
function pa_projection_a(a::Vector{Float64})
    pa_proj_ab(x::Vector{Float64}) = projection_ab(x,a)

    return pa_proj_ab
end
################################################################################

################################################################################
function ns_objective(c::Float64,b::Float64,h::Float64,
                      P::Vector{Float64},D::Vector{Float64},
                      x::Float64)
    obj = 0.0
    for i in 1:length(P)
        obj += P[i]*(b*max(0.0,D[i]-x)+h*max(0.0,x - D[i]))
    end
    return c*x + obj #+ 0.0001*x^2
end

function subgrad_ns_obj(c::Float64,b::Float64,h::Float64,
                        P::Vector{Float64},D::Vector{Float64},
                        x::Float64)
    subgrad = 0.0
    for i in 1:length(P)
        if D[i] > x
            subgrad += P[i]*b*(-1.0)
        elseif D[i] < x
            subgrad += P[i]*h*(1.0)
        else
            subgrad += P[i]*(h-b)*(rand()+rand())
        end
    end
    return c + subgrad #+ 0.0002*x
end
################################################################################

################################################################################
# This is a partial function application of ns_objective
function pa_ns_objective(c::Float64,b::Float64,h::Float64,
                        P::Vector{Float64},D::Vector{Float64})

    pa_ns_obj(x::Float64) = ns_objective(c,b,h,P,D,x)

    return pa_ns_obj
end

# This is a partial function application of subgrad_ns_obj
function pa_subgrad_ns_obj(c::Float64,b::Float64,h::Float64,
                        P::Vector{Float64},D::Vector{Float64})

    pa_d_ns_obj(x) = subgrad_ns_obj(c,b,h,P,D,x)

    return pa_d_ns_obj
end
################################################################################

################################################################################
function proj_sub_grad(f::Objective,
                       P_X::Projection,
                       g_t::StepSize,
                       maxit::Int64,
                       x_0::Union{Float64,Vector{Float64}},
                       step::String)

    it = 0
    f_vec    = zeros(maxit)
    f_vec[1] = f.Obj(x_0)
    # gam_t    = g_t.StepRule(maxit)

    if step == "fixed"
        while it < maxit
            x_1 = P_X.Proj(x_0 - g_t.FixedStep*f.dObj(x_0))

            # Behavior of iterates
            # Progress:
            # println("||x_0 - x_1|| = ", norm(x_0-x_1))
            # Optimality:
            # println("x - Proj_X(x - grad f(x)) = ", norm(x_1 - P_X.Proj(x_1 - f.dObj(x_1))))
            # Objective Function:
            # println("df(x_1) = ", f.dObj(x_1))

            x_0 = x_1
            it += 1
            f_vec[it] = f.Obj(x_0)
        end
    else
        while it < maxit
            x_1 = P_X.Proj(x_0 - g_t.StepRule(maxit)*f.dObj(x_0))

            # Behavior of iterates
            # Progress:
            # println("||x_0 - x_1|| = ", norm(x_0-x_1))
            # Optimality:
            # println("x - Proj_X(x - grad f(x)) = ", norm(x_1 - P_X.Proj(x_1 - f.dObj(x_1))))
            # Objective Function:
            # println("f(x_1) - f(x_0) = ", f.Obj(x_1) - f.Obj(x_0))
            x_0 = x_1
            it += 1
            f_vec[it] = f.Obj(x_0)
        end
    end

    return x_0, f_vec
end

K = 10000
P = ones(K)/K
D = rand(K) .+ 1
b = 2.0
c = 1.0
h = 1.0
x = 0.0

P_X = Projection(x -> max(0,x))
f   = Objective(pa_ns_objective(c,b,h,P,D),pa_subgrad_ns_obj(c,b,h,P,D))
g_t = StepSize(0.075)

x_sol, f_vec = proj_sub_grad(f,P_X,g_t,100,x,"fixed")
1+ 1/3
println(x_sol)
plot(f_vec)

err_vec     = zeros(10000)
err_vec_reg = zeros(10000)
for i in 1:10000
    K = i
    P = ones(K)/K
    D = rand(K) .+ 1
    b = 2.0
    c = 1.0
    h = 1.0
    x = 1.0
    f = Objective(pa_ns_objective(c,b,h,P,D),pa_subgrad_ns_obj(c,b,h,P,D))
    x_sol, f_vec = proj_sub_grad(f,P_X,g_t,100,x,"fixed")
    println(x_sol)
    err_vec[i] = abs(x_sol - (1+ 1/3))
    # err_vec_reg[i] = abs(x_sol - 1+ 1/3)
end

ep  = plot(err_vec,lw = 2,label=false)
ep2 = plot!(err_vec_reg,lw = 2,label=false)
# display(sp)
savefig(ep,"EmpiricalStability1D.pdf")
