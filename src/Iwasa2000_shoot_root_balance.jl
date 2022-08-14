#!/usr/bin/env julia

using ECOS
using JuMP
using Plots
using QHull

function gfunc(x1::Vector{Float64}, x2::Vector{Float64})::Vector{Float64}
  a1 = 2.0;
  a2 = 60.0;
  b1 = 0.5;
  b2 = 2.0;
  L = 1.0;
  W = 1.0;
  1.0 ./ ( a1./L./(x1.^b1) .+ a2./W./(x2.^b2) );
end

# Form an N x 2 matrix of sample points
pts = vec(collect.(
  Iterators.product(LinRange(0.01,100,80), LinRange(0.01,100,80))
));
pts = reduce(hcat, pts)';
# Get values of `gfunc at these points`
zz = gfunc(pts[:,1], pts[:,2]);

# Get a convex hull to approximate `gfunc`
hull_pts = hcat(pts,zz);

# Build the model
dt=0.8;
model = Model(with_optimizer(
  ECOS.Optimizer, maxit=10000, feastol=1e-4, reltol=1e-3, abstol=1e-3
));
timeseries = LinRange(0.0, 153.0, Int(round((153.0-0.0)/dt)));

N = length(timeseries);
@variable(model, u[1:N, 1:3], lower_bound=0);
@variable(model, X1[1:N],     lower_bound=0);
@variable(model, X2[1:N],     lower_bound=0);
@variable(model, R[1:N],      lower_bound=0);
@variable(model, g[1:N],      lower_bound=0);

@constraint(model, X1[1]==1.0);
@constraint(model, X2[1]==2.5);
@constraint(model, R[1]==0.0);

scatter(hull_pts[:,1], hull_pts[:,2], hull_pts[:,3]);

hull = chull(hull_pts);
good_facets = hull.facets[hull.facets[:,3].>0,:];

for ti in 1:(N-1)
  for f in eachrow(good_facets)
    @constraint(
      model, f[1] * X1[ti] + f[2] * X2[ti] + f[3] * g[ti] + f[4] <= 0
    );
  end
  @constraint(model, sum(u[ti,:])==g[ti]);
  @constraint(model, X1[ti+1] == X1[ti] + dt * u[ti, 1]);
  @constraint(model, X2[ti+1] == X2[ti] + dt * u[ti, 2]);
  @constraint(model, R[ti+1]  == R[ti]  + dt * u[ti, 3]);
end

@objective(model, Max, R[end]);

optimize!(model)

# Everything above is displayed in the paper

solution_summary(model)


plot(timeseries, value.(X1), linewidth=3)
plot!(timeseries, value.(X2), linewidth=3)
plot!(timeseries, value.(R), linewidth=3)
savefig("/z/iwasa1984_timeseries.pdf")

# plot(value.(X1), value.(X2))
# plot!(xlims=(0,100), ylims=(0,100), aspect_ratio=:equal)

plot(timeseries, reduced_cost.(X1))
plot!(timeseries, reduced_cost.(X2))
plot!(timeseries, reduced_cost.(R))

plot(timeseries, shadow_price.(c))
plot!(timeseries[2:end], shadow_price.(c1))
plot!(timeseries[2:end], shadow_price.(c2))
plot!(timeseries[2:end], shadow_price.(c3))

plot(timeseries[2:end], dual.(c1))
plot!(timeseries[2:end], dual.(c2))
plot!(timeseries[2:end], dual.(c3))

# plot(timeseries, reduced_cost.(X1))
# plot!(timeseries, reduced_cost.(X2))

actual_g = gfunc(value.(X1), value.(X2));
model_g = value.(g);
absdiff = abs.(actual_g.-model_g)
reldiff = absdiff./actual_g

maximum(absdiff)
maximum(reldiff)

maximum(value.(X1))
maximum(value.(X2))

maximum(absdiff)
maximum(reldiff)

# class Problem:
#   def __init__(self, tmin: float, tmax: float, desired_tstep: float, years: int = 1, seasonize: bool = False) -> None:
#     ts, dt = gen_timeseries(start=tmin, stop=tmax, timestep=desired_tstep)
#     self.vars: Dict[str, Variable] = {}
#     self.controls: Dict[str, Variable3D] = {}
#     self.constraints: List[Constraint] = []
#     self.timeseries: np.ndarray = ts
#     self.dt: float = dt

#     if years<=0:
#       raise RuntimeError("Seasons must be >0!")
#     if years!=int(years):
#       raise RuntimeError("Seasons must be an integer!")
#     self.years = years
#     self.seasonize = seasonize

#   def _time_shape(self) -> Tuple[int ,...]:
#     if self.years==1 and not self.seasonize:
#       return (len(self.timeseries), )
#     else:
#       return (self.years, len(self.timeseries))

#   def tfirst(self, var: str) -> Variable:
#     if self.years==1 and not self.seasonize:
#       return self.vars[var][0]
#     else:
#       return self.vars[var][0,0]

#   def tlast(self, var: str) -> Variable:
#     if self.years==1 and not self.seasonize:
#       return self.vars[var][-1]
#     else:
#       return self.vars[var][-1,-1]



#   def add_year_var(self,
#     name: str,
#     initial: Optional[float] = None,
#     lower_bound: Optional[float] = None,
#     upper_bound: Optional[float] = None,
#     anchor_last: bool = False
#   ) -> Variable:
#     self.vars[name] = Variable(
#       self.years,
#       name=name,
#       pos=(lower_bound>=0) # cvxpy gains extra analysis powers if pos is used
#     )
#     self.vars[name].ts_type = "year"

#     if lower_bound is not None:
#       self.constraint(lower_bound<=self.vars[name])
#     if upper_bound is not None:
#       self.constraint(self.vars[name]<=upper_bound)
#     if initial is not None:
#       self.constraint(self.vars[name][0]==initial)
#     if anchor_last:
#       self.constraint(self.vars[name][-1]==0)

#     return self.vars[name]

#   def add_control_var(self,
#     name: str,
#     dim: int,
#     lower_bound: Optional[float] = None,
#     upper_bound: Optional[float] = None,
#   ) -> Variable3D:
#     self.controls[name] = Variable3D(
#       (self.years, len(self.timeseries), dim),
#       name=name,
#       pos=(lower_bound>=0)
#     )
#     self.controls[name].ts_type = "control"

#     for x in self.controls[name]:
#       if lower_bound is not None:
#         self.constraint(lower_bound<=x)
#       if upper_bound is not None:
#         self.constraint(x<=upper_bound)

#     return self.controls[name]

#   def dconstraint(
#     self,
#     var: Variable,
#     t: Union[int, Tuple[int, int]],
#     dt: float,
#     rhs: Expression,
#   ) -> None:
#     if isinstance(t, tuple):
#       self.constraints.append(
#         var[t[0],t[1]+1] == var[t] + dt * rhs
#       )
#     elif isinstance(t, int):
#       self.constraints.append(
#         var[t+1] == var[t] + dt * rhs
#       )

#   def constraint(self, constraint: Constraint) -> None:
#     self.constraints.append(constraint)

#   def add_piecewise_function(
#     self,
#     func: Callable[[np.ndarray], float],
#     yvar: Variable,
#     xvar: Variable,
#     xmin: float,
#     xmax: float,
#     n: int,
#     last_ditch: Optional[float] = None,
#     use_sos2: bool = False,
#   ) -> Variable:
#     #William, p. 149
#     xvals = np.linspace(start=xmin, stop=xmax, num=n)
#     if last_ditch is not None:
#       xvals = np.hstack((xvals, last_ditch))
#     yvals = func(xvals)
#     l = Variable(len(xvals))
#     self.constraint(xvar==xvals*l)
#     self.constraint(yvar==yvals*l)
#     self.constraint(cp.sum(l)==1)
#     self.constraint(l>=0)
#     if use_sos2:
#       self.add_sos2_constraint(l)
#     return yvar

#   def add_piecewise1d_function(
#     self,
#     func: Piecewise1D,
#     xvar: Variable,
#   ) -> Variable:
#     fitted_y, constraints = func.fit(xvar)
#     self.constraints.extend(constraints)
#     return fitted_y

#   def add_piecewise2d_function(
#     self,
#     func: Union[Piecewise2DMIP, Piecewise2DConvex],
#     xvar: Variable,
#     yvar: Variable,
#   ) -> Variable:
#     fitted_z, constraints = func.fit(xvar, yvar)
#     self.constraints.extend(constraints)
#     return fitted_z

#   def michaelis_menten_constraint(
#     self,
#     lhs_var: Variable,
#     rhs_var: Variable,
#     β1: float = 1,
#     β2: float = 1,
#     β3: float = 1,
#   ) -> None:
#     """lhs_var <= β1*rhs_var/(β2+β3*rhs_var)"""
#     β1 = β1 / β3
#     β2 = β2 / β3
#     self.constraint(lhs_var <= β1 * (1-β2*cp.inv_pos(β2+rhs_var)))

#     # self.constraint(0<=rhs_var - lhs_var) #TODO: Is there any way to get rid of this constraint?
#     # self.constraint(cp.SOC(
#     #   β1 * β2  +  β1 * rhs_var  -  β2 * lhs_var,
#     #   cp.vstack([
#     #     β1 * rhs_var,
#     #     β2 * lhs_var,
#     #     β1 * β2
#     #   ])
#     # ))

#   def hyperbolic_constraint(self, w: Variable, x: Variable, y: Variable) -> None:
#     """dot(w,w)<=x*y"""
#     self.constraint(cp.SOC(x + y, cp.vstack([2 * w, x - y])))

#   def plotVariables(self, norm_controls: bool = True, hide_vars: Optional[List[str]] = None) -> plt.Figure:
#     fig, axs = plt.subplots(2)

#     if hide_vars is None:
#       hide_vars = []

#     for name, var in self.vars.items():
#       if name in hide_vars:
#         continue
#       elif var.ts_type=="time":
#         full_times = []
#         for n in range(self.years):
#           full_times.extend([n*self.timeseries[-1] + x for x in self.timeseries])
#         axs[0].plot(full_times, var.value.flatten(), label=name)
#       elif var.ts_type=="y":
#         full_times = [(n+1)*self.timeseries[-1] for n in range(self.years)]
#         axs[0].plot(full_times, var.value.flatten(), '.', label=name)

#     full_times = []
#     for n in range(self.years):
#       full_times.extend([n*self.timeseries[-1] + x for x in self.timeseries])

#     for name, var in self.controls.items():
#       val = np.vstack([var[n,:,:].value for n in range(self.years)])
#       if norm_controls:
#         val[val<1e-3] = 0
#         val = normalize(val, axis=1, norm='l2')
#       for ci in range(val.shape[1]):
#         axs[1].plot(full_times, val[:,ci], label=f"{name}_{ci}")
#     fig.legend()
#     return fig

#   def _error_on_bad_status(self, status: Optional[str]) -> None:
#     if status == "infeasible":
#       raise RuntimeError("Problem was infeasible!")
#     elif status == "unbounded":
#       raise RuntimeError("Problem was unbounded!")
#     elif status == "unbounded_inaccurate":
#       raise RuntimeError("Problem was unbounded and inaccurate!")

#   def solve(
#     self,
#     objective,
#     solver: str = "CBC",
#     **kwargs: Any,
#   ) -> Tuple[Union[str,None],float]:
#     problem = cp.Problem(objective, self.constraints)

#     print("Problem is DCP?", problem.is_dcp())

#     optval = problem.solve(solver, **kwargs)

#     self._error_on_bad_status(problem.status)

#     return problem.status, optval

#   def time_indices(self) -> Iterator[Tuple[int, int]]:
#     if self.years==1 and not self.seasonize:
#       return ((0,t) for t in range(len(self.timeseries)-1))
#     else:
#       return ((y,t) for y in range(self.years) for t in range(len(self.timeseries)-1))

#   def idx2time(self, time_index: int) -> float:
#     assert 0<=time_index<len(self.timeseries)
#     return self.timeseries[time_index]

#   def year_indices(self) -> Iterator[int]:
#     return range(self.years)

#   def time_discount(self, var: str, σ: float) -> Expression:
#     expr: Expression = 0
#     for n in range(self.years):
#       expr += self.vars[var][n] * (σ**n)
#     return expr

#   def constrain_control_sum_at_time(
#     self,
#     control: Variable3D,
#     sum_var: Expression,
#     t: int,
#     n: int = 0
#   ) -> None:
#     self.constraint(cp.sum(control[n, t, :]) <= sum_var) #TODO: Should this be == or <= ?

#   def add_sos2_constraint(self, x: Variable) -> None:
#     # TODO: From https://www.philipzucker.com/trajectory-optimization-of-a-pendulum-with-mixed-integer-linear-programming/
#     assert len(x.shape) == 1
#     n = x.size
#     z = Variable(n - 1, boolean=True)
#     self.constraint(0 <= x)
#     self.constraint(x[0] <= z[0])
#     self.constraint(x[-1] <= z[-1])
#     self.constraint(x[1:-1] <= z[:-1]+z[1:])
#     self.constraint(cp.sum(z) == 1)
#     self.constraint(cp.sum(x) == 1)














# struct TimeProblem
#   tmin::Float64
#   tmax::Float64
#   dt::Float64
#   years::Int64
#   seasonize::Bool
#   model::JuMP.Model
#   timeseries::Vector{<:AbstractFloat}
#   function TimeProblem(
#     optimizer,
#     tmin::Float64,
#     tmax::Float64,
#     dt::Float64,
#     years::Int64,
#     seasonize::Bool,
#   )
#     model = Model(GLPK.Optimizer)

#     timeseries = collect(tmin:dt:tmax)
#     if timeseries[end] != tmax
#       timeseries = vcat(timeseries, tmax)
#     end

#     return new(
#       tmin,
#       tmax,
#       dt,
#       years,
#       seasonize,
#       model,
#       timeseries,
#     )
#   end
# end

# macro add_time_var(
#   tp,
#   name,
#   initial=nothing,
#   lower_bound=nothing,
#   upper_bound=nothing,
#   anchor_last=false,
# )
#   quote
#     if tp.seasonize
#       x= @variable(tp.model, [1:10])
#       # JuMP.@variable(tp.model, name[tp.years, length(tp.timeseries)])
#       if !isnothing($initial)
#     #     @constraint(tp.model, name[1, 1] == initial)
#       end
#     else
#       # @variable(tp.model, $name[length(tp.timeseries)])
#       if !isnothing($initial)
#     #     @constraint(tp.model, name[1] == initial)
#       end
#     end
#   end
# end


# function add_time_var(
#   tp::TimeProblem,
#   name,
#   initial=nothing,
#   lower_bound=nothing,
#   upper_bound=nothing,
#   anchor_last::Bool=false,
# )
#   if tp.seasonize
#     var = @variable(tp.model, [tp.years, length(tp.timeseries)], base_name=name)
#     # JuMP.@variable(tp.model, name[tp.years, length(tp.timeseries)])
#     if !isnothing(initial)
#   #     @constraint(tp.model, name[1, 1] == initial)
#     end
#   else
#     var = @variable(tp.model, [length(tp.timeseries)], base_name=name)
#     if !isnothing(initial)
#   #     @constraint(tp.model, name[1] == initial)
#     end
#   end
#   return var
# end

# tp = TimeProblem(GLPK.Optimizer, 0.0, 10.0, 0.1, 1, true)
# add_time_var(tp, :F)
# all_variables(tp.model)