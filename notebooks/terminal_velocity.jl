### A Pluto.jl notebook ###
# v0.19.9

using Markdown
using InteractiveUtils

# This Pluto notebook uses @bind for interactivity. When running this notebook outside of Pluto, the following 'mock version' of @bind gives bound variables a default value (instead of an error).
macro bind(def, element)
    quote
        local iv = try Base.loaded_modules[Base.PkgId(Base.UUID("6e696c72-6542-2067-7265-42206c756150"), "AbstractPlutoDingetjes")].Bonds.initial_value catch; b -> missing; end
        local el = $(esc(element))
        global $(esc(def)) = Core.applicable(Base.get, el) ? Base.get(el) : iv(el)
        el
    end
end

# ╔═╡ e752f5da-d4b7-11ee-039b-e1e972398564
begin
	using PlutoUI
	using Gadfly	
	import PlutoUI: combine
	using LaTeXStrings
	using DataFrames
	using Colors
	using Printf
	using NumericIO
	using Logging
	using Compose
	
	Logging.disable_logging(Logging.Warn)

	la_url = "https://mdpetters.github.io/cee233/assets/Los_Angeles_Pollution.jpg"
	
	md"""
$(TableOfContents(depth=5))
# Terminal Velocity

$(Markdown.MD(Markdown.Admonition("warning", "Definition of Terminal Velocity", [md"The particular falling speed, for any given object moving through a fluid medium of specified physical properties, at which the drag forces and buoyant forces exerted by the fluid on the object just equal the gravitational force acting on the object. It falls at constant speed, unless it moves into air layers of different physical properties. In the atmosphere, the latter effect is so gradual that objects such as raindrops, which attain terminal velocity at great heights above the surface, may be regarded as continuously adjusting their speeds to remain at all times essentially in the terminal fall condition. 
"])))

## Calculation of Terminal Velocity

The terminal velocity can be computed from the force balance equating drag force and gravity. Initially, the particle is at rest and the drag force is zero. As the  velocity increases, the drag force increases, until the two forces are balanced. When the force balance is reached, the particle settles at it's terminal velocity. This is illustrated in the animation below. 
	"""
end


# ╔═╡ 74f4d2ab-7b4c-4673-aadf-67deab736052
begin
	const g = 9.81       # Acceleration due to graviation [m s-2]
	const Rd = 287.5     # Specific gas constant for dry air [J kg-1 K-1]
	
	λ(T::Float64,p::Float64) = 6.6e-8*(101315.0/p)*(T/293.15)
	η(T::Float64) = 1.83245e-5*exp(1.5*log(T/296.1))*(406.55)/(T+110.4)
	Cc(T::Float64,p::Float64,d::Float64) = 
		1.0+λ(T,p)/d*(2.34+1.05*exp.(-0.39*d/λ(T,p)))  
	ρg(T::Float64,p::Float64) = p/(Rd*T)
	Re(v::Float64,d::Float64,T::Float64,p::Float64) = ρg(T,p)*v*d/η(T)    
	Cd(Re::Float64) = 24.0/Re.*(1+0.15*Re^0.687)               
	
	function vtp(D::Float64) 
		if D < 100e-6
			3.7e7*D^2.0
		elseif (D >= 100e-6) & (D <= 1000e-6)
			4.3e3*D
		else
			9.65 - 10.3*exp(-600.0*D)
		end
	end
	
	function vt(d::Float64;T=298.15,p=1e5,ρp=1e3)  
	    f(v) = (4.0*ρp*d*g*Cc(T,p,d)/(3.0*Cd(Re(v,d,T,p))*ρg(T,p)))^0.5 
	    vts2(d, v) = (f(v)-v)^2.0 < 1e-30 ? v : vts2(d,f(v))
	    x = try 
			vts2(d, vtp(d))
		catch 
			NaN
		end
		return x
	end

	function relaxation(d)
	    Gadfly.set_default_plot_size(17Gadfly.cm, 9Gadfly.cm)
	
	    xnames = Dict(0 =>"0",1 =>"1 τ",2 =>"2 τ",3 =>"3 τ",4 =>"4 τ",5 =>"5 τ",
			6 =>"6 τ")
	    T,p,ρp = 298.15, 1e5, 1000.0
	    τ = ρp*d^2.0*Cc(T,p,d)/(18.0*η(T))
	    t = collect(0.0:0.1:6.0)
	
	    label1 = "v<sub>t</sub> = "*formatted(vt(d),:SI,ndigits=3)*"m s<sup>-1</sup>"
	    label2 = "τ =  "*formatted(τ, :SI, ndigits=3)*"s"
	    plot(layer(x=[1,1], y=[-10,1-1.0/exp(1.0)],color = [label2 for i = 1:2], 
			Geom.line, Geom.point),
	         layer(x=t, y=(1.0.-exp.(-t)),color = [label1 for i = 1:length(t)], 
				 Geom.line),
	         Scale.color_discrete_manual("darkgoldenrod3", "black"),
	         Guide.xlabel("Time"), Guide.ylabel("Velocity"),
	         Guide.xticks(ticks = [0,1,2,3,4,5,6]),
	         Guide.yticks(ticks = 0:0.1:1),
	         Scale.x_continuous(labels = i->xnames[round(Int,i)]),
	         Scale.y_continuous(labels = i->@sprintf("%.1f v<sub>t</sub>",i)),
	         Coord.cartesian(xmin=0, xmax=6.0, ymin = 0))
	end 
	vtx(τ::Float64, d::Float64, t::Float64) = vt(d)*(1.0-exp(-t/τ))
	
	dps = 1500e-6
	Tx,px,ρpx = 298.15,1e5,1000.0
	τ = ρpx*dps^2.0*Cc(Tx,px,dps)/(18.0*η(Tx))
	term = vt(dps)  
	label1 = "F<sub>g</sub> = mg = π/6 D<sup>3</sup> ρ<sub>p</sub> g"
	label2 = "F<sub>drag</sub> = π/8 C <sub>d</sub> ρ <sub>air</sub> D <sup>2</sup> v <sup>2</sup>"
	function anim1(t)
		set_default_graphic_size(10Compose.cm, 10Compose.cm)
		vc = vtx(τ,dps,t*1.0)
	    z = 0.02 + (0.40-0.17)*vc/term
	    i = vc*t
	    g1 = [[(y,0), (y,1)] for (x,y) in zip(zeros(11,1), 0:0.1:1)]
	    g2 = [[(1,y-(i%0.1)), (x,y-(i%0.1))] for (x,y) in zip(zeros(11,1), 0:0.1:1)]

		label = "(time (s),velocity (m/s),fraction of vt (-))"
		ct = round(t,digits = 1),round(vc,digits = 2),round(vc/term,digits = 2)
	    c = compose(context(), line([g1;g2]), stroke("black"),
			linewidth(0.1Compose.mm),
	    	(context(),Compose.circle(0.5, 0+0.5, 0.1), fill("steelblue3"), 
		   		stroke("white")),
	       	(context(), stroke("black"), arrow(), linewidth(0.5Compose.mm),
	            line([(0.5,0.6), (0.5, 0.85)]), line([(0.5,0.4), (0.5, 0.4-z)])), 
	       	(context(), text(0.53, 0.86, label1), text(0.53, 0.38-z, label2),
			    text(0.1, 0.025, label), text(0.1, 0.065, ct),
	            fill("black"), fontsize(10Compose.pt), stroke(RGBA(1,1,1,0))))
	end
	function mplt(T,p,ρ)  
		gengrid(r) = [vcat(map(x->x:x:9x,r)...);r[end]*10]
		Gadfly.set_default_plot_size(18Gadfly.cm, 9Gadfly.cm)
		Dp = gengrid(exp10.(-9:1:-4))
		colors = ["black", "steelblue3", "darkred"]

		xnames = Dict(-9 =>"1 nm",-8=>"10 nm",-7=>"100 nm",-6=> "1 μm",
			-5=>"10 μm",-4=>"100 μm",-3=>"1 mm")
		ynames = Dict(-9=>"1 nm/s",-6=>"1 μm/s",-3=>"1 mm/s",0=>"1 m/s")
		d = [0.001, 0.01, 0.1, 1, 10, 100, 1000]*1e-6
		vbook = [6.9e-9, 7.0e-8, 8.8e-7, 3.5e-5, 0.0031, 0.248, 3.86]
		
		l1 = layer(x=Dp, y=vt.(Dp; T=20+273.15,p=1000*100.0,ρp=1000.0), 
			Geom.line, color = ["Standard Conditions" for i = 1:length(Dp)])
		l2 = layer(x=Dp, y=vt.(Dp; T=T+273.15,p=p*100.0,ρp=ρ), 
			Geom.line, color = ["Slider Settings" for i = 1:length(Dp)])
		l3 = layer(x=d, y=vbook, 
			Geom.point, color = ["Tabulated" for i = 1:length(d)])
	
		plot(l3, l1, l2, 
			Guide.xlabel("Particle diameter"), 
			Guide.ylabel("Terminal velocity"), Guide.xticks(ticks = -9:1:-3), Guide.yticks(ticks = -9:1:1),
			Scale.color_discrete_manual(colors...),
			Scale.x_log10(labels = i->get(xnames, i, "")),
			Scale.y_log10(labels = i->get(ynames, i, "")),
			Coord.cartesian(xmin=-9, xmax=-3,ymin = -9, ymax = 1))
		end

	function parameterization()  
		gengrid(r) = [vcat(map(x->x:x:9x,r)...);r[end]*10]
	    Gadfly.set_default_plot_size(17Gadfly.cm, 9Gadfly.cm)
	    Dp1 = gengrid(exp10.(-9:1:-4))
	    Dp = [Dp1;[2e-3]]
	    xnames = Dict(-7=>"100 nm",-6=> "1 μm",-5=>"10 μm",-4=>"100 μm",
			-3=>"1 mm", -2=>"10 mm")
	    ynames = Dict(-7=>"100 nm/s",-6=>"1 μm/s",-5=>"10 μm/s",
			-4=>"100 μm/s",-3=>"1 mm/s",-2=>"10 mm/s",-1=>"100 mm/s",0=>"1 m/s",1=>"10 m/s")
	    
	    vt1(D) = 3e7*D^2.0
	    ii = (Dp .<= 100e-6) .& (Dp .>= 1.2e-9)
	    vt2(D) = 4e3*D
	    jj = (Dp .>= 100e-6) .& (Dp .<= 1.2e-3)
	    Dx = 1e-3:1e-3:1e-2
	    vt3(D) = 9.65-10.3exp(-600.0*D)
	    ll = (Dx .>= 1000e-6) .& (Dx .>= 1.2e-9)
	    label1 = ["v<sub>t</sub> = 3×10<sup>7</sup> D<sup>2</sup>" for 
		    i = 1:length(Dp)]
	    label2 = ["v<sub>t</sub> = 4×10<sup>3</sup> D" for i = 1:length(Dp)]
	    label3 = ["v<sub>t</sub> = 9.65 - 10.3e<sup>-600D</sup>" for i = 1:length(Dx)]
	    label4 = ["Numerical solution" for i = 1:length(Dp)]
	
	    plot(layer(x=Dp[ii], y=vt1.(Dp[ii]),color=label1[ii],  
			Theme(line_width=1.5Gadfly.pt, line_style=[:dash]), Geom.line),
	         layer(x=Dp[jj], y=vt2.(Dp[jj]),color=label2[jj], 
				 Theme(line_width=1.5Gadfly.pt, line_style=[:dash]), Geom.line),
	         layer(x=Dx[ll], y=vt3.(Dx[ll]),color=label3[ll], 
				 Theme(line_width=1.5Gadfly.pt, line_style=[:dash]), Geom.line),
	         layer(x=Dp, y=vt.(Dp),color=label4, 
				 Theme(line_width=2Gadfly.pt), Geom.line),
	         Guide.xlabel("Particle diameter"), Guide.ylabel("Terminal velocity"),
	         Scale.color_discrete_manual("darkgoldenrod3", "salmon", "steelblue3", 
				 "black"),
	         Guide.xticks(ticks = -9:1:-2), Guide.yticks(ticks = -9:1:1),
	         Scale.x_log10(labels = i->get(xnames, i, "")),
	         Scale.y_log10(labels = i->get(ynames, i, "")),
	         Coord.cartesian(xmin=-7, xmax=-2,ymin = -7, ymax = 1.3))
	end
	
	@bind timer Clock(interval = 0.1, max_value = 20, repeat = true)
end

# ╔═╡ d2107caa-d368-4735-9a1f-7d4018e7032c
begin

	sld = @bind vars combine() do Child
	md"""
	``T\; (\degree C)``   $(
		Child(Slider(-20:10:40.0, default = -20, show_value = true))
	) 
	
	``p\; (hPa)``  $(
		Child(Slider([10:10:90;100:100:1100.0], default = 10, show_value = true))
	) 
	
	``\rho_p\; (kg\; m^{-3})``  $(
		Child(Slider(600:200:2000.0, default = 2000, show_value = true))
	)
	"""
	end


	md"""


$(anim1(timer))
**Figure 1.** Animation of an accelerating droplet starting at rest. The force of gravity (``F_g``) accelerates the drop downward. As the velcocity ``v`` increases, the drag force (``F_{drag}``) opposes gravity and slows the acceleration until force balance is reached (``F_{drag} = F_g``).

The formulat to calculate terminal velocity is obtained by equation the drag force and gravity and is given by:
	
```math
v_t = \sqrt{\frac{4\rho_p g C_c D }{3C_d \rho_a  }} 
```

where ``v_t`` is the terminal velocity, ``\rho_p`` is the particle density, ``C_c`` is the Cunningham slip flow correction factor, ``g`` is the acceleration due to gravity, ``D`` is the particle diameter, ``C_d`` is the drag coefficient, and ``\rho_a`` is the air density. 

The slip flow correction factor is a function of particle diameter and depends on the mean free path of the molecules comprising the air. 

```math
C_c = 1.0+\frac{λ}{D}\left(2.34+1.05\exp\left(-0.39\frac{D}{\lambda}\right)\right) 
```

where ``\lambda`` is the mean free path of air:
	
```math
\lambda = 6.6\times 10^{-8} \left(\frac{101315}{p} \right) \left (\frac{T}{293.15} \right)
```

which in turn is a function of temperature and pressure. The Reynolds number depends on the velocity, the density of the gas (``\rho_a``), and viscosity (``\eta``) of the gas. 

```math
Re = \frac{\rho_a v D}{\eta}
```

where ``\rho_a`` depends on temperature and pressure, and viscocity depends on temperature. The drag coefficient is approximately constant for large diameters and increases strongly with decreasing diameter. 

```math
Cd = \begin{cases}
24.0/Re \;\;\; Re < 0.1 \\
24.0/Re (1+0.15Re^{0.687}) \;\; else
\end{cases}
```

Solving the equation for ``v_t(D)`` requires a numerical solution since ``Re`` depends on velocity. The [fixed point iteration algorithm](https://en.wikipedia.org/wiki/Fixed-point_iteration) provides a method to solve for ``v_t(D)``. 

## Temperature, Pressure, and Particle Density 

Terminal velocity depends on pressure and temperature through the dependence of several of the underlying properties on temperature and pressure, includsing air density (``p = \frac{\rho}{R_dT}``), mean free path ``\lambda(T,p)`` and viscosity ``\eta(T)``. 
	
Terminal velocity also directly depends on particle density, due to the dependency of the force of gravity on particle mass. 

$(sld)
"""

end

# ╔═╡ 92a3457f-4c2f-428e-ba17-447725a81727
md"""
$(mplt(vars[1], vars[2], vars[3]))
**Figure 2.** Dependence of terminal velocity on particle diameter. The blue lines is for standard conditions defined as ``T = 20^\circ C`` and ``p = 1000\; hPa``.  Circles show tabulated values from tabulated values Tables 3.1 and 3.3 in Hinds. The red line shows the influence of temperature, pressure, and particle density based on the slider settings.

$(Markdown.MD(Markdown.Admonition("note", "Exercises", [md"
1. If cloud base is 1 km above an observer and precipitation falling from the cloud is to reach the observer, what is the minimum velocity/droplet size required to fall through the layer in a reasonable amount of time?
2. Hypothesize why decreasing pressure increases the fall velocity.
3. Hypothesize why increasing particle density increases the fall velocity.
"])))
"""

# ╔═╡ 982fac1f-98dd-401d-9c6f-f606bfcdc678
md"""
## Parameterizations of Terminal Velocity

Closed form parameterizations can be used for different size ranges to facilitate (1) faster computation of terminal velocity and (2) derivation of physical relationships that require a closed form solution for terminal velocity. Common parameterizations are:

```math
v_t = k_1 D^2\;\;\;\;\;\;\;\;\;\;\;\;\;\;\;\;\;\;\;\;\;\;\;\;\;\;\;\;\;\;\;\;\;\;\;\;\;\;\;    1\; \mu m < D < 0.1\; m m
```

```math
v_t = k_2 D^1\;\;\;\;\;\;\;\;\;\;\;\;\;\;\;\;\;\;\;\;\;\;\;\;\;\;\;\;\;\;\;\;\;\;\;\;  0.1\;mm < D < 1\; mm
```

```math
v_t = 9.65 - 10.3 \exp(-600D)\;\;\;\;\;\;\;\;\;\;\;\;\;  1\;mm < D < 10\; mm
```

where ``k_1 = \frac{1}{18}\frac{\rho_p g}{\eta} = 3.7\times10^7\; m^{-1} s^{-1}`` (assuming water) and ``k_2 = 4\times10^{-3}\; s^{-1}``

$(Markdown.MD(Markdown.Admonition("warning", "Key Proportionalities", [md"Note that the proprtionality on diameter switches from 
- ``v_t \propto D^2`` for ``D < 100\mu m``
- ``v_t \propto D`` for ``0.1 mm < D < 1 mm``
- ``v_t \propto \sqrt{D}`` for ``1 mm < D < 5 mm``
- ``v_t \approx const`` for ``D > 5 mm``
"])))

$(parameterization())

**Figure 3.**Parameterizations of settling velocity compared to the numerical solution at standard pressure and temperature for water droplets. The implemented numerical solution does not account for droplet deformation and is only valid to ~2 mm in diameter.    

$(Markdown.MD(Markdown.Admonition("note", "Exercises", [md"
1. List at least two advantages and two disadvantages of using these parameterizations for modeling terminal velocity.
2. For large hard spheres, ``C_d \approx 0.44`` and Newtons' theory predicts that ``v_t \propto \sqrt{D}``. Propose an explanation why terminal velocity approaches a constant for drops with ``D >> 5 mm``.
"])))

"""

# ╔═╡ 79f7d6ed-ed5c-46eb-b9cb-89e8b0e65cff
begin
	sld1 = @bind vars1 combine() do Child
		md"""
		``D\; (m)``   $(
			Child(Slider(exp10.(-9:1:-3), default = 1e-8, show_value = true))
		) 
		"""
	end
	
	md"""
	# Relaxation Time
		
	At force balance the drop continues to fall at its terminal velocity. After release, the velocity increase with time is
	
	```math
	v(t) = v_{ts} \left [1 - \exp \left(-\frac{t}{\tau} \right) \right ]
	```
	
	where ``v_{ts}`` is the terminal settling velocity, ``t`` is time, and ``\tau`` is the relaxation time. 
	
	```math
	\tau = \frac{\rho_p D^2 C_c(D)}{18 \eta}
	```
	
	At ``t \gtrapprox 5\tau``, the particle has reached ``v_{ts}``. The relaxation time strongly varies with particle size and is controlled by the ratio of intertial and viscous forces acting on the particle.

	$(sld1)
	"""
	
end

# ╔═╡ 19a8a974-c341-4860-9316-96560ece10f2
relaxation(vars1[1])	

# ╔═╡ 45d7697c-beac-437e-bf2f-7a3df5e66c2e
md"""
**Figure 4.** Increase in velocity with time of a droplet after release from rest. ``\tau`` is the relaxation time and ``v_t`` is the terminal velocity. The values in the legend vary with the input diameter. 

$(Markdown.MD(Markdown.Admonition("note", "Exercises", [md"
1. Estimate the time it takes for a 1 micron bacterium to reach terminal velocity.
2. Compute the fraction of terminal velocity that the particle has reached at ``t = \tau``."])))
"""

# ╔═╡ 97592d4d-79a0-4d75-a022-ea607c85b184
md"""
# Synthesis Assignment

$(Markdown.MD(Markdown.Admonition("tip", "Synthesis Assignment", [md"
1. Write a single computer finction in Julia, Python, R or MATLAB that computes ``v_t(D,v)``.
2. Apply this function and reproduce Figure 2. The tabulated velocity values are

| D (μm)  | v (m/s) |
|---------|---------|
| 0.001   | 6.9e-9  |
| 0.01    | 7.0e-8  |
| 0.1     | 8.8e-7  |
| 1       | 3.5e-5  |
| 10      | 0.0031  |
| 100     | 0.248   |
| 1000    | 3.86    |

3. On the graph, identify the continuum and kinetic regimes, the Stokes, transition, and Newtonian regimes, and the typical size of Aitken mode, accumulation mode, and coarse mode aerosols as well as cloud, drizzle, and raindrops.
4. Discuss why the terminal velocity is sensitive to pressure in some but not all regimes."])))
"""

# ╔═╡ 00000000-0000-0000-0000-000000000001
PLUTO_PROJECT_TOML_CONTENTS = """
[deps]
Colors = "5ae59095-9a9b-59fe-a467-6f913c188581"
Compose = "a81c6b42-2e10-5240-aca2-a61377ecd94b"
DataFrames = "a93c6f00-e57d-5684-b7b6-d8193f3e46c0"
Gadfly = "c91e804a-d5a3-530f-b6f0-dfbca275c004"
LaTeXStrings = "b964fa9f-0449-5b57-a5c2-d3ea65f4040f"
Logging = "56ddb016-857b-54e1-b83d-db4d58db5568"
NumericIO = "6c575b1c-77cb-5640-a5dc-a54116c90507"
PlutoUI = "7f904dfe-b85e-4ff6-b463-dae2292396a8"
Printf = "de0858da-6303-5e67-8744-51eddeeeb8d7"

[compat]
Colors = "~0.12.10"
Compose = "~0.9.5"
DataFrames = "~1.6.1"
Gadfly = "~1.4.0"
LaTeXStrings = "~1.3.1"
NumericIO = "~0.3.2"
PlutoUI = "~0.7.58"
"""

# ╔═╡ 00000000-0000-0000-0000-000000000002
PLUTO_MANIFEST_TOML_CONTENTS = """
# This file is machine-generated - editing it directly is not advised

julia_version = "1.10.2"
manifest_format = "2.0"
project_hash = "49ba8c766172d81dc90b88af0921c74be2c093ad"

[[deps.AbstractFFTs]]
deps = ["LinearAlgebra"]
git-tree-sha1 = "d92ad398961a3ed262d8bf04a1a2b8340f915fef"
uuid = "621f4979-c628-5d54-868e-fcf4e3e8185c"
version = "1.5.0"
weakdeps = ["ChainRulesCore", "Test"]

    [deps.AbstractFFTs.extensions]
    AbstractFFTsChainRulesCoreExt = "ChainRulesCore"
    AbstractFFTsTestExt = "Test"

[[deps.AbstractPlutoDingetjes]]
deps = ["Pkg"]
git-tree-sha1 = "c278dfab760520b8bb7e9511b968bf4ba38b7acc"
uuid = "6e696c72-6542-2067-7265-42206c756150"
version = "1.2.3"

[[deps.Adapt]]
deps = ["LinearAlgebra", "Requires"]
git-tree-sha1 = "0fb305e0253fd4e833d486914367a2ee2c2e78d0"
uuid = "79e6a3ab-5dfb-504d-930d-738a2a938a0e"
version = "4.0.1"
weakdeps = ["StaticArrays"]

    [deps.Adapt.extensions]
    AdaptStaticArraysExt = "StaticArrays"

[[deps.ArgTools]]
uuid = "0dad84c5-d112-42e6-8d28-ef12dabb789f"
version = "1.1.1"

[[deps.Artifacts]]
uuid = "56f22d72-fd6d-98f1-02f0-08ddc0907c33"

[[deps.AxisAlgorithms]]
deps = ["LinearAlgebra", "Random", "SparseArrays", "WoodburyMatrices"]
git-tree-sha1 = "01b8ccb13d68535d73d2b0c23e39bd23155fb712"
uuid = "13072b0f-2c55-5437-9ae7-d433b7a33950"
version = "1.1.0"

[[deps.Base64]]
uuid = "2a0f44e3-6c83-55bd-87e4-b1978d98bd5f"

[[deps.Calculus]]
deps = ["LinearAlgebra"]
git-tree-sha1 = "f641eb0a4f00c343bbc32346e1217b86f3ce9dad"
uuid = "49dc2e85-a5d0-5ad3-a950-438e2897f1b9"
version = "0.5.1"

[[deps.CategoricalArrays]]
deps = ["DataAPI", "Future", "Missings", "Printf", "Requires", "Statistics", "Unicode"]
git-tree-sha1 = "1568b28f91293458345dabba6a5ea3f183250a61"
uuid = "324d7699-5711-5eae-9e2f-1d82baa6b597"
version = "0.10.8"

    [deps.CategoricalArrays.extensions]
    CategoricalArraysJSONExt = "JSON"
    CategoricalArraysRecipesBaseExt = "RecipesBase"
    CategoricalArraysSentinelArraysExt = "SentinelArrays"
    CategoricalArraysStructTypesExt = "StructTypes"

    [deps.CategoricalArrays.weakdeps]
    JSON = "682c06a0-de6a-54ab-a142-c8b1cf79cde6"
    RecipesBase = "3cdcf5f2-1ef4-517c-9805-6587b60abb01"
    SentinelArrays = "91c51154-3ec4-41a3-a24f-3f23e20d615c"
    StructTypes = "856f2bd8-1eba-4b0a-8007-ebc267875bd4"

[[deps.ChainRulesCore]]
deps = ["Compat", "LinearAlgebra"]
git-tree-sha1 = "aef70bb349b20aa81a82a19704c3ef339d4ee494"
uuid = "d360d2e6-b24c-11e9-a2a3-2a2ae2dbcce4"
version = "1.22.1"
weakdeps = ["SparseArrays"]

    [deps.ChainRulesCore.extensions]
    ChainRulesCoreSparseArraysExt = "SparseArrays"

[[deps.ColorTypes]]
deps = ["FixedPointNumbers", "Random"]
git-tree-sha1 = "eb7f0f8307f71fac7c606984ea5fb2817275d6e4"
uuid = "3da002f7-5984-5a60-b8a6-cbb66c0b333f"
version = "0.11.4"

[[deps.Colors]]
deps = ["ColorTypes", "FixedPointNumbers", "Reexport"]
git-tree-sha1 = "fc08e5930ee9a4e03f84bfb5211cb54e7769758a"
uuid = "5ae59095-9a9b-59fe-a467-6f913c188581"
version = "0.12.10"

[[deps.Compat]]
deps = ["TOML", "UUIDs"]
git-tree-sha1 = "c955881e3c981181362ae4088b35995446298b80"
uuid = "34da2185-b29b-5c13-b0c7-acf172513d20"
version = "4.14.0"
weakdeps = ["Dates", "LinearAlgebra"]

    [deps.Compat.extensions]
    CompatLinearAlgebraExt = "LinearAlgebra"

[[deps.CompilerSupportLibraries_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "e66e0078-7015-5450-92f7-15fbd957f2ae"
version = "1.1.0+0"

[[deps.Compose]]
deps = ["Base64", "Colors", "DataStructures", "Dates", "IterTools", "JSON", "LinearAlgebra", "Measures", "Printf", "Random", "Requires", "Statistics", "UUIDs"]
git-tree-sha1 = "bf6570a34c850f99407b494757f5d7ad233a7257"
uuid = "a81c6b42-2e10-5240-aca2-a61377ecd94b"
version = "0.9.5"

[[deps.Contour]]
git-tree-sha1 = "d05d9e7b7aedff4e5b51a029dced05cfb6125781"
uuid = "d38c429a-6771-53c6-b99e-75d170b6e991"
version = "0.6.2"

[[deps.CoupledFields]]
deps = ["LinearAlgebra", "Statistics", "StatsBase"]
git-tree-sha1 = "6c9671364c68c1158ac2524ac881536195b7e7bc"
uuid = "7ad07ef1-bdf2-5661-9d2b-286fd4296dac"
version = "0.2.0"

[[deps.Crayons]]
git-tree-sha1 = "249fe38abf76d48563e2f4556bebd215aa317e15"
uuid = "a8cc5b0e-0ffa-5ad4-8c14-923d3ee1735f"
version = "4.1.1"

[[deps.DataAPI]]
git-tree-sha1 = "abe83f3a2f1b857aac70ef8b269080af17764bbe"
uuid = "9a962f9c-6df0-11e9-0e5d-c546b8b5ee8a"
version = "1.16.0"

[[deps.DataFrames]]
deps = ["Compat", "DataAPI", "DataStructures", "Future", "InlineStrings", "InvertedIndices", "IteratorInterfaceExtensions", "LinearAlgebra", "Markdown", "Missings", "PooledArrays", "PrecompileTools", "PrettyTables", "Printf", "REPL", "Random", "Reexport", "SentinelArrays", "SortingAlgorithms", "Statistics", "TableTraits", "Tables", "Unicode"]
git-tree-sha1 = "04c738083f29f86e62c8afc341f0967d8717bdb8"
uuid = "a93c6f00-e57d-5684-b7b6-d8193f3e46c0"
version = "1.6.1"

[[deps.DataStructures]]
deps = ["Compat", "InteractiveUtils", "OrderedCollections"]
git-tree-sha1 = "1fb174f0d48fe7d142e1109a10636bc1d14f5ac2"
uuid = "864edb3b-99cc-5e75-8d2d-829cb0a9cfe8"
version = "0.18.17"

[[deps.DataValueInterfaces]]
git-tree-sha1 = "bfc1187b79289637fa0ef6d4436ebdfe6905cbd6"
uuid = "e2d170a0-9d28-54be-80f0-106bbe20a464"
version = "1.0.0"

[[deps.Dates]]
deps = ["Printf"]
uuid = "ade2ca70-3891-5945-98fb-dc099432e06a"

[[deps.Distances]]
deps = ["LinearAlgebra", "Statistics", "StatsAPI"]
git-tree-sha1 = "66c4c81f259586e8f002eacebc177e1fb06363b0"
uuid = "b4f34e82-e78d-54a5-968a-f98e89d6e8f7"
version = "0.10.11"
weakdeps = ["ChainRulesCore", "SparseArrays"]

    [deps.Distances.extensions]
    DistancesChainRulesCoreExt = "ChainRulesCore"
    DistancesSparseArraysExt = "SparseArrays"

[[deps.Distributed]]
deps = ["Random", "Serialization", "Sockets"]
uuid = "8ba89e20-285c-5b6f-9357-94700520ee1b"

[[deps.Distributions]]
deps = ["FillArrays", "LinearAlgebra", "PDMats", "Printf", "QuadGK", "Random", "SpecialFunctions", "Statistics", "StatsAPI", "StatsBase", "StatsFuns"]
git-tree-sha1 = "7c302d7a5fec5214eb8a5a4c466dcf7a51fcf169"
uuid = "31c24e10-a181-5473-b8eb-7969acd0382f"
version = "0.25.107"

    [deps.Distributions.extensions]
    DistributionsChainRulesCoreExt = "ChainRulesCore"
    DistributionsDensityInterfaceExt = "DensityInterface"
    DistributionsTestExt = "Test"

    [deps.Distributions.weakdeps]
    ChainRulesCore = "d360d2e6-b24c-11e9-a2a3-2a2ae2dbcce4"
    DensityInterface = "b429d917-457f-4dbc-8f4c-0cc954292b1d"
    Test = "8dfed614-e22c-5e08-85e1-65c5234f0b40"

[[deps.DocStringExtensions]]
deps = ["LibGit2"]
git-tree-sha1 = "2fb1e02f2b635d0845df5d7c167fec4dd739b00d"
uuid = "ffbed154-4ef7-542d-bbb7-c09d3a79fcae"
version = "0.9.3"

[[deps.Downloads]]
deps = ["ArgTools", "FileWatching", "LibCURL", "NetworkOptions"]
uuid = "f43a241f-c20a-4ad4-852c-f6b1247861c6"
version = "1.6.0"

[[deps.DualNumbers]]
deps = ["Calculus", "NaNMath", "SpecialFunctions"]
git-tree-sha1 = "5837a837389fccf076445fce071c8ddaea35a566"
uuid = "fa6b7ba4-c1ee-5f82-b5fc-ecf0adba8f74"
version = "0.6.8"

[[deps.FFTW]]
deps = ["AbstractFFTs", "FFTW_jll", "LinearAlgebra", "MKL_jll", "Preferences", "Reexport"]
git-tree-sha1 = "4820348781ae578893311153d69049a93d05f39d"
uuid = "7a1cc6ca-52ef-59f5-83cd-3a7055c09341"
version = "1.8.0"

[[deps.FFTW_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "c6033cc3892d0ef5bb9cd29b7f2f0331ea5184ea"
uuid = "f5851436-0d7a-5f13-b9de-f02708fd171a"
version = "3.3.10+0"

[[deps.FileWatching]]
uuid = "7b1f6079-737a-58dc-b8bc-7a2ca5c1b5ee"

[[deps.FillArrays]]
deps = ["LinearAlgebra", "Random"]
git-tree-sha1 = "5b93957f6dcd33fc343044af3d48c215be2562f1"
uuid = "1a297f60-69ca-5386-bcde-b61e274b549b"
version = "1.9.3"
weakdeps = ["PDMats", "SparseArrays", "Statistics"]

    [deps.FillArrays.extensions]
    FillArraysPDMatsExt = "PDMats"
    FillArraysSparseArraysExt = "SparseArrays"
    FillArraysStatisticsExt = "Statistics"

[[deps.FixedPointNumbers]]
deps = ["Statistics"]
git-tree-sha1 = "335bfdceacc84c5cdf16aadc768aa5ddfc5383cc"
uuid = "53c48c17-4a7d-5ca2-90c5-79b7896eea93"
version = "0.8.4"

[[deps.Future]]
deps = ["Random"]
uuid = "9fa8497b-333b-5362-9e8d-4d0656e87820"

[[deps.Gadfly]]
deps = ["Base64", "CategoricalArrays", "Colors", "Compose", "Contour", "CoupledFields", "DataAPI", "DataStructures", "Dates", "Distributions", "DocStringExtensions", "Hexagons", "IndirectArrays", "IterTools", "JSON", "Juno", "KernelDensity", "LinearAlgebra", "Loess", "Measures", "Printf", "REPL", "Random", "Requires", "Showoff", "Statistics"]
git-tree-sha1 = "d546e18920e28505e9856e1dfc36cff066907c71"
uuid = "c91e804a-d5a3-530f-b6f0-dfbca275c004"
version = "1.4.0"

[[deps.Grisu]]
git-tree-sha1 = "53bb909d1151e57e2484c3d1b53e19552b887fb2"
uuid = "42e2da0e-8278-4e71-bc24-59509adca0fe"
version = "1.0.2"

[[deps.Hexagons]]
deps = ["Test"]
git-tree-sha1 = "de4a6f9e7c4710ced6838ca906f81905f7385fd6"
uuid = "a1b4810d-1bce-5fbd-ac56-80944d57a21f"
version = "0.2.0"

[[deps.HypergeometricFunctions]]
deps = ["DualNumbers", "LinearAlgebra", "OpenLibm_jll", "SpecialFunctions"]
git-tree-sha1 = "f218fe3736ddf977e0e772bc9a586b2383da2685"
uuid = "34004b35-14d8-5ef3-9330-4cdb6864b03a"
version = "0.3.23"

[[deps.Hyperscript]]
deps = ["Test"]
git-tree-sha1 = "179267cfa5e712760cd43dcae385d7ea90cc25a4"
uuid = "47d2ed2b-36de-50cf-bf87-49c2cf4b8b91"
version = "0.0.5"

[[deps.HypertextLiteral]]
deps = ["Tricks"]
git-tree-sha1 = "7134810b1afce04bbc1045ca1985fbe81ce17653"
uuid = "ac1192a8-f4b3-4bfe-ba22-af5b92cd3ab2"
version = "0.9.5"

[[deps.IOCapture]]
deps = ["Logging", "Random"]
git-tree-sha1 = "8b72179abc660bfab5e28472e019392b97d0985c"
uuid = "b5f81e59-6552-4d32-b1f0-c071b021bf89"
version = "0.2.4"

[[deps.IndirectArrays]]
git-tree-sha1 = "012e604e1c7458645cb8b436f8fba789a51b257f"
uuid = "9b13fd28-a010-5f03-acff-a1bbcff69959"
version = "1.0.0"

[[deps.InlineStrings]]
deps = ["Parsers"]
git-tree-sha1 = "9cc2baf75c6d09f9da536ddf58eb2f29dedaf461"
uuid = "842dd82b-1e85-43dc-bf29-5d0ee9dffc48"
version = "1.4.0"

[[deps.IntelOpenMP_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "5fdf2fe6724d8caabf43b557b84ce53f3b7e2f6b"
uuid = "1d5cc7b8-4909-519e-a0f8-d0f5ad9712d0"
version = "2024.0.2+0"

[[deps.InteractiveUtils]]
deps = ["Markdown"]
uuid = "b77e0a4c-d291-57a0-90e8-8db25a27a240"

[[deps.Interpolations]]
deps = ["Adapt", "AxisAlgorithms", "ChainRulesCore", "LinearAlgebra", "OffsetArrays", "Random", "Ratios", "Requires", "SharedArrays", "SparseArrays", "StaticArrays", "WoodburyMatrices"]
git-tree-sha1 = "88a101217d7cb38a7b481ccd50d21876e1d1b0e0"
uuid = "a98d9a8b-a2ab-59e6-89dd-64a1c18fca59"
version = "0.15.1"

    [deps.Interpolations.extensions]
    InterpolationsUnitfulExt = "Unitful"

    [deps.Interpolations.weakdeps]
    Unitful = "1986cc42-f94f-5a68-af5c-568840ba703d"

[[deps.InvertedIndices]]
git-tree-sha1 = "0dc7b50b8d436461be01300fd8cd45aa0274b038"
uuid = "41ab1584-1d38-5bbf-9106-f11c6c58b48f"
version = "1.3.0"

[[deps.IrrationalConstants]]
git-tree-sha1 = "630b497eafcc20001bba38a4651b327dcfc491d2"
uuid = "92d709cd-6900-40b7-9082-c6be49f344b6"
version = "0.2.2"

[[deps.IterTools]]
git-tree-sha1 = "42d5f897009e7ff2cf88db414a389e5ed1bdd023"
uuid = "c8e1da08-722c-5040-9ed9-7db0dc04731e"
version = "1.10.0"

[[deps.IteratorInterfaceExtensions]]
git-tree-sha1 = "a3f24677c21f5bbe9d2a714f95dcd58337fb2856"
uuid = "82899510-4779-5014-852e-03e436cf321d"
version = "1.0.0"

[[deps.JLLWrappers]]
deps = ["Artifacts", "Preferences"]
git-tree-sha1 = "7e5d6779a1e09a36db2a7b6cff50942a0a7d0fca"
uuid = "692b3bcd-3c85-4b1f-b108-f13ce0eb3210"
version = "1.5.0"

[[deps.JSON]]
deps = ["Dates", "Mmap", "Parsers", "Unicode"]
git-tree-sha1 = "31e996f0a15c7b280ba9f76636b3ff9e2ae58c9a"
uuid = "682c06a0-de6a-54ab-a142-c8b1cf79cde6"
version = "0.21.4"

[[deps.Juno]]
deps = ["Base64", "Logging", "Media", "Profile"]
git-tree-sha1 = "07cb43290a840908a771552911a6274bc6c072c7"
uuid = "e5e0dc1b-0480-54bc-9374-aad01c23163d"
version = "0.8.4"

[[deps.KernelDensity]]
deps = ["Distributions", "DocStringExtensions", "FFTW", "Interpolations", "StatsBase"]
git-tree-sha1 = "fee018a29b60733876eb557804b5b109dd3dd8a7"
uuid = "5ab0869b-81aa-558d-bb23-cbf5423bbe9b"
version = "0.6.8"

[[deps.LaTeXStrings]]
git-tree-sha1 = "50901ebc375ed41dbf8058da26f9de442febbbec"
uuid = "b964fa9f-0449-5b57-a5c2-d3ea65f4040f"
version = "1.3.1"

[[deps.LazyArtifacts]]
deps = ["Artifacts", "Pkg"]
uuid = "4af54fe1-eca0-43a8-85a7-787d91b784e3"

[[deps.LibCURL]]
deps = ["LibCURL_jll", "MozillaCACerts_jll"]
uuid = "b27032c2-a3e7-50c8-80cd-2d36dbcbfd21"
version = "0.6.4"

[[deps.LibCURL_jll]]
deps = ["Artifacts", "LibSSH2_jll", "Libdl", "MbedTLS_jll", "Zlib_jll", "nghttp2_jll"]
uuid = "deac9b47-8bc7-5906-a0fe-35ac56dc84c0"
version = "8.4.0+0"

[[deps.LibGit2]]
deps = ["Base64", "LibGit2_jll", "NetworkOptions", "Printf", "SHA"]
uuid = "76f85450-5226-5b5a-8eaa-529ad045b433"

[[deps.LibGit2_jll]]
deps = ["Artifacts", "LibSSH2_jll", "Libdl", "MbedTLS_jll"]
uuid = "e37daf67-58a4-590a-8e99-b0245dd2ffc5"
version = "1.6.4+0"

[[deps.LibSSH2_jll]]
deps = ["Artifacts", "Libdl", "MbedTLS_jll"]
uuid = "29816b5a-b9ab-546f-933c-edad1886dfa8"
version = "1.11.0+1"

[[deps.Libdl]]
uuid = "8f399da3-3557-5675-b5ff-fb832c97cbdb"

[[deps.LinearAlgebra]]
deps = ["Libdl", "OpenBLAS_jll", "libblastrampoline_jll"]
uuid = "37e2e46d-f89d-539d-b4ee-838fcccc9c8e"

[[deps.Loess]]
deps = ["Distances", "LinearAlgebra", "Statistics", "StatsAPI"]
git-tree-sha1 = "a113a8be4c6d0c64e217b472fb6e61c760eb4022"
uuid = "4345ca2d-374a-55d4-8d30-97f9976e7612"
version = "0.6.3"

[[deps.LogExpFunctions]]
deps = ["DocStringExtensions", "IrrationalConstants", "LinearAlgebra"]
git-tree-sha1 = "18144f3e9cbe9b15b070288eef858f71b291ce37"
uuid = "2ab3a3ac-af41-5b50-aa03-7779005ae688"
version = "0.3.27"

    [deps.LogExpFunctions.extensions]
    LogExpFunctionsChainRulesCoreExt = "ChainRulesCore"
    LogExpFunctionsChangesOfVariablesExt = "ChangesOfVariables"
    LogExpFunctionsInverseFunctionsExt = "InverseFunctions"

    [deps.LogExpFunctions.weakdeps]
    ChainRulesCore = "d360d2e6-b24c-11e9-a2a3-2a2ae2dbcce4"
    ChangesOfVariables = "9e997f8a-9a97-42d5-a9f1-ce6bfc15e2c0"
    InverseFunctions = "3587e190-3f89-42d0-90ee-14403ec27112"

[[deps.Logging]]
uuid = "56ddb016-857b-54e1-b83d-db4d58db5568"

[[deps.MIMEs]]
git-tree-sha1 = "65f28ad4b594aebe22157d6fac869786a255b7eb"
uuid = "6c6e2e6c-3030-632d-7369-2d6c69616d65"
version = "0.1.4"

[[deps.MKL_jll]]
deps = ["Artifacts", "IntelOpenMP_jll", "JLLWrappers", "LazyArtifacts", "Libdl"]
git-tree-sha1 = "72dc3cf284559eb8f53aa593fe62cb33f83ed0c0"
uuid = "856f044c-d86e-5d09-b602-aeab76dc8ba7"
version = "2024.0.0+0"

[[deps.MacroTools]]
deps = ["Markdown", "Random"]
git-tree-sha1 = "2fa9ee3e63fd3a4f7a9a4f4744a52f4856de82df"
uuid = "1914dd2f-81c6-5fcd-8719-6d5c9610ff09"
version = "0.5.13"

[[deps.Markdown]]
deps = ["Base64"]
uuid = "d6f4376e-aef5-505a-96c1-9c027394607a"

[[deps.MbedTLS_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "c8ffd9c3-330d-5841-b78e-0817d7145fa1"
version = "2.28.2+1"

[[deps.Measures]]
git-tree-sha1 = "c13304c81eec1ed3af7fc20e75fb6b26092a1102"
uuid = "442fdcdd-2543-5da2-b0f3-8c86c306513e"
version = "0.3.2"

[[deps.Media]]
deps = ["MacroTools", "Test"]
git-tree-sha1 = "75a54abd10709c01f1b86b84ec225d26e840ed58"
uuid = "e89f7d12-3494-54d1-8411-f7d8b9ae1f27"
version = "0.5.0"

[[deps.Missings]]
deps = ["DataAPI"]
git-tree-sha1 = "f66bdc5de519e8f8ae43bdc598782d35a25b1272"
uuid = "e1d29d7a-bbdc-5cf2-9ac0-f12de2c33e28"
version = "1.1.0"

[[deps.Mmap]]
uuid = "a63ad114-7e13-5084-954f-fe012c677804"

[[deps.MozillaCACerts_jll]]
uuid = "14a3606d-f60d-562e-9121-12d972cd8159"
version = "2023.1.10"

[[deps.NaNMath]]
deps = ["OpenLibm_jll"]
git-tree-sha1 = "0877504529a3e5c3343c6f8b4c0381e57e4387e4"
uuid = "77ba4419-2d1f-58cd-9bb1-8ffee604a2e3"
version = "1.0.2"

[[deps.NetworkOptions]]
uuid = "ca575930-c2e3-43a9-ace4-1e988b2c1908"
version = "1.2.0"

[[deps.NumericIO]]
deps = ["Printf"]
git-tree-sha1 = "5e2bd9bee8b55b754ca61386df207abbfc266ef6"
uuid = "6c575b1c-77cb-5640-a5dc-a54116c90507"
version = "0.3.2"

[[deps.OffsetArrays]]
git-tree-sha1 = "6a731f2b5c03157418a20c12195eb4b74c8f8621"
uuid = "6fe1bfb0-de20-5000-8ca7-80f57d26f881"
version = "1.13.0"
weakdeps = ["Adapt"]

    [deps.OffsetArrays.extensions]
    OffsetArraysAdaptExt = "Adapt"

[[deps.OpenBLAS_jll]]
deps = ["Artifacts", "CompilerSupportLibraries_jll", "Libdl"]
uuid = "4536629a-c528-5b80-bd46-f80d51c5b363"
version = "0.3.23+4"

[[deps.OpenLibm_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "05823500-19ac-5b8b-9628-191a04bc5112"
version = "0.8.1+2"

[[deps.OpenSpecFun_jll]]
deps = ["Artifacts", "CompilerSupportLibraries_jll", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "13652491f6856acfd2db29360e1bbcd4565d04f1"
uuid = "efe28fd5-8261-553b-a9e1-b2916fc3738e"
version = "0.5.5+0"

[[deps.OrderedCollections]]
git-tree-sha1 = "dfdf5519f235516220579f949664f1bf44e741c5"
uuid = "bac558e1-5e72-5ebc-8fee-abe8a469f55d"
version = "1.6.3"

[[deps.PDMats]]
deps = ["LinearAlgebra", "SparseArrays", "SuiteSparse"]
git-tree-sha1 = "949347156c25054de2db3b166c52ac4728cbad65"
uuid = "90014a1f-27ba-587c-ab20-58faa44d9150"
version = "0.11.31"

[[deps.Parsers]]
deps = ["Dates", "PrecompileTools", "UUIDs"]
git-tree-sha1 = "8489905bcdbcfac64d1daa51ca07c0d8f0283821"
uuid = "69de0a69-1ddd-5017-9359-2bf0b02dc9f0"
version = "2.8.1"

[[deps.Pkg]]
deps = ["Artifacts", "Dates", "Downloads", "FileWatching", "LibGit2", "Libdl", "Logging", "Markdown", "Printf", "REPL", "Random", "SHA", "Serialization", "TOML", "Tar", "UUIDs", "p7zip_jll"]
uuid = "44cfe95a-1eb2-52ea-b672-e2afdf69b78f"
version = "1.10.0"

[[deps.PlutoUI]]
deps = ["AbstractPlutoDingetjes", "Base64", "ColorTypes", "Dates", "FixedPointNumbers", "Hyperscript", "HypertextLiteral", "IOCapture", "InteractiveUtils", "JSON", "Logging", "MIMEs", "Markdown", "Random", "Reexport", "URIs", "UUIDs"]
git-tree-sha1 = "71a22244e352aa8c5f0f2adde4150f62368a3f2e"
uuid = "7f904dfe-b85e-4ff6-b463-dae2292396a8"
version = "0.7.58"

[[deps.PooledArrays]]
deps = ["DataAPI", "Future"]
git-tree-sha1 = "36d8b4b899628fb92c2749eb488d884a926614d3"
uuid = "2dfb63ee-cc39-5dd5-95bd-886bf059d720"
version = "1.4.3"

[[deps.PrecompileTools]]
deps = ["Preferences"]
git-tree-sha1 = "03b4c25b43cb84cee5c90aa9b5ea0a78fd848d2f"
uuid = "aea7be01-6a6a-4083-8856-8a6e6704d82a"
version = "1.2.0"

[[deps.Preferences]]
deps = ["TOML"]
git-tree-sha1 = "00805cd429dcb4870060ff49ef443486c262e38e"
uuid = "21216c6a-2e73-6563-6e65-726566657250"
version = "1.4.1"

[[deps.PrettyTables]]
deps = ["Crayons", "LaTeXStrings", "Markdown", "PrecompileTools", "Printf", "Reexport", "StringManipulation", "Tables"]
git-tree-sha1 = "88b895d13d53b5577fd53379d913b9ab9ac82660"
uuid = "08abe8d2-0d0c-5749-adfa-8a2ac140af0d"
version = "2.3.1"

[[deps.Printf]]
deps = ["Unicode"]
uuid = "de0858da-6303-5e67-8744-51eddeeeb8d7"

[[deps.Profile]]
deps = ["Printf"]
uuid = "9abbd945-dff8-562f-b5e8-e1ebf5ef1b79"

[[deps.QuadGK]]
deps = ["DataStructures", "LinearAlgebra"]
git-tree-sha1 = "9b23c31e76e333e6fb4c1595ae6afa74966a729e"
uuid = "1fd47b50-473d-5c70-9696-f719f8f3bcdc"
version = "2.9.4"

[[deps.REPL]]
deps = ["InteractiveUtils", "Markdown", "Sockets", "Unicode"]
uuid = "3fa0cd96-eef1-5676-8a61-b3b8758bbffb"

[[deps.Random]]
deps = ["SHA"]
uuid = "9a3f8284-a2c9-5f02-9a11-845980a1fd5c"

[[deps.Ratios]]
deps = ["Requires"]
git-tree-sha1 = "1342a47bf3260ee108163042310d26f2be5ec90b"
uuid = "c84ed2f1-dad5-54f0-aa8e-dbefe2724439"
version = "0.4.5"
weakdeps = ["FixedPointNumbers"]

    [deps.Ratios.extensions]
    RatiosFixedPointNumbersExt = "FixedPointNumbers"

[[deps.Reexport]]
git-tree-sha1 = "45e428421666073eab6f2da5c9d310d99bb12f9b"
uuid = "189a3867-3050-52da-a836-e630ba90ab69"
version = "1.2.2"

[[deps.Requires]]
deps = ["UUIDs"]
git-tree-sha1 = "838a3a4188e2ded87a4f9f184b4b0d78a1e91cb7"
uuid = "ae029012-a4dd-5104-9daa-d747884805df"
version = "1.3.0"

[[deps.Rmath]]
deps = ["Random", "Rmath_jll"]
git-tree-sha1 = "f65dcb5fa46aee0cf9ed6274ccbd597adc49aa7b"
uuid = "79098fc4-a85e-5d69-aa6a-4863f24498fa"
version = "0.7.1"

[[deps.Rmath_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "6ed52fdd3382cf21947b15e8870ac0ddbff736da"
uuid = "f50d1b31-88e8-58de-be2c-1cc44531875f"
version = "0.4.0+0"

[[deps.SHA]]
uuid = "ea8e919c-243c-51af-8825-aaa63cd721ce"
version = "0.7.0"

[[deps.SentinelArrays]]
deps = ["Dates", "Random"]
git-tree-sha1 = "0e7508ff27ba32f26cd459474ca2ede1bc10991f"
uuid = "91c51154-3ec4-41a3-a24f-3f23e20d615c"
version = "1.4.1"

[[deps.Serialization]]
uuid = "9e88b42a-f829-5b0c-bbe9-9e923198166b"

[[deps.SharedArrays]]
deps = ["Distributed", "Mmap", "Random", "Serialization"]
uuid = "1a1011a3-84de-559e-8e89-a11a2f7dc383"

[[deps.Showoff]]
deps = ["Dates", "Grisu"]
git-tree-sha1 = "91eddf657aca81df9ae6ceb20b959ae5653ad1de"
uuid = "992d4aef-0814-514b-bc4d-f2e9a6c4116f"
version = "1.0.3"

[[deps.Sockets]]
uuid = "6462fe0b-24de-5631-8697-dd941f90decc"

[[deps.SortingAlgorithms]]
deps = ["DataStructures"]
git-tree-sha1 = "66e0a8e672a0bdfca2c3f5937efb8538b9ddc085"
uuid = "a2af1166-a08f-5f64-846c-94a0d3cef48c"
version = "1.2.1"

[[deps.SparseArrays]]
deps = ["Libdl", "LinearAlgebra", "Random", "Serialization", "SuiteSparse_jll"]
uuid = "2f01184e-e22b-5df5-ae63-d93ebab69eaf"
version = "1.10.0"

[[deps.SpecialFunctions]]
deps = ["IrrationalConstants", "LogExpFunctions", "OpenLibm_jll", "OpenSpecFun_jll"]
git-tree-sha1 = "e2cfc4012a19088254b3950b85c3c1d8882d864d"
uuid = "276daf66-3868-5448-9aa4-cd146d93841b"
version = "2.3.1"
weakdeps = ["ChainRulesCore"]

    [deps.SpecialFunctions.extensions]
    SpecialFunctionsChainRulesCoreExt = "ChainRulesCore"

[[deps.StaticArrays]]
deps = ["LinearAlgebra", "PrecompileTools", "Random", "StaticArraysCore"]
git-tree-sha1 = "bf074c045d3d5ffd956fa0a461da38a44685d6b2"
uuid = "90137ffa-7385-5640-81b9-e52037218182"
version = "1.9.3"
weakdeps = ["ChainRulesCore", "Statistics"]

    [deps.StaticArrays.extensions]
    StaticArraysChainRulesCoreExt = "ChainRulesCore"
    StaticArraysStatisticsExt = "Statistics"

[[deps.StaticArraysCore]]
git-tree-sha1 = "36b3d696ce6366023a0ea192b4cd442268995a0d"
uuid = "1e83bf80-4336-4d27-bf5d-d5a4f845583c"
version = "1.4.2"

[[deps.Statistics]]
deps = ["LinearAlgebra", "SparseArrays"]
uuid = "10745b16-79ce-11e8-11f9-7d13ad32a3b2"
version = "1.10.0"

[[deps.StatsAPI]]
deps = ["LinearAlgebra"]
git-tree-sha1 = "1ff449ad350c9c4cbc756624d6f8a8c3ef56d3ed"
uuid = "82ae8749-77ed-4fe6-ae5f-f523153014b0"
version = "1.7.0"

[[deps.StatsBase]]
deps = ["DataAPI", "DataStructures", "LinearAlgebra", "LogExpFunctions", "Missings", "Printf", "Random", "SortingAlgorithms", "SparseArrays", "Statistics", "StatsAPI"]
git-tree-sha1 = "d1bf48bfcc554a3761a133fe3a9bb01488e06916"
uuid = "2913bbd2-ae8a-5f71-8c99-4fb6c76f3a91"
version = "0.33.21"

[[deps.StatsFuns]]
deps = ["HypergeometricFunctions", "IrrationalConstants", "LogExpFunctions", "Reexport", "Rmath", "SpecialFunctions"]
git-tree-sha1 = "cef0472124fab0695b58ca35a77c6fb942fdab8a"
uuid = "4c63d2b9-4356-54db-8cca-17b64c39e42c"
version = "1.3.1"

    [deps.StatsFuns.extensions]
    StatsFunsChainRulesCoreExt = "ChainRulesCore"
    StatsFunsInverseFunctionsExt = "InverseFunctions"

    [deps.StatsFuns.weakdeps]
    ChainRulesCore = "d360d2e6-b24c-11e9-a2a3-2a2ae2dbcce4"
    InverseFunctions = "3587e190-3f89-42d0-90ee-14403ec27112"

[[deps.StringManipulation]]
deps = ["PrecompileTools"]
git-tree-sha1 = "a04cabe79c5f01f4d723cc6704070ada0b9d46d5"
uuid = "892a3eda-7b42-436c-8928-eab12a02cf0e"
version = "0.3.4"

[[deps.SuiteSparse]]
deps = ["Libdl", "LinearAlgebra", "Serialization", "SparseArrays"]
uuid = "4607b0f0-06f3-5cda-b6b1-a6196a1729e9"

[[deps.SuiteSparse_jll]]
deps = ["Artifacts", "Libdl", "libblastrampoline_jll"]
uuid = "bea87d4a-7f5b-5778-9afe-8cc45184846c"
version = "7.2.1+1"

[[deps.TOML]]
deps = ["Dates"]
uuid = "fa267f1f-6049-4f14-aa54-33bafae1ed76"
version = "1.0.3"

[[deps.TableTraits]]
deps = ["IteratorInterfaceExtensions"]
git-tree-sha1 = "c06b2f539df1c6efa794486abfb6ed2022561a39"
uuid = "3783bdb8-4a98-5b6b-af9a-565f29a5fe9c"
version = "1.0.1"

[[deps.Tables]]
deps = ["DataAPI", "DataValueInterfaces", "IteratorInterfaceExtensions", "LinearAlgebra", "OrderedCollections", "TableTraits"]
git-tree-sha1 = "cb76cf677714c095e535e3501ac7954732aeea2d"
uuid = "bd369af6-aec1-5ad0-b16a-f7cc5008161c"
version = "1.11.1"

[[deps.Tar]]
deps = ["ArgTools", "SHA"]
uuid = "a4e569a6-e804-4fa4-b0f3-eef7a1d5b13e"
version = "1.10.0"

[[deps.Test]]
deps = ["InteractiveUtils", "Logging", "Random", "Serialization"]
uuid = "8dfed614-e22c-5e08-85e1-65c5234f0b40"

[[deps.Tricks]]
git-tree-sha1 = "eae1bb484cd63b36999ee58be2de6c178105112f"
uuid = "410a4b4d-49e4-4fbc-ab6d-cb71b17b3775"
version = "0.1.8"

[[deps.URIs]]
git-tree-sha1 = "67db6cc7b3821e19ebe75791a9dd19c9b1188f2b"
uuid = "5c2747f8-b7ea-4ff2-ba2e-563bfd36b1d4"
version = "1.5.1"

[[deps.UUIDs]]
deps = ["Random", "SHA"]
uuid = "cf7118a7-6976-5b1a-9a39-7adc72f591a4"

[[deps.Unicode]]
uuid = "4ec0a83e-493e-50e2-b9ac-8f72acf5a8f5"

[[deps.WoodburyMatrices]]
deps = ["LinearAlgebra", "SparseArrays"]
git-tree-sha1 = "c1a7aa6219628fcd757dede0ca95e245c5cd9511"
uuid = "efce3f68-66dc-5838-9240-27a6d6f5f9b6"
version = "1.0.0"

[[deps.Zlib_jll]]
deps = ["Libdl"]
uuid = "83775a58-1f1d-513f-b197-d71354ab007a"
version = "1.2.13+1"

[[deps.libblastrampoline_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "8e850b90-86db-534c-a0d3-1478176c7d93"
version = "5.8.0+1"

[[deps.nghttp2_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "8e850ede-7688-5339-a07c-302acd2aaf8d"
version = "1.52.0+1"

[[deps.p7zip_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "3f19e933-33d8-53b3-aaab-bd5110c3b7a0"
version = "17.4.0+2"
"""

# ╔═╡ Cell order:
# ╟─e752f5da-d4b7-11ee-039b-e1e972398564
# ╟─74f4d2ab-7b4c-4673-aadf-67deab736052
# ╟─d2107caa-d368-4735-9a1f-7d4018e7032c
# ╟─92a3457f-4c2f-428e-ba17-447725a81727
# ╟─982fac1f-98dd-401d-9c6f-f606bfcdc678
# ╟─79f7d6ed-ed5c-46eb-b9cb-89e8b0e65cff
# ╟─19a8a974-c341-4860-9316-96560ece10f2
# ╟─45d7697c-beac-437e-bf2f-7a3df5e66c2e
# ╟─97592d4d-79a0-4d75-a022-ea607c85b184
# ╟─00000000-0000-0000-0000-000000000001
# ╟─00000000-0000-0000-0000-000000000002
