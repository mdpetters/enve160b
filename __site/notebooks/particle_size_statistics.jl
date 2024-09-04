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

# ╔═╡ 70924856-fa28-4ca2-b024-95664a461213
begin
	using PlutoUI
	using Gadfly	
	using LinearAlgebra
	import PlutoUI: combine
	using LaTeXStrings
	using DataFrames
	using Colors
	using Printf
	using CSV
	using DataFrames
	using HTTP
	using Logging

	Logging.disable_logging(Logging.Warn)

	la_url = "https://mdpetters.github.io/cee233/assets/Los_Angeles_Pollution.jpg" 
	pops1_url = "https://mdpetters.github.io/cee233/assets/pops_schematic.png"
	pops2_url = "https://mdpetters.github.io/cee233/assets/pops.png"
	
    smps1_url = "https://mdpetters.github.io/cee233/assets/smps_schematic.png"
	smps2_url = "https://mdpetters.github.io/cee233/assets/smps.jpg"

	hfdma_url = "https://mdpetters.github.io/cee233/assets/hfdma.csv"
	lhfdma_url = "https://mdpetters.github.io/cee233/assets/lhfdma.csv"
	uhfdma_url = "https://mdpetters.github.io/cee233/assets/uhfdma.csv"
	rdma_url = "https://mdpetters.github.io/cee233/assets/rdma.csv"
	lrdma_url = "https://mdpetters.github.io/cee233/assets/lrdma.csv"
	urdma_url = "https://mdpetters.github.io/cee233/assets/urdma.csv"
	pops_url ="https://mdpetters.github.io/cee233/assets/pops.csv"
	lpops_url = "https://mdpetters.github.io/cee233/assets/lpops.csv"
	upops_url = "https://mdpetters.github.io/cee233/assets/upops.csv"

	md"""
	$(TableOfContents(depth=5))
	# Particle Size Statistics
	"""
end

# ╔═╡ 4c2859cd-b599-41a5-a5eb-e34c6c35a74f
md"""
$(Markdown.MD(
	Markdown.Admonition("warning", "Definition if an Aerosol", [md"An aerosol is a collection of airborne solid or liquid particles, with a typical size between 10 and 1000 nm. Aerosols may be of either natural or anthropogenic origin. Aerosols are sometimes, but not always visible as haze or smog."])))

$(Resource(la_url))
**Figure 1.** Smog over Los Angeles. Photo by David Iliff. License: CC BY-SA 3.0.

Aerosols have acute and chronic negative impacts on human health and the environment. Aerosols scatter (redirect) light, which reduces visibility. The particles serve as condensation sites for water, thereby influencing the properties of clouds.

The aerosol size distribution is one of the most critical properties that determines how aerosol interacts with the body, the atmosphere, and light. In addition to helping assess aerosol impacts on the environment, the particle size distributions also provide information about the sources and age of particles.

## Tabular Represenation
"""

# ╔═╡ f1eed054-7d08-4952-b739-d0cfc3b2549a
begin
	mutable struct SizeDistribution
	    A::Any                        # Input parameters [[N1,Dg1,σg1], ...] or DMA
	    De::Vector{<:AbstractFloat}   # bin edges
	    Dp::Vector{<:AbstractFloat}   # bin midpoints
	    ΔlnD::Vector{<:AbstractFloat} # ΔlnD of the grid
	    S::Vector{<:AbstractFloat}    # spectral density
	    N::Vector{<:AbstractFloat}    # number concentration per bin
	    form::Symbol                  # form of the size distribution 
    end
	md(A, x) = @. A[1] / (√(2π) * log(A[3])) * exp(-(log(x / A[2]))^2 / (2log(A[3])^2))
	logn(A, x) = mapreduce((A) -> md(A, x), +, A)
	
	function lognormal(A; d1 = 8.0, d2 = 2000.0, bins = 256)
	    De = 10.0 .^ range(log10(d1), stop = log10(d2), length = bins + 1)
	    Dp = sqrt.(De[2:end] .* De[1:end-1])
	    ΔlnD = log.(De[2:end] ./ De[1:end-1])
	    S = logn(A, Dp)
	    N = S .* ΔlnD
	    return SizeDistribution(A, De, Dp, ΔlnD, S, N, :lognormal)
	end
	
	function aerosol_table1()
	    𝕟 = lognormal([[200.0, 80.0, 1.2]]; d1 = 30.0, d2 = 300.0, bins = 10)
	
	    Dp = (𝕟.De[2:end]+𝕟.De[1:end-1])/2.0
	    Dlow = round.(Int,𝕟.De[1:end-1])
	    Dup = round.(Int,𝕟.De[2:end])
	    N = round.(Int,𝕟.N)
	    S=π.*(Dp.*1e-3).^2.0.*N
	    V=π./6.0*(Dp.*1e-3).^3.0.*N
	    M=π./6.0*(Dp.*1e-3).^3.0.*N.*2.0
	
	    df = DataFrame(Dlow=Dlow,Dup=Dup,N=N)
	end

	function aerosol_table2()
	    𝕟 = lognormal([[200.0, 80.0, 1.2]]; d1 = 30.0, d2 = 300.0, bins = 10)
	
	    Dp = (𝕟.De[2:end]+𝕟.De[1:end-1])/2.0
	    Dlow = round.(Int,𝕟.De[1:end-1])
	    Dup = round.(Int,𝕟.De[2:end])
	    N = round.(Int,𝕟.N)
	    S=π.*(Dp.*1e-3).^2.0.*N
	    V=π./6.0*(Dp.*1e-3).^3.0.*N
	    M=π./6.0*(Dp.*1e-3).^3.0.*N.*2.0
	
	    df = DataFrame(Dlow=Dlow,Dup=Dup,N=N,SpectralDensity=[nothing for i = 1:10])
	end

	function aerosol_table3()
	    𝕟 = lognormal([[200.0, 80.0, 1.2]]; d1 = 30.0, d2 = 300.0, bins = 10)
	
	    D1 = round.(Int,(𝕟.De[2:end]+𝕟.De[1:end-1])/2.0)
	    Dlow = round.(Int,𝕟.De[1:end-1])
	    Dup = round.(Int,𝕟.De[2:end])
	    N = round.(Int,𝕟.N)
	    S1=round.(π.*(D1.*1e-3).^2.0.*N,digits=3)
	    V1=round.(π./6.0*(D1.*1e-3).^3.0.*N,digits=3)
	    M1=round.(π./6.0*(D1.*1e-3).^3.0.*N.*2.0, digits = 4)
	
	    Dp = Array{Union{Nothing,Float64}}(undef, 10)
	    S = Array{Union{Nothing,Float64}}(undef, 10)
	    V = Array{Union{Nothing,Float64}}(undef, 10)
	    M = Array{Union{Nothing,Float64}}(undef, 10)
	    Dp[1:3] = D1[1:3]
	    S[1:3] = S1[1:3]
	    V[1:3] = V1[1:3]
	    M[1:3] = M1[1:3]
	    Dp[7:end] = D1[7:end]
	    S[7:end] = S1[7:end]
	    V[7:end] = V1[7:end]
	    M[7:end] = M1[7:end]
	
	    df = DataFrame(Dlow=Dlow,Dup=Dup,Dp=Dp,N=N,S=S,V=V,M=M)
	end

	function aerosol_figure1()
	    Gadfly.set_default_plot_size(18Gadfly.cm, 7Gadfly.cm)
	    colors = ["black", "darkred", "steelblue3"]
	
	    𝕟 = lognormal([[200, 80, 1.2]]; d1 = 30.0, d2 = 300.0, bins = 10); 
	
	    plot(xmin = 𝕟.De[1:end-1], xmax = 𝕟.De[2:end], y = 𝕟.N, Geom.bar, 
	         Theme(default_color="steelblue3"),
	         Guide.xlabel("Particle diameter (nm)"),
	         Guide.ylabel("Number concentration (cm⁻³)"),
	         Guide.xticks(ticks = 0:50:300),
	         Coord.cartesian(xmin=0, xmax=300, ymax = 100)) 
	end

	function aerosol_figure2()
	    Gadfly.set_default_plot_size(18Gadfly.cm, 8Gadfly.cm)
	    colors = ["black", "darkred", "steelblue3"]
	
	    𝕟 = lognormal([[200, 80, 1.2]]; d1 = 30.0, d2 = 300.0, bins = 10); 
	
	    p1 = plot(xmin = 𝕟.De[1:end-1], xmax = 𝕟.De[2:end], y = 𝕟.N, Geom.bar, 
	            Theme(default_color="steelblue3"),
	            Guide.xlabel("Particle diameter (nm)"),
	            Guide.ylabel("Number concentration (cm⁻³)"),
	            Guide.title("Histogram Representation 10 Bins"),
	            Guide.xticks(ticks = 0:50:300),
	            Coord.cartesian(xmin=0, xmax=300, ymax = 100)) 
	
	    𝕟 = lognormal([[200, 80, 1.2]]; d1 = 30.0, d2 = 300.0, bins = 40); 
	
	    p2 = plot(xmin = 𝕟.De[1:end-1], xmax = 𝕟.De[2:end], y = 𝕟.N, Geom.bar, 
	            Theme(default_color="steelblue3"),
	            Guide.xlabel("Particle diameter (nm)"),
	            Guide.ylabel("Number concentration (cm⁻³)"),
	            Guide.title("Histogram Representation 40 Bins"),
	            Guide.xticks(ticks = 0:50:300),
	            Coord.cartesian(xmin=0, xmax=300, ymax = 100)) 
	
	    hstack(p1,p2)
	end

	function aerosol_figure3()
	    Gadfly.set_default_plot_size(18Gadfly.cm, 8Gadfly.cm)
	    colors = ["black", "darkred", "steelblue3"]
	
	    𝕟 = lognormal([[200, 80, 1.2]]; d1 = 30.0, d2 = 300.0, bins = 10); 
	
	    p1 = plot(xmin = 𝕟.De[1:end-1], xmax = 𝕟.De[2:end], y = 𝕟.S, Geom.bar, 
	            Theme(default_color="steelblue3"),
	            Guide.xlabel("Particle diameter (nm)"),
	            Guide.ylabel("dN/dlnD (cm⁻³)"),
	            Guide.title("Spectral Density Representation 10 Bins"),
	            Guide.xticks(ticks = 0:50:300),
	            Coord.cartesian(xmin=0, xmax=300)) 
	
	    𝕟 = lognormal([[200, 80, 1.2]]; d1 = 30.0, d2 = 300.0, bins = 40); 
	
	    p2 = plot(xmin = 𝕟.De[1:end-1], xmax = 𝕟.De[2:end], y = 𝕟.S, Geom.bar, 
	            Theme(default_color="steelblue3"),
	            Guide.xlabel("Particle diameter (nm)"),
	            Guide.ylabel("dN/dlnD (cm⁻³)"),
	            Guide.title("Spectral Density Representation 40 Bins"),
	            Guide.xticks(ticks = 0:50:300),
	            Coord.cartesian(xmin=0, xmax=300)) 
	
	    hstack(p1,p2)
	end
	
	function aerosol_figure4()
		Gadfly.set_default_plot_size(18Gadfly.cm, 8Gadfly.cm)
		colors = ["black", "darkred", "steelblue3"]
	
		𝕟 = lognormal([[200, 80, 1.2]]; d1 = 30.0, d2 = 300.0, bins = 10); 
		xlabel = log10.([30, 50, 100, 200, 300])
		lfunx = x->ifelse(sum(x .== xlabel) == 1, @sprintf("%i",exp10(x)), "")
	
		p1 = plot(xmin = 𝕟.De[1:end-1], xmax = 𝕟.De[2:end], y = 𝕟.S, Geom.bar, 
				Theme(default_color="steelblue3"),
				Guide.xlabel("Particle diameter (nm)"),
				Guide.ylabel("dN/dlnD (cm⁻³)"),
				Guide.title("Spectral Density Representation 10 Bins - Logscale"),
				Guide.xticks(ticks = log10.([30:10:100;200;300])),
				Scale.x_log10(labels = lfunx),
				Coord.cartesian(xmin=log10(30), xmax=log10(300))) 
	
		𝕟 = lognormal([[200, 80, 1.2]]; d1 = 30.0, d2 = 300.0, bins = 40); 
	
		p2 = plot(xmin = 𝕟.De[1:end-1], xmax = 𝕟.De[2:end], y = 𝕟.S, Geom.bar, 
				Theme(default_color="steelblue3"),
				Guide.xlabel("Particle diameter (nm)"),
				Guide.ylabel("dN/dlnD (cm⁻³)"),
				Guide.title("Spectral Density Representation 40 Bins - Logscale"),
				Guide.xticks(ticks = log10.([30:10:100;200;300])),
				Scale.x_log10(labels = lfunx),
				Coord.cartesian(xmin=log10(30), xmax=log10(300))) 
	
		hstack(p1,p2)
	end
	function aerosol_figure5()
	    Gadfly.set_default_plot_size(18Gadfly.cm, 8Gadfly.cm)
	    colors = ["black", "darkred", "steelblue3"]
	
	    𝕟 = lognormal([[200, 80, 1.2]]; d1 = 30.0, d2 = 300.0, bins = 10); 
	    𝕗 = lognormal([[200, 80, 1.2]]; d1 = 30.0, d2 = 300.0, bins = 500); 
	
	    xlabel = log10.([30, 50, 100, 200, 300])
	    lfunx = x->ifelse(sum(x .== xlabel) == 1, @sprintf("%i",exp10(x)), "")
	
	    p1 = plot(layer(x=𝕗.Dp, y = 𝕗.S, Geom.line, 		Gadfly.style(default_color=colorant"black")),
	            layer(xmin = 𝕟.De[1:end-1], xmax = 𝕟.De[2:end], y = 𝕟.S, Geom.bar), 
	            Theme(default_color="steelblue3"),
	            Guide.xlabel("Particle diameter (nm)"),
	            Guide.ylabel("dN/dlnD (cm⁻³)"),
	            Guide.title("Spectral Density Representation 10 Bins - Logscale"),
	            Guide.xticks(ticks = log10.([30:10:100;200;300])),
	            Scale.x_log10(labels = lfunx),
	            Coord.cartesian(xmin=log10(30), xmax=log10(300))) 
	
	    𝕟 = lognormal([[200, 80, 1.2]]; d1 = 30.0, d2 = 300.0, bins = 40); 
	
	    p2 = plot(layer(x=𝕗.Dp, y = 𝕗.S, Geom.line, Gadfly.style(default_color=colorant"black")),
	            xmin = 𝕟.De[1:end-1], xmax = 𝕟.De[2:end], y = 𝕟.S, Geom.bar,
	            Theme(default_color="steelblue3"),
	            Guide.xlabel("Particle diameter (nm)"),
	            Guide.ylabel("dN/dlnD (cm⁻³)"),
	            Guide.title("Spectral Density Representation 40 Bins - Logscale"),
	            Guide.xticks(ticks = log10.([30:10:100;200;300])),
	            Scale.x_log10(labels = lfunx),
	            Coord.cartesian(xmin=log10(30), xmax=log10(300))) 
	
	    hstack(p1,p2)
	end

    function aerosol_figure9()
	    Gadfly.set_default_plot_size(20Gadfly.cm, 16Gadfly.cm)
	    colors = ["black", "darkred", "steelblue3"]
	
	    𝕟 = lognormal([[200, 80, 1.2]]; d1 = 30.0, d2 = 300.0, bins = 10)
	    𝕗 = lognormal([[200, 80, 1.2]]; d1 = 30.0, d2 = 300.0, bins = 500); 
	
	    xlabel = log10.([30, 50, 100, 200, 300])
	    lfunx = x->ifelse(sum(x .== xlabel) == 1, @sprintf("%i",exp10(x)), "")
	
	    p1 = plot(layer(x=𝕗.Dp, y = 𝕗.S, Geom.line, Gadfly.style(default_color=colorant"black")),
	            layer(xmin = 𝕟.De[1:end-1], xmax = 𝕟.De[2:end], y = 𝕟.S, Geom.bar), 
	            Theme(default_color="steelblue3"),
	            Guide.xlabel("Particle diameter (nm)"),
	            Guide.ylabel("dN/dlnD (cm⁻³)"),
	            Guide.title("Number Spectral Density"),
	            Guide.xticks(ticks = log10.([30:10:100;200;300])),
	            Scale.x_log10(labels = lfunx),
	            Coord.cartesian(xmin=log10(30), xmax=log10(300))) 
	
	    p2 = plot(layer(x=𝕗.Dp, y = 𝕗.S.*π.*(𝕗.Dp.*1e-3).^2.0, Geom.line,
	            Gadfly.style(default_color=colorant"black")),
	            layer(xmin = 𝕟.De[1:end-1], xmax = 𝕟.De[2:end], y = 𝕟.S.*π.*(𝕟.Dp.*1e-3).^2.0, Geom.bar), 
	            Theme(default_color="steelblue3"),
	            Guide.xlabel("Particle diameter (nm)"),
	            Guide.ylabel("dS/dlnD (μm² cm⁻³)"),
	            Guide.title("Surface Area Spectral Density"),
	            Guide.xticks(ticks = log10.([30:10:100;200;300])),
	            Scale.x_log10(labels = lfunx),
	            Coord.cartesian(xmin=log10(30), xmax=log10(300))) 
	
	    p3 = plot(layer(x=𝕗.Dp, y = 𝕗.S.*π.*(𝕗.Dp.*1e-3).^3.0./6.0, Geom.line,
	            Gadfly.style(default_color=colorant"black")),
	            layer(xmin = 𝕟.De[1:end-1], xmax = 𝕟.De[2:end], y = 𝕟.S.*π.*(𝕟.Dp.*1e-3).^3.0./6.0, Geom.bar), 
	            Theme(default_color="steelblue3"),
	            Guide.xlabel("Particle diameter (nm)"),
	            Guide.ylabel("dV/dlnD (μm³ cm⁻³)"),
	            Guide.title("Volume  Spectral Density"),
	            Guide.xticks(ticks = log10.([30:10:100;200;300])),
	            Scale.x_log10(labels = lfunx),
	            Coord.cartesian(xmin=log10(30), xmax=log10(300))) 
	
	    p4 = plot(layer(x=𝕗.Dp, y = 𝕗.S.*π.*(𝕗.Dp.*1e-3).^3.0./6.0*2.0, Geom.line,
	            Gadfly.style(default_color=colorant"black")),
	            layer(xmin = 𝕟.De[1:end-1], xmax = 𝕟.De[2:end], y = 𝕟.S.*π.*(𝕟.Dp.*1e-3).^3.0./6.0*2.0, Geom.bar), 
	            Theme(default_color="steelblue3"),
	            Guide.xlabel("Particle diameter (nm)"),
	            Guide.ylabel("dM/dlnD (μg m⁻³)"),
	            Guide.title("Mass Spectral Density"),
	            Guide.xticks(ticks = log10.([30:10:100;200;300])),
	            Scale.x_log10(labels = lfunx),
	            Coord.cartesian(xmin=log10(30), xmax=log10(300))) 
	
	    vstack(hstack(p1,p2),hstack(p3,p4))
	end

	
	aerosol_table1()
end

# ╔═╡ 305ec81f-92a0-441b-a223-40b247c4d7eb
md"""
**Table 1.** Dlow is the lower bound diameter and Dup is the upper bound diameter of a size bin in nm. N is the number concentration of particles that falls into that size bin in ``cm^{-3}``.

  $(aerosol_figure1()) 
**Figure 2.** The same size distribution as in the table plotted as a histogram.

$(Markdown.Admonition("note", "A Exercises", [md"
1. Calculate the total particle number concentration from the table above.
2. Is the bin-width constant in the table and figure?
3. Identify strength and weaknesses of this representation of the size distribution."]))
"""

# ╔═╡ 6c2236b1-06a4-4cd6-b323-c14dc531db3b
md"""
  $(aerosol_figure2()) 
**Figure 3.** The same size distribution as in Figure 2 but varying the number of bins. Left measured using 10 bins and right measured using 40 bins.

$(Markdown.Admonition("note", "B Exercises", [md"
1. Explain why the y-values of the left and right histogram are different.
2. What would happen to the number observed in a size bin if the number of bins were to be increased to a very large number (e.g. 10000)?"]))

## Spectral Density

A better way to represent the size distribution is to normalize the number concentration by the width of the size bin interval in log-space. The resulting quantity is referred to as the spectral density

```math
\frac{dN}{dlnD}=\frac{N}{ln(D_{up})-ln(D_{low})}=\frac{N}{ln(D_{up}/D_{low})}
```

$(aerosol_table2()) 
$(Markdown.Admonition("note", "C Exercises", [md"
1. Determine the unit of spectral density.
2. Calculate the spectral density for the size distribution table above.
"]))

$(aerosol_figure3())
**Figure 4.** The same size distribution in the tables but plotted agains spectral density. Left measured using 10 bins and right measured using 40 bins.

$(Markdown.Admonition("note", "D Exercises", [md"
1. Sum up the dN/dlnD values in the table. Do they correspond the the total number concentration?
2. Based on figure, identify strength and weaknesses of this representation of the size distribution.
"]))


$(aerosol_figure4())
**Figure 5.** The same size distribution as in the tables, plotted on a log-diameter axis.

$(Markdown.Admonition("note", "E Exercises", [md"
Are the bins spaced regularly when plotting in log-space? If so, why?
"]))

## Lognormal Function

Note that the size distribution appears like bell-shaped curve when plotted in logarithmic-diameter space. Such distributions are represented using the lognormal size distribution function

The lognormal size distribution function is 

```math
\frac{dN}{d\ln D_p} = \frac{N_{t}}{\sqrt{2\pi}\ln\sigma_{g}} \exp \left(- \frac{\left[\ln D_p-\ln D_{pg}\right]^2}{2\ln \sigma_{g}^2}\right)
``` 

where ``\frac{dN}{d\ln D_p}`` is the spectral number density, ``N_{t}`` is the total number concentration, ``\sigma_{g}`` is the geometric standard deviation, and ``D_{pg}`` is the mode diameter. The size distribution is then described by the triplet ``\{N_{t},D_{pg},\sigma_{g}\}`` that best describes the data. 
    
$(Markdown.Admonition("warning", "Key Concept", [md"The mode diameter represents the most frequent observation, or highest spectral density. The geometric standard deviation varies between 1 for an infinitely narrow distribution and values exceeding 1. For a normal distribution, 68.21% of particle countes fall between ``\pm`` one standard deviation. For the lognormal distribution 68.21% of particle counts are within the interval ``{D_g}/{\sigma_g} < D < D_g\sigma_g``."]))

$(aerosol_figure5())
**Figure 5.** Representation of the same size distribution as histogram sampled with 10 bins (left) and 40 bins (right), plotted against spectral density and using a logarithmic diameter axis, and with the best fit size-distribution function overlaid. The parameters are ``{N_t = 200\;cm^{−3} , D_{pg} = 80\; nm, \sigma_g = 1.2}``.

$(Markdown.Admonition("note", "F Exercises", [md"
What does ``N_{t}`` represent? How does it compare to your answer from A Exercises?
"]))

"""

# ╔═╡ 812aae18-616d-4649-883d-caf028ab693b
begin
	gengrid(r) = [vcat(map(x->x:x:9x,r)...);r[end]*10]
	cfun(c) = RGBA{Float32}(c.r,c.g,c.b,1)
	
	function aerosol_app1(Nt, Dg, σg)
	    Gadfly.set_default_plot_size(18Gadfly.cm, 8Gadfly.cm)
	    𝕗 = lognormal([[Nt, Dg, σg]]; d1 = 1.0, d2 = 1000.0, bins = 1000); 
		𝕘 = deepcopy(𝕗)
	    colors = ["black", "darkred", "steelblue3", "darkgrey"]
	    xlabel = log10.([1, 10, 100, 1000])
	    lfunx = x->ifelse(sum(x .== xlabel) == 1, @sprintf("%i",exp10(x)), "")
	
	    Nmax1 = maximum(𝕗.S)
	    Nmax2 = maximum(𝕘.S[(𝕘.Dp .> Dg*σg) .| (𝕘.Dp .< Dg/σg)])
	    𝕘.S[(𝕘.Dp .> Dg*σg) .| (𝕘.Dp .< Dg/σg)] .= 0.0
	    label1 = ["D<sub>pg</sub> = $Dg nm" for i=1:2]
	    label2 = @sprintf("%.1f < D < %.1f nm", Dg/σg, Dg*σg)
	    label3 = [@sprintf("N shaded area = %.1f cm⁻³", 0.6827*Nt)  for i=1:1000]
	    l0 = layer(x=𝕗.Dp, y = 𝕗.S, color = ["Lognormal function" for i=1:1000],
			Geom.line)
	    l1 = layer(x = [Dg, Dg], y = [-2e3,Nmax1], color = label1, Geom.line,
			Geom.point,  Theme(alphas=[0.5], discrete_highlight_color=
				cfun,highlight_width=1Gadfly.pt))
	    l2 = layer(x=[Dg],y=[Nmax2],xmin=[Dg/σg],xmax=[Dg*σg],color =[label2],
			Geom.errorbar,Geom.line)    
	    l3 = layer(x=𝕘.Dp,y=𝕘.S,color=label3, Geom.bar)    
	
	    p1 = plot(l0, l1, l2,l3,
	            Guide.xlabel("Particle diameter (nm)"),
	            Guide.ylabel("dN/dlnD (cm⁻³)"),
	            Scale.color_discrete_manual(colors...),
	            Guide.xticks(ticks = log10.(gengrid([1,10,100]))),
	            Scale.x_log10(labels = lfunx),
	            Coord.cartesian(xmin=log10(1), xmax=log10(1000),ymin = 0)) 
	end

@bind vars combine() do Child
	md"""
	``N_t (cm^{-3})``   $(
		Child(Slider([10:10:90;100:100:1000], default = 100, show_value = true))
	) 
	
	``D_g (nm)``  $(
		Child(Slider([1:1:9;10:10:90;100:100:500], default = 50, show_value = true))
	) 
	
	``\sigma_g (-)``  $(
		Child(Slider(1:0.05:2.5, default = 1.6, show_value = true))
	)
	"""
end

end

# ╔═╡ d6df37c9-d86f-4e73-bcb9-fe776c7c817b
md"""
$(aerosol_app1(vars[1], vars[2], vars[3]))
**Figure 7.** Visualization of the lognormal distribution function.

$(Markdown.Admonition("note", "G Exercises", [md"
Explore how the parameters of the lognormal function influence the distribution.

1. What does ``D_{pg}`` represent? How would you identify it from the chart?

2. What does ``\sigma_{g}`` represent?

3. Move the sliders and observe the change in the legend. Verify that 68.21% of particles fall within the shaded area defined by ``{D_{pg}}/{\sigma_g} < D_p < D_{pg}\sigma_g``. Verify for at least three combinations of ``\{N_t, D_{pg},\sigma_g\}``

4. Assume you are provided with a plot of the size distribution that shows ``D_pg`` (red) and the spread (blue). Use this information from the legend reconstruct the observed ``\sigma_g``.  
"]))

"""

# ╔═╡ f10b12b0-4bc8-4d04-8848-ce0a7f391b10
md"""
# Observations of Size Distributions

Many techniques to measure size distributions exist. Two common instrument are the Scanning Mobility Particle Sizer (SMPS) and the Optical Particle Counter (OPC) and are introduced here.

## Scanning Mobility Particle Sizer

Aerosol flows through an annulus gap. An electric potential is applied between the inner and outer column. The electric potential drags charged particles through a sheath flow. Charged particles within a narrow electrical mobility band are steered to a sample slit. The electric potential selects a narrow size range. Particles within this size range are counted using a condensation particle counter. Scanning the electric potential with time, usually over a 1-5 min period produces a particle size distribution. 

The SMPS technique can used to measure particles between 1-1000 nm. However, a single instrument is limited to a narrower range which is determined by the length of the column and the flow rate through the instrument.

|                     |                      |
|---------------------|----------------------|
| $(Resource(smps1_url, :width => 1400px)) | $(Resource(smps2_url, :width => 1100px))|
**Figure 8.** (Left) Schematic of the SMPS. *Image Source:* Petters (2018), License CC BY-NC-ND 4.0. (Right) Commercial SMPS instrument. *Image Source:* manufacturer brochure.
  
## Optical Particle Spectrometer

Aerosol flow is directed through a laser beam. Particles scatter (redirect) light in the beam to photodetector. The intensity of the scattered light is related to the particle size. Concentration is obtained from the number of particles crossing the laser beam per unit time and the flow rate through the instrument. Particles are binned into size bins based on scattered light intensity. 

The OPC technique can be used to measure particles > 60 nm. However, most OPCs only detect particles > 400 nm. At large sizes concentrations become small and detection is limited by counting statistics. Thus the size range depends on the specific OPC model. The POPS particles 150 nm - 3000 nm. 

|                     |                      |
|---------------------|----------------------|
| $(Resource(pops1_url, :width => 1400px)) | $(Resource(pops2_url, :width => 900px)) |


**Figure 9.** (Left) Top and side view of an optical particle counter. The collimating lens (CL) is shown in green, the cylindrical lenses (CyLs) are shown in blue,the diode laser, DL, is in solid dark blue. (Right) A complete and functional instrument. Bottom panel: The optics box with the scattering signal digitizer board (shielded by a grounded brass EMI cover). *Image Source:* Gao et al. (2018), License: Non-commercial re-use, distribution, and reproduction in any medium, provided the original work is properly attributed, cited, and is not altered, transformed, or built upon in any way, is permitted. 

# Ambient Size Distributions
"""

# ╔═╡ a928db35-b4d0-486a-9870-0d68464b4e2c
begin
	hfdma = CSV.read(HTTP.get(hfdma_url).body, DataFrame)
	lhfdma = CSV.read(HTTP.get(lhfdma_url).body, DataFrame)
	uhfdma = CSV.read(HTTP.get(uhfdma_url).body, DataFrame)
	rdma = CSV.read(HTTP.get(rdma_url).body, DataFrame)
	lrdma = CSV.read(HTTP.get(lrdma_url).body, DataFrame)
	urdma = CSV.read(HTTP.get(urdma_url).body, DataFrame)
	pops = CSV.read(HTTP.get(pops_url).body, DataFrame)
	lpops = CSV.read(HTTP.get(lpops_url).body, DataFrame)
	upops = CSV.read(HTTP.get(upops_url).body, DataFrame)

	@bind tindex combine() do Child
	md"""
	Time Index   $(
		Child(Slider(1:1:49, default = 18, show_value = true))
	) 
	"""
end
end

# ╔═╡ 184b3fb6-6baf-4948-b648-68dcc697a1f1
begin
	function mode_figure()
		xlabel = log10.([1, 10, 100, 1000, 10000])
		colors = ["black", "darkred", "darkgoldenrod", "steelblue3", "purple", "black"]
		lfunx = x->ifelse(sum(x .== xlabel) == 1, @sprintf("%i",exp10(x)), "")
		n0 = lognormal([[3000, 7, 1.2]]; d1 = 1.0, d2 = 100.0, bins = 1000)
		l0 = layer(x=n0.Dp, y = n0.S, Geom.line, color = ["Nucleation Mode" for i=1:100], Theme(line_width=2pt))
		n1 = lognormal([[4900, 30, 1.6]]; d1 = 1.0, d2 = 300.0, bins = 1000)
		l1 = layer(x=n1.Dp, y = n1.S, Geom.line, color = ["Aitken Mode" for i=1:1000],
		Theme(line_width=2pt))
		n2 = lognormal([[900, 150, 1.7]]; d1 = 1.0, d2 = 10000.0, bins = 1000)
		l2 = layer(x=n2.Dp, y = n2.S, Geom.line, color = ["Accumulation Mode" for i=1:1000],
		Theme(line_width=2pt))
		n3 = lognormal([[300, 2000, 1.9]]; d1 = 1.0, d2 = 10000.0, bins = 1000)
		l3 = layer(x=n3.Dp, y = n3.S, Geom.line, color = ["Coarse Mode" for i=1:1000], Theme(line_width=2pt))

		n4 = lognormal([[3000, 7, 1.2], [4900, 30, 1.6], [800, 150, 1.7], [300, 2000, 1.9]]; d1 = 1.0, d2 = 10000.0, bins = 1000)
		l4 = layer(x=n4.Dp, y = n4.S, Geom.line, color = ["Sum of Modes" for i=1:1000], linestyle=[:dot])
		p1 = plot(l4, l0, l1, l2, l3, 
			Guide.xlabel("Particle diameter (nm)"),
			Guide.ylabel("dN/dlnD (cm⁻³)"),
			Scale.color_discrete_manual(colors...),
			Guide.xticks(ticks = log10.(gengrid([1,10,100,1000]))),
			Scale.x_log10(labels = lfunx),
			Coord.cartesian(xmin=log10(1), xmax=log10(10000)))
	end
	function aerosol_app2(j)
		Gadfly.set_default_plot_size(18Gadfly.cm, 8Gadfly.cm)
		Dhf = Vector(hfdma[1,2:end])
		Shf = Matrix(hfdma[2:end,2:end])
		Slhf = Matrix(lhfdma[2:end,2:end])
		Suhf = Matrix(uhfdma[2:end,2:end])
		Drd = Vector(rdma[1,2:end])
		Srd = Matrix(rdma[2:end,2:end])
		Slrd = Matrix(lrdma[2:end,2:end])
		Surd = Matrix(urdma[2:end,2:end])
		
		t = collect(skipmissing(hfdma[2:end,1]))
		
		Dpo = Vector(pops[1,2:end])
		Spo = Matrix(pops[2:end,2:end])
		Slpo = Matrix(lpops[2:end,2:end])
		Supo = Matrix(upops[2:end,2:end])
		
		colors = ["darkgoldenrod3", "darkred", "steelblue3", "darkgrey"]
		xlabel = log10.([5, 10, 20, 50, 100, 200, 500, 1000, 2000])
		lfunx = x->ifelse(sum(x .== xlabel) == 1, @sprintf("%i",exp10(x)), "")
		cfun(c) = RGBA{Float32}(c.r,c.g,c.b,1)
		
		l0 = layer(x=Dhf, y = Shf[j,:], ymin=Slhf[j,:], ymax=Suhf[j,:], 
			Geom.line, Geom.ribbon, color = ["SMPS2" for i=1:length(Dhf)], Theme(alphas=[0.3],lowlight_color=cfun))
		l1 = layer(x=Drd, y = Srd[j,:], ymin=Slrd[j,:], ymax=Surd[j,:], 
			Geom.line, Geom.ribbon, color = ["SMPS1" for i=1:length(Drd)], Theme(alphas=[0.3],lowlight_color=cfun))
		l2 = layer(x=Dpo, y = Spo[j,:], ymin=Slpo[j,:], ymax=Supo[j,:], 
			Geom.line, Geom.ribbon,   color = ["OPC" for i=1:length(Dpo)], Theme(alphas=[0.3],lowlight_color=cfun))
		p1 = plot(l1,l0,l2,
			Guide.xlabel("Particle diameter (nm)"),
			Guide.ylabel("dN/dlnD (cm⁻³)"),
			Guide.title("Composite Size Distribtion $(t[j])"),
			Scale.color_discrete_manual(colors...),
			Guide.xticks(ticks = log10.(gengrid([1,10,100,1000]))),
			Scale.x_log10(labels = lfunx),
			Coord.cartesian(xmin=log10(4), xmax=log10(3000)))
	end

md"""	
## Example Temporal Evolution

$(aerosol_app2(tindex[1]))
**Figure 10.** One day of aerosol size distribution data collected in Raleigh, NC, November 2019. The observations are using two SMPS instruments and one POPS optical particle counter. The distributions are a 30 min time average. The composite size distribution is obtained by stiching the three instruments together. Shaded areas show the 5% to 95% interquartile range of the observations.

$(Markdown.Admonition("note", "H Exercises", [md"
1. Use the slider to explore how the size distribution changes during the one day period. Analyze the output and write down at minimum three conclusions you can make based the data.
2. In your own words, describe how an SMPS and an OPC measure the particle size distribution.
3. List at least one advantage and one disadvantage of the optical particle counter (OPC) and scanning mobility particle sizer (SMPS)"]))

## Multimodal  Size Distributions

The data shows that that aerosol size distribution typically consists of 2-4 modes. Each mode is approximately lognormaldistributed. The total size distribution is discribed by the sum of the modes:

```math
\frac{dN}{d\ln D_p} = \sum_{i=1}^n \frac{N_{t,i}}{\sqrt{2\pi}\ln\sigma_{g,i}} \exp \left(- \frac{\left[\ln D_p-\ln D_{pg,i}\right]^2}{2\ln \sigma_{g,i}^2}\right)
```

where ``\frac{dN}{d\ln D_p}`` is the spectral number density, ``N_{t,i}`` is the total number concentration, ``\sigma_{g,i}`` is the geometric standard deviation, and ``D_{pg,i}`` is the geometric mean diameter of the ``i^{th}`` mode, and ``n`` is the number of modes. 

$(mode_figure())
**Figure 11.** Schematic representation of a multimodal lognormal distribution function. The number concentration in the coarse mode is exagerated to make it visibile. 

The four modes have been referred to by various names, depending on the author. However, commone names and associated sources with these modes are:

**1. Nucleation Mode:** Particles below 10 nm. These particles have been newly formed either by homogeneous nucleation in the atmosphere or by nucleation processes that occur within the emissions from high temperature sources and lead to the emission of primary nucleation mode particles. Homogenous nucleation is the processes by which low volatility gas-phase compounds form spontaneously new solid or liquid particles. For example, sulfuring acid (H2SO4) is formed from the atmospheric oxidation of SO2. This H2SO4 can nucleate and form sulfate aerosol.

**2. Aitken Mode:** Particles beween 10-80 nm. Names after the meteorologist John Aitken (1839 – 14 November 1919) who discovered these particles. Sources of Aitken mode particles include direct emissions, e.g. from vehicle exhaust, and growth of nucleation mode particles by condensation of vapors.

**3. Accumulation Mode:** Particles between 80-1000 nm. Named for their long lifetime in the atmosphere (7-30 days). Brownian motion and settling velocity of these particles is small. Therefore these particle accumulate in the atmosphere. Accumulation mode particles form from the growth of Aitken mode particles by condensation and by coagulation.  Further growth is inhibited because they do not coagulate as rapidly as fine particles and do not settle like coarse particles. Accumulation particles can be efficiently removed by rain.

**4. Coarse Mode:** Particles > 1000 nm in diameter. Coarse mode particles are generated mechanically. Sources include wind-blown soil/desert dusts, industrial emissions, biological particles (bacteria, viruses) and sea-spray aerosol. Gravitational settling is fast, which reduces the atmospheric lifetime of these particles relative to those of the accumulation mode. 

"""
	end

# ╔═╡ 5b62c4dd-02b0-4621-b40a-2f2a96a5b316
md"""
# Moment Distributions

Aerosol concentration can be characterized in terms of number concentration, surface area concentration, volume concentration, and mass concentration. As a rule of thumb:

- Number concentration is important to assess aerosol impacts on clouds and climate
- Surface area concentration is important to assess aerosol impacts on light scattering and surface mediated processes such as condensation, chemical reaction on particle surfaces, and ice nucleation
- Volume/Mass concentration is important to assess aerosol impacts on human health and the environment (e.g. nutrient cycling)

Surface area, volume, and mass concentration can be derived from the size distribution

```math
S=\pi D_{p}^{2}N
```

```math
V=\frac{\pi}{6}D_{p}^{3}N
```

```math
M=\frac{\pi}{6}D_{p}^{3}N\rho_{p}
```

where ``\rho_{p}`` is the particle density. The midpoint diameter of the bin, ``D_{p}=(D_{low}+D_{up})/2``, should be used in the calculation.

## Tabular Representation
$(aerosol_table3())
**Table 3.** Dlow and Dup are the lower and upper bound particle diameters in nm. Dp is the calulated midpoint diameter. N is the number concentration in the bin in units of ``cm^{-3}``. Columns 1,2, and 4 are the same as Table 1. S, V, and M are the surface area in ``\mu m^{2}\;cm^{-3}``, volume concentration in ``\mu m^{3}\;cm^{-3}`` and mass concentration in ``\mu g\;m^{-3}`` using the formulae above. A particle density of ``2 g cm^{-3}`` is assumed. Example calculations for row 3 are as follows:

```math
S=\pi D_{p}^{2}N=3.1415\times(54\times10^{-3}\;\mu m)^{2}\times9\;cm^{-3}=0.082\;\mu m^{2}\;cm^{-3}
```

```math
V=\pi/6D_{p}^{3}N=3.1415 /6 \times(54\times10^{-3}\;\mu m)^{3}\times9\;cm^{-3}=0.00073\;\mu m^{3}\;cm^{-3}
```

```math
\begin{array}
\\M=\pi/6D_{p}^{3}N =3.1415 / 6 \times(54\times10^{-3}\;\mu m)^{3}\times9\;cm^{-3}\times2\;g\;cm^{-3}\times10^{6}\;\mu g\;g^{-1} \\
\times10^{-18}\;m^{3}\;\mu m^{-3}\times10^{12}\;cm^{6}\;m^{-6}=0.0015\;\mu g\;m^{-3}
\end{array}

```

$(Markdown.MD(
	Markdown.Admonition("warning", "Important Shortcut", [md"
Note that for mass calculations, if the density is ``1000\; kg\;m^{-3} = 1\; g\; cm^{-3}``, the volume concentration in ``\mu m^{3}\;cm^{-3}`` equals to the mass concentration in ``\mu g\;m^{-3}``. This is an important conversion and explains why ``\mu m^{3}\;cm^{-3}`` is the preferred unit for volume concentration."])))

Important mass fractions of the aerosol are PM1, PM2.5 and PM10, denoting  particulate matters with D < 1 μm, 2.5 μm, and 10 μm, respectively. PM concentrations can be obtained by integration over the size distribution.

$(Markdown.MD(
	Markdown.Admonition("note", "J Exercises", [md"
1. Compute the surface area, volume, and mass distribution for the missing entries the table, assuming ``\rho_{p}=2000\;kg\;m^{-3}`` 
2. Compute the total surface area, volume, and mass concentration from the distribution."])))

## Spectral Density

As with number concentration, the surface area and mass distributions are usually written as spectral densities and plotted in logarithmic diameter space. 

```math
\frac{dS}{dlnD}=\pi D_{p}^{2}\frac{dN}{dlnD}
```

```math
\frac{dV}{dlnD}=\frac{\pi}{6}D_{p}^{3}\frac{dN}{dlnD}
```

```math
\frac{dM}{dlnD}=\frac{\pi}{6}D_{p}^{3}\frac{dN}{dlnD}\rho_{p}
```

$(aerosol_figure9())
**Figure 12.** Representation of the same size distribution as in table 3 plotted as a histogram sampled with 10 bins against spectral density of the number surface area and mass distribution, using a logarithmic x-axis, and with the best fit size-distribution function overlaid. A particle density of 2000 kg m-3 is assumed. The parameters are ``\{ N_{t}=200\;cm^{-3},D_{pg}=80\;nm,\sigma_{g}=1.2\}``.

$(Markdown.MD(
	Markdown.Admonition("note", "K Exercises", [md"
1. Is the mode diameter the same for the four different distributions? Why or why not?
2. Perform unit analysis to show that the units on the y-axes are correct."])))
"""

# ╔═╡ fd570630-252a-4b45-b574-7fe94d669a3a
begin
	challenge = Markdown.MD(
		Markdown.Admonition("tip", "Synthesis Assignment",  [md"""
	
	**The following a multimodal size distribution has been reported in the literature**
	
	| Mode 1 | Mode 2 | Mode 3 |
	|--------|--------|--------|
	| \{133, 8, 1.93\} | \{66.6, 266, 1.24\} | {3.1, 580, 1.49\} |
	
	where the numbers for each mode represent ``\{N_t, D_{pg},\sigma_g\}`` in units ``\{cm^{-3}, nm, -\}``.
		
	1. Plot the number, surface area and volume density distributions.
	2. Numerically estimate the total number, surface area, volume, and mass concentrations in the ultrafine mode (D < 100 nm), accumulation mode (100 nm < D < 1 µm), and total (i.e. all sizes).  
	"""]));
	
	md"""

	# Synthesis Problem
	$(challenge)
	Solution:
	
	|               | Number (cm⁻³) |  Surface Area (μm² cm⁻³)   |   Mass (μg m⁻³)  |
	|---------------|---------------|----------------------------|------------------|
	| Total         | 202.6         | 20.7  | 2.16    |
	|D < 0.1 μm     | 132.9         | 0.063 | 0.00036 |
	|0.1 < D < 1 μm | 69.4          | 19.4  | 1.75    |
	"""
end

# ╔═╡ 9e6f74f4-c32b-4873-9870-40d0d432c382
md"""
# References Cited

Gao, R. S., H. Telg, R. J. McLaughlin, S. J. Ciciora, L. A. Watts, M. S. Richardson, J. P. Schwarz, A. E. Perring, T. D. Thornberry, A. W. Rollins, M. Z. Markovic, T. S. Bates, J. E. Johnson & D. W. Fahey (2016) A light-weight, high-sensitivity particle spectrometer for PM2.5 aerosol measurements, Aerosol Science and Technology, 50:1, 88-99, doi:10.1080/02786826.2015.1131809. 

Laden F, Schwartz J, Speizer FE, Dockery DW. Reduction in fine particulate air pollution and mortality: Extended follow-up of the Harvard Six Cities study. Am J Respir Crit Care Med. 2006;173(6):667–672. doi:10.1164/rccm.200503-443OC.

Petters, M. D. (2018) A language to simplify computation of differential mobility analyzer response functions, Aerosol Science and Technology, 52:12, 1437-1451, doi:10.1080/02786826.2018.1530724.

Stolzenburg, M. R., McMurry, P. H., Sakurai, H., Smith, J. N., Mauldin III, R. L., Eisele, F. L., and Clement, C. F. ( 2005), Growth rates of freshly nucleated atmospheric particles in Atlanta, J. Geophys. Res., 110, D22S05, doi:10.1029/2005JD005935.


# License
    
Author and Copyright: [Markus Petters](https://mdpetters.github.io/)

The text of this notebook and images created by the author are licensed under the Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International Public License [CC BY-NC-SA 4.0](https://creativecommons.org/licenses/by-nc-sa/4.0/). The scripts are licensed under the [GNU General Public License v3](https://www.gnu.org/licenses/gpl-3.0.en.html). 

Images and software from other sources are licensed as indicated.
"""

# ╔═╡ 00000000-0000-0000-0000-000000000001
PLUTO_PROJECT_TOML_CONTENTS = """
[deps]
CSV = "336ed68f-0bac-5ca0-87d4-7b16caf5d00b"
Colors = "5ae59095-9a9b-59fe-a467-6f913c188581"
DataFrames = "a93c6f00-e57d-5684-b7b6-d8193f3e46c0"
Gadfly = "c91e804a-d5a3-530f-b6f0-dfbca275c004"
HTTP = "cd3eb016-35fb-5094-929b-558a96fad6f3"
LaTeXStrings = "b964fa9f-0449-5b57-a5c2-d3ea65f4040f"
LinearAlgebra = "37e2e46d-f89d-539d-b4ee-838fcccc9c8e"
Logging = "56ddb016-857b-54e1-b83d-db4d58db5568"
PlutoUI = "7f904dfe-b85e-4ff6-b463-dae2292396a8"
Printf = "de0858da-6303-5e67-8744-51eddeeeb8d7"

[compat]
CSV = "~0.10.12"
Colors = "~0.12.10"
DataFrames = "~1.6.1"
Gadfly = "~1.4.0"
HTTP = "~1.10.2"
LaTeXStrings = "~1.3.1"
PlutoUI = "~0.7.58"
"""

# ╔═╡ 00000000-0000-0000-0000-000000000002
PLUTO_MANIFEST_TOML_CONTENTS = """
# This file is machine-generated - editing it directly is not advised

julia_version = "1.10.2"
manifest_format = "2.0"
project_hash = "e11d353f2896d87b91319aff25b19ea4cc447b38"

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

[[deps.BitFlags]]
git-tree-sha1 = "2dc09997850d68179b69dafb58ae806167a32b1b"
uuid = "d1d4a3ce-64b1-5f1a-9ba4-7e7e69966f35"
version = "0.1.8"

[[deps.CSV]]
deps = ["CodecZlib", "Dates", "FilePathsBase", "InlineStrings", "Mmap", "Parsers", "PooledArrays", "PrecompileTools", "SentinelArrays", "Tables", "Unicode", "WeakRefStrings", "WorkerUtilities"]
git-tree-sha1 = "679e69c611fff422038e9e21e270c4197d49d918"
uuid = "336ed68f-0bac-5ca0-87d4-7b16caf5d00b"
version = "0.10.12"

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

[[deps.CodecZlib]]
deps = ["TranscodingStreams", "Zlib_jll"]
git-tree-sha1 = "59939d8a997469ee05c4b4944560a820f9ba0d73"
uuid = "944b1d66-785c-5afd-91f1-9de20f533193"
version = "0.7.4"

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
git-tree-sha1 = "d2c021fbdde94f6cdaa799639adfeeaa17fd67f5"
uuid = "34da2185-b29b-5c13-b0c7-acf172513d20"
version = "4.13.0"
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

[[deps.ConcurrentUtilities]]
deps = ["Serialization", "Sockets"]
git-tree-sha1 = "9c4708e3ed2b799e6124b5673a712dda0b596a9b"
uuid = "f0e56b4a-5159-44fe-b623-3e5288b988bb"
version = "2.3.1"

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

[[deps.ExceptionUnwrapping]]
deps = ["Test"]
git-tree-sha1 = "dcb08a0d93ec0b1cdc4af184b26b591e9695423a"
uuid = "460bff9d-24e4-43bc-9d9f-a8973cb893f4"
version = "0.1.10"

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

[[deps.FilePathsBase]]
deps = ["Compat", "Dates", "Mmap", "Printf", "Test", "UUIDs"]
git-tree-sha1 = "9f00e42f8d99fdde64d40c8ea5d14269a2e2c1aa"
uuid = "48062228-2e41-5def-b9a4-89aafe57970f"
version = "0.9.21"

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

[[deps.HTTP]]
deps = ["Base64", "CodecZlib", "ConcurrentUtilities", "Dates", "ExceptionUnwrapping", "Logging", "LoggingExtras", "MbedTLS", "NetworkOptions", "OpenSSL", "Random", "SimpleBufferStream", "Sockets", "URIs", "UUIDs"]
git-tree-sha1 = "ac7b73d562b8f4287c3b67b4c66a5395a19c1ae8"
uuid = "cd3eb016-35fb-5094-929b-558a96fad6f3"
version = "1.10.2"

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

[[deps.LoggingExtras]]
deps = ["Dates", "Logging"]
git-tree-sha1 = "c1dd6d7978c12545b4179fb6153b9250c96b0075"
uuid = "e6f89c97-d47a-5376-807f-9c37f3926c36"
version = "1.0.3"

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

[[deps.MbedTLS]]
deps = ["Dates", "MbedTLS_jll", "MozillaCACerts_jll", "NetworkOptions", "Random", "Sockets"]
git-tree-sha1 = "c067a280ddc25f196b5e7df3877c6b226d390aaf"
uuid = "739be429-bea8-5141-9913-cc70e7f3736d"
version = "1.1.9"

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

[[deps.OpenSSL]]
deps = ["BitFlags", "Dates", "MozillaCACerts_jll", "OpenSSL_jll", "Sockets"]
git-tree-sha1 = "51901a49222b09e3743c65b8847687ae5fc78eb2"
uuid = "4d8831e6-92b7-49fb-bdf8-b643e874388c"
version = "1.4.1"

[[deps.OpenSSL_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "60e3045590bd104a16fefb12836c00c0ef8c7f8c"
uuid = "458c3c95-2e84-50aa-8efc-19380b2a3a95"
version = "3.0.13+0"

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

[[deps.SimpleBufferStream]]
git-tree-sha1 = "874e8867b33a00e784c8a7e4b60afe9e037b74e1"
uuid = "777ac1f9-54b0-4bf8-805c-2214025038e7"
version = "1.1.0"

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

[[deps.TranscodingStreams]]
git-tree-sha1 = "54194d92959d8ebaa8e26227dbe3cdefcdcd594f"
uuid = "3bb67fe8-82b1-5028-8e26-92a6c54297fa"
version = "0.10.3"
weakdeps = ["Random", "Test"]

    [deps.TranscodingStreams.extensions]
    TestExt = ["Test", "Random"]

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

[[deps.WeakRefStrings]]
deps = ["DataAPI", "InlineStrings", "Parsers"]
git-tree-sha1 = "b1be2855ed9ed8eac54e5caff2afcdb442d52c23"
uuid = "ea10d353-3f73-51f8-a26c-33c1cb351aa5"
version = "1.4.2"

[[deps.WoodburyMatrices]]
deps = ["LinearAlgebra", "SparseArrays"]
git-tree-sha1 = "c1a7aa6219628fcd757dede0ca95e245c5cd9511"
uuid = "efce3f68-66dc-5838-9240-27a6d6f5f9b6"
version = "1.0.0"

[[deps.WorkerUtilities]]
git-tree-sha1 = "cd1659ba0d57b71a464a29e64dbc67cfe83d54e7"
uuid = "76eceee3-57b5-4d4a-8e66-0e911cebbf60"
version = "1.6.1"

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
# ╟─70924856-fa28-4ca2-b024-95664a461213
# ╟─4c2859cd-b599-41a5-a5eb-e34c6c35a74f
# ╟─f1eed054-7d08-4952-b739-d0cfc3b2549a
# ╟─305ec81f-92a0-441b-a223-40b247c4d7eb
# ╟─6c2236b1-06a4-4cd6-b323-c14dc531db3b
# ╟─812aae18-616d-4649-883d-caf028ab693b
# ╟─d6df37c9-d86f-4e73-bcb9-fe776c7c817b
# ╟─f10b12b0-4bc8-4d04-8848-ce0a7f391b10
# ╟─a928db35-b4d0-486a-9870-0d68464b4e2c
# ╟─184b3fb6-6baf-4948-b648-68dcc697a1f1
# ╟─5b62c4dd-02b0-4621-b40a-2f2a96a5b316
# ╟─fd570630-252a-4b45-b574-7fe94d669a3a
# ╟─9e6f74f4-c32b-4873-9870-40d0d432c382
# ╟─00000000-0000-0000-0000-000000000001
# ╟─00000000-0000-0000-0000-000000000002
