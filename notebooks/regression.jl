### A Pluto.jl notebook ###
# v0.19.32

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

# ╔═╡ 62287756-e122-406b-bfbf-8ac45779f91b
begin
	using CSV
	using DataFrames
	using LaTeXStrings
	using Plots,Plots.PlotMeasures
	using LinearAlgebra
	using StatsBase
	using PlutoUI
	using ColorSchemes
	using SymPy
	using HTTP
	using PlutoUI: combine
	using Random
	
	SSE(y,X,β) = (y-X*β)'*(y-X*β)
	linfit(x, y) = [ones(length(x), 1) x] \ y 
	df = DataFrame(x = [1,2,3,4,5,6,7,8.0], 
		y = [-0.2, 2.2, 3.5, 5.0, 8.8, 7, 10.0, 12.0])
	
	md"""
	# Linear Regression, Integral Equations, Data Inversion, and Regularization 

	This notebook introduces mathematical descriptions of linear regression and it's application in inverse modeling.
	$(TableOfContents(depth=4))
	"""
end

# ╔═╡ bf16cb46-2f7e-4f51-aca6-2a1f7fcae79b
md"""# Learning Objectives
"""

# ╔═╡ 6708a714-5946-4191-9758-b20fdd56e62b
Markdown.MD(
	Markdown.Admonition("tip", "Learning Objectives", [md"
- Apply matrix based methods to compute multiple linear regression models to describe observational data.
- Apply the quadrature method to transform Fredholm integral equations into matrix form.
- Apply Tikhonov regularization to perform matrix-based inversion from observational data and linear models.
- Apply Tikhonov regularization to filter/smooth regularly spaced data.
"]))

# ╔═╡ bb2ccd82-562d-48bc-99dc-6301126b3a41
md"""# Linear Regression

## Problem Definition

Suppose we have data with one dependent variable ``x`` and one independent variable ``y``. The two appear to be correleted. We seek the best fit line via

```math
\hat{y} = mx + b
```

where ``\hat{y}`` is the estimated observation from the linear model and ``m`` and ``b`` are coefficients. One way to define the best fit is to minimize the sum square error (SSE):

```math
SSE = \sum{(\hat{y}}_i - y_i)^2 = || \vec{\hat{y}}-\vec{y} ||_2^2
```
and ``||\vec{x} ||_2=\sqrt{x_1^2 + x_2^2 + ... + x_n^2}``  is the [two-norm](https://en.wiktionary.org/wiki/two-norm) of the vector ``\vec{x} = <x_1, x_2, ..., x_n>.``

## Example: Minimize SSE via Slider
"""

# ╔═╡ d1dc866c-33c0-41df-92aa-c426a78eee9d
begin
	aGrid = -10:0.1:10
	bGrid = -5:0.1:5
	aSlider = @bind a Slider(aGrid, default = mean(df.x), show_value = true)
	bSlider = @bind b Slider(bGrid, default = 0, show_value = true)

	residualsTextBox = @bind plotResiduals CheckBox(default = true)
	bestTextBox = @bind plotBest CheckBox(default = false)
	
	md"""
	
	**Regression coefficients**
	 
	m: $(bSlider)
	b: $(aSlider)   
	plot residuals: $(residualsTextBox)
	plot best fit: $(bestTextBox)
	"""
end

# ╔═╡ 32f6a4a8-50e2-44e4-9323-5eb137431d82
begin
	fitted = linfit(df.x, df.y)
	SSEreg = SSE(df.y, [ones(length(df.x)) df.x], [a,b])
	SSEols = SSE(df.y, [ones(length(df.x)) df.x], [fitted[1], fitted[2]])
	p = plot()
	for i in 1:size(df,1)
		if plotResiduals
			plot!(p, [df.x[i], df.x[i]], [a+b*df.x[i],df.y[i]], 
				color = :lightgray, label = :none)
		end
	end
	Plots.abline!(p, b, a,  lw = 2, color = :black, label = "slider")
	if plotBest
		Plots.abline!(p, fitted[2], fitted[1],  lw = 2, color = :darkred, 
			label = "best fit")
	end

	scatter!(p, df.x, df.y, xlab = "x", ylab = "y", label = nothing)

	q = heatmap(bGrid, aGrid, colorbar_title="log10(SSE)",
			[log10(SSE(df.y, [ones(length(df.x)) df.x], [a,b])) 
				for a in aGrid, b in bGrid], c = :viridis, 
			xlab = "m", ylab = "b", legend = true)
	
	q = scatter!([b], [a], markersize = 6, 
		series_annotations = text.(round(SSEreg, digits = 2), :top, :white, 10), label = "slider", color = :black)
	q = scatter!([fitted[2]], [fitted[1]],  markersize = 6, 
		series_annotations = text.(round(SSEols, digits = 2), :top, :yellow, 10), label = "best fit", color = :darkred)

	plot(p,q, layout = grid(1,2), size = (700,300), bottom_margin = 30px, 
		left_margin = 10px)
end

# ╔═╡ 7c3c6ddb-bb9f-456b-938f-451ac5efd5dc
md"""
## Estimating the Coefficients

We have ``n`` dependent points ``\vec{x}`` and ``n`` corresponding observations ``\vec{y}``. For example, ``\vec{x}`` = vehicle weight and ``\vec{y}`` = fuel consumption. (We would expect that fuel consumption increases with vehicle weight). Using the model ``y = mx + b`` thus results in ``n`` equations with two unknowns:

```math
\begin{eqnarray}
\hat{y}_1 &= mx_1 + b  \\
\hat{y}_2 &= mx_2 + b  \\
\vdots \\
\hat{y}_n &= mx_n + b  \\
\end{eqnarray}
```

In matrix form:

```math
\begin{bmatrix}
\hat{y}_1 \\ \hat{y}_2 \\ \vdots \\ \hat{y}_n
\end{bmatrix} = 
\begin{bmatrix}
1 & x_1 \\
1 & x_2 \\
\vdots & \vdots \\
1 & x_n \\
\end{bmatrix}
\begin{bmatrix}
b \\
m
\end{bmatrix}
```

```math
\vec{\hat{y}} = \bf{A}\vec{\beta} 
```

and ``\vec{\beta} = \begin{bmatrix} b \\ m \end{bmatrix}`` are the sought coefficients. The norm to minimize is ``|| \vec{\hat{y}}-\vec{y} ||_2^2``. Thus we can formulate the problem via

```math
	\textrm{minimize}\; ||\mathbf{A}\vec{\beta} - \vec{y}||_2^2.
```

First, we expand the squared-norm of the residual vector:
```math
	||\,\mathbf{A}\mathbf{\vec{\beta}} - \vec{y} \,||_2^2
	=
	\left(\mathbf{A}{\vec{\beta}} - \vec{y} \right)^\intercal
	\left(\mathbf{A}{\vec{\beta}} - \vec{y} \right)
	=
	\mathbf{\vec{\beta}}^\intercal\mathbf{A}^\intercal \mathbf{A}{\vec{\beta} 
	- 2\mathbf{\vec{\beta}}^\intercal\mathbf{A}^\intercal \vec{y}
	+ \vec{y}^\intercal \vec{y}} 
```

Since we would like to find a ``{\vec{\beta}}`` that minimizes the expression, we take the [matrix-derivative](https://www.matrixcalculus.org/) with respect to ``{\vec{\beta}}`` and set the expression to 0, i.e.
```math
	\dfrac{d}{d\mathbf{\vec{\beta}}}\left(\mathbf{\vec{\beta}}^\intercal\mathbf{A}^\intercal \mathbf{A}{\vec{\beta} 
	- 2\mathbf{\vec{\beta}}^\intercal\mathbf{A}^\intercal \vec{y}
	+ \vec{y}^\intercal \vec{y}}\right)
	=
	2\mathbf{A}^\intercal \mathbf{A}\mathbf{\vec{\beta}}
	-2\mathbf{A}^\intercal \vec{y}
	=
	0
```

which gives the expression for $\mathbf{\vec{\beta}}$:
```math
	\mathbf{A}^\intercal \mathbf{A}\mathbf{\vec{\beta}}
	=
	\mathbf{A}^\intercal \vec{y}
	\quad \rightarrow \quad
	{\vec{\beta}}
	=
	\left(\mathbf{A}^\intercal \mathbf{A}\right)^{-1}\mathbf{A}^\intercal \vec{y}
```

Furthermore, the sum square error is 

```math
SSE = || A\vec{\beta} - \vec{y} ||_2^2
```
## Example: Minimize SSE via Matrix
"""


# ╔═╡ 8b8ce0e8-1333-43af-afb4-a306546a128a
md"""
This is the example dataset used above
"""

# ╔═╡ b4d29f23-3910-4667-9c3b-9a6ae3a527bb
df

# ╔═╡ bb2dce04-b4b5-4166-8ea2-6a5135acdad9
# Thus the vector y is
y = df.y

# ╔═╡ ad47fc86-e8bf-4cfc-9902-1c548306f32a
# The Matrix A
A = [ones(length(df.x), 1) df.x]

# ╔═╡ 1ca8c66d-9041-4a8f-899f-35c35b2f1c82
# Regression coefficients β[1] = b and β[2] = m. Compare to "best fit" above!
β = inv(A'*A)*A'*df.y

# ╔═╡ d85306eb-a3a8-4075-b09b-8efab8f5efe2
# Sum square error. Compare to "best fit" above!
SSEcalc = norm(A*β - df.y)^2.0

# ╔═╡ 182ff5ae-d1d5-4232-a988-d58f5556d535
md"""
## Coefficient of Determination (R-Squared)

The coefficient of determination is given by 

```math
R^2 = 1 - \frac{SSE}{SSTotal}
```

where ``SSTotal = \sum_i \left ( \vec{y_i} - \bar{\vec{y}} \right )^2 = || \vec{y} - \bar{\vec{y}} ||_2^2`` 

A concise definition for ``R^2`` is that ``R^2 \times 100`` percent of the variation in y is reduced by taking into account predictor.

If ``R^2 = 1`` all of the variation is explained by the model. If ``R^2 = 0`` none of the variation is explained by the model.
"""

# ╔═╡ c1cc1638-9a26-4374-ad61-8b2a56c29b7d
# Calculated using SSE and SST
Rsquare = 1 - SSEcalc/norm(y .- mean(y))^2

# ╔═╡ 1d88d4cd-c8fb-4f04-9746-89e52ffe712d
# Calculated using the Julia cor function
cor(df.x, df.y).^2

# ╔═╡ 99ee2478-9029-422c-91ca-410492a103c7
md"""
## The Mysterious `linfit` Function

The regression coefficients can be found using a one-liner in Julia (and other languages)

```julia
linfit(x, y) = [ones(length(x), 1) x] \ y 
```

The `[ones(length(x), 1) x]` is the matrix A and the `\` is the Matrix division using a polyalgorithm. The solver that is used depends upon the structure of A. The idea is that `A \ y` solves for the equation ``A\beta = y`` 

Thus we could write alternatively:
"""

# ╔═╡ ed5f2181-eb9b-4238-9e91-438d7ccc3f9d
## Compare to β above
β1 = A \ y

# ╔═╡ b18ecd31-c760-41b9-9037-613da500101f
## Or (the linfit function is defined in the header and must be defined in your script
linfit(df.x, df.y)

# ╔═╡ 620341d3-bfbf-437a-a5cc-d146fafa9067
md"""
The advantage of the approach is that the numerical algorithm to find ``\beta`` is optimized.
"""

# ╔═╡ 2a186fc5-e755-417f-883b-c5f58b834c97
md"""
## Example: Regression on Real Data

The dataset below shows the fuel efficiency alongside a number of parameter for a number of vehicles. To conduct regression analysis the first step is to generate a plot of the dependent vs. the independent variable.
"""

# ╔═╡ 9b6ddee1-54cb-4b87-b10c-0d2b13a7ebdb
begin
	fptr = HTTP.get("https://mdpetters.github.io/cee200/assets/auto.txt")
	l = split(String(fptr.body), "\n")
	
	function splitit(line)
		x = string(line)
		m = split(x, "\t")
		n = split(m[1])

		o = m[2][2:end-1]
		mpg = parse(Float64, n[1])
		cylinders = parse(Int, n[2])
		displacement = parse(Float64, n[3])
		hp = parse(Float64, n[4])
		weight = parse(Float64, n[5])
		acceleration = parse(Float64, n[6])
		year =  parse(Int, n[7])
		return DataFrame(mpg = mpg, cylinders = cylinders, 
			displacement = displacement, hp = hp, 
			weight = weight, acceleration = acceleration, year = year, model = o)
	end
	
	df1 = mapfoldl(i -> splitit(l[i]), vcat, 1:length(l)-1)
end

# ╔═╡ 5a2d4a7b-a0eb-4e8d-b7e9-198976745cd2
begin
	xvar = df1.weight
	yvar = 100.0./df1.mpg
	scatter(xvar, yvar, xlabel = "weight (lb)", ylabel = "Gallons/100 Miles", 
		color = :lightgray, label = "data")
	coeff = linfit(xvar, yvar)
	R2 = round(cor(xvar, yvar).^2, digits = 2)
	xs = 0:1000:6000
	plot!(xs, coeff[1] .+ coeff[2].*xs, color = :black, label = "y = mx + b, R² = $(R2)")
end

# ╔═╡ dd6a5c2a-0ed5-4552-b094-368d3bd57577
md"""

## Extension: Multiple Linear Regression

The above analysis is easily extended to multiple variables. Suppose we want to fold in horespower in addition to weight as explanatory variable.


We have ``n`` dependent points ``\vec{w}`` (weight) and ``\vec{h}`` (horespower) and ``n`` corresponding observations ``\vec{y}`` (fuel consumption). Using the model ``y = c_1w + c_2h + c_3 wh + c_4`` thus results in ``n`` equations with four unknowns:

```math
\begin{eqnarray}
\hat{y}_1 &= c_1w_1 + c_2h_1 + c_3w_1h_1 + c_4  \\
\hat{y}_2 &= c_1w_2 + c_2h_2 + c_3w_2h_2 + c_4  \\
\vdots \\
\hat{y}_n &= c_1w_n + c_2h_n + c_3w_nh_n + c_4  \\
\end{eqnarray}
```

In matrix form:

```math
\begin{bmatrix}
\hat{y}_1 \\ \hat{y}_2 \\ \vdots \\ \hat{y}_n
\end{bmatrix} = 
\begin{bmatrix}
w_1 & h_1 & w_1h_1 & 1 \\
w_2 & h_2 & w_2h_2 & 1 \\
\vdots & \vdots \\
w_n & h_n & w_nh_n & 1 \\
\end{bmatrix}
\begin{bmatrix}
c_1 \\
c_2 \\ 
c_3 \\
c_4 
\end{bmatrix}
```

```math
\vec{\hat{y}} = \bf{A}\vec{\beta} 
```

This is trivially solved:
"""

# ╔═╡ c4b7dc55-bcd5-4c12-9fb4-af83b40ecd53
# Solve for c coefficients
c_coeff = let 
	w = df1.weight
	h = df1.hp
	y = 100.0./df1.mpg
	A = [w h w.*h ones(length(y), 1)]
	A \ y 
end

# ╔═╡ 340a16cc-9b66-4aa8-8849-d75de48195cb
let 
	xvar = c_coeff[1].*df1.weight .+ c_coeff[2].*df1.hp .+ 
		c_coeff[3].*df1.weight .* c_coeff[3].*df1.hp .+ c_coeff[4]
	yvar = 100.0./df1.mpg
	R2 = round(cor(xvar, yvar)^2, digits = 2)
	scatter(xvar, yvar, xlabel = L"c_1w_i + c_2h_i + c_3w_ih_i + c_4", 
		ylabel = "Gallons/100 Miles", color = :lightgray, label = "Model R² = $(R2)")
	
end

# ╔═╡ 6497be76-a639-430f-963b-0486e14151b2
md"""

# The Moore–Penrose Pseudo-Inverse 

The Moore-Penrose inverse, also sometimes referred to as the Pseudo-Inverse is defined by the pre-term in the ordinary least squares solution. For any matrix ``\mathbf{A}``, there is always one and only one pseudoinverse ``\mathbf{A^{+}}``

```math
\mathbf{A}^+ = \left(\mathbf{A}^\intercal \mathbf{A}\right)^{-1}\mathbf{A}^\intercal
```
"""

# ╔═╡ 4cfff7c9-de08-43d2-8551-c166d22df965
md"""
# Integral Equations

## Detour: Ordinary Differential Equation

Integral equations are the mirror image of differential equations.

### First order differential equation

```math
\frac{dy}{dx} = u(x) \quad u(a) =  u_0,
```

Intgration gives

```math
y(x) = u(a) + \int_a^x u(t) dt
```

### Second order differential equation

```math
\frac{d^2y}{dx^2} = u(x) \quad u(a) =  u_0,
```

Intgration gives

```math
y(x) = u(a) + x y'(a) + \int_a^x (x-t) u(t) dt 
```math

Now let's assume ``u(a) = 0`` and ``y'(a) = 0`` and reorder

```math
\int_a^x (x-t) u(t) dt = y(x)
```

Now let's generalize the ``(x-t)`` function as Kernel function ``K(x,t)``

```math
\int_a^x K(x,t) u(t) dt = y(x)
```


## Fredholm Integral Equation 

A [Fredholm integral equation of the first kind](https://mathworld.wolfram.com/FredholmIntegralEquationoftheFirstKind.html) is defined as

$\int_{a}^{b}K(x,t)u(t)dt=y(x)$
 
where the integration limits ``a`` and ``b`` do not depend on ``x``. Note that the Fourier transform is a Fredholm integral equation with ``a = -\infty`` and ``b = \infty``.

```math
\begin{array}
a & & \displaystyle\mathcal{F}\{f(t)\} =& \int_{-\infty}^{\infty} f(t)e^{ix t} dt = F(x) \\
& & \displaystyle\mathcal{F}\{f(t)\} =& \int_{-\infty}^{\infty} K(x, t) f(t) dt = F(x)
\end{array}
```

where ``K(x, t) = e^{ix t}``, where ``x`` is the frequency. Thus another way to think about integral equations is that they are *domain transformations* from the $t$-domain to the $x$-domain. 


## Volterra Integral Equation 

A [Volterra integral equation of the first kind](https://mathworld.wolfram.com/VolterraIntegralEquationoftheFirstKind.html) is defined as

$\int_{a}^{x}K(x,t)u(t)dt=y(x)$
 
The distinction to the Fredholm integral equation is that the integration limits  depend on ``x``.

"""


# ╔═╡ 341081cc-a36b-4cdb-99e7-9ac34730b254
md"""

## Integral Equations in Science and Engineering

- Surface temperature from interior measurements
- Measurement reconstruction (e.g. size distribution from mobility measurements)
- Inference of particle microphysical properties from optical and scattering measurements
- Subsurface density from surface microgravity measurements
- Image reconstruction (e.g. astronomy or medical) 
- Optical Interferometry (e.g. FTIR)
- Any application that uses spectral analysis 

These problems have in common that the taken measurement is the result of a convolution of the quantity of interest and a "transfer function" that produces the observed result. 

"""

# ╔═╡ d415e3d4-3034-4bab-8ee0-e2645e6daa8f
md"""

# Data Inversion of Integral Equations


The general Fredholm integral equation is

```math
\int_a^b K(x,t) u(t) dt = y(x)
```

This kernel serves as an illustrative example

$K(x,t)=\exp(x\cos(t))$

$u(t)=\sin(t)$

$y(x)=\frac{2\sinh x}{x}$

$t\in[0,\pi]\;\mathrm{and}\;x\in[0,\pi/2]$

Thus, the identity 

```math
\int_0^\pi \exp(x\cos(t)) \sin(t) dt = \frac{2\sinh x}{x}
```

holds. We can test this through `SymPy`:

"""



# ╔═╡ 90e078ad-088a-4415-bcc5-44fc01dc8406
let 
	@syms x::positive t 
	integrate(exp(x*cos(t))*sin(t), (t, 0, π)) |> simplify
end

# ╔═╡ ec5da1c6-3952-4576-b09d-b84af7b14bb2
md"""
The Fredholm integral equation 

```math
\int_a^b K(x,t) u(t) dt = y(x) \quad t \in [a, b] \;\; x \in [d, e]
```

can be discretized into a system of linear equations such that

$\mathbf{A}\vec{u}=\vec{y}$

where $\vec{u}=<u_{1},\dots,u_{n}>$ is a discrete vector representing ``u(t)``, $\vec{y}= <y_{1},\dots,y_{m}>$ is a vector representing $y(x)$ and $\mathrm{\mathbf{A}}$ is the design matrix.

Using the quadrature method, the integral is approximated by a weighted sum such that

$\int_{a}^{b}K(x,t)u(t)dt \rightarrow y_j \approx\sum_{i=1}^{n}w_nK(x_{j},t_{i})u(t_{i})$

where $w_n=\frac{b-a}{n}$, $w_m = \frac{c-d}{m}$, $t_{i}=(i-\frac{1}{2})w_n$, and $x_{j} = (j - \frac{1}{2})w_m$. The elements comprising the design matrix $\mathrm{\mathbf{A}}$ are $a_{j,i}=w_nK(x_{j},t_{i})$. The matrix $\mathbf{A}$ is $\mathbb{R}^{m\times n}. $ If $n = m$, the matrix is square.

Below is an example how to calculate the matrix ``\mathbf{A}\vec{u}``. The discrete solution ``\mathbf{A}\vec{u}`` and ``\vec{y}`` are close, but not identical due to discretization errors.
"""

# ╔═╡ 97567bbc-f652-4d54-81ae-028bb4b27b6b
function discretize_baart(n, m)
	A = zeros(m,n)
	a, b = 0.0, π
	c, d = 1e-20, π/2
	wₙ = (b-a)/n
	wₘ = (d-c)/m
	K(x,t) = exp(x*cos(t))
	
	for i = 1:n
	    for j = 1:m
	        A[j,i] = wₙ*K((j-0.5)*wₘ, (i-0.5)*wₙ)
	    end
	end

	t = [(i-0.5)*wₙ for i = 1:n]
	x = [(j-0.5)*wₘ for j = 1:m]

	u = sin.(t)
	y = 2.0*sinh.(x)./x
	A, u, A*u, y, t, x
end

# ╔═╡ 3ab8590a-bb4e-4e6b-9b53-3603c23aaeeb
@bind sv combine() do Child
	md"""
	n  $(
		Child(Slider(5:1:100, default = 15))
	) 
	"""
end

# ╔═╡ 59733493-4be0-4650-a83a-e4279b0174ae
@bind svm combine() do Child
	md"""
	m $(
		Child(Slider(15:1:100, default = 15))
	) 
	"""
end

# ╔═╡ 5cd36360-bd59-44c9-850e-79e5de8a21e8
myA, myu, myAu, myy, myt, myx = discretize_baart(sv[1],svm[1])

# ╔═╡ 3fa1bdd3-3fd0-43d4-b450-467fe636e9dc
md"""
##### Visualize
"""

# ╔═╡ 408f9681-325d-4d6f-89be-2ec8b518eac9
let
	t = 0.0:0.01:π
	x = 1e-10:0.001:(π/2)
	y = 2.0*sinh.(x)./x
	p1 = plot(t, sin.(t), xlabel = "t", ylabel = "u", label = L"u(t) = sin(t)", 
		color = :black)
	p1 = scatter!(myt, myu, label = L"\vec{u}", title = "t-domain")
	p2 = plot(x, y, xlabel = "x", ylabel = "y", label = L"y(x) = \frac{2sinh(x)}{x}", 
		color = :black)
	p2 = scatter!(myx, myA*myu, label = L"\vec{y} = \mathbf{A}\vec{u}", 
		title = "x-domain")
	plot(p1, p2, layout = grid(1,2), size = (700,300), bottom_margin = 15px)
end

# ╔═╡ 507d1240-a0a2-4979-8856-2790c3c34eca
Markdown.MD(
	Markdown.Admonition("warning", "Key Concepts", [md"

- Note that the matrix $\mathbf{A}$ need not be square. That is we can map from m points in $u(t)$ space to n points in $y(x)$ space. 
- Typically we observe $y(x)$ and then apply the inversion to find the true $u(t)$
	
	"]))

# ╔═╡ 17bcf2c6-5b89-4703-be18-c7c6c0433508
md"""

## Inversion by Ordinary Least Squares Regression

The inverse problem is to find $u(t)$ for a known kernel $K(x,t)$ and set of measurements $y(x)$. Thus we seek the best ``\vec{u}`` that matches observations  ``\vec{u}``. This is the same as the regression problem.


```math
	\textrm{minimize}\; ||\mathbf{A}\vec{u} - \vec{y}||_2^2.
```

to which we already know the solution:

```math
{\hat{\vec{u}}}
	=
	\left(\mathbf{A}^\intercal \mathbf{A}\right)^{-1}\mathbf{A}^\intercal \vec{y}
```

where ``\hat{\vec{u}}`` is the estimated inverted solution

"""


# ╔═╡ 151dc9af-5099-42be-86a9-f0ab59646b67
@bind sn2 combine() do Child
	md"""
	n  $(
		Child(Slider(5:1:100, default = 8))
	) 
	"""
end

# ╔═╡ f52780fe-14d6-476c-9cfc-a656c09a7a01
let
	A, u, Au, y, t, x = discretize_baart(sn2[1],sn2[1])
    uhat = (A'*A)^(-1)*A'*y

	p1 = plot(y, xlabel = "Sample # (n)", ylabel = "Observation y", 
		marker = :circle, color = :black, label = L"\vec{y}") 
	p2 = plot(u, xlabel = "Sample # (n)", color = :black, label = L"u(t)")
	p2 = plot!(uhat, color = :darkred, 
		label = 
			L"(\mathbf{A}^{\intercal}\mathbf{A})^{-1}\mathbf{A}^{\intercal} \vec{y}", 
		marker = :circle)
	plot(p1, p2, layout = grid(1,2), size = (700, 300), left_margin = 10px, bottom_margin = 15px)
end

# ╔═╡ f8918afb-d4cd-4d6e-b030-955a9434c2ff
md"""
## Ill-Posed Problems

In a well posed problem, three fundamental conditions must be satisfied:

1. The problem must have a unique solution.
2. The solution must depend continuously on the data or the parameters.
3. The solution must be stable against small changes in the data or the parameters.


"""

# ╔═╡ d26edc69-bd85-451d-99cc-5c0eee4aed40
Markdown.MD(
	Markdown.Admonition("warning", "Key Concept", [md"

Ill posed problems violate one or more of these conditions, making them inherently more difficult to solve. Common characteristics of an ill posed problem are

1. Lack of a Unique Solution
2. Sensitivity to Initial Conditions (or System Noise)
3. Nonexistence of a Solution

The problem above is an ill posed problem because the solution is very sensitive to system noise."]))

# ╔═╡ 59375052-fb7a-4728-9118-f5096850da9c
md"""
### Detour: Matrix Inverse.

#### Rank and Rank Deficiency
The `rank` of matrix A is the dimension of the vector space generated (or spanned) by its columns. This corresponds to the maximal number of linearly independent columns of A. This, in turn, is identical to the dimension of the vector space spanned by its rows.

A matrix is said to have full rank if its rank equals the largest possible for a matrix of the same dimensions, which is the lesser of the number of rows and columns. A matrix is said to be rank-deficient if it does not have full rank.

Matrices with full rank is invertible. Rank-deficient matrices are invertible using the Moore–Penrose inverse.
"""

# ╔═╡ 01451338-1a69-47fe-9727-d5c6aaa16758
theA = [1 2; 0 0]

# ╔═╡ 7f9348f6-508e-4ac8-a4bc-e2a75d127097
rank(theA)

# ╔═╡ 7995877d-99e8-434b-9d54-1d50eec61f79
det(theA)

# ╔═╡ 1f9bad57-f122-4a5c-bdc7-90543e50a5a8
inv(theA)

# ╔═╡ e72ae865-e53a-415c-a4d4-f9ac0db96646
md"""

#### The Moore–Penrose Inverse Revisited

As derived above:

```math
\mathbf{A}^+ = \left(\mathbf{A}^\intercal \mathbf{A}\right)^{-1}\mathbf{A}^\intercal
```

The Moore-Penrose pseudo-inverse and solution has the following properties

1. If ``m = n``, ``\mathbf{A}^+ = \mathbf{A}^{−1}`` if ``\mathbf{A}`` is full rank. The pseudo-inverse for the case where ``\mathbf{A}`` is not full rank will be considered below
2. If ``m > n`` the solution is the one that minimizes the quantity ``||\mathbf{A}\vec{y} - \vec{u} ||_2^2``. That is, in this case there are more constraining equations than there are free variables. In this case it is not generally possible find a solution to these equations. The pseudo-inverse gives the solution ``\vec{y}`` such that ``\mathbf{A}^+\vec{y}`` is closest (in a least-squared sense) to the desired solution vector ``\vec{u}``.
3. If ``m < n`` , then the Moore-Penrose solution minimizes the 2-norm of ``\vec{y}``: ``||\vec{y}||_2``. In this case, there are generally an infinite number of solutions, and the Moore-Penrose solution is the particular solution whose vector 2-norm is minimal.

When A is full rank, the Moore-Penrose pseudo-inverse can be directly calculated as follows:

- If ``m > n``: ``\mathbf{A}^+ = \left(\mathbf{A}^\intercal \mathbf{A}\right)^{-1}\mathbf{A}^\intercal`` (see above)
- If ``m < n``: ``\mathbf{A}^+ = \mathbf{A}^\intercal \left(\mathbf{A}\mathbf{A}^\intercal\right)^{-1}`` (which is known as the right inverse)

However, when A is not full rank, then these formulas can not be used. More generally, the pseudo-inverse is best computed using the Singular Value Decomposition.

"""

# ╔═╡ 1ced6465-0899-4393-b93d-c3aa688a67f3
md"""
#### Singular Value Decomposition

Let ``\mathbf{A} \in \mathbb{R}^{m \times n}``. Then there exists orthogonal matrices ``\mathbf{U} \in \mathbb{R}^{m \times m}`` and ``\mathbf{V} \in \mathbb{R}^{n \times n}`` such that the matrix ``\mathbf{A}`` can be decomposed as follows:

```math
\mathbf{A} = \mathbf{U} \mathbf{\Sigma} \mathbf{V}^T
```

where ``\Sigma`` is an ``m \times n`` diagonal matrix having the form

```math
\Sigma = \begin{bmatrix}
\sigma_1 & 0 & 0 & \dots & 0 & 0 \\
0 & \sigma_2 & 0 & \dots & 0 & 0 \\
0 & 0 & \sigma_3 & \dots & 0 & 0  \\
\vdots & \vdots & \vdots & \dots & \vdots \\
0 & 0 & 0 & \dots & \sigma_p & 0\\
\end{bmatrix}
```

and

$\sigma_1 > \sigma_2 > \sigma_3 > \dots > \sigma_p \quad \quad p = \min\{m,n\}$

The ``<\sigma_i>`` are termed the *singular values* of the matrix ``\mathbf{A}``. The columns of ``\mathbf{U}`` are termed the left singular vectors, while the columns of ``\mathbf{V}`` are termed the right singular vectors. The decomposition described  is called the *Singular Value Decomposition*, which is conveniently abbreviated as SVD.

Note that the SVD can be obtained usign virtually any Linear Algebra package. 
"""

# ╔═╡ e6eed0fa-ff80-4898-94d4-d44653b8b898
# Notes
# 1. Σ is returned as a vector (skinny). Diagonal(Σ) converts it to a Matrix
# 2. The identity is only approximate due to numerical errors in the SVD algorithm
let 
	U, Σ, V = svd(myA)
	myA ≈ U*Diagonal(Σ)*V'
end

# ╔═╡ 6ad8f345-fd60-40fa-a61c-dff02137171a
md"""

#### The Moore-Penrose Inverse from the SVD

Using the SVD, the pseudo-inverse of a matrix can be easily computed as follows. 

$\mathbf{A}^+ = \mathbf{V} \mathbf{\Sigma}^+ \mathbf{U}^T$

where the matrix ``\mathbf{\Sigma}^+`` takes the form:

```math
\mathbf{\Sigma}^+ =  
\begin{bmatrix}
\frac{1}{\sigma_1} & 0 & 0 & \dots & 0 & 0 \\
0 & \frac{1}{\sigma_2} & 0 & \dots & 0 & 0 \\
0 & 0 & \frac{1}{\sigma_3} & \dots & 0 & 0  \\
\vdots & \vdots & \vdots & \dots & \vdots \\
0 & 0 & 0 & \dots & \frac{1}{\sigma_p} & 0\\
\end{bmatrix}
```

for all of the non-zero singular values. If any of the ``σ_i`` are zero, then a zero is placed in corresponding entry of ``\mathbf{\Sigma}^+``. If the matrix ``\mathbf{A}`` is rank deficient, then one or more of its singular values will be zero. Hence, the SVD provides a means to compute the pseudo-inverse of a singular matrix.

"""

# ╔═╡ 6900e0ed-615e-4ede-87ca-0ba44382438b
let 
	U, Σ, V = svd(myA)
	pinv(myA) ≈ V*Diagonal(1.0./Σ)*U'
end

# ╔═╡ 83fc9f05-12ac-4c93-8d92-3590e892dd1e
md"""
### Consequence of Near-Zero Singular Values

Notice that the computation of ``\mathbf{A}^+`` via the SVD includes ``<\frac{1}{\sigma_i}>`` terms. If ``{\sigma_i}> = 0`` this is infinity and hence the term is removed for the pseudo inverse. The matrix is rank deficient, the true inverse is not defined, there are infinite solutions, and the proplem is ill-posed. Now consider near zero singular values. This results in very large ``<\frac{1}{\sigma_i}>``. These large entries in ``\mathbf{A}^+`` are then multiplied the observations ``\vec{y}`` to esitmate the solution:

```math
{\hat{\vec{u}}}
	=
	\left(\mathbf{A}^\intercal \mathbf{A}\right)^{-1}\mathbf{A}^\intercal \vec{y} = \mathbf{A}^+  \vec{y}
```

Small errors in ``\vec{y}`` amplify, thus causing the poor performance of the linear regression solution.

As shown in the plot below for the singular values decay near exponentially. As the number of samples increases the values ``<\sigma_i>`` drop into the numerical noise.

A consequence of near-zero singular values is that the matrix is **effectively rank-deficient** (and thus not invertible, and thus the problem is ill-posed) though it will appear as full rank. 
"""

# ╔═╡ d02b8d90-24e8-420f-934c-f3bc4d62554f
Markdown.MD(
	Markdown.Admonition("danger", "Random Error", [md"
So far the only error in $\vec{y}$ are discretization errors from our forward calculation. In practice, additional measurement errors will be superimposed on y such that 

```math
\vec{y} = \vec{y}_{ideal} + \vec{\epsilon}
```

where $\epsilon$ is a random perturbation on the observation $\vec{y}$. 
"]))

# ╔═╡ 3ad80698-f495-4740-9ece-dbf908bd9734
@bind sn3 combine() do Child
	md"""
	n  $(
		Child(Slider(5:1:100, default = 8))
	) 
	"""
end

# ╔═╡ 0d399316-d9b9-4980-bff0-d60d8c3fdb94
@bind sn4 combine() do Child
	md"""
	Random error (fraction): ϵ  $(
		Child(Slider(0:0.001:0.01, default = 0))
	) 
	"""
end

# ╔═╡ aabdeb85-2ca7-43ae-b413-92cc73c5fff1
let
	A, u, Au, y, t, x = discretize_baart(sn3[1],sn3[1])
	y1 = y + sn4[1]*(rand(length(y)) .- 0.5)
    uhat = A \ y1 
	#(A'*A)^(-1)*A'*y # Use A \ y to solve for better numerical resilience

	p1 = plot(y1, xlabel = "Sample # (n)", ylabel = "Observation y", 
		marker = :circle, color = :black, label = L"\vec{y}") 
	p1 = plot!(y, label = "Noise free")
	p2 = plot(u, xlabel = "Sample # (n)", color = :black, label = L"u(t)")
	p2 = plot!(uhat, color = :darkred, 
		label = 
			L"\mathbf{A} \backslash \vec{y}", 
		marker = :circle)

	U, Σ, V = svd(A)
	p3 = plot(Σ, yscale = :log10, color = :black, ylabel = "σᵢ", 
		xlabel = "Sample # (m)", 
		ylim = [1e-20, 10], label = :none)
	p12 = plot(p1, p2, layout = grid(1,2))
	plot(p12, p3, layout = grid(2,1), size = (700, 500), left_margin = 10px, bottom_margin = 15px)
end

# ╔═╡ 46a31a35-e5f9-4e71-8d44-7093c6feb6a1
Markdown.MD(
	Markdown.Admonition("info", "Exercise", [md"
- Change the number of samples n using the slider above.
- As you increase n, observe how the ordindary least squares inversion becomes noisy, and eventually useless. At the same time, observe how the singular values decay into the machine precision (~2e-16) for a Float64.
- Even imperceptible noise superimposed on the observation can *destroy* the solution
"]))

# ╔═╡ 42bd1e5f-25ea-4375-8a9c-3c5e8ded2867
Markdown.MD(
	Markdown.Admonition("warning", "Key Concept", [md"
Machine epsilon or machine precision is an upper bound on the relative approximation error due to rounding in floating point arithmetic. This value characterizes computer arithmetic in the field of numerical analysis, and by extension in the subject of computational science. 

In `Julia`, the epsilon of a data type can be determined using `eps(x)`.
"]))

# ╔═╡ f7dbe9d9-b709-47fe-9a7b-9a6cedb8d3c9
eps(Float64(1.0))  # Precision of 64 bin floating point

# ╔═╡ df2601bc-e560-4801-93bf-226700e1bde8
eps(Float32(1.0))  # Precision of 32 bin floating point

# ╔═╡ e386f4ba-3cce-44e1-b289-082afb765e37
Markdown.MD(
	Markdown.Admonition("warning", "Key Concepts", [md"
- The purpose of the SVD is to express the matrix in the terms of their singular values.  
- The SVD is provides a numerical method to compute the pseudo inverse.
- The SVD is a rank-revealing algorithm, i.e. allows you the measure the rank of a matrix.
- The SVD is computationally expensive (``> O(n^2)``) and thus calculating the SVD for large problems (``n,m > 10^6``) is not feasible.
"]))

# ╔═╡ da679600-4485-47c1-9999-6d4392a0e6f2
md"""
##  Inversion by Regularization

Tikhonov regularization is a means to filter this noise by solving the minimization problem 

```math
{ {\vec{u}_{\lambda}}}=\mathrm{minimize}\left\{ \left\lVert {{{\mathbf{A}}{\vec{ u}}-{\vec{y}}}}\right\rVert _{2}^{2}+\lambda^{2}\left\lVert { {\bf L}({\vec{u}}-{\vec{u}_{0}})}\right\rVert _{2}^{2}\right\} 
```

where ``\vec{u}_{\lambda}`` is the regularized estimate of ``\vec{u}``,
``\left\lVert \cdot\right\rVert _{2}`` is the Euclidean norm, ``{\rm {\bf L}}`` is the Tikhonov filter matrix, ``\lambda`` is the regularization parameter, and ``\vec{u}_{0}`` is a vector of an *a priori* guess of the solution. The initial guess can be taken to be ``\vec{u}_{0}=0`` if no *a priori* information is known. The matrix ``{\mathbf{A}}`` does not need to be square. 

For ``\lambda=0`` the Tikhonov problem reverts to the ordinary least
squares solution. If ``\mathbf{A}`` is square and ``\lambda=0``,
the least-squares solution is ``\vec{u}_\lambda={{\mathbf{A}}^{+}\vec{y}}``. For large ``\lambda`` the solution reverts to the initial guess,
i.e. ``\lim_{\lambda\rightarrow\infty}{\vec{u}_{\lambda}}={\vec{u}_{0}}``.
Therefore, the regularization parameter ``\lambda`` interpolates between
the initial guess and the noisy ordinary least squares solution. The
filter matrix ``{{\mathbf{L}}}`` provides additional smoothness constraints on the solution. The simplest form is to use the identity matrix, ``{\mathbf{L}} =\mathbf{I}``.

The formal solution to the Tikhonov problem is given by

```math
\vec{u}_{\lambda}=\left({\rm {\bf A}^{T}}{\rm {\bf A}}+\lambda^{2}{{\bf L}^{T}{\rm {\bf L}}}\right)^{-1}\left({\rm {\bf A}^{T}}{\vec{y}}+\lambda^{2}{{\mathbf{L}}^{T}{{\mathbf{L}}}\vec{u}_{0}}\right)
```

The equation is readily derived by writing ``f=\left\lVert {{{\mathbf{A}}{\vec{ u}}-{\vec{y}}}}\right\rVert _{2}^{2}+\lambda^{2}\left\lVert {{\mathbf{L}}({\vec{u}}-{\vec{u}_{0}})}\right\rVert _{2}^{2}``,
take ``\frac{df}{d{\rm {\rm x}}}=0``, and solve for ``\vec{u}``. Use
[http://www.matrixcalculus.org/](http://www.matrixcalculus.org/)
to validate symbolic matrix derivatives.

"""

# ╔═╡ 45835413-d81b-4816-8f9b-ef7060efa272
Markdown.MD(
	Markdown.Admonition("info", "Exercises", [md"
1. Show that for ``\lambda = 0``, the Tikhonov solution is equal to 
	
```math
{\hat{\vec{u}}}
	=
	\left(\mathbf{A}^\intercal \mathbf{A}\right)^{-1}\mathbf{A}^\intercal \vec{y} = \mathbf{A}^+  \vec{y}
```

2. Show that ``{\vec{u}_{\lambda}} = \mathrm{minimize}\left\{ \lambda^{2}\left\lVert { {\bf L}({\vec{u}}-{\vec{u_{0}}})}\right\rVert _{2}^{2}\right\}`` implies 
	
$\vec{u_{\lambda}} = \vec{u_0}$

3. Show that ``\mathbf{L} = \mathbf{I}`` and ``\vec{u_0} = <0, \dots, 0>`` implies 
	
```math
\vec{u}_{\lambda} = \left({\rm {\bf A}^{T}}{\rm {\bf A}}+\lambda^{2} \mathbf{I}\right)^{-1}{\rm {\bf A}^{T}}{\vec{y}}
```

This simplified version of Tikhonov regularization is also known as *Ridge Regression*
"]))

# ╔═╡ 018e7cba-d12c-4b9d-999a-98f4908d308f
md"""
Number of samples: $(@bind sv1 Slider(5:1:100, default = 10))
"""

# ╔═╡ b3ed3084-b382-47b5-94d1-939afde61036
md"""
Regularization Parameter λ: $(@bind λ Slider([0; 1e-9:1e-9:1e-7;1e-7:1e-7:1e-5; 1e-5:1e-5:1e-3; 1e-3:1e-3:1e-1; 1e-1:1e-1:10; 10:10:1000], default = 1e-6))
"""

# ╔═╡ 4c2bbd0c-fa7a-4892-9a00-3ed740b5450a
let
	A, u, Au, y, t, x = discretize_baart(sv1[1],sv1[1])
	
	uhat = ((A'*A)^-1 * A') * y
	
	p0 = plot(y, xlabel = "Sample # (n)", ylabel = "Observation y", 
		marker = :circle, color = :black, label = L"\vec{y}") 

	p1 = plot(u, xlabel = "Sample # (n)", color = :black, label = L"u(t)")
	ymax = 1.5*maximum(uhat)
	ymin = minimum(uhat)
	p1 = plot!(uhat, color = :darkred, ylim = [ymin, ymax], label = 
		L"\vec{u} = (\mathbf{A}^\intercal \mathbf{A})^{-1} \mathbf{A}^\intercal \vec{y}")

	Id = Matrix(I, sv1[1], sv1[1])
	uhat2 =  ((A'*A + λ^2.0*Id)) \ (A' * y)
	p2 = plot(u, xlabel = "Sample # (n)", color = :black, label = L"u(t)")
	p2 = plot!(uhat2, color = :darkred, label = L"\vec{u}_{\lambda} = \left({\rm \mathbf{A}^{\intercal}}{\rm \mathbf{A}}+\lambda^{2} \mathbf{I}\right)^{-1}{\rm \mathbf{A}^{\intercal}}{\vec{y}}", ylim = [0,1.5], title = "λ = $(λ)")

	plot(p0, p1, p2, layout = grid(1,3), size = (700, 300), 
		left_margin = 10px, bottom_margin = 15px)
end

# ╔═╡ 6f24d102-3b91-440e-aeb7-2a8e1abf7558
Markdown.MD(
	Markdown.Admonition("warning", "Key Concepts", [md"
- Note that even the simplest form of regularization is excellent at removing noise.
- Note how the solution converges to the initial guess or zero for $\lambda = 0$ and $\lambda = \infty$, respectively.
- However, there is no obvious choice for the best regularization parameter.
"]))

# ╔═╡ 0a784c5f-bc9b-4a9f-9ae8-9c7aa82f7192
md"""
## Selecting the Regularization Parameter

### Residual and Solution Norm

The regularized inverse contains two terms:

```math
{ {\vec{u}_{\lambda}}}=\mathrm{minimize}\left\{ \left\lVert {{{\mathbf{A}}{\vec{ u}}-{\vec{y}}}}\right\rVert _{2}^{2}+\lambda^{2}\left\lVert { {\bf L}({\vec{u}}-{\vec{u}_{0}})}\right\rVert _{2}^{2}\right\} 
```

The first term is called the *residual norm*:

```math
R = \left\lVert {{{\mathbf{A}}{\vec{ u}}-{\vec{y}}}}\right\rVert
```

The second term is called the *solution norm*

```math
S = \left\lVert { {\bf L}({\vec{u}}-{\vec{u}_{0}})}\right\rVert
```

### Morozov's Discrepancy Principle

The discrepancy principle, due to Morozov, chooses the regularization parameter to be the largest value of $\lambda$

such that the residual norm is bounded by the noise level in the data, i.e.,

```math
\left\lVert {{{\mathbf{A}}{\vec{u}_\lambda}-{\vec{y}}}}\right\rVert  < \delta
```

where $\delta$ is the noise level. Here, $\vec{u}_\lambda$ denotes the parameter found minimizing the Tikhonov regularized minimization problem with parameter $\lambda$. This choice aims to avoid overfitting of the data, i.e., fitting the noise. The main shortcoming of the approach is that the noise level in the data must be known.
"""

# ╔═╡ 771b2e05-d6bc-4a27-a4c1-930332535d06
md"""
Other metrics like generalized cross validation or the L-curve criterion are available to automate the search for the optimal λ. In general, the approach selected may depend on how \"bad\" the matrix is, or how ill-conditioned the problem is. Plotting the behavior of the residual and solution norm may help decide on the criterion.
"""

# ╔═╡ 00000000-0000-0000-0000-000000000001
PLUTO_PROJECT_TOML_CONTENTS = """
[deps]
CSV = "336ed68f-0bac-5ca0-87d4-7b16caf5d00b"
ColorSchemes = "35d6a980-a343-548e-a6ea-1d62b119f2f4"
DataFrames = "a93c6f00-e57d-5684-b7b6-d8193f3e46c0"
HTTP = "cd3eb016-35fb-5094-929b-558a96fad6f3"
LaTeXStrings = "b964fa9f-0449-5b57-a5c2-d3ea65f4040f"
LinearAlgebra = "37e2e46d-f89d-539d-b4ee-838fcccc9c8e"
Plots = "91a5bcdd-55d7-5caf-9e0b-520d859cae80"
PlutoUI = "7f904dfe-b85e-4ff6-b463-dae2292396a8"
Random = "9a3f8284-a2c9-5f02-9a11-845980a1fd5c"
StatsBase = "2913bbd2-ae8a-5f71-8c99-4fb6c76f3a91"
SymPy = "24249f21-da20-56a4-8eb1-6a02cf4ae2e6"

[compat]
CSV = "~0.10.11"
ColorSchemes = "~3.23.0"
DataFrames = "~1.6.1"
HTTP = "~1.9.14"
LaTeXStrings = "~1.3.0"
Plots = "~1.38.17"
PlutoUI = "~0.7.52"
StatsBase = "~0.34.0"
SymPy = "~1.1.12"
"""

# ╔═╡ 00000000-0000-0000-0000-000000000002
PLUTO_MANIFEST_TOML_CONTENTS = """
# This file is machine-generated - editing it directly is not advised

julia_version = "1.9.3"
manifest_format = "2.0"
project_hash = "1bb3a28e34f82aac50484e87577840821859f0ad"

[[deps.AbstractPlutoDingetjes]]
deps = ["Pkg"]
git-tree-sha1 = "91bd53c39b9cbfb5ef4b015e8b582d344532bd0a"
uuid = "6e696c72-6542-2067-7265-42206c756150"
version = "1.2.0"

[[deps.ArgTools]]
uuid = "0dad84c5-d112-42e6-8d28-ef12dabb789f"
version = "1.1.1"

[[deps.Artifacts]]
uuid = "56f22d72-fd6d-98f1-02f0-08ddc0907c33"

[[deps.Base64]]
uuid = "2a0f44e3-6c83-55bd-87e4-b1978d98bd5f"

[[deps.BitFlags]]
git-tree-sha1 = "43b1a4a8f797c1cddadf60499a8a077d4af2cd2d"
uuid = "d1d4a3ce-64b1-5f1a-9ba4-7e7e69966f35"
version = "0.1.7"

[[deps.Bzip2_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "19a35467a82e236ff51bc17a3a44b69ef35185a2"
uuid = "6e34b625-4abd-537c-b88f-471c36dfa7a0"
version = "1.0.8+0"

[[deps.CSV]]
deps = ["CodecZlib", "Dates", "FilePathsBase", "InlineStrings", "Mmap", "Parsers", "PooledArrays", "PrecompileTools", "SentinelArrays", "Tables", "Unicode", "WeakRefStrings", "WorkerUtilities"]
git-tree-sha1 = "44dbf560808d49041989b8a96cae4cffbeb7966a"
uuid = "336ed68f-0bac-5ca0-87d4-7b16caf5d00b"
version = "0.10.11"

[[deps.Cairo_jll]]
deps = ["Artifacts", "Bzip2_jll", "CompilerSupportLibraries_jll", "Fontconfig_jll", "FreeType2_jll", "Glib_jll", "JLLWrappers", "LZO_jll", "Libdl", "Pixman_jll", "Pkg", "Xorg_libXext_jll", "Xorg_libXrender_jll", "Zlib_jll", "libpng_jll"]
git-tree-sha1 = "4b859a208b2397a7a623a03449e4636bdb17bcf2"
uuid = "83423d85-b0ee-5818-9007-b63ccbeb887a"
version = "1.16.1+1"

[[deps.CodecZlib]]
deps = ["TranscodingStreams", "Zlib_jll"]
git-tree-sha1 = "02aa26a4cf76381be7f66e020a3eddeb27b0a092"
uuid = "944b1d66-785c-5afd-91f1-9de20f533193"
version = "0.7.2"

[[deps.ColorSchemes]]
deps = ["ColorTypes", "ColorVectorSpace", "Colors", "FixedPointNumbers", "PrecompileTools", "Random"]
git-tree-sha1 = "d9a8f86737b665e15a9641ecbac64deef9ce6724"
uuid = "35d6a980-a343-548e-a6ea-1d62b119f2f4"
version = "3.23.0"

[[deps.ColorTypes]]
deps = ["FixedPointNumbers", "Random"]
git-tree-sha1 = "eb7f0f8307f71fac7c606984ea5fb2817275d6e4"
uuid = "3da002f7-5984-5a60-b8a6-cbb66c0b333f"
version = "0.11.4"

[[deps.ColorVectorSpace]]
deps = ["ColorTypes", "FixedPointNumbers", "LinearAlgebra", "Requires", "Statistics", "TensorCore"]
git-tree-sha1 = "a1f44953f2382ebb937d60dafbe2deea4bd23249"
uuid = "c3611d14-8923-5661-9e6a-0046d554d3a4"
version = "0.10.0"
weakdeps = ["SpecialFunctions"]

    [deps.ColorVectorSpace.extensions]
    SpecialFunctionsExt = "SpecialFunctions"

[[deps.Colors]]
deps = ["ColorTypes", "FixedPointNumbers", "Reexport"]
git-tree-sha1 = "fc08e5930ee9a4e03f84bfb5211cb54e7769758a"
uuid = "5ae59095-9a9b-59fe-a467-6f913c188581"
version = "0.12.10"

[[deps.CommonEq]]
git-tree-sha1 = "d1beba82ceee6dc0fce8cb6b80bf600bbde66381"
uuid = "3709ef60-1bee-4518-9f2f-acd86f176c50"
version = "0.2.0"

[[deps.CommonSolve]]
git-tree-sha1 = "0eee5eb66b1cf62cd6ad1b460238e60e4b09400c"
uuid = "38540f10-b2f7-11e9-35d8-d573e4eb0ff2"
version = "0.2.4"

[[deps.Compat]]
deps = ["UUIDs"]
git-tree-sha1 = "e460f044ca8b99be31d35fe54fc33a5c33dd8ed7"
uuid = "34da2185-b29b-5c13-b0c7-acf172513d20"
version = "4.9.0"
weakdeps = ["Dates", "LinearAlgebra"]

    [deps.Compat.extensions]
    CompatLinearAlgebraExt = "LinearAlgebra"

[[deps.CompilerSupportLibraries_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "e66e0078-7015-5450-92f7-15fbd957f2ae"
version = "1.0.5+0"

[[deps.ConcurrentUtilities]]
deps = ["Serialization", "Sockets"]
git-tree-sha1 = "5372dbbf8f0bdb8c700db5367132925c0771ef7e"
uuid = "f0e56b4a-5159-44fe-b623-3e5288b988bb"
version = "2.2.1"

[[deps.Conda]]
deps = ["Downloads", "JSON", "VersionParsing"]
git-tree-sha1 = "8c86e48c0db1564a1d49548d3515ced5d604c408"
uuid = "8f4d0f93-b110-5947-807f-2305c1781a2d"
version = "1.9.1"

[[deps.Contour]]
git-tree-sha1 = "d05d9e7b7aedff4e5b51a029dced05cfb6125781"
uuid = "d38c429a-6771-53c6-b99e-75d170b6e991"
version = "0.6.2"

[[deps.Crayons]]
git-tree-sha1 = "249fe38abf76d48563e2f4556bebd215aa317e15"
uuid = "a8cc5b0e-0ffa-5ad4-8c14-923d3ee1735f"
version = "4.1.1"

[[deps.DataAPI]]
git-tree-sha1 = "8da84edb865b0b5b0100c0666a9bc9a0b71c553c"
uuid = "9a962f9c-6df0-11e9-0e5d-c546b8b5ee8a"
version = "1.15.0"

[[deps.DataFrames]]
deps = ["Compat", "DataAPI", "DataStructures", "Future", "InlineStrings", "InvertedIndices", "IteratorInterfaceExtensions", "LinearAlgebra", "Markdown", "Missings", "PooledArrays", "PrecompileTools", "PrettyTables", "Printf", "REPL", "Random", "Reexport", "SentinelArrays", "SortingAlgorithms", "Statistics", "TableTraits", "Tables", "Unicode"]
git-tree-sha1 = "04c738083f29f86e62c8afc341f0967d8717bdb8"
uuid = "a93c6f00-e57d-5684-b7b6-d8193f3e46c0"
version = "1.6.1"

[[deps.DataStructures]]
deps = ["Compat", "InteractiveUtils", "OrderedCollections"]
git-tree-sha1 = "3dbd312d370723b6bb43ba9d02fc36abade4518d"
uuid = "864edb3b-99cc-5e75-8d2d-829cb0a9cfe8"
version = "0.18.15"

[[deps.DataValueInterfaces]]
git-tree-sha1 = "bfc1187b79289637fa0ef6d4436ebdfe6905cbd6"
uuid = "e2d170a0-9d28-54be-80f0-106bbe20a464"
version = "1.0.0"

[[deps.Dates]]
deps = ["Printf"]
uuid = "ade2ca70-3891-5945-98fb-dc099432e06a"

[[deps.DelimitedFiles]]
deps = ["Mmap"]
git-tree-sha1 = "9e2f36d3c96a820c678f2f1f1782582fcf685bae"
uuid = "8bb1440f-4735-579b-a4ab-409b98df4dab"
version = "1.9.1"

[[deps.DocStringExtensions]]
deps = ["LibGit2"]
git-tree-sha1 = "2fb1e02f2b635d0845df5d7c167fec4dd739b00d"
uuid = "ffbed154-4ef7-542d-bbb7-c09d3a79fcae"
version = "0.9.3"

[[deps.Downloads]]
deps = ["ArgTools", "FileWatching", "LibCURL", "NetworkOptions"]
uuid = "f43a241f-c20a-4ad4-852c-f6b1247861c6"
version = "1.6.0"

[[deps.EpollShim_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "8e9441ee83492030ace98f9789a654a6d0b1f643"
uuid = "2702e6a9-849d-5ed8-8c21-79e8b8f9ee43"
version = "0.0.20230411+0"

[[deps.ExceptionUnwrapping]]
deps = ["Test"]
git-tree-sha1 = "e90caa41f5a86296e014e148ee061bd6c3edec96"
uuid = "460bff9d-24e4-43bc-9d9f-a8973cb893f4"
version = "0.1.9"

[[deps.Expat_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "4558ab818dcceaab612d1bb8c19cee87eda2b83c"
uuid = "2e619515-83b5-522b-bb60-26c02a35a201"
version = "2.5.0+0"

[[deps.FFMPEG]]
deps = ["FFMPEG_jll"]
git-tree-sha1 = "b57e3acbe22f8484b4b5ff66a7499717fe1a9cc8"
uuid = "c87230d0-a227-11e9-1b43-d7ebe4e7570a"
version = "0.4.1"

[[deps.FFMPEG_jll]]
deps = ["Artifacts", "Bzip2_jll", "FreeType2_jll", "FriBidi_jll", "JLLWrappers", "LAME_jll", "Libdl", "Ogg_jll", "OpenSSL_jll", "Opus_jll", "PCRE2_jll", "Pkg", "Zlib_jll", "libaom_jll", "libass_jll", "libfdk_aac_jll", "libvorbis_jll", "x264_jll", "x265_jll"]
git-tree-sha1 = "74faea50c1d007c85837327f6775bea60b5492dd"
uuid = "b22a6f82-2f65-5046-a5b2-351ab43fb4e5"
version = "4.4.2+2"

[[deps.FilePathsBase]]
deps = ["Compat", "Dates", "Mmap", "Printf", "Test", "UUIDs"]
git-tree-sha1 = "e27c4ebe80e8699540f2d6c805cc12203b614f12"
uuid = "48062228-2e41-5def-b9a4-89aafe57970f"
version = "0.9.20"

[[deps.FileWatching]]
uuid = "7b1f6079-737a-58dc-b8bc-7a2ca5c1b5ee"

[[deps.FixedPointNumbers]]
deps = ["Statistics"]
git-tree-sha1 = "335bfdceacc84c5cdf16aadc768aa5ddfc5383cc"
uuid = "53c48c17-4a7d-5ca2-90c5-79b7896eea93"
version = "0.8.4"

[[deps.Fontconfig_jll]]
deps = ["Artifacts", "Bzip2_jll", "Expat_jll", "FreeType2_jll", "JLLWrappers", "Libdl", "Libuuid_jll", "Pkg", "Zlib_jll"]
git-tree-sha1 = "21efd19106a55620a188615da6d3d06cd7f6ee03"
uuid = "a3f928ae-7b40-5064-980b-68af3947d34b"
version = "2.13.93+0"

[[deps.Formatting]]
deps = ["Printf"]
git-tree-sha1 = "8339d61043228fdd3eb658d86c926cb282ae72a8"
uuid = "59287772-0a20-5a39-b81b-1366585eb4c0"
version = "0.4.2"

[[deps.FreeType2_jll]]
deps = ["Artifacts", "Bzip2_jll", "JLLWrappers", "Libdl", "Zlib_jll"]
git-tree-sha1 = "d8db6a5a2fe1381c1ea4ef2cab7c69c2de7f9ea0"
uuid = "d7e528f0-a631-5988-bf34-fe36492bcfd7"
version = "2.13.1+0"

[[deps.FriBidi_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "aa31987c2ba8704e23c6c8ba8a4f769d5d7e4f91"
uuid = "559328eb-81f9-559d-9380-de523a88c83c"
version = "1.0.10+0"

[[deps.Future]]
deps = ["Random"]
uuid = "9fa8497b-333b-5362-9e8d-4d0656e87820"

[[deps.GLFW_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Libglvnd_jll", "Pkg", "Xorg_libXcursor_jll", "Xorg_libXi_jll", "Xorg_libXinerama_jll", "Xorg_libXrandr_jll"]
git-tree-sha1 = "d972031d28c8c8d9d7b41a536ad7bb0c2579caca"
uuid = "0656b61e-2033-5cc2-a64a-77c0f6c09b89"
version = "3.3.8+0"

[[deps.GR]]
deps = ["Artifacts", "Base64", "DelimitedFiles", "Downloads", "GR_jll", "HTTP", "JSON", "Libdl", "LinearAlgebra", "Pkg", "Preferences", "Printf", "Random", "Serialization", "Sockets", "TOML", "Tar", "Test", "UUIDs", "p7zip_jll"]
git-tree-sha1 = "d73afa4a2bb9de56077242d98cf763074ab9a970"
uuid = "28b8d3ca-fb5f-59d9-8090-bfdbd6d07a71"
version = "0.72.9"

[[deps.GR_jll]]
deps = ["Artifacts", "Bzip2_jll", "Cairo_jll", "FFMPEG_jll", "Fontconfig_jll", "FreeType2_jll", "GLFW_jll", "JLLWrappers", "JpegTurbo_jll", "Libdl", "Libtiff_jll", "Pixman_jll", "Qt6Base_jll", "Zlib_jll", "libpng_jll"]
git-tree-sha1 = "1596bab77f4f073a14c62424283e7ebff3072eca"
uuid = "d2c73de3-f751-5644-a686-071e5b155ba9"
version = "0.72.9+1"

[[deps.Gettext_jll]]
deps = ["Artifacts", "CompilerSupportLibraries_jll", "JLLWrappers", "Libdl", "Libiconv_jll", "Pkg", "XML2_jll"]
git-tree-sha1 = "9b02998aba7bf074d14de89f9d37ca24a1a0b046"
uuid = "78b55507-aeef-58d4-861c-77aaff3498b1"
version = "0.21.0+0"

[[deps.Glib_jll]]
deps = ["Artifacts", "Gettext_jll", "JLLWrappers", "Libdl", "Libffi_jll", "Libiconv_jll", "Libmount_jll", "PCRE2_jll", "Pkg", "Zlib_jll"]
git-tree-sha1 = "d3b3624125c1474292d0d8ed0f65554ac37ddb23"
uuid = "7746bdde-850d-59dc-9ae8-88ece973131d"
version = "2.74.0+2"

[[deps.Graphite2_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "344bf40dcab1073aca04aa0df4fb092f920e4011"
uuid = "3b182d85-2403-5c21-9c21-1e1f0cc25472"
version = "1.3.14+0"

[[deps.Grisu]]
git-tree-sha1 = "53bb909d1151e57e2484c3d1b53e19552b887fb2"
uuid = "42e2da0e-8278-4e71-bc24-59509adca0fe"
version = "1.0.2"

[[deps.HTTP]]
deps = ["Base64", "CodecZlib", "ConcurrentUtilities", "Dates", "ExceptionUnwrapping", "Logging", "LoggingExtras", "MbedTLS", "NetworkOptions", "OpenSSL", "Random", "SimpleBufferStream", "Sockets", "URIs", "UUIDs"]
git-tree-sha1 = "cb56ccdd481c0dd7f975ad2b3b62d9eda088f7e2"
uuid = "cd3eb016-35fb-5094-929b-558a96fad6f3"
version = "1.9.14"

[[deps.HarfBuzz_jll]]
deps = ["Artifacts", "Cairo_jll", "Fontconfig_jll", "FreeType2_jll", "Glib_jll", "Graphite2_jll", "JLLWrappers", "Libdl", "Libffi_jll", "Pkg"]
git-tree-sha1 = "129acf094d168394e80ee1dc4bc06ec835e510a3"
uuid = "2e76f6c2-a576-52d4-95c1-20adfe4de566"
version = "2.8.1+1"

[[deps.Hyperscript]]
deps = ["Test"]
git-tree-sha1 = "8d511d5b81240fc8e6802386302675bdf47737b9"
uuid = "47d2ed2b-36de-50cf-bf87-49c2cf4b8b91"
version = "0.0.4"

[[deps.HypertextLiteral]]
deps = ["Tricks"]
git-tree-sha1 = "c47c5fa4c5308f27ccaac35504858d8914e102f9"
uuid = "ac1192a8-f4b3-4bfe-ba22-af5b92cd3ab2"
version = "0.9.4"

[[deps.IOCapture]]
deps = ["Logging", "Random"]
git-tree-sha1 = "d75853a0bdbfb1ac815478bacd89cd27b550ace6"
uuid = "b5f81e59-6552-4d32-b1f0-c071b021bf89"
version = "0.2.3"

[[deps.InlineStrings]]
deps = ["Parsers"]
git-tree-sha1 = "9cc2baf75c6d09f9da536ddf58eb2f29dedaf461"
uuid = "842dd82b-1e85-43dc-bf29-5d0ee9dffc48"
version = "1.4.0"

[[deps.InteractiveUtils]]
deps = ["Markdown"]
uuid = "b77e0a4c-d291-57a0-90e8-8db25a27a240"

[[deps.InvertedIndices]]
git-tree-sha1 = "0dc7b50b8d436461be01300fd8cd45aa0274b038"
uuid = "41ab1584-1d38-5bbf-9106-f11c6c58b48f"
version = "1.3.0"

[[deps.IrrationalConstants]]
git-tree-sha1 = "630b497eafcc20001bba38a4651b327dcfc491d2"
uuid = "92d709cd-6900-40b7-9082-c6be49f344b6"
version = "0.2.2"

[[deps.IteratorInterfaceExtensions]]
git-tree-sha1 = "a3f24677c21f5bbe9d2a714f95dcd58337fb2856"
uuid = "82899510-4779-5014-852e-03e436cf321d"
version = "1.0.0"

[[deps.JLFzf]]
deps = ["Pipe", "REPL", "Random", "fzf_jll"]
git-tree-sha1 = "f377670cda23b6b7c1c0b3893e37451c5c1a2185"
uuid = "1019f520-868f-41f5-a6de-eb00f4b6a39c"
version = "0.1.5"

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

[[deps.JpegTurbo_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "6f2675ef130a300a112286de91973805fcc5ffbc"
uuid = "aacddb02-875f-59d6-b918-886e6ef4fbf8"
version = "2.1.91+0"

[[deps.LAME_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "f6250b16881adf048549549fba48b1161acdac8c"
uuid = "c1c5ebd0-6772-5130-a774-d5fcae4a789d"
version = "3.100.1+0"

[[deps.LERC_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "bf36f528eec6634efc60d7ec062008f171071434"
uuid = "88015f11-f218-50d7-93a8-a6af411a945d"
version = "3.0.0+1"

[[deps.LLVMOpenMP_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "f689897ccbe049adb19a065c495e75f372ecd42b"
uuid = "1d63c593-3942-5779-bab2-d838dc0a180e"
version = "15.0.4+0"

[[deps.LZO_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "e5b909bcf985c5e2605737d2ce278ed791b89be6"
uuid = "dd4b983a-f0e5-5f8d-a1b7-129d4a5fb1ac"
version = "2.10.1+0"

[[deps.LaTeXStrings]]
git-tree-sha1 = "f2355693d6778a178ade15952b7ac47a4ff97996"
uuid = "b964fa9f-0449-5b57-a5c2-d3ea65f4040f"
version = "1.3.0"

[[deps.Latexify]]
deps = ["Formatting", "InteractiveUtils", "LaTeXStrings", "MacroTools", "Markdown", "OrderedCollections", "Printf", "Requires"]
git-tree-sha1 = "f428ae552340899a935973270b8d98e5a31c49fe"
uuid = "23fbe1c1-3f47-55db-b15f-69d7ec21a316"
version = "0.16.1"

    [deps.Latexify.extensions]
    DataFramesExt = "DataFrames"
    SymEngineExt = "SymEngine"

    [deps.Latexify.weakdeps]
    DataFrames = "a93c6f00-e57d-5684-b7b6-d8193f3e46c0"
    SymEngine = "123dc426-2d89-5057-bbad-38513e3affd8"

[[deps.LibCURL]]
deps = ["LibCURL_jll", "MozillaCACerts_jll"]
uuid = "b27032c2-a3e7-50c8-80cd-2d36dbcbfd21"
version = "0.6.3"

[[deps.LibCURL_jll]]
deps = ["Artifacts", "LibSSH2_jll", "Libdl", "MbedTLS_jll", "Zlib_jll", "nghttp2_jll"]
uuid = "deac9b47-8bc7-5906-a0fe-35ac56dc84c0"
version = "7.84.0+0"

[[deps.LibGit2]]
deps = ["Base64", "NetworkOptions", "Printf", "SHA"]
uuid = "76f85450-5226-5b5a-8eaa-529ad045b433"

[[deps.LibSSH2_jll]]
deps = ["Artifacts", "Libdl", "MbedTLS_jll"]
uuid = "29816b5a-b9ab-546f-933c-edad1886dfa8"
version = "1.10.2+0"

[[deps.Libdl]]
uuid = "8f399da3-3557-5675-b5ff-fb832c97cbdb"

[[deps.Libffi_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "0b4a5d71f3e5200a7dff793393e09dfc2d874290"
uuid = "e9f186c6-92d2-5b65-8a66-fee21dc1b490"
version = "3.2.2+1"

[[deps.Libgcrypt_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Libgpg_error_jll", "Pkg"]
git-tree-sha1 = "64613c82a59c120435c067c2b809fc61cf5166ae"
uuid = "d4300ac3-e22c-5743-9152-c294e39db1e4"
version = "1.8.7+0"

[[deps.Libglvnd_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Xorg_libX11_jll", "Xorg_libXext_jll"]
git-tree-sha1 = "6f73d1dd803986947b2c750138528a999a6c7733"
uuid = "7e76a0d4-f3c7-5321-8279-8d96eeed0f29"
version = "1.6.0+0"

[[deps.Libgpg_error_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "c333716e46366857753e273ce6a69ee0945a6db9"
uuid = "7add5ba3-2f88-524e-9cd5-f83b8a55f7b8"
version = "1.42.0+0"

[[deps.Libiconv_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "c7cb1f5d892775ba13767a87c7ada0b980ea0a71"
uuid = "94ce4f54-9a6c-5748-9c1c-f9c7231a4531"
version = "1.16.1+2"

[[deps.Libmount_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "9c30530bf0effd46e15e0fdcf2b8636e78cbbd73"
uuid = "4b2f31a3-9ecc-558c-b454-b3730dcb73e9"
version = "2.35.0+0"

[[deps.Libtiff_jll]]
deps = ["Artifacts", "JLLWrappers", "JpegTurbo_jll", "LERC_jll", "Libdl", "XZ_jll", "Zlib_jll", "Zstd_jll"]
git-tree-sha1 = "2da088d113af58221c52828a80378e16be7d037a"
uuid = "89763e89-9b03-5906-acba-b20f662cd828"
version = "4.5.1+1"

[[deps.Libuuid_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "7f3efec06033682db852f8b3bc3c1d2b0a0ab066"
uuid = "38a345b3-de98-5d2b-a5d3-14cd9215e700"
version = "2.36.0+0"

[[deps.LinearAlgebra]]
deps = ["Libdl", "OpenBLAS_jll", "libblastrampoline_jll"]
uuid = "37e2e46d-f89d-539d-b4ee-838fcccc9c8e"

[[deps.LogExpFunctions]]
deps = ["DocStringExtensions", "IrrationalConstants", "LinearAlgebra"]
git-tree-sha1 = "7d6dd4e9212aebaeed356de34ccf262a3cd415aa"
uuid = "2ab3a3ac-af41-5b50-aa03-7779005ae688"
version = "0.3.26"

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
git-tree-sha1 = "a03c77519ab45eb9a34d3cfe2ca223d79c064323"
uuid = "e6f89c97-d47a-5376-807f-9c37f3926c36"
version = "1.0.1"

[[deps.MIMEs]]
git-tree-sha1 = "65f28ad4b594aebe22157d6fac869786a255b7eb"
uuid = "6c6e2e6c-3030-632d-7369-2d6c69616d65"
version = "0.1.4"

[[deps.MacroTools]]
deps = ["Markdown", "Random"]
git-tree-sha1 = "9ee1618cbf5240e6d4e0371d6f24065083f60c48"
uuid = "1914dd2f-81c6-5fcd-8719-6d5c9610ff09"
version = "0.5.11"

[[deps.Markdown]]
deps = ["Base64"]
uuid = "d6f4376e-aef5-505a-96c1-9c027394607a"

[[deps.MbedTLS]]
deps = ["Dates", "MbedTLS_jll", "MozillaCACerts_jll", "Random", "Sockets"]
git-tree-sha1 = "03a9b9718f5682ecb107ac9f7308991db4ce395b"
uuid = "739be429-bea8-5141-9913-cc70e7f3736d"
version = "1.1.7"

[[deps.MbedTLS_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "c8ffd9c3-330d-5841-b78e-0817d7145fa1"
version = "2.28.2+0"

[[deps.Measures]]
git-tree-sha1 = "c13304c81eec1ed3af7fc20e75fb6b26092a1102"
uuid = "442fdcdd-2543-5da2-b0f3-8c86c306513e"
version = "0.3.2"

[[deps.Missings]]
deps = ["DataAPI"]
git-tree-sha1 = "f66bdc5de519e8f8ae43bdc598782d35a25b1272"
uuid = "e1d29d7a-bbdc-5cf2-9ac0-f12de2c33e28"
version = "1.1.0"

[[deps.Mmap]]
uuid = "a63ad114-7e13-5084-954f-fe012c677804"

[[deps.MozillaCACerts_jll]]
uuid = "14a3606d-f60d-562e-9121-12d972cd8159"
version = "2022.10.11"

[[deps.NaNMath]]
deps = ["OpenLibm_jll"]
git-tree-sha1 = "0877504529a3e5c3343c6f8b4c0381e57e4387e4"
uuid = "77ba4419-2d1f-58cd-9bb1-8ffee604a2e3"
version = "1.0.2"

[[deps.NetworkOptions]]
uuid = "ca575930-c2e3-43a9-ace4-1e988b2c1908"
version = "1.2.0"

[[deps.Ogg_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "887579a3eb005446d514ab7aeac5d1d027658b8f"
uuid = "e7412a2a-1a6e-54c0-be00-318e2571c051"
version = "1.3.5+1"

[[deps.OpenBLAS_jll]]
deps = ["Artifacts", "CompilerSupportLibraries_jll", "Libdl"]
uuid = "4536629a-c528-5b80-bd46-f80d51c5b363"
version = "0.3.21+4"

[[deps.OpenLibm_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "05823500-19ac-5b8b-9628-191a04bc5112"
version = "0.8.1+0"

[[deps.OpenSSL]]
deps = ["BitFlags", "Dates", "MozillaCACerts_jll", "OpenSSL_jll", "Sockets"]
git-tree-sha1 = "51901a49222b09e3743c65b8847687ae5fc78eb2"
uuid = "4d8831e6-92b7-49fb-bdf8-b643e874388c"
version = "1.4.1"

[[deps.OpenSSL_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "bbb5c2115d63c2f1451cb70e5ef75e8fe4707019"
uuid = "458c3c95-2e84-50aa-8efc-19380b2a3a95"
version = "1.1.22+0"

[[deps.OpenSpecFun_jll]]
deps = ["Artifacts", "CompilerSupportLibraries_jll", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "13652491f6856acfd2db29360e1bbcd4565d04f1"
uuid = "efe28fd5-8261-553b-a9e1-b2916fc3738e"
version = "0.5.5+0"

[[deps.Opus_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "51a08fb14ec28da2ec7a927c4337e4332c2a4720"
uuid = "91d4177d-7536-5919-b921-800302f37372"
version = "1.3.2+0"

[[deps.OrderedCollections]]
git-tree-sha1 = "2e73fe17cac3c62ad1aebe70d44c963c3cfdc3e3"
uuid = "bac558e1-5e72-5ebc-8fee-abe8a469f55d"
version = "1.6.2"

[[deps.PCRE2_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "efcefdf7-47ab-520b-bdef-62a2eaa19f15"
version = "10.42.0+0"

[[deps.Parsers]]
deps = ["Dates", "PrecompileTools", "UUIDs"]
git-tree-sha1 = "716e24b21538abc91f6205fd1d8363f39b442851"
uuid = "69de0a69-1ddd-5017-9359-2bf0b02dc9f0"
version = "2.7.2"

[[deps.Pipe]]
git-tree-sha1 = "6842804e7867b115ca9de748a0cf6b364523c16d"
uuid = "b98c9c47-44ae-5843-9183-064241ee97a0"
version = "1.3.0"

[[deps.Pixman_jll]]
deps = ["Artifacts", "CompilerSupportLibraries_jll", "JLLWrappers", "LLVMOpenMP_jll", "Libdl"]
git-tree-sha1 = "64779bc4c9784fee475689a1752ef4d5747c5e87"
uuid = "30392449-352a-5448-841d-b1acce4e97dc"
version = "0.42.2+0"

[[deps.Pkg]]
deps = ["Artifacts", "Dates", "Downloads", "FileWatching", "LibGit2", "Libdl", "Logging", "Markdown", "Printf", "REPL", "Random", "SHA", "Serialization", "TOML", "Tar", "UUIDs", "p7zip_jll"]
uuid = "44cfe95a-1eb2-52ea-b672-e2afdf69b78f"
version = "1.9.2"

[[deps.PlotThemes]]
deps = ["PlotUtils", "Statistics"]
git-tree-sha1 = "1f03a2d339f42dca4a4da149c7e15e9b896ad899"
uuid = "ccf2f8ad-2431-5c83-bf29-c5338b663b6a"
version = "3.1.0"

[[deps.PlotUtils]]
deps = ["ColorSchemes", "Colors", "Dates", "PrecompileTools", "Printf", "Random", "Reexport", "Statistics"]
git-tree-sha1 = "f92e1315dadf8c46561fb9396e525f7200cdc227"
uuid = "995b91a9-d308-5afd-9ec6-746e21dbc043"
version = "1.3.5"

[[deps.Plots]]
deps = ["Base64", "Contour", "Dates", "Downloads", "FFMPEG", "FixedPointNumbers", "GR", "JLFzf", "JSON", "LaTeXStrings", "Latexify", "LinearAlgebra", "Measures", "NaNMath", "Pkg", "PlotThemes", "PlotUtils", "PrecompileTools", "Preferences", "Printf", "REPL", "Random", "RecipesBase", "RecipesPipeline", "Reexport", "RelocatableFolders", "Requires", "Scratch", "Showoff", "SparseArrays", "Statistics", "StatsBase", "UUIDs", "UnicodeFun", "UnitfulLatexify", "Unzip"]
git-tree-sha1 = "9f8675a55b37a70aa23177ec110f6e3f4dd68466"
uuid = "91a5bcdd-55d7-5caf-9e0b-520d859cae80"
version = "1.38.17"

    [deps.Plots.extensions]
    FileIOExt = "FileIO"
    GeometryBasicsExt = "GeometryBasics"
    IJuliaExt = "IJulia"
    ImageInTerminalExt = "ImageInTerminal"
    UnitfulExt = "Unitful"

    [deps.Plots.weakdeps]
    FileIO = "5789e2e9-d7fb-5bc7-8068-2c6fae9b9549"
    GeometryBasics = "5c1252a2-5f33-56bf-86c9-59e7332b4326"
    IJulia = "7073ff75-c697-5162-941a-fcdaad2a7d2a"
    ImageInTerminal = "d8c32880-2388-543b-8c61-d9f865259254"
    Unitful = "1986cc42-f94f-5a68-af5c-568840ba703d"

[[deps.PlutoUI]]
deps = ["AbstractPlutoDingetjes", "Base64", "ColorTypes", "Dates", "FixedPointNumbers", "Hyperscript", "HypertextLiteral", "IOCapture", "InteractiveUtils", "JSON", "Logging", "MIMEs", "Markdown", "Random", "Reexport", "URIs", "UUIDs"]
git-tree-sha1 = "e47cd150dbe0443c3a3651bc5b9cbd5576ab75b7"
uuid = "7f904dfe-b85e-4ff6-b463-dae2292396a8"
version = "0.7.52"

[[deps.PooledArrays]]
deps = ["DataAPI", "Future"]
git-tree-sha1 = "a6062fe4063cdafe78f4a0a81cfffb89721b30e7"
uuid = "2dfb63ee-cc39-5dd5-95bd-886bf059d720"
version = "1.4.2"

[[deps.PrecompileTools]]
deps = ["Preferences"]
git-tree-sha1 = "03b4c25b43cb84cee5c90aa9b5ea0a78fd848d2f"
uuid = "aea7be01-6a6a-4083-8856-8a6e6704d82a"
version = "1.2.0"

[[deps.Preferences]]
deps = ["TOML"]
git-tree-sha1 = "7eb1686b4f04b82f96ed7a4ea5890a4f0c7a09f1"
uuid = "21216c6a-2e73-6563-6e65-726566657250"
version = "1.4.0"

[[deps.PrettyTables]]
deps = ["Crayons", "LaTeXStrings", "Markdown", "Printf", "Reexport", "StringManipulation", "Tables"]
git-tree-sha1 = "ee094908d720185ddbdc58dbe0c1cbe35453ec7a"
uuid = "08abe8d2-0d0c-5749-adfa-8a2ac140af0d"
version = "2.2.7"

[[deps.Printf]]
deps = ["Unicode"]
uuid = "de0858da-6303-5e67-8744-51eddeeeb8d7"

[[deps.PyCall]]
deps = ["Conda", "Dates", "Libdl", "LinearAlgebra", "MacroTools", "Serialization", "VersionParsing"]
git-tree-sha1 = "43d304ac6f0354755f1d60730ece8c499980f7ba"
uuid = "438e738f-606a-5dbb-bf0a-cddfbfd45ab0"
version = "1.96.1"

[[deps.Qt6Base_jll]]
deps = ["Artifacts", "CompilerSupportLibraries_jll", "Fontconfig_jll", "Glib_jll", "JLLWrappers", "Libdl", "Libglvnd_jll", "OpenSSL_jll", "Pkg", "Xorg_libXext_jll", "Xorg_libXrender_jll", "Xorg_libxcb_jll", "Xorg_xcb_util_image_jll", "Xorg_xcb_util_keysyms_jll", "Xorg_xcb_util_renderutil_jll", "Xorg_xcb_util_wm_jll", "Zlib_jll", "xkbcommon_jll"]
git-tree-sha1 = "364898e8f13f7eaaceec55fd3d08680498c0aa6e"
uuid = "c0090381-4147-56d7-9ebc-da0b1113ec56"
version = "6.4.2+3"

[[deps.REPL]]
deps = ["InteractiveUtils", "Markdown", "Sockets", "Unicode"]
uuid = "3fa0cd96-eef1-5676-8a61-b3b8758bbffb"

[[deps.Random]]
deps = ["SHA", "Serialization"]
uuid = "9a3f8284-a2c9-5f02-9a11-845980a1fd5c"

[[deps.RecipesBase]]
deps = ["PrecompileTools"]
git-tree-sha1 = "5c3d09cc4f31f5fc6af001c250bf1278733100ff"
uuid = "3cdcf5f2-1ef4-517c-9805-6587b60abb01"
version = "1.3.4"

[[deps.RecipesPipeline]]
deps = ["Dates", "NaNMath", "PlotUtils", "PrecompileTools", "RecipesBase"]
git-tree-sha1 = "45cf9fd0ca5839d06ef333c8201714e888486342"
uuid = "01d81517-befc-4cb6-b9ec-a95719d0359c"
version = "0.6.12"

[[deps.Reexport]]
git-tree-sha1 = "45e428421666073eab6f2da5c9d310d99bb12f9b"
uuid = "189a3867-3050-52da-a836-e630ba90ab69"
version = "1.2.2"

[[deps.RelocatableFolders]]
deps = ["SHA", "Scratch"]
git-tree-sha1 = "90bc7a7c96410424509e4263e277e43250c05691"
uuid = "05181044-ff0b-4ac5-8273-598c1e38db00"
version = "1.0.0"

[[deps.Requires]]
deps = ["UUIDs"]
git-tree-sha1 = "838a3a4188e2ded87a4f9f184b4b0d78a1e91cb7"
uuid = "ae029012-a4dd-5104-9daa-d747884805df"
version = "1.3.0"

[[deps.SHA]]
uuid = "ea8e919c-243c-51af-8825-aaa63cd721ce"
version = "0.7.0"

[[deps.Scratch]]
deps = ["Dates"]
git-tree-sha1 = "30449ee12237627992a99d5e30ae63e4d78cd24a"
uuid = "6c6a2e73-6563-6170-7368-637461726353"
version = "1.2.0"

[[deps.SentinelArrays]]
deps = ["Dates", "Random"]
git-tree-sha1 = "04bdff0b09c65ff3e06a05e3eb7b120223da3d39"
uuid = "91c51154-3ec4-41a3-a24f-3f23e20d615c"
version = "1.4.0"

[[deps.Serialization]]
uuid = "9e88b42a-f829-5b0c-bbe9-9e923198166b"

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
git-tree-sha1 = "c60ec5c62180f27efea3ba2908480f8055e17cee"
uuid = "a2af1166-a08f-5f64-846c-94a0d3cef48c"
version = "1.1.1"

[[deps.SparseArrays]]
deps = ["Libdl", "LinearAlgebra", "Random", "Serialization", "SuiteSparse_jll"]
uuid = "2f01184e-e22b-5df5-ae63-d93ebab69eaf"

[[deps.SpecialFunctions]]
deps = ["IrrationalConstants", "LogExpFunctions", "OpenLibm_jll", "OpenSpecFun_jll"]
git-tree-sha1 = "e2cfc4012a19088254b3950b85c3c1d8882d864d"
uuid = "276daf66-3868-5448-9aa4-cd146d93841b"
version = "2.3.1"

    [deps.SpecialFunctions.extensions]
    SpecialFunctionsChainRulesCoreExt = "ChainRulesCore"

    [deps.SpecialFunctions.weakdeps]
    ChainRulesCore = "d360d2e6-b24c-11e9-a2a3-2a2ae2dbcce4"

[[deps.Statistics]]
deps = ["LinearAlgebra", "SparseArrays"]
uuid = "10745b16-79ce-11e8-11f9-7d13ad32a3b2"
version = "1.9.0"

[[deps.StatsAPI]]
deps = ["LinearAlgebra"]
git-tree-sha1 = "45a7769a04a3cf80da1c1c7c60caf932e6f4c9f7"
uuid = "82ae8749-77ed-4fe6-ae5f-f523153014b0"
version = "1.6.0"

[[deps.StatsBase]]
deps = ["DataAPI", "DataStructures", "LinearAlgebra", "LogExpFunctions", "Missings", "Printf", "Random", "SortingAlgorithms", "SparseArrays", "Statistics", "StatsAPI"]
git-tree-sha1 = "75ebe04c5bed70b91614d684259b661c9e6274a4"
uuid = "2913bbd2-ae8a-5f71-8c99-4fb6c76f3a91"
version = "0.34.0"

[[deps.StringManipulation]]
git-tree-sha1 = "46da2434b41f41ac3594ee9816ce5541c6096123"
uuid = "892a3eda-7b42-436c-8928-eab12a02cf0e"
version = "0.3.0"

[[deps.SuiteSparse_jll]]
deps = ["Artifacts", "Libdl", "Pkg", "libblastrampoline_jll"]
uuid = "bea87d4a-7f5b-5778-9afe-8cc45184846c"
version = "5.10.1+6"

[[deps.SymPy]]
deps = ["CommonEq", "CommonSolve", "Latexify", "LinearAlgebra", "Markdown", "PyCall", "RecipesBase", "SpecialFunctions"]
git-tree-sha1 = "ed1605d9415cccb50e614b8fe0035753877b5303"
uuid = "24249f21-da20-56a4-8eb1-6a02cf4ae2e6"
version = "1.1.12"

    [deps.SymPy.extensions]
    SymPySymbolicUtilsExt = "SymbolicUtils"

    [deps.SymPy.weakdeps]
    SymbolicUtils = "d1185830-fcd6-423d-90d6-eec64667417b"

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
deps = ["DataAPI", "DataValueInterfaces", "IteratorInterfaceExtensions", "LinearAlgebra", "OrderedCollections", "TableTraits", "Test"]
git-tree-sha1 = "1544b926975372da01227b382066ab70e574a3ec"
uuid = "bd369af6-aec1-5ad0-b16a-f7cc5008161c"
version = "1.10.1"

[[deps.Tar]]
deps = ["ArgTools", "SHA"]
uuid = "a4e569a6-e804-4fa4-b0f3-eef7a1d5b13e"
version = "1.10.0"

[[deps.TensorCore]]
deps = ["LinearAlgebra"]
git-tree-sha1 = "1feb45f88d133a655e001435632f019a9a1bcdb6"
uuid = "62fd8b95-f654-4bbd-a8a5-9c27f68ccd50"
version = "0.1.1"

[[deps.Test]]
deps = ["InteractiveUtils", "Logging", "Random", "Serialization"]
uuid = "8dfed614-e22c-5e08-85e1-65c5234f0b40"

[[deps.TranscodingStreams]]
deps = ["Random", "Test"]
git-tree-sha1 = "9a6ae7ed916312b41236fcef7e0af564ef934769"
uuid = "3bb67fe8-82b1-5028-8e26-92a6c54297fa"
version = "0.9.13"

[[deps.Tricks]]
git-tree-sha1 = "aadb748be58b492045b4f56166b5188aa63ce549"
uuid = "410a4b4d-49e4-4fbc-ab6d-cb71b17b3775"
version = "0.1.7"

[[deps.URIs]]
git-tree-sha1 = "b7a5e99f24892b6824a954199a45e9ffcc1c70f0"
uuid = "5c2747f8-b7ea-4ff2-ba2e-563bfd36b1d4"
version = "1.5.0"

[[deps.UUIDs]]
deps = ["Random", "SHA"]
uuid = "cf7118a7-6976-5b1a-9a39-7adc72f591a4"

[[deps.Unicode]]
uuid = "4ec0a83e-493e-50e2-b9ac-8f72acf5a8f5"

[[deps.UnicodeFun]]
deps = ["REPL"]
git-tree-sha1 = "53915e50200959667e78a92a418594b428dffddf"
uuid = "1cfade01-22cf-5700-b092-accc4b62d6e1"
version = "0.4.1"

[[deps.Unitful]]
deps = ["Dates", "LinearAlgebra", "Random"]
git-tree-sha1 = "a72d22c7e13fe2de562feda8645aa134712a87ee"
uuid = "1986cc42-f94f-5a68-af5c-568840ba703d"
version = "1.17.0"

    [deps.Unitful.extensions]
    ConstructionBaseUnitfulExt = "ConstructionBase"
    InverseFunctionsUnitfulExt = "InverseFunctions"

    [deps.Unitful.weakdeps]
    ConstructionBase = "187b0558-2788-49d3-abe0-74a17ed4e7c9"
    InverseFunctions = "3587e190-3f89-42d0-90ee-14403ec27112"

[[deps.UnitfulLatexify]]
deps = ["LaTeXStrings", "Latexify", "Unitful"]
git-tree-sha1 = "e2d817cc500e960fdbafcf988ac8436ba3208bfd"
uuid = "45397f5d-5981-4c77-b2b3-fc36d6e9b728"
version = "1.6.3"

[[deps.Unzip]]
git-tree-sha1 = "ca0969166a028236229f63514992fc073799bb78"
uuid = "41fe7b60-77ed-43a1-b4f0-825fd5a5650d"
version = "0.2.0"

[[deps.VersionParsing]]
git-tree-sha1 = "58d6e80b4ee071f5efd07fda82cb9fbe17200868"
uuid = "81def892-9a0e-5fdd-b105-ffc91e053289"
version = "1.3.0"

[[deps.Wayland_jll]]
deps = ["Artifacts", "EpollShim_jll", "Expat_jll", "JLLWrappers", "Libdl", "Libffi_jll", "Pkg", "XML2_jll"]
git-tree-sha1 = "ed8d92d9774b077c53e1da50fd81a36af3744c1c"
uuid = "a2964d1f-97da-50d4-b82a-358c7fce9d89"
version = "1.21.0+0"

[[deps.Wayland_protocols_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "4528479aa01ee1b3b4cd0e6faef0e04cf16466da"
uuid = "2381bf8a-dfd0-557d-9999-79630e7b1b91"
version = "1.25.0+0"

[[deps.WeakRefStrings]]
deps = ["DataAPI", "InlineStrings", "Parsers"]
git-tree-sha1 = "b1be2855ed9ed8eac54e5caff2afcdb442d52c23"
uuid = "ea10d353-3f73-51f8-a26c-33c1cb351aa5"
version = "1.4.2"

[[deps.WorkerUtilities]]
git-tree-sha1 = "cd1659ba0d57b71a464a29e64dbc67cfe83d54e7"
uuid = "76eceee3-57b5-4d4a-8e66-0e911cebbf60"
version = "1.6.1"

[[deps.XML2_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Libiconv_jll", "Pkg", "Zlib_jll"]
git-tree-sha1 = "93c41695bc1c08c46c5899f4fe06d6ead504bb73"
uuid = "02c8fc9c-b97f-50b9-bbe4-9be30ff0a78a"
version = "2.10.3+0"

[[deps.XSLT_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Libgcrypt_jll", "Libgpg_error_jll", "Libiconv_jll", "Pkg", "XML2_jll", "Zlib_jll"]
git-tree-sha1 = "91844873c4085240b95e795f692c4cec4d805f8a"
uuid = "aed1982a-8fda-507f-9586-7b0439959a61"
version = "1.1.34+0"

[[deps.XZ_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "cf2c7de82431ca6f39250d2fc4aacd0daa1675c0"
uuid = "ffd25f8a-64ca-5728-b0f7-c24cf3aae800"
version = "5.4.4+0"

[[deps.Xorg_libX11_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Xorg_libxcb_jll", "Xorg_xtrans_jll"]
git-tree-sha1 = "afead5aba5aa507ad5a3bf01f58f82c8d1403495"
uuid = "4f6342f7-b3d2-589e-9d20-edeb45f2b2bc"
version = "1.8.6+0"

[[deps.Xorg_libXau_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "6035850dcc70518ca32f012e46015b9beeda49d8"
uuid = "0c0b7dd1-d40b-584c-a123-a41640f87eec"
version = "1.0.11+0"

[[deps.Xorg_libXcursor_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Xorg_libXfixes_jll", "Xorg_libXrender_jll"]
git-tree-sha1 = "12e0eb3bc634fa2080c1c37fccf56f7c22989afd"
uuid = "935fb764-8cf2-53bf-bb30-45bb1f8bf724"
version = "1.2.0+4"

[[deps.Xorg_libXdmcp_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "34d526d318358a859d7de23da945578e8e8727b7"
uuid = "a3789734-cfe1-5b06-b2d0-1dd0d9d62d05"
version = "1.1.4+0"

[[deps.Xorg_libXext_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Xorg_libX11_jll"]
git-tree-sha1 = "b7c0aa8c376b31e4852b360222848637f481f8c3"
uuid = "1082639a-0dae-5f34-9b06-72781eeb8cb3"
version = "1.3.4+4"

[[deps.Xorg_libXfixes_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Xorg_libX11_jll"]
git-tree-sha1 = "0e0dc7431e7a0587559f9294aeec269471c991a4"
uuid = "d091e8ba-531a-589c-9de9-94069b037ed8"
version = "5.0.3+4"

[[deps.Xorg_libXi_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Xorg_libXext_jll", "Xorg_libXfixes_jll"]
git-tree-sha1 = "89b52bc2160aadc84d707093930ef0bffa641246"
uuid = "a51aa0fd-4e3c-5386-b890-e753decda492"
version = "1.7.10+4"

[[deps.Xorg_libXinerama_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Xorg_libXext_jll"]
git-tree-sha1 = "26be8b1c342929259317d8b9f7b53bf2bb73b123"
uuid = "d1454406-59df-5ea1-beac-c340f2130bc3"
version = "1.1.4+4"

[[deps.Xorg_libXrandr_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Xorg_libXext_jll", "Xorg_libXrender_jll"]
git-tree-sha1 = "34cea83cb726fb58f325887bf0612c6b3fb17631"
uuid = "ec84b674-ba8e-5d96-8ba1-2a689ba10484"
version = "1.5.2+4"

[[deps.Xorg_libXrender_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Xorg_libX11_jll"]
git-tree-sha1 = "19560f30fd49f4d4efbe7002a1037f8c43d43b96"
uuid = "ea2f1a96-1ddc-540d-b46f-429655e07cfa"
version = "0.9.10+4"

[[deps.Xorg_libpthread_stubs_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "8fdda4c692503d44d04a0603d9ac0982054635f9"
uuid = "14d82f49-176c-5ed1-bb49-ad3f5cbd8c74"
version = "0.1.1+0"

[[deps.Xorg_libxcb_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "XSLT_jll", "Xorg_libXau_jll", "Xorg_libXdmcp_jll", "Xorg_libpthread_stubs_jll"]
git-tree-sha1 = "b4bfde5d5b652e22b9c790ad00af08b6d042b97d"
uuid = "c7cfdc94-dc32-55de-ac96-5a1b8d977c5b"
version = "1.15.0+0"

[[deps.Xorg_libxkbfile_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Xorg_libX11_jll"]
git-tree-sha1 = "730eeca102434283c50ccf7d1ecdadf521a765a4"
uuid = "cc61e674-0454-545c-8b26-ed2c68acab7a"
version = "1.1.2+0"

[[deps.Xorg_xcb_util_image_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Xorg_xcb_util_jll"]
git-tree-sha1 = "0fab0a40349ba1cba2c1da699243396ff8e94b97"
uuid = "12413925-8142-5f55-bb0e-6d7ca50bb09b"
version = "0.4.0+1"

[[deps.Xorg_xcb_util_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Xorg_libxcb_jll"]
git-tree-sha1 = "e7fd7b2881fa2eaa72717420894d3938177862d1"
uuid = "2def613f-5ad1-5310-b15b-b15d46f528f5"
version = "0.4.0+1"

[[deps.Xorg_xcb_util_keysyms_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Xorg_xcb_util_jll"]
git-tree-sha1 = "d1151e2c45a544f32441a567d1690e701ec89b00"
uuid = "975044d2-76e6-5fbe-bf08-97ce7c6574c7"
version = "0.4.0+1"

[[deps.Xorg_xcb_util_renderutil_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Xorg_xcb_util_jll"]
git-tree-sha1 = "dfd7a8f38d4613b6a575253b3174dd991ca6183e"
uuid = "0d47668e-0667-5a69-a72c-f761630bfb7e"
version = "0.3.9+1"

[[deps.Xorg_xcb_util_wm_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Xorg_xcb_util_jll"]
git-tree-sha1 = "e78d10aab01a4a154142c5006ed44fd9e8e31b67"
uuid = "c22f9ab0-d5fe-5066-847c-f4bb1cd4e361"
version = "0.4.1+1"

[[deps.Xorg_xkbcomp_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Xorg_libxkbfile_jll"]
git-tree-sha1 = "330f955bc41bb8f5270a369c473fc4a5a4e4d3cb"
uuid = "35661453-b289-5fab-8a00-3d9160c6a3a4"
version = "1.4.6+0"

[[deps.Xorg_xkeyboard_config_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Xorg_xkbcomp_jll"]
git-tree-sha1 = "691634e5453ad362044e2ad653e79f3ee3bb98c3"
uuid = "33bec58e-1273-512f-9401-5d533626f822"
version = "2.39.0+0"

[[deps.Xorg_xtrans_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "e92a1a012a10506618f10b7047e478403a046c77"
uuid = "c5fb5394-a638-5e4d-96e5-b29de1b5cf10"
version = "1.5.0+0"

[[deps.Zlib_jll]]
deps = ["Libdl"]
uuid = "83775a58-1f1d-513f-b197-d71354ab007a"
version = "1.2.13+0"

[[deps.Zstd_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "49ce682769cd5de6c72dcf1b94ed7790cd08974c"
uuid = "3161d3a3-bdf6-5164-811a-617609db77b4"
version = "1.5.5+0"

[[deps.fzf_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "868e669ccb12ba16eaf50cb2957ee2ff61261c56"
uuid = "214eeab7-80f7-51ab-84ad-2988db7cef09"
version = "0.29.0+0"

[[deps.libaom_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "3a2ea60308f0996d26f1e5354e10c24e9ef905d4"
uuid = "a4ae2306-e953-59d6-aa16-d00cac43593b"
version = "3.4.0+0"

[[deps.libass_jll]]
deps = ["Artifacts", "Bzip2_jll", "FreeType2_jll", "FriBidi_jll", "HarfBuzz_jll", "JLLWrappers", "Libdl", "Pkg", "Zlib_jll"]
git-tree-sha1 = "5982a94fcba20f02f42ace44b9894ee2b140fe47"
uuid = "0ac62f75-1d6f-5e53-bd7c-93b484bb37c0"
version = "0.15.1+0"

[[deps.libblastrampoline_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "8e850b90-86db-534c-a0d3-1478176c7d93"
version = "5.8.0+0"

[[deps.libfdk_aac_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "daacc84a041563f965be61859a36e17c4e4fcd55"
uuid = "f638f0a6-7fb0-5443-88ba-1cc74229b280"
version = "2.0.2+0"

[[deps.libpng_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Zlib_jll"]
git-tree-sha1 = "94d180a6d2b5e55e447e2d27a29ed04fe79eb30c"
uuid = "b53b4c65-9356-5827-b1ea-8c7a1a84506f"
version = "1.6.38+0"

[[deps.libvorbis_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Ogg_jll", "Pkg"]
git-tree-sha1 = "b910cb81ef3fe6e78bf6acee440bda86fd6ae00c"
uuid = "f27f6e37-5d2b-51aa-960f-b287f2bc3b7a"
version = "1.3.7+1"

[[deps.nghttp2_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "8e850ede-7688-5339-a07c-302acd2aaf8d"
version = "1.48.0+0"

[[deps.p7zip_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "3f19e933-33d8-53b3-aaab-bd5110c3b7a0"
version = "17.4.0+0"

[[deps.x264_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "4fea590b89e6ec504593146bf8b988b2c00922b2"
uuid = "1270edf5-f2f9-52d2-97e9-ab00b5d0237a"
version = "2021.5.5+0"

[[deps.x265_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "ee567a171cce03570d77ad3a43e90218e38937a9"
uuid = "dfaa095f-4041-5dcd-9319-2fabd8486b76"
version = "3.5.0+0"

[[deps.xkbcommon_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Wayland_jll", "Wayland_protocols_jll", "Xorg_libxcb_jll", "Xorg_xkeyboard_config_jll"]
git-tree-sha1 = "9ebfc140cc56e8c2156a15ceac2f0302e327ac0a"
uuid = "d8fb68d0-12a3-5cfd-a85a-d49703b185fd"
version = "1.4.1+0"
"""

# ╔═╡ Cell order:
# ╟─62287756-e122-406b-bfbf-8ac45779f91b
# ╟─bf16cb46-2f7e-4f51-aca6-2a1f7fcae79b
# ╟─6708a714-5946-4191-9758-b20fdd56e62b
# ╟─bb2ccd82-562d-48bc-99dc-6301126b3a41
# ╟─d1dc866c-33c0-41df-92aa-c426a78eee9d
# ╟─32f6a4a8-50e2-44e4-9323-5eb137431d82
# ╟─7c3c6ddb-bb9f-456b-938f-451ac5efd5dc
# ╟─8b8ce0e8-1333-43af-afb4-a306546a128a
# ╟─b4d29f23-3910-4667-9c3b-9a6ae3a527bb
# ╠═bb2dce04-b4b5-4166-8ea2-6a5135acdad9
# ╠═ad47fc86-e8bf-4cfc-9902-1c548306f32a
# ╠═1ca8c66d-9041-4a8f-899f-35c35b2f1c82
# ╠═d85306eb-a3a8-4075-b09b-8efab8f5efe2
# ╟─182ff5ae-d1d5-4232-a988-d58f5556d535
# ╠═c1cc1638-9a26-4374-ad61-8b2a56c29b7d
# ╠═1d88d4cd-c8fb-4f04-9746-89e52ffe712d
# ╟─99ee2478-9029-422c-91ca-410492a103c7
# ╠═ed5f2181-eb9b-4238-9e91-438d7ccc3f9d
# ╠═b18ecd31-c760-41b9-9037-613da500101f
# ╟─620341d3-bfbf-437a-a5cc-d146fafa9067
# ╟─2a186fc5-e755-417f-883b-c5f58b834c97
# ╟─9b6ddee1-54cb-4b87-b10c-0d2b13a7ebdb
# ╟─5a2d4a7b-a0eb-4e8d-b7e9-198976745cd2
# ╟─dd6a5c2a-0ed5-4552-b094-368d3bd57577
# ╠═c4b7dc55-bcd5-4c12-9fb4-af83b40ecd53
# ╟─340a16cc-9b66-4aa8-8849-d75de48195cb
# ╟─6497be76-a639-430f-963b-0486e14151b2
# ╟─4cfff7c9-de08-43d2-8551-c166d22df965
# ╟─341081cc-a36b-4cdb-99e7-9ac34730b254
# ╟─d415e3d4-3034-4bab-8ee0-e2645e6daa8f
# ╠═90e078ad-088a-4415-bcc5-44fc01dc8406
# ╟─ec5da1c6-3952-4576-b09d-b84af7b14bb2
# ╠═97567bbc-f652-4d54-81ae-028bb4b27b6b
# ╟─3ab8590a-bb4e-4e6b-9b53-3603c23aaeeb
# ╟─59733493-4be0-4650-a83a-e4279b0174ae
# ╠═5cd36360-bd59-44c9-850e-79e5de8a21e8
# ╟─3fa1bdd3-3fd0-43d4-b450-467fe636e9dc
# ╟─408f9681-325d-4d6f-89be-2ec8b518eac9
# ╟─507d1240-a0a2-4979-8856-2790c3c34eca
# ╟─17bcf2c6-5b89-4703-be18-c7c6c0433508
# ╟─151dc9af-5099-42be-86a9-f0ab59646b67
# ╟─f52780fe-14d6-476c-9cfc-a656c09a7a01
# ╟─f8918afb-d4cd-4d6e-b030-955a9434c2ff
# ╟─d26edc69-bd85-451d-99cc-5c0eee4aed40
# ╟─59375052-fb7a-4728-9118-f5096850da9c
# ╠═01451338-1a69-47fe-9727-d5c6aaa16758
# ╠═7f9348f6-508e-4ac8-a4bc-e2a75d127097
# ╠═7995877d-99e8-434b-9d54-1d50eec61f79
# ╠═1f9bad57-f122-4a5c-bdc7-90543e50a5a8
# ╟─e72ae865-e53a-415c-a4d4-f9ac0db96646
# ╟─1ced6465-0899-4393-b93d-c3aa688a67f3
# ╠═e6eed0fa-ff80-4898-94d4-d44653b8b898
# ╟─6ad8f345-fd60-40fa-a61c-dff02137171a
# ╠═6900e0ed-615e-4ede-87ca-0ba44382438b
# ╟─83fc9f05-12ac-4c93-8d92-3590e892dd1e
# ╟─d02b8d90-24e8-420f-934c-f3bc4d62554f
# ╟─3ad80698-f495-4740-9ece-dbf908bd9734
# ╟─0d399316-d9b9-4980-bff0-d60d8c3fdb94
# ╟─aabdeb85-2ca7-43ae-b413-92cc73c5fff1
# ╟─46a31a35-e5f9-4e71-8d44-7093c6feb6a1
# ╟─42bd1e5f-25ea-4375-8a9c-3c5e8ded2867
# ╠═f7dbe9d9-b709-47fe-9a7b-9a6cedb8d3c9
# ╠═df2601bc-e560-4801-93bf-226700e1bde8
# ╟─e386f4ba-3cce-44e1-b289-082afb765e37
# ╟─da679600-4485-47c1-9999-6d4392a0e6f2
# ╟─45835413-d81b-4816-8f9b-ef7060efa272
# ╟─018e7cba-d12c-4b9d-999a-98f4908d308f
# ╟─b3ed3084-b382-47b5-94d1-939afde61036
# ╟─4c2bbd0c-fa7a-4892-9a00-3ed740b5450a
# ╟─6f24d102-3b91-440e-aeb7-2a8e1abf7558
# ╟─0a784c5f-bc9b-4a9f-9ae8-9c7aa82f7192
# ╟─771b2e05-d6bc-4a27-a4c1-930332535d06
# ╟─00000000-0000-0000-0000-000000000001
# ╟─00000000-0000-0000-0000-000000000002
