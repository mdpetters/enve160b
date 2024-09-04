### A Pluto.jl notebook ###
# v0.19.29

#> [frontmatter]
#> title = "Review of Vector Calculus"

using Markdown
using InteractiveUtils

# ╔═╡ 9cb51ec8-40ff-11ee-0ddd-3335291342a5
begin
	
	using LinearAlgebra
	using PlutoUI, LaTeXStrings
	using HypertextLiteral
	using Plots,  PlotThemes
	using FileIO
	using Images 
	using SpecialFunctions
	using FastGaussQuadrature
	using ForwardDiff
	using Roots
	using FiniteDifferences
	using Plots.PlotMeasures: px
	using CalculusWithJulia
	using Tensors

	md"""
	$(TableOfContents(depth=4))
	# Review of Numeric Differentiation, Vectors, Matrices, Vector Calculus
	This notebook provides an illustrated review over key concepts in vector calculus with the objective to apply these operators in a computational context.
	# Learning Objectives
	"""
end

# ╔═╡ 8cba9988-b750-448e-9b94-df412c4af5a3

Markdown.MD(
	Markdown.Admonition("tip", "Learning Objectives", [md"
- Perform finite difference and automatic differentiation methods on functions.
- Provide a geometeric explanation of standard vector operations (dot product, cross product).
- Fluently work with vectors and matrices in a computational environment
"]))


# ╔═╡ 2466d645-3367-423a-815d-d86d5c2228b3
md"""
# Taylor Series Approximation

A function can be approximated near a point ``x_0`` using the Taylor series expansion

```math
f(x_0 + h) = \sum_{n=0}^\infty \frac{f^{(n)}(x_0)}{n!}h^n = f(x_0) + f'(x_0)h + \frac{f''(x_0)}{2!}h^2 + \frac{f'''(x_0)}{3!}h^3 + \dots
```

The example below is for ``f(x) = sin(x)`` and ``x_0 = \frac{pi}{4}``. Not how the first order term represents the tangent line at ``x_0``.
"""

# ╔═╡ be9dc539-b848-410b-bb37-4aa91155e306
let 
	x0 = pi/4.0
	x = 0:0.1:2pi
	p = plot(x, sin.(x), color = :black, label = "f(x)", ylim = [-1,2], 
		xlabel = "x", ylabel = "y")
	f1(a,h) = sin(a) + cos(a)*h
	h = -pi:0.1:pi
	x2 = x0 .+ h
	plot!(x2, f1.(x0,h), label = "n = 0..1", color = :darkred) 
	f2(a,h) = sin(a) + cos(a)*h - 1/2*sin(a)*h^2
	plot!(x2, f2.(x0,h), label = "n = 0..2", color = :steelblue3)
	f3(a,h) = sin(a) + cos(a)*h - 1/2*sin(a)*h^2 - 1/6*cos(a)*h^3.0
	plot!(x2, f3.(x0,h), label = "n = 0..3", color = :darkgoldenrod)
	plot!(size = (700,300), bottom_margin = 20px)
end

# ╔═╡ 52f31447-9fb3-4c43-91ac-9cb27f209e04
md"""
# Numeric Differentiation

## Forward and Central Difference

By approximating the limit of the secant line with a value for a small, but positive, $h$, we get an approximation to the derivative. That is


```math
f'(x) \approx \frac{f(x+h) - f(x)}{h}.
```

This is the forward-difference approximation. The central difference approximation looks both ways:


```math
f'(x) \approx \frac{f(x+h) - f(x-h)}{2h}.
```


Though in general they are different, they are both approximations. The central difference is usually more accurate for the same size $h$. However, both are susceptible to round-off errors. The numerator is a subtraction of like-size numbers - a perfect opportunity to lose precision.

As such there is a balancing act:

  * if $h$ is too small the round-off errors are problematic,
  * if $h$ is too big the approximation to the limit is not good.

For the forward difference $h$ values around $10^{-8}$ are typically good, for the central difference, values around $10^{-6}$ are typically good.

##### Higher Order Derivatives

```math
f''(x) \approx \frac{{\frac{f(x+h) - f(x)}{h}} - \frac{f(x) - f(x-h)}{h}}{h} = \frac{f(x+h) - 2f(x) + f(x-h)}{h^2}.
```

Note that thefinite difference approximation can be straightforwardly derived from the Taylor series expansion.

Example: 

```math
f(x) = \exp(-x^2/2)
```

```math
f'(x) = -x \exp(-x^2/2)
```

```math
f''(x) = (x^2-1) \exp(-x^2/2)
```

"""

# ╔═╡ 0348024a-027a-4ac2-9530-44803c6a6289
# Forward difference approximation
let
	fp = -1.0 * exp(-1.0^2.0/2.0) # True derivative
	f(x) = exp(-x^2/2)
	c = 1
	h = 1e-8
	fapprox = (f(c+h) - f(c)) / h
	(fapprox, (fapprox - fp)/fp *100) # derivative and relative error in %
end

# ╔═╡ 1a994a91-bcb2-4df6-9862-3f6eb696eb21
# Central difference approximation
let
	fp = -1.0 * exp(-1.0^2.0/2.0) # True derivative
	f(x) = exp(-x^2/2)
	c = 1
	h = 1e-8
	cdapprox = (f(c+h) - f(c-h)) / (2h)
	(cdapprox, (cdapprox - fp)/fp *100) # derivative and relative error in %
end

# ╔═╡ ccdb8284-755a-424c-ba86-f212d16813e9
md"""
## Finite Difference Methods in Julia

The [FiniteDifferences](https://github.com/JuliaDiff/FiniteDifferences.jl) and [FiniteDiff](https://github.com/JuliaDiff/FiniteDiff.jl) packages provide performant interfaces for differentiation based on finite differences.

The package implements

`forward_fdm`, `central_fdm`, and `backward_fdm` to compute the derivative from functions.

"""

# ╔═╡ 0daf4a2e-4dbb-4087-b4d0-796c97423356
let
	fp = -1.0 * exp(-1.0^2.0/2.0) # True derivative
	f(x) = exp(-x^2/2)
	# 4: number of grid points, 1: order of derivative, f: function, 1: value
	cfdm = central_fdm(6, 1)(f, 1) 
	(cfdm, (cfdm - fp)/fp *100) # derivative and relative error in %
end

# ╔═╡ a8e2053a-4557-490e-9a4e-babc6a0fc8e6
# Test second derivative around x = 2 
let	
	fpp = (2.0^2.0 - 1) * exp(-2.0^2.0/2.0) # True derivative
	f(x) = exp(-x^2/2)
	# 4: number of grid points, 2: order of derivative, f: function, 2: value
	cfdm = central_fdm(6, 2)(f, 2) 
	(cfdm, (cfdm - fpp)/fpp * 100) # derivative and relative error in %
end

# ╔═╡ 737aadb2-41d2-4ca4-b096-349b17d67666
md"""
## Automatic Differentiation

*Stop approximating derivatives!*

The field of automatic differentiation provides methods for automatically computing exact derivatives (up to floating-point error) given only the function `f` itself. Some methods use many fewer evaluations of ``f`` than would be required when using finite differences. In the best case, the **exact gradient of ``f``** can be evaluated for the cost of ``O(1)`` evaluations of `f` itself. 

 The `ForwardDiff` package provides one of [several](https://juliadiff.org/) ways for `Julia` to compute automatic derivatives. 
"""


# ╔═╡ e41e96c0-2c3b-497d-8a39-2aaaccc17423
let
	f(x) = exp(-x^2/2)
	fp = -1.0 * exp(-1.0^2.0/2.0)
	autod = ForwardDiff.derivative(f, 1)   # derivative is qualified by a module name
	(autod, (autod - fp)/fp * 100) # derivative and relative error in %
end

# ╔═╡ cb12b4de-5957-499b-b45e-cef684675903
md"""
# Vector Operations

## Vectors vs. Points
A [vector](https://en.wikipedia.org/wiki/Euclidean_vector)
mathematically is a geometric object with two attributes a magnitude
and a direction. (The direction is undefined in the case the magnitude
is $0$.) Vectors are typically visualized with an arrow, where the
anchoring of the arrow is context dependent and is not particular to a
given vector.

Vectors and points are related, but distinct. They are identified when the tail of the vector is taken to be the origin. Let's focus on ``3`` dimensions. Mathematically, the notation for a point is $p=(x,y,z)$ while the notation for a vector is $\vec{v} = \langle x, y, z \rangle$. The $i$th component in a vector is referenced by a subscript: $v_i$. With this, we may write a typical vector as $\vec{v} = \langle v_1, v_2, \dots, v_n \rangle$ and a vector in $n=3$ as $\vec{v} =\langle v_1, v_2, v_3 \rangle$.
The different grouping notation distinguishes the two objects. As another example, the notation $\{x, y, z\}$ indicates a set. Vectors and points may be *identified* by anchoring the vector at the origin. Sets are quite different from both, as the order of their entries is not unique.

## Vectors in Julia

In `Julia`, the notation to define a point and a vector would be identical, using square brackets to group like-type values: `[x, y, z]`. The notation `(x,y,z)` would form a [tuple](https://en.wikipedia.org/wiki/Euclidean_vector) which though similar in many respects, are different, as tuples do not have the operations associated with a point or a vector defined for them.

The square bracket constructor has some subtleties:

* `[x,y,z]` calls `vect` and creates a 1-dimensional array
* `[x; y; z]` calls `vcat` to **v**ertically con**cat**enate values together. With simple (scalar) values `[x,y,z]` and `[x; y; z]` are identical, but not in other cases. (For example, is `A` is a matrix then `[A, A]` is a vector of matrices, `[A; A]` is a matrix combined from the two pieces.
* `[x y z]`	 calls `hcat` to **h**orizontally con**cat**enate values together. If `x`, `y` are numbers then `[x y]` is *not* a vector, but rather a ``2``D array with a single row and two columns.
* finally `[w x; y z]` calls `hvcat` to horizontally and vertically concatenate values together to create a container in two dimensions, like a matrix.

Mathematically, a vector is a one-dimensional collection of numbers, a matrix a two-dimensional *rectangular* collection of numbers, and an array an $n$-dimensional rectangular-like collection of numbers. In `Julia`, a vector can hold a collection of objects of arbitrary type, though each will be promoted to a common type.
"""

# ╔═╡ 9b43916d-3b24-4fe7-8eea-b044aecf0681
x, y = [1, 2, 3], [4, 5, 6]

# ╔═╡ b9de777d-aae7-45fa-bc3f-34899b911fa9
xx = [1,2.0,3]

# ╔═╡ 7878efe3-04fd-4632-bca9-045565e82824
[x; y]

# ╔═╡ c49a2915-ca5a-4405-8625-3167dbc767a6
[x, y]

# ╔═╡ 5eca5911-287a-4252-9393-d8469acb34af
[x y]

# ╔═╡ 08aa5e12-5e0d-44bf-827d-eafb2ad5a238
aa= [1 2 3]

# ╔═╡ e0f30602-8498-4592-b461-b17c12c9e6b7
aa[:]

# ╔═╡ a8999060-be95-40ce-9d86-7aa48c97d317
[1,2,3]

# ╔═╡ 446c2b03-5204-4380-b250-c1bcb3f6dba5
md"""

## Addition an Subtraction

Addition can be visualized geometrically: put the tail of $\vec{v}$ at the tip of $\vec{u}$ and draw a vector from the tail of $\vec{u}$ to the tip of $\vec{v}$ and you have $\vec{u}+\vec{v}$. This is identical by $\vec{v} + \vec{u}$ as vector addition is commutative. Unless $\vec{u}$ and $\vec{v}$ are parallel or one has $0$ length, the addition will create a vector with a different direction from the two. Vector subtraction is performed by reversing the sign of one vector and adding. Addition and subtraction is

```math
\begin{array}
\textit{Commutative}: & \vec{u}+\vec{v} = \vec{v}+\vec{u} \\
\textit{Associative}: & (\vec{v}+\vec{w}) + \vec{u} = \vec{v}+ (\vec{u} + \vec{w})
\end{array}
```
"""

# ╔═╡ a2174988-7bb9-4ec9-a586-b38bf0d4c821
u, v, w = [1, 2], [4, 2], [3, 4]

# ╔═╡ 9f2348dc-726f-4320-9611-f94c736ba0fb
w1 = u + v

# ╔═╡ c4ad6ae2-065f-4438-a984-865e6440d43d
w2 = u - v

# ╔═╡ e40b137b-a15e-433a-9a3a-5838458b45af
begin 
	function plot1()
		u = [1, 2]
		v = [4, 2]
		w = u + v
		x = u - v
		p = [0,0]
		p1 = plot(xlabel = "x", ylabel = "y", title = "w1 = u+v")
		p1 = quiver!(unzip([p])..., quiver=unzip([u]), color = :darkred,)
		p1 = quiver!(unzip([u])..., quiver=unzip([v]), color = :steelblue3)
		p1 = quiver!(unzip([p])..., quiver=unzip([w]), color = :black)

		p2 = plot(xlabel = "x", ylabel = "y", title = "w2 = u-v")
		p2 = quiver!(unzip([p])..., quiver=unzip([u]), color = :darkred)
		p2 = quiver!(unzip([u])..., quiver=unzip([v]), color = :steelblue3)
		p2 = quiver!(unzip([u])..., quiver=unzip([-v]), color = :steelblue3, 
			ls = :dash)
		p2 = quiver!(unzip([p])..., quiver=unzip([x]), color = :black)
		
		plot(p1, p2, size = (500, 250))
	end
	plot1()
end

# ╔═╡ 031753c1-7fcf-4e74-bf37-66e241248bbb
md"""
## Mutliplication of a Vector by a Scalar

When a vector is multiplied by a scalr, the magnitdue of the vector is altered the the direction is not. When the scalar is negative, the direction is reversed.

```math
\begin{array}
\textit{Commutative}: & s\vec{v} = \vec{v}s \\
\textit{Associative}: & r(s\vec{v}) = (rs)\vec{v} \\
\textit{Distributive}: & (q+r+s)\vec{v} = q \vec{v} + r \vec{v} + s \vec{v}	
\end{array}
```

"""

# ╔═╡ 74a35dd7-bd9a-4d02-bcd7-85e58b3ff4bf
2*u

# ╔═╡ 7e34a490-1a94-4bc1-b657-61bc197282b3
-3*u

# ╔═╡ f26144bc-3dac-4218-a76a-23a2d7baeb69
begin 
	function plot2()
		u = [1, 2]
	
		p = [0,0]
		p1 = plot(xlabel = "x", ylabel = "y", title = "w1 = 2u")
		p1 = quiver!(unzip([p])..., quiver=unzip([u]), color = :darkred, lw = 3)
		p1 = quiver!(unzip([p])..., quiver=unzip([2.0*u]), color = :steelblue3)

		p2 = plot(xlabel = "x", ylabel = "y", title = "w2 = -3u")
		p2 = quiver!(unzip([p])..., quiver=unzip([u]), color = :darkred, lw = 3)
		p2 = quiver!(unzip([p])..., quiver=unzip([-3u]), color = :steelblue3)
		
		plot(p1, p2, size = (500, 250))
	end
	plot2()
end

# ╔═╡ 646917d4-b5b0-4e0d-8a68-84ba3971e28e
md"""
## Magnitude and Direction

If a vector $\vec{v} = \langle v_1, v_2, \dots, v_n\rangle$ then the *norm* (also Euclidean norm or length) of $\vec{v}$ is defined by:

```math
\| \vec{v} \| = \sqrt{ v_1^2 + v_2^2 + \cdots + v_n^2}.
```

```math
\begin{array}
\text{Associative} & \| c\vec{v} \| = |c| \| \vec{v} \| \\
\textit{Triangle Propery} & \| \vec{v} + \vec{w} \| \leq \| \vec{v} \| + \| \vec{w} \| \\
\textit{Symmetry} & \| -\vec{v} \| = \| \vec{v} \|	
\end{array}
```

In `Julia`, the `norm` function is in the standard library, `LinearAlgebra`, which must be loaded first through the command `using LinearAlgebra`.
"""

# ╔═╡ ad7f995e-bc2d-4192-ba3c-4a2665af73fd
norm([2,2,1])

# ╔═╡ d1024fa4-5a13-4ce7-8efd-b0321e675cd9
norm(2.0*[2,2,1]) === 2.0*norm([2,2,1]) 

# ╔═╡ 4e203179-00f9-4de0-bcd3-e4c4e1e08b2f
norm(-2*[2,2,1]) == norm(2*[2,2,1])

# ╔═╡ d6b4f284-1638-45db-ad25-5749aefd550f
md"""
## Scalar Product (or Dot Product) 

The scalar product of two vectors ``\vec{v}`` and ``\vec{w}`` is defined by

```math
\vec{v} \cdot \vec{w} = \|\vec{v}\| \|\vec{w}\| \cos(\theta).
```

where ``\theta`` is the angle betweent the vectors ``\vec{v}`` and ``\vec{w}``. If $\vec{v} = \langle v_1, v_2, \dots, v_n\rangle$ and $\vec{w} = \langle w_1, w_2, \dots, w_n\rangle$, then the *dot product* of $\vec{v}$ and $\vec{w}$ is defined by:

```math
\vec{v} \cdot \vec{w} = v_1 w_1 + v_2 w_2 + \cdots + v_n w_n.
```

It follows that

```math
\vec{v} \cdot \vec{v} = \| \vec{v} \|^2  
```

The rules governing the dot product are

```math
\begin{array}
\textit{Commutative}: & \vec{u} \cdot \vec{v} = \vec{v} \cdot \vec{u} \\
\textit{Not Associative}: & (\vec{u} \cdot \vec{v})\vec{w} \ne \vec{u}  (\vec{v} \cdot \vec{w}) \\
\textit{Distributive}: & \vec{u} \cdot [ \vec{v} + \vec{w} ] = \vec{u} \cdot \vec{v} + \vec{v} \cdot \vec{w} 
\end{array}
```

The dot product is computed in `Julia` by the `dot` function, which is in the `LinearAlgebra` package of the standard library.

In `Julia`, the unicode operator entered by `\cdot[tab]` can also be used to mirror the math notation:

```julia;
𝒖 ⋅ 𝒗   # u \cdot[tab] v
```
"""

# ╔═╡ 5badd08a-bb90-4401-8dd9-3cefc7f45159
[1, 2] ⋅ [4, 2] # 1*4 + 2*2 = 8

# ╔═╡ 3d38088d-9ed7-4497-9132-2ada083c20b1
(u ⋅ v) == (v ⋅ u) # test communtative property

# ╔═╡ bd349f83-58a1-4965-9750-18cec4ab2542
(u ⋅ v) * w # Note (u ⋅ v) is a scalar thus (u ⋅ v)*w ≠ u*(v ⋅ w)

# ╔═╡ fa153793-fcc2-43e5-bebb-96ccaa1094d2
u * (v ⋅ w) # Note (u ⋅ v) is a scalar thus (u ⋅ v)*w ≠ u*(v ⋅ w)

# ╔═╡ 811c3176-2dc2-4d7d-87d7-320d0e0ce4b9
u ⋅ (v + w) == u ⋅ v + u ⋅ w # Test distributive

# ╔═╡ 0217effe-642e-4408-b14d-848ca12b6ec0
md"""

Two (non-parallel) vectors will lie in the same "plane", even in higher dimensions. Within this plane, there will be an angle between them within $0 < θ < \pi$. The angle between the two vectors is the same regardless of their order of consideration.


Denoting $\hat{v} = \vec{v} / \| \vec{v} \|$, the unit vector in the direction of $\vec{v}$, then 

```math 
\cos(\theta) = \hat{v} \cdot \hat{w}
```

The angle between the vectors does not depend on their magnitude. Since 
``cos(\pi/2)=0``,  

$\vec{v} \cdot \vec{w} = 0$ 

implies that the two vectors are orthogonal (at right angles to each other). This holds in any dimension.
"""

# ╔═╡ bf0ab7c8-5826-42a0-ba41-6f4879e7a080
begin 
	function plot3()
		u = [1, 2]
		v = [-1, 0.5]
		
		a = [2, 2]
		b = [-1, 4]
		p = [0,0]
		p1 = plot(xlabel = "x", ylabel = "y", title = "u ⋅ v = $(u ⋅ v)")
		p1 = quiver!(unzip([p])..., quiver=unzip([u]), color = :darkred, lw = 1)
		p1 = quiver!(unzip([p])..., quiver=unzip([v]), color = :steelblue3, lw =1)

		p2 = plot(xlabel = "x", ylabel = "y", title = "u ⋅ v = $(a ⋅ b)")
		p2 = quiver!(unzip([p])..., quiver=unzip([a]), color = :darkred, lw = 3)
		p2 = quiver!(unzip([p])..., quiver=unzip([b]), color = :steelblue3)
		
		plot(p1, p2, size = (500, 250))
	end
	plot3()
end

# ╔═╡ e41fd78b-5385-43a4-aded-42663016ebc2
md"""
## Vector Product (or Cross Product) 

The vector product (or cross product) of two ``3``-dimensional vectors $\vec{v}$ and ``\vec{w}`` is defined 

```math
[\vec{v} \times \vec{w}] =  \| \vec{v} \| \| \vec{w} \| \sin(\theta) \hat{n}
```


with $\theta$ being the angle in $[0, \pi]$ between $\vec{v}$ and $\vec{w}$.
The direction of the cross product is such that it is *orthogonal* to *both* $\vec{v}$ and $\vec{w}$. There are two such directions, to identify which is correct, the [right-hand rule](https://en.wikipedia.org/wiki/Cross_product#Definition) is used. This rule points the right hand fingers in the direction of $\vec{v}$ and curls them towards $\vec{w}$ (so that the angle between the two vectors is in $[0, \pi]$). The thumb will point in the direction. Call this direction $\hat{n}$, a normal unit vector. It immediately follows that

```math
[\vec{v} \times \vec{v}] = 0
```

The definition in terms of its components is:

```math
\vec{v} \times \vec{w} = \langle v_2 w_3 - v_3 w_2, v_3 w_1 - v_1 w_3, v_1 w_2 - v_2 w_1 \rangle.
```

Define $\hat{i}$, $\hat{j}$, and $\hat{k}$ to represent unit vectors in the $x$, $y$, and $z$ direction. Then a vector $\langle v_1, v_2, v_3 \rangle$ could be written $v_1\hat{i} + v_2\hat{j} + v_3\hat{k}$. With this the cross product of $\vec{v}$ and $\vec{w}$ is the vector associated with the *determinant* of the matrix

```math
\left[
\begin{array}{}
\hat{i} & \hat{j} & \hat{k}\\
v_1   & v_2   & v_3\\
w_1   & w_2   & w_3
\end{array}
\right]
```

The rules governing the cross product are

```math
\begin{array}
\textit{Scalar\ multiplication}: & (c\vec{u})\times\vec{v} = c(\vec{u}\times\vec{v}) \\
\textit{Distributive over addition}: & \vec{u} \times (\vec{v} + \vec{w}) = \vec{u}\times\vec{v} + \vec{u}\times\vec{w} \\
\textit{Anti commutative}: & \vec{u} \times \vec{v} = - \vec{v} \times \vec{u} \\
\textit{Not associative}: & (\vec{u}\times\vec{v})\times\vec{w} \ne \vec{u}\times(\vec{v}\times\vec{w}) 
\end{array}
```

In `Julia`, the `cross` function from the `LinearAlgebra` package implements the cross product. 

```julia;
u × v # u \times[tab] v
```

"""

# ╔═╡ a2e1d98e-a755-42e8-9a16-13e2bed00bdc
[0, 2, 4] × [2, 4, 5] # 2*5 - 4*4 = -6; 4*2 - 0*5 = 8, 0*4 - 2*2 = -4 

# ╔═╡ 4a5ece52-bf23-4e22-9819-7fa507e17517
[0, 2, 4] × [2, 4, 5] == - ([2, 4, 5] × [0, 2, 4]) # anti-commutative

# ╔═╡ 891b5c2d-bfe3-4eb3-ad9f-508b48dd4f98
[0, 2, 4] × ([2, 4, 5] + [1, -1, 10]) == [0, 2, 4] × [2, 4, 5] + [0, 2, 4] × [1, -1, 10] # Distributive over addition

# ╔═╡ 24727b89-5b90-4a93-868b-f11901bb2c87
begin 
	function plot4()

		p = plot(xlabel = "x", ylabel = "y", zlabel = "z", 
			title = "u (black), v (blue), u × v (red)" )
		c = [-1, 0, 0.1]
		d = [0, 1, 0.1]
		e = c × d
		p = quiver!([0], [0], [0], quiver = ([c[1]], [c[2]], [c[3]]), color = :black)
		p = quiver!([0], [0], [0], quiver = ([d[1]], [d[2]], [d[3]]), 
			color = :steelblue3)
		p = quiver!([0], [0], [0], quiver = ([e[1]], [e[2]], [e[3]]), 
			color = :darkred, lw = 3)
	end
	plot4()
end

# ╔═╡ e3bf87cd-0574-4804-80cf-34951a469044
md"""

## Summary of Vector Operations in Julia

This table shows common usages of the symbols for various multiplication types: `*`, $\cdot$, and $\times$:


| Symbol   | inputs         | output      | type                  |
|:--------:|:-------------- |:----------- |:------                |
| `*`      | scalar, scalar | scalar      | regular multiplication |
| `*`      | scalar, vector | vector      | scalar multiplication |
| `*`      | vector, vector | *undefined* |                       |
| $\cdot$  | scalar, scalar | scalar      | regular multiplication |
| $\cdot$  | scalar, vector | vector      | scalar multiplication  |
| $\cdot$  | vector, vector | scalar      | dot product           |
| $\times$ | scalar, scalar | scalar      | regular multiplication |
| $\times$ | scalar, vector | undefined   |                       |
| $\times$ | vector, vector | vector      | cross product (``3``D)|

"""

# ╔═╡ 8d69e12e-eedb-4dd8-b2c7-61011915981e
md"""
# Matrix Operations

When there is a system of equations, something like:

```math
\begin{array}{}
3x  &+& 4y  &- &5z &= 10\\
3x  &-& 5y  &+ &7z &= 11\\
-3x &+& 6y  &+ &9z &= 12,
\end{array}
```

Then we might think of $3$ vectors $\langle 3,4,-5\rangle$, $\langle 3,-5,7\rangle$, and $\langle -3,6,9\rangle$ being dotted with $\langle x,y,z\rangle$. Mathematically, matrices and their associated algebra are used to represent this. In this example, the system of equations above would be represented by a matrix and two vectors:

```math
M = \left[
\begin{array}{}
3 & 4 & -5\\
5 &-5 &  7\\
-3& 6 & 9
\end{array}
\right],\quad
\vec{x} = \langle x, y , z\rangle,\quad
\vec{b} = \langle 10, 11, 12\rangle,
```

and the expression $M\vec{x} = \vec{b}$. The matrix $M$ is a rectangular collection of numbers or expressions arranged in rows and columns with certain algebraic definitions. There are $m$ rows and $n$ columns in an $m\times n$ matrix. In this example $m=n=3$, and in such a case the matrix is called square. A vector, like $\vec{x}$ is usually identified with the $n \times 1$ matrix (a column vector). Were that done, the system of equations would be written $M\vec{x}=\vec{b}$.

If we refer to a matrix $M$ by its components, a convention is to use $(M)_{ij}$ or $m_{ij}$ to denote the entry in the $i$th *row* and $j$th *column*. Following `Julia`'s syntax, we would use $m_{i:}$ to refer to *all* entries in the $i$th row, and $m_{:j}$ to denote *all* entries in the $j$ column.

"""

# ╔═╡ 1964d79e-cfa8-4b38-bd02-ba5fda5a64bc
md"""
## Matrices in Julia
A matrix in `Julia` is defined component by component with `[]`. Row entries are separated with spaces and columns with semicolons:

```julia;
M = [3 4 -5; 5 -5 7; -3 6 9]
```

Space is the separator, which means computing a component during definition (i.e., writing `2 + 3` in place of `5`) can be problematic, as no space can be used in the computation, lest it be parsed as a separator. In `Julia`, entries in a matrix (or a vector) are stored in a container with a type wide enough accomodate each entry. 

Matrices may also be defined from blocks. This example shows how to make two column vectors into a matrix:

```julia;
u = [10, 11, 12]
v = [13, 14, 15]
[u v]   # horizontally combine
```

Vertically combining the two will stack them:

```julia;
[u; v]
```
"""


# ╔═╡ 0977c4d2-54b9-48ef-8cc7-6c7827c52cb3
[3 4 -5; 5 -5 7; -3 (6 +2im) 9]

# ╔═╡ 5e36c68f-6dcb-4ce4-8084-401153d201f2
[[10, 11, 12] [13, 14, 15]]

# ╔═╡ ecc22fa1-d7b2-4b56-913e-4707f0b4d68e
md"""
## Multiplication of a Matrix by a Scalar

Scalar multiplication of a matrix $M$ by $c$ just multiplies each entry by $c$, so the new matrix would have components defined by $cm_{ij}$.
"""

# ╔═╡ af98df01-1a8d-4059-9a3e-71196b3e4ad5
2.0*[[1, 2, 3] [4, 5, 6] [7, 8, 9]]

# ╔═╡ cd140c37-4f2e-421a-b643-82f25481c3f6
md"""
## Addition and Subtraction

Matrices of the same size, like vectors, have addition defined for them. As with scalar multiplication, addition is defined component wise. So $A+B$ is the matrix with $ij$ entry $A_{ij} + B_{ij}$.
"""

# ╔═╡ d7f6c238-3fe2-4a18-aff5-11c6324edd67
M1 = [[1, 2] [3, 4]] 

# ╔═╡ 342e5e8f-455b-44de-9f67-9ee27717cf3c
M2 = [[1, 1] [1, 2]]

# ╔═╡ f88f2cca-4e26-4d53-9617-993786ed0353
M1 + M2

# ╔═╡ 948f625e-8e47-46ca-be5e-beddd92b3237
md"""
## Matrix Multiplication

Matrix multiplication may be viewed as a collection of dot product operations. First, matrix multiplication is only  defined between $A$ and $B$, as $AB$, if the size of $A$ is $m\times n$ and the size of $B$ is $n \times k$. That is the number of columns of $A$ must match the number of rows of $B$ for the left multiplication of $AB$ to be defined. If this is so, then we have the $ij$ entry of $AB$ is:

```math
(AB)_{ij} = A_{i:} \cdot B_{:j}.
```

That is, if we view the $i$th row of $A$ and the $j$th column of B as  *vectors*, then the $ij$ entry is the dot product.

This is why $M$ in the example above, has the coefficients for each equation in a row and not a column, and why $\vec{x}$ is thought of as a $n\times 1$ matrix (a column vector) and not as a row vector.

Matrix multiplication between $A$ and $B$ is not, in general, commutative. Not only may the sizes not permit $BA$ to be found when $AB$ may be, there is just no guarantee when the sizes match that the components will be the same.

Matrix multiplication in `Julia` is obtained via the `*` symbol.


When the multiplication is broadcasted though, with `.*`, the operation will be component wise:

```julia;
M .* M   # component wise (Hadamard product)
```


"""

# ╔═╡ cd19c076-9943-4223-9ef5-3a72f43c3540
[1 2; 3 4] * [1, 2] # 1*1 + 2*2 = 5; 3*1 + 4*2 = 11 

# ╔═╡ 17cb4739-33fc-41bb-86c8-2c063bb65cad
[1 2] ⋅ [1 2]

# ╔═╡ 29d53bd2-6517-41f2-9d2e-679a8c0f8f90
[3 4] ⋅ [1 2]

# ╔═╡ 83801d66-89da-4161-b4b8-1962a97ac6a8
[1 2; 3 4] .* [1 2; 3 4] # Hadamard product

# ╔═╡ 0331ab7a-5529-4320-89ec-c448dbe54194
[1 2; 3 4] * [1 2; 3 4] # Matrix Multiplication

# ╔═╡ 9ba04cc5-b33e-4478-b0eb-8f0d8ab93ce2
[1 3] ⋅ [1 3]

# ╔═╡ 4be5acc1-b66e-41a2-8818-6411c1f67054
exp([1, 2])

# ╔═╡ 1308e420-e393-4c24-a508-106f346c8ad6
md"""

## Matrix Transpose

The *transpose* of a matrix flips the difference between row and column, so the $ij$ entry of the transpose is the $ji$ entry of the matrix. This means the transpose will have size $n \times m$ when $M$ has size $m \times n$. Mathematically, the transpose is denoted $M^t$.

In `Julia` the transpose is either the `'` operator or the `transpose` function.

(However, this is not true for matrices with complex numbers as `'` is the "adjoint," that is, the transpose of the matrix *after* taking complex conjugates.)
"""

# ╔═╡ 911b5ad6-5991-4472-8229-7a091dfc8a80
matrix = [[1,2,3] [4,5,6] [7,8,9]]

# ╔═╡ aaccb6c0-c3c5-4e2e-b529-58aeef9bd8e7
matrix' # Transpose

# ╔═╡ 6fd6892a-d7ef-435d-89d7-3490d0aa392a
transpose(matrix)

# ╔═╡ f6a62dc5-5449-4e63-8344-afd8d97275f0
(matrix')' # Transposing twice restores the original matrix

# ╔═╡ 0d01c8fe-de93-43b3-bc3a-fd1f3d8426c4
md"""
## Matrix Determinant

The *determinant* of a *square* matrix is a number that can be used to infer if the matrix is invertible (or the system of equations has a unique solution). The determinant may be computed different ways, but its [definition](https://en.wikipedia.org/wiki/Leibniz_formula_for_determinants) by the Leibniz formula is common. Two special cases are all we need. The $2\times 2$ case and the $3 \times 3$ case:

```math
\left|
\begin{array}{}
a&b\\
c&d
\end{array}
\right| =
ad - bc, \quad
\left|
\begin{array}{}
a&b&c\\
d&e&f\\
g&h&i
\end{array}
\right| =
a \left|
\begin{array}{}
e&f\\
h&i
\end{array}
\right|
- b \left|
\begin{array}{}
d&f\\
g&i
\end{array}
\right|
+c \left|
\begin{array}{}
d&e\\
g&h
\end{array}
\right|.
```

The $3\times 3$ case shows how determinants may be [computed recursively](https://en.wikipedia.org/wiki/Determinant#Definition), using "cofactor" expansion.



The determinant is found by `det` provided by the `LinearAlgebra` package:

```julia;
det(M)
```

"""

# ╔═╡ 1ba9f7d8-0777-4991-a4be-aa1faabc59b0
det([[1,2,3] [4,5,6] [7,-8,9]]) # If the determinant is ≠ 0 the matrix is invertible

# ╔═╡ 98717010-786e-47a6-b930-bdc424fab88f
det([[0,1,2] [2,1,0] [0,0,0]]) # No unique solution exists

# ╔═╡ 6d262382-efd0-42f2-b2c4-bb0cb4699124
md"""

## Matrix Inverse
If $M$ is a square matrix and its determinant is non-zero, then there is an *inverse* matrix, denoted $M^{-1}$, with the properties that $MM^{-1} = M^{-1}M = I$, where $I$ is the diagonal matrix of all $1$s called the identify matrix.

In `Julia` the matrix inverse is either obtained via `^-1` or `inv(M)`
"""

# ╔═╡ 9a9ffa95-3a1a-4bf3-a915-ea72406817a7
Matrix1 = [[1,2,3] [4,5,6] [7,-8,9]]

# ╔═╡ da83cb32-0ea0-4a3b-a4d0-6d358d17dbc2
inv(Matrix1)

# ╔═╡ dd976bda-0281-4196-a1ce-6eff9e7d3717
Matrix1^-1

# ╔═╡ de1c4baf-2318-4a68-a905-6c769bc66a7f
inv([[0,1,2] [2,1,0] [0,0,0]]) # Cannot do this, det = 0

# ╔═╡ 40c1a126-178b-4fad-84f9-58846a599043
md"""
## Other Named Matrices

##### Diagonal Matrix
Square matrices with $0$ entries below the diagonal are called upper triangular; square matrices with $0$ entries above the diagonal are called lower triangular matrices
"""

# ╔═╡ 5e63c04e-887c-409a-8336-0fbbae1032d0
c = LowerTriangular(rand(10,10))

# ╔═╡ b8d79025-2b53-4710-8727-58cf14ebcdc3
md"""
##### Identity Matrix 
Square matrices which are $0$ except possibly along the diagonal are diagonal
matrices; and a diagonal matrix whose diagonal entries are all $1$ are
called an *identity matrix*.
"""

# ╔═╡ 41291c12-a2fa-4897-a181-21709e47ddb0
Matrix(1I, 5, 5)

# ╔═╡ 28de043e-2255-4318-be1b-e011f599a09f
md"""
# Vector Calculus

"""

# ╔═╡ 3b0fb866-9547-4b0d-b3df-d6d6ae60baff
md"""

## Vector-Valued Functions

### Definition

A function $\vec{f}: R \rightarrow R^n$, $n > 1$ is called a vector-valued function. Some examples:


```math
\vec{f}(t) = \langle \sin(t), 2\cos(t) \rangle, \quad
\vec{g}(t) = \langle \sin(t), \cos(t), t \rangle, \quad
\vec{h}(t) = \langle 2, 3 \rangle + t \cdot \langle 1, 2 \rangle.
```

The components themselves are also functions of $t$, in this case univariate functions. Depending on the context, it can be useful to view vector-valued functions as a function that returns a vector, or a vector of the component functions.


The above example functions have $n$ equal $2$, $3$, and $2$ respectively. We will see that many concepts of calculus for univariate functions ($n=1$) have direct counterparts.


(We use $\vec{f}$ above to emphasize the return value is a vector, but will quickly drop that notation and let context determine if $f$ refers to a scalar- or vector-valued function.)

### Representation in Julia

In `Julia`, the representation of a vector-valued function is straightforward: we define a function of a single variable that returns a vector. For example, the three functions above would be represented by:
"""

# ╔═╡ 87dfec70-511c-431e-a1d1-8ee752c812c1
# For a given t, these evaluate to a vector. For example:

let
	f(t) = [sin(t), 2*cos(t)]
	g(t) = [sin(t), cos(t), t]
	h(t) = [2, 3] + t * [1, 2]

	h(2)
end

# ╔═╡ 32fd4867-e125-4ac1-8bb4-604bdc56ad11
md"""

### Derivatives


If $\vec{f}(t)$ is  vector valued,  and $\Delta t > 0$ then we can consider the vector:


```math
\vec{f}(t + \Delta t) - \vec{f}(t)
```

For example, if $\vec{f}(t) = \langle 3\cos(t), 2\sin(t) \rangle$ and $t=\pi/4$ and $\Delta t = \pi/16$ we have this picture:
"""

# ╔═╡ 84f964e3-f1b3-48e1-8510-48c2149b1480
let
	f(t) = [3cos(t), 2sin(t)]
	t, Δt = pi/4, pi/16
	df = f(t + Δt) - f(t)

	plot(legend=false, size = (300,200))
	arrow!([0,0], f(t))
	arrow!([0,0], f(t + Δt))
	arrow!(f(t), df)
end

# ╔═╡ b2b58488-3ddd-4275-bf37-1923333e074e
md"""
The length of the difference appears to be related to the length of $\Delta t$, in a similar manner as the univariate derivative. The following limit defines the *derivative* of a vector-valued function:


```math
\vec{f}'(t) = \lim_{\Delta t \rightarrow 0} \frac{f(t + \Delta t) - f(t)}{\Delta t}.
```

The limit exists if the component limits do. The component limits are just the derivatives of the component functions. So, if $\vec{f}(t) = \langle x(t), y(t) \rangle$, then $\vec{f}'(t) = \langle x'(t), y'(t) \rangle$.


If the derivative is never $\vec{0}$, the curve is called *regular*. For a regular curve the derivative is a tangent vector to the parameterized curve, akin to the case for a univariate function. We can use `ForwardDiff` to compute the derivative in the exact same manner as was done for univariate functions:
"""

# ╔═╡ d688d12d-661c-4ac0-8651-ce26dc856518
# Visualize the tangential property through a graph:
let
	f(t) = [3cos(t), 2sin(t)]
	p = plot_parametric(0..2pi, f, legend=false, aspect_ratio=:equal)
	for t in [1,2,3]
    	arrow!(f(t), f'(t)) # add arrow with tail on curve, in direction of derivative
	end
	plot(p, size = (400,300))
end

# ╔═╡ 46ea45d8-90f8-42a6-9fb4-49fc46c2aef6
let
	t = 1
	f(t) = [3cos(t), 2sin(t)]
	f'(t), f''(t)  # Derivatives f' f'' etc is defined in CalculusWithJulia.jl 
end

# ╔═╡ 0ce54fa8-fda6-48f1-804b-1fd6100bc490
# Derivatives f' f'' etc is defined recursively in CalculusWithJulia.jl 
let
	D(f,n=1) = n > 1 ? D(D(f),n-1) : x -> ForwardDiff.derivative(f, float(x))
	Base.adjoint(f::Function) = D(f)         # allow f' to compute derivative
end

# ╔═╡ 56508bf0-2b27-4db7-871f-0522c3addbd7
# Standard approach using forward diff
let
	t = 1
	f(t) = [3cos(t), 2sin(t)]
	ForwardDiff.derivative(f, 1)
end

# ╔═╡ cd6e3483-14c8-42d5-b8be-c6abe6b86f74
md"""
### Derivative Rules

These two derivative formulas hold for vector-valued functions $R \rightarrow R^n$:

```math
\begin{align*}
(\vec{u} \cdot \vec{v})' &= \vec{u}' \cdot \vec{v} + \vec{u} \cdot \vec{v}',\\
(\vec{u} \times \vec{v})' &= \vec{u}' \times \vec{v} + \vec{u} \times \vec{v}'.
\end{align*}
```
"""

# ╔═╡ 90698737-5c04-41b1-bf81-a9e638fc407f
md""" 

##  Multivariate Scalar Functions 

### Definition

Consider a function $f: R^n \rightarrow R$. It has multiple arguments for its input (an $x_1, x_2, \dots, x_n$) and only one, *scalar*, value for an output. Some simple examples might be:

```math
\begin{align*}
f(x,y) &= x^2 + y^2\\
g(x,y) &= x \cdot y\\
h(x,y) &= \sin(x) \cdot \sin(y)
\end{align*}
```

For two examples from real life consider the elevation Point Query Service (of the [USGS](https://nationalmap.gov/epqs/)) returns the elevation in international feet or meters for a specific latitude/longitude within the United States. The longitude can be associated to an $x$ coordinate, the latitude to a $y$ coordinate, and the elevation a $z$ coordinate, and as long as the region is small enough, the $x$-$y$ coordinates can be thought to lie on a plane. (A flat earth assumption.)

Similarly,  a weather map, say of the United States, may show the maximum predicted temperature for a given day. This describes a function that take a position ($x$, $y$) and returns a predicted temperature ($z$).

Mathematically, we may describe the values $(x,y)$ in terms of a point, $P=(x,y)$ or a vector $\vec{v} = \langle x, y \rangle$ using the identification of a point with a vector. As convenient, we may write any of $f(x,y)$, $f(P)$, or $f(\vec{v})$ to describe the evaluation of $f$ at the value $x$ and $y$

In `Julia`, defining a scalar function is straightforward, the syntax following mathematical notation:

"""

# ╔═╡ 55d464e4-d04c-464e-9b3b-aa6f0b0021dc
# Examples on how to construct a scalar multivariate function
let 
	f(x,y) = x^2 + y^2
	g(x,y) = x * y
	h(x,y) = sin(x) * sin(y)
	f(1,2), g(2, 3), h(3,4)
end

# ╔═╡ 6fff5a73-33b4-4e77-828d-60995bd0cd7c
# Alternatively, the function may be defined using a vector argument
let
	g1(v) = v[1] * v[2]  # Single line function (no function keyword)

	# Multiline function 
	function g2(v)
    	x, y = v      # unpacks the vector
    	return x * y  # returns the value
	end
	g1([2, 3]), g2([2,3])
end

# ╔═╡ 97e14b3a-a056-40d4-ba55-abfeb9835dd7
md"""
### Visualizing Multivariate Scalar Functions

Suppose for the moment that $f:R^2 \rightarrow R$. The equation $z = f(x,y)$ may be visualized by the set of points in $3$-dimensions $\{(x,y,z): z = f(x,y)\}$. This will render as a surface, and that surface will pass a "vertical line test", in that each $(x,y)$ value corresponds to at most one $z$ value. We will see alternatives for describing surfaces beyond through a function of the form $z=f(x,y)$. These are similar to how a curve in the $x$-$y$ plane can be described by a function of the form $y=f(x)$ but also through an equation of the form $F(x,y) = c$ or through a parametric description, such as is used for planar curves. For now though we focus on the case where $z=f(x,y)$.

In `Julia`, plotting such a surface requires a generalization to plotting a univariate function where, typically, a grid of evenly spaced values is given between some $a$ and $b$, the corresponding $y$ or $f(x)$ values are found, and then the points are connected in a dot-to-dot manner.

Here, a two-dimensional grid of $x$-$y$ values needs specifying, and the corresponding $z$ values found. As the grid will be assumed to be regular only the $x$ and $y$ values need specifying, the set of pairs can be computed. The $z$ values, it will be seen, are easily computed. This cloud of points is plotted and each cell in the $x$-$y$ plane is plotted with a surface giving the $x$-$y$-$z$, $3$-dimensional, view. One way to plot such a surface is to tessalate the cell and then for each triangle, represent a plane made up of the $3$ boundary points.
"""

# ╔═╡ 60ab4bde-a284-4588-849a-f78deb8bc01f
let 
	f(x, y) = x^2 + y^2

	xs = range(-2, 2, length=100)
	ys = range(-2, 2, length=100)

	plot(xs, ys, f, xlabel = "x", ylabel = "y", zlabel = "z", size = (500, 250),
		seriestype=:surface)
end

# ╔═╡ 2f4fdb5f-4cb3-46e3-ad36-46024e4ca686
md"""

It is possible to call `surface(xs, ys, zs)` where `zs` is not a vector, but rather a *matrix* of values corresponding to a grid described by the `xs` and `ys`. A matrix is a rectangular collection of values indexed by row and column through indices `i` and `j`. Here the values in `zs` should satisfy: the $i$th row and $j$th column entry should be $z_{ij} = f(x_i, y_j)$ where $x_i$ is the $i$th entry from the `xs` and $y_j$ the $j$th entry from the `ys`.

This array comprehension appraoch `zs = [f(x,y) for y in ys, x in xs]` can be also used in `Python`. `Matlab` provides the `meshgrid` function to achieve this.

"""

# ╔═╡ cb9bfdbe-b338-42d3-a194-fc13b42346f1
let 
	f(x, y) = x^2 + y^2
	xs = range(-2, 2, length=100)
	ys = range(-2, 2, length=100)
	zs = [f(x,y) for y in ys, x in xs]
	plot(xs, ys, zs, xlabel = "x", ylabel = "y", zlabel = "z", size = (500, 250),
		seriestype=:surface)
end

# ╔═╡ a13b9789-fa4b-45bb-a083-81d3df4d4c86
let # zs is a matrix!
	f(x, y) = x^2 + y^2
	xs = range(-2, 2, length=10)
	ys = range(-2, 2, length=100)
	zs = [f(x,y) for y in ys, x in xs]
end

# ╔═╡ 762b2923-09f9-4a1d-aba6-aa0fedfb75b9
let # 3D surface plots are also visualizes as contours or h
	f(x, y) = x^2 + y^2
	xs = range(-2, 2, length=100)
	ys = range(-2, 2, length=100)
	zs = [f(x,y) for y in ys, x in xs]
	p1 = plot(xs, ys, zs, xlabel = "x", ylabel = "y", zlabel = "z", 
		seriestype=:contour)
	p2 = plot(xs, ys, zs, xlabel = "x", ylabel = "y", zlabel = "z", 
		seriestype=:heatmap)
	plot(p1, p2, layout = grid(1,2), size = (650, 250), bottom_margin = 20px)
end

# ╔═╡ 93e4bce7-6b75-410b-bf55-6d3cafbd7cc7
md"""
## The Gradient
### Definition

The *gradient* of a scalar function $f$ indicates the direction of steepest ascent and is a vector comprised of the partial derivatives:

```math
\nabla f(x_1, x_2, \dots, x_n) = \langle
\frac{\partial f}{\partial x_1},
\frac{\partial f}{\partial x_2}, \dots,
\frac{\partial f}{\partial x_n} \rangle.
```

As seen, the gradient is a vector-valued function, but has, also, multivariable inputs. It is a function from $R^n \rightarrow R^n$.

The gradient of such a function is designated ``\text{grad} f`` or ``\nabla f``. The rules governing the *gradient* operation are 


```math
\begin{array}
\textit{Not Communatative}: & \nabla f \ne f {\nabla} \\
\textit{Not Associative}: & ({\nabla} f)g \ne {\nabla} (fg) \\
\textit{Distributive}: & {\nabla} (f+g) = {\nabla} f + {\nabla} g
\end{array}
```


##### Example


Let $f(x,y) = x^2 - 2xy$, then to compute the partials, we just treat the other variables like a constant. (This is consistent with the view that the partial derivative is just a regular derivative along a line where all other variables are constant.)


Then

```math
\begin{align*}
\frac{\partial (x^2 - 2xy)}{\partial x} &= 2x - 2y\\
\frac{\partial (x^2 - 2xy)}{\partial y} &= 0 - 2x = -2x.
\end{align*}
```

Combining, gives $\nabla{f} = \langle 2x -2y, -2x \rangle$.
"""

# ╔═╡ 040b970b-b5cc-4798-b59c-3a4dab6752d6
md"""

### Finding Partial Derivatives in Julia


Two different methods are described, one for working with functions, the other symbolic expressions. This mirrors our treatment for vector-valued functions, where `ForwardDiff.derivative` was used for functions, and `SymPy`'s `diff` function for symbolic expressions.


Suppose, we consider $f(x,y) = x^2 - 2xy$. We may define it with `Julia` through:
"""

# ╔═╡ 5dde81b8-0f0c-4f95-a9ed-53cc45cf25f9
# The numeric gradient at a point, can be found using ForwardDiff.gradient:
let 
	f₂(x,y) = x^2 - 2x*y
	f₂(v) = f₂(v...)       # to handle vectors. Need not be defined each time
	pt₂ = [1, 2]
	ForwardDiff.gradient(f₂, pt₂), [2*pt₂[1] - 2*pt₂[2], -2*pt₂[1]]      
end

# ╔═╡ ee144366-3888-4511-859d-73894596c48c
md"""
### Example

The gradient is not a univariate function, a simple vector-valued function, or a scalar function, but rather a *vector field* (which will  be discussed later). For the case, $f: R^2 \rightarrow R$, the gradient will be a function which takes a point $(x,y)$ and returns a vector , $\langle \partial{f}/\partial{x}(x,y), \partial{f}/\partial{y}(x,y) \rangle$. We can visualize this by plotting a vector at several points on a grid. This task is made easier with a function like the following, which handles the task of vectorizing the values. It is provided within the `CalculusWithJulia` package:


```julia
function vectorfieldplot!(p, V; xlim=(-5,5), ylim=(-5,5), nx=10, ny=10, kwargs...)
    dx, dy = (xlim[2]-xlim[1])/nx, (ylim[2]-ylim[1])/ny
    xs, ys = xlim[1]:dx:xlim[2], ylim[1]:dy:ylim[2]

    ps = [[x,y] for x in xs for y in ys]
    vs = V.(ps)
	λ = 0.9 * minimum([u/maximum(getindex.(vs,i)) for (i,u) in enumerate((dx,dy))])

    quiver!(unzip(ps)..., quiver=unzip(λ * vs))
end
```
"""

# ╔═╡ 8cc764ce-147d-45d7-ae0e-19aafd34bba2
let
	f(x,y) = 2 - x^2 - 3y^2
	f(v) = f(v...)

	xs = ys = range(-2,2, length=50)
	p = contour(xs, ys, f, nlevels=12)
	vectorfieldplot!(p, ForwardDiff.gradient(f), xlim=(-2,2), ylim=(-2,2),
		nx=10, ny=10)
	plot(p, size = (400,300))
end

# ╔═╡ 1b8fa2fd-d0a1-4c5d-8d79-c9d3d26f38b0
md"""
## The Divergence
### Definition

The *divergence* of a vector field $F:R^3 \rightarrow R^3$ is given by

```math
\text{divergence}(F) =
\lim \frac{1}{\Delta V} \oint_S F\cdot\hat{N} dS =
\frac{\partial{F_x}}{\partial{x}} +\frac{\partial{F_y}}{\partial{y}} + \frac{\partial{F_z}}{\partial{z}}.
```

The limit expression for the divergence will hold for any smooth closed surface, $S$, converging on $(x,y,z)$, not just box-like ones.

The common denotation of divergence is either ``\operatorname {div} \mathbf {F}`` or  ``\nabla \cdot \mathbf {F}``. 

```math
\operatorname {div} \mathbf {F} = \nabla \cdot \mathbf {F} =\left({\frac {\partial }{\partial x}},{\frac {\partial }{\partial y}},{\frac {\partial }{\partial z}}\right)\cdot (F_{x},F_{y},F_{z})={\frac {\partial F_{x}}{\partial x}}+{\frac {\partial F_{y}}{\partial y}}+{\frac {\partial F_{z}}{\partial z}}.
```

The rules governing the divergence are

```math
\begin{array}
\textit{Not Communatative}: & ({\nabla}\cdot \vec{u}) \ne (\vec{u} \cdot {\nabla}) \\
\textit{Not Associative}: & ({\nabla} \cdot s\vec{u}) \ne ({\nabla}s \cdot \vec{u}) \\
\textit{Distributive}: & ({\nabla}\cdot [\vec{u} +\vec{w}]) = ({\nabla}\cdot \vec{u}) + ({\nabla}\cdot \vec{v}) 
\end{array}
```

In `Julia`, the divergence can be calculated using the [`Tensors.jl`](https://ferrite-fem.github.io/Tensors.jl/stable/man/automatic_differentiation/#Tensors.divergence) package. 

### Example

What is the divergence of $\mathbf{F}=(−y,xy,0)$


```math
\begin{array} & 
\frac{\partial F_x}{\partial x} = 0 & 
\frac{\partial F_y}{\partial y} = x & 
\frac{\partial F_z}{\partial z} = 0 
\end{array}
```

```math
\nabla \cdot \mathbf{F} = \frac{\partial F_x}{\partial x} + \frac{\partial F_y}{\partial y} + \frac{\partial F_z}{\partial z} = x
```

"""

# ╔═╡ 7ff231b1-a054-446c-8188-5e4111d03741
let
	f(a,b) = [-b, a*b]
	g(x) = Vec{3}([-x[2], x[1]*x[2], 0.0])
	xs = range(-1,1, length=50) 
	ys = range(-1,1, length=50)
	p = plot()
    vectorfieldplot!(p, f, xlim=(-1,1), ylim=(-1,1), nx=8, ny=8, color = :black)
	p1 = plot(p,  xlabel = "x", ylabel = "y", title = "F")

	# Note: Tensors requires the type to Vec{3} to work properly
	div = [Tensors.divergence(g, Vec{3}([x,y,0])) for y in ys, x in xs]
	p2 = heatmap(xs, ys, div, xlabel ="x", ylabel = "y", title = "∇⋅F")
	plot(p1, p2, layout = grid(1,2), size = (700,300), bottom_margin = 20px)
end

# ╔═╡ 92c5d7db-00ad-4d14-ad76-39deae233357
md"""
## The Curl

### Definition

The *curl* of a $3$-dimensional vector field $F=\langle F_x,F_y,F_z\rangle$ is defined by:

```math
\text{curl}\ \bf{F} \it = 
\langle \frac{\partial{F_z}}{\partial{y}} - \frac{\partial{F_y}}{\partial{z}},
\frac{\partial{F_x}}{\partial{z}} - \frac{\partial{F_z}}{\partial{x}},
\frac{\partial{F_y}}{\partial{x}} - \frac{\partial{F_x}}{\partial{y}} \rangle.
```

The curl has a formal representation in terms of a $3\times 3$ determinant, similar to that used to compute the cross product, that is useful for computation:


```math
\text{curl}\ \bf{F} = \nabla \times \bf{F} \it = \det
\begin{bmatrix}
\hat{i} & \hat{j} & \hat{k}\\
\frac{\partial}{\partial{x}} & \frac{\partial}{\partial{y}} & \frac{\partial}{\partial{z}}\\
F_x & F_y & F_z
\end{bmatrix}
```

The *curl* operation, like the divergence, is distributiver but not communative or associative. 

In `Julia`, the *curl* can be calculated using the [`Tensors.jl`](https://ferrite-fem.github.io/Tensors.jl/stable/man/automatic_differentiation/#Tensors.divergence) package. 

### Example

What is the curl of $\mathbf{F}=(−y,xy,0)$


```math
\begin{array} & 
\frac{\partial F_x}{\partial y} = -1 & 
\frac{\partial F_y}{\partial x} = y  & 
\frac{\partial F_x}{\partial z} = \frac{\partial F_y}{\partial z} = \frac{\partial F_z}{\partial x} = \frac{\partial F_z}{\partial y} = 0    
\end{array}
```

```math
\nabla \times \bf{F} \rm = (0-0, 0-0, y + 1) = (0,0,y + 1)
```


"""

# ╔═╡ 4ffe6f6e-8b1a-4b99-b742-689951842be5
let
	f(a,b) = [-b, a*b]
	g(x) = Vec{3}([-x[2], x[1]*x[2], 0.0])
	xs = range(-1,1, length=50) 
	ys = range(-1,1, length=50)
	
	# Note: Tensors requires the type to Vec{3} to work properly
	curl = [Tensors.curl(g, Vec{3}([x,y,0])) for y in ys, x in xs]
end

# ╔═╡ 935ef6ad-6f41-4074-ad24-5f1ad0de8869
md"""
## Interpretation of the Divergence and Curl 

The divergence and curl measure complementary aspects of a vector field. The divergence is defined in terms of flow out of an infinitesimal box, the curl is about rotational flow around an infinitesimal area patch. The radial vector field $F(x,y,z) = \langle x, y, z \rangle$ is an example of a divergent field. There is a constant outward flow, emanating from the origin. Here we picture the field when $z=0$:
"""

# ╔═╡ 62dba3f4-9649-4cca-a42a-e49c95adfa3d
let
	F12(x,y) = [x,y]
	F12(v) = F12(v...)
	p = plot(legend=false)
	vectorfieldplot!(p, F12, xlim=(-5,5), ylim=(-5,5), nx=10, ny=10)
	t0, dt = -pi/6, 2pi/6
	r0, dr = 3, 1
	plot!(p, unzip(r -> r * [cos(t0), sin(t0)], r0, r0 + dr)..., linewidth=3)
	plot!(p, unzip(r -> r * [cos(t0+dt), sin(t0+dt)], r0, r0 + dr)..., linewidth=3)
	plot!(p, unzip(t -> r0 * [cos(t), sin(t)], t0, t0 + dt)..., linewidth=3)
	plot!(p, unzip(t -> (r0+dr) * [cos(t), sin(t)], t0, t0 + dt)..., linewidth=3)

	plot(p, size = (300,250))
end

# ╔═╡ 9e1cacf6-8bcc-4342-bb54-c8de1524ec51
md"""
The vector field $F(x,y,z) = \langle -y, x, 0 \rangle$ is an example of a rotational field. This vector field rotates as seen in  this figure showing slices for different values of $z$:
"""

# ╔═╡ 6037feef-2817-49d0-82c4-97a41b2422b2
let
V(x,y,z) = [-y, x,0]
V(v) = V(v...)
p = plot([NaN],[NaN],[NaN], legend=false)
ys = xs = range(-2,2, length=10 )
zs = range(0, 4, length=3)
CalculusWithJulia.vectorfieldplot3d!(p, V, xs, ys, zs, nz=3)
plot!(p, [0,0], [0,0],[-1,5], linewidth=3)
p
end

# ╔═╡ dbd55643-6cb1-4e29-9520-ff05099976ef
md"""
The field has a clear rotation about the $z$ axis (illustrated with a line), the curl is a vector that points in the direction of the *right hand* rule as the right hand fingers follow the flow with magnitude given by the amount of rotation.


This is a bit misleading though, the curl is defined by a limit, and not in terms of a large box. The key point for this field is that the strength of the field is stronger as the points get farther away, so for a properly oriented small box, the integral along the closer edge will be less than that along the outer edge.


Consider a related field where the strength gets smaller as the point gets farther away but otherwise has the same circular rotation pattern

The curl of the  vector field $F(x,y,z) = \langle 0, 1+y^2, 0\rangle$ is $0$, as there is clearly no rotation as seen in this slice where $z=0$:
"""

# ╔═╡ 6bc47359-a215-48da-acf8-79700c8c75ac
let 
	p = vectorfieldplot((x,y) -> [0, 1+y^2], xlim=(-1,1), ylim=(-1,1), nx=10, ny=8)
	plot(p, size = (300,200))
end

# ╔═╡ 45094072-406c-49e2-a8a3-2ffdcc981562
md"""
Now consider a similar field  $F(x,y,z) = \langle 0, 1+x^2, 0,\rangle$. A slice is somewhat similar, in that the flow lines are all in the $\hat{j}$ direction:
"""

# ╔═╡ e923fac4-f348-4e5d-a566-610d1a7649fc
let
	p = vectorfieldplot((x,y) -> [0, 1+x^2], xlim=(-1,1), ylim=(-1,1), nx=10, ny=8)
	plot(p, size = (300,200))
end

# ╔═╡ 7fca8fb2-52ea-4cf6-8997-bcee8e4e3f84
let
	g(x) = Vec{3}([0.0, 1 + x[1]^2, 0.0])
	xs = range(-1,1, length=5) 
	ys = range(-1,1, length=5)
	curl = [Tensors.curl(g, Vec{3}([x,y,0])) for y in ys, x in xs]
end

# ╔═╡ 0fc62a68-07d1-45d1-8f15-b022e3625eb7
md"""
However, this vector field has a curl. The curl points in the $\hat{k}$ direction (out of the figure). A useful visualization is to mentally place a small paddlewheel at a point and imagine if it will turn. In the constant field case, there is equal flow on both sides of the axis, so it any forces on the wheel blades will balance out. In the latter example, if $x > 0$, the force on the right side will be greater than the force on the left so the paddlewheel would rotate counter clockwise. The right hand rule for this rotation will point in the upward, or $\hat{k}$ direction, as seen algebraically in the curl.
"""

# ╔═╡ 7d5c2ca6-130d-4050-9fed-64ace363fee5
html"<iframe width=\"720\" height=\"440\" src=\"https://www.youtube.com/embed/rB83DpBJQsE\" title=\"YouTube video player
\" frameborder=\"0\" allow=\"accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture\" allowfullscreen></iframe>"

# ╔═╡ 1d5901fd-bb47-4819-8a82-4fa433040401
md"""
## Summary of Vector Calculus

The divergence, gradient, and curl satisfy several algebraic [properties](https://en.wikipedia.org/wiki/Vector_calculus_identities).

Let $f$ and $g$ denote scalar functions, $R^3 \rightarrow R$ and $F$ and $G$ be vector fields, $R^3 \rightarrow R^3$.


### Linearity


As with the sum rule of univariate derivatives, these operations satisfy:

```math
\begin{align*}
\nabla(f + g) &= \nabla{f} + \nabla{g}\\
\nabla\cdot(F+G) &= \nabla\cdot{F} + \nabla\cdot{G}\\
\nabla\times(F+G) &= \nabla\times{F} + \nabla\times{G}.
\end{align*}
```

### Product Rule

The product rule $(uv)' = u'v + uv'$ has related formulas:

```math
\begin{align*}
\nabla{(fg)} &= (\nabla{f}) g + f\nabla{g} = g\nabla{f}  + f\nabla{g}\\
\nabla\cdot{fF} &= (\nabla{f})\cdot{F} +  f(\nabla\cdot{F})\\
\nabla\times{fF} &= (\nabla{f})\times{F} +  f(\nabla\times{F}).
\end{align*}
```

### Rules over Cross Products

The cross product of two vector fields is a vector field for which the divergence and curl may be taken. There are formulas to relate to the individual terms:

```math
\begin{align*}
\nabla\cdot(F \times G) &= (\nabla\times{F})\cdot G - F \cdot (\nabla\times{G})\\
\nabla\times(F \times G) &= F(\nabla\cdot{G}) - G(\nabla\cdot{F}) + (G\cdot\nabla)F-(F\cdot\nabla)G
\end{align*}
```

### Vanishing Properties
The curl of a gradient field is $\vec{0}$

```math
\nabla \times \nabla{f} = \vec{0},
```

The divergence of a curl field is $0$:

```math
\nabla \cdot(\nabla\times{F}) = 0.
```
"""

# ╔═╡ 00000000-0000-0000-0000-000000000001
PLUTO_PROJECT_TOML_CONTENTS = """
[deps]
CalculusWithJulia = "a2e0e22d-7d4c-5312-9169-8b992201a882"
FastGaussQuadrature = "442a2c76-b920-505d-bb47-c5924d526838"
FileIO = "5789e2e9-d7fb-5bc7-8068-2c6fae9b9549"
FiniteDifferences = "26cc04aa-876d-5657-8c51-4c34ba976000"
ForwardDiff = "f6369f11-7733-5829-9624-2563aa707210"
HypertextLiteral = "ac1192a8-f4b3-4bfe-ba22-af5b92cd3ab2"
Images = "916415d5-f1e6-5110-898d-aaa5f9f070e0"
LaTeXStrings = "b964fa9f-0449-5b57-a5c2-d3ea65f4040f"
LinearAlgebra = "37e2e46d-f89d-539d-b4ee-838fcccc9c8e"
PlotThemes = "ccf2f8ad-2431-5c83-bf29-c5338b663b6a"
Plots = "91a5bcdd-55d7-5caf-9e0b-520d859cae80"
PlutoUI = "7f904dfe-b85e-4ff6-b463-dae2292396a8"
Roots = "f2b01f46-fcfa-551c-844a-d8ac1e96c665"
SpecialFunctions = "276daf66-3868-5448-9aa4-cd146d93841b"
Tensors = "48a634ad-e948-5137-8d70-aa71f2a747f4"

[compat]
CalculusWithJulia = "~0.1.2"
FastGaussQuadrature = "~0.5.1"
FileIO = "~1.16.1"
FiniteDifferences = "~0.12.30"
ForwardDiff = "~0.10.36"
HypertextLiteral = "~0.9.4"
Images = "~0.26.0"
LaTeXStrings = "~1.3.0"
PlotThemes = "~3.1.0"
Plots = "~1.38.17"
PlutoUI = "~0.7.52"
Roots = "~2.0.19"
SpecialFunctions = "~2.3.1"
Tensors = "~1.15.0"
"""

# ╔═╡ 00000000-0000-0000-0000-000000000002
PLUTO_MANIFEST_TOML_CONTENTS = """
# This file is machine-generated - editing it directly is not advised

julia_version = "1.9.3"
manifest_format = "2.0"
project_hash = "32c1c09ea1648c0688243283537dbf646f8ce074"

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
git-tree-sha1 = "91bd53c39b9cbfb5ef4b015e8b582d344532bd0a"
uuid = "6e696c72-6542-2067-7265-42206c756150"
version = "1.2.0"

[[deps.Adapt]]
deps = ["LinearAlgebra", "Requires"]
git-tree-sha1 = "76289dc51920fdc6e0013c872ba9551d54961c24"
uuid = "79e6a3ab-5dfb-504d-930d-738a2a938a0e"
version = "3.6.2"
weakdeps = ["StaticArrays"]

    [deps.Adapt.extensions]
    AdaptStaticArraysExt = "StaticArrays"

[[deps.ArgTools]]
uuid = "0dad84c5-d112-42e6-8d28-ef12dabb789f"
version = "1.1.1"

[[deps.ArnoldiMethod]]
deps = ["LinearAlgebra", "Random", "StaticArrays"]
git-tree-sha1 = "62e51b39331de8911e4a7ff6f5aaf38a5f4cc0ae"
uuid = "ec485272-7323-5ecc-a04f-4719b315124d"
version = "0.2.0"

[[deps.ArrayInterface]]
deps = ["Adapt", "LinearAlgebra", "Requires", "SparseArrays", "SuiteSparse"]
git-tree-sha1 = "f83ec24f76d4c8f525099b2ac475fc098138ec31"
uuid = "4fba245c-0d91-5ea0-9b3e-6abc04ee57a9"
version = "7.4.11"

    [deps.ArrayInterface.extensions]
    ArrayInterfaceBandedMatricesExt = "BandedMatrices"
    ArrayInterfaceBlockBandedMatricesExt = "BlockBandedMatrices"
    ArrayInterfaceCUDAExt = "CUDA"
    ArrayInterfaceGPUArraysCoreExt = "GPUArraysCore"
    ArrayInterfaceStaticArraysCoreExt = "StaticArraysCore"
    ArrayInterfaceTrackerExt = "Tracker"

    [deps.ArrayInterface.weakdeps]
    BandedMatrices = "aae01518-5342-5314-be14-df237901396f"
    BlockBandedMatrices = "ffab5731-97b5-5995-9138-79e8c1846df0"
    CUDA = "052768ef-5323-5732-b1bb-66c8b64840ba"
    GPUArraysCore = "46192b85-c4d5-4398-a991-12ede77f4527"
    StaticArraysCore = "1e83bf80-4336-4d27-bf5d-d5a4f845583c"
    Tracker = "9f7883ad-71c0-57eb-9f7f-b5c9e6d3789c"

[[deps.ArrayInterfaceCore]]
deps = ["LinearAlgebra", "SnoopPrecompile", "SparseArrays", "SuiteSparse"]
git-tree-sha1 = "e5f08b5689b1aad068e01751889f2f615c7db36d"
uuid = "30b0a656-2188-435a-8636-2ec0e6a096e2"
version = "0.1.29"

[[deps.Artifacts]]
uuid = "56f22d72-fd6d-98f1-02f0-08ddc0907c33"

[[deps.AxisAlgorithms]]
deps = ["LinearAlgebra", "Random", "SparseArrays", "WoodburyMatrices"]
git-tree-sha1 = "66771c8d21c8ff5e3a93379480a2307ac36863f7"
uuid = "13072b0f-2c55-5437-9ae7-d433b7a33950"
version = "1.0.1"

[[deps.AxisArrays]]
deps = ["Dates", "IntervalSets", "IterTools", "RangeArrays"]
git-tree-sha1 = "16351be62963a67ac4083f748fdb3cca58bfd52f"
uuid = "39de3d68-74b9-583c-8d2d-e117c070f3a9"
version = "0.4.7"

[[deps.Base64]]
uuid = "2a0f44e3-6c83-55bd-87e4-b1978d98bd5f"

[[deps.BitFlags]]
git-tree-sha1 = "43b1a4a8f797c1cddadf60499a8a077d4af2cd2d"
uuid = "d1d4a3ce-64b1-5f1a-9ba4-7e7e69966f35"
version = "0.1.7"

[[deps.BitTwiddlingConvenienceFunctions]]
deps = ["Static"]
git-tree-sha1 = "0c5f81f47bbbcf4aea7b2959135713459170798b"
uuid = "62783981-4cbd-42fc-bca8-16325de8dc4b"
version = "0.1.5"

[[deps.Bzip2_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "19a35467a82e236ff51bc17a3a44b69ef35185a2"
uuid = "6e34b625-4abd-537c-b88f-471c36dfa7a0"
version = "1.0.8+0"

[[deps.CEnum]]
git-tree-sha1 = "eb4cb44a499229b3b8426dcfb5dd85333951ff90"
uuid = "fa961155-64e5-5f13-b03f-caf6b980ea82"
version = "0.4.2"

[[deps.CPUSummary]]
deps = ["CpuId", "IfElse", "PrecompileTools", "Static"]
git-tree-sha1 = "89e0654ed8c7aebad6d5ad235d6242c2d737a928"
uuid = "2a0fbf3d-bb9c-48f3-b0a9-814d99fd7ab9"
version = "0.2.3"

[[deps.Cairo_jll]]
deps = ["Artifacts", "Bzip2_jll", "CompilerSupportLibraries_jll", "Fontconfig_jll", "FreeType2_jll", "Glib_jll", "JLLWrappers", "LZO_jll", "Libdl", "Pixman_jll", "Pkg", "Xorg_libXext_jll", "Xorg_libXrender_jll", "Zlib_jll", "libpng_jll"]
git-tree-sha1 = "4b859a208b2397a7a623a03449e4636bdb17bcf2"
uuid = "83423d85-b0ee-5818-9007-b63ccbeb887a"
version = "1.16.1+1"

[[deps.CalculusWithJulia]]
deps = ["Base64", "Contour", "ForwardDiff", "HCubature", "IntervalSets", "JSON", "LinearAlgebra", "PlotUtils", "Random", "RecipesBase", "Reexport", "Requires", "Roots", "SpecialFunctions", "SplitApplyCombine", "Test"]
git-tree-sha1 = "049194aa15becc95f65f2cf38ec0a221e486d1c3"
uuid = "a2e0e22d-7d4c-5312-9169-8b992201a882"
version = "0.1.2"

[[deps.CatIndices]]
deps = ["CustomUnitRanges", "OffsetArrays"]
git-tree-sha1 = "a0f80a09780eed9b1d106a1bf62041c2efc995bc"
uuid = "aafaddc9-749c-510e-ac4f-586e18779b91"
version = "0.2.2"

[[deps.ChainRulesCore]]
deps = ["Compat", "LinearAlgebra", "SparseArrays"]
git-tree-sha1 = "e30f2f4e20f7f186dc36529910beaedc60cfa644"
uuid = "d360d2e6-b24c-11e9-a2a3-2a2ae2dbcce4"
version = "1.16.0"

[[deps.CloseOpenIntervals]]
deps = ["Static", "StaticArrayInterface"]
git-tree-sha1 = "70232f82ffaab9dc52585e0dd043b5e0c6b714f1"
uuid = "fb6a15b2-703c-40df-9091-08a04967cfa9"
version = "0.1.12"

[[deps.Clustering]]
deps = ["Distances", "LinearAlgebra", "NearestNeighbors", "Printf", "Random", "SparseArrays", "Statistics", "StatsBase"]
git-tree-sha1 = "b86ac2c5543660d238957dbde5ac04520ae977a7"
uuid = "aaaa29a8-35af-508c-8bc3-b662a17a0fe5"
version = "0.15.4"

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

[[deps.Combinatorics]]
git-tree-sha1 = "08c8b6831dc00bfea825826be0bc8336fc369860"
uuid = "861a8166-3701-5b0c-9a16-15d98fcdc6aa"
version = "1.0.2"

[[deps.CommonSolve]]
git-tree-sha1 = "0eee5eb66b1cf62cd6ad1b460238e60e4b09400c"
uuid = "38540f10-b2f7-11e9-35d8-d573e4eb0ff2"
version = "0.2.4"

[[deps.CommonSubexpressions]]
deps = ["MacroTools", "Test"]
git-tree-sha1 = "7b8a93dba8af7e3b42fecabf646260105ac373f7"
uuid = "bbf7d656-a473-5ed7-a52c-81e309532950"
version = "0.3.0"

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

[[deps.ComputationalResources]]
git-tree-sha1 = "52cb3ec90e8a8bea0e62e275ba577ad0f74821f7"
uuid = "ed09eef8-17a6-5b46-8889-db040fac31e3"
version = "0.3.2"

[[deps.ConcurrentUtilities]]
deps = ["Serialization", "Sockets"]
git-tree-sha1 = "5372dbbf8f0bdb8c700db5367132925c0771ef7e"
uuid = "f0e56b4a-5159-44fe-b623-3e5288b988bb"
version = "2.2.1"

[[deps.ConstructionBase]]
deps = ["LinearAlgebra"]
git-tree-sha1 = "fe2838a593b5f776e1597e086dcd47560d94e816"
uuid = "187b0558-2788-49d3-abe0-74a17ed4e7c9"
version = "1.5.3"
weakdeps = ["IntervalSets", "StaticArrays"]

    [deps.ConstructionBase.extensions]
    ConstructionBaseIntervalSetsExt = "IntervalSets"
    ConstructionBaseStaticArraysExt = "StaticArrays"

[[deps.Contour]]
git-tree-sha1 = "d05d9e7b7aedff4e5b51a029dced05cfb6125781"
uuid = "d38c429a-6771-53c6-b99e-75d170b6e991"
version = "0.6.2"

[[deps.CoordinateTransformations]]
deps = ["LinearAlgebra", "StaticArrays"]
git-tree-sha1 = "f9d7112bfff8a19a3a4ea4e03a8e6a91fe8456bf"
uuid = "150eb455-5306-5404-9cee-2592286d6298"
version = "0.6.3"

[[deps.CpuId]]
deps = ["Markdown"]
git-tree-sha1 = "fcbb72b032692610bfbdb15018ac16a36cf2e406"
uuid = "adafc99b-e345-5852-983c-f28acb93d879"
version = "0.3.1"

[[deps.CustomUnitRanges]]
git-tree-sha1 = "1a3f97f907e6dd8983b744d2642651bb162a3f7a"
uuid = "dc8bdbbb-1ca9-579f-8c36-e416f6a65cce"
version = "1.0.2"

[[deps.DataAPI]]
git-tree-sha1 = "8da84edb865b0b5b0100c0666a9bc9a0b71c553c"
uuid = "9a962f9c-6df0-11e9-0e5d-c546b8b5ee8a"
version = "1.15.0"

[[deps.DataStructures]]
deps = ["Compat", "InteractiveUtils", "OrderedCollections"]
git-tree-sha1 = "3dbd312d370723b6bb43ba9d02fc36abade4518d"
uuid = "864edb3b-99cc-5e75-8d2d-829cb0a9cfe8"
version = "0.18.15"

[[deps.Dates]]
deps = ["Printf"]
uuid = "ade2ca70-3891-5945-98fb-dc099432e06a"

[[deps.DelimitedFiles]]
deps = ["Mmap"]
git-tree-sha1 = "9e2f36d3c96a820c678f2f1f1782582fcf685bae"
uuid = "8bb1440f-4735-579b-a4ab-409b98df4dab"
version = "1.9.1"

[[deps.Dictionaries]]
deps = ["Indexing", "Random", "Serialization"]
git-tree-sha1 = "e82c3c97b5b4ec111f3c1b55228cebc7510525a2"
uuid = "85a47980-9c8c-11e8-2b9f-f7ca1fa99fb4"
version = "0.3.25"

[[deps.DiffResults]]
deps = ["StaticArraysCore"]
git-tree-sha1 = "782dd5f4561f5d267313f23853baaaa4c52ea621"
uuid = "163ba53b-c6d8-5494-b064-1a9d43ac40c5"
version = "1.1.0"

[[deps.DiffRules]]
deps = ["IrrationalConstants", "LogExpFunctions", "NaNMath", "Random", "SpecialFunctions"]
git-tree-sha1 = "23163d55f885173722d1e4cf0f6110cdbaf7e272"
uuid = "b552c78f-8df3-52c6-915a-8e097449b14b"
version = "1.15.1"

[[deps.Distances]]
deps = ["LinearAlgebra", "Statistics", "StatsAPI"]
git-tree-sha1 = "b6def76ffad15143924a2199f72a5cd883a2e8a9"
uuid = "b4f34e82-e78d-54a5-968a-f98e89d6e8f7"
version = "0.10.9"
weakdeps = ["SparseArrays"]

    [deps.Distances.extensions]
    DistancesSparseArraysExt = "SparseArrays"

[[deps.Distributed]]
deps = ["Random", "Serialization", "Sockets"]
uuid = "8ba89e20-285c-5b6f-9357-94700520ee1b"

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

[[deps.FFTViews]]
deps = ["CustomUnitRanges", "FFTW"]
git-tree-sha1 = "cbdf14d1e8c7c8aacbe8b19862e0179fd08321c2"
uuid = "4f61f5a4-77b1-5117-aa51-3ab5ef4ef0cd"
version = "0.3.2"

[[deps.FFTW]]
deps = ["AbstractFFTs", "FFTW_jll", "LinearAlgebra", "MKL_jll", "Preferences", "Reexport"]
git-tree-sha1 = "b4fbdd20c889804969571cc589900803edda16b7"
uuid = "7a1cc6ca-52ef-59f5-83cd-3a7055c09341"
version = "1.7.1"

[[deps.FFTW_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "c6033cc3892d0ef5bb9cd29b7f2f0331ea5184ea"
uuid = "f5851436-0d7a-5f13-b9de-f02708fd171a"
version = "3.3.10+0"

[[deps.FastGaussQuadrature]]
deps = ["LinearAlgebra", "SpecialFunctions", "StaticArrays"]
git-tree-sha1 = "0f478d8bad6f52573fb7658a263af61f3d96e43a"
uuid = "442a2c76-b920-505d-bb47-c5924d526838"
version = "0.5.1"

[[deps.FileIO]]
deps = ["Pkg", "Requires", "UUIDs"]
git-tree-sha1 = "299dc33549f68299137e51e6d49a13b5b1da9673"
uuid = "5789e2e9-d7fb-5bc7-8068-2c6fae9b9549"
version = "1.16.1"

[[deps.FileWatching]]
uuid = "7b1f6079-737a-58dc-b8bc-7a2ca5c1b5ee"

[[deps.FiniteDifferences]]
deps = ["ChainRulesCore", "LinearAlgebra", "Printf", "Random", "Richardson", "SparseArrays", "StaticArrays"]
git-tree-sha1 = "98a4571ba51bf172c798e7cde9b16c11c1e22e6d"
uuid = "26cc04aa-876d-5657-8c51-4c34ba976000"
version = "0.12.30"

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

[[deps.ForwardDiff]]
deps = ["CommonSubexpressions", "DiffResults", "DiffRules", "LinearAlgebra", "LogExpFunctions", "NaNMath", "Preferences", "Printf", "Random", "SpecialFunctions"]
git-tree-sha1 = "cf0fe81336da9fb90944683b8c41984b08793dad"
uuid = "f6369f11-7733-5829-9624-2563aa707210"
version = "0.10.36"
weakdeps = ["StaticArrays"]

    [deps.ForwardDiff.extensions]
    ForwardDiffStaticArraysExt = "StaticArrays"

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
git-tree-sha1 = "8e2d86e06ceb4580110d9e716be26658effc5bfd"
uuid = "28b8d3ca-fb5f-59d9-8090-bfdbd6d07a71"
version = "0.72.8"

[[deps.GR_jll]]
deps = ["Artifacts", "Bzip2_jll", "Cairo_jll", "FFMPEG_jll", "Fontconfig_jll", "GLFW_jll", "JLLWrappers", "JpegTurbo_jll", "Libdl", "Libtiff_jll", "Pixman_jll", "Qt5Base_jll", "Zlib_jll", "libpng_jll"]
git-tree-sha1 = "da121cbdc95b065da07fbb93638367737969693f"
uuid = "d2c73de3-f751-5644-a686-071e5b155ba9"
version = "0.72.8+0"

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

[[deps.Graphics]]
deps = ["Colors", "LinearAlgebra", "NaNMath"]
git-tree-sha1 = "d61890399bc535850c4bf08e4e0d3a7ad0f21cbd"
uuid = "a2bd30eb-e257-5431-a919-1863eab51364"
version = "1.1.2"

[[deps.Graphite2_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "344bf40dcab1073aca04aa0df4fb092f920e4011"
uuid = "3b182d85-2403-5c21-9c21-1e1f0cc25472"
version = "1.3.14+0"

[[deps.Graphs]]
deps = ["ArnoldiMethod", "Compat", "DataStructures", "Distributed", "Inflate", "LinearAlgebra", "Random", "SharedArrays", "SimpleTraits", "SparseArrays", "Statistics"]
git-tree-sha1 = "1cf1d7dcb4bc32d7b4a5add4232db3750c27ecb4"
uuid = "86223c79-3864-5bf0-83f7-82e725a168b6"
version = "1.8.0"

[[deps.Grisu]]
git-tree-sha1 = "53bb909d1151e57e2484c3d1b53e19552b887fb2"
uuid = "42e2da0e-8278-4e71-bc24-59509adca0fe"
version = "1.0.2"

[[deps.HCubature]]
deps = ["Combinatorics", "DataStructures", "LinearAlgebra", "QuadGK", "StaticArrays"]
git-tree-sha1 = "e95b36755023def6ebc3d269e6483efa8b2f7f65"
uuid = "19dc6840-f33b-545b-b366-655c7e3ffd49"
version = "1.5.1"

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

[[deps.HistogramThresholding]]
deps = ["ImageBase", "LinearAlgebra", "MappedArrays"]
git-tree-sha1 = "7194dfbb2f8d945abdaf68fa9480a965d6661e69"
uuid = "2c695a8d-9458-5d45-9878-1b8a99cf7853"
version = "0.3.1"

[[deps.HostCPUFeatures]]
deps = ["BitTwiddlingConvenienceFunctions", "IfElse", "Libdl", "Static"]
git-tree-sha1 = "eb8fed28f4994600e29beef49744639d985a04b2"
uuid = "3e5b6fbb-0976-4d2c-9146-d79de83f2fb0"
version = "0.1.16"

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

[[deps.IfElse]]
git-tree-sha1 = "debdd00ffef04665ccbb3e150747a77560e8fad1"
uuid = "615f187c-cbe4-4ef1-ba3b-2fcf58d6d173"
version = "0.1.1"

[[deps.ImageAxes]]
deps = ["AxisArrays", "ImageBase", "ImageCore", "Reexport", "SimpleTraits"]
git-tree-sha1 = "2e4520d67b0cef90865b3ef727594d2a58e0e1f8"
uuid = "2803e5a7-5153-5ecf-9a86-9b4c37f5f5ac"
version = "0.6.11"

[[deps.ImageBase]]
deps = ["ImageCore", "Reexport"]
git-tree-sha1 = "eb49b82c172811fd2c86759fa0553a2221feb909"
uuid = "c817782e-172a-44cc-b673-b171935fbb9e"
version = "0.1.7"

[[deps.ImageBinarization]]
deps = ["HistogramThresholding", "ImageCore", "LinearAlgebra", "Polynomials", "Reexport", "Statistics"]
git-tree-sha1 = "f5356e7203c4a9954962e3757c08033f2efe578a"
uuid = "cbc4b850-ae4b-5111-9e64-df94c024a13d"
version = "0.3.0"

[[deps.ImageContrastAdjustment]]
deps = ["ImageBase", "ImageCore", "ImageTransformations", "Parameters"]
git-tree-sha1 = "eb3d4365a10e3f3ecb3b115e9d12db131d28a386"
uuid = "f332f351-ec65-5f6a-b3d1-319c6670881a"
version = "0.3.12"

[[deps.ImageCore]]
deps = ["AbstractFFTs", "ColorVectorSpace", "Colors", "FixedPointNumbers", "MappedArrays", "MosaicViews", "OffsetArrays", "PaddedViews", "PrecompileTools", "Reexport"]
git-tree-sha1 = "fc5d1d3443a124fde6e92d0260cd9e064eba69f8"
uuid = "a09fc81d-aa75-5fe9-8630-4744c3626534"
version = "0.10.1"

[[deps.ImageCorners]]
deps = ["ImageCore", "ImageFiltering", "PrecompileTools", "StaticArrays", "StatsBase"]
git-tree-sha1 = "24c52de051293745a9bad7d73497708954562b79"
uuid = "89d5987c-236e-4e32-acd0-25bd6bd87b70"
version = "0.1.3"

[[deps.ImageDistances]]
deps = ["Distances", "ImageCore", "ImageMorphology", "LinearAlgebra", "Statistics"]
git-tree-sha1 = "08b0e6354b21ef5dd5e49026028e41831401aca8"
uuid = "51556ac3-7006-55f5-8cb3-34580c88182d"
version = "0.2.17"

[[deps.ImageFiltering]]
deps = ["CatIndices", "ComputationalResources", "DataStructures", "FFTViews", "FFTW", "ImageBase", "ImageCore", "LinearAlgebra", "OffsetArrays", "PrecompileTools", "Reexport", "SparseArrays", "StaticArrays", "Statistics", "TiledIteration"]
git-tree-sha1 = "432ae2b430a18c58eb7eca9ef8d0f2db90bc749c"
uuid = "6a3955dd-da59-5b1f-98d4-e7296123deb5"
version = "0.7.8"

[[deps.ImageIO]]
deps = ["FileIO", "IndirectArrays", "JpegTurbo", "LazyModules", "Netpbm", "OpenEXR", "PNGFiles", "QOI", "Sixel", "TiffImages", "UUIDs"]
git-tree-sha1 = "bca20b2f5d00c4fbc192c3212da8fa79f4688009"
uuid = "82e4d734-157c-48bb-816b-45c225c6df19"
version = "0.6.7"

[[deps.ImageMagick]]
deps = ["FileIO", "ImageCore", "ImageMagick_jll", "InteractiveUtils"]
git-tree-sha1 = "b0b765ff0b4c3ee20ce6740d843be8dfce48487c"
uuid = "6218d12a-5da1-5696-b52f-db25d2ecc6d1"
version = "1.3.0"

[[deps.ImageMagick_jll]]
deps = ["JpegTurbo_jll", "Libdl", "Libtiff_jll", "Pkg", "Zlib_jll", "libpng_jll"]
git-tree-sha1 = "1c0a2295cca535fabaf2029062912591e9b61987"
uuid = "c73af94c-d91f-53ed-93a7-00f77d67a9d7"
version = "6.9.10-12+3"

[[deps.ImageMetadata]]
deps = ["AxisArrays", "ImageAxes", "ImageBase", "ImageCore"]
git-tree-sha1 = "355e2b974f2e3212a75dfb60519de21361ad3cb7"
uuid = "bc367c6b-8a6b-528e-b4bd-a4b897500b49"
version = "0.9.9"

[[deps.ImageMorphology]]
deps = ["DataStructures", "ImageCore", "LinearAlgebra", "LoopVectorization", "OffsetArrays", "Requires", "TiledIteration"]
git-tree-sha1 = "6f0a801136cb9c229aebea0df296cdcd471dbcd1"
uuid = "787d08f9-d448-5407-9aad-5290dd7ab264"
version = "0.4.5"

[[deps.ImageQualityIndexes]]
deps = ["ImageContrastAdjustment", "ImageCore", "ImageDistances", "ImageFiltering", "LazyModules", "OffsetArrays", "PrecompileTools", "Statistics"]
git-tree-sha1 = "783b70725ed326340adf225be4889906c96b8fd1"
uuid = "2996bd0c-7a13-11e9-2da2-2f5ce47296a9"
version = "0.3.7"

[[deps.ImageSegmentation]]
deps = ["Clustering", "DataStructures", "Distances", "Graphs", "ImageCore", "ImageFiltering", "ImageMorphology", "LinearAlgebra", "MetaGraphs", "RegionTrees", "SimpleWeightedGraphs", "StaticArrays", "Statistics"]
git-tree-sha1 = "3ff0ca203501c3eedde3c6fa7fd76b703c336b5f"
uuid = "80713f31-8817-5129-9cf8-209ff8fb23e1"
version = "1.8.2"

[[deps.ImageShow]]
deps = ["Base64", "ColorSchemes", "FileIO", "ImageBase", "ImageCore", "OffsetArrays", "StackViews"]
git-tree-sha1 = "3b5344bcdbdc11ad58f3b1956709b5b9345355de"
uuid = "4e3cecfd-b093-5904-9786-8bbb286a6a31"
version = "0.3.8"

[[deps.ImageTransformations]]
deps = ["AxisAlgorithms", "CoordinateTransformations", "ImageBase", "ImageCore", "Interpolations", "OffsetArrays", "Rotations", "StaticArrays"]
git-tree-sha1 = "7ec124670cbce8f9f0267ba703396960337e54b5"
uuid = "02fcd773-0e25-5acc-982a-7f6622650795"
version = "0.10.0"

[[deps.Images]]
deps = ["Base64", "FileIO", "Graphics", "ImageAxes", "ImageBase", "ImageBinarization", "ImageContrastAdjustment", "ImageCore", "ImageCorners", "ImageDistances", "ImageFiltering", "ImageIO", "ImageMagick", "ImageMetadata", "ImageMorphology", "ImageQualityIndexes", "ImageSegmentation", "ImageShow", "ImageTransformations", "IndirectArrays", "IntegralArrays", "Random", "Reexport", "SparseArrays", "StaticArrays", "Statistics", "StatsBase", "TiledIteration"]
git-tree-sha1 = "d438268ed7a665f8322572be0dabda83634d5f45"
uuid = "916415d5-f1e6-5110-898d-aaa5f9f070e0"
version = "0.26.0"

[[deps.Imath_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "3d09a9f60edf77f8a4d99f9e015e8fbf9989605d"
uuid = "905a6f67-0a94-5f89-b386-d35d92009cd1"
version = "3.1.7+0"

[[deps.Indexing]]
git-tree-sha1 = "ce1566720fd6b19ff3411404d4b977acd4814f9f"
uuid = "313cdc1a-70c2-5d6a-ae34-0150d3930a38"
version = "1.1.1"

[[deps.IndirectArrays]]
git-tree-sha1 = "012e604e1c7458645cb8b436f8fba789a51b257f"
uuid = "9b13fd28-a010-5f03-acff-a1bbcff69959"
version = "1.0.0"

[[deps.Inflate]]
git-tree-sha1 = "5cd07aab533df5170988219191dfad0519391428"
uuid = "d25df0c9-e2be-5dd7-82c8-3ad0b3e990b9"
version = "0.1.3"

[[deps.IntegralArrays]]
deps = ["ColorTypes", "FixedPointNumbers", "IntervalSets"]
git-tree-sha1 = "be8e690c3973443bec584db3346ddc904d4884eb"
uuid = "1d092043-8f09-5a30-832f-7509e371ab51"
version = "0.1.5"

[[deps.IntelOpenMP_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "ad37c091f7d7daf900963171600d7c1c5c3ede32"
uuid = "1d5cc7b8-4909-519e-a0f8-d0f5ad9712d0"
version = "2023.2.0+0"

[[deps.InteractiveUtils]]
deps = ["Markdown"]
uuid = "b77e0a4c-d291-57a0-90e8-8db25a27a240"

[[deps.Interpolations]]
deps = ["Adapt", "AxisAlgorithms", "ChainRulesCore", "LinearAlgebra", "OffsetArrays", "Random", "Ratios", "Requires", "SharedArrays", "SparseArrays", "StaticArrays", "WoodburyMatrices"]
git-tree-sha1 = "721ec2cf720536ad005cb38f50dbba7b02419a15"
uuid = "a98d9a8b-a2ab-59e6-89dd-64a1c18fca59"
version = "0.14.7"

[[deps.IntervalSets]]
deps = ["Dates", "Random"]
git-tree-sha1 = "8e59ea773deee525c99a8018409f64f19fb719e6"
uuid = "8197267c-284f-5f27-9208-e0e47529a953"
version = "0.7.7"
weakdeps = ["Statistics"]

    [deps.IntervalSets.extensions]
    IntervalSetsStatisticsExt = "Statistics"

[[deps.IrrationalConstants]]
git-tree-sha1 = "630b497eafcc20001bba38a4651b327dcfc491d2"
uuid = "92d709cd-6900-40b7-9082-c6be49f344b6"
version = "0.2.2"

[[deps.IterTools]]
git-tree-sha1 = "4ced6667f9974fc5c5943fa5e2ef1ca43ea9e450"
uuid = "c8e1da08-722c-5040-9ed9-7db0dc04731e"
version = "1.8.0"

[[deps.JLD2]]
deps = ["FileIO", "MacroTools", "Mmap", "OrderedCollections", "Pkg", "Printf", "Reexport", "Requires", "TranscodingStreams", "UUIDs"]
git-tree-sha1 = "aa6ffef1fd85657f4999030c52eaeec22a279738"
uuid = "033835bb-8acc-5ee8-8aae-3f567f8a3819"
version = "0.4.33"

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

[[deps.JpegTurbo]]
deps = ["CEnum", "FileIO", "ImageCore", "JpegTurbo_jll", "TOML"]
git-tree-sha1 = "327713faef2a3e5c80f96bf38d1fa26f7a6ae29e"
uuid = "b835a17e-a41a-41e7-81f0-2f016b05efe0"
version = "0.1.3"

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

[[deps.LayoutPointers]]
deps = ["ArrayInterface", "LinearAlgebra", "ManualMemory", "SIMDTypes", "Static", "StaticArrayInterface"]
git-tree-sha1 = "88b8f66b604da079a627b6fb2860d3704a6729a1"
uuid = "10f19ff3-798f-405d-979b-55457f8fc047"
version = "0.1.14"

[[deps.LazyArtifacts]]
deps = ["Artifacts", "Pkg"]
uuid = "4af54fe1-eca0-43a8-85a7-787d91b784e3"

[[deps.LazyModules]]
git-tree-sha1 = "a560dd966b386ac9ae60bdd3a3d3a326062d3c3e"
uuid = "8cdb02fc-e678-4876-92c5-9defec4f444e"
version = "0.3.1"

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
deps = ["Artifacts", "JLLWrappers", "JpegTurbo_jll", "LERC_jll", "Libdl", "Pkg", "Zlib_jll", "Zstd_jll"]
git-tree-sha1 = "3eb79b0ca5764d4799c06699573fd8f533259713"
uuid = "89763e89-9b03-5906-acba-b20f662cd828"
version = "4.4.0+0"

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

[[deps.LoopVectorization]]
deps = ["ArrayInterface", "ArrayInterfaceCore", "CPUSummary", "CloseOpenIntervals", "DocStringExtensions", "HostCPUFeatures", "IfElse", "LayoutPointers", "LinearAlgebra", "OffsetArrays", "PolyesterWeave", "PrecompileTools", "SIMDTypes", "SLEEFPirates", "Static", "StaticArrayInterface", "ThreadingUtilities", "UnPack", "VectorizationBase"]
git-tree-sha1 = "c88a4afe1703d731b1c4fdf4e3c7e77e3b176ea2"
uuid = "bdcacae8-1622-11e9-2a5c-532679323890"
version = "0.12.165"
weakdeps = ["ChainRulesCore", "ForwardDiff", "SpecialFunctions"]

    [deps.LoopVectorization.extensions]
    ForwardDiffExt = ["ChainRulesCore", "ForwardDiff"]
    SpecialFunctionsExt = "SpecialFunctions"

[[deps.MIMEs]]
git-tree-sha1 = "65f28ad4b594aebe22157d6fac869786a255b7eb"
uuid = "6c6e2e6c-3030-632d-7369-2d6c69616d65"
version = "0.1.4"

[[deps.MKL_jll]]
deps = ["Artifacts", "IntelOpenMP_jll", "JLLWrappers", "LazyArtifacts", "Libdl", "Pkg"]
git-tree-sha1 = "eb006abbd7041c28e0d16260e50a24f8f9104913"
uuid = "856f044c-d86e-5d09-b602-aeab76dc8ba7"
version = "2023.2.0+0"

[[deps.MacroTools]]
deps = ["Markdown", "Random"]
git-tree-sha1 = "9ee1618cbf5240e6d4e0371d6f24065083f60c48"
uuid = "1914dd2f-81c6-5fcd-8719-6d5c9610ff09"
version = "0.5.11"

[[deps.ManualMemory]]
git-tree-sha1 = "bcaef4fc7a0cfe2cba636d84cda54b5e4e4ca3cd"
uuid = "d125e4d3-2237-4719-b19c-fa641b8a4667"
version = "0.1.8"

[[deps.MappedArrays]]
git-tree-sha1 = "2dab0221fe2b0f2cb6754eaa743cc266339f527e"
uuid = "dbb5928d-eab1-5f90-85c2-b9b0edb7c900"
version = "0.4.2"

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

[[deps.MetaGraphs]]
deps = ["Graphs", "JLD2", "Random"]
git-tree-sha1 = "1130dbe1d5276cb656f6e1094ce97466ed700e5a"
uuid = "626554b9-1ddb-594c-aa3c-2596fe9399a5"
version = "0.7.2"

[[deps.Missings]]
deps = ["DataAPI"]
git-tree-sha1 = "f66bdc5de519e8f8ae43bdc598782d35a25b1272"
uuid = "e1d29d7a-bbdc-5cf2-9ac0-f12de2c33e28"
version = "1.1.0"

[[deps.Mmap]]
uuid = "a63ad114-7e13-5084-954f-fe012c677804"

[[deps.MosaicViews]]
deps = ["MappedArrays", "OffsetArrays", "PaddedViews", "StackViews"]
git-tree-sha1 = "7b86a5d4d70a9f5cdf2dacb3cbe6d251d1a61dbe"
uuid = "e94cdb99-869f-56ef-bcf0-1ae2bcbe0389"
version = "0.3.4"

[[deps.MozillaCACerts_jll]]
uuid = "14a3606d-f60d-562e-9121-12d972cd8159"
version = "2022.10.11"

[[deps.NaNMath]]
deps = ["OpenLibm_jll"]
git-tree-sha1 = "0877504529a3e5c3343c6f8b4c0381e57e4387e4"
uuid = "77ba4419-2d1f-58cd-9bb1-8ffee604a2e3"
version = "1.0.2"

[[deps.NearestNeighbors]]
deps = ["Distances", "StaticArrays"]
git-tree-sha1 = "2c3726ceb3388917602169bed973dbc97f1b51a8"
uuid = "b8a86587-4115-5ab1-83bc-aa920d37bbce"
version = "0.4.13"

[[deps.Netpbm]]
deps = ["FileIO", "ImageCore", "ImageMetadata"]
git-tree-sha1 = "d92b107dbb887293622df7697a2223f9f8176fcd"
uuid = "f09324ee-3d7c-5217-9330-fc30815ba969"
version = "1.1.1"

[[deps.NetworkOptions]]
uuid = "ca575930-c2e3-43a9-ace4-1e988b2c1908"
version = "1.2.0"

[[deps.OffsetArrays]]
deps = ["Adapt"]
git-tree-sha1 = "2ac17d29c523ce1cd38e27785a7d23024853a4bb"
uuid = "6fe1bfb0-de20-5000-8ca7-80f57d26f881"
version = "1.12.10"

[[deps.Ogg_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "887579a3eb005446d514ab7aeac5d1d027658b8f"
uuid = "e7412a2a-1a6e-54c0-be00-318e2571c051"
version = "1.3.5+1"

[[deps.OpenBLAS_jll]]
deps = ["Artifacts", "CompilerSupportLibraries_jll", "Libdl"]
uuid = "4536629a-c528-5b80-bd46-f80d51c5b363"
version = "0.3.21+4"

[[deps.OpenEXR]]
deps = ["Colors", "FileIO", "OpenEXR_jll"]
git-tree-sha1 = "327f53360fdb54df7ecd01e96ef1983536d1e633"
uuid = "52e1d378-f018-4a11-a4be-720524705ac7"
version = "0.3.2"

[[deps.OpenEXR_jll]]
deps = ["Artifacts", "Imath_jll", "JLLWrappers", "Libdl", "Zlib_jll"]
git-tree-sha1 = "a4ca623df1ae99d09bc9868b008262d0c0ac1e4f"
uuid = "18a262bb-aa17-5467-a713-aee519bc75cb"
version = "3.1.4+0"

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

[[deps.PNGFiles]]
deps = ["Base64", "CEnum", "ImageCore", "IndirectArrays", "OffsetArrays", "libpng_jll"]
git-tree-sha1 = "9b02b27ac477cad98114584ff964e3052f656a0f"
uuid = "f57f5aa1-a3ce-4bc8-8ab9-96f992907883"
version = "0.4.0"

[[deps.PaddedViews]]
deps = ["OffsetArrays"]
git-tree-sha1 = "0fac6313486baae819364c52b4f483450a9d793f"
uuid = "5432bcbf-9aad-5242-b902-cca2824c8663"
version = "0.5.12"

[[deps.Parameters]]
deps = ["OrderedCollections", "UnPack"]
git-tree-sha1 = "34c0e9ad262e5f7fc75b10a9952ca7692cfc5fbe"
uuid = "d96e819e-fc66-5662-9728-84c9c7592b0a"
version = "0.12.3"

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

[[deps.PkgVersion]]
deps = ["Pkg"]
git-tree-sha1 = "f6cf8e7944e50901594838951729a1861e668cb8"
uuid = "eebad327-c553-4316-9ea0-9fa01ccd7688"
version = "0.3.2"

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

[[deps.PolyesterWeave]]
deps = ["BitTwiddlingConvenienceFunctions", "CPUSummary", "IfElse", "Static", "ThreadingUtilities"]
git-tree-sha1 = "240d7170f5ffdb285f9427b92333c3463bf65bf6"
uuid = "1d0040c9-8b98-4ee7-8388-3f51789ca0ad"
version = "0.2.1"

[[deps.Polynomials]]
deps = ["LinearAlgebra", "RecipesBase"]
git-tree-sha1 = "3aa2bb4982e575acd7583f01531f241af077b163"
uuid = "f27b6e38-b328-58d1-80ce-0feddd5e7a45"
version = "3.2.13"

    [deps.Polynomials.extensions]
    PolynomialsChainRulesCoreExt = "ChainRulesCore"
    PolynomialsMakieCoreExt = "MakieCore"
    PolynomialsMutableArithmeticsExt = "MutableArithmetics"

    [deps.Polynomials.weakdeps]
    ChainRulesCore = "d360d2e6-b24c-11e9-a2a3-2a2ae2dbcce4"
    MakieCore = "20f20a25-4f0e-4fdf-b5d1-57303727442b"
    MutableArithmetics = "d8a4904e-b15c-11e9-3269-09a3773c0cb0"

[[deps.PrecompileTools]]
deps = ["Preferences"]
git-tree-sha1 = "9673d39decc5feece56ef3940e5dafba15ba0f81"
uuid = "aea7be01-6a6a-4083-8856-8a6e6704d82a"
version = "1.1.2"

[[deps.Preferences]]
deps = ["TOML"]
git-tree-sha1 = "7eb1686b4f04b82f96ed7a4ea5890a4f0c7a09f1"
uuid = "21216c6a-2e73-6563-6e65-726566657250"
version = "1.4.0"

[[deps.Printf]]
deps = ["Unicode"]
uuid = "de0858da-6303-5e67-8744-51eddeeeb8d7"

[[deps.ProgressMeter]]
deps = ["Distributed", "Printf"]
git-tree-sha1 = "d7a7aef8f8f2d537104f170139553b14dfe39fe9"
uuid = "92933f4c-e287-5a05-a399-4b506db050ca"
version = "1.7.2"

[[deps.QOI]]
deps = ["ColorTypes", "FileIO", "FixedPointNumbers"]
git-tree-sha1 = "18e8f4d1426e965c7b532ddd260599e1510d26ce"
uuid = "4b34888f-f399-49d4-9bb3-47ed5cae4e65"
version = "1.0.0"

[[deps.Qt5Base_jll]]
deps = ["Artifacts", "CompilerSupportLibraries_jll", "Fontconfig_jll", "Glib_jll", "JLLWrappers", "Libdl", "Libglvnd_jll", "OpenSSL_jll", "Pkg", "Xorg_libXext_jll", "Xorg_libxcb_jll", "Xorg_xcb_util_image_jll", "Xorg_xcb_util_keysyms_jll", "Xorg_xcb_util_renderutil_jll", "Xorg_xcb_util_wm_jll", "Zlib_jll", "xkbcommon_jll"]
git-tree-sha1 = "0c03844e2231e12fda4d0086fd7cbe4098ee8dc5"
uuid = "ea2cea3b-5b76-57ae-a6ef-0a8af62496e1"
version = "5.15.3+2"

[[deps.QuadGK]]
deps = ["DataStructures", "LinearAlgebra"]
git-tree-sha1 = "6ec7ac8412e83d57e313393220879ede1740f9ee"
uuid = "1fd47b50-473d-5c70-9696-f719f8f3bcdc"
version = "2.8.2"

[[deps.Quaternions]]
deps = ["LinearAlgebra", "Random", "RealDot"]
git-tree-sha1 = "da095158bdc8eaccb7890f9884048555ab771019"
uuid = "94ee1d12-ae83-5a48-8b1c-48b8ff168ae0"
version = "0.7.4"

[[deps.REPL]]
deps = ["InteractiveUtils", "Markdown", "Sockets", "Unicode"]
uuid = "3fa0cd96-eef1-5676-8a61-b3b8758bbffb"

[[deps.Random]]
deps = ["SHA", "Serialization"]
uuid = "9a3f8284-a2c9-5f02-9a11-845980a1fd5c"

[[deps.RangeArrays]]
git-tree-sha1 = "b9039e93773ddcfc828f12aadf7115b4b4d225f5"
uuid = "b3c3ace0-ae52-54e7-9d0b-2c1406fd6b9d"
version = "0.3.2"

[[deps.Ratios]]
deps = ["Requires"]
git-tree-sha1 = "1342a47bf3260ee108163042310d26f2be5ec90b"
uuid = "c84ed2f1-dad5-54f0-aa8e-dbefe2724439"
version = "0.4.5"
weakdeps = ["FixedPointNumbers"]

    [deps.Ratios.extensions]
    RatiosFixedPointNumbersExt = "FixedPointNumbers"

[[deps.RealDot]]
deps = ["LinearAlgebra"]
git-tree-sha1 = "9f0a1b71baaf7650f4fa8a1d168c7fb6ee41f0c9"
uuid = "c1ae055f-0cd5-4b69-90a6-9a35b1a98df9"
version = "0.1.0"

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

[[deps.RegionTrees]]
deps = ["IterTools", "LinearAlgebra", "StaticArrays"]
git-tree-sha1 = "4618ed0da7a251c7f92e869ae1a19c74a7d2a7f9"
uuid = "dee08c22-ab7f-5625-9660-a9af2021b33f"
version = "0.3.2"

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

[[deps.Richardson]]
deps = ["LinearAlgebra"]
git-tree-sha1 = "e03ca566bec93f8a3aeb059c8ef102f268a38949"
uuid = "708f8203-808e-40c0-ba2d-98a6953ed40d"
version = "1.4.0"

[[deps.Roots]]
deps = ["ChainRulesCore", "CommonSolve", "Printf", "Setfield"]
git-tree-sha1 = "ff42754a57bb0d6dcfe302fd0d4272853190421f"
uuid = "f2b01f46-fcfa-551c-844a-d8ac1e96c665"
version = "2.0.19"

    [deps.Roots.extensions]
    RootsForwardDiffExt = "ForwardDiff"
    RootsIntervalRootFindingExt = "IntervalRootFinding"
    RootsSymPyExt = "SymPy"

    [deps.Roots.weakdeps]
    ForwardDiff = "f6369f11-7733-5829-9624-2563aa707210"
    IntervalRootFinding = "d2bf35a9-74e0-55ec-b149-d360ff49b807"
    SymPy = "24249f21-da20-56a4-8eb1-6a02cf4ae2e6"

[[deps.Rotations]]
deps = ["LinearAlgebra", "Quaternions", "Random", "StaticArrays"]
git-tree-sha1 = "54ccb4dbab4b1f69beb255a2c0ca5f65a9c82f08"
uuid = "6038ab10-8711-5258-84ad-4b1120ba62dc"
version = "1.5.1"

[[deps.SHA]]
uuid = "ea8e919c-243c-51af-8825-aaa63cd721ce"
version = "0.7.0"

[[deps.SIMD]]
deps = ["PrecompileTools"]
git-tree-sha1 = "0e270732477b9e551d884e6b07e23bb2ec947790"
uuid = "fdea26ae-647d-5447-a871-4b548cad5224"
version = "3.4.5"

[[deps.SIMDTypes]]
git-tree-sha1 = "330289636fb8107c5f32088d2741e9fd7a061a5c"
uuid = "94e857df-77ce-4151-89e5-788b33177be4"
version = "0.1.0"

[[deps.SLEEFPirates]]
deps = ["IfElse", "Static", "VectorizationBase"]
git-tree-sha1 = "4b8586aece42bee682399c4c4aee95446aa5cd19"
uuid = "476501e8-09a2-5ece-8869-fb82de89a1fa"
version = "0.6.39"

[[deps.Scratch]]
deps = ["Dates"]
git-tree-sha1 = "30449ee12237627992a99d5e30ae63e4d78cd24a"
uuid = "6c6a2e73-6563-6170-7368-637461726353"
version = "1.2.0"

[[deps.Serialization]]
uuid = "9e88b42a-f829-5b0c-bbe9-9e923198166b"

[[deps.Setfield]]
deps = ["ConstructionBase", "Future", "MacroTools", "StaticArraysCore"]
git-tree-sha1 = "e2cc6d8c88613c05e1defb55170bf5ff211fbeac"
uuid = "efcf1570-3423-57d1-acb7-fd33fddbac46"
version = "1.1.1"

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

[[deps.SimpleTraits]]
deps = ["InteractiveUtils", "MacroTools"]
git-tree-sha1 = "5d7e3f4e11935503d3ecaf7186eac40602e7d231"
uuid = "699a6c99-e7fa-54fc-8d76-47d257e15c1d"
version = "0.9.4"

[[deps.SimpleWeightedGraphs]]
deps = ["Graphs", "LinearAlgebra", "Markdown", "SparseArrays"]
git-tree-sha1 = "4b33e0e081a825dbfaf314decf58fa47e53d6acb"
uuid = "47aef6b3-ad0c-573a-a1e2-d07658019622"
version = "1.4.0"

[[deps.Sixel]]
deps = ["Dates", "FileIO", "ImageCore", "IndirectArrays", "OffsetArrays", "REPL", "libsixel_jll"]
git-tree-sha1 = "2da10356e31327c7096832eb9cd86307a50b1eb6"
uuid = "45858cf5-a6b0-47a3-bbea-62219f50df47"
version = "0.1.3"

[[deps.SnoopPrecompile]]
deps = ["Preferences"]
git-tree-sha1 = "e760a70afdcd461cf01a575947738d359234665c"
uuid = "66db9d55-30c0-4569-8b51-7e840670fc0c"
version = "1.0.3"

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
weakdeps = ["ChainRulesCore"]

    [deps.SpecialFunctions.extensions]
    SpecialFunctionsChainRulesCoreExt = "ChainRulesCore"

[[deps.SplitApplyCombine]]
deps = ["Dictionaries", "Indexing"]
git-tree-sha1 = "48f393b0231516850e39f6c756970e7ca8b77045"
uuid = "03a91e81-4c3e-53e1-a0a4-9c0c8f19dd66"
version = "1.2.2"

[[deps.StackViews]]
deps = ["OffsetArrays"]
git-tree-sha1 = "46e589465204cd0c08b4bd97385e4fa79a0c770c"
uuid = "cae243ae-269e-4f55-b966-ac2d0dc13c15"
version = "0.1.1"

[[deps.Static]]
deps = ["IfElse"]
git-tree-sha1 = "f295e0a1da4ca425659c57441bcb59abb035a4bc"
uuid = "aedffcd0-7271-4cad-89d0-dc628f76c6d3"
version = "0.8.8"

[[deps.StaticArrayInterface]]
deps = ["ArrayInterface", "Compat", "IfElse", "LinearAlgebra", "Requires", "SnoopPrecompile", "SparseArrays", "Static", "SuiteSparse"]
git-tree-sha1 = "33040351d2403b84afce74dae2e22d3f5b18edcb"
uuid = "0d7ed370-da01-4f52-bd93-41d350b8b718"
version = "1.4.0"
weakdeps = ["OffsetArrays", "StaticArrays"]

    [deps.StaticArrayInterface.extensions]
    StaticArrayInterfaceOffsetArraysExt = "OffsetArrays"
    StaticArrayInterfaceStaticArraysExt = "StaticArrays"

[[deps.StaticArrays]]
deps = ["LinearAlgebra", "Random", "StaticArraysCore"]
git-tree-sha1 = "9cabadf6e7cd2349b6cf49f1915ad2028d65e881"
uuid = "90137ffa-7385-5640-81b9-e52037218182"
version = "1.6.2"
weakdeps = ["Statistics"]

    [deps.StaticArrays.extensions]
    StaticArraysStatisticsExt = "Statistics"

[[deps.StaticArraysCore]]
git-tree-sha1 = "36b3d696ce6366023a0ea192b4cd442268995a0d"
uuid = "1e83bf80-4336-4d27-bf5d-d5a4f845583c"
version = "1.4.2"

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

[[deps.SuiteSparse]]
deps = ["Libdl", "LinearAlgebra", "Serialization", "SparseArrays"]
uuid = "4607b0f0-06f3-5cda-b6b1-a6196a1729e9"

[[deps.SuiteSparse_jll]]
deps = ["Artifacts", "Libdl", "Pkg", "libblastrampoline_jll"]
uuid = "bea87d4a-7f5b-5778-9afe-8cc45184846c"
version = "5.10.1+6"

[[deps.TOML]]
deps = ["Dates"]
uuid = "fa267f1f-6049-4f14-aa54-33bafae1ed76"
version = "1.0.3"

[[deps.Tar]]
deps = ["ArgTools", "SHA"]
uuid = "a4e569a6-e804-4fa4-b0f3-eef7a1d5b13e"
version = "1.10.0"

[[deps.TensorCore]]
deps = ["LinearAlgebra"]
git-tree-sha1 = "1feb45f88d133a655e001435632f019a9a1bcdb6"
uuid = "62fd8b95-f654-4bbd-a8a5-9c27f68ccd50"
version = "0.1.1"

[[deps.Tensors]]
deps = ["ForwardDiff", "LinearAlgebra", "PrecompileTools", "SIMD", "StaticArrays", "Statistics"]
git-tree-sha1 = "bcbb366323add300742c9e4a5447e584640aeff2"
uuid = "48a634ad-e948-5137-8d70-aa71f2a747f4"
version = "1.15.0"

[[deps.Test]]
deps = ["InteractiveUtils", "Logging", "Random", "Serialization"]
uuid = "8dfed614-e22c-5e08-85e1-65c5234f0b40"

[[deps.ThreadingUtilities]]
deps = ["ManualMemory"]
git-tree-sha1 = "eda08f7e9818eb53661b3deb74e3159460dfbc27"
uuid = "8290d209-cae3-49c0-8002-c8c24d57dab5"
version = "0.5.2"

[[deps.TiffImages]]
deps = ["ColorTypes", "DataStructures", "DocStringExtensions", "FileIO", "FixedPointNumbers", "IndirectArrays", "Inflate", "Mmap", "OffsetArrays", "PkgVersion", "ProgressMeter", "UUIDs"]
git-tree-sha1 = "8621f5c499a8aa4aa970b1ae381aae0ef1576966"
uuid = "731e570b-9d59-4bfa-96dc-6df516fadf69"
version = "0.6.4"

[[deps.TiledIteration]]
deps = ["OffsetArrays", "StaticArrayInterface"]
git-tree-sha1 = "1176cc31e867217b06928e2f140c90bd1bc88283"
uuid = "06e1c1a7-607b-532d-9fad-de7d9aa2abac"
version = "0.5.0"

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

[[deps.UnPack]]
git-tree-sha1 = "387c1f73762231e86e0c9c5443ce3b4a0a9a0c2b"
uuid = "3a884ed6-31ef-47d7-9d2a-63182c4928ed"
version = "1.0.2"

[[deps.Unicode]]
uuid = "4ec0a83e-493e-50e2-b9ac-8f72acf5a8f5"

[[deps.UnicodeFun]]
deps = ["REPL"]
git-tree-sha1 = "53915e50200959667e78a92a418594b428dffddf"
uuid = "1cfade01-22cf-5700-b092-accc4b62d6e1"
version = "0.4.1"

[[deps.Unitful]]
deps = ["Dates", "LinearAlgebra", "Random"]
git-tree-sha1 = "607c142139151faa591b5e80d8055a15e487095b"
uuid = "1986cc42-f94f-5a68-af5c-568840ba703d"
version = "1.16.3"

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

[[deps.VectorizationBase]]
deps = ["ArrayInterface", "CPUSummary", "HostCPUFeatures", "IfElse", "LayoutPointers", "Libdl", "LinearAlgebra", "SIMDTypes", "Static", "StaticArrayInterface"]
git-tree-sha1 = "b182207d4af54ac64cbc71797765068fdeff475d"
uuid = "3d5dd08c-fd9d-11e8-17fa-ed2836048c2f"
version = "0.21.64"

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

[[deps.WoodburyMatrices]]
deps = ["LinearAlgebra", "SparseArrays"]
git-tree-sha1 = "de67fa59e33ad156a590055375a30b23c40299d3"
uuid = "efce3f68-66dc-5838-9240-27a6d6f5f9b6"
version = "0.5.5"

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

[[deps.libsixel_jll]]
deps = ["Artifacts", "JLLWrappers", "JpegTurbo_jll", "Libdl", "Pkg", "libpng_jll"]
git-tree-sha1 = "d4f63314c8aa1e48cd22aa0c17ed76cd1ae48c3c"
uuid = "075b6546-f08a-558a-be8f-8157d0f608a5"
version = "1.10.3+0"

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
# ╟─9cb51ec8-40ff-11ee-0ddd-3335291342a5
# ╟─8cba9988-b750-448e-9b94-df412c4af5a3
# ╟─2466d645-3367-423a-815d-d86d5c2228b3
# ╟─be9dc539-b848-410b-bb37-4aa91155e306
# ╟─52f31447-9fb3-4c43-91ac-9cb27f209e04
# ╠═0348024a-027a-4ac2-9530-44803c6a6289
# ╠═1a994a91-bcb2-4df6-9862-3f6eb696eb21
# ╟─ccdb8284-755a-424c-ba86-f212d16813e9
# ╠═0daf4a2e-4dbb-4087-b4d0-796c97423356
# ╠═a8e2053a-4557-490e-9a4e-babc6a0fc8e6
# ╟─737aadb2-41d2-4ca4-b096-349b17d67666
# ╠═e41e96c0-2c3b-497d-8a39-2aaaccc17423
# ╟─cb12b4de-5957-499b-b45e-cef684675903
# ╠═9b43916d-3b24-4fe7-8eea-b044aecf0681
# ╠═b9de777d-aae7-45fa-bc3f-34899b911fa9
# ╠═7878efe3-04fd-4632-bca9-045565e82824
# ╠═c49a2915-ca5a-4405-8625-3167dbc767a6
# ╠═5eca5911-287a-4252-9393-d8469acb34af
# ╠═08aa5e12-5e0d-44bf-827d-eafb2ad5a238
# ╠═e0f30602-8498-4592-b461-b17c12c9e6b7
# ╠═a8999060-be95-40ce-9d86-7aa48c97d317
# ╠═446c2b03-5204-4380-b250-c1bcb3f6dba5
# ╠═a2174988-7bb9-4ec9-a586-b38bf0d4c821
# ╠═9f2348dc-726f-4320-9611-f94c736ba0fb
# ╠═c4ad6ae2-065f-4438-a984-865e6440d43d
# ╟─e40b137b-a15e-433a-9a3a-5838458b45af
# ╟─031753c1-7fcf-4e74-bf37-66e241248bbb
# ╠═74a35dd7-bd9a-4d02-bcd7-85e58b3ff4bf
# ╠═7e34a490-1a94-4bc1-b657-61bc197282b3
# ╟─f26144bc-3dac-4218-a76a-23a2d7baeb69
# ╟─646917d4-b5b0-4e0d-8a68-84ba3971e28e
# ╠═ad7f995e-bc2d-4192-ba3c-4a2665af73fd
# ╠═d1024fa4-5a13-4ce7-8efd-b0321e675cd9
# ╠═4e203179-00f9-4de0-bcd3-e4c4e1e08b2f
# ╟─d6b4f284-1638-45db-ad25-5749aefd550f
# ╠═5badd08a-bb90-4401-8dd9-3cefc7f45159
# ╠═3d38088d-9ed7-4497-9132-2ada083c20b1
# ╠═bd349f83-58a1-4965-9750-18cec4ab2542
# ╠═fa153793-fcc2-43e5-bebb-96ccaa1094d2
# ╠═811c3176-2dc2-4d7d-87d7-320d0e0ce4b9
# ╟─0217effe-642e-4408-b14d-848ca12b6ec0
# ╟─bf0ab7c8-5826-42a0-ba41-6f4879e7a080
# ╟─e41fd78b-5385-43a4-aded-42663016ebc2
# ╠═a2e1d98e-a755-42e8-9a16-13e2bed00bdc
# ╠═4a5ece52-bf23-4e22-9819-7fa507e17517
# ╠═891b5c2d-bfe3-4eb3-ad9f-508b48dd4f98
# ╟─24727b89-5b90-4a93-868b-f11901bb2c87
# ╟─e3bf87cd-0574-4804-80cf-34951a469044
# ╟─8d69e12e-eedb-4dd8-b2c7-61011915981e
# ╟─1964d79e-cfa8-4b38-bd02-ba5fda5a64bc
# ╠═0977c4d2-54b9-48ef-8cc7-6c7827c52cb3
# ╠═5e36c68f-6dcb-4ce4-8084-401153d201f2
# ╟─ecc22fa1-d7b2-4b56-913e-4707f0b4d68e
# ╠═af98df01-1a8d-4059-9a3e-71196b3e4ad5
# ╟─cd140c37-4f2e-421a-b643-82f25481c3f6
# ╠═d7f6c238-3fe2-4a18-aff5-11c6324edd67
# ╠═342e5e8f-455b-44de-9f67-9ee27717cf3c
# ╠═f88f2cca-4e26-4d53-9617-993786ed0353
# ╟─948f625e-8e47-46ca-be5e-beddd92b3237
# ╠═cd19c076-9943-4223-9ef5-3a72f43c3540
# ╠═17cb4739-33fc-41bb-86c8-2c063bb65cad
# ╠═29d53bd2-6517-41f2-9d2e-679a8c0f8f90
# ╠═83801d66-89da-4161-b4b8-1962a97ac6a8
# ╠═0331ab7a-5529-4320-89ec-c448dbe54194
# ╠═9ba04cc5-b33e-4478-b0eb-8f0d8ab93ce2
# ╠═4be5acc1-b66e-41a2-8818-6411c1f67054
# ╟─1308e420-e393-4c24-a508-106f346c8ad6
# ╠═911b5ad6-5991-4472-8229-7a091dfc8a80
# ╠═aaccb6c0-c3c5-4e2e-b529-58aeef9bd8e7
# ╠═6fd6892a-d7ef-435d-89d7-3490d0aa392a
# ╠═f6a62dc5-5449-4e63-8344-afd8d97275f0
# ╟─0d01c8fe-de93-43b3-bc3a-fd1f3d8426c4
# ╠═1ba9f7d8-0777-4991-a4be-aa1faabc59b0
# ╠═98717010-786e-47a6-b930-bdc424fab88f
# ╟─6d262382-efd0-42f2-b2c4-bb0cb4699124
# ╠═9a9ffa95-3a1a-4bf3-a915-ea72406817a7
# ╠═da83cb32-0ea0-4a3b-a4d0-6d358d17dbc2
# ╠═dd976bda-0281-4196-a1ce-6eff9e7d3717
# ╠═de1c4baf-2318-4a68-a905-6c769bc66a7f
# ╟─40c1a126-178b-4fad-84f9-58846a599043
# ╠═5e63c04e-887c-409a-8336-0fbbae1032d0
# ╟─b8d79025-2b53-4710-8727-58cf14ebcdc3
# ╠═41291c12-a2fa-4897-a181-21709e47ddb0
# ╟─28de043e-2255-4318-be1b-e011f599a09f
# ╟─3b0fb866-9547-4b0d-b3df-d6d6ae60baff
# ╠═87dfec70-511c-431e-a1d1-8ee752c812c1
# ╟─32fd4867-e125-4ac1-8bb4-604bdc56ad11
# ╟─84f964e3-f1b3-48e1-8510-48c2149b1480
# ╟─b2b58488-3ddd-4275-bf37-1923333e074e
# ╟─d688d12d-661c-4ac0-8651-ce26dc856518
# ╠═46ea45d8-90f8-42a6-9fb4-49fc46c2aef6
# ╠═0ce54fa8-fda6-48f1-804b-1fd6100bc490
# ╠═56508bf0-2b27-4db7-871f-0522c3addbd7
# ╟─cd6e3483-14c8-42d5-b8be-c6abe6b86f74
# ╟─90698737-5c04-41b1-bf81-a9e638fc407f
# ╠═55d464e4-d04c-464e-9b3b-aa6f0b0021dc
# ╠═6fff5a73-33b4-4e77-828d-60995bd0cd7c
# ╟─97e14b3a-a056-40d4-ba55-abfeb9835dd7
# ╠═60ab4bde-a284-4588-849a-f78deb8bc01f
# ╟─2f4fdb5f-4cb3-46e3-ad36-46024e4ca686
# ╠═cb9bfdbe-b338-42d3-a194-fc13b42346f1
# ╠═a13b9789-fa4b-45bb-a083-81d3df4d4c86
# ╠═762b2923-09f9-4a1d-aba6-aa0fedfb75b9
# ╟─93e4bce7-6b75-410b-bf55-6d3cafbd7cc7
# ╟─040b970b-b5cc-4798-b59c-3a4dab6752d6
# ╠═5dde81b8-0f0c-4f95-a9ed-53cc45cf25f9
# ╟─ee144366-3888-4511-859d-73894596c48c
# ╠═8cc764ce-147d-45d7-ae0e-19aafd34bba2
# ╟─1b8fa2fd-d0a1-4c5d-8d79-c9d3d26f38b0
# ╟─7ff231b1-a054-446c-8188-5e4111d03741
# ╟─92c5d7db-00ad-4d14-ad76-39deae233357
# ╟─4ffe6f6e-8b1a-4b99-b742-689951842be5
# ╟─935ef6ad-6f41-4074-ad24-5f1ad0de8869
# ╟─62dba3f4-9649-4cca-a42a-e49c95adfa3d
# ╟─9e1cacf6-8bcc-4342-bb54-c8de1524ec51
# ╟─6037feef-2817-49d0-82c4-97a41b2422b2
# ╟─dbd55643-6cb1-4e29-9520-ff05099976ef
# ╟─6bc47359-a215-48da-acf8-79700c8c75ac
# ╟─45094072-406c-49e2-a8a3-2ffdcc981562
# ╟─e923fac4-f348-4e5d-a566-610d1a7649fc
# ╠═7fca8fb2-52ea-4cf6-8997-bcee8e4e3f84
# ╟─0fc62a68-07d1-45d1-8f15-b022e3625eb7
# ╟─7d5c2ca6-130d-4050-9fed-64ace363fee5
# ╟─1d5901fd-bb47-4819-8a82-4fa433040401
# ╟─00000000-0000-0000-0000-000000000001
# ╟─00000000-0000-0000-0000-000000000002
