### A Pluto.jl notebook ###
# v0.19.29

#> [frontmatter]
#> title = "Fourier Transform"

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

# ╔═╡ c6b0dc22-2f17-11ee-3378-4d9f003bef9e
begin
	using PlutoUI
	using Plots
	using Plots.PlotMeasures
	using FFTW
	using Statistics
	using Printf
	using DataFrames
	using CSV
	using HTTP
	using Dates
	using SymPy

	import PlutoUI: combine

	md"""
	# Fourier Transform

	This notebook introduces mathematical descriptions of the wave equation and, the discrete Fourier transform, and important applications of the discrete Fourier transform including amplitude spectra, phase spectra, power spectra, and frequency filtering.
	$(TableOfContents(depth=4))
	"""
end

# ╔═╡ 7fb213ec-4168-405b-9ccd-3d209412f320
md"""# Mathematical Description of Waves

## The general cosine wave equation

The wave equation defines the wave as a sine function. The wave is defined in terms of it's amplitude (``A``), frequency (``f``), and phase (``ϕ``). In the equation below the phase is in units of radians.  

```math
\begin{equation}
y_1(t) = A\cos(2πft + ϕ)
\end{equation}
```
"""

# ╔═╡ 7bdac999-3fb8-4956-8fe8-d8e3da626f8b
@bind values combine() do Child
	md"""
	A (a.u.) $(
		Child(Slider(0:0.1:10, default = 1))
	) f (Hz) $(
		Child(Slider(0.1:0.1:5, default = 0.5))
	) Φ (deg) $( 
		Child(Slider(0:1:360, default = 0))
	)
	"""
end

# ╔═╡ 65b1e136-a659-4cb0-83a1-d7d5711ee4f4
begin
	t = range(0.0, stop = 5, length = 500)
	A = values[1]
	f = 2.0.*values[2].*pi
	ϕ = values[3]./360.0.*2.0.*pi
	basewave = A.*cos.(f.*t .+ ϕ)
	plot(t, basewave, legend = :none, color = :black, 
		title = "A = $(values[1]) f = $(values[2]) Hz, ϕ = $(values[3])°", 
		xticks = 0:10, xlabel = "t (s)", ylabel = "Amplitude (a.u.)", 
		size = (800, 250), left_margin = 20px, bottom_margin = 20px)
end

# ╔═╡ 4f155487-04a8-4d3b-9085-cd8baf611934
md"""

## The complex wave equation

One can express the same wave via

```math
y_2(t) = a_n \cos(2πft) + b_n \sin(2πft)
```

where ``a_n = A\cos(ϕ)`` and ``b_n = -A\sin(ϕ)``. 

Furthermore this can be expressed as

```math
y_3(t) = X_n^* \exp^{-i 2πft}  + X_n \exp^{i 2πft} 
```

where ``X_n = \frac{1}{2} \left (a_n - i b_n \right )``, ``X_n^*`` is the conjugate of ``X_n`` and ``\exp^{ix} = \cos(x) + i\sin(x)``
"""


# ╔═╡ d782a289-abf8-416b-84ee-67f7c10eca99
begin
	an = A*cos.(ϕ)
	bn = -A*sin.(ϕ)
	wave =  an*cos.(f.*t) .+ bn*sin.(f.*t) 
	cf = 0.5*(an - im*bn)
	wave30 = real.(conj(cf) .* exp.(-im .* f .* t) + cf .* exp.(im .* f .* t))
	anr = round(an, digits = 2)
	bnr = round(bn, digits = 2)
	fr = round(f, digits = 2)
	ϕr = round(ϕ, digits = 2)
	cr = round(cf, digits = 2)
	crc = round(conj(cf), digits = 2)
	plot(t, basewave, color = :black, lw = 5, label = "$A cos($(fr)t +  $(ϕr))")
	
	plot!(t, wave,  label = "$(anr) sin($(fr)t) + $(bnr) cos($(fr)t)", 
		lw = 3, color = :darkred, ls = :dash)
	
	plot!(t, wave30, label = "($(cr))exp(-im $(fr)t}  + ($(crc))exp(im $(fr)t)", 
		color = :darkgray)
	plot!(size = (800, 200))
end

# ╔═╡ c26f651a-0417-470c-a7b4-731f2c7a5dfa
md"""## Interpretation of complex coefficients ``X_n``


"""

# ╔═╡ 1c505532-6083-4201-a6a6-ecdec9c3ce74
begin
	Θ = atan(imag(cf), real(cf)) 
	r = 2*sqrt(cf*conj(cf)) |> real
	plot([0, Θ], [0, r], proj = :polar, m = 2, gridlinewidth = 2, 
		lw = 2, color = :black, size = (200,200), label = :none)
end

# ╔═╡ 043b09fc-fbb2-46e4-8829-ee19bb4ddcad
md"""
The amplitude and the phase angle of the wave are 

``A = 2\sqrt{X_n X_n^*}`` 

``ϕ = \tan^{-1}\left ( \frac{\Im(X_n)}{\Re(X_m)} \right)``. 

Note that four-quadrant inverse tangent must be computed to get the accurate phase angle. In most languages that corresponds to the ```atan2``` function, or the [2-argument arctangent](https://en.wikipedia.org/wiki/Atan2). Note that in Julia atan should be used. The amplitude and phase angle are typically visualized by plotting the complex numbner in polar coordinates. 
"""

# ╔═╡ 29ed7682-81f5-4a90-8960-e69f1979c018
Markdown.MD(
	Markdown.Admonition("info", "Exercises", [md"
1. What is the phase angle required to express ``s(t) = A\sin(2πft)`` by applying ``y(t) = A\cos(2πft + ϕ)``? In other words, what is the phase difference between a sine wave and cosine wave? 

2. Show mathematically that ``y_1(t) = y_2(t) = y_3(t)``
"]))

# ╔═╡ 3b3ac786-d3d5-49f5-af43-fe4800bb6dac
md"""## Superposition of multiple waves

The superposition of multiple cosine waves produces more complex functions.
```math
y(t) = \sum_{i=1}^{3} A_i\cos(2πf_it + ϕ_i)
```


"""

# ╔═╡ 6ced3576-4768-4d92-8e07-8420bc77e2de
@bind v combine() do Child
	md"""
	A₁ (a.u.) $(
		Child(Slider(0:0.1:10, default = 0.3))
	) f₁ (Hz) $(
		Child(Slider(0.1:0.1:50, default = 3))
	) Φ₁ (deg) $( 
		Child(Slider(0:1:180, default = 45))
	)
	
	A₂ (a.u.) $(
		Child(Slider(0:0.1:10, default = 1))
	) f₂ (Hz) $(
		Child(Slider(0.1:0.1:50, default = 30))
	) Φ₂ (deg) $( 
		Child(Slider(0:1:180, default = 90))
	)

	A₃ (a.u.) $(
		Child(Slider(0:0.1:10, default = 0.5))
	) f₃ (Hz) $(
		Child(Slider(0.1:0.1:50, default = 10))
	) Φ₃ (deg) $( 
		Child(Slider(0:1:180, default = 1809))
	)
	"""
end

# ╔═╡ 009069c3-a918-449b-a01e-58671bc8c189
begin
	wave1 = v[1].*cos.(2.0.*v[2].*pi.*t .+ v[3]./360.0.*2.0.*pi)
	wave2 = v[4].*cos.(2.0.*v[5].*pi.*t .+ v[6]./360.0.*2.0.*pi)
	wave3 = v[7].*cos.(2.0.*v[8].*pi.*t .+ v[9]./360.0.*2.0.*pi)

	p1 = plot(t, wave1,  color = :black, label = "A = $(v[1]) f = $(v[2]) Hz, ϕ = $(v[3])°", xticks = 1:10, ylabel = "A (a.u.)")

	p2 = plot(t, wave2,  color = :black, label = "A = $(v[4]) f = $(v[5]) Hz, ϕ = $(v[6])°", xticks = 1:10, ylabel = "A (a.u.)")

	p3 = plot(t, wave3,  color = :black, label = "A = $(v[7]) f = $(v[8]) Hz, ϕ = $(v[9])°", xticks = 1:10, ylabel = "A (a.u.)")

	p4 = plot(t, wave1 + wave2 + wave3, color = :black, xlabel = "t (s)", ylabel = "A (a.u.)", label = "∑waves")
	
	plot(p1, p2, p3, p4, layout = grid(4,1), size = (800, 500), left_margin = 20px, bottom_margin = 0px)
	
end

# ╔═╡ 555b410b-4a51-4810-8be2-389a2e02589f
md"""# Discrete Fourier Transform

## Fourier series

The Fourier series states that a periodic function in the time domain can be expressed as sum of ``\cos`` and ``\sin`` functions over all frequencies 
```math
f(t) = \frac{1}{2} a_0 + \sum_{k = 1}^{\infty} \left (a_k\cos \left ( 2 \pi \frac{ k}{N} t\right ) + b_k\cos \left( 2 \pi \frac{ k}{N} t\right)\right) = \sum_{k = -\infty}^\infty X_k \exp^{i 2 \pi \frac{ k}{N} t}
```

Where ``N`` is the period and ``\frac{k}{N}`` is the frequency, Further basic mathematical details about the Fourier series can be found in the [text](https://en.wikibooks.org/wiki/Advanced_Mathematics_for_Engineers_and_Scientists/Details_and_Applications_of_Fourier_Series). The most salient points are that 
the Fourier series  will converge to ``f(t)``, except at discontinuities, if the following conditions hold:

- ``f(t) = f(t + N)``, i.e. ``f(t)`` has period ``N``.
- ``f(t)``, ``f'(t)``, and ``f''(t)`` are piecewise continuous on the interval ``-N/2 ≤ t ≤ N/2``.
- The pieces that make up ``f(t)``, ``f'(t)``, and ``f''(t)`` are continuous over closed subintervals.


## Discrete Fourier Transform
The discrete Fourier transform can be used to identify the complex wave coefficients, and hence the amplitudes (``A_k``), frequencies (``f_k``), and phases (``ϕ_k``) that comprise a complex periodic signal.

```math
X(k) = \sum_{n = 0}^{N-1} x_n \exp^{-i 2\pi \frac{k}{N} n}
```

where ``N`` is the sample size, ``x_n`` are the discrete sampled values of the time series, ``n \approx t`` approximates the time coordinate, ``\frac{k}{N}`` approximates the frequency, and ``X(k)`` are the Fourier coefficients corresponding to the ``k^{th}`` frequency bin. 

### Example 1: DFT of a single wave

- Consider a 1Hz sine wave with no phase shift: ``f(t) = A\sin(2πt)`` (black curve). 
- The wave is sampled at a sampling frequency of 8 Hz 
- There are ``N = 8`` discrete sample points (orange points).
- The ``x_n`` are the amplitude values for each point (annotations)
"""

# ╔═╡ bca75450-bef8-40da-aaa9-21af44dae187
@bind sv combine() do Child
	md"""
	A (a.u.) $(
		Child(Slider(0:0.1:10, default = 1))
	) 
	"""
end

# ╔═╡ 76e08834-ee2b-466f-a364-95e0cfb30908
begin
	function foo(i)
		x = yy[i] 
		s = (i % 2 == 0) ? @sprintf("%.3f", x) : @sprintf("%.1f", x)
		return Plots.text(s, 10, :left)
	end
	t2 = 0:0.01:1

	t3 = [0,pi/4, pi/2, 3pi/4, pi, 5pi/4, 6pi/4, 7pi/4]./(2pi)
	
	
	plot(t2, sv[1].*sin.(2.0*π*1*t2), legend = :none, color = :black, 
	xticks = 0:1, framestyle = :origin, xlabel = "time (s)", 
	ylabel = "Amplitude (a.u.)", size = (800, 200), left_margin = 20px, 
	bottom_margin = 20px)

	yy = sv[1].*sin.(2.0*π*1*t3)  
	plot!(t3, yy, marker = :circle, lw = 0, ms = 5)
	xx = map(foo, 1:8)
	# x = map(a -> , yy)
	annotate!(t3 .+ 0.02, yy, xx)
end

# ╔═╡ a34a3543-4b77-4166-91b8-cbc97364ab51
md"""
Thus the eight ``x_n`` values are
"""

# ╔═╡ 7e8c3346-2921-4e20-a2db-62a3263afaa2
x = round.(yy, digits = 3) 

# ╔═╡ fb4895ee-184f-4213-b918-a242df4d1dd8

md"""
Computing each Fourier coefficient involves a sum over ``N`` samples. 

Computing ``X(0) = \sum_{n=0}^{N-1} x_n \exp^{-i\frac{2\pi k}{N} n} = \sum_{n=0}^{N-1} x_n``, because ``\exp(0) = 1``. 
"""

# ╔═╡ 8ebac18f-ee13-4875-aaad-a3704320c960
begin
	function get_all(k)
		get_term(k, n) = "$(x[n+1])*exp(-im*2*π*$(k)/8*$n)"
		terms = map(n -> get_term(k, n), 0:7)
		expr = mapfoldl(x -> x * " + ", *, terms)[1:end-3]
		str = "X($k) = " * expr
		with_terminal() do 
			println(str)
			println("X($k) = ", Meta.parse(expr) |> eval |> round)
		end
	end
	get_all(0)
end

# ╔═╡ 34c4c94f-ef8c-4e4c-b703-b6aa61bff729
get_all(1)

# ╔═╡ 81879994-19b8-4ba2-a602-7f0306c192a3
get_all(2)

# ╔═╡ a555702d-1e69-4ef6-bfcb-a62c287c2923
get_all(3)

# ╔═╡ 5bf53b8d-45ed-46e8-8abe-95d0a9cfa2bf
get_all(4)

# ╔═╡ a00ab2a1-5b5e-4c66-9833-50504986d739
get_all(5)

# ╔═╡ 3bc44bfb-1055-4401-96b2-43067865b0e5
get_all(6)

# ╔═╡ a0055782-9f10-4a47-be68-a88442ffc582
get_all(7)

# ╔═╡ 6cb0e462-67de-4ed5-a671-51131a5c3c6c
Markdown.MD(
	Markdown.Admonition("warning", "Key Concepts", [md"
- There are ``N`` Fourier coefficients for ``N`` samples.
- Computing the Discrete Fourier Transform (DFT) requires evaluation of ``N^2`` terms (each X(i) is the summation over ``N`` terms). This means that the time complexity of the DFT is ``O(N^2)``. This makes it computationally expensive to compute the DFT for large datasets.
- The first Fourier coefficient is the fundamental frequency. It is always a real number and equals the average of the samples time series
- The Fourier coefficients are mirrored (conjugated). That is ``X(1) = X^*(7)``, ``X(2) = X^*(6)``, ``...``. Or: ``X(i) = X^*(N-i)``
- There is no new physical information in the second half of the Fourier coefficients. These are the negative frequencies resulting from using complex notation. This makes sense as each Fourier coefficent contains two pieces of information itself (Amplitude and phase), thus ``N`` datapoints map to ``N/2`` unique Fourier coefficients. 
"]))

# ╔═╡ 320dfe95-d647-4106-953a-3bdb6ef3133c
md"""
#### The Fourier coefficients

The DFT deconvolves the contribution of each frequency to the signal. Since the DFT operates in discrete space, the DFT samples at discrete frequencies. These are

`` 
f = k \frac{f_s}{N}
``

where ``f_s`` is the sampling frequency (in the example above 8 Hz), ``N`` is the number of samples (in the example above 8) and ``k`` is frequncy bin of the DFT. For an even number of samples, the unique spectrum is limited to $0 < f < f_s/2$. The frequency $f_s/2$ is the [Nyquist frequncy](https://en.wikipedia.org/wiki/Nyquist_frequency) and the highest frequency that can be resolved using the DFT analysis. 

The amplitude of each wave ``k \ge 1`` is given by

``
A = \frac{2}{N} \sqrt{X(k)X^*(k)}
``

As before, the factor 2 comes from combining the amplitudes of the positive and negative frequency Fourier coefficient. The amplitude needs to be normalized by the number of sample points.

The phase angle in radians of each wave is given by 

``
θ = \tan^{-1} \left ( \frac{\Im(X(k))}{\Re(X(k))} \right)
``

The 2-argument arctangent returns values betwenn ``-\pi`` and ``pi``. A convient short algorithm (expressed in Julia) to convert to the phase angle in ``0 ^\circ < \theta < 360 ^\circ`` is

```julia
θ = atan(imag(Xₖ), real(Xₖ)) * (180.0/pi) + 360.0) % 360.0
```

where ```x % y```  returns the remainder or signed remainder of a division and is known as the [modulo operator](https://en.wikipedia.org/wiki/Modulo).




"""

# ╔═╡ dea1d104-0ece-4540-99da-3ea58387235c
begin
	Xₖ  = round.(fft(x))
	mangle = map(Θ -> (Θ * (180.0/pi) + 360.0) % 360.0, atan.(imag.(Xₖ), real.(Xₖ)))
	Amp = real(2*sqrt.(Xₖ .* conj.(Xₖ))) / 8.0
	ftable = DataFrame(k = 0:4, f = map(k -> k * 8/8, 0:4), X = Xₖ[1:5], 
		A = Amp[1:5], Θ = mangle[1:5])
	rename(ftable, :k => "kᵗʰ bin", :f => "f (Hz)", :X => "X(k)", 
		:A => "Amplitude (-)", :Θ => "Θ (°)")
end

# ╔═╡ e4489e46-2a57-4ab9-badd-aa2895b48e98
md"""

The table shows a typical summary of the Fourier coefficients. The amplitude and phase spectrum are graphs of frequency vs. amplitude and frequency vs. phase
"""

# ╔═╡ 460e85b2-6b50-4d3e-8f57-73d6cc21b929
begin
	p1a = plot(ftable[!,2], ftable[!,4], line=:stem, marker=:circle, color = :black,
		xlabel = "Frequency(Hz)", ylabel = "Amplitude (-)", label = :none)
	p2a = plot(ftable[!,2], ftable[!,5], line=:stem, marker=:circle, color = :black,
	xlabel = "Frequency(Hz)", ylabel = "Phase (°)", label = :none, ylim = [-20,360])

	plot(p1a, p2a, size = (700,300), bottom_margin = 20px, left_margin = 10px)
end

# ╔═╡ 025ed5b0-3982-4ba4-8b78-9514da4adcab
Markdown.MD(
	Markdown.Admonition("danger", "Important Algorithn", [md"
As shown above the time complexity of DFT is ``O(N^2)``. In practice this is prohibitive. The **Fast Fourier Transform** (FFT) has time complexity ``O(N\log N)`` and is the universally used and widely implemented algorithm to compute the Fourier coefficients. The function ```fft(x)``` should work in virtually any language. A popular highly optimized C library is [FFTW](https://www.fftw.org/), which has bindings in many languages [Julia FFTW](https://github.com/JuliaMath/FFTW.jl), [Python FFTW](https://pypi.org/project/pyFFTW/), [R FFTW](https://cran.r-project.org/web/packages/fftw/index.html), 
"]))

# ╔═╡ ff846c7a-e507-4fa6-8dc4-aca16f65dde0
round.(fft(x)) 

# ╔═╡ 195d5da6-f246-4e4d-801f-a6d11b4d449d
html"""
<iframe width="690px" height="370px" src="https://www.youtube.com/embed/nmgFG7PUHfo" style="border:none;"> </iframe>
"""

# ╔═╡ 48baf264-8507-40bb-b4c7-15e0b82d2aae
md"""
### Example 2: DFT of three waves

- Consider the superposition of three cosine waves with amplitudes ``A_i``, frequencies ``f_i``, and phase angles ``\phi_i``. 
- There are ``N = 500`` discrete sample points comprising the trace.
- The total time is 5s. Thus the sampling frequency is 100 Hz.
- The highest resolved frequency (Nyquist frequency) is 50 Hz
- The lowest resolved frequency is 0.2 Hz (first Fourier coefficient)
"""

# ╔═╡ d7a0e2b4-e6e7-437e-a089-0987f81fc9b7
@bind v2 combine() do Child
	md"""
	A₁ (a.u.) $(
		Child(Slider(0:0.1:10, default = 1))
	) f₁ (Hz) $(
		Child(Slider(0.1:0.1:50, default = 2))
	) Φ₁ (deg) $( 
		Child(Slider(0:1:360, default = 15))
	)
	
	A₂ (a.u.) $(
		Child(Slider(0:0.1:10, default = 0.5))
	) f₂ (Hz) $(
		Child(Slider(0.1:0.1:50, default = 10))
	) Φ₂ (deg) $( 
		Child(Slider(0:1:360, default = 77))
	)

	A₃ (a.u.) $(
		Child(Slider(0:0.1:10, default = 2))
	) f₃ (Hz) $(
		Child(Slider(0.1:0.1:50, default = 40))
	) Φ₃ (deg) $( 
		Child(Slider(0:1:360, default = 270))
	)
	"""
end

# ╔═╡ e9ea80d2-48c2-4ce6-8f2d-d12c4d8a7a7d
begin
	wave1a = v2[1].*cos.(2.0.*v2[2].*pi.*t .+ v2[3]./360.0.*2.0.*pi)
	wave2a = v2[4].*cos.(2.0.*v2[5].*pi.*t .+ v2[6]./360.0.*2.0.*pi)
	wave3a = v2[7].*cos.(2.0.*v2[8].*pi.*t .+ v2[9]./360.0.*2.0.*pi)
			
	X1  = (fft(wave1a + wave2a + wave3a))
	mangle1 = map(Θ -> (Θ * (180.0/pi) + 360.0) % 360.0, atan.(imag.(X1), real.(X1)))
	Amp1 = real(2*sqrt.(X1 .* conj.(X1))) / 500
	ftable1 = DataFrame(k = 0:250, f = map(k -> k * 100/500, 0:250), X = X1[1:251], 
		A = Amp1[1:251], Θ = mangle1[1:251])
	rename(ftable1, :k => "kᵗʰ bin", :f => "f (Hz)", :X => "X(k)", 
		:A => "Amplitude (-)", :Θ => "Θ (°)")

	p4a = plot(t, wave1a + wave2a + wave3a, color = :black, xlabel = "t (s)", 
		ylabel = "A (a.u.)", label = "∑waves")

	p5a = plot(ftable1[2:end, 2], ftable1[2:end, 4], color = :black, 
		xscale = :log10, xlim = [0.1, 100], minorgrid = :true, 
		label = "Fourier Transform", xlabel = "Frequency (Hz)", ylabel = "Amplitude")
	p5b = plot!([v2[2]], [v2[1]], line=:stem, marker = :circle, color = :darkred, 
		label = "Wave 1")
	p5b = plot!([v2[5]], [v2[4]], line=:stem, marker = :circle, color = :steelblue3, 
		label = "Wave 2")
	p5b = plot!([v2[8]], [v2[7]], line=:stem, marker = :circle, 
		color = :darkgoldenrod, label = "Wave 3")


	p6a = plot(ftable1[2:end, 2], ftable1[2:end, 5], color = :black, 
		xscale = :log10, xlim = [0.1, 100], minorgrid = :true, ylim = [-20,360],
		label = "Fourier Transform", xlabel = "Frequency (Hz)", ylabel = "Phase")
	p6a = plot!([v2[2]], [v2[3]], line=:stem, marker = :circle, color = :darkred, 
		label = "Wave 1")
	p6a = plot!([v2[5]], [v2[6]], line=:stem, marker = :circle, color = :steelblue3, 
		label = "Wave 2")
	p6a = plot!([v2[8]], [v2[9]], line=:stem, marker = :circle, 
		color = :darkgoldenrod, label = "Wave 3")
	
	plot(p4a, p5a, p6a, layout = grid(3,1), size = (800, 600), 
		left_margin = 20px, bottom_margin = 20px, legend = :topleft)	
end

# ╔═╡ 1dd0225b-c974-49ae-99b2-b98d2360f0c5
ftable1

# ╔═╡ 9826982f-0d7c-43e2-8328-9b686612eaf5
Markdown.MD(
	Markdown.Admonition("warning", "Key Concepts", [md"
- The amplitude spectrum is broadened around the true amplitude. This smearing effect is also referred to as leakage, and arises from the fact that the DFT can only resolve discrete frequencies that may not match the true underlying wave frequency. 
- The phase spectrum is noisy/ambiguous. It may be required to clean up the signal prior to using/interpreting the phase spectrum.  
"]))

# ╔═╡ f0a64881-805a-459d-9e8a-31efa787af7a
begin
sspot_url = "https://upload.wikimedia.org/wikipedia/commons/6/67/Sunspots_1302_Sep_2011_by_NASA.jpg"
md"""
### Example 3: Amplitude of sunspot time series
$(Resource(sspot_url, :height => 500px)) 

[Sunspots](https://en.wikipedia.org/wiki/Sunspot) are phenomena on the Sun's photosphere that appear as temporary spots that are darker than the surrounding areas. They are regions of reduced surface temperature caused by concentrations of magnetic flux that inhibit convection. Their number varies according to the approximately 11-year solar cycle. 

The table below shows raw sunspot data download from ["http://serc.carleton.edu/files/introgeo/teachingwdata/examples/GreenwichSSNvstime.txt"]("http://serc.carleton.edu/files/introgeo/teachingwdata/examples/GreenwichSSNvstime.txt")
"""
end

# ╔═╡ c698ffa3-db8b-4920-b6ff-00416dc99094
begin 
	sunspot_data = CSV.read(HTTP.get("http://serc.carleton.edu/files/introgeo/teachingwdata/examples/GreenwichSSNvstime.txt").body, DataFrame; header = ["Year", "Sunspots"])
	sunspot_data
end

# ╔═╡ 09d0aa06-b1bd-4793-a08b-fa4cb4fa08c8
begin 
	year = Matrix(sunspot_data)[:,1]
	ss = Matrix(sunspot_data)[:,2]
	plot(year, ss, color = :black, xlabel = "Year", ylabel = "Sunspot Number", 
		label = :none, size = (700, 300), bottom_margin = 20px, left_margin = 20px)
end

# ╔═╡ 0605a57f-7cb2-4d83-ae54-ae1b65756efc
md"""
- Before performing a Fourier transform, it is important to test that a dataset is "clean". Missing data and irregularly spaced data are not suitable. That means no missing data and that time is regularly spaces. One test is to examine the dataset table/file. Another test is plotting the data. 
- Interpolation and/or averaging can be used to fill gaps of missing data and ensure constant spacing in the time domain. 
- The sample period is 0.08 y. (Evaluating the spacing in the table gives 0.08 and 0.09 likely due to rounding errors. It is close enough for this example). Thus the sample frequency is 1/0.08y = 12.5 y⁻¹. The Nyqyist frequency is 6.25 y⁻¹
- There are 3067 sampling points. To use FFT one should must restrict the number of samples to a number of 2-to-the-nth-power, i.e. 512, 1024, 2048, etc to avoid artefacts. Thus we will evaluate only the first 2048 samples from the dataset. An alternative to truncating the data is to zero-pad the signal to the next available size, here 4096 samples.
- The signal is assumed to periodic. That is, the value of sunspot numbers for sample 1 and sample 2048 should be identical. This is not the case and will result in spectral leakage. A [window function](https://en.wikipedia.org/wiki/Window_function) is typically applied to taper the signal at the edges. 
- Reasoning for the latter few points can be found [here](https://www.dataq.com/data-acquisition/general-education-tutorials/fft-fast-fourier-transform-waveform-analysis.html).
- The ``L``-point cosine taper (Tukey window), is one of many reasonable taper functions.

```math
w(x) = 
\begin{cases}
      \frac{1}{2} \left [ 1 + \cos \left (\frac{2\pi}{r}[x-r/2] \right ) \right ], &  0 \le x < \frac{r}{2} \\
      1, & \frac{r}{2}  \le x < 1 - \frac{r}{2} \\
	  \frac{1}{2} \left [ 1 + \cos \left (\frac{2\pi}{r}[x-1+r/2] \right ) \right ], & 1 - \frac{r}{2} \le x \le 1
\end{cases}
```

where ``x`` is an ``L``-point linearly spaced vector between 0 and 1 and the the parameter ``r`` is the ratio of the tapered section to the entire window length. A value of ``r = 0`` corresponds to no tapering/rectangular window. A value of ``r = 1`` corresponds to the Hanning window.
"""


# ╔═╡ b7a8a268-d12e-40ff-b95f-cf5669a96a1c
@bind tr combine() do Child
	md"""
	r $(
		Child(Slider(0:0.1:1, default = 0.1))
	) 
	"""
end

# ╔═╡ 5a9d6385-af99-4a4a-b70d-4cd8a2463353
begin
	function cosine_taper(L, r)
		function f(x)
			if (x < r/2.0) 
				return 0.5*(1.0+cos(2.0*pi/r*(x-r/2.0)))
			elseif ((x ≥ r/2.0) && (x < 1.0-r/2)) 
				return 1.0
			else
				a = 0.5*(1.0+cos(2.0*pi/r*(x-1.0+r/2.0)))
				b = isnan(a) ? 1.0 : a
				return b
			end
		end
		
		return map(f, range(0, stop = 1, length = L))
	end
	ct = cosine_taper(2048, tr[1])
	X3  = fft(ss[1:2048].*ct)
	mangle3 = map(Θ -> (Θ * (180.0/pi) + 360.0) % 360.0, atan.(imag.(X3), real.(X3)))
	Amp3 = real(2*sqrt.(X3 .* conj.(X3))) / 2048
	ftable3 = DataFrame(k = 0:1024, f = map(k -> k * 12.5/2048, 0:1024), 
		X = X3[1:1025], A = Amp3[1:1025], Θ = mangle3[1:1025])
	ftable3[1,4] = ftable3[1,4]/2 
	rename(ftable3, :k => "kᵗʰ bin", :f => "f (Hz)", :X => "X(k)", 
		:A => "Amplitude (-)", :Θ => "Θ (°)")

	p11 = plot(ct, color = :black, label = "Tukey window", xticks = 0:128:2048,
		xlabel = "Sample number", ylabel = "Amplitude", ylim = [0,1])

	p12 = plot(ss[1:2048].*ct, color = :black, label = "Windowed sample", 
		xticks = 0:128:2048, xlabel = "Sample number", ylabel = "Amplitude")

	p13 = plot(ftable3[2:end, 2], ftable3[2:end, 4], color = :black, 
		xscale = :log10,  minorgrid = :true, label = "Fourier Transform", 
		xlabel = "Frequency (y⁻¹)", ylabel = "Amplitude", legend = :right)

	p13 = plot!([1.0/10.98], [25.0], line=:stem, marker = :circle, color = :darkred, 
	 	label = "10.98 years")

	p13 = plot!([1.0/86], [20.0], line=:stem, marker = :circle, 
		color = :steelblue3, label = "86 years")

	plot(p11, p12, p13, layout = grid(3,1), size = (800, 600), 
		left_margin = 20px, bottom_margin = 10px)	
end

# ╔═╡ 007c3147-3f53-4ef3-a5b3-9a43203ecfce
ftable3

# ╔═╡ d410fc70-565e-46d9-af87-6585c74cdda3
Markdown.MD(
	Markdown.Admonition("warning", "Key Concept", [md"
- Recall that Fourier coefficient ``k = 0`` corresponds to the mean. Note that when computing the amplitude, there is no coefficient of 2, i.e. ``A = X(k)/N``. This because there is no mirrored coefficient for that frequency. It is easy to confirm that the first amplitude in the table corresponds to the mean of the signal.  
"]))

# ╔═╡ f1c3da44-e3f2-4249-a194-28c1caf1b82a
mean(ss[1:2048].*ct) # Windowed subsetted data

# ╔═╡ 5e046263-ff18-40b4-9810-f32b89316e3d
Markdown.MD(
	Markdown.Admonition("warning", "Key Concept", [md"
- The windowing will slightly affect the spectrum. Tapering at the edges will make it difficult/impossible to resolve the longest frequencies. Observe how the 86 year frequency is obscured when narrowing the window.
"]))

# ╔═╡ f44e4292-97bc-4372-b28e-6868f8e64449
begin
	sonic_url = "https://upload.wikimedia.org/wikipedia/commons/8/87/WindMaster.jpg"
	bdl_url = "https://mdpetters.github.io/cee200/assets/boundary_layer.png"
	
	md"""
	### Example 4: Power spectra of vertical velocity 

	Pollution dispersion occurs within the atmospheric boundary layer. The dynamics of the boundary layer are complex due to the presence of turbulent motions. 
	
	$(Resource(bdl_url)) 

	**Figure.** Schematic of the atmopspheric boundary layer, adapted from  Kadivar et al. (2021), https://doi.org/10.1016/j.ijft.2021.100077.**


	$(Resource(sonic_url, :height => 400px)) 
	
	A sonic anemometer measures the 3-dimensional wind speed and (sonic) temperature at high frequency. These measurements are critical to understand the turbulent structure of the atmosphere. The current descriptions of atmospheric turbulence are mostly through statistics and time-series analysis. Spectral analysis of vertical velocity data is a common tool applied in the field.

	- Below is a dataset of 32768 points (2^15 samples) taken with a sonic anemometer at 10 m height inside a convective boundary in the southeastern United States in July 2022.   
	- The unit of vertical velocity (w) is m s⁻¹.
	- The unit of temperature (T) is °C.
	- The time resolution is 0.1 s. The sample frequency is 10 Hz. 
	- The Nyquist frequency is 5 Hz
	"""
end

# ╔═╡ c72f0b73-75ca-4e42-b73d-eb573081fca0
begin
	data = HTTP.get("https://mdpetters.github.io/cee200/assets/velocity_data.csv")
	turbulence_data = CSV.read(data.body, DataFrame)
	turbulence_data
end

# ╔═╡ 45114e1c-baac-4334-bff8-6b019a16b11c
begin

	ct4 = cosine_taper(2^15, 0.1)
	tdata = turbulence_data[!,2]
	tapered_data = tdata.*ct4
	
	X4  = fft(tapered_data) 
	mangle4 = map(Θ -> (Θ * (180.0/pi) + 360.0) % 360.0, atan.(imag.(X4), real.(X4)))
	Amp4 = real(2*sqrt.(X4 .* conj.(X4))) / 32768
	P4 = 0.5.*Amp4.^2
	summary_table = DataFrame(k = 0:16384, f = map(k -> k * 10/32768, 0:16384), 
		X = X4[1:16385], A = Amp4[1:16385], P = P4[1:16385])
	summary_table[1,4] = summary_table[1,4]/2 
	rename(summary_table, :k => "kᵗʰ bin", :f => "f (Hz)", :X => "X(k)", 
		:A => "Amplitude (-)", :P => "P (°)")

	deltaf = (summary_table[2:end,:f] .- summary_table[1:end-1,:f])[1]
	
	p21 = plot(ct4, color = :black, label = "Tukey window", 
		xticks = (0:8192:32768, ["0", "8192", "16384", "24576", "32768"]),
		xlabel = "Sample number", ylabel = "m s⁻¹", ylim = [0,1])

	p22 = plot(tapered_data, color = :black, label = "Windowed sample", 
		xticks = (0:8192:32768, ["0", "8192", "16384", "24576", "32768"]), 
		xlabel = "Sample number", ylabel = "Amplitude")

	p23a = plot(summary_table[2:end, 2], summary_table[2:end, 5]./deltaf, 
		color = :black,  minorgrid = :true, label = "Fourier Transform", 
		xlabel = "Frequency (Hz)", ylabel = "S (m² s⁻² Hz⁻¹)", 
		legend = :right, xlim = [-0.1, 1])

	p23b = plot(summary_table[2:end, 2], summary_table[2:end, 5]./deltaf, 
		color = :black, xscale = :log10, 
		minorgrid = :true, label = "Fourier Transform", 
		xlabel = "Frequency (Hz)", ylabel = "S (m² s⁻² Hz⁻¹)", 
		yscale = :log10, legend = :left, xlim = [0.0001, 10])

	p23 = plot(p23a, p23b, layout = grid(1,2))
	plot(p22, p23, layout = grid(2,1), size = (800, 800), 
		left_margin = 20px, bottom_margin = 10px)	
end


# ╔═╡ 09f4a93f-8e88-49f4-81a5-ec194713d86c
let
	ct4 = cosine_taper(2^15, 0.1)
	tdata = turbulence_data[!,2]
	tapered_data = tdata.*ct4
	
	X4  = fft(tapered_data) 
	mangle4 = map(Θ -> (Θ * (180.0/pi) + 360.0) % 360.0, atan.(imag.(X4), real.(X4)))
	Amp4 = real(2*sqrt.(X4 .* conj.(X4))) / 32768
	P4 = 0.5.*Amp4.^2

	P5 = 2/(32768)^2 * (real.(X4).^2 + imag.(X4).^2)
	P4 .== P5
end

# ╔═╡ c0072a98-ddb5-4ac7-9d10-37bcf460b8e3
md"""

Often we are less interested in the amplitude spectrum, but in the power spectrum. The [energy (or power) associated with a wave[(http://hyperphysics.phy-astr.gsu.edu/hbase/Waves/powstr.html) is proportional to the square of the amplitude. The power at each frequency is computed from the amplitude spectrum via

```math
P(k) = \frac{1}{2}A(k)^2
```

where ``A(k)`` is the amplitude at the ``k^{th}`` frequency bin. The power spectrum has an intersting property: 

```math
\sum_{k=1}^{N/2} P(k) = VAR(y(t))
```

i.e., the sum of the powers at each frequency equals to the total variance of the time-series. Note that the power of the fundamental frequency ``k = 0`` is not included. The power spectrum gives the contribution of each frequency to the variance. It is therefore sometimes referred to as the variance spectrum.

The timeseries is that of a physical quantity (vertical velocity m s⁻¹). Thus the unit of ``P(k)`` is m² s⁻². The variance spectrum is sometimes normalized by bin-width of the frequency spectrum ``\Delta f``

```math
S(k) = \frac{P(k)}{\Delta f} 
```

In the case of the vertical velocity timeseries, this quantity has units of m² s⁻² Hz⁻¹. The normalization is performed to allow comparing spectra from different studies that have been measured at different frequency resolution. Fortunately, the ``\Delta f`` from the fft is constant across the spectrum: 
"""



# ╔═╡ ba1b5fbe-0543-47aa-a789-dbbd342c478b
Δf = summary_table[2:end,:f] .- summary_table[1:end-1,:f]

# ╔═╡ e064dc71-2012-4cba-a974-f7a5af86cbd8
summary_table

# ╔═╡ 37c58238-8109-464a-83f4-b9f886a2fccf
md"""
You should **always** check your computed results for self consistency. The two "truisms" that should always hold are:

```math
A(0) = MEAN(y(t))
```

and

```math
\sum_{k=1}^{N/2} P(k) = VAR(y(t))
```
"""

# ╔═╡ 42ceebb9-e01f-4ae5-9f17-7e30d6d69fce
mean_from_table, variance_from_table = summary_table[1,:A], sum(summary_table[2:end,:P])

# ╔═╡ 4c8ae3a1-837d-44f1-8f79-ea18dd815e71
mean_from_time_series, variance_from_timeseries = mean(tapered_data), var(tapered_data)

# ╔═╡ f010d059-eaba-454a-a19d-d762e0562a3b
Markdown.MD(
	Markdown.Admonition("warning", "Key Concept", [md"
The statement
```math
VAR(y(t)) = \sum_{k=1}^{N/2} P(k)  
```
is equivalent to 

```math
VAR(y(t)) = \int S(k) df 
```

That is, the area under the curve of the energy spectrum (in linear coordinates) is the contribution of the signal to it's variance. Often the spectrum varies over orders of magnitude in both frequency and power. Those spectra are generally presented in log-log coordinates. Note, however that the log-log representation does not conserve area.  
"]))



# ╔═╡ 766bbe53-9916-4138-be08-382336f7c5c2
begin
butter_url = "https://upload.wikimedia.org/wikipedia/commons/c/cd/Butterworth_Filter_Orders.svg"
md"""### Example 5: Frequency filtering using the DFT

Another important application of the Fourier transform is to filter/smooth noisy data. A simple smoothing algorithm is to take a running average. Another common method is to Fourier transform the data, "delete" the high frequencies by setting them to zero, and then perform the inverse Fourier transform of the filtered data. Below is an example using - again - sunspot data. The difference to the previous dataset is that it is higher frequency daily data. Changing the frequency cutoff clearly smoothes the time series by removing noise. This is also a good method to remove outliers from a dataset.

Applying this idea filter "quick and dirty" low-pass filter is a good first start but may lead to "ringing" artifacts. It is preferrable to apply a window filter that gradually attenuates the frequencies. This is also how real world filters work, e.g. the RC circuit. You can learn more about frequency filtering [here](https://brianmcfee.net/dstbook-site/content/ch10-convtheorem/intro.html). An example of a good filter is the Butterworth filter:

$(Resource(butter_url)) 
**Figure.** Plot of the gain of Butterworth low-pass filters of orders 1 through 5, with cutoff frequency ``\omega _{0}=1``. Source: [Wikipedia](https://en.wikipedia.org/wiki/Butterworth_filter).
"""
end

# ╔═╡ 63d2185e-de51-4f6d-b97b-a5f2b193fa87
@bind cutf combine() do Child
	md"""
	frequency cutoff $(
		Child(Slider([0.001:0.001:0.009;0.01:0.01:0.09; 0.1], default = 0.01))
	) 
	"""
end

# ╔═╡ de2e72fc-a747-40b7-903b-24e5480f3441
begin
	function get_sun(c)
		sundata = HTTP.get("https://mdpetters.github.io/cee200/assets/SN_d_tot_V2_clean.csv")
		sun2 = CSV.read(sundata.body, DataFrame, header = false)
		d = Matrix(sun2)
		n,m = size(d)
		t = map(i -> Date(d[i,1], d[i,2], d[i,3]), 1:n)
		ii = d[:,5] .== -1.0
		d[ii,5] .= 0
		(t[end-32367:end], d[end-32367:end,5])	

		ct = cosine_taper(2^15, 0.1)
		tdata = d[end-32767:end,5]
		length(tdata)
		tapered_data = tdata.*ct
	
		X  = fft(tapered_data) 
		mangle = map(Θ -> (Θ * (180.0/pi) + 360.0) % 360.0, 
		 	atan.(imag.(X), real.(X)))
		Amp = real(2*sqrt.(X .* conj.(X))) / 32768
		P = 0.5.*Amp.^2
		
		freq = map(k -> k * 1/32768, 0:16384)
		i = c .< freq 
		j = sum(i)		
		Y1 = deepcopy(X[1:16384])
		Y2 = deepcopy(X[16385:end])
		Y1[end-j:end] .= 1e-10
		Y2[1:j] .= 1e-10
		Z =[Y1;Y2]
		Amp2 = real(2*sqrt.(Z .* conj.(Z))) / 32768
		P2 = 0.5.*Amp2.^2

		summary_table = DataFrame(k = 0:16384, f = freq, 
			X = X[1:16385], A = Amp[1:16385], P = P[1:16385], P2 = P2[1:16385])
		summary_table[1,4] = summary_table[1,4]/2 
		
		filtered_data = ifft(Z) |> real
		deltaf = (summary_table[2:end,:f] .- summary_table[1:end-1,:f])[1]
	
		p22 = plot(tapered_data, color = :black, label = "Windowed sample", 
		xticks = (0:8192:32768, ["0", "8192", "16384", "24576", "32768"]), 
		xlabel = "Sample number", ylabel = "Sunspots")
		p22 = plot!(filtered_data, label = "Filtered data")

		p23 = plot(summary_table[2:end, 2], summary_table[2:end, 5]./deltaf, 
			color = :black, xscale = :log10, 
			minorgrid = :true, label = "Fourier Transform", 
			xlabel = "Frequency (d⁻¹)", ylabel = "P (Sunspots²)", 
			yscale = :log10, legend = :left, xlim = [0.00001, 1], ylim = [1e-2, 1e8])
		p23 = plot!(summary_table[2:end, 2], summary_table[2:end, 6]./deltaf, 
			label = "Low-pass-filtered spectrum")

		plot(p22, p23, layout = grid(2,1), size = (800, 600), 
		left_margin = 20px, bottom_margin = 10px)	
	end

	 get_sun(cutf[1])
end

# ╔═╡ 53c64a93-9da8-4f6a-8fe6-8edd1017e179
md"""
# Fourier Transform 

```math
\begin{array}{lll}
 & \text{Discrete Fourier transform:} & 
X(k) = \sum_{n = 0}^{N-1} x_n \exp^{-i 2\pi \frac{k}{N} n}
 \\
& \text{Inverse discrete Fourier transform:} & x(n) = \sum_{k =0}^{N-1} X_k \exp^{i 2 \pi \frac{k}{N} n}
\end{array}
```


## Fourier Transform Pairs
```math
\begin{array}{lll}
\text{(i)} & \text{Fourier transform:} & 
\displaystyle\mathcal{F}\{f(x)\} = \int_{-\infty}^{\infty} f(x)e^{i\alpha x} dx = F(\alpha)
\\

			& \text{Inverse Fourier transform:} & 
\displaystyle\mathcal{F}^{-1}\{F(\alpha)\} = \frac{1}{2\pi}\int_{-\infty}^{\infty} F(\alpha)e^{-i\alpha x} d\alpha = f(x)
\\
\text{(ii)} & \text{Fourier sine transform:} & 
\displaystyle\mathcal{F}_s\{f(x)\} = \int_{0}^{\infty} f(x)\sin\alpha x dx = F(\alpha)
\\
 & \text{Inveres Fourier sine transform:} & 
\displaystyle\mathcal{F}_s^{-1}\{F(\alpha)\} = \frac{2}{\pi}\int_{0}^{\infty} F(\alpha)\sin \alpha x d\alpha = f(x)
\\
\text{(iii)} & \text{Fourier cosine transform:} & 
\displaystyle\mathcal{F}_c\{f(x)\} = \int_{0}^{\infty} f(x)\cos\alpha x dx = F(\alpha)
\\
 & \text{Inveres Fourier cosine transform:} & 
\displaystyle\mathcal{F}_c^{-1}\{F(\alpha)\} = \frac{2}{\pi}\int_{0}^{\infty} F(\alpha)\cos \alpha x d\alpha = f(x)
\\

\end{array}
```
"""

# ╔═╡ d5a0dc74-14f7-4002-a61a-47f09c95c977
md"""
## Fourier Transform of derivatives
Suppose that ``f`` is continuous and absolutely integrable on the interfval ``(-\infty, \infty)`` and ``f'`` is piecewise continuous on every finite interval. If ``f (x) \to 0`` as ``x\to\pm \infty``, then integration by parts gives
```math
\mathcal{F}\{f'(x)\} = -i\alpha F(\alpha).
```
and 
```math
\mathcal{F}\{f''(x)\} = (-i\alpha)^2 F(\alpha)=-\alpha^2F(\alpha).
```
In general 
```math
\mathcal{F}\{f^{(n)}(x)\} = (-i\alpha)^n F(\alpha), \quad n=0,1,2,\cdots
```
"""

# ╔═╡ dbaa63ee-5d9e-416f-ab63-30962c117114
md"""
# Laplace Transform

## Definition

Let ``f`` be a function defined for ``t \ge 0``. Then the integral
```math
\mathcal{L}\{f(t)\}=\int_0^{\infty}e^{-st}f(t) dt 
```
is said to be the one-sided __Laplace transform__ of ``f``, provided the integral converges.

__Note__: For the Laplace transform ``s = a + bi`` is a complex number. The two-sided Laplace transform is 

$\mathcal{L}\{f(t)\}=\int_{-\infty}^{\infty}e^{-st}f(t) dt$ 

Therefore, the Fourier transform is a special case of the two-sided Laplace transform where the ``\Re({s}) = 0``.
"""

# ╔═╡ 08f94fa7-55f3-4983-88aa-838e73be3b8a
md"""
## Transforms of Some Basic Functions
```math
\begin{array}{llcl}
\textbf{(a)} & \mathcal{L}\{1\} &=& \frac{1}{s}\\
\textbf{(b)} & \mathcal{L}\{t^n\} &=& \frac{n!}{s^{n+1}},\quad n=1,2,3,\cdots\\
\textbf{(c)} & \mathcal{L}\{e^{at}\} &=& \frac{1}{s-a}\\
\textbf{(d)} & \mathcal{L}\{\sin kt\} &=& \frac{k}{s^2+k^2}\\
\textbf{(e)} & \mathcal{L}\{\cos kt\} &=& \frac{s}{s^2+k^2}\\
\textbf{(f)} & \mathcal{L}\{\sinh kt\} &=& \frac{k}{s^2-k^2}\\
\textbf{(g)} & \mathcal{L}\{\cosh kt\} &=& \frac{s}{s^2-k^2}\\
\end{array}
```
"""

# ╔═╡ 56fcab84-9174-4207-9a54-5f81fbab59c3
let 
	@syms s k::positive t
	integrate(exp(-s*t)*sin(k*t), (t, 0, oo)) |> simplify
end

# ╔═╡ 453650c6-395d-4710-921f-961e40fcd0b4
md"""
## The Inverse Transform and Transforms of Derivatives
If ``F(s)`` represents the __Laplace transform__ of a function ``f(t)``, 
that is, ``\mathcal{L}\{ f (t)\} = F (s)``, we then say ``f(t)`` is the __inverse Laplace transform__ of ``F(s)`` and write
```math
f(t) = \mathcal{L}^{-1}\{F(s)\}.
```

__Some Inverse Transforms__

```math
\begin{array}{lcl}
1 &=& \mathcal{L}^{-1}\left\{\frac{1}{s}\right\} \\
t^n &=& \mathcal{L}^{-1}\left\{\frac{n!}{s^{n+1}}\right\}, \quad n=1,2,3,\cdots \\
e^{at} &=& \mathcal{L}^{-1}\left\{\frac{1}{s-a}\right\} \\
\sin kt &=& \mathcal{L}^{-1}\left\{\frac{k}{s^2+k^2}\right\} \\
\cos kt &=& \mathcal{L}^{-1}\left\{\frac{s}{s^2+k^2}\right\} \\
\sinh kt &=& \mathcal{L}^{-1}\left\{\frac{k}{s^2-k^2}\right\} \\
\cosh kt &=& \mathcal{L}^{-1}\left\{\frac{s}{s^2-k^2}\right\} \\
\end{array}
```
"""

# ╔═╡ c84358b4-09ff-4ddc-a4c3-31068a8a8662
md"""
## Transforms of Derivatives

If ``f, f', \cdots , f^{(n-1)}`` are continuous on ``[0, \infty)`` and are of exponential order and if ``f^{(n)}(t)`` is piecewise continuous on ``[0, \infty)``, then
```math
\mathcal{L}\{f^{(n)}(t)\} = s^n F(s) - s^{n-1}f(0) - s^{n-2} f'(0) - \cdots - f^{(n-1)}(0),
```
where ``F(s) = \mathcal{L}\{f (t)\}``.
"""

# ╔═╡ c7263946-d128-41d7-9f4e-0c6d033a168d


# ╔═╡ 1b2a2ca4-3092-43e7-b5d8-689dc213069a


# ╔═╡ ad437aee-00a0-4ac0-9685-2d8314fe66a1


# ╔═╡ bbcd415e-42bc-4d85-9354-f32675a391fc


# ╔═╡ 927da28f-a16f-4f6b-a662-4490b28c12fb


# ╔═╡ 00000000-0000-0000-0000-000000000001
PLUTO_PROJECT_TOML_CONTENTS = """
[deps]
CSV = "336ed68f-0bac-5ca0-87d4-7b16caf5d00b"
DataFrames = "a93c6f00-e57d-5684-b7b6-d8193f3e46c0"
Dates = "ade2ca70-3891-5945-98fb-dc099432e06a"
FFTW = "7a1cc6ca-52ef-59f5-83cd-3a7055c09341"
HTTP = "cd3eb016-35fb-5094-929b-558a96fad6f3"
Plots = "91a5bcdd-55d7-5caf-9e0b-520d859cae80"
PlutoUI = "7f904dfe-b85e-4ff6-b463-dae2292396a8"
Printf = "de0858da-6303-5e67-8744-51eddeeeb8d7"
Statistics = "10745b16-79ce-11e8-11f9-7d13ad32a3b2"
SymPy = "24249f21-da20-56a4-8eb1-6a02cf4ae2e6"

[compat]
CSV = "~0.10.11"
DataFrames = "~1.6.1"
FFTW = "~1.7.1"
HTTP = "~1.9.14"
Plots = "~1.38.17"
PlutoUI = "~0.7.52"
SymPy = "~1.1.12"
"""

# ╔═╡ 00000000-0000-0000-0000-000000000002
PLUTO_MANIFEST_TOML_CONTENTS = """
# This file is machine-generated - editing it directly is not advised

julia_version = "1.9.3"
manifest_format = "2.0"
project_hash = "b4df7c7302d7bb8ec7ad1e3c28af42ed98063e38"

[[deps.AbstractFFTs]]
deps = ["LinearAlgebra"]
git-tree-sha1 = "d92ad398961a3ed262d8bf04a1a2b8340f915fef"
uuid = "621f4979-c628-5d54-868e-fcf4e3e8185c"
version = "1.5.0"

    [deps.AbstractFFTs.extensions]
    AbstractFFTsChainRulesCoreExt = "ChainRulesCore"
    AbstractFFTsTestExt = "Test"

    [deps.AbstractFFTs.weakdeps]
    ChainRulesCore = "d360d2e6-b24c-11e9-a2a3-2a2ae2dbcce4"
    Test = "8dfed614-e22c-5e08-85e1-65c5234f0b40"

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
git-tree-sha1 = "dd3000d954d483c1aad05fe1eb9e6a715c97013e"
uuid = "35d6a980-a343-548e-a6ea-1d62b119f2f4"
version = "3.22.0"

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
git-tree-sha1 = "5ce999a19f4ca23ea484e92a1774a61b8ca4cf8e"
uuid = "34da2185-b29b-5c13-b0c7-acf172513d20"
version = "4.8.0"
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
git-tree-sha1 = "cf25ccb972fec4e4817764d01c82386ae94f77b4"
uuid = "864edb3b-99cc-5e75-8d2d-829cb0a9cfe8"
version = "0.18.14"

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
git-tree-sha1 = "f61f768bf090d97c532d24b64e07b237e9bb7b6b"
uuid = "d2c73de3-f751-5644-a686-071e5b155ba9"
version = "0.72.9+0"

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

[[deps.IntelOpenMP_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "0cb9352ef2e01574eeebdb102948a58740dcaf83"
uuid = "1d5cc7b8-4909-519e-a0f8-d0f5ad9712d0"
version = "2023.1.0+0"

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
deps = ["Preferences"]
git-tree-sha1 = "abc9885a7ca2052a736a600f7fa66209f96506e1"
uuid = "692b3bcd-3c85-4b1f-b108-f13ce0eb3210"
version = "1.4.1"

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

[[deps.LazyArtifacts]]
deps = ["Artifacts", "Pkg"]
uuid = "4af54fe1-eca0-43a8-85a7-787d91b784e3"

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
git-tree-sha1 = "c3ce8e7420b3a6e071e0fe4745f5d4300e37b13f"
uuid = "2ab3a3ac-af41-5b50-aa03-7779005ae688"
version = "0.3.24"

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
git-tree-sha1 = "cedb76b37bc5a6c702ade66be44f831fa23c681e"
uuid = "e6f89c97-d47a-5376-807f-9c37f3926c36"
version = "1.0.0"

[[deps.MIMEs]]
git-tree-sha1 = "65f28ad4b594aebe22157d6fac869786a255b7eb"
uuid = "6c6e2e6c-3030-632d-7369-2d6c69616d65"
version = "0.1.4"

[[deps.MKL_jll]]
deps = ["Artifacts", "IntelOpenMP_jll", "JLLWrappers", "LazyArtifacts", "Libdl", "Pkg"]
git-tree-sha1 = "154d7aaa82d24db6d8f7e4ffcfe596f40bff214b"
uuid = "856f044c-d86e-5d09-b602-aeab76dc8ba7"
version = "2023.1.0+0"

[[deps.MacroTools]]
deps = ["Markdown", "Random"]
git-tree-sha1 = "42324d08725e200c23d4dfb549e0d5d89dede2d2"
uuid = "1914dd2f-81c6-5fcd-8719-6d5c9610ff09"
version = "0.5.10"

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
git-tree-sha1 = "1aa4b74f80b01c6bc2b89992b861b5f210e665b5"
uuid = "458c3c95-2e84-50aa-8efc-19380b2a3a95"
version = "1.1.21+0"

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
git-tree-sha1 = "9673d39decc5feece56ef3940e5dafba15ba0f81"
uuid = "aea7be01-6a6a-4083-8856-8a6e6704d82a"
version = "1.1.2"

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
git-tree-sha1 = "074f993b0ca030848b897beff716d93aca60f06a"
uuid = "5c2747f8-b7ea-4ff2-ba2e-563bfd36b1d4"
version = "1.4.2"

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
git-tree-sha1 = "c4d2a349259c8eba66a00a540d550f122a3ab228"
uuid = "1986cc42-f94f-5a68-af5c-568840ba703d"
version = "1.15.0"

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
git-tree-sha1 = "2222b751598bd9f4885c9ce9cd23e83404baa8ce"
uuid = "ffd25f8a-64ca-5728-b0f7-c24cf3aae800"
version = "5.4.3+1"

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
# ╟─c6b0dc22-2f17-11ee-3378-4d9f003bef9e
# ╟─7fb213ec-4168-405b-9ccd-3d209412f320
# ╟─7bdac999-3fb8-4956-8fe8-d8e3da626f8b
# ╟─65b1e136-a659-4cb0-83a1-d7d5711ee4f4
# ╟─4f155487-04a8-4d3b-9085-cd8baf611934
# ╟─d782a289-abf8-416b-84ee-67f7c10eca99
# ╟─c26f651a-0417-470c-a7b4-731f2c7a5dfa
# ╟─1c505532-6083-4201-a6a6-ecdec9c3ce74
# ╟─043b09fc-fbb2-46e4-8829-ee19bb4ddcad
# ╟─29ed7682-81f5-4a90-8960-e69f1979c018
# ╟─3b3ac786-d3d5-49f5-af43-fe4800bb6dac
# ╟─6ced3576-4768-4d92-8e07-8420bc77e2de
# ╟─009069c3-a918-449b-a01e-58671bc8c189
# ╟─555b410b-4a51-4810-8be2-389a2e02589f
# ╟─bca75450-bef8-40da-aaa9-21af44dae187
# ╟─76e08834-ee2b-466f-a364-95e0cfb30908
# ╟─a34a3543-4b77-4166-91b8-cbc97364ab51
# ╟─7e8c3346-2921-4e20-a2db-62a3263afaa2
# ╟─fb4895ee-184f-4213-b918-a242df4d1dd8
# ╟─8ebac18f-ee13-4875-aaad-a3704320c960
# ╟─34c4c94f-ef8c-4e4c-b703-b6aa61bff729
# ╟─81879994-19b8-4ba2-a602-7f0306c192a3
# ╟─a555702d-1e69-4ef6-bfcb-a62c287c2923
# ╟─5bf53b8d-45ed-46e8-8abe-95d0a9cfa2bf
# ╟─a00ab2a1-5b5e-4c66-9833-50504986d739
# ╟─3bc44bfb-1055-4401-96b2-43067865b0e5
# ╟─a0055782-9f10-4a47-be68-a88442ffc582
# ╟─6cb0e462-67de-4ed5-a671-51131a5c3c6c
# ╟─320dfe95-d647-4106-953a-3bdb6ef3133c
# ╠═dea1d104-0ece-4540-99da-3ea58387235c
# ╟─e4489e46-2a57-4ab9-badd-aa2895b48e98
# ╟─460e85b2-6b50-4d3e-8f57-73d6cc21b929
# ╟─025ed5b0-3982-4ba4-8b78-9514da4adcab
# ╠═ff846c7a-e507-4fa6-8dc4-aca16f65dde0
# ╠═195d5da6-f246-4e4d-801f-a6d11b4d449d
# ╟─48baf264-8507-40bb-b4c7-15e0b82d2aae
# ╟─d7a0e2b4-e6e7-437e-a089-0987f81fc9b7
# ╟─e9ea80d2-48c2-4ce6-8f2d-d12c4d8a7a7d
# ╟─1dd0225b-c974-49ae-99b2-b98d2360f0c5
# ╟─9826982f-0d7c-43e2-8328-9b686612eaf5
# ╟─f0a64881-805a-459d-9e8a-31efa787af7a
# ╠═c698ffa3-db8b-4920-b6ff-00416dc99094
# ╟─09d0aa06-b1bd-4793-a08b-fa4cb4fa08c8
# ╟─0605a57f-7cb2-4d83-ae54-ae1b65756efc
# ╟─b7a8a268-d12e-40ff-b95f-cf5669a96a1c
# ╟─5a9d6385-af99-4a4a-b70d-4cd8a2463353
# ╟─007c3147-3f53-4ef3-a5b3-9a43203ecfce
# ╟─d410fc70-565e-46d9-af87-6585c74cdda3
# ╠═f1c3da44-e3f2-4249-a194-28c1caf1b82a
# ╟─5e046263-ff18-40b4-9810-f32b89316e3d
# ╟─f44e4292-97bc-4372-b28e-6868f8e64449
# ╟─c72f0b73-75ca-4e42-b73d-eb573081fca0
# ╠═45114e1c-baac-4334-bff8-6b019a16b11c
# ╠═09f4a93f-8e88-49f4-81a5-ec194713d86c
# ╟─c0072a98-ddb5-4ac7-9d10-37bcf460b8e3
# ╠═ba1b5fbe-0543-47aa-a789-dbbd342c478b
# ╠═e064dc71-2012-4cba-a974-f7a5af86cbd8
# ╟─37c58238-8109-464a-83f4-b9f886a2fccf
# ╠═42ceebb9-e01f-4ae5-9f17-7e30d6d69fce
# ╠═4c8ae3a1-837d-44f1-8f79-ea18dd815e71
# ╟─f010d059-eaba-454a-a19d-d762e0562a3b
# ╟─766bbe53-9916-4138-be08-382336f7c5c2
# ╟─63d2185e-de51-4f6d-b97b-a5f2b193fa87
# ╟─de2e72fc-a747-40b7-903b-24e5480f3441
# ╟─53c64a93-9da8-4f6a-8fe6-8edd1017e179
# ╟─d5a0dc74-14f7-4002-a61a-47f09c95c977
# ╟─dbaa63ee-5d9e-416f-ab63-30962c117114
# ╟─08f94fa7-55f3-4983-88aa-838e73be3b8a
# ╠═56fcab84-9174-4207-9a54-5f81fbab59c3
# ╟─453650c6-395d-4710-921f-961e40fcd0b4
# ╟─c84358b4-09ff-4ddc-a4c3-31068a8a8662
# ╠═c7263946-d128-41d7-9f4e-0c6d033a168d
# ╠═1b2a2ca4-3092-43e7-b5d8-689dc213069a
# ╠═ad437aee-00a0-4ac0-9685-2d8314fe66a1
# ╠═bbcd415e-42bc-4d85-9354-f32675a391fc
# ╠═927da28f-a16f-4f6b-a662-4490b28c12fb
# ╟─00000000-0000-0000-0000-000000000001
# ╟─00000000-0000-0000-0000-000000000002
