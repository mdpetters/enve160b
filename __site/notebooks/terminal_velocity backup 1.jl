### A Pluto.jl notebook ###
# v0.19.39

using Markdown
using InteractiveUtils

# ╔═╡ e752f5da-d4b7-11ee-039b-e1e972398564
begin
	using PlutoUI
	using Plots	
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

	# Logging.disable_logging(Logging.Warn)

	la_url = "https://mdpetters.github.io/cee233/assets/Los_Angeles_Pollution.jpg"
	
	md"""
	$(TableOfContents(depth=5))
	# Terminal Velocity"
	"""
end


# ╔═╡ d2107caa-d368-4735-9a1f-7d4018e7032c


# ╔═╡ Cell order:
# ╠═e752f5da-d4b7-11ee-039b-e1e972398564
# ╠═d2107caa-d368-4735-9a1f-7d4018e7032c
