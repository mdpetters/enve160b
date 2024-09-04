### A Pluto.jl notebook ###
# v0.19.29

using Markdown
using InteractiveUtils

# ╔═╡ 05381ce2-6a14-11ee-291e-5fa1e2346bfa
#using Plots

# ╔═╡ d9d68372-ba7d-4dd8-8071-1eb8e80722ad
function my_function(a)
	x = 0:0.01:10
	y = a.*sin.(x)
	return y, x
end

# ╔═╡ 28253019-1472-4890-bca4-72edf8c4d456
function my_function(a::Int, b::Int)
	a + b
end


# ╔═╡ af256094-577f-440a-ab78-abb25894d495
function my_function(a::String, b::String)
	a * b
end

# ╔═╡ 05aa63ab-d83d-4099-b213-5ecc21deb940
aa, bb = my_function(4.0)

# ╔═╡ 5c129a5b-b836-40c2-a731-be933f37ce95
my_function(2,3)

# ╔═╡ 2f31c2c6-6d26-49e6-b7e8-d735252351d9
my_function("a", "n")

# ╔═╡ Cell order:
# ╠═05381ce2-6a14-11ee-291e-5fa1e2346bfa
# ╠═d9d68372-ba7d-4dd8-8071-1eb8e80722ad
# ╠═28253019-1472-4890-bca4-72edf8c4d456
# ╠═af256094-577f-440a-ab78-abb25894d495
# ╠═05aa63ab-d83d-4099-b213-5ecc21deb940
# ╠═5c129a5b-b836-40c2-a731-be933f37ce95
# ╠═2f31c2c6-6d26-49e6-b7e8-d735252351d9
