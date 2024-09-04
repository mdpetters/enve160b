
begin
		# Load data
		df = DataFrame(CSV.File(download("https://github.com/mattiasvillani/Regression/raw/master/Data/healthdata.csv");header=true));
		df.spending = df.spending;

		# Define Squared Error function
		function SSE(y,X,β)
			return (y-X*β)'*(y-X*β)
		end

		square(w, x, y) = Shape(x .+ [0,w,w,0], y .+ [0,0,w,w])

		nothing
end


begin
	l = open("../data/auto.txt", "r") do f
		readlines(f)
	end
	function splitit(x)
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
	l[1]
	split(l[1], "\t")
	df1 = mapfoldl(splitit, vcat, l)
end