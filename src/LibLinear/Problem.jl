
mutable struct Problem
    prob::Dict{Symbol, Any}
end

function init__prob(self::Problem)
    self.prob[:l] = nothing
    self.prob[:n] = nothing
    self.prob[:y] = nothing
    self.prob[:x] = nothing
    self.prob[:bias] = 0.

    return self
end
