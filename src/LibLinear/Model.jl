

mutable struct Model
    m::Dict{Symbol, Any}
end

function init__m(self::Model)
    self.m[:serialVersionUID] = -6456047576741854834
    self.m[:bias] = nothing
    self.m[:label] = nothing
    self.m[:nr_class] = 0
    self.m[:nr_feature] = 0
    self.m[:solverType] = nothing
    self.m[:w] = nothing
    return self
end
