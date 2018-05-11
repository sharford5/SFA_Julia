mutable struct Parameter
    para::Dict{Symbol, Any}
end

function init__para(self::Parameter, solverType, c, iter, p)
    self.para[:solverType] = solverType
    self.para[:c] = c
    self.para[:iter] = iter
    self.para[:eps] = p

    return self
end
