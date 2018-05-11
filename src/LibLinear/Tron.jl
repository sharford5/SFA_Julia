
mutable struct Tron
    tron::Dict{Symbol, Any}
end

function init__tron(self::Tron)
    self.tron[:fun_obj] = nothing
    self.tron[:eps] = nothing
    self.tron[:max_iter] = nothing

    return self
end
