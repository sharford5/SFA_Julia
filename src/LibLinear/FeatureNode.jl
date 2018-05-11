

mutable struct FeatureNode
    index::Int64
    value::Float64
end

function getIndex(self::FeatureNode)
    return self.index
end

function getValue(self::FeatureNode)
    return self.value
end

function setValue(self::FeatureNode, value)
    self.value = value
end
