global solverdict = Dict("L2R_LR"=>Any[0, true, false],
                    "L2R_L2LOSS_SVC_DUAL"=>Any[1, false, false],
                    "L2R_L2LOSS_SVC"=>Any[2, false, false],
                    "L2R_L1LOSS_SVC_DUAL"=>Any[3, false, false],
                    "MCSVM_CS"=>Any[4, false, false],
                    "L1R_L2LOSS_SVC"=>Any[5, false, false],
                    "L1R_LR"=>Any[6, true, false],
                    "L2R_LR_DUAL"=>Any[7, true, false],
                    "L2R_L2LOSS_SVR"=>Any[11, false, true],
                    "L2R_L2LOSS_SVR_DUAL"=>Any[12, false, true],
                    "L2R_L1LOSS_SVR_DUAL"=>Any[13, false, true])

mutable struct SolverType
    st::Dict{Symbol, Any}
end
function __init__solver(self::SolverType, string)
    self.st[:solvertype] = string
    self.st[:id], self.st[:logisticRegressionSolver], self.st[:supportVectorRegression] = solverdict[string]
    return self
end

function isLogisticRegressionSolver(self::SolverType)
    return self.st[:logisticRegressionSolver]
end

function isSupportVectorRegression(self::SolverType)
    return self.st[:supportVectorRegression]
end
