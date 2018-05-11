include("/Users/Sam/Documents/SFA_Julia/src/LibLinear/Model.jl")
include("/Users/Sam/Documents/SFA_Julia/src/LibLinear/Problem.jl")
include("/Users/Sam/Documents/SFA_Julia/src/LibLinear/Tron.jl")


mutable struct Linear
    lin::Dict{Symbol, Any}
end


function train_linear(self::Linear, prob, param)
    var2 = prob.prob[:x]
    n = length(var2)

    for w_size in 1:n
        nodes = var2[w_size]
        indexBefore = 0
        var7 = nodes
        nr_class = length(nodes)
        for var9 in 1:nr_class
            n = var7[var9]
            if getIndex(n) <= indexBefore
                return "feature nodes must be sorted by index in ascending order"
            end
            indexBefore = getIndex(n)
        end
    end

    l = prob.prob[:l]
    n = prob.prob[:n]
    w_size = prob.prob[:n]
    model = Model(Dict())
    if prob.prob[:bias] >= 0.0
        model.m[:nr_feature] = n - 1
    else
        model.m[:nr_feature] = n
    end

    model.m[:solverType] = param.para[:solverType]
    model.m[:bias] = prob.prob[:bias]

    if isSupportVectorRegression(param.para[:solverType])
        model.m[:w] = Any[nothing for _ in 1:w_size]
        model.m[:nr_class] = 2
        model.m[:label] = nothing
        return "Not Set for Regression"
        # checkProblemSize(n, model.nr_class);
        # train_one(prob, param, model.w, 0.0D, 0.0D);
    else
        perm = [0 for _ in 1:l]
        perm, rv = groupClasses(self, prob, perm)
        nr_class = rv.gcr[:nr_class]
        label = rv.gcr[:label]
        start = rv.gcr[:start]
        count = rv.gcr[:count]
        # checkProblemSize(n, nr_class);
        model.m[:nr_class] = nr_class
        model.m[:label] = label
        weighted_C = [param.para[:c] for _ in 1:nr_class]

        #Removed part with param weights
        x = [prob.prob[:x][perm[j]] for j in 1:length(perm)]

        subprob = init__prob(Problem(Dict()))
        subprob.prob[:n] = n
        subprob.prob[:l] = l
        subprob.prob[:x] = [x[u] for u in 1:l]
        subprob.prob[:y] = [0.0 for _ in 1:l]

        if param.para[:solverType].st[:solvertype] == "MCSVM_CS"
            model.m[:w] = [0.0 for _ in 1:convert(Int64, n * nr_class)]
            for i in 1:nr_class
                i = start[i]
                while i < start[i] + count[i]
                    subprob.prob[:y][i] = i
                    i += 1
                end
            end
            #TODO Not relevant for me now
            # SolverMCSVM_CS solver = new SolverMCSVM_CS(subprob, nr_class, weighted_C, param.eps);
            # solver.solve(model.w);
        elseif nr_class == 2
            model.m[:w] = [0. for _ in 1:w_size]
            i = start[1] + count[1]# -1

            for i in 1:i
                subprob.prob[:y][i] = 1.0
            end

            # i += 1
            while i <= subprob.prob[:l]
                subprob.prob[:y][i] = -1.0
                i += 1
            end
            w = train_one(self, subprob, param, model.m[:w], weighted_C[1], weighted_C[2])
        else
            model.m[:w] = [0. for _ in 1:convert(Int64, w_size * nr_class)]
            w = [0. for _ in 1:w_size]

            for i in 1:nr_class
                si = start[i]
                ei = si + count[i]

                K = 1
                for _ in 1:si
                    subprob.prob[:y][K] = -1.0
                    K += 1
                end

                while K < ei
                    subprob.prob[:y][K] = 1.0
                    K += 1
                end

                while K < subprob.prob[:l]
                    subprob.prob[:y][K] = -1.0
                    K += 1
                end

                w = train_one(self, subprob, param, w, weighted_C[i], param.para[:c])

                for j in 1:n
                    model.m[:w][(j-1) * nr_class + i] = w[j]
                end
            end
        end
    end
    return model
end

function train_one(self, prob, param, w, Cp, Cn)
    eps = param.para[:eps]
    pos = 0

    for i in 1:prob.prob[:l]
        if prob.prob[:y][i] > 0.0
            pos += 1
        end
    end

    i = prob.prob[:l] - pos
    primal_solver_tol = eps * max(min(pos, i), 1) / prob.prob[:l]
    fun_obj = nothing
    C = 0.0
    i = 0.0
    prob_col = Problem(Dict())
    tron_obj = init__tron(Tron(Dict()))

    if param.para[:solverType].st[:solvertype] == "L2R_LR"
        C = [0. for _ in 1:prob.prob[:l]]

        for i in 1:prob.prob[:l]
            if prob.prob[:y][i] > 0.0
                C[i] = Cp
            else
                C[i] = Cn
            end
        end

        # fun_obj = L2R_LrFunction(prob, C)
        # tron_obj = new Tron(fun_obj, primal_solver_tol);
        # tron_obj.tron(w);
        # break;
    # elif param.solverType.solvertype == 'L2R_L2LOSS_SVC':
        # C = [0. for _ in range(prob.l)]
        #
        # for(i = 0; i < prob.l; ++i) {
        #     if (prob.y[i] > 0.0D) {
        #         C[i] = Cp;
        #     } else {
        #         C[i] = Cn;
        #     }
        # }
        #
        # Function fun_obj = new L2R_L2_SvcFunction(prob, C);
        # tron_obj = new Tron(fun_obj, primal_solver_tol);
        # tron_obj.tron(w);
    # elif param.solverType.solvertype == 'L2R_L2LOSS_SVC_DUAL':
        # solve_l2r_l1l2_svc(prob, w, eps, Cp, Cn, SolverType.L2R_L2LOSS_SVC_DUAL);
    # elif param.solverType.solvertype == 'L2R_L1LOSS_SVC_DUAL':
        # solve_l2r_l1l2_svc(prob, w, eps, Cp, Cn, SolverType.L2R_L1LOSS_SVC_DUAL);
    # elif param.solverType.solvertype == 'L1R_L2LOSS_SVC':
        # prob_col = transpose(prob);
        # solve_l1r_l2_svc(prob_col, w, primal_solver_tol, Cp, Cn);
    # elif param.solverType.solvertype == 'L1R_LR':
        # prob_col = transpose(prob);
        # solve_l1r_lr(prob_col, w, primal_solver_tol, Cp, Cn);
    elseif param.para[:solverType].st[:solvertype] == "L2R_LR_DUAL"   #TODO Only one that works
        w2 = solve_l2r_lr_dual(self, prob, w, eps, Cp, Cn)
    # elif param.solverType.solvertype == 'L2R_L2LOSS_SVR':
        # C = new double[prob.l];
        #
        # for(i = 0; i < prob.l; ++i) {
        #     C[i] = param.C;
        # }
        #
        # fun_obj = new L2R_L2_SvrFunction(prob, C, param.p);
        # tron_obj = new Tron(fun_obj, param.eps);
        # tron_obj.tron(w);
    # elif param.solverType.solvertype == 'L2R_L1LOSS_SVR_DUAL':
    # elif param.solverType.solvertype == 'L2R_L2LOSS_SVR_DUAL':
        # solve_l2r_l1l2_svr(prob, w, param);
    else
        print("unknown solver type: ",  param.para[:solverType].st[:solvertype])
    end

    return w2
end

function groupClasses(self::Linear, prob, perm)
    l = prob.prob[:l]
    label = []
    for this in prob.prob[:y]
        if this in label
        else
            append!(label, this)
        end
    end
    count = [0 for _ in 1:length(label)]
    data_label = [findin(label, lab)[1] for lab in prob.prob[:y]]
    nr_class = length(label)

    for lab in prob.prob[:y]
        count[findin(label, lab)] += 1
    end

    if (nr_class == 2) & (label[1] == -1) & (label[2] == 1)
        label = swap(self, label, 1, 2)
        count = swap(self, count, 1, 2)

        for i in 1:l
            if data_label[i] == 1
                data_label[i] = 2
            else
                data_label[i] = 1
            end
        end
    end

    start = [1 for _ in 1:nr_class]
    # start[1] = 0

    for i in 2:nr_class
        start[i] = start[i - 1] + count[i - 1]# - 1
    end

    for i in 1:l
        perm[start[data_label[i]]] = i
        start[data_label[i]] += 1
    end

    start[1] = 1

    for i in 2:nr_class
        start[i] = start[i - 1] + count[i - 1]# - 1
    end
    # start[1] = 1

    return perm, init__g(GroupClassesReturn(Dict()), nr_class, label, start, count)
end

# def copyOf(self, original, newLength):
#     copy = [None for _ in range(newLength)]
#     copy = self.arrayCopy(original, 0, copy, 0, min(original.length, newLength))
#     return copy
#
#
# def arrayCopy(self, src, srcPos, dest, destPos, length):
#     for i in range(length):
#         dest[i + destPos] = src[i + srcPos]
#     return dest
#

function swap(self, array, idxA, idxB)
    temp = array[idxA]
    array[idxA] = array[idxB]
    array[idxB] = temp
    return array
end

function solve_l2r_lr_dual(self, prob, w, eps, Cp, Cn)
    l = prob.prob[:l]
    w_size = prob.prob[:n]
    iter = 0
    xTx = [0. for _ in 1:l]
    max_iter = 1000
    index = [0 for _ in 1:l]
    alpha = [0. for _ in 1:(2 * l)]
    y = [0 for _ in 1:l]
    max_inner_iter = 100
    innereps = 0.01
    innereps_min = min(1.0E-8, eps)
    upper_bound = [Cn, 0.0, Cp]

    for i in 1:l
        if prob.prob[:y][i] > 0.
            y[i] = 2 # 1
        else
            y[i] = 0 # -1
        end
    end

    for i in 1:l
        alpha[(2 * i) - 1] = min(0.001 * upper_bound[y[i]+1], 1.0E-8)
        alpha[2 * i] = upper_bound[y[i]+1] - alpha[(2 * i) - 1]
    end

    for i in 1:w_size
        w[i] = 0.0
    end

    var10001 = 0.
    C = 0.
    i = 1
    for i in 1:l
        xTx[i] = 0.0
        var24 = prob.prob[:x][i]
        var25 = length(var24)
        for var26 in 1:var25
            xi = var24[var26]
            C = xi.value
            xTx[i] += C * C
            var10001 = xi.index # - 1
            w[var10001] += (y[i]-1) * alpha[2 * i - 1] * C
        end
        index[i] += i
    end

    while iter < max_iter
        newton_iter = 0
        for i in 1:(l-1)
            r = rand(1:l-i)
            newton_iter = i + r
            # index = swap(self, index, i, newton_iter)
        end

        newton_iter = 0
        Gmax = 0.0

        for s in 1:l
            i = index[s]
            yi = y[i] - 1
            C = upper_bound[y[i] + 1]
            ywTx = 0.0
            xisq = xTx[i]
            var34 = prob.prob[:x][i]
            var35 = length(var34)

            for var36 in 1:var35
                xi = var34[var36]
                ywTx += w[xi.index] * xi.value
            end
            ywTx *= y[i] - 1
            a = xisq
            b = ywTx
            ind1 = (2 * i) - 1
            ind2 = 2 * i #+ 1
            sign = 1

            condition = 0.5 * xisq * (alpha[ind2] - alpha[ind1]) + ywTx

            if condition < 0.0
                ind1 = 2 * i
                ind2 = 2 * i - 1
                sign = -1
            end

            alpha_old = alpha[ind1]
            z = alpha_old
            if C - alpha_old < 0.5 * C
                z = 0.1 * alpha_old
            end

            gp = xisq * (z - alpha_old) + sign * ywTx + log(z / (C - z))
            Gmax = max(Gmax, abs(gp))
            eta = 0.1

            inner_iter = 0
            while (inner_iter <= max_inner_iter) & (abs(gp) >= innereps)
                gpp = a + C / (C - z) / z
                tmpz = z - gp / gpp
                if tmpz <= 0.0
                    z *= 0.1
                else
                    z = tmpz
                end

                gp = a * (z - alpha_old) + sign * b + log(z / (C - z))
                newton_iter += 1
                inner_iter += 1
            end

            if inner_iter > 0
                alpha[ind1] = z
                alpha[ind2] = C - z
                var60 = prob.prob[:x][i]
                var51 = length(var60)

                for var61 in 1:var51
                    xi = var60[var61]
                    var10001 = xi.index #- 1
                    w[var10001] += sign * (z - alpha_old) * yi * xi.value
                end
            end
        end

        iter += 1

        if Gmax < eps
            break
        end

        if newton_iter <= l / 10
            innereps = max(innereps_min, 0.1 * innereps)
        end
    end

    #Ordering of W is different that JAVA
    v = 0.0

    for i in 1:w_size
        v += w[i] * w[i]
    end

    v *= 0.5

    for i in 1:l
        v += alpha[2 * i] * log(alpha[2 * i - 1]) + alpha[2 * i] * log(alpha[2 * i]) - upper_bound[y[i]+1] * log(upper_bound[y[i] + 1])
    end

    return w
end

function predict(self::Linear, model, x)
    dec_values = [0. for _  in 1:model.m[:nr_class]]
    return predictValues(self, model, x, dec_values)
end

function predictValues(self, model, x, dec_values)
    n = 0
    if model.m[:bias] >= 0.0
        n = model.m[:nr_feature] + 1
    else
        n = model.m[:nr_feature]
    end
    w = model.m[:w]
    nr_w = 0
    if (model.m[:nr_class] == 2) & (model.m[:solverType].st[:solvertype] != "MCSVM_CS")
        nr_w = 1
    else
        nr_w = model.m[:nr_class]
    end

    for dec_max_idx in 1:nr_w
        dec_values[dec_max_idx] = 0.0
    end

    var12 = x
    i = length(x)

    for var8 in 1:i
        lx = var12[var8]
        idx = lx.index
        if idx <= n
            for i in 1:nr_w
                dec_values[i] += (w[(idx-1) * nr_w + i] * lx.value)
            end
        end
    end

    if model.m[:nr_class] == 2
        if isSupportVectorRegression(model.m[:solverType])
            return dec_values[1]
        else
            if dec_values[1] > 0.0
                return model.m[:label][1]
            else
                return model.m[:label][2]
            end
        end
    else
        dec_max_idx = 1

        for i in 2:model.m[:nr_class]
            if dec_values[i] > dec_values[dec_max_idx]
                dec_max_idx = i
            end
        end

        return model.m[:label][dec_max_idx]
    end
end

mutable struct GroupClassesReturn
    gcr::Dict{Symbol, Any}
end

function init__g(self::GroupClassesReturn, nr_class, label, start, count)
    self.gcr[:nr_class] = nr_class
    self.gcr[:label] = label
    self.gcr[:start] = start
    self.gcr[:count] = count
    return self
end
