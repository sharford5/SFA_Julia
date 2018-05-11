
@everywhere include("/Users/Sam/Documents/SFA_Julia/src/transformation/MUSE.jl")
@everywhere include("/Users/Sam/Documents/SFA_Julia/src/LibLinear/Feature.jl")
@everywhere include("/Users/Sam/Documents/SFA_Julia/src/LibLinear/FeatureNode.jl")
@everywhere include("/Users/Sam/Documents/SFA_Julia/src/LibLinear/Linear.jl")
@everywhere include("/Users/Sam/Documents/SFA_Julia/src/LibLinear/Parameter.jl")
@everywhere include("/Users/Sam/Documents/SFA_Julia/src/LibLinear/Problem.jl")
@everywhere include("/Users/Sam/Documents/SFA_Julia/src/LibLinear/SolverType.jl")


"
The WEASEL+MUSE classifier as published in

 Schäfer, P., Leser, U.: Multivariate Time Series Classification with WEASEL+MUSE. arXiv 2017
 http://arxiv.org/abs/1711.11343
"

mutable struct MUSEClassifier
    mc::Dict{Symbol, Any}
end

function init_MUSEClassifier(self::MUSEClassifier, d)
    self.mc[:NAME] = d
    self.mc[:maxF] = 6
    self.mc[:minF] = 4
    self.mc[:maxS] = 4
    self.mc[:histTypes] = ["EQUI_DEPTH", "EQUI_FREQUENCY"]

    self.mc[:chi] = 2
    self.mc[:bias] = 1
    self.mc[:p] = 0.1
    self.mc[:iter] = 5000
    self.mc[:c] = 1
    self.mc[:MAX_WINDOW_SIZE] = 450
    self.mc[:solverType] = __init__solver(SolverType(Dict()), "L2R_LR_DUAL")
    self.mc[:word_model] = nothing
    self.mc[:linear_model] = nothing
    self.mc[:TIMESERIES_NORM] = false

    return self
end

function eval_MUSEClassifier(self::MUSEClassifier, train, test)
    if !self.mc[:TIMESERIES_NORM]
        for i in 1:train[:Samples]
            for j in 1:train[:Dimensions]
                train[i][j].ts[:NORM_CHECK] = self.mc[:TIMESERIES_NORM]
            end
        end
        for i in 1:test[:Samples]
            for j in 1:test[:Dimensions]
                test[i][j].ts[:NORM_CHECK] = self.mc[:TIMESERIES_NORM]
            end
        end
    end

    self, scores = fit(self, train)
    acc, labels = predict(self, scores, test)

    return string("WEASEL+MUSE; ",round(scores.train_correct/scores.train_size,3),"; ",round(acc,3)), labels
end

function fit(self::MUSEClassifier, trainSamples)
    musemodel = fitMuse(self, trainSamples)
    return musemodel
end

function predict(self::MUSEClassifier, scores, test)
    _, words = createWORDS(self.mc[:word_model], test)
    _, bag = createBagOfPatterns(self.mc[:word_model], words, test, test[:Dimensions], scores.f)
    bag = Remap(self.mc[:word_model], bag)

    features = initLibLinear(self, bag, scores.n_features)

    pred_labels = []
    for f in features
        append!(pred_labels, predict(self.mc[:model], self.mc[:linear_model], f))
    end

    acc = sum([pred_labels[i] == test[i][1].ts[:label] for i in 1:test[:Samples]])/test[:Samples]

    return acc, pred_labels
end

function fitMuse(self::MUSEClassifier, samples)
    dimensionality = samples[:Dimensions]

    maxCorrect = -1
    bestF = -1
    bestNorm = false
    bestHistType = nothing

    min = 4
    Max = GetMax(self, samples, self.mc[:MAX_WINDOW_SIZE])

    self.mc[:windowLengths] = [a for a in min:Max]

    breaker = false
    for histType in self.mc[:histTypes]
        for normMean in [true, false]
            model = init__m(MUSE(Dict()), self.mc[:maxF], self.mc[:maxS], histType, self.mc[:windowLengths], normMean, true)
            model, words = createWORDS(model, samples)

            f = self.mc[:minF]
            while f <= self.mc[:maxF]
                model, bag = createBagOfPatterns(model, words, samples, dimensionality, f)
                model, bag = filterChiSquared(model, bag, self.mc[:chi])

                problem = initLibLinearProblem(self, bag, model.m[:dict], self.mc[:bias])
                correct = trainLibLinear(self, problem, 10)
                # println(correct)
                if correct > maxCorrect
                    maxCorrect = correct
                    bestF = f
                    bestNorm = normMean
                    bestHistType = histType
                end

                if correct == samples[:Samples]
                    breaker = true
                    break
                end

                f += 2
            end
            if breaker
                break
            end
        end
        if breaker
            break
        end
    end

    self.mc[:word_model] = init__m(MUSE(Dict()), bestF, self.mc[:maxS], bestHistType, self.mc[:windowLengths], bestNorm, true)
    self.mc[:word_model], words = createWORDS(self.mc[:word_model], samples)
    self.mc[:word_model], bag = createBagOfPatterns(self.mc[:word_model], words, samples, dimensionality, bestF)
    self.mc[:word_model], bag = filterChiSquared(self.mc[:word_model], bag, self.mc[:chi])
    problem = initLibLinearProblem(self, bag, self.mc[:word_model].m[:dict], self.mc[:bias])
    param = init__para(Parameter(Dict()), self.mc[:solverType], self.mc[:c], self.mc[:iter], self.mc[:p])
    self.mc[:model] = Linear(Dict())
    self.mc[:linear_model] = train_linear(self.mc[:model], problem, param)

    return self, MUSEMODEL(bestNorm, bestHistType, bestF, maxCorrect, samples[:Samples], problem.prob[:n])
end

function GetMax(self, samples, number)
    m = length(samples[1][1].ts[:data])
    for i in 1:samples[:Samples]
        for j in 1:length(keys(samples[i]))
            m = max(m, length(samples[i][j].ts[:data]))
        end
    end
    return min(m, number)
end

function initLibLinearProblem(self::MUSEClassifier, bob, dict, bias)
    problem = init__prob(Problem(Dict()))
    problem.prob[:bias] = bias
    problem.prob[:n] = SIZE(dict)
    problem.prob[:y] = [bob[j].bob[:label] for j in 1:length(bob)]

    features = initLibLinear(self, bob, problem.prob[:n])

    problem.prob[:l] = length(features)
    problem.prob[:x] = features
    return problem
end

function initLibLinear(self::MUSEClassifier, bob, max_feature)
    featuresTrain = Any[nothing for _ in 1:length(bob)]
    for j in 1:length(bob)
        features = []
        bop = bob[j]
        try
            max_key = maximum(keys(bop.bob[:bob]))
            for word_key in 1:max_key
                try
                    word_value = bop.bob[:bob][word_key]
                    if (word_value > 0) & (word_key <= max_feature)
                        append!(features, [FeatureNode(word_key, word_value)])
                    end
                end
            end
        end

        featuresTrain[j] = features
    end
    return featuresTrain
end

function trainLibLinear(self::MUSEClassifier, prob, n_folds = 10)
    param = init__para(Parameter(Dict()), self.mc[:solverType], self.mc[:c], self.mc[:iter], self.mc[:p])
    srand(1)
    l = prob.prob[:l]

    if n_folds > l
        n_folds = l
    end

    fold_start = [0]
    perm = [i for i in 1:l]
    perm = shuffle(perm)

    for i in 1:n_folds
        append!(fold_start, convert(Int64, floor(i*l/n_folds)))
    end

    # fold_start.append(l)
    correct = 0

    ## 10 fold cross validation of training set
    for i in 1:n_folds
        model = Linear(Dict())
        b = fold_start[i]
        e = fold_start[i + 1]+1

        subprob = init__prob(Problem(Dict()))
        subprob.prob[:bias] = prob.prob[:bias]
        subprob.prob[:n] = prob.prob[:n]
        subprob.prob[:l] = l - (e - b) + 1
        subprob.prob[:y] = []

        rows = []
        for j in 1:b
            append!(rows, perm[j])
            append!(subprob.prob[:y], prob.prob[:y][perm[j]])
        end

        for j in e:l
            append!(rows, perm[j])
            append!(subprob.prob[:y], prob.prob[:y][perm[j]])
        end

        subprob.prob[:x] = [prob.prob[:x][j] for j in rows]
        fold_model = train_linear(model, subprob, param)

        fold_x = []
        fold_y = []
        for u in (b+1):(e-1)
            append!(fold_x, [prob.prob[:x][perm[u]]])
            append!(fold_y, prob.prob[:y][perm[u]])
        end

        fold_labels = []
        for h in 1:length(fold_y)
            append!(fold_labels, predict(model, fold_model, fold_x[h]))
        end

        for u in 1:length(fold_y)
            if fold_y[u] == fold_labels[u]
                correct += 1
            end
        end
    end

    return correct
end


mutable struct MUSEMODEL
    norm::Bool
    histType::Any
    f::Int64
    train_correct::Int64
    train_size::Int64
    n_features::Int64
end
