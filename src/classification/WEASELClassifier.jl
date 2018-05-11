include("/Users/Sam/Documents/SFA_Julia/src/transformation/WEASEL.jl")
include("/Users/Sam/Documents/SFA_Julia/src/LibLinear/Feature.jl")
include("/Users/Sam/Documents/SFA_Julia/src/LibLinear/FeatureNode.jl")
include("/Users/Sam/Documents/SFA_Julia/src/LibLinear/Linear.jl")
include("/Users/Sam/Documents/SFA_Julia/src/LibLinear/Parameter.jl")
include("/Users/Sam/Documents/SFA_Julia/src/LibLinear/Problem.jl")
include("/Users/Sam/Documents/SFA_Julia/src/LibLinear/SolverType.jl")

"
The WEASEL (Word ExtrAction for time SEries cLassification) classifier as published in
 Schäfer, P., Leser, U.: Fast and Accurate Time Series
 Classification with WEASEL. CIKM 2017
"


mutable struct WEASELClassifier
    weasel::Dict{Symbol, Any}
end

function init_WEASELClassifier(self::WEASELClassifier, d)
    self.weasel[:NAME] = d
    self.weasel[:maxF] = 6
    self.weasel[:minF] = 4
    self.weasel[:maxS] = 4
    self.weasel[:chi] = 2
    self.weasel[:bias] = 1
    self.weasel[:p] = 0.1
    self.weasel[:iter] = 5000
    self.weasel[:c] = 1
    self.weasel[:MAX_WINDOW_LENGTH] = 250
    self.weasel[:solverType] = __init__solver(SolverType(Dict()), "L2R_LR_DUAL")
    self.weasel[:word_model] = nothing
    self.weasel[:linear_model] = nothing

    return self
end

function eval_WEASELClassifier(self::WEASELClassifier, train, test)
    self, scores = fitWeasel(self, train)
    self, acc, labels = predict(self, scores, test)

    return string("WEASEL; ",round(scores.train_correct/scores.train_size,3),"; ",round(acc,3)), labels
end

function fitWeasel(self::WEASELClassifier, train)
    maxCorrect = -1
    bestF = -1
    bestNorm = false

    self.weasel[:minWindowLength] = 4
    maxWindowLength = self.weasel[:MAX_WINDOW_LENGTH]
    for i in 1:train[:Samples]
        maxWindowLength = min(length(train[i].ts[:data]), maxWindowLength)
    end
    self.weasel[:windows] = [i for i in self.weasel[:minWindowLength]:maxWindowLength]

    keep_going = true
    for normMean in [true, false]
        if keep_going
            model = init__w(WEASEL(Dict()), self.weasel[:maxF], self.weasel[:maxS], self.weasel[:windows], normMean)
            model, words = createWORDS(model, train)

            f = self.weasel[:minF]
            while (f <= self.weasel[:maxF]) & (keep_going == true)
                model = reset(model)
                model, bop = createBagOfPatterns(model, words, train, f)
                model, bop = filterChiSquared(model, bop, self.weasel[:chi])

                problem = initLibLinearProblem(self, bop, model.w[:dict], self.weasel[:bias])
                correct = trainLibLinear(self, problem, 10)

                if correct > maxCorrect
                    maxCorrect = correct
                    bestF = f
                    bestNorm = normMean
                end
                if correct == train[:Samples]
                    keep_going = false
                end
                f += 2
            end
        end
    end

    self.weasel[:word_model] = init__w(WEASEL(Dict()), self.weasel[:maxF], self.weasel[:maxS], self.weasel[:windows], bestNorm)
    self.weasel[:word_model], words = createWORDS(self.weasel[:word_model], train)
    self.weasel[:word_model], bop = createBagOfPatterns(self.weasel[:word_model], words, train, bestF)
    self.weasel[:word_model], bop = filterChiSquared(self.weasel[:word_model], bop, self.weasel[:chi])
    problem = initLibLinearProblem(self, bop, self.weasel[:word_model].w[:dict], self.weasel[:bias])

    param = init__para(Parameter(Dict()), self.weasel[:solverType], self.weasel[:c], self.weasel[:iter], self.weasel[:p])

    self.weasel[:model] = Linear(Dict())
    self.weasel[:linear_model] = train_linear(self.weasel[:model], problem, param)

    return self, WEASELMODEL(bestNorm, bestF, maxCorrect, train[:Samples], problem.prob[:n])
end

function predict(self::WEASELClassifier, scores, test)
    _, words = createWORDS(self.weasel[:word_model], test)
    _, bag = createBagOfPatterns(self.weasel[:word_model], words, test, scores.f)
    bag = Remap(self.weasel[:word_model], bag)

    features = initLibLinear(self, bag, scores.n_features)

    pred_labels = []
    for f in features
        append!(pred_labels, predict(self.weasel[:model], self.weasel[:linear_model], f))
    end

    acc = sum([pred_labels[i] == test[i].ts[:label] for i in 1:test[:Samples]])/test[:Samples]

    return self, acc, pred_labels
end

function initLibLinearProblem(self::WEASELClassifier, bob, dict, bias)
    problem = init__prob(Problem(Dict()))
    problem.prob[:bias] = bias
    problem.prob[:n] = SIZE(dict)
    problem.prob[:y] = [bob[j].bob[:label] for j in 1:length(bob)]

    features = initLibLinear(self, bob, problem.prob[:n])

    problem.prob[:l] = length(features)
    problem.prob[:x] = features
    return problem
end

function initLibLinear(self::WEASELClassifier, bob, max_feature)
    featuresTrain = Any[nothing for _ in 1:length(bob)]
    for j in 1:length(bob)
        features = []
        bop = bob[j]
        max_key = maximum(keys(bop.bob[:bob]))
        for word_key in 1:max_key
            try
                word_value = bop.bob[:bob][word_key]
                if (word_value > 0) & (word_key <= max_feature)
                    append!(features, [FeatureNode(word_key, word_value)])
                end
            end
        end

        featuresTrain[j] = features
    end
    return featuresTrain
end

function trainLibLinear(self::WEASELClassifier, prob, n_folds = 10)
    param = init__para(Parameter(Dict()), self.weasel[:solverType], self.weasel[:c], self.weasel[:iter], self.weasel[:p])
    srand(1234)
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


mutable struct WEASELMODEL
    norm::Bool
    f::Int64
    train_correct::Int64
    train_size::Int64
    n_features::Int64
end
