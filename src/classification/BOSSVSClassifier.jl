include("/Users/Sam/Documents/SFA_Julia/src/transformation/BOSSVS.jl")
using Stats


"
The Bag-of-SFA-Symbols in Vector Space classifier as published in
 Schäfer, P.: Scalable time series classification. DMKD (2016)
"


mutable struct BOSSVSClassifier
    bossVS::Dict{Symbol, Any}
end

function init_BOSSVSClassifier(self::BOSSVSClassifier, d)
    self.bossVS[:NAME] = d
    self.bossVS[:factor] = 0.95
    self.bossVS[:maxF] = 16
    self.bossVS[:minF] = 4
    self.bossVS[:maxS] = 4
    self.bossVS[:MAX_WINDOW_LENGTH] = 250
    self.bossVS[:folds] = 10
    return self
end

function createFoldIndex(self::BOSSVSClassifier, l, n_folds)
    srand(0)
    fold_index = [0]
    perm = [i for i in 1:l]
    perm = shuffle(perm)

    for i in 1:n_folds
        append!(fold_index, convert(Int64, floor(i * l / n_folds)))
    end
    # fold_index.append(l)
    self.bossVS[:train_indices] = Dict(i=>[] for i in 1:n_folds)
    self.bossVS[:test_indices] = Dict(i=>[] for i in 1:n_folds)
    for i in 1:n_folds
        for j in 1:l
            if (j < fold_index[i]) | (j >= fold_index[i+1])
                append!(self.bossVS[:train_indices][i], perm[j])
            else
                append!(self.bossVS[:test_indices][i], perm[j])
            end
        end
    end
    return self
end

function eval_BOSSVSClassifier(self::BOSSVSClassifier, train, test)
    self = createFoldIndex(self, train[:Samples], self.bossVS[:folds])

    correctTraining = fit(self, train)
    train_acc = correctTraining/train[:Samples]

    correctTesting, labels = prediction(self, self.bossVS[:model], test)
    test_acc = correctTesting/test[:Samples]

    return string("BOSSVS; ",round(train_acc,3),"; ",round(test_acc,3)), labels
end

function fit(self::BOSSVSClassifier, train)
    maxCorrect = -1

    self.bossVS[:minWindowLength] = 10
    maxWindowLength = min(self.bossVS[:MAX_WINDOW_LENGTH], train[:Size])

    count = sqrt(maxWindowLength)
    distance = (maxWindowLength - self.bossVS[:minWindowLength]) / count

    windows = []
    c = self.bossVS[:minWindowLength]
    while c <= maxWindowLength
        append!(windows, convert(Int64, c))
        c += floor(distance)
    end

    for normMean in [true, false]
        model = fitEnsemble(self, windows, normMean, train)
        correct, labels = prediction(self, model, train)

        if maxCorrect <= correct
            maxCorrect = correct
            self.bossVS[:model] = model
        end
    end
    return maxCorrect
end

function fitIndividual(self::BOSSVSClassifier, NormMean, samples, i)
    uniqueLabels = unique(samples[:Labels])
    model = Dict(:window=> i, :normMean=> NormMean, :correctTraining=> 0)
    bossvs = init__bossvs(BOSSVS(Dict()), self.bossVS[:maxF], self.bossVS[:maxS], i, NormMean)
    words = createWords(bossvs, samples)

    f = self.bossVS[:minF]
    keep_going = true
    while (keep_going) & (f <= min(i, self.bossVS[:maxF]))
        bag = createBagOfPattern(bossvs, words, samples, f)

        correct = 0
        for s in 1:self.bossVS[:folds]
            idf = createTfIdf(bossvs, bag, self.bossVS[:train_indices][s], uniqueLabels, samples[:Labels])
            correct += predict(self, self.bossVS[:test_indices][s], bag, idf, samples[:Labels])[1]
        end

        if correct > model[:correctTraining]
            model[:correctTraining] = correct
            model[:f] = f
        end
        if correct == samples[:Samples]
            keep_going = false
        end

        f += 2
    end

    bag = createBagOfPattern(bossvs, words, samples, model[:f])
    model[:idf] = createTfIdf(bossvs, bag, [i for i in 1:samples[:Samples]], uniqueLabels, samples[:Labels])
    model[:bossvs] = bossvs
    append!(self.bossVS[:results], [model])
    return self
end

function fitEnsemble(self::BOSSVSClassifier, windows, normMean, samples)
    correctTraining = 0
    self.bossVS[:results] = []

    println(self.bossVS[:NAME], "  Fitting for a norm of ", normMean)
    for i in windows
        print(i)
        print("; ")
        self = fitIndividual(self, normMean, samples, i)
    end
    println("")

    # Find best correctTraining
    for i in 1:length(self.bossVS[:results])
        if self.bossVS[:results][i][:correctTraining] > correctTraining
            correctTraining = self.bossVS[:results][i][:correctTraining]
        end
    end


    # Remove Results that are no longer satisfactory
    new_results = []
    for i in 1:length(self.bossVS[:results])
        if self.bossVS[:results][i][:correctTraining] >= (correctTraining * self.bossVS[:factor])
            append!(new_results, [self.bossVS[:results][i]])
        end
    end

    return new_results
end

function predict(self::BOSSVSClassifier, indices, bagOfPatternsTestSamples, matrixTrain, labels)
    pred_labels = Any[0 for _ in 1:length(indices)]
    correct = 0

    for x in 1:length(indices)
        i = indices[x]
        bestDistance = 0.
        for key in keys(matrixTrain)
            value = matrixTrain[key]
            label = key
            stat = matrixTrain[key]
            distance = 0.0
            for key2 in keys(bagOfPatternsTestSamples[i])
                value2 = bagOfPatternsTestSamples[i][key2]
                if key2 in keys(stat)
                    Value = stat[key2]
                else
                    Value = 0.
                end
                distance += value2 * (Value + 1.0)
            end

            #No mag normal option

            if distance > bestDistance
                bestDistance = distance
                pred_labels[x] = label
            end
        end

        if pred_labels[x] == labels[i]
            correct += 1
        end
    end

    return correct, pred_labels
end

function prediction(self::BOSSVSClassifier, model, samples)
    uniqueLabels = unique(samples[:Labels])
    pred_labels = convert(Array{Any,2},zeros(samples[:Samples], length(model)))
    pred_vector = convert(Array{Any},zeros(samples[:Samples]))
    indicesTest = [i for i in 1:samples[:Samples]]

    for i in 1:length(model)
        score = model[i]
        bossvs = score[:bossvs]
        wordsTest = createWords(bossvs, samples)
        bagTest = createBagOfPattern(bossvs, wordsTest, samples, score[:f])

        p = predict(self, indicesTest, bagTest, score[:idf], samples[:Labels])

        for j in 1:length(p[2])
            pred_labels[j,i] = p[2][j]
        end
    end

    for i in 1:samples[:Samples]
        m = modes(pred_labels[i,1:end])
        pred_vector[i] = m[1]
    end


    correct = sum([pred_vector[i] == samples[i].ts[:label] for i in 1:samples[:Samples]])
    return correct, pred_labels
end
