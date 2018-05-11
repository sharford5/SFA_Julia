include("/Users/Sam/Documents/SFA_Julia/src/transformation/BOSS.jl")



"
The Bag-of-SFA-Symbols Ensemble Classifier as published in
 Schäfer, P.: The boss is concerned with time series classification
 in the presence of noise. DMKD (2015)
"


mutable struct BOSSEnsembleClassifier
    boss::Dict{Symbol, Any}
end

function init_BOSSEnsembleClassifier(self::BOSSEnsembleClassifier, d)
    self.boss[:NAME] = d
    self.boss[:factor] = 0.96
    self.boss[:maxF] = 16
    self.boss[:minF] = 6
    self.boss[:maxS] = 4
    self.boss[:MAX_WINDOW_LENGTH] = 250
    return self
end

function eval_BOSSEnsembleClassifier(self::BOSSEnsembleClassifier, train, test)
    self, scores = fit(self, train)
    self, labels, correctTesting = predict(self, self.boss[:model], test)
    test_acc = correctTesting/test[:Samples]

    return string("BOSS Ensemble; ",round(scores,3),"; ",round(test_acc,3)), labels
end

function fit(self::BOSSEnsembleClassifier, train)
    self.boss[:minWindowLength] = 10
    maxWindowLength = self.boss[:MAX_WINDOW_LENGTH]
    for i in 1:train[:Samples]
        maxWindowLength = min(length(train[i].ts[:data])-1, maxWindowLength)
    end

    self.boss[:windows] = [i for i in maxWindowLength:-1:self.boss[:minWindowLength]]

    NORMALIZATION = [true, false]
    bestCorrectTraining = 0.
    bestScore = nothing

    for norm in NORMALIZATION
        self, models, correctTraining = fitEnsemble(self, norm, train)
        self, labels, correctTesting = predict(self, models, train)

        if bestCorrectTraining < correctTesting
            bestCorrectTraining = correctTesting
            bestScore = correctTesting/train[:Samples]
            self.boss[:model] = models
        end
    end

    return self, bestScore
end

function fitIndividual(self::BOSSEnsembleClassifier, NormMean, samples, i)
    model = BOSSModel(self, NormMean, i)
    boss = init__boss(BOSS(Dict()), self.boss[:maxF], self.boss[:maxS], i, NormMean)

    train_words = createWords(boss, samples)
    f = self.boss[:minF]
    keep_going = true
    while (f <= self.boss[:maxF]) & (keep_going == true)
        bag = createBagOfPattern(boss, train_words, samples, f)
        s = prediction(self, bag, bag, samples[:Labels], samples[:Labels], false)
        # println(i, s)

        if s[1] > model[2]
            model[2] = s[1]
            model[3] = f
            model[4] = boss
            model[6] = bag
            model[7] = samples[:Labels]
        end
        if s[1] == samples[:Samples]
            keep_going = false
        end
        f += 2
    end

    append!(self.boss[:results], [model])
    return self
end

function fitEnsemble(self::BOSSEnsembleClassifier, NormMean, samples)
    correctTraining = 0
    self.boss[:results] = []

    println(self.boss[:NAME], "  Fitting for a norm of ", NormMean)
    # Parallel(n_jobs=3, backend="threading")(delayed(self.fitIndividual, check_pickle=False)(NormMean, samples, i, bar) for i in range(len(self.windows)))
    for i in self.boss[:windows]
        print(i)
        print("; ")
        self = fitIndividual(self, NormMean, samples, i)
    end
    println("")

    #Find best correctTraining
    for i in 1:length(self.boss[:results])
        if self.boss[:results][i][2] > correctTraining
            correctTraining = self.boss[:results][i][2]
        end
    end

    # Remove Results that are no longer satisfactory
    new_results = []
    for i in 1:length(self.boss[:results])
        if self.boss[:results][i][2] >= (correctTraining * self.boss[:factor])
            append!(new_results, [self.boss[:results][i]])
        end
    end

    return self, new_results, correctTraining
end

function BossScore(self::BOSSEnsembleClassifier, normed, windowLength)
    return ["BOSS Ensemble", 0, 0, normed, windowLength, zeros(), 0]
end

function BOSSModel(self::BOSSEnsembleClassifier, normed, windowLength)
    return BossScore(self, normed, windowLength)
end

function prediction(self::BOSSEnsembleClassifier, bag_test, bag_train, label_test, label_train, training_check)
    p_labels = Any[0 for i in 1:length(label_test)]
    p_correct = 0

    for i in 1:length(bag_test)
        minDistance = 2147483647
        p_labels[i] = "Nan"

        noMatchDistance = 0
        for key in keys(bag_test[i])
            noMatchDistance += bag_test[i][key]^2
        end

        for j in 1:length(bag_train)
            if (bag_train[j] != bag_test[i]) | (training_check)
                distance = 0
                for key in keys(bag_test[i])
                    if key in keys(bag_train[j])
                        buf = bag_test[i][key] - bag_train[j][key]
                    else
                        buf = bag_test[i][key]
                    end
                    distance += buf^2

                    if distance >= minDistance
                        continue
                    end
                end

                if (distance != noMatchDistance) & (distance < minDistance)
                    minDistance = distance
                    p_labels[i] = label_train[j]
                end
            end
        end

        if label_test[i] == p_labels[i]
            p_correct += 1
        end
    end

    return p_correct, p_labels
end

function predict(self::BOSSEnsembleClassifier, models, samples)
    Label_Matrix = convert(Array{Any,2},zeros(samples[:Samples], length(models)))
    Label_Vector = convert(Array{Any},zeros(samples[:Samples]))

    for i in 1:length(models)
        model = models[i]
        wordsTest = createWords(model[4], samples)

        test_bag = createBagOfPattern(model[4], wordsTest, samples, model[3])
        p_correct, p_labels = prediction(self, test_bag, model[6], samples[:Labels], model[7], true)

        for j in 1:length(p_labels)
            Label_Matrix[j, i] = p_labels[j]
        end
    end

    unique_labels = unique(samples[:Labels])

    for i in 1:length(Label_Vector)
        maximum = 0
        best = 0
        d = Label_Matrix[i, :]
        for key in unique_labels
            if sum(d .== key) > maximum
                maximum = sum(d .== key)
                best = key
            end
        end
        Label_Vector[i] = best
    end

    correctTesting = 0
    for i in 1:length(Label_Vector)
        if convert(Int64, Label_Vector[i]) == convert(Int64, samples[:Labels][i])
            correctTesting += 1
        end
    end

    return self, Label_Vector, correctTesting
end
