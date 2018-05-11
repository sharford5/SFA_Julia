include("/Users/Sam/Documents/SFA_Julia/src/classification/ShotgunClassifier.jl")


"
The Shotgun Ensemble Classifier as published in:
 Schäfer, P.: Towards time series classification without human preprocessing.
 In Machine Learning and Data Mining in Pattern Recognition,
 pages 228–242. Springer, 2014.
"

mutable struct ShotgunEnsembleClassifier
    sec::Dict{Symbol, Any}
end

function init_ShotgunEnsembleClassifier(self::ShotgunEnsembleClassifier, d)
    self.sec[:NAME] = d
    self.sec[:factor] = 0.92
    self.sec[:MAX_WINDOW_LENGTH] = 250

    return self
end

function eval_ShotgunEnsembleClassifier(self::ShotgunEnsembleClassifier, train, test)
    self, correctTraining = fit(self, train)
    train_acc = correctTraining / train[:Samples]

    correctTesting, labels = predictEnsemble(self, self.sec[:model], test)
    test_acc = correctTesting / test[:Samples]

    return string("Shotgun Ensemble; ",round(train_acc,3),"; ",round(test_acc,3)), labels
end

function fit(self::ShotgunEnsembleClassifier, train)
    bestCorrectTraining = 0

    for normMean in [true, false]
        self, model, correct = fitEnsemble(self, normMean, train, self.sec[:factor])

        if correct > bestCorrectTraining
            bestCorrectTraining = correct
            self.sec[:model] = model
        end
    end
    return self, bestCorrectTraining
end

function fitIndividual(self::ShotgunEnsembleClassifier, NormMean, samples, i)
    model = ShotgunModel(NormMean, i, samples, samples[:Labels], nothing)
    correct, pred_labels = predict(self, model, samples)
    model.correct = correct
    append!(self.sec[:results], [model])
    return self
end

function fitEnsemble(self::ShotgunEnsembleClassifier, normMean, samples, factor)
    minWindowLength = 5
    maxWindowLength = min(self.sec[:MAX_WINDOW_LENGTH], samples[:Size])
    windows = [i for i in minWindowLength:maxWindowLength]

    correctTraining = 0
    self.sec[:results] = []

    println(self.sec[:NAME],"  Fitting for a norm of ",normMean)
    # with progressbar.ProgressBar(max_value=len(windows)) as bar:
    #     Parallel(n_jobs=3, backend="threading")(delayed(self.fitIndividual, check_pickle=False)(normMean, samples, windows, i, bar) for i in range(len(windows)))
    for i in windows
        print(i)
        print("; ")
        self = fitIndividual(self, normMean, samples, i)
    end
    println()

    # Find best correctTraining
    for i in 1:length(self.sec[:results])
        if self.sec[:results][i].correct > correctTraining
            correctTraining = self.sec[:results][i].correct
        end
    end

    # Remove Results that are no longer satisfactory
    new_results = []
    for i in 1:length(self.sec[:results])
        if self.sec[:results][i].correct >= (correctTraining * factor)
            append!(new_results, [self.sec[:results][i]])
        end
    end

    return self, new_results, correctTraining
end

function predictEnsemble(self::ShotgunEnsembleClassifier, models, testSamples)
    uniqueLabels = unique(testSamples[:Labels])
    pred_labels = []
    final_labels = Any[nothing for _ in 1:testSamples[:Samples]]
    for i in 1:length(models)
        model = models[i]
        correct, labels = predict(self, model, testSamples)
        append!(pred_labels, [labels])
    end

    for i in 1:testSamples[:Samples]
        ensemble_labels = [pred_labels[j][i] for j in 1:length(pred_labels)]
        try
            final_labels[i] = mode(ensemble_labels)
        catch #Guess if there is no favorite
            final_labels[i] = rand(uniqueLabels)
        end
    end
    final_correct = sum([final_labels[i] == testSamples[i].ts[:label] for i in 1:testSamples[:Samples]])
    return final_correct, final_labels
end

function predict(self::ShotgunEnsembleClassifier, model, test_samples)
    p = Any[nothing for _ in 1:test_samples[:Samples]]
    means = Any[nothing for _ in 1:length(model.labels)]
    stds = Any[nothing for _ in 1:length(model.labels)]
    means, stds = calcMeansStds(self, model.window, model.samples, means, stds, model.norm)

    for i in 1:test_samples[:Samples]
        query = test_samples[i]
        distanceTo1NN = Inf

        wQueryLen = min(length(query.ts[:data]), model.window)
        disjointWindows, _ = getDisjointSequences(query, wQueryLen, model.norm)
        for j in 1:length(model.labels)
            ts = model.samples[j].ts[:data]
            if ts != query.ts[:data]
                totalDistance = 0.

                for Q in 1:length(disjointWindows)
                    q = disjointWindows[Q]
                    resultDistance = distanceTo1NN

                    for w in 1:(length(ts) - model.window)
                        distance = getEuclideanDistance(self, ts, q.ts[:data], means[j][w], stds[j][w], resultDistance, w)
                        resultDistance = min(distance, resultDistance)
                    end
                    totalDistance += resultDistance
                    if totalDistance > distanceTo1NN
                        break
                    end
                end

                if totalDistance < distanceTo1NN
                    p[i] = model.samples[j].ts[:label]
                    distanceTo1NN = totalDistance
                end
            end
        end
    end

    correct = sum([p[i] == test_samples[i].ts[:label] for i in 1:test_samples[:Samples]])
    return correct, p
end

function getEuclideanDistance(self::ShotgunEnsembleClassifier, ts, q, meanTs, stdTs, minValue, w)
    distance = 0.0
    for ww in 1:(length(q)-1)
        value1 = (ts[w + ww - 1] - meanTs) * stdTs
        value = q[ww] - value1
        distance += (value * value)

        if distance >= minValue
            return Inf
        end
    end
    return distance
end

function calcMeansStds(self::ShotgunEnsembleClassifier, windowLength, trainSamples, means, stds, normMean)
    for i in 1:trainSamples[:Samples]
        w = min(windowLength, trainSamples[:Size])
        means[i] = Any[nothing for _ in 1:(trainSamples[:Size] - w + 1)]
        stds[i] = Any[nothing for _ in 1:(trainSamples[:Size] - w + 1)]
        means[i], stds[i] = calcIncreamentalMeanStddev(self, w, trainSamples[i].ts[:data], means[i], stds[i])
        for j in 1:length(stds[i])
            if stds[i][j] > 0
                stds[i][j] = 1.0 / stds[i][j]
            else
                stds[i][j] = 1.0
            end
            if normMean
                means[i][j] = means[i][j]
            else
                means[i][j] = 0
            end
        end
    end
    return means, stds
end

function calcIncreamentalMeanStddev(self::ShotgunEnsembleClassifier, windowLength, series, MEANS, STDS)
    SUM = 0.
    squareSum = 0.

    rWindowLength = 1.0 / windowLength
    for ww in 1:windowLength
        SUM += series[ww]
        squareSum += series[ww] * series[ww]
    end
    MEANS[1] = SUM * rWindowLength
    buf = squareSum * rWindowLength - MEANS[1] * MEANS[1]

    if buf > 0
        STDS[1] = sqrt(buf)
    else
        STDS[1] = 0
    end

    for w in 2:(length(series) - windowLength + 1)
        SUM += series[w + windowLength - 1] - series[w - 1]
        MEANS[w] = SUM * rWindowLength

        squareSum += series[w + windowLength - 1] * series[w + windowLength - 1] - series[w - 1] * series[w - 1]
        buf = squareSum * rWindowLength - MEANS[w] * MEANS[w]
        if buf > 0
            STDS[w] = sqrt(buf)
        else
            STDS[w] = 0
        end
    end

    return MEANS, STDS
end


mutable struct ShotgunModel
    norm::Bool
    window::Int64
    samples::Dict
    labels::Array{Any}
    correct::Any
end
