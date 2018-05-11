include("/Users/Sam/Documents/SFA_Julia/src/transformation/SFA.jl")


mutable struct WEASEL
    w::Dict{Symbol, Any}
end

function init__w(self::WEASEL, maxF, maxS, windowLength, normMean)
    self.w[:maxF] = maxF
    self.w[:symbols] = maxS
    self.w[:windowLengths] = windowLength
    self.w[:normMean] = normMean
    self.w[:signature] = Any[nothing for w in 1:length(windowLength)]
    self.w[:dict] = Dictionary(Dict())
    self = init__d(self)
    self.w[:MAX_WINDOW_LENGTH] = 250

    return self
end

function createWORDS(self::WEASEL, samples)
    self.w[:words] = Any[nothing for _ in 1:length(self.w[:windowLengths])]

    println("Fitting Words ")
    # with progressbar.ProgressBar(max_value=len(self.windowLengths)) as bar:
    #     Parallel(n_jobs=4, backend="threading")(delayed(self.createWords, check_pickle=False)(samples, w, bar) for w in range(len(self.windowLengths)))
    for w in 1:length(self.w[:windowLengths])
        print(self.w[:windowLengths][w])
        print("; ")
        self = createWords(self, samples, w)
    end
    println()

    return self, self.w[:words]
end

function createWords(self::WEASEL, samples, index)
    if self.w[:signature][index] == nothing
        self.w[:signature][index] = init_sfa(SFA(Dict(:initialized=>false, :HistogramType=>"INFORMATION_GAIN")), true, false)
        self.w[:signature][index] = fitWindowing(self.w[:signature][index], samples, self.w[:windowLengths][index], self.w[:maxF], self.w[:symbols], self.w[:normMean], false)
        # printBins(self.w[:signature][index])
    end

    words = []
    for i in 1:samples[:Samples]
        append!(words, [transformWindowingInt(self.w[:signature][index], samples[i], self.w[:maxF])])
    end

    self.w[:words][index] = words
    return self
end

function int2byte(self::WEASEL, number)
    log = 0
    if (number & 0xffff0000) != 0
        number >>= 16
        log = 16
    end
    if number >= 256
        number >>= 8
        log += 8
    end
    if number >= 16
        number >>= 4
        log += 4
    end
    if number >= 4
        number >>= 2
        log += 2
    end
    return log + (number >> 1)
end

function createBagOfPatterns(self::WEASEL, words, samples, f)
    bagOfPatterns = [init__bob(BagOfBigrams(Dict()), samples[j].ts[:label]) for j in 1:samples[:Samples]]
    usedBits = int2byte(self, self.w[:symbols])
    mask = (1 << (usedBits * f)) - 1
    highestBit = int2byte(self, self.w[:MAX_WINDOW_LENGTH])+1

    for j in 1:samples[:Samples]
        for w in 1:length(self.w[:windowLengths])
            for offset in 1:length(words[w][j])
                self, word = getWord(self, (words[w][j][offset] & mask) << highestBit | w)
                if word in keys(bagOfPatterns[j].bob[:bob])
                    bagOfPatterns[j].bob[:bob][word] = bagOfPatterns[j].bob[:bob][word] + 1
                else
                    bagOfPatterns[j].bob[:bob][word] = 1
                end

                if offset - self.w[:windowLengths][w] > 0
                    self, prevWord = getWord(self, (words[w][j][offset - self.w[:windowLengths][w]] & mask) << highestBit | w)
                    self, newWord = getWord(self, (prevWord << 32 | word ) << highestBit)
                    if newWord in keys(bagOfPatterns[j].bob[:bob])
                        bagOfPatterns[j].bob[:bob][newWord] = bagOfPatterns[j].bob[:bob][newWord] + 1
                    else
                        bagOfPatterns[j].bob[:bob][newWord] = 1
                    end
                end
            end
        end
    end

    return self, bagOfPatterns
end

function filterChiSquared(self::WEASEL, bob, chi_limit)
    classFrequencies = Dict()
    for list in bob
        label = list.bob[:label]
        if label in keys(classFrequencies)
            classFrequencies[label] = classFrequencies[label] + 1
        else
            classFrequencies[label] = 1
        end
    end

    featureCount = Dict()
    classProb = Dict()
    observed = Dict()
    chiSquare = Dict()

    # count number of samples with this word
    for bagOfPattern in bob
        label = bagOfPattern.bob[:label]
        bag_dict = bagOfPattern.bob[:bob]
        for key in keys(bag_dict)
            if bag_dict[key] > 0
                if key in keys(featureCount)
                    featureCount[key] = featureCount[key] + 1
                else
                    featureCount[key] = 1
                end
                key2 = label << 32 | key
                if key2 in keys(observed)
                    observed[key2] = observed[key2] + 1
                else
                    observed[key2] = 1
                end
            end
        end
    end

    # samples per class
    for list in bob
        label = list.bob[:label]
        if label in keys(classProb)
            classProb[label] = classProb[label] + 1
        else
            classProb[label] = 1
        end
    end

    # chi square: observed minus expected occurence
    for prob_key in keys(classProb)
        prob_value = classProb[prob_key]/length(bob)
        for feature_key in keys(featureCount)
            feature_value = featureCount[feature_key]
            key = prob_key << 32 | feature_key
            expected = prob_value * feature_value
            chi = get(observed, key) - expected
            newChi = chi * chi / expected

            if (newChi >= chi_limit) & (newChi > get(chiSquare, feature_key))
                chiSquare[feature_key] = newChi
            end
        end
    end

    #best elements above limit
    for j in 1:length(bob)
        for key in keys(bob[j].bob[:bob])
            if get(chiSquare, key) < chi_limit
                bob[j].bob[:bob][key] = 0
            end
        end
    end

    bob = Remap(self, bob)

    return self, bob
end

#--------------------------------------------------------------------------------------------------

mutable struct BagOfBigrams
    bob::Dict{Symbol, Any}
end

function init__bob(self::BagOfBigrams, label)
    self.bob[:bob] = Dict()
    self.bob[:label] =  convert(Int64, label)
    return self
end

#--------------------------------------------------------------------------------------------------

mutable struct Dictionary
    d::Dict{Symbol, Any}
end

function init__d(self::WEASEL)
    self.w[:dict].d[:dict] = Dict()
    self.w[:dict].d[:dictChi] = Dict()
    return self
end

function reset(self::WEASEL)
    self.w[:dict].d[:dict] = Dict()
    self.w[:dict].d[:dictChi] = Dict()
    return self
end

function getWord(self::WEASEL, word)
    word2 = 0
    if word in keys(self.w[:dict].d[:dict])
        word2 = self.w[:dict].d[:dict][word]
    else
        word2 = length(keys(self.w[:dict].d[:dict])) + 1
        self.w[:dict].d[:dict][word] = word2
    end
    return self, word2
end

function getWordChi(self::WEASEL, word)
    word2 = 0
    if word in keys(self.w[:dict].d[:dictChi])
        word2 = self.w[:dict].d[:dictChi][word]
    else
        word2 = length(keys(self.w[:dict].d[:dictChi])) + 1
        self.w[:dict].d[:dictChi][word] = word2
    end
    return self, word2
end

function SIZE(self::Dictionary)
    if length(self.d[:dictChi]) != 0
        return length(self.d[:dictChi])+1
    else
        return length(self.d[:dict])
    end
end

function Remap(self::WEASEL, bagOfPatterns)
    for j in 1:length(bagOfPatterns)
        oldMap = bagOfPatterns[j].bob[:bob]
        bagOfPatterns[j].bob[:bob] = Dict()
        for word_key in keys(oldMap)
            word_value = oldMap[word_key]
            if word_value > 0
                self, k = getWordChi(self, word_key)
                bagOfPatterns[j].bob[:bob][k] = word_value
            end
        end
    end

    return bagOfPatterns
end

function get(dictionary, key)
    if key in keys(dictionary)
        return dictionary[key]
    else
        return 0
    end
end
