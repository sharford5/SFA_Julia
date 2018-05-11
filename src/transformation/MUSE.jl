include("/Users/Sam/Documents/SFA_Julia/src/transformation/SFA.jl")

mutable struct MUSE
    m::Dict{Symbol, Any}
end

function init__m(self::MUSE, maxF, maxS, histogramType, windowLength, normMean, lowerBounding)
    self.m[:maxF] = maxF + maxF % 2
    self.m[:symbols] = maxS
    self.m[:windowLengths] = windowLength
    self.m[:normMean] = normMean
    self.m[:signature] = Any[nothing for w in 1:length(windowLength)]
    self.m[:dict] = Dictionary(Dict())
    self = init__d(self)
    self.m[:MAX_WINDOW_LENGTH] = 450
    self.m[:lowerBounding] = lowerBounding
    self.m[:histogramType] = histogramType

    return self
end

function createWORDS(self::MUSE, samples)
    self.m[:words] = Any[nothing for _ in 1:length(self.m[:windowLengths])]

    println("Fitting Words ")
    for w in 1:length(self.m[:windowLengths])
        print(self.m[:windowLengths][w])
        print("; ")
        self = createWords(self, samples, w)
    end
    println()

    return self, self.m[:words]
end

function createWords(self::MUSE, samples, index)
    if self.m[:signature][index] == nothing
        self.m[:signature][index] = init_sfa(SFA(Dict(:initialized=>false, :HistogramType=>self.m[:histogramType])), false, self.m[:lowerBounding], false)
        self.m[:signature][index] = mv_fitWindowing(self.m[:signature][index], samples, self.m[:windowLengths][index], self.m[:maxF], self.m[:symbols], self.m[:normMean], false)
        # printBins(self.m[:signature][index])
    end

    words = []
    for m in 1:samples[:Samples]
        for n in 1:samples[:Dimensions]
            if length(samples[m][n].ts[:data]) >= self.m[:windowLengths][index]
                append!(words, [transformWindowingInt(self.m[:signature][index], samples[m][n], self.m[:maxF])])
            else
                append!(words, [[]])
            end
        end
    end

    self.m[:words][index] = words
    return self
end

function int2byte(self::MUSE, number)
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

function createBagOfPatterns(self::MUSE, words, samples, dimensionality, f)
    bagOfPatterns = []
    usedBits = int2byte(self, self.m[:symbols])
    mask = (1 << (usedBits * f)) - 1
    highestBit = int2byte(self, self.m[:MAX_WINDOW_LENGTH])+1

    j = 0
    for dim in 1:samples[:Samples]
        bop = init__bob(BagOfBigrams(Dict()), samples[dim][1].ts[:label])
        for w in 1:length(self.m[:windowLengths])
            if self.m[:windowLengths][1] >= f
                for d in 1:dimensionality
                    dLabel = string(convert(Int64, d))
                    for offset in 1:length(words[w][j+d])
                        word = string(w,"_",dLabel,"_",words[w][j + d][offset] & mask)
                        self, dict = getWord(self, word)
                        if dict in keys(bop.bob[:bob])
                            bop.bob[:bob][dict] = bop.bob[:bob][dict] + 1
                        else
                            bop.bob[:bob][dict] = 1
                        end

                        if offset - self.m[:windowLengths][w] > 0
                            prevWord = string(w, "_", dLabel, "_", words[w][j + d][offset - self.m[:windowLengths][w]] & mask)
                            self, newWord = getWord(self, string(word,"_", prevWord))
                            if newWord in keys(bop.bob[:bob])
                                bop.bob[:bob][newWord] = bop.bob[:bob][newWord] + 1
                            else
                                bop.bob[:bob][newWord] = 1
                            end
                        end
                    end
                end
            end
        end
        append!(bagOfPatterns, [bop])
        j += dimensionality
    end
    return self, bagOfPatterns
end

function filterChiSquared(self::MUSE, bob, chi_limit)
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

function init__d(self::MUSE)
    self.m[:dict].d[:dict] = Dict()
    self.m[:dict].d[:dictChi] = Dict()
    return self
end

function reset(self::MUSE)
    self.m[:dict].d[:dict] = Dict()
    self.m[:dict].d[:dictChi] = Dict()
    return self
end

function getWord(self::MUSE, word)
    word2 = 0
    if word in keys(self.m[:dict].d[:dict])
        word2 = self.m[:dict].d[:dict][word]
    else
        word2 = length(keys(self.m[:dict].d[:dict])) + 1
        self.m[:dict].d[:dict][word] = word2
    end
    return self, word2
end

function getWordChi(self::MUSE, word)
    word2 = 0
    if word in keys(self.m[:dict].d[:dictChi])
        word2 = self.m[:dict].d[:dictChi][word]
    else
        word2 = length(keys(self.m[:dict].d[:dictChi])) + 1
        self.m[:dict].d[:dictChi][word] = word2
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

function Remap(self::MUSE, bagOfPatterns)
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
