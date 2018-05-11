include("/Users/Sam/Documents/SFA_Julia/src/transformation/SFA.jl")


mutable struct BOSSVS
    bossvs::Dict{Symbol, Any}
end

function init__bossvs(self::BOSSVS, maxF, maxS, windowLength, normMean)
    self.bossvs[:maxF] = maxF
    self.bossvs[:symbols] = maxS
    self.bossvs[:windowLength] = windowLength
    self.bossvs[:normMean] = normMean
    self.bossvs[:signature] = nothing
    return self
end

function createWords(self::BOSSVS, samples)
    if self.bossvs[:signature] == nothing
        self.bossvs[:signature] = init_sfa(SFA(Dict(:initialized=>false, :HistogramType=>"EQUI_DEPTH")))
        self.bossvs[:signature] = fitWindowing(self.bossvs[:signature], samples, self.bossvs[:windowLength], self.bossvs[:maxF], self.bossvs[:symbols], self.bossvs[:normMean], true)
        # printBins(self.bossvs[:signature])
    end

    words = []
    for i in 1:samples[:Samples]
        sfaWords = transformWindowing(self.bossvs[:signature], samples[i])
        words_small = []
        for word in sfaWords
            append!(words_small, [createWord(self, word, self.bossvs[:maxF], int2byte(self, self.bossvs[:symbols]))])
        end
        append!(words, [words_small])
    end
    return words
end

function createWord(self::BOSSVS, numbers, maxF, bits)
    shortsPerLong = convert(Int64, round(60 / bits))
    to = min(length(numbers), maxF)

    b = 0
    s = 1
    shiftOffset = 1
    for i in s:min(to, shortsPerLong + s)
        shift = 1
        for j in 1:bits
            if (numbers[i] & shift) != 0
                b |= shiftOffset
            end
            shiftOffset <<= 1
            shift <<= 1
        end
    end

    limit = 2147483647
    total = 2147483647 + 2147483648
    while b > limit
        b = b - total - 1
    end
    return b
end

function createBagOfPattern(self::BOSSVS, words, samples, f)
    bagOfPatterns = []
    usedBits = int2byte(self, self.bossvs[:symbols])
    mask = (1 << (usedBits * f)) - 1

    for j in 1:length(words)
        BOP = Dict()
        lastWord = -9223372036854775808
        for offset in 1:length(words[j])
            word = words[j][offset] & mask
            if word != lastWord
                if word in keys(BOP)
                    BOP[word] += 1
                else
                    BOP[word] = 1
                end
            end
            lastWord = word
        end
        append!(bagOfPatterns, [BOP])
    end

    return bagOfPatterns
end

function int2byte(self::BOSSVS, number)
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

function createTfIdf(self::BOSSVS, bagOfPatterns, sampleIndices, uniqueLabels, labels)
    matrix = Dict()
    for label in uniqueLabels
        matrix[label] = Dict()
    end

    for j in sampleIndices
        label = labels[j]
        for key in keys(bagOfPatterns[j])
            value = bagOfPatterns[j][key]
            if key in keys(matrix[label])
                matrix[label][key] = matrix[label][key] + value
            else
                matrix[label][key] = value
            end
        end
    end
    wordInClassFreq = Dict()
    for key in keys(matrix)
        value = matrix[key]
        for key2 in keys(matrix[key])
            value2 = matrix[key][key2]
            if key2 in keys(wordInClassFreq)
                wordInClassFreq[key2] = wordInClassFreq[key2] + 1
            else
                wordInClassFreq[key2] = 1
            end
        end
    end

    for key in keys(matrix)
        value = matrix[key]
        tfIDFs = matrix[key]
        for key2 in keys(tfIDFs)
            value2 = tfIDFs[key2]
            wordCount = wordInClassFreq[key2]
            if (value2 > 0) & (length(uniqueLabels) != wordCount)
                tfValue = 1. + log10(value2)
                idfValue = log10(1. + length(uniqueLabels) / wordCount)
                tfIdf = tfValue / idfValue
                tfIDFs[key2] = tfIdf
            else
                tfIDFs[key2] = 0.
            end
        end
        matrix[key] = tfIDFs
    end

    matrix = normalizeTfIdf(self, matrix)
    return matrix
end

function normalizeTfIdf(self, classStatistics)
    for key in keys(classStatistics)
        value = classStatistics[key]
        squareSum = 0.
        for key2 in keys(classStatistics[key])
            value2 = classStatistics[key][key2]
            squareSum += value2 ^ 2
        end
        squareRoot = sqrt(squareSum)
        if squareRoot > 0
            for key2 in keys(classStatistics[key])
                value2 = classStatistics[key][key2]
                classStatistics[key][key2] =  classStatistics[key][key2]/squareRoot
            end
        end
    end
    return classStatistics
end
