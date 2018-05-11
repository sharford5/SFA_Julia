include("/Users/Sam/Documents/SFA_Julia/src/transformation/SFA.jl")


mutable struct BOSS
    boss::Dict{Symbol, Any}
end

function init__boss(self::BOSS, maxF, maxS, windowLength, normMean)
    self.boss[:maxF] = maxF
    self.boss[:symbols] = maxS
    self.boss[:windowLength] = windowLength
    self.boss[:normMean] = normMean
    self.boss[:signature] = nothing
    return self
end

function createWords(self::BOSS, samples)
    if self.boss[:signature] == nothing
        self.boss[:signature] = init_sfa(SFA(Dict(:initialized=>false, :HistogramType=>"EQUI_DEPTH")))
        self.boss[:signature] = fitWindowing(self.boss[:signature], samples, self.boss[:windowLength], self.boss[:maxF], self.boss[:symbols], self.boss[:normMean], true)
        # printBins(self.boss[:signature])
    end

    words = []
    for i in 1:samples[:Samples]
        sfaWords = transformWindowing(self.boss[:signature], samples[i])
        words_small = []
        for word in sfaWords
            append!(words_small, [createWord(self, word, self.boss[:maxF], int2byte(self, self.boss[:symbols]))])
        end
        append!(words, [words_small])
    end
    return words
end

function createWord(self::BOSS, numbers, maxF, bits)
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

function createBagOfPattern(self::BOSS, words, samples, f)
    bagOfPatterns = []
    usedBits = int2byte(self, self.boss[:symbols])
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

function int2byte(self::BOSS, number)
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
#
# def bag2dict(self, bag):
#     bag_dict = []
#     for list in bag:
#         new_dict = {}
#         for element in list:
#             if element in new_dict.keys():
#                 new_dict[element] += 1
#             else:
#                 new_dict[element] = 1
#         bag_dict.append(new_dict)
#     return bag_dict
