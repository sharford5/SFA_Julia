

mutable struct timeseries
    ts::Dict{Symbol, Any}
end

function __init__(self::timeseries, data, label, NORM_CHECK)
    self.ts[:data] = data
    self.ts[:label] = label
    self.ts[:NORM_CHECK] = NORM_CHECK
    self.ts[:normed] = false
    self.ts[:mean] = 0.
    self.ts[:std] = 1.
    return self
end

function NORM(self::timeseries, normMean)
    self.ts[:mean] = mean(self.ts[:data])
    self = calculate_std(self)

    thisNorm = !self.ts[:normed]

    if (self.ts[:NORM_CHECK]) & (thisNorm)
        self = NORM_WORK(self, normMean)
    end
    return self
end

function calculate_std(self::timeseries)
    var = 0.
    for i in 1:length(self.ts[:data])
        var += self.ts[:data][i] * self.ts[:data][i]
    end

    norm = 1.0 / length(self.ts[:data])
    buf = (norm * var) - (self.ts[:mean] * self.ts[:mean])

    try
        self.ts[:std] = buf != 0 ? sqrt(buf) : 0.
    catch
        self.ts[:std] = 0.
    end
    return self
end

function NORM_WORK(self::timeseries, normMean)
    ISTD = self.ts[:std] == 0 ? 1. : 1./self.ts[:std]

    if normMean
        self.ts[:data] = [(self.ts[:data][i] - self.ts[:mean]) * ISTD for i in 1:length(self.ts[:data])]
        self.ts[:mean] = 0.0
    elseif ISTD != 1.
        self.ts[:data] = [self.ts[:data][i] * ISTD for i in 1:length(self.ts[:data])]
    end

    self.ts[:normed] = true
    return self
end


##====================================================================================================
function getDisjointSequences(series::timeseries, windowSize, normMean)
    amount = convert(Int64, floor(length(series.ts[:data]) / windowSize))
    subseqences = Dict()

    for i in 0:(amount-1)
        subseqences_data = __init__(timeseries(Dict()), series.ts[:data][(i*windowSize)+1:((i+1)*windowSize)], series.ts[:label], series.ts[:NORM_CHECK])
        subseqences_data = NORM(subseqences_data, normMean)
        subseqences[i+1] =  subseqences_data
    end

    return subseqences, amount
end

function calcIncreamentalMeanStddev(windowLength, series, MEANS, STDS)
    SUM = 0.
    squareSum = 0.

    rWindowLength = 1.0 / windowLength
    for ww in 1:windowLength
        SUM += series[ww]
        squareSum += series[ww]*series[ww]
    end
    append!(MEANS, SUM * rWindowLength)
    buf = squareSum*rWindowLength - MEANS[1]*MEANS[1]

    if buf > 0
        append!(STDS, sqrt(buf))
    else
        append!(STDS, 0)
    end

    for w in 2:(length(series)-windowLength+1)
        SUM += series[w+windowLength-1] - series[w-1]
        append!(MEANS, SUM * rWindowLength)

        squareSum += series[w+windowLength-1]*series[w+windowLength-1] - series[w-1]*series[w-1]
        buf = squareSum * rWindowLength - MEANS[w]*MEANS[w]
        if buf > 0
            append!(STDS, sqrt(buf))
        else
            append!(STDS, 0)
        end
    end

    return MEANS, STDS
end
#
# def createWord(numbers, maxF, bits):
#     shortsPerLong = int(round(60 / bits))
#     to = min([len(numbers), maxF])
#
#     b = 0
#     s = 0
#     shiftOffset = 1
#     for i in range(s, (min(to, shortsPerLong + s))):
#         shift = 1
#         for j in range(bits):
#             if (numbers[i] & shift) != 0:
#                 b |= shiftOffset
#             shiftOffset <<= 1
#             shift <<= 1
#
#     limit = 2147483647
#     total = 2147483647 + 2147483648
#     while b > limit:
#         b = b - total - 1
#     return b
#
#
# def int2byte(number):
#     log = 0
#     if (number & 0xffff0000) != 0:
#         number >>= 16
#         log = 16
#     if number >= 256:
#         number >>= 8
#         log += 8
#     if number >= 16:
#         number >>= 4
#         log += 4
#     if number >= 4:
#         number >>= 2
#         log += 2
#     return log + (number >> 1)
#
#
# def compareTo(score, bestScore):
#     if score[1] > bestScore[1]:
#         return -1
#     elif (score[1] == bestScore[1]) & (score[4] > bestScore[4]):
#         return -1
#     return 1
