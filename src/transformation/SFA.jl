include("/Users/Sam/Documents/SFA_Julia/src/timeseries/TimeSeries.jl")
include("/Users/Sam/Documents/SFA_Julia/src/transformation/MFT.jl")


"
 Symbolic Fourier Approximation as published in
 Schäfer, P., Högqvist, M.: SFA: a symbolic fourier approximation and
 index for similarity search in high dimensional datasets.
 In: EDBT, ACM (2012)
"

mutable struct SFA
    sfa::Dict{Symbol, Any}
end

function init_sfa(self::SFA, SUP = false, LB = true, MB = false)
    self.sfa[:initialized] = false
    self.sfa[:SUP] = SUP
    self.sfa[:lowerBounding] = LB
    self.sfa[:MUSE_Bool] = MB
    self.sfa[:transformation] = init_mft(MFT(Dict(:initialized=>false)))

    return self
end

function initialize_sfa(self::SFA, wordLength, symbols, normMean)
    self.sfa[:initialized] = true
    self.sfa[:wordLength] = wordLength
    self.sfa[:maxWordLength] = wordLength
    self.sfa[:symbols] = symbols
    self.sfa[:normMean] = normMean
    self.sfa[:alphabetSize] = symbols
    self.sfa[:transformation] = init_mft(MFT(Dict(:initialized=>false)))

    self.sfa[:orderLine] = []
    self.sfa[:bins] = zeros(wordLength, self.sfa[:alphabetSize]) + Inf
    self.sfa[:bins][:, 1] = -Inf
    return self
end

function printBins(self::SFA)
    println(self.sfa[:bins])
end

function fitTransform(self::SFA, samples, wordLength, symbols, normMean)
    self, transformedSamples = fitTransformDouble(self, samples, wordLength, symbols, normMean)
    return self, transform(self, samples, transformedSamples)
end

function fitTransformDouble(self::SFA, samples, wordLength, symbols, normMean)
    if self.sfa[:initialized] == false
        self = initialize_sfa(self, wordLength, symbols, normMean)
        if self.sfa[:transformation].mft[:initialized] == false
            self.sfa[:transformation] = initialize_mft(self.sfa[:transformation], length(samples[1].ts[:data]), normMean, self.sfa[:lowerBounding], self.sfa[:MUSE_Bool])
        end
    end

    self, transformedSamples = fillOrderline(self, samples, wordLength)

    if self.sfa[:HistogramType] == "EQUI_DEPTH"
        self = divideEquiDepthHistogram(self)
    elseif self.sfa[:HistogramType] == "EQUI_FREQUENCY"
        self = divideEquiWidthHistogram(self)
    elseif self.sfa[:HistogramType] == "INFORMATION_GAIN"
        self = divideHistogramInformationGain(self)
    end

    return self, transformedSamples
end

function transform(self::SFA, samples, approximate)
    transformed = []
    for i in 1:samples[:Samples]
        append!(transformed, [transform2(self, samples[i].ts[:data], approximate[i])])
    end

    return transformed
end

function transform2(self::SFA, series, one_approx)
    if one_approx == "null"
        one_approx = transform_mft(self.sfa[:transformation], series, self.sfa[:maxWordLength])
    end

    if self.sfa[:SUP]
        return quantizationSupervised(self, one_approx)
    else
        return quantization(self, one_approx)
    end
end

function fillOrderline(self::SFA, samples, wordLength)
    self.sfa[:orderLine] = [[[0., 0] for _ in 1:samples[:Samples]] for _ in 1:wordLength]

    transformedSamples = []
    for i in 1:samples[:Samples]
        transformedSamples_small = transform_mft(self.sfa[:transformation], samples[i].ts[:data], wordLength)
        append!(transformedSamples, [transformedSamples_small])
        for j in 1:length(transformedSamples_small)
            value = round(transformedSamples_small[j], 2) + 0. #is a bad way of removing values of -0.0
            obj = [value, samples[i].ts[:label]]
            self.sfa[:orderLine][j][i] = obj
        end
    end

    for (l, list) in enumerate(self.sfa[:orderLine])
        del_list = list
        new_list = []
        while length(del_list) != 0
            current_min_value = Inf
            current_min_location = -1
            label = -Inf
            for j in 1:length(del_list)
                if (del_list[j][1] < current_min_value) | ((del_list[j][1] == current_min_value) & (del_list[j][2] < label))
                    current_min_value = del_list[j][1]
                    label = del_list[j][2]
                    current_min_location = j
                end
            end
            append!(new_list, [del_list[current_min_location]])
            deleteat!(del_list,current_min_location)
        end
        self.sfa[:orderLine][l] = new_list
    end
    return self, transformedSamples
end

function divideEquiDepthHistogram(self::SFA)
    for i in 1:size(self.sfa[:bins],1)
        depth = length(self.sfa[:orderLine][i]) / self.sfa[:alphabetSize]
        try
            depth = length(self.sfa[:orderLine][i]) / self.sfa[:alphabetSize]
        catch
            depth = 0
        end

        pos = 1
        count = 0
        try
            for j in 1:length(self.sfa[:orderLine][i])
                count += 1
                condition1 = count > ceil(depth * (pos))
                condition2 = pos == 1
                condition3 = self.sfa[:bins][i, pos] != self.sfa[:orderLine][i][j][1]
                if (condition1) & (condition2 | condition3)
                    self.sfa[:bins][i, pos+1] = round(self.sfa[:orderLine][i][j][1],2)
                    pos += 1
                end
            end
        end
    end
    self.sfa[:bins][:, 1] = -Inf

    return self
end

function quantization(self::SFA, one_approx)
    i = 1
    word = [0 for _ in 1:length(one_approx)]
    for v in one_approx
        c = 1
        for C in 1:size(self.sfa[:bins],2)
            if v < self.sfa[:bins][i,c]
                break
            else
                c += 1
            end
        end
        word[i] = c-2
        i += 1
    end
    return word
end

function fitWindowing(self::SFA, samples, windowSize, wordLength, symbols, normMean, lowerBounding)
    self.sfa[:transformation] = initialize_mft(self.sfa[:transformation], windowSize, normMean, lowerBounding, self.sfa[:MUSE_Bool])

    sa = Dict()
    index = 1

    for i in 1:samples[:Samples]
        new_list, amount = getDisjointSequences(samples[i], windowSize, normMean)
        for j in 1:amount
            sa[index] = new_list[j]
            index += 1
        end
    end

    sa[:Samples] = index-1

    if self.sfa[:SUP]
        self, _ = fitTransformSupervised(self, sa, wordLength, symbols, normMean)
    else
        self, _ = fitTransform(self, sa, wordLength, symbols, normMean)
    end
    return self
end

function transformWindowing(self::SFA, series)
    mft = transformWindowing(self.sfa[:transformation], series, self.sfa[:maxWordLength])

    words = []
    for i in 1:length(mft)
        if self.sfa[:SUP]
            append!(words, [quantizationSupervised(self, mft[i])])
        else
            append!(words, [quantization(self, mft[i])])
        end
    end
    return words
end

function mv_fitWindowing(self::SFA, samples, windowSize, wordLength, symbols, normMean, lowerBounding)
    sa = Dict()
    index = 1
    for i in 1:samples[:Samples]
        for k in 1:length(keys(samples[i]))
            new_list, _ = getDisjointSequences(samples[i][k], windowSize, normMean)
            for j in 1:length(new_list)
                sa[index] = new_list[j]
                index += 1
            end
        end
    end

    sa[:Samples] = index - 1
    self = fitWindowing(self, sa, windowSize, wordLength, symbols, normMean, lowerBounding)
    return self
end

function transformWindowingInt(self::SFA, series, wordLength)
    words = transformWindowing(self, series)
    intWords = []
    for i in 1:length(words)
        append!(intWords, [createWord(self, words[i], wordLength, int2byte(self, self.sfa[:alphabetSize]))])
    end
    return intWords
end

function divideEquiWidthHistogram(self::SFA)
    i = 1
    for element in self.sfa[:orderLine]
        if length(element) != 0
            first = element[1][1]
            last = element[end][1]
            intervalWidth = (last - first) / self.sfa[:alphabetSize]

            for c in 1:(self.sfa[:alphabetSize]-1)
                self.sfa[:bins][i,c] = intervalWidth * (c + 1) + first
            end
        end
        i += 1
    end

    self.sfa[:bins][:, 1] = -Inf
    return self
end

function divideHistogramInformationGain(self::SFA)
    for i in 1:length(self.sfa[:orderLine])
        element = self.sfa[:orderLine][i]
        self.sfa[:splitPoints] = []
        self = findBestSplit(self, element, 1, length(element), self.sfa[:alphabetSize])
        self.sfa[:splitPoints] = sort(self.sfa[:splitPoints])
        for j in 1:length(self.sfa[:splitPoints])
            self.sfa[:bins][i, j + 1] = element[self.sfa[:splitPoints][j]+1][1]
        end
    end
    return self
end

function findBestSplit(self::SFA, element, start, last, remainingSymbols)
    bestGain = -1
    bestPos = -1
    total = last - start + 1

    self.sfa[:cOut] = Dict()
    self.sfa[:cIn] = Dict()

    for pos in start:last
        label = element[pos][2]
        if label in keys(self.sfa[:cOut])
            self.sfa[:cOut][label] = self.sfa[:cOut][label] + 1.
        else
            self.sfa[:cOut][label] = 1.
        end
    end

    class_entropy = entropy(self, self.sfa[:cOut], total)

    i = start
    lastLabel = element[i][2]
    self, i2 = moveElement(self, lastLabel)
    i += i2

    for split in (start+1):(last-1)
        label = element[i][2]
        self, i2 = moveElement(self, label)
        i += i2
        if label != lastLabel
            gain = calculateInformationGain(self, class_entropy, i-1, total)
            if gain >= bestGain
                bestGain = gain
                bestPos = split
            end
        end
        lastLabel = label
    end

    if bestPos > -1
        append!(self.sfa[:splitPoints], bestPos)
        remainingSymbols = remainingSymbols / 2
        if remainingSymbols > 1
            if (bestPos - start >= 2) & (last - bestPos >= 2)
                self = findBestSplit(self, element, start, bestPos-1, remainingSymbols)
                self = findBestSplit(self, element, bestPos, last, remainingSymbols)
            elseif last - bestPos >= 4
                self = findBestSplit(self, element, bestPos, convert(Int64, round((last - bestPos)/2,0))-1, remainingSymbols)
                self = findBestSplit(self, element, convert(Int64, round((last - bestPos)/2,0)), last, remainingSymbols)
            elseif bestPos - start >= 4
                self = findBestSplit(self, element, start, convert(Int64, round((bestPos-start)/2,0))-1, remainingSymbols)
                self = findBestSplit(self, element, convert(Int64, round((bestPos-start)/2,0)), last, remainingSymbols)
            end
        end
    end
    return self
end

function moveElement(self::SFA, label)
    if label in keys(self.sfa[:cIn])
        self.sfa[:cIn][label] = self.sfa[:cIn][label] + 1.
    else
        self.sfa[:cIn][label] = 1.
    end
    if label in keys(self.sfa[:cOut])
        self.sfa[:cOut][label] = self.sfa[:cOut][label] - 1.
    else
        self.sfa[:cOut][label] = -1.
    end
    return self, 1
end

function entropy(self::SFA, freq, total)
    e = 0.
    if total != 0
        log2 = 1.0 / log(2)
        for k in keys(freq)
            p = freq[k] / total
            if p > 0
                e += -1 * p * log(p) * log2
            end
        end
    else
        e = Inf
    end
    return e
end

function calculateInformationGain(self::SFA, class_entropy, total_c_in, total)
    total_c_out = total - total_c_in
    return class_entropy - total_c_in / total * entropy(self, self.sfa[:cIn], total_c_in) - total_c_out / total * entropy(self, self.sfa[:cOut], total_c_out)
end

function createWord(self::SFA, numbers, maxF, bits)
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

function int2byte(self::SFA, number)
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

## Supervised
function fitTransformSupervised(self::SFA, samples, wordLength, symbols, normMean)
    l = length(samples[1].ts[:data])
    self, transformedSignal = fitTransformDouble(self, samples, l, symbols, normMean)

    best = calcBestCoefficients(self, samples, transformedSignal)

    self.sfa[:bestValues] = [0 for i in 1:min(length(best), wordLength)]
    self.sfa[:maxWordLength] = 0

    for i in 1:length(self.sfa[:bestValues])
        self.sfa[:bestValues][i] = best[i][1]
        self.sfa[:maxWordLength] = max(best[i][1] + 1, self.sfa[:maxWordLength])
    end
    self.sfa[:maxWordLength] += self.sfa[:maxWordLength] % 2

    return self, transform(self, samples, transformedSignal)
end

function calcBestCoefficients(self::SFA, samples, transformedSignal)
    classes = Dict()
    for i in 1:samples[:Samples]
        if samples[i].ts[:label] in keys(classes)
            append!(classes[samples[i].ts[:label]], [transformedSignal[i]])
        else
            classes[samples[i].ts[:label]] = [transformedSignal[i]]
        end
    end

    nSamples = length(transformedSignal)
    nClasses = length(keys(classes))
    l = length(transformedSignal[2])

    f = getFoneway(self, l, classes, nSamples, nClasses)
    f_sorted = sort(f, rev = true)

    best = []
    inf_index = 0

    for value in f_sorted
        if value == Inf
            index = findin(f, value)[1] + inf_index
            inf_index += 1
        else
            index = findin(f, value)[1]
        end
        append!(best, [[convert(Int64, index), value]])
    end

    return best
end

function getFoneway(self, l, classes, nSamples, nClasses)
    ss_alldata = [0. for i in 1:l]
    sums_args = Dict()
    keys_class = keys(classes)

    for key in keys_class
        allTs = classes[key]
        sums = [0. for i in 1:length(ss_alldata)]
        sums_args[key] = sums
        for ts in allTs
            for i in 1:length(ts)
                ss_alldata[i] += ts[i] * ts[i]
                sums[i] += ts[i]
            end
        end
    end

    square_of_sums_alldata = [0. for i in 1:length(ss_alldata)]
    square_of_sums_args = Dict()
    for key in keys_class
        # square_of_sums_alldata2 = [0. for i in range(len(ss_alldata))]
        sums = sums_args[key]
        for i in 1:length(sums)
            square_of_sums_alldata[i] += sums[i]
        end
        # square_of_sums_alldata += square_of_sums_alldata2

        squares = [0. for i in 1:length(sums)]
        square_of_sums_args[key] = squares
        for i in 1:length(sums)
            squares[i] += sums[i]*sums[i]
        end
    end

    for i in 1:length(square_of_sums_alldata)
        square_of_sums_alldata[i] *= square_of_sums_alldata[i]
    end

    sstot = [0. for i in 1:length(ss_alldata)]
    for i in 1:length(sstot)
        sstot[i] = ss_alldata[i] - square_of_sums_alldata[i]/nSamples
    end

    ssbn = [0. for i in 1:length(ss_alldata)]    ## sum of squares between
    sswn = [0. for i in 1:length(ss_alldata)]    ## sum of squares within

    for key in keys_class
        sums = square_of_sums_args[key]
        n_samples_per_class = length(classes[key])
        for i in 1:length(sums)
            ssbn[i] += sums[i]/n_samples_per_class
        end
    end

    for i in 1:length(square_of_sums_alldata)
        ssbn[i] += -square_of_sums_alldata[i]/nSamples
    end

    dfbn = nClasses-1                          ## degrees of freedom between
    dfwn = nSamples - nClasses                 ## degrees of freedom within
    msb = [0. for i in 1:length(ss_alldata)]   ## variance (mean square) between classes
    msw = [0. for i in 1:length(ss_alldata)]   ## variance (mean square) within samples
    f = [0. for i in 1:length(ss_alldata)]     ## f-ratio

    for i in 1:length(sswn)
        sswn[i] = sstot[i]-ssbn[i]
        msb[i] = ssbn[i]/dfbn
        msw[i] = sswn[i]/dfwn
        if msw[i] != 0.
            f[i] = msb[i]/msw[i]
        else
            f[i] = Inf
        end
    end

    return f
end

function quantizationSupervised(self, one_approx)
    signal = [0 for _ in 1:min(length(one_approx), length(self.sfa[:bestValues]))]

    for a in 1:length(signal)
        i = self.sfa[:bestValues][a]
        b = 0
        for beta in 1:size(self.sfa[:bins],2)
            if one_approx[i] < self.sfa[:bins][i,beta]
                break
            else
                b += 1
            end
        end
        signal[a] = b-1
    end

    return signal
end
