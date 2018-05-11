include("/Users/Sam/Documents/SFA_Julia/src/timeseries/TimeSeries.jl")

mutable struct MFT
    mft::Dict{Symbol, Any}
    # initialized::Bool
    # windowSize::Int64
    # MUSE::Bool
    # startOffset::Int64
    # norm::Float64
end

function init_mft(self::MFT)
    self.mft[:initialized] = false
    return self
end

function initialize_mft(self::MFT, windowSize, normMean, lowerBounding, MUSE_Bool = false)
    self.mft[:initialized] = true
    self.mft[:windowSize] = windowSize
    self.mft[:MUSE] = MUSE_Bool

    self.mft[:startOffset] = normMean ? 2 : 0
    # self.startOffset = normMean ? 3 : 1
    self.mft[:norm] = lowerBounding ? 1.0 / sqrt(windowSize) : 1.0
    return self
end

function transform_mft(self::MFT, series, wordlength)
    FFT_series = fft(series)
    data_new = []
    windowSize = length(series)

    for i in 1:convert(Int64, ceil(length(series) / 2))
        append!(data_new, real(FFT_series[i]))
        append!(data_new, imag(FFT_series[i]))
    end
    data_new[2] = 0.0
    data_new = data_new[1:self.mft[:windowSize]]

    l = min(windowSize - self.mft[:startOffset], wordlength)
    copy = data_new[(self.mft[:startOffset]+1):(l + self.mft[:startOffset])]
    while length(copy) != wordlength
        append!(copy, 0)
    end

    sign = 1
    for i in 1:length(copy)
        copy[i] *= self.mft[:norm] * sign
        sign *= -1
    end
    return copy
end

function transformWindowing(self::MFT, series_full, wordLength)
    series = series_full.ts[:data]
    if self.mft[:MUSE]
        WORDLENGTH = convert(Int64, max(self.mft[:windowSize], wordLength + self.mft[:startOffset]))
    else
        WORDLENGTH = convert(Int64, min(self.mft[:windowSize], wordLength + self.mft[:startOffset]))
    end
    WORDLENGTH = WORDLENGTH + WORDLENGTH % 2
    phis = [0. for i in 1:WORDLENGTH]
    for u in 1:2:WORDLENGTH
        uHalve = -(u-1) / 2
        phis[u] = cos(2 * pi * uHalve / self.mft[:windowSize])
        phis[u+1] = -sin(2 * pi * uHalve / self.mft[:windowSize])
    end

    final = max(1, length(series) - self.mft[:windowSize] + 1)
    self.mft[:MEANS] = []
    self.mft[:STDS] = []

    self.mft[:MEANS], self.mft[:STDS] = calcIncreamentalMeanStddev(self.mft[:windowSize], series, self.mft[:MEANS], self.mft[:STDS])
    transformed = []

    data = series
    mftData_FFT = []

    for t in 1:final
        if t > 1
            k = 1
            while k <= WORDLENGTH
                real1 = mftData_FFT[k] + data[t + self.mft[:windowSize]-1] - data[t-1]
                imag1 = mftData_FFT[k + 1]

                real = (real1 * phis[k]) - (imag1 * phis[k + 1])
                imag = (real1 * phis[k + 1]) + (imag1 * phis[k])
                mftData_FFT[k] = real
                mftData_FFT[k + 1] = imag
                k += 2
            end
        else
            mftData_fft = fft(data[1:self.mft[:windowSize]])
            mftData_FFT = [0. for _ in 1:WORDLENGTH]

            i = 1
            for j in 1:min(self.mft[:windowSize], WORDLENGTH)
                if j % 2 == 1
                    mftData_FFT[j] = real(mftData_fft[i])
                else
                    mftData_FFT[j] = imag(mftData_fft[i])
                    i += 1
                end
            end

            mftData_FFT[2] = 0.
        end

        copy = [0. for i in 1:wordLength]
        copy_value = mftData_FFT
        try
            copy_value = mftData_FFT[(self.mft[:startOffset]+1):(self.mft[:startOffset] + wordLength)]
        catch
            copy_value = mftData_FFT[(self.mft[:startOffset]+1):end]
        end
        for s in 1:length(copy_value)
            copy[s] = copy_value[s]
        end
        copy = __init__(timeseries(Dict()), copy, series_full.ts[:label], series_full.ts[:NORM_CHECK])
        copy = normalizeFT(self, copy, self.mft[:STDS][t])
        append!(transformed, [copy])
    end
    return transformed
end

function normalizeFT(self::MFT, copy, std)
    normalisingFactor = (copy.ts[:NORM_CHECK]) & (std > 0) ? 1. / std  : 1.
    normalisingFactor *= self.mft[:norm]

    sign = 1
    for i in 1:length(copy.ts[:data])
        copy.ts[:data][i] *= sign * normalisingFactor
        sign *= -1
    end
    return copy.ts[:data]
end
