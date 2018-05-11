include("/Users/Sam/Documents/SFA_Julia/src/timeseries/TimeSeriesLoader.jl")
include("/Users/Sam/Documents/SFA_Julia/src/transformation/SFA.jl")

symbols = 8
wordLength = 8
normMean = false

function sfaToWord(word)
    word_string = ""
    alphabet = "abcdefghijklmnopqrstuv"
    for w in word
        word_string = string(word_string,alphabet[w+1])
    end
    return word_string
end

train, test = uv_load("Gun_Point")

sfa = init_sfa(SFA(Dict(:initialized=>false, :HistogramType=>"EQUI_DEPTH")))
sfa, _ = fitTransform(sfa, train, wordLength, symbols, normMean)
printBins(sfa)


for i in 1:test[:Samples]
    wordList = transform2(sfa, test[i].ts[:data], "null")
    println(string(i, "-th transformed time series SFA word ", "\t", sfaToWord(wordList)))
end
