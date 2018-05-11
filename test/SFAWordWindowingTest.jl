include("/Users/Sam/Documents/SFA_Julia/src/timeseries/TimeSeriesLoader.jl")
include("/Users/Sam/Documents/SFA_Julia/src/transformation/SFA.jl")

symbols = 4
wordLength = 4
windowLength = 64
normMean = true


function sfaToWord(word)
    word_string = ""
    alphabet = "abcdefghijklmnopqrstuv"
    for w in word
        word_string = string(word_string,alphabet[w+1])
    end
    return word_string
end

function sfaToWordList(wordList)
    list_string = ""
    for word in wordList
        list_string = string(list_string,sfaToWord(word))
        list_string = string(list_string, "; ")
    end
    return list_string
end

train, test = uv_load("Gun_Point")

sfa = init_sfa(SFA(Dict(:initialized=>false, :HistogramType=>"EQUI_DEPTH")))
sfa = fitWindowing(sfa, train, windowLength, wordLength, symbols, normMean, true)
printBins(sfa)


for i in 1:2# 1:test[:Samples]
    wordList = transformWindowing(sfa, test[i])
    println(string(i, "-th transformed time series SFA word ", "\t", sfaToWordList(wordList)))
end
