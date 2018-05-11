include("/Users/Sam/Documents/SFA_Julia/src/timeseries/TimeSeriesLoader.jl")
include("/Users/Sam/Documents/SFA_Julia/src/classification/MUSEClassifier.jl")


Datasets = [#"PenDigits",
             # "ShapesRandom",
             "DigitShapeRandom",
             # "ECG",
             # "JapaneseVowels",
             # "Libras"
]


for data in Datasets
    train, test = mv_load(data, true)

    #The MUSE Classifier
    tic()
    muse = init_MUSEClassifier(MUSEClassifier(Dict()), data)
    scoreMUSE = eval_MUSEClassifier(muse, train, test)[1]
    println(string(data,"; ",scoreMUSE), " ", toq())
end
