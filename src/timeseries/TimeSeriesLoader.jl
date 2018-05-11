include("/Users/Sam/Documents/SFA_Julia/src/timeseries/TimeSeries.jl")

uv_dir = "/Users/Sam/Documents/SFA_Julia/datasets/univariate/"
mv_dir = "/Users/Sam/Documents/SFA_Julia/datasets/multivariate/"
sep = ' '
extension = ""
#Personal Archieve
uv_dir = "/Archive/"
sep = ','
extension = ".txt"

function uv_load(dataset_name)
    try
        train = Dict()
        test = Dict()

        train_raw = readdlm(string(uv_dir, dataset_name, "\\", dataset_name, "_TRAIN", extension), sep)
        test_raw = readdlm(string(uv_dir, dataset_name, "\\", dataset_name, "_TEST", extension), sep)

        train[:Type] = "UV"
        train[:Samples] = size(train_raw,1)
        train[:Size] = size(train_raw,2)-1
        train[:Labels] = []

        test[:Type] = "UV"
        test[:Samples] = size(test_raw,1)
        test[:Size] = size(test_raw,2)-1
        test[:Labels] = []

        for i in 1:train[:Samples]
            label = Int(train_raw[i,1])
            append!(train[:Labels], label)
            series = train_raw[i,2:end]
            train[i] = __init__(timeseries(Dict()), series, label, true)
            train[i] = NORM(train[i], true)
        end

        for i in 1:test[:Samples]
            label = Int(test_raw[i,1])
            append!(test[:Labels], label)
            series = test_raw[i, 2:end]
            test[i] = __init__(timeseries(Dict()), series, label, true)
            test[i] = NORM(test[i], true)
        end

        println("Done reading ", dataset_name, " Training Data...  Samples: ", train[:Samples], "   Length: ", train[:Size])
        println("Done reading ", dataset_name, " Testing Data...  Samples: ", test[:Samples], "   Length: ", test[:Size])
        println()

        return train, test

    catch
        print("Data not loaded Try changing the data path in the TimeSeriesLoader file")
    end
end

function uv_load_prune(dataset_name, keep)
    try
        train = Dict()
        test = Dict()

        train_raw = readdlm(string(uv_dir, dataset_name, "\\", dataset_name, "_TRAIN", extension), sep)
        test_raw = readdlm(string(uv_dir, dataset_name, "\\", dataset_name, "_TEST", extension), sep)

        train[:Type] = "UV"
        train[:Samples] = length(keep)
        train[:Size] = size(train_raw,2)-1
        train[:Labels] = []

        test[:Type] = "UV"
        test[:Samples] = size(test_raw,1)
        test[:Size] = size(test_raw,2)-1
        test[:Labels] = []

        idx = 1
        for i in 1:size(train_raw,1)
            if i in keep
                label = Int(train_raw[i,1])
                append!(train[:Labels], label)
                series = train_raw[i,2:end]
                train[idx] = __init__(timeseries(Dict()), series, label, true)
                train[idx] = NORM(train[idx], true)
                idx += 1
            end
        end

        for i in 1:test[:Samples]
            label = Int(test_raw[i,1])
            append!(test[:Labels], label)
            series = test_raw[i, 2:end]
            test[i] = __init__(timeseries(Dict()), series, label, true)
            test[i] = NORM(test[i], true)
        end

        println("Done reading ", dataset_name, " Training Data...  Samples: ", train[:Samples], "   Length: ", train[:Size])
        println("Done reading ", dataset_name, " Testing Data...  Samples: ", test[:Samples], "   Length: ", test[:Size])
        println()

        return train, test

    catch
        print("Data not loaded Try changing the data path in the TimeSeriesLoader file")
    end
end

function mv_load(dataset_name, useDerivatives = true)
    try
        train = Dict()
        test = Dict()

        train_raw = readdlm(string(mv_dir, dataset_name, "\\", dataset_name, "_TRAIN3"), ' ')
        test_raw = readdlm(string(mv_dir, dataset_name, "\\", dataset_name, "_TEST3"), ' ')

        train[:Type] = "MV"
        train[:Samples] = convert(Int64, train_raw[end,1])
        if useDerivatives
            train[:Dimensions] = 2*(size(train_raw,2)-3)
        else
            train[:Dimensions] = size(train_raw,2)-3
        end
        train[:Labels] = []

        test[:Type] = "MV"
        test[:Samples] = convert(Int64, test_raw[end,1])
        if useDerivatives
            test[:Dimensions] = 2*(size(test_raw,2)-3)
        else
            test[:Dimensions] = size(test_raw,2)-3
        end
        test[:Labels] = []

        for i in 1:convert(Int64, train_raw[end,1])
            row_info = train_raw[train_raw[:,1] .== i,:]
            label = row_info[1,3]
            append!(train[:Labels], label)
            channel = 1
            train[i] = Dict()
            for j in 4:size(row_info,2)
                series = row_info[:,j]
                train[i][channel] = __init__(timeseries(Dict()), series, label, true)
                channel += 1
                if useDerivatives
                    series2= [0. for _ in 1:(length(series)-1)]
                    for u in 2:length(series)
                        series2[u-1] = series[u] - series[u-1]
                    end
                    train[i][channel] = __init__(timeseries(Dict()), series2, label, true)
                    channel += 1
                end
            end
        end

        for i in 1:convert(Int64, test_raw[end,1])
            row_info = test_raw[test_raw[:,1] .== i,:]
            label = row_info[1,3]
            append!(test[:Labels], label)
            channel = 1
            test[i] = Dict()
            for j in 4:size(row_info,2)
                series = row_info[:,j]
                test[i][channel] = __init__(timeseries(Dict()), series, label, true)
                channel += 1
                if useDerivatives
                    series2= [0. for _ in 1:(length(series)-1)]
                    for u in 2:length(series)
                        series2[u-1] = series[u] - series[u-1]
                    end
                    test[i][channel] = __init__(timeseries(Dict()), series2, label, true)
                    channel += 1
                end
            end
        end

        println("Done reading ",dataset_name," Training Data...  Samples: ", train[:Samples], "   Dimensions: ", train[:Dimensions])
        println("Done reading ",dataset_name," Testing Data...  Samples: ", test[:Samples], "   Dimensions: ", test[:Dimensions])
        println()

        return train, test
    catch
        println("Data not loaded Try changing the data path in the TimeSeriesLoader file")
    end
end
