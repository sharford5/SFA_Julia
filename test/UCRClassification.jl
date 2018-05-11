include("/Users/Sam/Documents/SFA_Julia/src/timeseries/TimeSeriesLoader.jl")
include("/Users/Sam/Documents/SFA_Julia/src/classification/WEASELClassifier.jl")
include("/Users/Sam/Documents/SFA_Julia/src/classification/BOSSEnsembleClassifier.jl")
include("/Users/Sam/Documents/SFA_Julia/src/classification/BOSSVSClassifier.jl")
include("/Users/Sam/Documents/SFA_Julia/src/classification/ShotgunEnsembleClassifier.jl")
include("/Users/Sam/Documents/SFA_Julia/src/classification/ShotgunClassifier.jl")


Datasets = [#"Coffee",
            # "Beef",
            # "ECG200",
            # "Gun_Point",
            # "BeetleFly",
            "ItalyPowerDemand",
            "TwoLeadECG",
            "SonyAIBORobotSurface",
            # "MiddlePhalanxTW",
            # "Earthquakes"
            ]

# Datasets = ["ItalyPowerDemand",
#             "50words",
#             "Worms",
#             "WormsTwoClass",
#             "Coffee",
#             "Gun_Point",
#             "Ham",
#             "Computers",
#             "Trace",
#             "ArrowHead",
#             "FacesUCR",
#             "ChlorineConcentration",
#             "SmallKitchenAppliances",
#             "wafer",
#             "wine",
#             "MedicalImages",
#             "BeetleFly",
#             "BirdChicken",
#             "Phoneme",
#             "MiddlePhalanxOutlineAgeGroup",
#             "MiddlePhalanxOutlineCorrect",
#             "MiddlePhalanxTW",
#             "FaceAll",
#             "Plane",
#             "ProximalPhalanxOutlineAgeGroup",
#             "ProximalPhalanxOutlineCorrect",
#             "ProximalPhalanxTW",
#             "RefrigerationDevices",
#             "ShapeletSim",
#             "OSULeaf",
#             "ShapesAll",
#             "Car",
#             "Adiac",
#             "yoga",
#             "LargeKitchenAppliances",
#             "HandOutlines",
#             "fish",
#             "Lighting7",
#             "Meat",
#             "Lighting2",
#             "synthetic_control",
#             "CinC_ECG_torso",
#             "MALLAT",
#             "Symbols",
#             "ECG200",
#             "ECG5000",
#             "ElectricDevices",
#             "FaceFour",
#             "OliveOil",
#             "Beef",
#             "DiatomSizeReduction",
#             "DistalPhalanxOutlineAgeGroup",
#             "PhalangesOutlinesCorrect",
#             "Strawberry",
#             "DistalPhalanxOutlineCorrect",
#             "DistalPhalanxTW",
#             "InsectWingbeatSound",
#             "CBF",
#             "ECGFiveDays",
#             "TwoLeadECG",
#             "SonyAIBORobotSurfaceII",
#             "MoteStrain",
#             "SonyAIBORobotSurface",
#             "Two_Patterns",
#             "uWaveGestureLibrary_X",
#             "uWaveGestureLibrary_Y",
#             "uWaveGestureLibraryAll",
#             "StarLightCurves",
#             "Haptics",
#             "SwedishLeaf",
#             "ToeSegmentation1",
#             "ToeSegmentation2",
#             "InlineSkate",
#             "NonInvasiveFatalECG_Thorax1",
#             "NonInvasiveFatalECG_Thorax2",
#             "Cricket_Y",
#             "Cricket_X",
#             "Cricket_Z",
#             "Herring",
#             "Earthquakes",
#             "FordA",
#             "FordB",
#             "WordsSynonyms",
#             "ScreenType",
#             "uWaveGestureLibrary_Z"
# ]



for data in Datasets
    train, test = uv_load(data)

    #The WEASEL Classifier
    tic()
    weasel = init_WEASELClassifier(WEASELClassifier(Dict()), data)
    scoreWEASEL = eval_WEASELClassifier(weasel, train, test)[1]
    println(string(data,"; ",scoreWEASEL), " ", toq())

    #The BOSS Ensemble Classifier
    tic()
    boss = init_BOSSEnsembleClassifier(BOSSEnsembleClassifier(Dict()), data)
    scoreBOSS = eval_BOSSEnsembleClassifier(boss, train, test)[1]
    println(string(data,"; ",scoreBOSS), " ", toq())

    # #The BOSS VS Classifier
    # tic()
    # bossVS = init_BOSSVSClassifier(BOSSVSClassifier(Dict()), data)
    # scoreBOSSVS = eval_BOSSVSClassifier(bossVS, train, test)[1]
    # println(string(data,"; ",scoreBOSSVS))

    # #The Shotgun Ensemble Classifier
    # tic()
    # shotgunEnsemble = init_ShotgunEnsembleClassifier(ShotgunEnsembleClassifier(Dict()), data)
    # scoreShotgunEnsemble = eval_ShotgunEnsembleClassifier(shotgunEnsemble, train, test)[1]
    # println(string(data,"; ",scoreShotgunEnsemble), " ", toq())

    # #The Shotgun Classifier
    # tic()
    # shotgun = init_ShotgunClassifier(ShotgunClassifier(Dict()), data)
    # scoreShotgun = eval_ShotgunClassifier(shotgun, train, test)[1]
    # println(string(data,"; ",scoreShotgun), " ", toq())
end
