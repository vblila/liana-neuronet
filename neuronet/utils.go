package neuronet

import (
    "math"
)

func CalculateAnnMse(ann AnnInterface, inputData [][]float64, targetData [][]float64) float64 {
    if len(inputData) == 0 {
        return 0
    }

    var mse float64
    for i := 0; i < len(inputData); i++ {
        mse += CalculateAnnVectorMse(ann, inputData[i], targetData[i])
    }

    return mse / float64(len(inputData))
}

func CalculateAnnVectorMse(ann AnnInterface, inputVector []float64, targetVector []float64) float64 {
    var vectorMSE float64

    outputVector := ann.Query(inputVector)
    for i := 0; i < len(targetVector); i++ {
        vectorMSE += math.Pow(targetVector[i] - outputVector[i], 2)
    }
    vectorMSE = vectorMSE / float64(len(targetVector))

    return vectorMSE
}

func DumpAnnFlatWeights(ann AnnInterface) []float64 {
    var dumpWeights []float64
    layers := ann.GetLayers()
    for l := 0; l < len(layers); l++ {
        for n := 0; n < len(layers[l].Neurons); n++ {
            dumpWeights = append(dumpWeights, layers[l].Neurons[n].Weights...)
        }
    }

    return dumpWeights
}

func AcceptAnnFlatWeights (ann AnnInterface, weights []float64) {
    wI := 0

    layers := ann.GetLayers()
    for l := 0; l < len(layers); l++ {
        for n := 0; n < len(layers[l].Neurons); n++ {
            for w := 0; w < len(layers[l].Neurons[n].Weights); w++ {
                layers[l].Neurons[n].Weights[w] = weights[wI]
                wI++
            }
        }
    }
}