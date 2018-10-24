package neuronet

type MlpAnn struct {
    Layers      []*Layer
    teachingMSE float64
}

func (ann *MlpAnn) Query(x []float64) []float64 {
    var layerX []float64
    layerX = append(layerX, x...)

    var layerY []float64

    for l := 0; l < len(ann.Layers); l++ {
        layerNeuronsCount := len(ann.Layers[l].Neurons)
        layerY = make([]float64, layerNeuronsCount, layerNeuronsCount)
        for n := 0; n < layerNeuronsCount; n++ {
            layerY[n] = ann.Layers[l].Neurons[n].Query(layerX)
        }

        layerX = make([]float64, layerNeuronsCount, layerNeuronsCount)
        copy(layerX, layerY)
    }

    return layerX
}

func (ann *MlpAnn) GetOutputLayer() *Layer {
    return ann.Layers[len(ann.Layers) - 1]
}

func (ann *MlpAnn) GetLayers() []*Layer {
    return ann.Layers
}

func (ann *MlpAnn) GetTeachingMse() float64 {
    return ann.teachingMSE
}

func (ann *MlpAnn) SetTeachingMse(mse float64) {
    ann.teachingMSE = mse
}

func (ann *MlpAnn) ResetBeforeQuerying() {
    // MLP не нуждается в подготовке перед опросом
}

func (ann *MlpAnn) Init(inputLength int64, structure []int64, optimizerStep float64) {
    ann.Layers = make([]*Layer, len(structure), len(structure))
    for l := 0; l < len(structure); l++ {
        layer := new(Layer)
        layer.Neurons = make([]*Neuron, structure[l], structure[l])

        for n := int64(0); n < structure[l]; n++ {
            neuron := new(Neuron)

            // Создаем веса нейрона
            var weightsLength int64
            if l == 0 {
                weightsLength = inputLength
            } else {
                weightsLength = structure[l - 1]
            }
            neuron.InitWeights(weightsLength)

            // Создаем опитимизатора фукнции ошибки нейрона
            neuron.Optimizer = new(Optimizer)
            neuron.Optimizer.StepValue = optimizerStep
            neuron.Optimizer.Neuron = neuron

            neuron.ObjectiveFunction = new(ObjectiveFunction)
            neuron.ObjectiveFunction.Neuron = neuron

            layer.Neurons[n] = neuron
        }

        ann.Layers[l] = layer
    }
}

func (ann *MlpAnn) IsRecurrentLayersExist() bool {
    return false
}