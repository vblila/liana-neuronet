package neuronet

type RecurrentAnn struct {
    Layers             []*Layer
    teachingMSE        float64
    contextInputVector []float64
}

func (ann *RecurrentAnn) Query(x []float64) []float64 {
    var layerX []float64
    layerX = append(layerX, x...)
    layerX = append(layerX, ann.contextInputVector...)

    var layerY []float64

    for l := 0; l < len(ann.Layers); l++ {
        layerNeuronsCount := len(ann.Layers[l].Neurons)
        layerY = make([]float64, layerNeuronsCount, layerNeuronsCount)
        for n := 0; n < layerNeuronsCount; n++ {
            layerY[n] = ann.Layers[l].Neurons[n].Query(layerX)
        }

        if l == 0 {
            ann.contextInputVector = layerY
        }

        layerX = make([]float64, layerNeuronsCount, layerNeuronsCount)
        copy(layerX, layerY)
    }

    return layerX
}

func (ann *RecurrentAnn) GetOutputLayer() *Layer {
    return ann.Layers[len(ann.Layers) - 1]
}

func (ann *RecurrentAnn) GetLayers() []*Layer {
    return ann.Layers
}

func (ann *RecurrentAnn) GetTeachingMse () float64 {
    return ann.teachingMSE
}

func (ann *RecurrentAnn) SetTeachingMse (mse float64) {
    ann.teachingMSE = mse
}

func (ann *RecurrentAnn) ResetBeforeQuerying() {
    ann.contextInputVector = make([]float64, len(ann.Layers[0].Neurons), len(ann.Layers[0].Neurons))
}

func (ann *RecurrentAnn) Init(inputLength int64, structure []int64, optimizerStep float64) {
    ann.Layers = make([]*Layer, len(structure), len(structure))
    for l := 0; l < len(structure); l++ {
        layer := new(Layer)
        layer.Neurons = make([]*Neuron, structure[l], structure[l])

        for n := int64(0); n < structure[l]; n++ {
            neuron := new(Neuron)

            // Создаем веса нейрона
            var weightsLength int64
            if l == 0 {
                // В первом скрытом слое будут подаваться значения его выхода на предыдущем опросном примере
                weightsLength = inputLength * 2
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

func (ann *RecurrentAnn) IsRecurrentLayersExist() bool {
    return true
}