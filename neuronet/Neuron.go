package neuronet

import (
    "math"
    "math/rand"
)

type Neuron struct {
    Weights           []float64
    LastNet           float64
    LastOut           float64
    LastInputVector   []float64
    Optimizer         *Optimizer
    ObjectiveFunction *ObjectiveFunction
}

func (neuron *Neuron) InitWeights(count int64) {
    // Добавляем к весам вес сигнала "смещения"
    neuron.Weights = make([]float64, count + 1, count + 1)

    // Инициализируем значение весов
    for w := 0; w < len(neuron.Weights); w++ {
        neuron.Weights[w] = (rand.NormFloat64() - rand.NormFloat64()) / math.MaxFloat64
    }
}

func (neuron *Neuron) fNet(x []float64) float64 {
    var result float64

    for i := 0; i < len(neuron.Weights); i++ {
        result += x[i] * neuron.Weights[i]
    }

    return result
}

func (neuron *Neuron) Query(x []float64) float64 {
    // Сигнал нейрона смещения
    x = append(x, 1)

    neuron.LastInputVector = x
    neuron.LastNet = neuron.fNet(x)
    neuron.LastOut = neuron.GetActivateF(neuron.LastNet)

    return neuron.LastOut
}

func (neuron *Neuron) GetActivateF(fNetValue float64) float64 {
    return 1 / (1 + math.Exp(-fNetValue))
}

func (neuron *Neuron) GetActivateF1(fNetValue float64) float64 {
    activateF1 := neuron.GetActivateF(fNetValue)
    return activateF1 * (1 - activateF1)
}