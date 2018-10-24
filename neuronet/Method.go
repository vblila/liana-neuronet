package neuronet

import (
    "math/rand"
    "time"
)

type Method struct {
    IsMutationUsed bool
}

func (method *Method) ExecuteTact(tactNumber int64, ann AnnInterface, inputData *[][]float64, targetData *[][]float64) {
    // В 1 такт обучения входит 1 эпоха обучения (подача всех примеров с распространением ошибки)

    ann.ResetBeforeQuerying()
    for i := 0; i < len(*inputData); i++ {
        var vectorIndex int64
        if ann.IsRecurrentLayersExist() {
            vectorIndex = int64(i)
        } else {
            vectorIndex = rand.Int63n(int64(len(*inputData)))
        }
        inputVector := (*inputData)[vectorIndex]
        targetVector := (*targetData)[vectorIndex]

        // Выходной вектор здесь нам не нужен, после опроса сети нейрон хранит последнее состояние
        ann.Query(inputVector)

        method.goBack(ann, targetVector)
    }

    if method.IsMutationUsed {
        rand.Seed(time.Now().UnixNano())

        flatWeightsDump := DumpAnnFlatWeights(ann)

        mse := CalculateAnnMse(ann, *inputData, *targetData)
        newMse := 0.0
        layers := ann.GetLayers()
        for i := 0; i < 10; i++ {
            for l := 0; l < len(layers); l++ {
                for n := 0; n < len(layers[l].Neurons); n++ {
                    if rand.Float64() < 0.5 {
                        continue
                    }

                    for w := 0; w < len(layers[l].Neurons[n].Weights); w++ {
                        mutation := float64(float64(rand.Int63n(1000000) - rand.Int63n(1000000)) / 1000000)
                        layers[l].Neurons[n].Weights[w] += mutation
                    }
                }
            }

            ann.ResetBeforeQuerying()
            newMse = CalculateAnnMse(ann, *inputData, *targetData)
            if newMse < mse {
                break
            }
        }

        if newMse > mse {
            AcceptAnnFlatWeights(ann, flatWeightsDump)
        }
    }
}

func (method *Method) goBack(ann AnnInterface, targetVector []float64) {
    // Считаем target целевой функции (не нейрона) выходного слоя
    for n := 0; n < len(ann.GetOutputLayer().Neurons); n++ {
        neuron := ann.GetOutputLayer().Neurons[n]

        // Sigma нужна для обратного распространения ошибки.
        // На основе ее значения вычисляется Target-значение для нейрона.
        neuron.Optimizer.Sigma = neuron.GetActivateF1(neuron.LastNet) * (targetVector[n] - neuron.LastOut)

        // Можно записать как в учебниках:
        // neuron.Optimizer.Target = neuron.Optimizer.Sigma / neuron.GetActivateF1(neuron.LastNet) + neuron.LastOut
        // Мы запишем уже известное нам значение
        neuron.Optimizer.Target = targetVector[n]

        p := neuron.Optimizer.Vector()
        step := neuron.Optimizer.StepValue
        for w := 0; w < len(neuron.Weights); w++ {
            neuron.Weights[w] += step * p[w]
        }
    }

    // Идем по скрытым слоям для входного слоя включительно
    layers := ann.GetLayers()
    for l := len(layers) - 2; l >= 0; l-- {
        for n := 0; n < len(layers[l].Neurons); n++ {
            neuron := layers[l].Neurons[n]

            // Обнулим sigma у каждого нейрона на текущем слое
            neuron.Optimizer.Sigma = 0

            // Рассчитываем sigma для каждого нейрона
            // sigma вектора i на слое l = E ('wi вектора k слоя l+1' * 'sigma вектора k слоя l+1') * 'p - вектор направления в общем случае'
            for k := 0; k < len(layers[l + 1].Neurons); k++ {
                layerNeuron := layers[l + 1].Neurons[k]
                neuron.Optimizer.Sigma += layerNeuron.Weights[n] * layerNeuron.Optimizer.Sigma
            }

            // Досчитываем Sigma
            neuron.Optimizer.Sigma *= neuron.GetActivateF1(neuron.LastNet)

            // Вычисляем для каждого нейрона target
            // по формулам по аналогии выходного слоя получается именно так
            neuron.Optimizer.Target = neuron.Optimizer.Sigma / neuron.GetActivateF1(neuron.LastNet) + neuron.LastOut

            p := neuron.Optimizer.Vector()
            step := neuron.Optimizer.StepValue
            for w := 0; w < len(neuron.Weights); w++ {
                neuron.Weights[w] += step * p[w]
            }
        }
    }
}