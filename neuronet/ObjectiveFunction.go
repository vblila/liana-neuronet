package neuronet

type ObjectiveFunction struct {
   Neuron *Neuron
}

// Целевая функция минимизации ошибки нейрона: Функция наименьших квадратов
func (objective *ObjectiveFunction) F1(optimizerTarget float64) []float64 {

    // Функция оптимизации вида 1/2 * (target - out)^2
    // f' = (target - out) * (-1) * out'
    // out' = f'(net) * {x1, x2, ... , xN}		| out' = f'(net) = f'(net) * net' = f'(net) * f'(x,w)
    // Где f'(net) - значение первой производной активационной функции

    neuron := objective.Neuron

    // Формируем вектор значений частных производных
    result := make([]float64, len(neuron.LastInputVector), len(neuron.LastInputVector))
    for i := 0; i < len(neuron.LastInputVector); i++ {
        result[i] = (optimizerTarget - neuron.LastOut) * (-1) * neuron.GetActivateF1(neuron.LastNet) * neuron.LastInputVector[i]
    }

    return result
}