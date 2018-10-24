package neuronet

type Optimizer struct {
    Sigma     float64
    Target    float64
    StepValue float64
    Neuron    *Neuron
}

func (optimizer *Optimizer) antiGradient() []float64 {
    // Значение первой производной функции ошибки в точке - вектор градиента
    neuron := optimizer.Neuron
    gradient := neuron.ObjectiveFunction.F1(optimizer.Target)

    antiGradient := gradient
    for i := 0; i < len(antiGradient); i++ {
        antiGradient[i] = -1 * gradient[i]
    }

    return antiGradient
}

func (optimizer *Optimizer) Vector() []float64 {
    return optimizer.antiGradient()
}