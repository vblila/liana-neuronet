package neuronet

type NormalizationParam struct {
    Min float64
    Max float64
}

type Normalization struct {
    InputParams []*NormalizationParam
    OutputParams []*NormalizationParam
}

func (normalization *Normalization) NormalizeVector(vector []float64, params []*NormalizationParam) []float64 {
    result := make([]float64, len(vector), len(vector))

    for i := 0; i < len(vector); i++ {
        result[i] = (vector[i] - params[i].Min) / (params[i].Max - params[i].Min)
    }

    return result
}

func (normalization *Normalization) DenormalizeVector(vector []float64, params []*NormalizationParam) []float64 {
    result := make([]float64, len(vector), len(vector))

    for i := 0; i < len(vector); i++ {
        result[i] = vector[i] * (params[i].Max - params[i].Min) + params[i].Min
    }

    return result
}

func (normalization *Normalization) NormalizeInputVector(vector []float64) []float64 {
    return normalization.NormalizeVector(vector, normalization.InputParams)
}

func (normalization *Normalization) DenormalizeInputVector(vector []float64) []float64 {
    return normalization.DenormalizeVector(vector, normalization.InputParams)
}

func (normalization *Normalization) NormalizeOutputVector(vector []float64) []float64 {
    return normalization.NormalizeVector(vector, normalization.OutputParams)
}

func (normalization *Normalization) DenormalizeOutputVector(vector []float64) []float64 {
    return normalization.DenormalizeVector(vector, normalization.OutputParams)
}

func (normalization *Normalization) CalculateInputParams(inputData [][]float64) {
    normalization.InputParams = nil
    for _, value := range inputData[0] {
        param := new(NormalizationParam)
        param.Min = value
        param.Max = value
        normalization.InputParams = append(normalization.InputParams, param)
    }

    for _, vector := range inputData {
        for column, value := range vector {
            if normalization.InputParams[column].Min > value {
                normalization.InputParams[column].Min = value
            }
            if normalization.InputParams[column].Max < value {
                normalization.InputParams[column].Max = value
            }
        }
    }
}

func (normalization *Normalization) CalculateOutputParams(outputData [][]float64) {
    normalization.OutputParams = nil
    for _, value := range outputData[0] {
        param := new(NormalizationParam)
        param.Min = value
        param.Max = value
        normalization.OutputParams = append(normalization.OutputParams, param)
    }

    for _, vector := range outputData {
        for column, value := range vector {
            if normalization.OutputParams[column].Min > value {
                normalization.OutputParams[column].Min = value
            }
            if normalization.OutputParams[column].Max < value {
                normalization.OutputParams[column].Max = value
            }
        }
    }
}

func (normalization *Normalization) NormalizeInputData(data [][]float64) [][]float64 {
    result := make([][]float64, len(data), len(data))

    for row, vector := range data {
        result[row] = normalization.NormalizeInputVector(vector)
    }

    return result
}

func (normalization *Normalization) NormalizeOutputData(data [][]float64) [][]float64 {
    result := make([][]float64, len(data), len(data))

    for row, vector := range data {
        result[row] = normalization.NormalizeOutputVector(vector)
    }

    return result
}