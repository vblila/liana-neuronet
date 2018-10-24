package neuronet

// Artificial neural network (ANN) Interface
type AnnInterface interface {
    Query(x []float64) []float64
    GetLayers() []*Layer
    GetOutputLayer() *Layer
    SetTeachingMse (mse float64)
    GetTeachingMse() float64
    ResetBeforeQuerying()
    Init(inputLength int64, structure []int64, optimizerStep float64)
    IsRecurrentLayersExist() bool
}

