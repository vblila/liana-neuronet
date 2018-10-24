package neuronet

type Teacher struct {
    Method              *Method
    RequiredTeachingMSE float64
    TactsLimit          int64
    Tacts               int64
    isManualStopped     bool
    tactListeners       []func()
}

func (teacher *Teacher) Start(ann AnnInterface, inputData [][]float64, targetData [][]float64)  {
    teacher.isManualStopped = false
    teacher.Tacts = 0

    for t := int64(1); t <= teacher.TactsLimit; t++ {
        teacher.Method.ExecuteTact(t, ann, &inputData, &targetData)

        teachingMse := CalculateAnnMse(ann, inputData, targetData)
        ann.SetTeachingMse(teachingMse)

        for _, f := range teacher.tactListeners {
            (f)()
        }

        if teachingMse <= teacher.RequiredTeachingMSE {
            break
        }

        teacher.Tacts++

        if teacher.isManualStopped {
            break
        }
    }
}

func (teacher *Teacher) Stop() {
    teacher.isManualStopped = true
}

func (teacher *Teacher) AddTactListener(f func()) {
    teacher.tactListeners = append(teacher.tactListeners, f)
}