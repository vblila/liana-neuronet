package main

import (
    "../neuronet"
    "../parsers"
    "./console"
    "./prediction"
    "encoding/csv"
    "fmt"
    "os"
    "path/filepath"
    "strconv"
    "time"
)

func main() {
    var opts *prediction.EmulatorOptions
    args := os.Args[1:]
    if len(args) == 0 {
        fmt.Println("Эмулятор искусственной нейронной сети. Решение задачи прогнозирования.")
        fmt.Println("Автор: Лила Владимир Борисович, email: v.b.lila@yandex.ru, 2018г")
        fmt.Println()
        fmt.Println("Параметры:")

        params := [][]string{
            {"csv", "Путь к источнику, формат файла csv"},
            {"firstRow", "Номер строки из файла, с которого начинается обучающая выборка (нумерация с нуля)"},
            {"teachLength", "Кол-во значений обучающей выборки"},
            {"testLength", "Кол-во значений тестовой выборки (идут после обучающих примеров)"},
            {"predictLength", "Кол-во прогнозируемых значений продолжения ряда (по умолчанию 3)"},
            {"x", "Номер столбца, содержащего входные значения"},
            {"window", "Размер входного окна (по умолчанию 3)"},
            {"ann", "Тип ИНС: mlp, recurrent (по умолчанию mlp)"},
            {"structure", "Кол-во нейронов по слоям через запятую (по умолчанию по теореме Колмогорова-Арнольда-Морозова)"},
            {"step", "Шаг обучения (по умолчанию 0.3)"},
            {"method", "Метод обучения: bp, hybrid (по умолчанию hybrid)"},
            {"tacts", "Лимит эпох обучения (по умолчанию 1000)"},
            {"mse", "Требуемая MSE на обучающей выборке (по умолчанию 0.001)"},
        }

        for _, param := range params {
            fmt.Println(fmt.Sprintf("  --%-15s %s", param[0], param[1]))
        }

        return
    } else {
        opts = prediction.CreateOptionsFromArgs(args)
    }

    if opts.Csv == "" {
        fmt.Println(console.Error("CSV источник не задан"))
        return
    }

    // 1. Создаем обучающую и тестовую выборки
    csvParser := new(parsers.CsvParser)
    csvParser.Source = opts.Csv

    var denormalizedTeachingInputData [][]float64
    var denormalizedTeachingTargetData [][]float64

    teachingSerie := csvParser.GetData(opts.FirstRow, []int64{opts.InputColumn}, opts.TeachLength)
    for i := 0; i < len(teachingSerie) - int(opts.WindowLength); i++ {
        row := make([]float64, opts.WindowLength, opts.WindowLength)
        for w := 0; w < int(opts.WindowLength); w++ {
            row[w] = teachingSerie[i + w][0]
        }

        denormalizedTeachingInputData = append(denormalizedTeachingInputData, row)
        denormalizedTeachingTargetData = append(denormalizedTeachingTargetData, teachingSerie[i + int(opts.WindowLength)])
    }

    var denormalizedTestingInputData [][]float64
    var denormalizedTestingTargetData [][]float64

    testingSerie := csvParser.GetData(opts.FirstRow + opts.TeachLength - opts.WindowLength, []int64{opts.InputColumn}, opts.TestLength + opts.WindowLength)
    for i := 0; i < len(testingSerie) - int(opts.WindowLength); i++ {
        row := make([]float64, opts.WindowLength, opts.WindowLength)
        for w := 0; w < int(opts.WindowLength); w++ {
            row[w] = testingSerie[i + w][0]
        }

        denormalizedTestingInputData = append(denormalizedTestingInputData, row)
        denormalizedTestingTargetData = append(denormalizedTestingTargetData, testingSerie[i + int(opts.WindowLength)])
    }

    querySerie := append(teachingSerie, testingSerie...)

    denormalizedQueryInputData := append(denormalizedTeachingInputData, denormalizedTestingInputData...)
    denormalizedQueryTargetData := append(denormalizedTeachingTargetData, denormalizedTestingTargetData...)

    fmt.Println(
        "Обработано из csv файла",
        console.Value(console.FormatInt(int64(len(csvParser.ParsedData)))),
        "строк.",
    )

    // 2. Нормализуем данные.

    // Определяем параметры нормализации на основе обучающей и тестовой выборок
    normalization := new(neuronet.Normalization)

    minValue := querySerie[0][0]
    maxValue := querySerie[0][0]
    for i := 0; i < len(querySerie); i++ {
        if minValue > querySerie[i][0] {
            minValue = querySerie[i][0]
        }
        if maxValue < querySerie[i][0] {
            maxValue = querySerie[i][0]
        }
    }
    // Раздвинем границы для прогнозируемых значений
    minValue = minValue - minValue
    maxValue = maxValue + maxValue

    normalization.InputParams = make([]*neuronet.NormalizationParam, opts.WindowLength, opts.WindowLength)
    for i, _ := range denormalizedTestingInputData[0] {
        normalization.InputParams[i] = &neuronet.NormalizationParam{Min:minValue, Max:maxValue}
    }
    normalization.OutputParams = []*neuronet.NormalizationParam{
        &neuronet.NormalizationParam{Min:minValue, Max:maxValue},
    }

    // Нормализуем обучающую выборку
    teachingInputData := normalization.NormalizeInputData(denormalizedTeachingInputData)
    teachingTargetData := normalization.NormalizeOutputData(denormalizedTeachingTargetData)

    // Нормализуем тестовую выборку
    testingInputData := normalization.NormalizeInputData(denormalizedTestingInputData)
    testingTargetData := normalization.NormalizeOutputData(denormalizedTestingTargetData)

    // Нормализуем опросную выборку
    queryInputData := normalization.NormalizeInputData(denormalizedQueryInputData)

    fmt.Println(
        "Выборки сформированы (обуч., тест., опрос.):",
        console.Value(console.FormatInt(int64(len(teachingInputData)))),
        console.Value(console.FormatInt(int64(len(testingInputData)))),
        console.Value(console.FormatInt(int64(len(queryInputData)))),
    )

    // 3. Создаем ИНС
    var ann neuronet.AnnInterface

    if opts.AnnType == "mlp" {
        ann = new(neuronet.MlpAnn)
    } else if opts.AnnType == "recurrent" {
        ann = new(neuronet.RecurrentAnn)
    } else {
        fmt.Println(console.Error("Указан неизвестный тип ИНС"))
        os.Exit(0)
    }
    ann.Init(opts.WindowLength, opts.Structure, opts.StepValue)

    // 4. Настраиваем учителя
    teacher := new(neuronet.Teacher)
    teacher.RequiredTeachingMSE = opts.MSELimit
    teacher.TactsLimit = opts.TactsLimit
    teacher.Method = new(neuronet.Method)
    if opts.Method == "hybrid" {
        teacher.Method.IsMutationUsed = true
    }

    // 5. Обучение
    unixTimestampNanoseconds := time.Now().UnixNano()
    teacher.AddTactListener(func() {
        tactDurationMs := float64(time.Now().UnixNano() - unixTimestampNanoseconds) / 1000000000

        testingMSE := neuronet.CalculateAnnMse(ann, testingInputData, testingTargetData)

        var tactInfo string
        tactInfo = "MSE обуч.: " + console.Value(console.FormatFloat(ann.GetTeachingMse(), 6))
        tactInfo += ", MSE тест.: " + console.Value(console.FormatFloat(testingMSE, 6))
        tactInfo += ", эпох: " + console.Value(console.FormatInt(teacher.Tacts + 1))
        tactInfo += ", время эпохи: " + console.Value(console.FormatFloat(tactDurationMs, 6)) + " сек"
        tactInfo += ", вес[0] нейрона[0][0]: " + console.Value(console.FormatFloat(ann.GetLayers()[0].Neurons[0].Weights[0], 4))
        tactInfo += ", вес[0] нейрона[y][0]: " + console.Value(console.FormatFloat(ann.GetOutputLayer().Neurons[0].Weights[0], 4))

        fmt.Print(tactInfo + "\r")
        unixTimestampNanoseconds = time.Now().UnixNano()
    })

    fmt.Println()
    fmt.Println("Обучение началось. Нажмите Escape, чтобы прервать обучение и получить результаты.")
    teacher.Start(ann, teachingInputData, teachingTargetData)
    fmt.Println()
    fmt.Println("Обучение завершено.")

    // 6. Опрос сети на опросной выборке
    fmt.Println()

    // Нужно сформировать csv файл, состоящий из опросных примеров + заголовка
    csvRows := make([][]string, len(teachingSerie) + len(testingSerie) + int(opts.PredictLength), len(teachingSerie) + len(testingSerie) + int(opts.PredictLength))
    csvRows[0] = []string{"Out series", "Target series"}

    for i := 0; i < int(opts.WindowLength); i++ {
        csvRows[i + 1] = []string{
            strconv.FormatFloat(teachingSerie[i][0], 'g', 8, 64),
            "",
        }
    }

    ann.ResetBeforeQuerying()
    inputVector := queryInputData[0]
    for r := 0; r < len(queryInputData) + 3; r++ {
        outputVector := ann.Query(inputVector)
        denormalizedOutputVector := normalization.DenormalizeOutputVector(outputVector)

        targetString := ""
        if r < len(denormalizedQueryTargetData) {
            targetString = strconv.FormatFloat(denormalizedQueryTargetData[r][0], 'g', 8, 64)
        }

        csvRows[r + 1 + int(opts.WindowLength)] = []string{
            strconv.FormatFloat(denormalizedOutputVector[0], 'g', 8, 64),
            targetString,
        }

        for i := 1; i < len(inputVector); i++ {
            inputVector[i - 1] = inputVector[i]
        }

        inputVector[len(inputVector) - 1] = outputVector[0]

        fmt.Print("Опрос сети. Значений ряда опрошено: " + console.Value(strconv.FormatInt(int64(r + 1 + int(opts.WindowLength)), 10)) + "\r")
    }
    fmt.Println()

    dir, _ := filepath.Abs(filepath.Dir(os.Args[0]))
    csvDestination := dir + "/results.csv"
    csvFile, err := os.Create(csvDestination)
    if err != nil {
        fmt.Println(console.Error(err.Error()))
    }
    defer csvFile.Close()

    writer := csv.NewWriter(csvFile)
    defer writer.Flush()

    for _, value := range csvRows {
        err := writer.Write(value)
        if err != nil {
            fmt.Println(console.Error(err.Error()))
        }
    }

    fmt.Println()
    fmt.Println("Результаты опроса ИНС выгружены в csv:", csvDestination)
}