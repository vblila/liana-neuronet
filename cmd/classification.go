package main

import (
    "../neuronet"
    "../parsers"
    "./classification"
    "./console"
    "encoding/csv"
    "fmt"
    "os"
    "path/filepath"
    "strconv"
    "time"
)

func main() {
    var opts *classification.EmulatorOptions
    args := os.Args[1:]
    if len(args) == 0 {
        fmt.Println("Эмулятор искусственной нейронной сети. Решение задачи классификации.")
        fmt.Println("Автор: Лила Владимир Борисович, email: v.b.lila@yandex.ru, 2018г")
        fmt.Println()
        fmt.Println("Параметры:")

        params := [][]string{
            {"csv", "Путь к источнику, формат файла csv"},
            {"firstRow", "Номер строки из файла, с которого начинается обучающая выборка (нумерация с нуля)"},
            {"teachLength", "Кол-во примеров обучающей выборки"},
            {"testLength", "Кол-во примеров тестовой выборки (идут после примеров обучающей выборки)"},
            {"queryLength", "Кол-во примеров для опроса сети (по умолчанию, teachLength + testLength)"},
            {"x", "Номера столбцов, содержащих входные сигналы (через запятую, нумерация с нуля)"},
            {"y", "Номера столбцов, содержащих выходные сигналы (через запятую, нумерация с нуля)"},
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
        opts = classification.CreateOptionsFromArgs(args)
    }

    if opts.Csv == "" {
        fmt.Println(console.Error("CSV источник не задан"))
        return
    }

    // 1. Создаем обучающую и тестовую выборки
    csvParser := new(parsers.CsvParser)
    csvParser.Source = opts.Csv

    denormalizedTeachingInputData := csvParser.GetData(opts.FirstRow, opts.InputColumns, opts.TeachLength)
    denormalizedTeachingTargetData := csvParser.GetData(opts.FirstRow, opts.OutputColumns, opts.TeachLength)

    var denormalizedTestingInputData [][]float64
    var denormalizedTestingTargetData [][]float64
    if opts.TestLength > 0 {
        denormalizedTestingInputData = csvParser.GetData(opts.FirstRow + opts.TeachLength, opts.InputColumns, opts.TestLength)
        denormalizedTestingTargetData = csvParser.GetData(opts.FirstRow + opts.TeachLength, opts.OutputColumns, opts.TestLength)
    }

    var denormalizedQueryInputData [][]float64
    var denormalizedQueryTargetData [][]float64
    if opts.QueryLength > 0 {
        denormalizedQueryInputData = csvParser.GetData(opts.FirstRow, opts.InputColumns, opts.QueryLength)
        denormalizedQueryTargetData = csvParser.GetData(opts.FirstRow, opts.OutputColumns, opts.QueryLength)
    } else {
        denormalizedQueryInputData = append(denormalizedTeachingInputData, denormalizedTestingInputData...)
        denormalizedQueryTargetData = append(denormalizedTeachingTargetData, denormalizedTestingTargetData...)
    }

    fmt.Println(
        "Обработано из csv файла",
        console.Value(console.FormatInt(int64(len(csvParser.ParsedData)))),
        "строк.",
    )

    // 2. Нормализуем данные.

    // Определяем параметры нормализации на основе обучающей и тестовой выборок
    normalization := new(neuronet.Normalization)
    normalization.CalculateInputParams(denormalizedQueryInputData)
    normalization.CalculateOutputParams(denormalizedQueryTargetData)

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
    ann.Init(int64(len(opts.InputColumns)), opts.Structure, opts.StepValue)

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
    csvRows := make([][]string, len(queryInputData) + 1, len(queryInputData) + 1)

    xLength := len(teachingInputData[0])
    yLength := len(teachingTargetData[0])

    csvRows[0] = make([]string, xLength + yLength * 2, xLength + yLength * 2)
    for c, _ := range teachingInputData[0] {
        csvRows[0][c] = "Input " + strconv.FormatInt(int64(c), 10)
    }
    for c, _ := range teachingTargetData[0] {
        csvRows[0][c + xLength] = "Output " + strconv.FormatInt(int64(c), 10)
    }
    for c, _ := range teachingTargetData[0] {
        csvRows[0][c + xLength + yLength] = "Target " + strconv.FormatInt(int64(c), 10)
    }

    ann.ResetBeforeQuerying()
    for r, inputVector := range queryInputData {
        denormalizedOutputVector := normalization.DenormalizeOutputVector(ann.Query(inputVector))

        xLength := len(inputVector)
        yLength := len(denormalizedOutputVector)

        csvRows[r + 1] = make([]string, xLength + yLength * 2, xLength + yLength * 2)

        for c, value := range denormalizedQueryInputData[r] {
           csvRows[r + 1][c] = strconv.FormatFloat(value, 'g', 8, 64)
        }

        for c, value := range denormalizedOutputVector {
            csvRows[r + 1][c + xLength] = strconv.FormatFloat(value, 'g', 8, 64)
        }

        for c, value := range denormalizedQueryTargetData[r] {
            csvRows[r + 1][c + xLength + yLength] = strconv.FormatFloat(value, 'g', 8, 64)
        }

        fmt.Print("Опрос сети. Примеров опрошено: " + console.Value(strconv.FormatInt(int64(r + 1), 10)) + "\r")
    }

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