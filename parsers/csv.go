package parsers

import (
    "bufio"
    "encoding/csv"
    "io"
    "os"
    "strconv"
    "strings"
)

type CsvParser struct {
    Source           string
    ParsedData       [][]float64
    isDataParsed     bool
    readRowListeners []*func()
}

func (parser *CsvParser) GetData(startStringNumber int64, columnNumbers []int64, length int64) [][]float64 {
    parser.initParsedData()

    data := make([][]float64, length, length)
    d := 0
    for i := startStringNumber; i < startStringNumber + length; i++ {
        row := make([]float64, len(columnNumbers), len(columnNumbers))
        for c, columnNumber := range columnNumbers {
            row[c] = parser.ParsedData[i][columnNumber]
        }

        data[d] = row
        d++
    }

    return data
}

func (parser *CsvParser) initParsedData() {
    if parser.isDataParsed {
        return
    }

    csvFile, error := os.Open(parser.Source)
    if error != nil {
        panic(error)
    }
    reader := csv.NewReader(bufio.NewReader(csvFile))

    for {
        line, error := reader.Read()
        if error == io.EOF {
            break
        } else if error != nil {
            panic(error)
        }

        row := make([]float64, len(line), len(line))
        for s := 0; s < len(line); s++ {
            line[s] = strings.Replace(line[s], ",", ".", -1)
            row[s], _ = strconv.ParseFloat(line[s], 64)
        }

        parser.ParsedData = append(parser.ParsedData, row)
    }

    parser.isDataParsed = true
}