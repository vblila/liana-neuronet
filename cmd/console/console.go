package console

import (
    "fmt"
    "strconv"
)

func FormatFloat(value float64, precision int64) string {
    return fmt.Sprintf("%." + strconv.FormatInt(precision, 10) + "f", value)
}

func FormatInt(value int64) string {
    return strconv.FormatInt(value, 10)
}

func Error(s string) string {
    return "\x1b[1;31;40m" + s + "\x1b[0m"
}

func Value(s string) string {
    return "\x1b[1;36;40m" + s + "\x1b[0m";
}