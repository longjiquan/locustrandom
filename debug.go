package locustrandom

import (
	"fmt"
	"os"
	"strconv"
	"sync"
)

var InsertLatencys []int
var SearchLatencys []int

var insertLock sync.RWMutex
var searchLock sync.RWMutex

func save(f string, latencys []int) {
	fmt.Println("write to: ", f, ", num: ", len(latencys))

	file, err := os.OpenFile(f, os.O_WRONLY|os.O_CREATE, os.ModePerm)
	if err != nil {
		panic(err)
	}
	defer file.Close()

	for _, latency := range latencys {
		_, err := file.WriteString(strconv.Itoa(latency) + "\n")
		if err != nil {
			panic(err)
		}
	}
}

func PutToInsertLatencys(latency int) {
	insertLock.Lock()
	defer insertLock.Unlock()
	InsertLatencys = append(InsertLatencys, latency)
}

func PutToSearchLatencys(latency int) {
	searchLock.Lock()
	defer searchLock.Unlock()
	SearchLatencys = append(SearchLatencys, latency)
}

func Save() {
	save("/home/ljq/debug/15583/gosdk-insert.txt", InsertLatencys)
	save("/home/ljq/debug/15583/gosdk-search.txt", SearchLatencys)
}

