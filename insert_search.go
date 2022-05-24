package locustrandom

import (
	"context"
	"fmt"
	milvusclient "github.com/milvus-io/milvus-sdk-go/v2/client"
	"github.com/milvus-io/milvus-sdk-go/v2/entity"
	"math/rand"
	"strconv"
	"sync"
	"time"
)

func newClient(address string) milvusclient.Client {
	client, err := milvusclient.NewGrpcClient(context.Background(), address)
	if err != nil {
		panic(err)
	}
	return client
}

func closeClient(client milvusclient.Client) {
	if err := client.Close(); err != nil {
		panic(err)
	}
}

var collectionName = "test_go_random_locust"
var partitionName = "default"
var f1 = "pk"
var f2 = "random"
var f3 = "embeddings"

func prepareSchema(dim int) *entity.Schema {
	return &entity.Schema{
		CollectionName: collectionName,
		Description:    "",
		AutoID:         false,
		Fields: []*entity.Field{
			{
				ID:         100,
				Name:       f1,
				PrimaryKey: true,
				DataType:   entity.FieldTypeInt64,
			},
			{
				ID:       101,
				Name:     f2,
				DataType: entity.FieldTypeDouble,
			},
			{
				ID:       102,
				Name:     f3,
				DataType: entity.FieldTypeFloatVector,
				TypeParams: map[string]string{
					"dim": strconv.Itoa(dim),
				},
			},
		},
	}
}

func prepare(client milvusclient.Client, dim int) {
	ctx := context.Background()
	has, err := client.HasCollection(ctx, collectionName)
	if err != nil {
		panic(err)
	}
	if has {
		err := client.DropCollection(ctx, collectionName)
		if err != nil {
			panic(err)
		}
	}
	schema := prepareSchema(dim)
	if err := client.CreateCollection(ctx, schema, 2); err != nil {
		panic(err)
	}
	if err := client.CreatePartition(ctx, collectionName, partitionName); err != nil {
		panic(err)
	}
	if err := client.LoadCollection(ctx, collectionName, false) ; err != nil {
		panic(err)
	}
}

func preparePks(iter int, num int) []int64 {
	ret := make([]int64, num)
	start := iter * num
	end := iter*num + num
	for i := start; i < end; i++ {
		ret[i-start] = int64(i)
	}
	return ret
}

func generateFloat64Array(numRows int) []float64 {
	ret := make([]float64, 0, numRows)
	for i := 0; i < numRows; i++ {
		ret = append(ret, rand.Float64())
	}
	return ret
}

func generateFloat32Array(numRows int) []float32 {
	ret := make([]float32, 0, numRows)
	for i := 0; i < numRows; i++ {
		ret = append(ret, rand.Float32())
	}
	return ret
}

func generateFloatVectors(numRows, dim int) [][]float32 {
	ret := make([][]float32, 0, numRows)
	for i := 0; i < numRows; i++ {
		x := make([]float32, 0, dim)
		for j := 0; j < dim; j++ {
			x = append(x, rand.Float32())
		}
		ret = append(ret, x)
	}
	return ret
}

func prepareInsertData(iter int, num int, dim int) []entity.Column {
	pkColumn := entity.NewColumnInt64(f1, preparePks(iter, num))
	randomColumn := entity.NewColumnDouble(f2, generateFloat64Array(num))
	embeddingColumn := entity.NewColumnFloatVector(f3, dim, generateFloatVectors(num, dim))
	return []entity.Column{pkColumn, randomColumn, embeddingColumn}
}

func insert(client milvusclient.Client, iter, num, dim int) {
	columns := prepareInsertData(iter, num, dim)
	start := time.Now()
	_, err := client.Insert(context.Background(), collectionName, partitionName, columns...)
	PutToInsertLatencys(int(time.Since(start).Milliseconds()))
	if err != nil {
		panic(err)
	}
}

func prepareVectors(nq int, dim int) []entity.Vector {
	ret := make([]entity.Vector, 0, nq)
	for i := 0; i < nq; i++ {
		x := generateFloat32Array(dim)
		var vector entity.FloatVector
		vector = append(vector, x...)
		ret = append(ret, vector)
	}
	return ret
}

func search(client milvusclient.Client, vectors []entity.Vector) {
	topk := 10
	nprobe := 20
	sp, err := entity.NewIndexFlatSearchParam(nprobe)
	if err != nil {
		panic(err)
	}
	start := time.Now()
	_, err = client.Search(context.Background(), collectionName, []string{partitionName}, fmt.Sprintf("%s >= 0", f1), []string{f1, f2}, vectors, f3, entity.L2, topk, sp)
	PutToSearchLatencys(int(time.Since(start).Milliseconds()))
	if err != nil {
		panic(err)
	}
}

func main() {
	address := "localhost:19530"

	defer Save()

	insertClient := newClient(address)
	defer closeClient(insertClient)

	searchClient := newClient(address)
	defer closeClient(searchClient)

	dim := 128
	prepare(insertClient, dim)

	iter := 0
	num := 500
	insert(insertClient, iter, num, dim)
	iter++

	vectors := prepareVectors(10, dim)

	var wg sync.WaitGroup

	per := 12
	n := 400
	for i := 0; i < n; i++ {
		for j := 0; j < per; j++ {
			wg.Add(1)
			go func() {
				defer wg.Done()
				if rand.Int()%2 == 0 {
					insert(insertClient, iter, num, dim)
					iter++
				} else {
					search(searchClient, vectors)
				}
			}()
		}
		wg.Wait()
	}
}
