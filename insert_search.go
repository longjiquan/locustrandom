package locustrandom

import (
	"context"
	"fmt"
	milvusclient "github.com/milvus-io/milvus-sdk-go/v2/client"
	"github.com/milvus-io/milvus-sdk-go/v2/entity"
	"math/rand"
	"strconv"
	"sync"
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
var dim = 128
var topk = 50
var address = "localhost:19530"
var nq = 1
var sampleCount = 1000
var threadCount = 20
var shardNum = 10

var vectors = prepareVectors(nq)

func prepareSchema() *entity.Schema {
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

func generateFloatVectors(numRows int) [][]float32 {
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

func prepareInsertData(iter int, num int) []entity.Column {
	pkColumn := entity.NewColumnInt64(f1, preparePks(iter, num))
	randomColumn := entity.NewColumnDouble(f2, generateFloat64Array(num))
	embeddingColumn := entity.NewColumnFloatVector(f3, dim, generateFloatVectors(num))
	return []entity.Column{pkColumn, randomColumn, embeddingColumn}
}

func insert(client milvusclient.Client, iter, num int) {
	columns := prepareInsertData(iter, num)
	_, err := client.Insert(context.Background(), collectionName, partitionName, columns...)
	if err != nil {
		panic(err)
	}
}

func prepareVectors(nq int) []entity.Vector {
	ret := make([]entity.Vector, 0, nq)
	for i := 0; i < nq; i++ {
		x := generateFloat32Array(dim)
		var vector entity.FloatVector
		vector = append(vector, x...)
		ret = append(ret, vector)
	}
	return ret
}

func setupCollection(client milvusclient.Client) {
	ctx := context.Background()
	schema := prepareSchema()
	if has, err := client.HasCollection(ctx, collectionName); err != nil || has {
		fmt.Println("collection already exists: ", collectionName)
		return
	}
	if err := client.CreateCollection(ctx, schema, int32(shardNum)); err != nil {
		panic(err)
	}
	fmt.Println("create collection: ", collectionName)

	if err := client.CreatePartition(ctx, collectionName, partitionName); err != nil {
		panic(err)
	}
	fmt.Println("create partition: ", partitionName)

	repeat := 670
	num := 1000
	for i := 0; i < repeat; i++ {
		insert(client, i, num)
		fmt.Println("insert into collection: ", i)
	}

	index, err := entity.NewIndexHNSW("IP", 64, 250)
	if err != nil {
		panic(err)
	}
	if err := client.CreateIndex(ctx, collectionName, f3, index, false); err != nil {
		panic(err)
	}
	fmt.Println("create index")
}

func search() {
	client := newClient(address)
	defer closeClient(client)

	sp, err := entity.NewIndexHNSWSearchParam(250)
	if err != nil {
		panic(err)
	}

	for i := 0 ; i < sampleCount ; i++ {
		_, err = client.Search(context.Background(), collectionName, []string{partitionName}, fmt.Sprintf("%s >= 0", f1), []string{f1, f2}, vectors, f3, entity.L2, topk, sp)
		if err != nil {
			panic(err)
		}
		if (i+1)%100 == 0 {
			fmt.Println("search done: ", i)
		}
	}
}

func ebay() {
	client := newClient(address)
	defer closeClient(client)
	setupCollection(client)

	ctx := context.Background()
	if err := client.LoadCollection(ctx, collectionName, false); err != nil {
		panic(err)
	}
	fmt.Println("load collection: ", collectionName)

	wg := sync.WaitGroup{}
	for i := 0; i < threadCount; i++ {
		wg.Add(1)
		go func() {
			defer wg.Done()
			search()
		}()
	}
	wg.Wait()
}

