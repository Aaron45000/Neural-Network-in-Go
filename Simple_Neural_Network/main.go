package main

import (
	"bufio"
	"encoding/csv"
	"fmt"
	"io"
	"math"
	"os"
	"strconv"
	"sync"

	"gonum.org/v1/gonum/mat"
	"gonum.org/v1/gonum/stat/distuv"
)

// For predictions to be made concurrently
type TrainData struct {
	inputs []float64
	target []float64
}

// Struct representing the parts of a simple neural network
type Network struct {
	inputs        int
	hiddenNodes   int
	outputs       int
	hiddenWeights *mat.Dense
	outputWeights *mat.Dense
	learningRate  float64
}

// create one instance of the struct Network
func CreateNetwork(input, hidden, output int, rate float64) Network {
	return Network{
		inputs:        input,
		hiddenNodes:   hidden,
		outputs:       output,
		hiddenWeights: mat.NewDense(hidden, input, randomArray(hidden*input, float64(input))),
		outputWeights: mat.NewDense(output, hidden, randomArray(output*hidden, float64(hidden))),
		learningRate:  rate,
	}
}

// predict is a method of Network
func (network *Network) Predict(slice []float64) mat.Matrix {

	inputs := mat.NewDense(len(slice), 1, slice)
	hiddenOutputs := apply(sigmoid, dotProduct(network.hiddenWeights, inputs))
	finalInputs := dotProduct(network.outputWeights, hiddenOutputs)
	return apply(sigmoid, finalInputs)

}

func (network *Network) Train(input, output []float64) {

	inputs := mat.NewDense(len(input), 1, input)
	hiddenInputs := dotProduct(network.hiddenWeights, inputs)
	hiddenOutputs := apply(sigmoid, hiddenInputs)
	finalInputs := dotProduct(network.outputWeights, hiddenOutputs)
	finalOutputs := apply(sigmoid, finalInputs)

	outputErrors := sub(mat.NewDense(len(output), 1, output), finalOutputs)
	hiddenErrors := dotProduct(network.outputWeights.T(), outputErrors)

	backPropagation1 := elemProduct(outputErrors, sigmoidPrime(finalOutputs))
	backPropagation1 = dotProduct(backPropagation1, hiddenOutputs.T())
	backPropagation1 = scale(backPropagation1, network.learningRate)
	network.outputWeights = sum(network.outputWeights, backPropagation1).(*mat.Dense)

	backPropagation2 := elemProduct(hiddenErrors, sigmoidPrime(hiddenOutputs))
	backPropagation2 = dotProduct(backPropagation2, inputs.T())
	backPropagation2 = scale(backPropagation2, network.learningRate)
	network.hiddenWeights = sum(network.hiddenWeights, backPropagation2).(*mat.Dense)

}

// Create a random array using the continous uniform probability distribution
func randomArray(size int, v float64) []float64 {
	dist := distuv.Uniform{
		Min: -1 / math.Sqrt(v),
		Max: 1 / math.Sqrt(v),
	}
	data := make([]float64, size)
	for i := 0; i < size; i++ {
		data[i] = dist.Rand()
	}
	return data
}

// apply sigmoid funtion  1/(1 + e^(-x))
func sigmoid(r, c int, z float64) float64 {

	return 1.0 / (1.0 + math.Exp(-z))
}

// to do the dot product between two matrix
func dotProduct(m, n mat.Matrix) mat.Matrix {
	r, _ := m.Dims()
	_, c := n.Dims()
	o := mat.NewDense(r, c, nil)
	o.Product(m, n)
	return o
}

func sub(m, n mat.Matrix) mat.Matrix {
	r, c := m.Dims()
	sub := mat.NewDense(r, c, nil)
	sub.Sub(m, n)
	return sub
}

func sum(m, n mat.Matrix) mat.Matrix {
	r, c := m.Dims()
	sum := mat.NewDense(r, c, nil)
	sum.Add(m, n)
	return sum
}

func elemProduct(m, n mat.Matrix) mat.Matrix {
	r, c := m.Dims()
	product := mat.NewDense(r, c, nil)
	product.MulElem(m, n)
	return product
}

func sigmoidPrime(m mat.Matrix) mat.Matrix {

	r, c := m.Dims()
	arrayone := make([]float64, r*c)

	for i := 0; i < r*c; i++ {

		arrayone[i] = 1
	}
	ones := mat.NewDense(r, c, arrayone)

	OneminusM := sub(ones, m)

	return elemProduct(m, OneminusM)

}

func scale(n mat.Matrix, s float64) mat.Matrix {
	r, c := n.Dims()
	scale := mat.NewDense(r, c, nil)
	scale.Scale(s, n)
	return scale
}

// To apply sigmoid function to the matrix
func apply(fn func(r, c int, v float64) float64, m mat.Matrix) mat.Matrix {
	r, c := m.Dims()
	o := mat.NewDense(r, c, nil)
	o.Apply(fn, m)
	return o
}

func testWorker(network *Network, wg *sync.WaitGroup, lineChannel chan []string, resultsChannel chan int) {

	defer wg.Done()

	for line := range lineChannel {

		header, err := strconv.Atoi(line[0])
		if err != nil {

			fmt.Printf("Hubo un error an intentar convertir el valor ASCII del header del csv a int en testfile")
			return
		}

		inputs := make([]float64, 784)

		for i := 0; i < 784; i++ {

			pixelStr := line[i+1]
			pixelVal, _ := strconv.ParseFloat(pixelStr, 64)
			inputs[i] = (pixelVal / 255 * 0.99) + 0.01
		}

		outputs := network.Predict(inputs)

		Maxval := -1.0
		prediction := 0

		for j := 0; j < 10; j++ {

			val := outputs.At(j, 0)

			if val > Maxval {

				Maxval = val
				prediction = j

			}
		}

		if prediction == header {

			resultsChannel <- 1

		} else {

			resultsChannel <- 0
		}

	}

}

func fetchResults(results chan int, score *int, total *int, fetchFinale *sync.WaitGroup) {

	defer fetchFinale.Done()
	for res := range results {

		*score += res
		*total++
	}
}

func main() {

	network := CreateNetwork(784, 200, 10, 0.1)

	trainFile, err := os.Open("csv/mnist_train.csv")
	lineChannel := make(chan []string, 50)
	resultChannel := make(chan int, 50)
	var wg sync.WaitGroup
	var fetchFinale sync.WaitGroup

	if err != nil {

		fmt.Printf("There was an error opening mnist_train.csv")
		return
	}
	reader := csv.NewReader(bufio.NewReader(trainFile))

	for {

		line, err := reader.Read()

		if err == io.EOF {

			break

		}

		header, err1 := strconv.Atoi(line[0])
		if err1 != nil {

			fmt.Printf("Hubo un error an intentar convertir el valor ASCII del header del csv a int en trainFile")
		}
		target := make([]float64, 10)
		for i := range target {

			target[i] = 0.01

		}

		target[header] = 0.99

		inputs := make([]float64, 784)

		for i := 0; i < 784; i++ {

			pixelStr := line[i+1]
			pixelVal, _ := strconv.ParseFloat(pixelStr, 64)
			inputs[i] = (pixelVal / 255 * 0.99) + 0.01
		}

		network.Train(inputs, target)

	}

	trainFile.Close()

	score := 0
	total := 0

	testFile, err := os.Open("csv/mnist_test.csv")

	if err != nil {

		fmt.Printf("There was an error opening mnist_test.csv")
		return
	}

	reader = csv.NewReader(bufio.NewReader(testFile))

	fetchFinale.Add(1)
	go fetchResults(resultChannel, &score, &total, &fetchFinale)

	for j := 0; j < 4; j++ {

		wg.Add(1)
		go testWorker(&network, &wg, lineChannel, resultChannel)

	}

	for {

		line, err := reader.Read()

		if err == io.EOF {

			break

		}

		lineChannel <- line

	}

	close(lineChannel)
	wg.Wait()
	close(resultChannel)
	testFile.Close()
	fetchFinale.Wait()

	if total > 0 {

		fmt.Printf("The neural network has score %d/%d \n", score, total)
		fmt.Printf("The percentage of accuracy is %.2f%% \n", (float64(score)/float64(total))*100)

	} else {

		fmt.Printf("There was no image to process")

	}

}
