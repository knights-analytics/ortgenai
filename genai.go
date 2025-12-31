package ortgenai

/*
#cgo CFLAGS: -O2 -g
#include "ort_genai_wrapper.h"
*/
import "C"

import (
	"context"
	"encoding/base64"
	"encoding/json"
	"errors"
	"fmt"
	"runtime"
	"strings"
	"sync"
	"time"
	"unsafe"
)

var ErrNotInitialized = fmt.Errorf("InitializeEnvironment() has either " +
	"not yet been called, or did not return successfully")

var onnxGenaiSharedLibraryPath string

var genAiEnv bool

func IsInitialized() bool {
	return genAiEnv
}

func InitializeEnvironment() error {
	if IsInitialized() {
		return fmt.Errorf("GenAI environment already initialized")
	}
	if err := InitializeGenAiLibrary(); err != nil {
		return fmt.Errorf("error initializing GenAI library: %w", err)
	}
	genAiEnv = true
	return nil
}

// DestroyEnvironment Call this function to clean up the internal onnxruntime environment when it
// is no longer required.
func DestroyEnvironment() error {
	if !IsInitialized() {
		return ErrNotInitialized
	}
	if err := platformCleanup(); err != nil {
		return fmt.Errorf("error during platform cleanup: %w", err)
	}
	genAiEnv = false
	return nil
}

func SetSharedLibraryPath(path string) {
	onnxGenaiSharedLibraryPath = path
}

type GenerationOptions struct {
	MaxLength int
	BatchSize int
}

type Message struct {
	Role    string `json:"role"`
	Content string `json:"content"`
}

type generator struct {
	generatorParamsPtr *C.OgaGeneratorParams
	generatorPtr       *C.OgaGenerator
}

func (g *generator) destroy() {
	C.DestroyOgaGenerator(g.generatorPtr)
	g.generatorPtr = nil
	C.DestroyOgaGeneratorParams(g.generatorParamsPtr)
	g.generatorParamsPtr = nil
}

func (g *generator) setInputs(namedTensors *NamedTensors) error {
	if namedTensors == nil || namedTensors.tensorsPtr == nil {
		return errors.New("named tensors is nil")
	}
	res := C.GeneratorSetInputs(g.generatorPtr, namedTensors.tensorsPtr)
	if err := OgaResultToError(res); err != nil {
		return fmt.Errorf("GeneratorSetInputs failed: %w", err)
	}
	return nil
}

type tokenizer struct {
	tokenizerPtr *C.OgaTokenizer
}

func newTokenizerFromModel(model model) (tokenizer, error) {
	var cTokenizer *C.OgaTokenizer
	res := C.CreateOgaTokenizer(model.modelPtr, &cTokenizer)
	if err := OgaResultToError(res); err != nil {
		return tokenizer{}, fmt.Errorf("CreateOgaTokenizer failed: %w", err)
	}
	if cTokenizer == nil {
		return tokenizer{}, errors.New("CreateOgaTokenizer returned nil without error")
	}
	return tokenizer{tokenizerPtr: cTokenizer}, nil
}

func (t *tokenizer) encode(prompt string, sequences *sequences) error {
	cStr := C.CString(prompt)
	defer C.free(unsafe.Pointer(cStr))
	result := C.TokenizerEncode(t.tokenizerPtr, cStr, sequences.sequencesPtr)
	if err := OgaResultToError(result); err != nil {
		return fmt.Errorf("TokenizerEncode failed: %w", err)
	}
	return nil
}

func (t *tokenizer) destroy() {
	C.DestroyOgaTokenizer(t.tokenizerPtr)
	t.tokenizerPtr = nil
}

type tokenizerStream struct {
	streamPtr *C.OgaTokenizerStream
}

func (ts *tokenizerStream) Decode(token C.int32_t) (string, error) {
	var cOutput *C.char
	result := C.TokenizerStreamDecode(ts.streamPtr, token, &cOutput)
	if err := OgaResultToError(result); err != nil {
		return "", fmt.Errorf("TokenizerStreamDecode failed: %w", err)
	}
	if cOutput == nil {
		return "", nil
	}
	decoded := C.GoString(cOutput) // don't free this one -- owned by the tokenizer stream
	return decoded, nil
}

func (ts *tokenizerStream) destroy() {
	C.DestroyOgaTokenizerStream(ts.streamPtr)
	ts.streamPtr = nil
}

type sequences struct {
	sequencesPtr *C.OgaSequences
}

type model struct {
	modelPtr *C.OgaModel
}

func (m *model) destroy() {
	if m.modelPtr != nil {
		C.DestroyOgaModel(m.modelPtr)
	}
	m.modelPtr = nil
}

type Session struct {
	model      *model
	tokenizer  *tokenizer
	statistics Statistics
	mutex      sync.Mutex // the C API is not thread-safe
}

type SequenceDelta struct {
	Sequence int
	Tokens   string
}

// Images represents a collection of loaded images for multimodal processing
type Images struct {
	imagesPtr *C.OgaImages
}

func (i *Images) destroy() {
	if i.imagesPtr != nil {
		C.DestroyOgaImages(i.imagesPtr)
	}
	i.imagesPtr = nil
}

// Destroy releases the images resources
func (i *Images) Destroy() {
	i.destroy()
}

// MultiModalProcessor processes images and text together for multimodal models
type MultiModalProcessor struct {
	processorPtr *C.OgaMultiModalProcessor
}

func (p *MultiModalProcessor) destroy() {
	if p.processorPtr != nil {
		C.DestroyOgaMultiModalProcessor(p.processorPtr)
	}
	p.processorPtr = nil
}

// Destroy releases the processor resources
func (p *MultiModalProcessor) Destroy() {
	p.destroy()
}

// NamedTensors represents a collection of named tensor inputs
type NamedTensors struct {
	tensorsPtr *C.OgaNamedTensors
}

func (nt *NamedTensors) destroy() {
	if nt.tensorsPtr != nil {
		C.DestroyOgaNamedTensors(nt.tensorsPtr)
	}
	nt.tensorsPtr = nil
}

// Destroy releases the named tensors resources
func (nt *NamedTensors) Destroy() {
	nt.destroy()
}

// Statistics captures generation performance metrics.
type Statistics struct {
	AvgPrefillSeconds float64
	TokensPerSecond   float64
	// cumulative
	CumulativePrefillSum           float64
	CumulativePrefillCount         int
	CumulativeTokens               int
	CumulativeTokenDurationSeconds float64
}

// GetStatistics returns the last computed statistics for the session.
func (s *Session) GetStatistics() Statistics {
	return s.statistics
}

// GetModel returns the underlying model for creating multimodal processors
func (s *Session) GetModel() *model {
	return s.model
}

func (g *generator) IsDone() bool {
	return bool(C.IsDone(g.generatorPtr))
}

func (t *tokenizer) ApplyChatTemplate(inputMessages []byte, addGenerationPrompt bool) (string, error) {
	if t.tokenizerPtr == nil {
		return "", errors.New("tokenizer is not initialized")
	}
	cInput := C.CString(string(inputMessages))
	defer C.free(unsafe.Pointer(cInput))
	var cOutput *C.char
	res := C.ApplyOgaTokenizerChatTemplate(t.tokenizerPtr, nil, cInput, nil, C.bool(addGenerationPrompt), &cOutput)
	if err := OgaResultToError(res); err != nil {
		return "", fmt.Errorf("ApplyOgaChatTemplate failed: %w", err)
	}
	if cOutput == nil {
		return "", errors.New("ApplyOgaChatTemplate returned nil output without error")
	}
	output := C.GoString(cOutput)
	C.DestroyOgaString(cOutput)
	return output, nil
}

func (t *tokenizer) tokenizeMessages(messages [][]Message) (*sequences, []*tokenizerStream, error) {
	if t.tokenizerPtr == nil {
		return nil, nil, errors.New("tokenizer is not initialized")
	}
	if len(messages) == 0 {
		return nil, nil, errors.New("no messages provided")
	}

	var cSequences *C.OgaSequences
	res := C.CreateOgaSequences(&cSequences)
	if err := OgaResultToError(res); err != nil {
		return nil, nil, fmt.Errorf("CreateOgaSequences failed: %w", err)
	}
	if cSequences == nil {
		return nil, nil, errors.New("CreateOgaSequences returned nil without error")
	}
	sequences := &sequences{sequencesPtr: cSequences}
	tokenizerStreams := make([]*tokenizerStream, 0, len(messages))

	for _, message := range messages {
		messageJSON, err := json.Marshal(message)
		if err != nil {
			return nil, nil, fmt.Errorf("failed to marshal input message: %w", err)
		}
		prompt, templateErr := t.ApplyChatTemplate(messageJSON, true)
		if templateErr != nil {
			return nil, nil, fmt.Errorf("failed to apply chat template: %w", templateErr)
		}
		if err = t.encode(prompt, sequences); err != nil {
			return nil, nil, fmt.Errorf("encode failed: %w", err)
		}
		var cStream *C.OgaTokenizerStream
		res = C.CreateOgaTokenizerStream(t.tokenizerPtr, &cStream)
		if err = OgaResultToError(res); err != nil {
			return nil, nil, fmt.Errorf("CreateOgaTokenizerStream failed: %w", err)
		}
		if cStream == nil {
			return nil, nil, errors.New("CreateOgaTokenizerStream returned nil without error")
		}
		tokenizerStreams = append(tokenizerStreams, &tokenizerStream{streamPtr: cStream})
	}
	return sequences, tokenizerStreams, nil
}

func (s *Session) createGenerator(sequences *sequences, generationOptions *GenerationOptions) (*generator, error) {
	var cGeneratorParams *C.OgaGeneratorParams
	res := C.CreateOgaGeneratorParams(s.model.modelPtr, &cGeneratorParams)
	if err := OgaResultToError(res); err != nil {
		return nil, fmt.Errorf("CreateOgaGeneratorParams failed: %w", err)
	}
	if cGeneratorParams == nil {
		return nil, errors.New("CreateOgaGeneratorParams returned nil generator params without error")
	}
	maxLengthName := C.CString("max_length")
	defer C.free(unsafe.Pointer(maxLengthName))
	res = C.GeneratorParamsSetSearchNumber(cGeneratorParams, maxLengthName, C.double(generationOptions.MaxLength))
	if err := OgaResultToError(res); err != nil {
		return nil, fmt.Errorf("GeneratorParamsSetSearchNumber failed: %w", err)
	}

	batchSizeName := C.CString("batch_size")
	defer C.free(unsafe.Pointer(batchSizeName))
	res = C.GeneratorParamsSetSearchNumber(cGeneratorParams, batchSizeName, C.double(generationOptions.BatchSize))
	if err := OgaResultToError(res); err != nil {
		return nil, fmt.Errorf("GeneratorParamsSetSearchNumber(batch_size) failed: %w", err)
	}

	// create a generator with those params
	var cGenerator *C.OgaGenerator
	res = C.CreateOgaGenerator(s.model.modelPtr, cGeneratorParams, &cGenerator)
	if err := OgaResultToError(res); err != nil {
		return nil, fmt.Errorf("CreateOgaGenerator failed: %w", err)
	}
	if cGenerator == nil {
		return nil, errors.New("CreateOgaGenerator returned nil generator without error")
	}

	// add sequences to generator
	res = C.GeneratorAppendTokenSequences(cGenerator, sequences.sequencesPtr)
	if err := OgaResultToError(res); err != nil {
		return nil, fmt.Errorf("GeneratorAppendTokenSequences failed: %w", err)
	}

	// Sequences are no longer needed after appending; destroy to avoid leaks.
	C.DestroyOgaSequences(sequences.sequencesPtr)
	sequences.sequencesPtr = nil

	return &generator{
		generatorParamsPtr: cGeneratorParams,
		generatorPtr:       cGenerator,
	}, nil
}

func sendGenerationError(errChan chan<- error, err error) {
	select {
	case errChan <- err:
	default:
	}
}

func (s *Session) Generate(ctx context.Context, messages [][]Message, generationOptions *GenerationOptions) (<-chan SequenceDelta, <-chan error, error) {
	s.mutex.Lock()
	sequences, tokenizerStreams, tokenizeErr := s.tokenizer.tokenizeMessages(messages)
	if tokenizeErr != nil {
		return nil, nil, fmt.Errorf("TokenizeMessages failed: %w", tokenizeErr)
	}

	if generationOptions == nil {
		generationOptions = &GenerationOptions{
			MaxLength: 2024,
			BatchSize: len(messages),
		}
	}
	if generationOptions.BatchSize <= 0 {
		generationOptions.BatchSize = len(messages)
	}

	generator, err := s.createGenerator(sequences, generationOptions)
	if err != nil {
		return nil, nil, err
	}

	outputChan := make(chan SequenceDelta, 10)
	errChan := make(chan error, 1)
	go func() {
		defer close(outputChan)
		defer close(errChan)
		defer generator.destroy()
		defer func() {
			for _, ts := range tokenizerStreams {
				ts.destroy()
			}
		}()

		var result *C.OgaResult

		// Use goroutine-local variables for per-run statistics to avoid race conditions
		// when multiple Generate calls run concurrently on the same session.
		runStart := time.Now()
		runFirstTokenTimes := make([]time.Time, len(messages))
		runTokenCount := 0

		// finalize tokens/sec at the end of the run
		defer func() {
			var earliest time.Time
			for _, ft := range runFirstTokenTimes {
				if !ft.IsZero() && (earliest.IsZero() || ft.Before(earliest)) {
					earliest = ft
				}
			}
			if !earliest.IsZero() && runTokenCount > 0 {
				dur := time.Since(earliest).Seconds()
				if dur > 0 {
					s.statistics.CumulativeTokenDurationSeconds += dur
					s.statistics.TokensPerSecond = float64(s.statistics.CumulativeTokens) / s.statistics.CumulativeTokenDurationSeconds
				}
			}
		}()
		defer s.mutex.Unlock()

		firstEmitted := make([]bool, len(messages))
		lastChar := make([]rune, len(messages))

		for {
			select {
			case <-ctx.Done():
				return
			default:
			}

			if generator.IsDone() {
				return
			}

			result = C.GeneratorGenerateNextToken(generator.generatorPtr)
			if err = OgaResultToError(result); err != nil {
				sendGenerationError(errChan, err)
				return
			}
			// For each sequence, decode only the last token just appended.
			for i := 0; i < len(messages); i++ {
				numTokens := C.GeneratorGetSequenceCount(generator.generatorPtr, C.size_t(i))
				if numTokens == 0 {
					continue
				}
				seqData := C.GeneratorGetSequenceData(generator.generatorPtr, C.size_t(i))
				arr := (*[1 << 30]C.int32_t)(unsafe.Pointer(seqData))
				lastToken := arr[numTokens-1]
				decoded, decodeErr := tokenizerStreams[i].Decode(lastToken)
				if decodeErr != nil {
					sendGenerationError(errChan, decodeErr)
					return
				}
				if decoded == "" {
					continue
				}

				// some normalization: skip leading spaces for first token, avoid repeated periods at the end.
				if !firstEmitted[i] {
					trim := strings.TrimLeft(decoded, " ")
					if trim == "" {
						continue
					}
					decoded = trim
					firstEmitted[i] = true
				}
				if decoded == "." && lastChar[i] == '.' {
					continue
				}
				r := []rune(decoded)
				lastChar[i] = r[len(r)-1]

				// stats
				if runFirstTokenTimes[i].IsZero() {
					runFirstTokenTimes[i] = time.Now()
					prefill := runFirstTokenTimes[i].Sub(runStart).Seconds()
					s.statistics.CumulativePrefillSum += prefill
					s.statistics.CumulativePrefillCount++
					s.statistics.AvgPrefillSeconds = s.statistics.CumulativePrefillSum / float64(s.statistics.CumulativePrefillCount)
				}
				s.statistics.CumulativeTokens++
				runTokenCount++
				select {
				case outputChan <- SequenceDelta{Sequence: i, Tokens: decoded}:
				case <-ctx.Done():
					return
				}
			}
		}
	}()
	return outputChan, errChan, nil
}

// GenerateWithTensors generates text using pre-processed named tensors (for multimodal inputs)
func (s *Session) GenerateWithTensors(ctx context.Context, namedTensors *NamedTensors, generationOptions *GenerationOptions) (<-chan SequenceDelta, <-chan error, error) {
	s.mutex.Lock()

	if namedTensors == nil || namedTensors.tensorsPtr == nil {
		return nil, nil, errors.New("named tensors is nil")
	}

	if generationOptions == nil {
		generationOptions = &GenerationOptions{
			MaxLength: 2024,
			BatchSize: 1,
		}
	}
	if generationOptions.BatchSize <= 0 {
		generationOptions.BatchSize = 1
	}

	// Create generator params
	var cGeneratorParams *C.OgaGeneratorParams
	res := C.CreateOgaGeneratorParams(s.model.modelPtr, &cGeneratorParams)
	if err := OgaResultToError(res); err != nil {
		return nil, nil, fmt.Errorf("CreateOgaGeneratorParams failed: %w", err)
	}
	if cGeneratorParams == nil {
		return nil, nil, errors.New("CreateOgaGeneratorParams returned nil")
	}

	maxLengthName := C.CString("max_length")
	defer C.free(unsafe.Pointer(maxLengthName))
	res = C.GeneratorParamsSetSearchNumber(cGeneratorParams, maxLengthName, C.double(generationOptions.MaxLength))
	if err := OgaResultToError(res); err != nil {
		C.DestroyOgaGeneratorParams(cGeneratorParams)
		return nil, nil, fmt.Errorf("GeneratorParamsSetSearchNumber(max_length) failed: %w", err)
	}

	batchSizeName := C.CString("batch_size")
	defer C.free(unsafe.Pointer(batchSizeName))
	res = C.GeneratorParamsSetSearchNumber(cGeneratorParams, batchSizeName, C.double(generationOptions.BatchSize))
	if err := OgaResultToError(res); err != nil {
		C.DestroyOgaGeneratorParams(cGeneratorParams)
		return nil, nil, fmt.Errorf("GeneratorParamsSetSearchNumber(batch_size) failed: %w", err)
	}

	// Create generator
	var cGenerator *C.OgaGenerator
	res = C.CreateOgaGenerator(s.model.modelPtr, cGeneratorParams, &cGenerator)
	if err := OgaResultToError(res); err != nil {
		C.DestroyOgaGeneratorParams(cGeneratorParams)
		return nil, nil, fmt.Errorf("CreateOgaGenerator failed: %w", err)
	}
	if cGenerator == nil {
		C.DestroyOgaGeneratorParams(cGeneratorParams)
		return nil, nil, errors.New("CreateOgaGenerator returned nil")
	}

	generator := &generator{
		generatorParamsPtr: cGeneratorParams,
		generatorPtr:       cGenerator,
	}

	// Set the named tensors as inputs
	if err := generator.setInputs(namedTensors); err != nil {
		generator.destroy()
		return nil, nil, fmt.Errorf("failed to set inputs: %w", err)
	}

	// Create tokenizer stream for decoding (we still need this for output)
	var cStream *C.OgaTokenizerStream
	res = C.CreateOgaTokenizerStream(s.tokenizer.tokenizerPtr, &cStream)
	if err := OgaResultToError(res); err != nil {
		generator.destroy()
		return nil, nil, fmt.Errorf("CreateOgaTokenizerStream failed: %w", err)
	}
	if cStream == nil {
		generator.destroy()
		return nil, nil, errors.New("CreateOgaTokenizerStream returned nil")
	}
	tokStream := &tokenizerStream{streamPtr: cStream}

	outputChan := make(chan SequenceDelta, 10)
	errChan := make(chan error, 1)
	go func() {
		defer close(outputChan)
		defer close(errChan)
		defer generator.destroy()
		defer tokStream.destroy()

		// Use goroutine-local variables for per-run statistics to avoid race conditions
		// when multiple Generate calls run concurrently on the same session.
		runStart := time.Now()
		// Use a map for first token times since the number of sequences may vary
		// (especially for multimodal models where sequence count can differ from batch size)
		runFirstTokenTimes := make(map[int]time.Time)
		runTokenCount := 0

		defer func() {
			var earliest time.Time
			for _, ft := range s.statistics.runFirstTokenTimes {
				if !ft.IsZero() && (earliest.IsZero() || ft.Before(earliest)) {
					earliest = ft
				}
			}
			if !earliest.IsZero() && s.statistics.runTokenCount > 0 {
				dur := time.Since(earliest).Seconds()
				if dur > 0 {
					s.statistics.cumulativeTokenDurationSeconds += dur
					s.statistics.TokensPerSecond = float64(s.statistics.cumulativeTokens) / s.statistics.cumulativeTokenDurationSeconds
				}
			}
			s.statistics.runFirstTokenTimes = nil
			s.statistics.runTokenCount = 0
			s.statistics.runStart = time.Time{}
		}()
		defer s.mutex.Unlock()

		for !generator.IsDone() {
			select {
			case <-ctx.Done():
				return
			default:
			}

			var result *C.OgaResult
			result = C.GeneratorGenerateNextToken(generator.generatorPtr)
			if err := OgaResultToError(result); err != nil {
				select {
				case errChan <- err:
				default:
				}
				return
			}

			// Iterate over each sequence in the batch
			// Note: generationOptions.BatchSize tells us how many sequences we have
			for i := range generationOptions.BatchSize {
				seqLen := C.GeneratorGetSequenceCount(generator.generatorPtr, C.size_t(i))
				if seqLen == 0 {
					continue
				}
				cData := C.GeneratorGetSequenceData(generator.generatorPtr, C.size_t(i))
				if cData == nil {
					continue
				}
				lastToken := C.int32_t(*(*C.int32_t)(unsafe.Pointer(uintptr(unsafe.Pointer(cData)) + uintptr((seqLen-1)*4))))
				decoded, decodeErr := tokStream.Decode(lastToken)
				if decodeErr != nil {
					select {
					case errChan <- decodeErr:
					default:
					}
					return
				}
				if decoded == "" {
					continue
				}

				// stats
				if _, seen := runFirstTokenTimes[i]; !seen {
					runFirstTokenTimes[i] = time.Now()
					prefill := runFirstTokenTimes[i].Sub(runStart).Seconds()
					// Update cumulative statistics with mutex protection
					s.mutex.Lock()
					s.statistics.cumulativePrefillSum += prefill
					s.statistics.cumulativePrefillCount++
					s.statistics.AvgPrefillSeconds = s.statistics.cumulativePrefillSum / float64(s.statistics.cumulativePrefillCount)
				}
				s.statistics.cumulativeTokens++
				s.statistics.runTokenCount++

				select {
				case outputChan <- SequenceDelta{Sequence: i, Tokens: decoded}:
				case <-ctx.Done():
					return
				}
			}
		}
	}()
	return outputChan, errChan, nil
}

func (s *Session) Destroy() {
	s.model.destroy()
	s.model = nil
	s.tokenizer.destroy()
	s.tokenizer = nil
}

func OgaResultToError(result *C.OgaResult) error {
	if result == nil {
		return nil
	}
	cString := C.GetOgaResultErrorString(result)
	msg := C.GoString(cString)
	C.DestroyOgaResult(result)
	return errors.New(msg)
}

// parseDataURI parses a data URI and returns the decoded data.
// Supports format: data:image/png;base64,<base64-encoded-data>
func parseDataURI(dataURI string) ([]byte, error) {
	// Check if it starts with "data:"
	if !strings.HasPrefix(dataURI, "data:") {
		return nil, fmt.Errorf("invalid data URI: must start with 'data:'")
	}

	// Find the comma that separates metadata from data
	commaIdx := strings.Index(dataURI, ",")
	if commaIdx == -1 {
		return nil, fmt.Errorf("invalid data URI: missing comma separator")
	}

	// Extract metadata and check for base64
	metadata := dataURI[5:commaIdx] // skip "data:"
	if !strings.Contains(metadata, "base64") {
		return nil, fmt.Errorf("unsupported data URI encoding: only base64 is supported")
	}

	// Decode base64 data
	encodedData := dataURI[commaIdx+1:]
	decodedData, err := base64.StdEncoding.DecodeString(encodedData)
	if err != nil {
		return nil, fmt.Errorf("failed to decode base64 data: %w", err)
	}

	return decodedData, nil
}

// LoadImage loads a single image from a file path or data URI
func LoadImage(imagePath string) (*Images, error) {
	if !IsInitialized() {
		return nil, ErrNotInitialized
	}

	// Check if it's a data URI
	if strings.HasPrefix(imagePath, "data:") {
		imageData, err := parseDataURI(imagePath)
		if err != nil {
			return nil, fmt.Errorf("failed to parse data URI: %w", err)
		}
		return LoadImageFromBuffer(imageData)
	}

	// Load from file path
	cPath := C.CString(imagePath)
	defer C.free(unsafe.Pointer(cPath))

	var cImages *C.OgaImages
	res := C.LoadOgaImage(cPath, &cImages)
	if err := OgaResultToError(res); err != nil {
		return nil, fmt.Errorf("LoadImage failed: %w", err)
	}
	if cImages == nil {
		return nil, errors.New("LoadImage returned nil without error")
	}

	return &Images{imagesPtr: cImages}, nil
}

// LoadImageFromBuffer loads a single image from a byte buffer
func LoadImageFromBuffer(imageData []byte) (*Images, error) {
	if !IsInitialized() {
		return nil, ErrNotInitialized
	}

	if len(imageData) == 0 {
		return nil, errors.New("image data is empty")
	}

	// Pin the Go memory before passing to C
	var pinner runtime.Pinner
	pinner.Pin(&imageData[0])
	defer pinner.Unpin()

	// Create C array of pointers and sizes
	dataPtr := unsafe.Pointer(&imageData[0])
	dataSize := C.size_t(len(imageData))

	var cImages *C.OgaImages
	res := C.LoadOgaImagesFromBuffers(
		&dataPtr,
		&dataSize,
		1, // count = 1 for single image
		&cImages,
	)
	if err := OgaResultToError(res); err != nil {
		return nil, fmt.Errorf("LoadImageFromBuffer failed: %w", err)
	}
	if cImages == nil {
		return nil, errors.New("LoadImageFromBuffer returned nil without error")
	}

	return &Images{imagesPtr: cImages}, nil
}

// LoadImages loads multiple images from file paths or data URIs
func LoadImages(imagePaths []string) (*Images, error) {
	if !IsInitialized() {
		return nil, ErrNotInitialized
	}

	if len(imagePaths) == 0 {
		return nil, errors.New("no image paths provided")
	}

	// Check if any paths are data URIs - if so, we need to use buffer loading
	hasDataURI := false
	for _, path := range imagePaths {
		if strings.HasPrefix(path, "data:") {
			hasDataURI = true
			break
		}
	}

	if hasDataURI {
		// Decode all images to buffers and use buffer loading
		buffers := make([][]byte, len(imagePaths))
		for i, path := range imagePaths {
			if strings.HasPrefix(path, "data:") {
				data, err := parseDataURI(path)
				if err != nil {
					return nil, fmt.Errorf("failed to parse data URI at index %d: %w", i, err)
				}
				buffers[i] = data
			} else {
				// For file paths, we'd need to read the file
				// For now, return an error if mixing data URIs with file paths
				return nil, errors.New("cannot mix data URIs with file paths in LoadImages")
			}
		}
		return LoadImagesFromBuffers(buffers)
	}

	// All are file paths - use the C API directly
	// Create OgaStringArray
	var cStringArray *C.OgaStringArray
	res := C.CreateOgaStringArray(&cStringArray)
	if err := OgaResultToError(res); err != nil {
		return nil, fmt.Errorf("CreateOgaStringArray failed: %w", err)
	}
	defer C.DestroyOgaStringArray(cStringArray)

	// Add each path to the string array
	for _, path := range imagePaths {
		cPath := C.CString(path)
		res = C.AddStringToOgaStringArray(cStringArray, cPath)
		C.free(unsafe.Pointer(cPath))
		if err := OgaResultToError(res); err != nil {
			return nil, fmt.Errorf("AddStringToOgaStringArray failed: %w", err)
		}
	}

	// Load images
	var cImages *C.OgaImages
	res = C.LoadOgaImages(cStringArray, &cImages)
	if err := OgaResultToError(res); err != nil {
		return nil, fmt.Errorf("LoadImages failed: %w", err)
	}
	if cImages == nil {
		return nil, errors.New("LoadImages returned nil without error")
	}

	return &Images{imagesPtr: cImages}, nil
}

// LoadImagesFromBuffers loads multiple images from byte buffers
func LoadImagesFromBuffers(imageBuffers [][]byte) (*Images, error) {
	if !IsInitialized() {
		return nil, ErrNotInitialized
	}

	if len(imageBuffers) == 0 {
		return nil, errors.New("no image buffers provided")
	}

	// Create arrays for pointers and sizes
	dataPtrs := make([]unsafe.Pointer, len(imageBuffers))
	dataSizes := make([]C.size_t, len(imageBuffers))

	// Pin all buffer memory before passing to C
	var pinner runtime.Pinner
	defer pinner.Unpin()

	for i, buf := range imageBuffers {
		if len(buf) == 0 {
			return nil, fmt.Errorf("image buffer at index %d is empty", i)
		}
		pinner.Pin(&buf[0])
		dataPtrs[i] = unsafe.Pointer(&buf[0])
		dataSizes[i] = C.size_t(len(buf))
	}

	var cImages *C.OgaImages
	res := C.LoadOgaImagesFromBuffers(
		&dataPtrs[0],
		&dataSizes[0],
		C.size_t(len(imageBuffers)),
		&cImages,
	)
	if err := OgaResultToError(res); err != nil {
		return nil, fmt.Errorf("LoadImagesFromBuffers failed: %w", err)
	}
	if cImages == nil {
		return nil, errors.New("LoadImagesFromBuffers returned nil without error")
	}

	return &Images{imagesPtr: cImages}, nil
}

// CreateMultiModalProcessor creates a multimodal processor from a model
func CreateMultiModalProcessor(model *model) (*MultiModalProcessor, error) {
	if !IsInitialized() {
		return nil, ErrNotInitialized
	}

	if model == nil || model.modelPtr == nil {
		return nil, errors.New("model is nil")
	}

	var cProcessor *C.OgaMultiModalProcessor
	res := C.CreateOgaMultiModalProcessor(model.modelPtr, &cProcessor)
	if err := OgaResultToError(res); err != nil {
		return nil, fmt.Errorf("CreateMultiModalProcessor failed: %w", err)
	}
	if cProcessor == nil {
		return nil, errors.New("CreateMultiModalProcessor returned nil without error")
	}

	return &MultiModalProcessor{processorPtr: cProcessor}, nil
}

// ProcessImages processes images with a prompt and returns named tensors
func (p *MultiModalProcessor) ProcessImages(prompt string, images *Images) (*NamedTensors, error) {
	if p.processorPtr == nil {
		return nil, errors.New("processor is not initialized")
	}
	if images == nil || images.imagesPtr == nil {
		return nil, errors.New("images is nil")
	}

	cPrompt := C.CString(prompt)
	defer C.free(unsafe.Pointer(cPrompt))

	var cTensors *C.OgaNamedTensors
	res := C.ProcessOgaImages(p.processorPtr, cPrompt, images.imagesPtr, &cTensors)
	if err := OgaResultToError(res); err != nil {
		return nil, fmt.Errorf("ProcessImages failed: %w", err)
	}
	if cTensors == nil {
		return nil, errors.New("ProcessImages returned nil without error")
	}

	return &NamedTensors{tensorsPtr: cTensors}, nil
}

// CreateGenerativeSessionAdvanced builds a GenAI config from a config directory,
// applies execution providers and options, creates the model and tokenizer, and returns a Session.
// providers: list of EP names in priority order (e.g., ["cuda"], ["NvTensorRtRtx"], ["OpenVINO"]).
// providerOptions: map of EP name -> map of key/value options.
func CreateGenerativeSessionAdvanced(configDirectoryPath string, providers []string, providerOptions map[string]map[string]string) (*Session, error) {
	if !IsInitialized() {
		return nil, ErrNotInitialized
	}

	// Create Config
	var cfg *C.OgaConfig
	cConfigPath := C.CString(configDirectoryPath)
	defer C.free(unsafe.Pointer(cConfigPath))
	res := C.CreateOgaConfig(cConfigPath, &cfg)
	if err := OgaResultToError(res); err != nil {
		return nil, fmt.Errorf("CreateOgaConfig failed: %w", err)
	}
	if cfg == nil {
		return nil, errors.New("CreateOgaConfig returned nil without error")
	}

	// Clear default providers to allow explicit configuration
	res = C.OgaConfigClearProviders(cfg)
	if err := OgaResultToError(res); err != nil {
		return nil, fmt.Errorf("OgaConfigClearProviders failed: %w", err)
	}

	// Append providers and set options
	for _, providerName := range providers {
		cp := C.CString(providerName)
		res = C.OgaConfigAppendProvider(cfg, cp)
		if err := OgaResultToError(res); err != nil {
			C.free(unsafe.Pointer(cp))
			return nil, fmt.Errorf("OgaConfigAppendProvider(%s) failed: %w", providerName, err)
		}
		if opts, ok := providerOptions[providerName]; ok {
			for k, v := range opts {
				ck := C.CString(k)
				cv := C.CString(v)
				res = C.OgaConfigSetProviderOption(cfg, cp, ck, cv)
				C.free(unsafe.Pointer(ck))
				C.free(unsafe.Pointer(cv))
				if err := OgaResultToError(res); err != nil {
					C.free(unsafe.Pointer(cp))
					return nil, fmt.Errorf("OgaConfigSetProviderOption(%s,%s=%s) failed: %w", providerName, k, v, err)
				}
			}
		}
		C.free(unsafe.Pointer(cp))
	}

	// Create Model from Config
	var cModel *C.OgaModel
	res = C.CreateOgaModelFromConfig(cfg, &cModel)
	if err := OgaResultToError(res); err != nil {
		return nil, fmt.Errorf("CreateOgaModelFromConfig failed: %w", err)
	}
	if cModel == nil {
		return nil, errors.New("CreateOgaModelFromConfig returned nil without error")
	}

	// Create Tokenizer
	model := model{modelPtr: cModel}
	tokenizer, err := newTokenizerFromModel(model)
	if err != nil {
		C.DestroyOgaModel(cModel)
		return nil, fmt.Errorf("newTokenizerFromModel failed: %w", err)
	}
	return &Session{
		model:      &model,
		tokenizer:  &tokenizer,
		statistics: Statistics{},
	}, nil
}

func CreateGenerativeSession(modelPath string) (*Session, error) {
	if !IsInitialized() {
		return nil, ErrNotInitialized
	}
	if modelPath == "" {
		return nil, errors.New("modelPath is empty")
	}
	cPath := C.CString(modelPath)
	defer C.free(unsafe.Pointer(cPath))

	var cModel *C.OgaModel
	res := C.CreateOgaModel(cPath, &cModel)
	if err := OgaResultToError(res); err != nil {
		return nil, fmt.Errorf("CreateOgaModel failed: %w", err)
	}
	if cModel == nil {
		return nil, errors.New("CreateOgaModel returned nil model without error")
	}

	model := model{modelPtr: cModel}
	tokenizer, err := newTokenizerFromModel(model)
	if err != nil {
		C.DestroyOgaModel(cModel)
		return nil, fmt.Errorf("newTokenizerFromModel failed: %w", err)
	}
	return &Session{
		model:     &model,
		tokenizer: &tokenizer,
	}, nil
}
