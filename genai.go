package ortgenai

/*
#cgo CFLAGS: -O2 -g
#include "ort_genai_wrapper.h"
*/
import "C"

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
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

var defaultMaxLength = 2024

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

func (g *generator) addSequences(sequences *sequences) error {
	// add sequences to generator
	res := C.GeneratorAppendTokenSequences(g.generatorPtr, sequences.sequencesPtr)
	if err := OgaResultToError(res); err != nil {
		return fmt.Errorf("GeneratorAppendTokenSequences failed: %w", err)
	}
	// Sequences are no longer needed after appending; destroy to avoid leaks.
	C.DestroyOgaSequences(sequences.sequencesPtr)
	sequences.sequencesPtr = nil
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

type SessionOptions struct {
	Multimodal bool
}

type Session struct {
	model      *model
	processor  *multiModalProcessor
	tokenizer  *tokenizer
	statistics Statistics
	options    SessionOptions
	mutex      sync.Mutex // the C API is not thread-safe
}

type SequenceDelta struct {
	Sequence int
	Tokens   string
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

func (g *generator) IsDone() bool {
	return bool(C.IsDone(g.generatorPtr))
}

// getLastToken returns the last token for the sequence at index seqIdx.
// It returns ok=false if the sequence has no tokens or data is unavailable.
func getLastToken(g *generator, seqIdx int) (C.int32_t, bool) {
	numTokens := C.GeneratorGetSequenceCount(g.generatorPtr, C.size_t(seqIdx))
	if numTokens == 0 {
		return 0, false
	}
	seqData := C.GeneratorGetSequenceData(g.generatorPtr, C.size_t(seqIdx))
	if seqData == nil {
		return 0, false
	}
	arr := (*[1 << 30]C.int32_t)(unsafe.Pointer(seqData))
	return arr[numTokens-1], true
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

func (t *tokenizer) createTokenizerStream() (*tokenizerStream, error) {
	var cStream *C.OgaTokenizerStream
	res := C.CreateOgaTokenizerStream(t.tokenizerPtr, &cStream)
	if err := OgaResultToError(res); err != nil {
		return nil, fmt.Errorf("CreateOgaTokenizerStream failed: %w", err)
	}
	if cStream == nil {
		return nil, errors.New("CreateOgaTokenizerStream returned nil without error")
	}
	return &tokenizerStream{streamPtr: cStream}, nil
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
		stream, err := t.createTokenizerStream()
		if err != nil {
			return nil, nil, fmt.Errorf("createTokenizerStream failed: %w", err)
		}
		tokenizerStreams = append(tokenizerStreams, stream)
	}
	return sequences, tokenizerStreams, nil
}

func (s *Session) createGenerator(generationOptions *GenerationOptions) (*generator, error) {
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

// startGenerationGoroutine launches the unified generation loop and returns output and error channels.
// Assumes the session mutex is already locked; it will be unlocked inside the goroutine when done.
func startGenerationGoroutine(ctx context.Context, s *Session, generator *generator, tokenizerStreams []*tokenizerStream, seqCount int) (<-chan SequenceDelta, <-chan error) {
	outputChan := make(chan SequenceDelta, 10)
	errChan := make(chan error, 1)
	go func() {
		defer close(outputChan)
		defer close(errChan)
		defer generator.destroy()
		defer func() {
			for _, ts := range tokenizerStreams {
				if ts != nil {
					ts.destroy()
				}
			}
		}()

		// Per-run statistics (goroutine-local to avoid races)
		runStart := time.Now()
		runFirstTokenTimes := map[int]time.Time{}
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

		firstEmitted := make([]bool, seqCount)
		lastChar := make([]rune, seqCount)

		for {
			select {
			case <-ctx.Done():
				return
			default:
			}

			if generator.IsDone() {
				return
			}

			result := C.GeneratorGenerateNextToken(generator.generatorPtr)
			if err := OgaResultToError(result); err != nil {
				sendGenerationError(errChan, err)
				return
			}

			// Iterate over each sequence in the batch
			for i := 0; i < seqCount; i++ {
				lastToken, ok := getLastToken(generator, i)
				if !ok {
					continue
				}

				decoded, decodeErr := tokenizerStreams[i].Decode(lastToken)
				if decodeErr != nil {
					sendGenerationError(errChan, decodeErr)
					return
				}
				if decoded == "" {
					continue
				}
				// normalization: skip leading spaces for first token, avoid repeated '.' at end
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
	return outputChan, errChan
}

func (s *Session) Generate(ctx context.Context, messages [][]Message, generationOptions *GenerationOptions) (<-chan SequenceDelta, <-chan error, error) {
	s.mutex.Lock()
	sequences, tokenizerStreams, tokenizeErr := s.tokenizer.tokenizeMessages(messages)
	if tokenizeErr != nil {
		s.mutex.Unlock()
		return nil, nil, fmt.Errorf("TokenizeMessages failed: %w", tokenizeErr)
	}

	if generationOptions == nil {
		generationOptions = &GenerationOptions{
			MaxLength: defaultMaxLength,
			BatchSize: len(messages),
		}
	}
	if generationOptions.BatchSize <= 0 {
		generationOptions.BatchSize = len(messages)
	}

	generator, err := s.createGenerator(generationOptions)
	if err != nil {
		s.mutex.Unlock()
		return nil, nil, err
	}
	if err = generator.addSequences(sequences); err != nil {
		generator.destroy()
		s.mutex.Unlock()
		return nil, nil, fmt.Errorf("failed to add sequences to generator: %w", err)
	}

	outputChan, errChan := startGenerationGoroutine(ctx, s, generator, tokenizerStreams, len(messages))
	return outputChan, errChan, nil
}

// GenerateWithImages generates text using pre-processed named tensors (for multimodal inputs).
// Currently only supports a single prompt.
func (s *Session) GenerateWithImages(ctx context.Context, prompts []string, images *Images, generationOptions *GenerationOptions) (<-chan SequenceDelta, <-chan error, error) {
	s.mutex.Lock()

	if !s.options.Multimodal {
		s.mutex.Unlock()
		return nil, nil, errors.New("session was not created with multimodal support")
	}

	if len(prompts) != 1 {
		s.mutex.Unlock()
		return nil, nil, errors.New("GenerateWithImages currently supports only a single prompt")
	}

	tensors, err := s.processor.ProcessImages(prompts[0], images)
	if err != nil {
		s.mutex.Unlock()
		return nil, nil, fmt.Errorf("ProcessImages failed: %w", err)
	}
	defer tensors.Destroy()

	if generationOptions == nil {
		generationOptions = &GenerationOptions{MaxLength: defaultMaxLength}
	}
	if generationOptions.BatchSize != len(prompts) {
		generationOptions.BatchSize = len(prompts)
	}

	generator, err := s.createGenerator(generationOptions)
	if err != nil {
		return nil, nil, fmt.Errorf("failed to create generator: %w", err)
	}

	// Set the named tensors as inputs
	if err := generator.setInputs(tensors); err != nil {
		generator.destroy()
		s.mutex.Unlock()
		return nil, nil, fmt.Errorf("failed to set inputs: %w", err)
	}

	// Create tokenizer streams per sequence (align with Generate behavior)
	numSeq := generationOptions.BatchSize
	tokenizerStreams := make([]*tokenizerStream, 0, numSeq)
	for i := 0; i < numSeq; i++ {
		ts, err := s.tokenizer.createTokenizerStream()
		if err != nil {
			for _, t := range tokenizerStreams {
				if t != nil {
					t.destroy()
				}
			}
			generator.destroy()
			s.mutex.Unlock()
			return nil, nil, fmt.Errorf("failed to create tokenizer stream: %w", err)
		}
		tokenizerStreams = append(tokenizerStreams, ts)
	}
	outputChan, errChan := startGenerationGoroutine(ctx, s, generator, tokenizerStreams, numSeq)
	return outputChan, errChan, nil
}

func (s *Session) Destroy() {
	if s.model != nil {
		s.model.destroy()
		s.model = nil
	}
	if s.tokenizer != nil {
		s.tokenizer.destroy()
		s.tokenizer = nil
	}
	if s.processor != nil {
		s.processor.destroy()
		s.processor = nil
	}
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

func CreateGenerativeSession(modelPath string, options *SessionOptions) (*Session, error) {
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

	var processor *multiModalProcessor
	var errProcessor error

	if options.Multimodal {
		processor, errProcessor = createMultiModalProcessor(&model)
		if errProcessor != nil {
			tokenizer.destroy()
			model.destroy()
			return nil, fmt.Errorf("createMultiModalProcessor failed: %w", errProcessor)
		}
	}

	return &Session{
		model:     &model,
		tokenizer: &tokenizer,
		processor: processor,
		options:   *options,
	}, nil
}
