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

// Statistics captures generation performance metrics.
type Statistics struct {
	AvgPrefillSeconds float64
	TokensPerSecond   float64
	// cumulative
	cumulativePrefillSum           float64
	cumulativePrefillCount         int
	cumulativeTokens               int
	cumulativeTokenDurationSeconds float64
	// per-run
	runStart           time.Time
	runFirstTokenTimes []time.Time
	runTokenCount      int
}

// GetStatistics returns the last computed statistics for the session.
func (s *Session) GetStatistics() Statistics {
	return s.statistics
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
	defer s.mutex.Unlock()
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

		s.statistics.runStart = time.Now()
		s.statistics.runFirstTokenTimes = make([]time.Time, len(messages))
		s.statistics.runTokenCount = 0

		// finalize tokens/sec at the end of the run
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
				if s.statistics.runFirstTokenTimes[i].IsZero() {
					s.statistics.runFirstTokenTimes[i] = time.Now()
					prefill := s.statistics.runFirstTokenTimes[i].Sub(s.statistics.runStart).Seconds()
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
		defer C.free(unsafe.Pointer(cp))
		if err := OgaResultToError(res); err != nil {
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
					return nil, fmt.Errorf("OgaConfigSetProviderOption(%s,%s=%s) failed: %w", providerName, k, v, err)
				}
			}
		}
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
