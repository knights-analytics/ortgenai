package ortgenai

/*
#include "ort_genai_wrapper.h"
*/
import "C"

import (
	"context"
	"errors"
	"fmt"
	"strings"
	"unsafe"
)

// StreamingProcessor buffers mono float32 PCM at the model's sample rate and
// produces owned model inputs. Serialize its calls with those of its Generator.
// Destroy returned NamedTensors after generation finishes for that chunk.
type StreamingProcessor struct {
	ptr       *C.OgaStreamingProcessor
	generator *Generator
}

// NewStreamingProcessor creates a processor for a compatible streaming ASR model.
// The generator owns the model and destroys any remaining processors on Destroy.
func (g *Generator) NewStreamingProcessor() (*StreamingProcessor, error) {
	if err := g.check(); err != nil {
		return nil, err
	}
	var ptr *C.OgaStreamingProcessor
	if err := OgaResultToError(C.CreateStreamingProcessor(g.model.modelPtr, &ptr)); err != nil {
		if ptr != nil {
			C.DestroyStreamingProcessor(ptr)
		}
		return nil, fmt.Errorf("creating streaming processor: %w", err)
	}
	if ptr == nil {
		return nil, errors.New("streaming processor creation returned nil without error")
	}
	p := &StreamingProcessor{ptr: ptr, generator: g}
	if g.streamingProcessors == nil {
		g.streamingProcessors = make(map[*StreamingProcessor]struct{})
	}
	g.streamingProcessors[p] = struct{}{}
	return p, nil
}

func (p *StreamingProcessor) check() error {
	if p == nil || p.ptr == nil || p.generator == nil {
		return errors.New("streaming processor is destroyed")
	}
	return p.generator.check()
}

// Process copies audio into the native processor. Supply at most the package's
// configured number of samples per chunk on each call. A nil result means more
// audio is needed, or voice activity detection omitted the chunk. Samples are not retained.
func (p *StreamingProcessor) Process(samples []float32) (*NamedTensors, error) {
	if err := p.check(); err != nil {
		return nil, err
	}
	if len(samples) == 0 {
		return nil, errors.New("audio samples are empty")
	}
	var ptr *C.OgaNamedTensors
	result := C.StreamingProcessorProcess(p.ptr, (*C.float)(unsafe.Pointer(&samples[0])), C.size_t(len(samples)), &ptr)
	return streamingTensors(result, ptr)
}

// Flush pads buffered audio with silence. It returns nil when no input remains.
func (p *StreamingProcessor) Flush() (*NamedTensors, error) {
	if err := p.check(); err != nil {
		return nil, err
	}
	var ptr *C.OgaNamedTensors
	result := C.StreamingProcessorFlush(p.ptr, &ptr)
	return streamingTensors(result, ptr)
}

func streamingTensors(result *C.OgaResult, ptr *C.OgaNamedTensors) (*NamedTensors, error) {
	if err := OgaResultToError(result); err != nil {
		if ptr != nil {
			C.DestroyOgaNamedTensors(ptr)
		}
		return nil, fmt.Errorf("processing streaming audio: %w", err)
	}
	if ptr == nil {
		return nil, nil
	}
	return &NamedTensors{tensorsPtr: ptr}, nil
}

// SetOption configures use_vad, vad_threshold, silence_duration_ms, or prefix_padding_ms.
// Values and model support are validated by the runtime.
func (p *StreamingProcessor) SetOption(key, value string) error {
	if err := p.check(); err != nil {
		return err
	}
	if key == "" || strings.ContainsRune(key, '\x00') || strings.ContainsRune(value, '\x00') {
		return errors.New("processor option must have a non-empty key and contain no NUL bytes")
	}
	k, v := C.CString(key), C.CString(value)
	defer C.free(unsafe.Pointer(k))
	defer C.free(unsafe.Pointer(v))
	if err := OgaResultToError(C.StreamingProcessorSetOption(p.ptr, k, v)); err != nil {
		return fmt.Errorf("setting streaming option: %w", err)
	}
	return nil
}

// Option returns a copied processor setting.
func (p *StreamingProcessor) Option(key string) (string, error) {
	if err := p.check(); err != nil {
		return "", err
	}
	if key == "" || strings.ContainsRune(key, '\x00') {
		return "", errors.New("processor option key must be non-empty and contain no NUL bytes")
	}
	k := C.CString(key)
	defer C.free(unsafe.Pointer(k))
	var value *C.char
	result := C.StreamingProcessorGetOption(p.ptr, k, (**C.char)(unsafe.Pointer(&value)))
	if value != nil {
		defer C.DestroyOgaString(value)
	}
	if err := OgaResultToError(result); err != nil {
		return "", fmt.Errorf("getting streaming option: %w", err)
	}
	if value == nil {
		return "", errors.New("streaming option returned nil without error")
	}
	return C.GoString(value), nil
}

// Destroy releases the native processor and its buffered audio.
func (p *StreamingProcessor) Destroy() {
	if p == nil {
		return
	}
	if p.ptr != nil {
		C.DestroyStreamingProcessor(p.ptr)
		p.ptr = nil
	}
	if p.generator != nil {
		delete(p.generator.streamingProcessors, p)
		p.generator = nil
	}
}

// StreamingASR owns a generator, audio processor, and ASR post-processor for one
// audio stream. Serialize all calls on one goroutine pinned with runtime.LockOSThread.
// Configure timestamps in genai_config.json before creation. Destroy the stream
// after any processing error; a failed chunk cannot be replayed safely.
type StreamingASR struct {
	generator     *Generator
	processor     *StreamingProcessor
	postProcessor *ASRPostProcessor
	finished      bool
	failed        bool
}

// CreateStreamingASR loads a streaming ASR package using its native search defaults.
func CreateStreamingASR(modelPath string) (*StreamingASR, error) {
	if !IsInitialized() {
		return nil, ErrNotInitialized
	}
	if modelPath == "" || strings.ContainsRune(modelPath, '\x00') {
		return nil, errors.New("model path must be non-empty and contain no NUL bytes")
	}
	path := C.CString(modelPath)
	defer C.free(unsafe.Pointer(path))
	g := &Generator{model: &model{}}
	s := &StreamingASR{generator: g}
	ok := false
	defer func() {
		if !ok {
			s.Destroy()
		}
	}()
	if err := OgaResultToError(C.CreateOgaModel(path, &g.model.modelPtr)); err != nil {
		return nil, fmt.Errorf("creating ASR model: %w", err)
	}
	if g.model.modelPtr == nil {
		return nil, errors.New("ASR model creation returned nil without error")
	}
	if err := OgaResultToError(C.CreateOgaGeneratorParams(g.model.modelPtr, &g.params)); err != nil {
		return nil, fmt.Errorf("creating ASR parameters: %w", err)
	}
	if g.params == nil {
		return nil, errors.New("ASR parameters creation returned nil without error")
	}
	if err := OgaResultToError(C.CreateOgaGenerator(g.model.modelPtr, g.params, &g.ptr)); err != nil {
		return nil, fmt.Errorf("creating ASR generator: %w", err)
	}
	if g.ptr == nil {
		return nil, errors.New("ASR generator creation returned nil without error")
	}
	var err error
	s.processor, err = g.NewStreamingProcessor()
	if err != nil {
		return nil, err
	}
	s.postProcessor, err = g.NewASRPostProcessor()
	if err != nil {
		return nil, err
	}
	ok = true
	return s, nil
}

func (s *StreamingASR) check() error {
	if s == nil || s.generator == nil {
		return errors.New("ASR stream is destroyed")
	}
	if s.failed {
		return errors.New("ASR stream failed; destroy it and create a new stream")
	}
	if s.finished {
		return errors.New("ASR stream is finalized")
	}
	return s.generator.check()
}

// SetOption configures the audio processor before supplying audio.
func (s *StreamingASR) SetOption(key, value string) error {
	if err := s.check(); err != nil {
		return err
	}
	return s.processor.SetOption(key, value)
}

// Option returns a copied audio processor setting.
func (s *StreamingASR) Option(key string) (string, error) {
	if err := s.check(); err != nil {
		return "", err
	}
	return s.processor.Option(key)
}

// SetRuntimeOption configures generator settings such as the model's lang_id.
func (s *StreamingASR) SetRuntimeOption(key, value string) error {
	if err := s.check(); err != nil {
		return err
	}
	return s.generator.RuntimeOption(key, value)
}

// Process transcribes one PCM chunk and returns copied incremental results.
// Split audio at the model's sample rate into chunks no larger than its configured
// chunk size; the native processor produces at most one input batch per call.
// Results returned alongside an error contain output produced before the failure.
// Cancellation is observed between native calls.
func (s *StreamingASR) Process(ctx context.Context, samples []float32) ([]ASRResult, error) {
	if err := s.check(); err != nil {
		return nil, err
	}
	if err := ctx.Err(); err != nil {
		return nil, err
	}
	if len(samples) == 0 {
		return nil, errors.New("audio samples are empty")
	}
	inputs, err := s.processor.Process(samples)
	if err != nil {
		s.failed = true
		return nil, err
	}
	results, err := s.decode(ctx, inputs)
	if err != nil {
		s.failed = true
	}
	return results, err
}

// Flush transcribes remaining audio and finalizes trailing word and segment
// timestamps. A successful call ends the stream; repeated Flush calls return no results.
func (s *StreamingASR) Flush(ctx context.Context) ([]ASRResult, error) {
	if s != nil && s.generator != nil && s.finished {
		return nil, nil
	}
	if err := s.check(); err != nil {
		return nil, err
	}
	if err := ctx.Err(); err != nil {
		return nil, err
	}
	inputs, err := s.processor.Flush()
	if err != nil {
		s.failed = true
		return nil, err
	}
	results, err := s.decode(ctx, inputs)
	if err == nil {
		err = ctx.Err()
	}
	if err == nil {
		err = s.postProcessor.Finalize()
	}
	if err == nil {
		var trailing []ASRResult
		trailing, err = s.postProcessor.Results()
		results = append(results, trailing...)
	}
	if err != nil {
		s.failed = true
		return results, err
	}
	s.finished = true
	return results, nil
}

func (s *StreamingASR) decode(ctx context.Context, inputs *NamedTensors) ([]ASRResult, error) {
	if inputs == nil {
		return nil, nil
	}
	defer inputs.Destroy()
	if err := s.generator.SetInputs(inputs); err != nil {
		return nil, err
	}
	var results []ASRResult
	for {
		if err := ctx.Err(); err != nil {
			return results, err
		}
		done, err := s.generator.IsDone()
		if err != nil {
			return results, err
		}
		if done {
			return results, nil
		}
		if err := s.generator.GenerateNextToken(); err != nil {
			return results, err
		}
		if err := s.postProcessor.ProcessNext(); err != nil {
			return results, err
		}
		batch, err := s.postProcessor.Results()
		if err != nil {
			return results, err
		}
		results = append(results, batch...)
	}
}

// Destroy releases post-processing, audio, generator, and model resources in order.
func (s *StreamingASR) Destroy() {
	if s == nil {
		return
	}
	if s.postProcessor != nil {
		s.postProcessor.Destroy()
		s.postProcessor = nil
	}
	if s.processor != nil {
		s.processor.Destroy()
		s.processor = nil
	}
	if s.generator != nil {
		s.generator.Destroy()
		s.generator = nil
	}
}
