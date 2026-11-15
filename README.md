# ortgenai

Go bindings for the ONNX Runtime GenAI C API.

This package provides a thin, idiomatic Go wrapper around the ONNX Runtime GenAI shared library, exposing:

- Session and model creation from local model folders
- Tokenization and chat templating
- Owned tokenizer and tensor resources, including batch encode/decode
- Session-based batched, streaming text generation with per-token deltas
- Synchronous engine requests with reusable event buffers and per-turn options
- Standalone generators, tensor/logit access, and MTP speculative generation
- Audio resources, model adapters, runtime settings, and process-wide device/logging controls
- Runtime statistics (tokens/sec, prefill timings)
- Provider selection and advanced provider options
- Multimodal input support (text, images, and audio)

The GenAI shared library is loaded dynamically. Linux (`.so`) and Windows (`.dll`) are supported.

Note: This implementation is still alpha so the API may change in future releases. You might want to rely on [hugot](https://github.com/knights-analytics/hugot) for a higher-level interface to ONNX Runtime GenAI in Go.

## Contents

- Requirements
- Installation
- Quick start
- Advanced usage
- Running tests
- Docker and containerized tests
- Troubleshooting
- License


## Requirements

- Go 1.27+
- Linux with glibc, or 64-bit Windows
- ONNX Runtime GenAI shared library and dependencies available at runtime:
  - ORT GenAI `0.20.0` or newer with the explicit engine/request API
  - Linux: `libonnxruntime-genai.so` and `libonnxruntime.so`
  - Windows: `onnxruntime-genai.dll` and `onnxruntime.dll`
- A local model directory compatible with ONNX Runtime GenAI (e.g., a converted `Phi-3.5` model folder)


## Installation

```bash
go get github.com/knights-analytics/ortgenai
```

At runtime, the wrapper needs to load the ONNX Runtime GenAI shared library. You can:

1) Place the GenAI library and its ONNX Runtime dependency next to your application binary, or
2) Call `genai.SetSharedLibraryPath(...)` before initialization with the library path (`.so` on Linux or `.dll` on Windows).


## Quick start

```go
package main

import (
    "context"
    "fmt"
    "time"

    genai "github.com/knights-analytics/ortgenai"
)

func main() {
    // Optional: set explicit path if the .so isn't on the default loader path
    // Note: ensure libonnxruntime.so is colocated with libonnxruntime-genai.so
    genai.SetSharedLibraryPath("/usr/lib/libonnxruntime-genai.so")

    if err := genai.InitializeEnvironment(); err != nil {
        panic(fmt.Errorf("init failed: %w", err))
    }
    defer func() {
        err = genai.DestroyEnvironment()
        if err != nil {
            panic(fmt.Errorf("destroy environment: %w", err))
        }
    }()

    // Point to a local model folder compatible with ONNX Runtime GenAI
    session, err := genai.CreateSession("./models/phi3.5")
    if err != nil {
        panic(fmt.Errorf("create session: %w", err))
    }
    defer session.Destroy()

    // Prepare one or more conversations (batched)
    conv1 := []genai.Message{
        {Role: "system", Content: "You are a helpful assistant."},
        {Role: "user", Content: "What is the capital of France?"},
    }

    ctx, cancel := context.WithTimeout(context.Background(), 2*time.Minute)
    defer cancel()

    deltas, errs, err := session.Generate(ctx, [][]genai.Message{conv1})
    if err != nil {
        panic(fmt.Errorf("generate: %w", err))
    }

    // Stream tokens as they arrive
    for {
        select {
        case d, ok := <-deltas:
            if !ok { // stream completed
                stats := session.GetStatistics()
                fmt.Printf("\nTokens/sec: %.2f\n", stats.TokensPerSecond)
                return
            }
            fmt.Print(d.Tokens) // append to your buffer
        case e := <-errs:
            if e != nil { panic(e) }
        }
    }
}
```


## Advanced usage

### Explicit shared library path

```go
genai.SetSharedLibraryPath("/opt/onnxruntime/lib/libonnxruntime-genai.so")
if err := genai.InitializeEnvironment(); err != nil { /* handle */ }
```

If not set, the code tries `libonnxruntime-genai.so` on Linux or `onnxruntime-genai.dll` on Windows. Keep the matching ONNX Runtime library (`libonnxruntime.so` or `onnxruntime.dll`) alongside it.

### Provider selection and options

If you have a JSON config path and custom providers to use (e.g., CUDA, CPU) you can create a session with advanced settings:

```go
providers := []string{"cpu"} // or e.g., []string{"cuda", "cpu"}
providerOptions := map[string]map[string]string{
    "cpu": {"intra_op_num_threads": "4"},
}

session, err := genai.CreateSessionWithOptions(
    "./path/to/config.json", // model/session config from GenAI tooling
    providers,
    providerOptions,
)
```

### Batched generation

`Session.Generate` accepts multiple conversations in one call: `[][]Message`. The returned channel carries `SequenceDelta` items, each labeled with the `Sequence` index so you can route output per-conversation.

### Generative ASR with timestamps

`Session.GenerateASR` runs audio conversations through a generative ASR model and returns copied text, word timestamps, and segment timestamps. Supply the audio and the model-specific prompt in each `MultimodalConversation`; results are labeled with the conversation index and emitted as each conversation completes. Keep the conversations, media, and session alive and unchanged until the result and error channels close.

### Explicit engine turns

`Engine` uses a synchronous, owner-thread API rather than channels or background dispatch. Keep all calls for an engine, its requests, and event buffers serialized on the same OS thread. Native events and usage views are copied before `Run` returns, so Go values remain valid when the event buffer is reused.

```go
runtime.LockOSThread()
defer runtime.UnlockOSThread()

engine, err := genai.CreateEngine("./models/phi3.5")
if err != nil { panic(err) }
defer engine.Destroy()

tokenizer, err := genai.CreateTokenizer("./models/phi3.5")
if err != nil { panic(err) }
defer tokenizer.Destroy()
inputTokenIDs, err := tokenizer.Encode("What is the capital of France?")
if err != nil { panic(err) }

buffer, err := engine.CreateEventBuffer(8)
if err != nil { panic(err) }
defer buffer.Destroy()

request, err := engine.CreateRequest(nil)
if err != nil { panic(err) }
defer request.Destroy()

turnOptions, err := request.CreateTurnOptions()
if err != nil { panic(err) }
defer turnOptions.Destroy()
if err := turnOptions.SetMaxGeneratedTokens(128); err != nil { panic(err) }

turnID, err := request.BeginTurn(inputTokenIDs, turnOptions)
if err != nil { panic(err) }
for {
    events, err := engine.Run(buffer)
    if err != nil { panic(err) }
    finished := false
    for _, event := range events {
        if event.TurnID == turnID && event.Flags&genai.EventFlagTurnFinished != 0 {
            finished = true
        }
    }
    if finished { break }
}
```

Token IDs can be supplied directly to `BeginTurn`; `Tokenizer.Encode` and `Tokenizer.Decode` provide copied Go values. Destroy requests, tokenizers, and buffers before shutting down the GenAI environment.

### Engine vision turns

`Request.BeginTurnWithImages(formattedPrompt, images, turnOptions)` accepts an incremental UTF-8 prompt and ordered images. Apply the package's chat template first, with one model-specific image marker per image. The Engine performs image preparation before returning and schedules neural encoding during `Engine.Run`. You may destroy the images after a successful `BeginTurnWithImages` call. Supply `nil` images for a later text-only continuation using the same API.

Use an Engine-compatible vision package, such as a packed Qwen vision export. Ordinary Generator vision exports are not interchangeable. The same owner-thread rules apply as for token turns.

Completed events expose `TerminalTokenID`: the sampled EOS token that ended the turn, or `-1` when no EOS token ended it. It is completion metadata and should not be appended to generated text. `FinishReason` and `MatchedStopStringIndex` remain available separately.

### Chunked streaming ASR

`CreateStreamingASR` owns the model, generator, streaming audio processor, and ASR post-processor for one audio stream. Supply mono float32 PCM at the sample rate specified by the package, with no more than one configured chunk of samples per `Process` call. Calls are synchronous; serialize them on a goroutine pinned with `runtime.LockOSThread`.

Enable timestamps under `model` in `genai_config.json` before loading a compatible Nemotron streaming speech package:

```json
{
  "timestamp_level": "all",
  "segment_separators": [".", "?", "!"],
  "segment_gap_threshold_seconds": 1.0
}
```

Choose `word`, `segment`, `all`, or `off`. `ASRTimestamp.StartTime` and `StopTime` are **int64 milliseconds**, matching ORT GenAI 0.20. Frame indices are also int64. Plain transcript fragments work with timestamps disabled.

After initializing the GenAI environment, the streaming flow is:

```go
runtime.LockOSThread()
defer runtime.UnlockOSThread()

stream, err := genai.CreateStreamingASR("./models/nemotron-speech")
if err != nil { return err }
defer stream.Destroy()

// VAD requires a compatible VAD configuration in the model package.
if err := stream.SetOption("use_vad", "false"); err != nil { return err }

// pcmChunks contains mono float32 PCM chunks at the package's sample rate.
for _, chunk := range pcmChunks {
    results, err := stream.Process(ctx, chunk)
    for _, result := range results {
        fmt.Print(result.Text)
        // result.Words and result.Segments contain copied timestamp records.
    }
    if err != nil { return err }
}
results, err := stream.Flush(ctx)
for _, result := range results { fmt.Print(result.Text) }
if err != nil { return err }
```

`Process` can return no results while audio is buffered or VAD skips a chunk. `Flush` pads remaining audio and emits trailing words and segments. A successful flush ends the stream. Results remain valid after later calls or destruction. Append transcript fragments or timestamp-record text to a transcript; appending both duplicates text. Context cancellation is checked between native calls. Destroy a stream after a processing failure and create a new one for new audio.

For direct control, `Generator.NewStreamingProcessor` exposes `Process`, `Flush`, `SetOption`, and `Option`. Pass each non-nil `NamedTensors` result intact to `Generator.SetInputs`, generate until done, and copy `ASRPostProcessor.Results` after **every** `ProcessNext` and `Finalize`. The native processor replaces its result batch on each mutation. Keep the tensor inputs alive until generation for that chunk finishes, then destroy them. Destroy the ASR post-processor before its generator. The generator releases its remaining streaming processors when destroyed.

### GPU memory sizing

`CreateEngineWithOptions` applies per-model CUDA cache-sizing allowances to a package configured with `engine.dynamic_batching`:

```go
engine, err := genai.CreateEngineWithOptions("./models/qwen", genai.EngineOptions{
    DeviceMemoryBudgetBytes:   12 << 30,
    DeviceMemoryHeadroomBytes: 1 << 30,
})
```

Zero fields preserve the package settings. Equivalent package JSON fields are `engine.dynamic_batching.device_memory_budget_bytes` and `device_memory_headroom_bytes`. The runtime validates provider and package support. These values guide cache sizing and retained GPU snapshots; they are not hard allocation quotas. Native Qwen and Bolt retain fixed graph caches and apply the allowance to snapshots. Profile headroom with the intended package, context length, and concurrency.

### Statistics

After generation, inspect `session.GetStatistics()` for fields such as `TokensPerSecond`, cumulative token counts, and prefill timings.


## Running tests

Local tests require the matching GenAI shared library and local model directories. Model-dependent tests skip when their fixtures are absent. On Windows, set `ONNXRUNTIME_DIR` to the directory containing the DLLs before running tests, for example:

- `libonnxruntime-genai.so` on Linux or `onnxruntime-genai.dll` on Windows (adjust via `SetSharedLibraryPath`), and
- the model directories referenced by the integration tests.

Run:

```bash
go test ./...
```

```powershell
$env:ONNXRUNTIME_DIR='E:\ort-lib'; go test ./...
```


## Docker and containerized tests

Two Dockerfiles are provided:

- `Dockerfile` — base image with dependencies
- `test.Dockerfile` — image to run unit tests

Helper scripts are available in `scripts/`:

- `scripts/run-unit-tests-container.sh` — build the test image and run tests in a container
- `scripts/run-unit-test.sh` — run tests directly (expects environment to be prepared)

You may also use `compose-test.yaml` to orchestrate test runs.


## Troubleshooting

- error loading GenAI shared library: Ensure the path to the platform-specific GenAI library is correct and readable by the process. Set it explicitly with `SetSharedLibraryPath`.
- Windows build errors about a missing C compiler: install a CGo-compatible compiler such as MinGW-w64 GCC, ensure it is on `PATH`, and enable CGo.
- missing `Oga...` symbols or "missing Oga..." errors: The GenAI library must export required symbols (e.g., `OgaCreateModel`). Make sure the GenAI and ONNX Runtime libraries are compatible and colocated.
- segmentation fault on load: Verify that your system’s CUDA/CPU provider dependencies match the `.so` build (driver/runtime versions).
- no output / stuck: Ensure your model folder is valid for ONNX Runtime GenAI and accessible; increase timeouts during first-run warmup.


## License

This project is licensed under the terms of the MIT License. See `LICENSE` for details.
