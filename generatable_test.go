package ortgenai

import (
	"context"
	"encoding/base64"
	"encoding/json"
	"fmt"
	"os"
	"regexp"
	"strings"
	"sync"
	"testing"
	"time"
)

type Generatable interface {
	Generate(ctx context.Context, messages [][]Message, tools []string, options *GenerationOptions) (<-chan SequenceDelta, <-chan error, error)
	GetStatistics() *Statistics
}

// testImagePNG is a minimal valid 1x1 red PNG image (base64 encoded).
var testImagePNG = "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mP8z8BQDwAEhQGAhKmMIQAAAABJRU5ErkJggg=="

var testJSONs = []string{
	`{
	"id": "12345",
	"name": "John Doe",
	"email": "john.doe@example.com",
	"age": 30,
	"address": {
		"street": "123 Main St",
		"city": "Anytown",
		"state": "CA",
		"zip": "12345"
	},
	"phone_numbers": [
		{
			"type": "home",
			"number": "555-1234"
		},
		{
			"type": "work",
			"number": "555-5678"
		}
	],
	"preferences": {
		"contact_method": "email",
		"newsletter_subscribed": true
	},
	"tags": ["customer", "premium", "active"],
	"metadata": {
		"last_login": "2024-01-15T10:30:00Z",
		"account_created": "2020-06-20T14:45:00Z"
	}
}`,
	`{
	"id": "67890",
	"name": "Jane Smith",
	"email": "jane.smith@example.com",
	"age": 25,
	"address": {
		"street": "456 Elm St",
		"city": "Othertown",
		"state": "NY",
		"zip": "67890"
	},
	"phone_numbers": [
		{
			"type": "mobile",
			"number": "555-8765"
		}
	],
	"preferences": {
		"contact_method": "phone",
		"newsletter_subscribed": false
	},
	"tags": ["lead", "new"],
	"metadata": {
		"last_login": "2024-02-20T09:15:00Z",
		"account_created": "2023-03-10T11:20:00Z"
	}
}`,
}

var inputMessagesFirstGeneration = []Message{
	{Role: "system", Content: "You are a helpful assistant."},
	{Role: "user", Content: fmt.Sprintf(`Hello, I have the following two
		jsons that represent two users:

		first: %s

		second: %s

		Please compare them and tell me the main differences between these users.
		`, testJSONs[0], testJSONs[1])},
}

var inputMessagesSecondGeneration = []Message{
	{Role: "system", Content: "You are a helpful assistant."},
	{Role: "user", Content: "What is the capital of France?"},
}

func TestLoadImageFromDataURI(t *testing.T) {
	SetSharedLibraryPath(getLibraryPath())

	if err := InitializeEnvironment(); err != nil {
		t.Fatalf("failed to initialize environment: %v", err)
	}
	defer func() {
		if err := DestroyEnvironment(); err != nil {
			t.Fatalf("failed to destroy environment: %v", err)
		}
	}()

	dataURI := "data:image/png;base64," + testImagePNG

	images, err := LoadImage(dataURI)
	if err != nil {
		t.Fatalf("LoadImage with data URI failed: %v", err)
	}
	defer images.Destroy()

	if images.imagesPtr == nil {
		t.Fatal("images.imagesPtr is nil after LoadImage with data URI")
	}
}

func TestLoadImagesFromBuffers(t *testing.T) {
	SetSharedLibraryPath(getLibraryPath())

	if err := InitializeEnvironment(); err != nil {
		t.Fatalf("failed to initialize environment: %v", err)
	}
	defer func() {
		if err := DestroyEnvironment(); err != nil {
			t.Fatalf("failed to destroy environment: %v", err)
		}
	}()

	imageData, err := base64.StdEncoding.DecodeString(testImagePNG)
	if err != nil {
		t.Fatalf("failed to decode test image: %v", err)
	}

	buffers := [][]byte{imageData, imageData}
	images, err := LoadImagesFromBuffers(buffers)
	if err != nil {
		t.Fatalf("LoadImagesFromBuffers failed: %v", err)
	}
	defer images.Destroy()

	if images.imagesPtr == nil {
		t.Fatal("images.imagesPtr is nil after LoadImagesFromBuffers")
	}
}

func TestParseDataURI(t *testing.T) {
	tests := []struct {
		name    string
		input   string
		wantErr bool
	}{
		{"valid PNG data URI", "data:image/png;base64," + testImagePNG, false},
		{"valid JPEG data URI", "data:image/jpeg;base64," + testImagePNG, false},
		{"missing data: prefix", "image/png;base64," + testImagePNG, true},
		{"missing comma separator", "data:image/png;base64" + testImagePNG, true},
		{"not base64 encoded", "data:image/png," + testImagePNG, true},
		{"invalid base64", "data:image/png;base64,!!!invalid!!!", true},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			_, err := parseDataURI(tt.input)
			if (err != nil) != tt.wantErr {
				t.Errorf("parseDataURI() error = %v, wantErr %v", err, tt.wantErr)
			}
		})
	}
}

func TestLoadImageFromBuffer(t *testing.T) {
	SetSharedLibraryPath(getLibraryPath())

	if err := InitializeEnvironment(); err != nil {
		t.Fatalf("failed to initialize environment: %v", err)
	}
	defer func() {
		if err := DestroyEnvironment(); err != nil {
			t.Fatalf("failed to destroy environment: %v", err)
		}
	}()

	imageData, err := base64.StdEncoding.DecodeString(testImagePNG)
	if err != nil {
		t.Fatalf("failed to decode test image: %v", err)
	}

	images, err := LoadImageFromBuffer(imageData)
	if err != nil {
		t.Fatalf("LoadImageFromBuffer failed: %v", err)
	}
	defer images.Destroy()

	if images.imagesPtr == nil {
		t.Fatal("images.imagesPtr is nil after LoadImageFromBuffer")
	}
}

func testGenericGeneration(t *testing.T, g Generatable, messages [][]Message, options *GenerationOptions) {
	t.Helper()
	ctx, cancel := context.WithTimeout(context.Background(), 5*time.Minute)
	defer cancel()

	generateChan, errChan, err := g.Generate(ctx, messages, nil, options)
	if err != nil {
		t.Fatalf("failed to start generation: %v", err)
	}

	outputs := make([][]string, len(messages))

	for token := range generateChan {
		if !token.EOSReached {
			if token.Sequence < len(outputs) {
				outputs[token.Sequence] = append(outputs[token.Sequence], token.Token)
			}
		} else {
			fmt.Printf("EOS token reached for sequence %d\n", token.Sequence)
		}
	}
	for err = range errChan {
		if err != nil {
			t.Fatalf("generation error: %v", err)
		}
	}

	for i, out := range outputs {
		fmt.Printf("Sequence %d output: %s\n", i, strings.Join(out, ""))
		if len(out) == 0 {
			t.Errorf("no output generated for sequence %d", i)
		}
	}

	stats := g.GetStatistics()
	fmt.Printf("Statistics: %+v\n", stats)
}

func testGenericConcurrentGeneration(t *testing.T, g Generatable) {
	t.Helper()
	ctx, cancel := context.WithTimeout(context.Background(), 5*time.Minute)
	defer cancel()

	temperature := 0.0
	topP := 0.9
	seed := 42
	opts := &GenerationOptions{
		MaxLength:   2048,
		Temperature: &temperature,
		TopP:        &topP,
		Seed:        &seed,
	}

	prompts := [][]Message{
		inputMessagesFirstGeneration,
		inputMessagesSecondGeneration,
	}

	type result struct {
		tokens []string
		err    error
	}

	var wg sync.WaitGroup
	results := make([]result, len(prompts))

	for i, msgs := range prompts {
		wg.Add(1)
		go func(idx int, messages []Message) {
			defer wg.Done()
			outputChan, errChan, err := g.Generate(ctx, [][]Message{messages}, nil, opts)
			if err != nil {
				results[idx] = result{err: err}
				return
			}

			var tokens []string
			for delta := range outputChan {
				if !delta.EOSReached {
					tokens = append(tokens, delta.Token)
				}
			}
			for err := range errChan {
				if err != nil {
					results[idx] = result{err: err}
					return
				}
			}
			results[idx] = result{tokens: tokens}
		}(i, msgs)
	}

	wg.Wait()

	for i, r := range results {
		if r.err != nil {
			t.Fatalf("request %d failed: %v", i, r.err)
		}
		output := strings.Join(r.tokens, "")
		fmt.Printf("Concurrent request %d output: %s\n", i, output)
		if len(r.tokens) == 0 {
			t.Fatalf("request %d produced no tokens", i)
		}
	}
}

func testGenericContextCancellation(t *testing.T, g Generatable) {
	t.Helper()
	ctx, cancel := context.WithTimeout(context.Background(), 2*time.Second)
	defer cancel()

	temperature := 0.0
	topP := 0.9
	seed := 42
	opts := &GenerationOptions{
		MaxLength:   4096, // long enough that we'll hit the timeout
		Temperature: &temperature,
		TopP:        &topP,
		Seed:        &seed,
	}

	outputChan, _, err := g.Generate(ctx, [][]Message{inputMessagesFirstGeneration}, nil, opts)
	if err != nil {
		t.Fatalf("Generate failed: %v", err)
	}

	var tokens []string
	for delta := range outputChan {
		if !delta.EOSReached {
			tokens = append(tokens, delta.Token)
		}
	}

	fmt.Printf("Cancelled after %d tokens\n", len(tokens))
}

// getLibraryPath returns the path to libonnxruntime-genai from ONNXRUNTIME_GENAI_LIB env var
// or defaults to /usr/lib/libonnxruntime-genai.so.
func getLibraryPath() string {
	if path := os.Getenv("ONNXRUNTIME_GENAI_LIB"); path != "" {
		return path
	}
	return "/usr/lib/libonnxruntime-genai.so"
}

func testGenericGenerationWithTools(t *testing.T, g Generatable) {
	t.Helper()
	// Two minimal Hermes-style tool definitions.
	tools := []string{
		`{
			"type": "function",
			"function": {
				"name": "get_current_time",
				"description": "Returns the current UTC time.",
				"parameters": {
					"type": "object",
					"properties": {},
					"required": []
				}
			}
		}`,
		`{
			"type": "function",
			"function": {
				"name": "get_weather",
				"description": "Returns the current weather for a given city.",
				"parameters": {
					"type": "object",
					"properties": {
						"city": {
							"type": "string",
							"description": "The name of the city."
						}
					},
					"required": ["city"]
				}
			}
		}`,
	}

	messages := []Message{
		{Role: "user", Content: "What time is it right now, and what's the weather like in Paris?"},
	}

	ctx, cancel := context.WithTimeout(context.Background(), 5*time.Minute)
	defer cancel()

	// Lark grammar that constrains both <tool_call> blocks to valid JSON.
	//
	// Requirements:
	//   - <tool_call> and </tool_call> must be "special": false in tokenizer.json so that
	//     llguidance can match them as regular byte sequences. Marking them "special": true
	//     promotes them to control tokens that the guidance system cannot byte-force, causing
	//     "token doesn't satisfy the grammar" errors.
	//   - The grammar expects exactly two tool calls for this query.
	toolGrammar := `start: fun_call fun_call /\n?/
fun_call: "<tool_call>\n" tool_json "\n</tool_call>\n"
tool_json: %json {"anyOf": [` +
		`{"type":"object","required":["name","arguments"],"additionalProperties":false,` +
		`"properties":{"name":{"const":"get_current_time"},"arguments":{"type":"object","properties":{},"additionalProperties":false}}},` +
		`{"type":"object","required":["name","arguments"],"additionalProperties":false,` +
		`"properties":{"name":{"const":"get_weather"},"arguments":{"type":"object","properties":{"city":{"type":"string"}},"required":["city"],"additionalProperties":false}}}` +
		`]}`

	temperature := 0.0
	topP := 0.9
	seed := 42
	options := &GenerationOptions{
		MaxLength:   512,
		BatchSize:   1,
		Temperature: &temperature,
		TopP:        &topP,
		Seed:        &seed,
		Guidance: &Guidance{
			Type:           GuidanceTypeLarkGrammar,
			Data:           toolGrammar,
			EnableFFTokens: true,
		},
	}

	outputChan, errChan, err := g.Generate(ctx, [][]Message{messages}, tools, options)
	if err != nil {
		t.Fatalf("Generate failed: %v", err)
	}

	var tokens []string
	for delta := range outputChan {
		if !delta.EOSReached {
			tokens = append(tokens, delta.Token)
		}
	}
	for genErr := range errChan {
		if genErr != nil {
			t.Fatalf("generation error: %v", genErr)
		}
	}

	output := strings.Join(tokens, "")
	fmt.Printf("Tool-calling output: %s\n", output)

	if len(output) == 0 {
		t.Fatal("expected non-empty output from tool-calling generation")
	}
	checkToolCalls(t, output)
}

func checkToolCalls(t *testing.T, output string) {
	t.Helper()
	calls := parseToolCalls(output)
	if len(calls) < 2 {
		t.Fatalf("expected at least 2 tool calls, got %d: %s", len(calls), output)
	}

	names := make(map[string]bool, len(calls))
	for _, c := range calls {
		names[c.Name] = true
	}
	if !names["get_current_time"] {
		t.Errorf("expected get_current_time tool call, got calls: %v", calls)
	}
	if !names["get_weather"] {
		t.Errorf("expected get_weather tool call, got calls: %v", calls)
	}
	for _, c := range calls {
		if c.Name == "get_weather" {
			city, _ := c.Arguments["city"].(string)
			if !strings.EqualFold(city, "Paris") {
				t.Errorf("expected get_weather city=Paris, got %q", city)
			}
		}
	}
}

type toolCall struct {
	Name      string         `json:"name"`
	Arguments map[string]any `json:"arguments"`
}

// parseToolCalls extracts all <tool_call>...</tool_call> blocks from s and unmarshals
// each as a toolCall. Uses json.Decoder so that a trailing stray `}` (a common
// int4-quantisation artefact) does not cause the block to be skipped — Decode reads
// exactly one JSON value and stops, leaving trailing garbage unread.
func parseToolCalls(s string) []toolCall {
	re := regexp.MustCompile(`(?s)<tool_call>\s*(.+?)\s*</tool_call>`)
	matches := re.FindAllStringSubmatch(s, -1)
	var calls []toolCall
	for _, m := range matches {
		dec := json.NewDecoder(strings.NewReader(m[1]))
		var tc toolCall
		if err := dec.Decode(&tc); err != nil {
			continue
		}
		calls = append(calls, tc)
	}
	return calls
}
