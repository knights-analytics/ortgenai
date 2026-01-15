package ortgenai

import (
	"context"
	"encoding/base64"
	"fmt"
	"os"
	"strings"
	"testing"
	"time"
)

// testImagePNG is a minimal valid 1x1 red PNG image (base64 encoded)
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

func TestGeneration(t *testing.T) {
	SetSharedLibraryPath(getLibraryPath())

	if err := InitializeEnvironment(); err != nil {
		t.Fatalf("failed to initialize environment: %v", err)
	}
	defer func() {
		if err := DestroyEnvironment(); err != nil {
			t.Fatalf("failed to destroy environment: %v", err)
		}
	}()

	modelPath := "./_models/phi3.5"

	session, err := CreateGenerativeSession(modelPath)
	if err != nil {
		t.Fatalf("failed to create session: %v", err)
	}

	inputMessagesFirstGeneration := []Message{
		{Role: "system", Content: "You are a helpful assistant."},
		{Role: "user", Content: fmt.Sprintf(`Hello, I have the following two
		jsons that represent two users:

		first: %s

		second: %s

		Please compare them and tell me the main differences between these users.
		`, testJSONs[0], testJSONs[1])},
	}

	inputMessagesSecondGeneration := []Message{
		{Role: "system", Content: "You are a helpful assistant."},
		{Role: "user", Content: "What is the capital of France?"},
	}

	ctx, cancel := context.WithTimeout(context.Background(), 5*time.Minute)
	defer cancel()
	options := &GenerationOptions{
		MaxLength: 2024,
		BatchSize: 2,
	}
	generateChan, errChan, err := session.Generate(ctx, [][]Message{inputMessagesFirstGeneration, inputMessagesSecondGeneration}, options)
	if err != nil {
		t.Fatalf("failed to start generation: %v", err)
	}
	var firstSequenceOutput []string
	var secondSequenceOutput []string

	for token := range generateChan {
		switch token.Sequence {
		case 0:
			firstSequenceOutput = append(firstSequenceOutput, token.Tokens)
		case 1:
			secondSequenceOutput = append(secondSequenceOutput, token.Tokens)
		}
	}
	for err := range errChan {
		if err != nil {
			t.Fatalf("generation error: %v", err)
		}
	}

	fmt.Printf("First sequence output: %s", strings.Join(firstSequenceOutput, "")+"\n")
	fmt.Printf("Second sequence output: %s", strings.Join(secondSequenceOutput, "")+"\n")

	fmt.Println("statistics:")
	stats := session.GetStatistics()
	fmt.Printf("Cumulative prefill count: %d\n", stats.CumulativePrefillCount)
	fmt.Printf("Cumulative prefill seconds: %.2f\n", stats.CumulativePrefillSum)
	fmt.Printf("Average prefill seconds: %.2f\n", stats.AvgPrefillSeconds)
	fmt.Printf("Cumulative tokens: %d\n", stats.CumulativeTokens)
	fmt.Printf("Cumulative token duration seconds: %.2f\n", stats.CumulativeTokenDurationSeconds)
	fmt.Printf("Tokens per second: %.2f\n", stats.TokensPerSecond)
}

// getLibraryPath returns the path to libonnxruntime-genai from ONNXRUNTIME_GENAI_LIB env var
// or defaults to /usr/lib/libonnxruntime-genai.so
func getLibraryPath() string {
	if path := os.Getenv("ONNXRUNTIME_GENAI_LIB"); path != "" {
		return path
	}
	return "/usr/lib/libonnxruntime-genai.so"
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

// TestMultimodalGeneration tests the full multimodal pipeline.
// Requires a vision-language model (e.g., phi-3.5-vision).
func TestMultimodalGeneration(t *testing.T) {
	visionModelPath := "./_models/phi-3.5-vision"
	if _, err := os.Stat(visionModelPath); os.IsNotExist(err) {
		t.Skip("Vision model not found at " + visionModelPath)
	}

	SetSharedLibraryPath(getLibraryPath())

	if err := InitializeEnvironment(); err != nil {
		t.Fatalf("failed to initialize environment: %v", err)
	}
	defer func() {
		if err := DestroyEnvironment(); err != nil {
			t.Fatalf("failed to destroy environment: %v", err)
		}
	}()

	session, err := CreateGenerativeSession(visionModelPath)
	if err != nil {
		t.Fatalf("failed to create session: %v", err)
	}
	defer session.Destroy()

	imageData, err := base64.StdEncoding.DecodeString(testImagePNG)
	if err != nil {
		t.Fatalf("failed to decode test image: %v", err)
	}

	images, err := LoadImageFromBuffer(imageData)
	if err != nil {
		t.Fatalf("LoadImageFromBuffer failed: %v", err)
	}
	defer images.Destroy()

	processor, err := CreateMultiModalProcessor(session.model)
	if err != nil {
		t.Fatalf("CreateMultiModalProcessor failed: %v", err)
	}
	defer processor.Destroy()

	prompt := "<|user|>\n<|image_1|>\nWhat is in this image?<|end|>\n<|assistant|>\n"
	tensors, err := processor.ProcessImages(prompt, images)
	if err != nil {
		t.Fatalf("ProcessImages failed: %v", err)
	}
	defer tensors.Destroy()

	ctx, cancel := context.WithTimeout(context.Background(), 2*time.Minute)
	defer cancel()

	options := &GenerationOptions{
		MaxLength: 4096,
		BatchSize: 1,
	}

	outputChan, errChan, err := session.GenerateWithTensors(ctx, tensors, options)
	if err != nil {
		t.Fatalf("GenerateWithTensors failed: %v", err)
	}

	var output []string
	for token := range outputChan {
		output = append(output, token.Tokens)
	}

	for err := range errChan {
		if err != nil {
			t.Fatalf("generation error: %v", err)
		}
	}

	fmt.Printf("Multimodal output: %s\n", strings.Join(output, ""))

	if len(output) == 0 {
		t.Fatal("no output generated from multimodal model")
	}
}
