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

func TestSession(t *testing.T) {
	SetSharedLibraryPath(getLibraryPath())

	if err := InitializeEnvironment(); err != nil {
		t.Fatalf("failed to initialize environment: %v", err)
	}
	defer func() {
		if err := DestroyEnvironment(); err != nil {
			t.Fatalf("failed to destroy environment: %v", err)
		}
	}()

	modelPath := "./models/phi3.5"
	if _, err := os.Stat(modelPath); os.IsNotExist(err) {
		t.Skip("Model not found at " + modelPath)
	}

	session, err := CreateSession(modelPath)
	if err != nil {
		t.Fatalf("failed to create session: %v", err)
	}
	defer session.Destroy()

	t.Run("Generation", func(t *testing.T) {
		temperature := 0.0
		topP := 0.9
		seed := 42
		options := &GenerationOptions{
			MaxLength:   2048,
			BatchSize:   2,
			Temperature: &temperature,
			TopP:        &topP,
			Seed:        &seed,
		}
		testGenericGeneration(t, session, [][]Message{inputMessagesFirstGeneration, inputMessagesSecondGeneration}, options)
	})

	t.Run("ConcurrentGeneration", func(t *testing.T) {
		testGenericConcurrentGeneration(t, session)
	})

	t.Run("ContextCancellation", func(t *testing.T) {
		testGenericContextCancellation(t, session)
	})

	t.Run("MultimodalGeneration", func(t *testing.T) {
		visionModelPath := "./models/phi3.5vision"
		if _, err := os.Stat(visionModelPath); os.IsNotExist(err) {
			t.Skip("Vision model not found at " + visionModelPath)
		}
		visionSession, err := CreateSession(visionModelPath)
		if err != nil {
			t.Fatalf("failed to create vision session: %v", err)
		}
		defer visionSession.Destroy()
		testGenericMultimodal(t, visionSession)
	})

	t.Run("GenerationWithTools", func(t *testing.T) {
		if os.Getenv("CI") == "true" {
			t.Skip("Skipping tool-calling test in CI as it requires qwen, we run this locally")
		}

		toolModelPath := "./models/qwen3-4B-int4"
		if _, err = os.Stat(toolModelPath); os.IsNotExist(err) {
			t.Skip("Model not found at " + toolModelPath)
		}

		toolSession, err := CreateSession(toolModelPath)
		if err != nil {
			t.Fatalf("failed to create tool session: %v", err)
		}
		defer toolSession.Destroy()
		testGenericGenerationWithTools(t, toolSession)
	})

	t.Run("LoRAAdapter", func(t *testing.T) {
		modelPath := "./models/phi3.5"
		if _, err := os.Stat(modelPath); os.IsNotExist(err) {
			t.Skip("Model not found at " + modelPath)
		}

		session, err := CreateSession(modelPath)
		if err != nil {
			t.Fatalf("failed to create session: %v", err)
		}
		defer session.Destroy()

		adapters, err := session.model.CreateAdapters()
		if err != nil {
			t.Fatalf("failed to create adapters: %v", err)
		}
		defer adapters.Destroy()

		// Attempt to load a non-existent adapter
		err = adapters.LoadAdapter("nonexistent.adapter", "my-adapter")
		if err == nil {
			t.Fatal("expected error loading non-existent adapter, got nil")
		}

		// Try generating with this adapter
		options := &GenerationOptions{
			MaxLength:     10,
			BatchSize:     1,
			Adapters:      adapters,
			ActiveAdapter: "my-adapter",
		}

		ctx, cancel := context.WithTimeout(context.Background(), 1*time.Minute)
		defer cancel()

		_, _, err = session.Generate(ctx, [][]Message{{{Role: "user", Content: "Hello"}}}, nil, options)
		if err == nil {
			t.Error("expected error during generation with non-existent adapter, got nil")
		}
	})
}

func testGenericMultimodal(t *testing.T, s *Session) {
	t.Helper()
	imageData, err := base64.StdEncoding.DecodeString(testImagePNG)
	if err != nil {
		t.Fatalf("failed to decode test image: %v", err)
	}

	images, err := LoadImageFromBuffer(imageData)
	if err != nil {
		t.Fatalf("LoadImageFromBuffer failed: %v", err)
	}
	defer images.Destroy()

	messages := []Message{
		{Role: "system", Content: "You are a helpful assistant."},
		// Include image token in the user content so chat template preserves it
		{Role: "user", Content: "<|image_1|>\nWhat is in this image?"},
	}

	ctx, cancel := context.WithTimeout(context.Background(), 20*time.Minute) // give this one a wide berth
	defer cancel()

	generationOptions := &GenerationOptions{
		MaxLength: 4096,
		BatchSize: 1,
	}

	outputChan, errChan, err := s.GenerateWithImages(ctx, [][]Message{messages}, images, nil, generationOptions)
	if err != nil {
		t.Fatalf("GenerateWithImages failed: %v", err)
	}

	var output []string
	for token := range outputChan {
		output = append(output, token.Token)
	}

	for err = range errChan {
		if err != nil {
			t.Fatalf("generation error: %v", err)
		}
	}

	fmt.Printf("Multimodal output: %s\n", strings.Join(output, ""))

	if len(output) == 0 {
		t.Fatal("no output generated from multimodal model")
	}
}
