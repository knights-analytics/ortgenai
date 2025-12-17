package ortgenai

import (
	"context"
	"fmt"
	"strings"
	"testing"
	"time"
)

func TestGenerationGPU(t *testing.T) {
	t.Skip() // skip by default in CI
	SetSharedLibraryPath("/usr/lib/libonnxruntime-genai.so")

	if err := InitializeEnvironment(); err != nil {
		t.Fatalf("failed to initialize environment: %v", err)
	}
	defer func() {
		if err := DestroyEnvironment(); err != nil {
			t.Fatalf("failed to destroy environment: %v", err)
		}
	}()

	modelPath := "./_models/phi3.5"

	providers := []string{"cuda"}
	providerOptions := map[string]map[string]string{}
	session, err := CreateGenerativeSessionAdvanced(modelPath, providers, providerOptions)
	if err != nil {
		t.Fatalf("failed to create advanced session: %v", err)
	}

	inputMessagesFirstGeneration := []Message{
		{Role: "system", Content: "You are a helpful assistant."},
		{Role: "user", Content: "What is the capital of France?"},
	}
	inputMessagesSecondGeneration := []Message{
		{Role: "system", Content: "You are a helpful assistant."},
		{Role: "user", Content: "What is the capital of Germany?"},
	}

	ctx, cancel := context.WithTimeout(context.Background(), 5*time.Minute)
	defer cancel()
	options := &GenerationOptions{
		MaxLength: 512,
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

	fmt.Printf("[GPU] First sequence output: %s", strings.Join(firstSequenceOutput, "")+"\n")
	fmt.Printf("[GPU] Second sequence output: %s", strings.Join(secondSequenceOutput, "")+"\n")

	fmt.Println("[GPU] statistics:")
	stats := session.GetStatistics()
	fmt.Printf("Cumulative prefill count: %d\n", stats.cumulativePrefillCount)
	fmt.Printf("Cumulative prefill seconds: %.2f\n", stats.cumulativePrefillSum)
	fmt.Printf("Average prefill seconds: %.2f\n", stats.AvgPrefillSeconds)
	fmt.Printf("Cumulative tokens: %d\n", stats.cumulativeTokens)
	fmt.Printf("Cumulative token duration seconds: %.2f\n", stats.cumulativeTokenDurationSeconds)
	fmt.Printf("Tokens per second: %.2f\n", stats.TokensPerSecond)
}
