package ortgenai

import (
	"context"
	"fmt"
	"os"
	"strings"
	"sync"
	"testing"
	"time"
)

func TestEngineSingleRequest(t *testing.T) {
	SetSharedLibraryPath(getLibraryPath())

	if err := InitializeEnvironment(); err != nil {
		t.Fatalf("failed to initialize environment: %v", err)
	}
	defer func() {
		if err := DestroyEnvironment(); err != nil {
			t.Fatalf("failed to destroy environment: %v", err)
		}
	}()

	if !IsEngineApiAvailable() {
		t.Skip("Engine API not available in this ORT GenAI version")
	}

	modelPath := "./models/phi3.5"
	if _, err := os.Stat(modelPath); os.IsNotExist(err) {
		t.Skip("Model not found at " + modelPath)
	}

	engine, err := CreateEngine(modelPath)
	if err != nil {
		t.Fatalf("CreateEngine failed: %v", err)
	}
	defer engine.Destroy()

	ctx, cancel := context.WithTimeout(context.Background(), 5*time.Minute)
	defer cancel()

	temperature := 0.0
	topP := 0.9
	seed := 42
	opts := &GenerationOptions{
		MaxLength:   512,
		Temperature: &temperature,
		TopP:        &topP,
		Seed:        &seed,
	}

	outputChan, errChan, err := engine.Submit(ctx, inputMessagesSecondGeneration, opts)
	if err != nil {
		t.Fatalf("Submit failed: %v", err)
	}

	var tokens []string
	for delta := range outputChan {
		if delta.EOSReached {
			fmt.Println("EOS reached")
		} else {
			tokens = append(tokens, delta.Token)
		}
	}

	for err := range errChan {
		if err != nil {
			t.Fatalf("generation error: %v", err)
		}
	}

	output := strings.Join(tokens, "")
	fmt.Printf("Engine single request output: %s\n", output)

	if len(tokens) == 0 {
		t.Fatal("no tokens generated")
	}
}

func TestEngineConcurrentRequests(t *testing.T) {
	SetSharedLibraryPath(getLibraryPath())

	if err := InitializeEnvironment(); err != nil {
		t.Fatalf("failed to initialize environment: %v", err)
	}
	defer func() {
		if err := DestroyEnvironment(); err != nil {
			t.Fatalf("failed to destroy environment: %v", err)
		}
	}()

	if !IsEngineApiAvailable() {
		t.Skip("Engine API not available in this ORT GenAI version")
	}

	modelPath := "./models/phi3.5"
	if _, err := os.Stat(modelPath); os.IsNotExist(err) {
		t.Skip("Model not found at " + modelPath)
	}

	engine, err := CreateEngine(modelPath)
	if err != nil {
		t.Fatalf("CreateEngine failed: %v", err)
	}
	defer engine.Destroy()

	ctx, cancel := context.WithTimeout(context.Background(), 5*time.Minute)
	defer cancel()

	temperature := 0.0
	topP := 0.9
	seed := 42
	opts := &GenerationOptions{
		MaxLength:   256,
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
			outputChan, errChan, err := engine.Submit(ctx, messages, opts)
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
		fmt.Printf("Engine concurrent request %d output: %s\n", i, output)
		if len(r.tokens) == 0 {
			t.Fatalf("request %d produced no tokens", i)
		}
	}
}

func TestEngineContextCancellation(t *testing.T) {
	SetSharedLibraryPath(getLibraryPath())

	if err := InitializeEnvironment(); err != nil {
		t.Fatalf("failed to initialize environment: %v", err)
	}
	defer func() {
		if err := DestroyEnvironment(); err != nil {
			t.Fatalf("failed to destroy environment: %v", err)
		}
	}()

	if !IsEngineApiAvailable() {
		t.Skip("Engine API not available in this ORT GenAI version")
	}

	modelPath := "./models/phi3.5"
	if _, err := os.Stat(modelPath); os.IsNotExist(err) {
		t.Skip("Model not found at " + modelPath)
	}

	engine, err := CreateEngine(modelPath)
	if err != nil {
		t.Fatalf("CreateEngine failed: %v", err)
	}
	defer engine.Destroy()

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

	outputChan, errChan, err := engine.Submit(ctx, inputMessagesFirstGeneration, opts)
	if err != nil {
		t.Fatalf("Submit failed: %v", err)
	}

	var tokens []string
	for delta := range outputChan {
		if !delta.EOSReached {
			tokens = append(tokens, delta.Token)
		}
	}

	// Drain error channel.
	for range errChan {
	}

	fmt.Printf("Engine cancelled after %d tokens\n", len(tokens))
	// The test passes if we get here without hanging — cancellation worked.
}
