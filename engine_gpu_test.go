package ortgenai

import (
	"os"
	"testing"
)

func TestEngineGPU(t *testing.T) {
	if os.Getenv("CI") == "true" {
		t.Skip("skip by default in CI")
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

	modelPath := "./models/phi3.5gpu" // use the gpu-optimized model
	if _, err := os.Stat(modelPath); os.IsNotExist(err) {
		t.Skip("GPU model not found at " + modelPath)
	}

	providers := []string{"cuda"}
	providerOptions := map[string]map[string]string{}
	engine, err := CreateEngineWithOptions(modelPath, providers, providerOptions)
	if err != nil {
		t.Fatalf("failed to create engine with CUDA: %v", err)
	}
	defer engine.Destroy()

	t.Run("Generation", func(t *testing.T) {
		options := &GenerationOptions{
			MaxLength: 2048,
			BatchSize: 1,
		}
		testGenericGeneration(t, engine, [][]Message{inputMessagesFirstGeneration, inputMessagesSecondGeneration}, options)
	})

	t.Run("ConcurrentGeneration", func(t *testing.T) {
		testGenericConcurrentGeneration(t, engine)
	})

	t.Run("ContextCancellation", func(t *testing.T) {
		testGenericContextCancellation(t, engine)
	})

	t.Run("GenerationWithTools", func(t *testing.T) {
		if os.Getenv("CI") == "true" {
			t.Skip("Skipping tool-calling test in CI as it requires qwen, we run this locally")
		}

		toolModelPath := "./models/qwen3-4B-int4"
		if _, err := os.Stat(toolModelPath); os.IsNotExist(err) {
			t.Skip("Model not found at " + toolModelPath)
		}

		toolEngine, err := CreateEngineWithOptions(toolModelPath, providers, providerOptions)
		if err != nil {
			t.Fatalf("failed to create tool engine: %v", err)
		}
		defer toolEngine.Destroy()
		testGenericGenerationWithTools(t, toolEngine)
	})
}
