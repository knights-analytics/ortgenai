package ortgenai

import (
	"os"
	"testing"
)

func TestSessionGPU(t *testing.T) {
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

	modelPath := "./models/phi3.5gpu" // use the gpu optimized model
	if _, err := os.Stat(modelPath); os.IsNotExist(err) {
		t.Skip("GPU model not found at " + modelPath)
	}

	providers := []string{"cuda"}
	providerOptions := map[string]map[string]string{}
	session, err := CreateSessionWithOptions(modelPath, providers, providerOptions)
	if err != nil {
		t.Fatalf("failed to create session with CUDA: %v", err)
	}
	defer session.Destroy()

	t.Run("Generation", func(t *testing.T) {
		options := &GenerationOptions{
			MaxLength: 2048,
			BatchSize: 2,
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
		// Note: Vision model may require specific GPU providers as well,
		// but for now we follow the existing pattern.
		visionSession, err := CreateSessionWithOptions(visionModelPath, providers, providerOptions)
		if err != nil {
			t.Fatalf("failed to create vision session with CUDA: %v", err)
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

		toolSession, err := CreateSessionWithOptions(toolModelPath, providers, providerOptions)
		if err != nil {
			t.Fatalf("failed to create tool session: %v", err)
		}
		defer toolSession.Destroy()
		testGenericGenerationWithTools(t, toolSession)
	})
}
