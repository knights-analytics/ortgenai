package ortgenai

import (
	"os"
	"runtime"
	"testing"
)

func TestGeneratorTokenLifecycle(t *testing.T) {
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
	runtime.LockOSThread()
	defer runtime.UnlockOSThread()

	tokenizer, err := CreateTokenizer(modelPath)
	if err != nil {
		t.Fatalf("failed to create tokenizer: %v", err)
	}
	defer tokenizer.Destroy()
	input, err := tokenizer.Encode("Hello")
	if err != nil {
		t.Fatalf("failed to encode prompt: %v", err)
	}

	generator, err := CreateGenerator(modelPath, 8, 1)
	if err != nil {
		t.Fatalf("failed to create generator: %v", err)
	}
	defer generator.Destroy()
	if err := generator.AppendTokens(input); err != nil {
		t.Fatalf("failed to append prompt tokens: %v", err)
	}
	if err := generator.SnapshotState(); err != nil {
		t.Fatalf("failed to snapshot generator state: %v", err)
	}
	if err := generator.GenerateNextToken(); err != nil {
		t.Fatalf("failed to generate next token: %v", err)
	}
	sequence, err := generator.Sequence(0)
	if err != nil {
		t.Fatalf("failed to copy generated sequence: %v", err)
	}
	if len(sequence) <= len(input) {
		t.Fatalf("generated sequence length = %d, prompt length = %d", len(sequence), len(input))
	}
	if _, err := generator.NextTokens(); err != nil {
		t.Fatalf("failed to copy next-token values: %v", err)
	}
}
