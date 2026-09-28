package ortgenai

import (
	"os"
	"testing"
)

func TestTokenizerEncodeDecode(t *testing.T) {
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
	tokenizer, err := CreateTokenizer(modelPath)
	if err != nil {
		t.Fatalf("failed to create tokenizer: %v", err)
	}
	defer tokenizer.Destroy()

	ids, err := tokenizer.Encode("Hello")
	if err != nil {
		t.Fatalf("failed to encode text: %v", err)
	}
	if len(ids) == 0 {
		t.Fatal("tokenizer returned no token IDs")
	}
	decoded, err := tokenizer.Decode(ids)
	if err != nil {
		t.Fatalf("failed to decode token IDs: %v", err)
	}
	if decoded == "" {
		t.Fatal("tokenizer returned empty decoded text")
	}

	batch, err := tokenizer.EncodeBatch([]string{"Hello", "World"})
	if err != nil {
		t.Fatalf("failed to batch encode text: %v", err)
	}
	defer batch.Destroy()
	shape, err := batch.Shape()
	if err != nil {
		t.Fatalf("failed to read batch tensor shape: %v", err)
	}
	if len(shape) == 0 || shape[0] != 2 {
		t.Fatalf("batch tensor shape = %v, want leading dimension 2", shape)
	}
	if _, err := batch.CopyData(); err != nil {
		t.Fatalf("failed to copy batch tensor data: %v", err)
	}
	decodedBatch, err := tokenizer.DecodeBatch(batch)
	if err != nil {
		t.Fatalf("failed to batch decode token IDs: %v", err)
	}
	if len(decodedBatch) != 2 || decodedBatch[0] == "" || decodedBatch[1] == "" {
		t.Fatalf("batch decode returned %v, want two non-empty strings", decodedBatch)
	}
}
