package ortgenai

import (
	"bytes"
	"testing"
)

func TestTensorOwnsBufferAndCopiesData(t *testing.T) {
	SetSharedLibraryPath(getLibraryPath())
	if err := InitializeEnvironment(); err != nil {
		t.Fatalf("failed to initialize environment: %v", err)
	}
	defer func() {
		if err := DestroyEnvironment(); err != nil {
			t.Fatalf("failed to destroy environment: %v", err)
		}
	}()

	input := []byte{1, 2, 3, 4}
	tensor, err := NewTensorFromBuffer(input, []int64{4}, ElementTypeUint8)
	if err != nil {
		t.Fatalf("failed to create tensor: %v", err)
	}
	input[0] = 9
	defer tensor.Destroy()

	shape, err := tensor.Shape()
	if err != nil {
		t.Fatalf("failed to read tensor shape: %v", err)
	}
	if len(shape) != 1 || shape[0] != 4 {
		t.Fatalf("tensor shape = %v, want [4]", shape)
	}
	data, err := tensor.CopyData()
	if err != nil {
		t.Fatalf("failed to copy tensor data: %v", err)
	}
	if !bytes.Equal(data, []byte{1, 2, 3, 4}) {
		t.Fatalf("tensor data = %v, want [1 2 3 4]", data)
	}
	if _, err := NewTensorFromBuffer([]byte{1}, []int64{2}, ElementTypeUint8); err == nil {
		t.Fatal("expected a tensor buffer size mismatch error")
	}
}
