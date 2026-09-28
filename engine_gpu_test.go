package ortgenai

import (
	"os"
	"runtime"
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

	modelPath := "./models/phi3.5gpu"
	if _, err := os.Stat(modelPath); os.IsNotExist(err) {
		t.Skip("GPU model not found at " + modelPath)
	}

	runtime.LockOSThread()
	defer runtime.UnlockOSThread()
	engine, err := CreateEngine(modelPath)
	if err != nil {
		t.Fatalf("failed to create GPU engine: %v", err)
	}
	defer engine.Destroy()
	request, err := engine.CreateRequest(nil)
	if err != nil {
		t.Fatalf("failed to create GPU engine request: %v", err)
	}
	defer request.Destroy()
}
