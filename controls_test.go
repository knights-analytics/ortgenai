package ortgenai

import "testing"

func TestRuntimeSettingsAndLogCallback(t *testing.T) {
	SetSharedLibraryPath(getLibraryPath())
	if err := InitializeEnvironment(); err != nil {
		t.Fatalf("failed to initialize environment: %v", err)
	}
	defer func() {
		if err := DestroyEnvironment(); err != nil {
			t.Fatalf("failed to destroy environment: %v", err)
		}
	}()

	settings, err := CreateRuntimeSettings()
	if err != nil {
		t.Fatalf("failed to create runtime settings: %v", err)
	}
	settings.Destroy()
	settings.Destroy()

	if err := SetLogCallback(func(string) {}); err != nil {
		t.Fatalf("failed to set log callback: %v", err)
	}
	if err := SetLogCallback(nil); err != nil {
		t.Fatalf("failed to clear log callback: %v", err)
	}
	if err := SetLogString("filename", ""); err != nil {
		t.Fatalf("failed to reset log destination: %v", err)
	}
}
