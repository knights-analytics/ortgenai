package ortgenai

import (
	"os"
	"path/filepath"
	"testing"
)

func TestResolveMTPModelPath(t *testing.T) {
	for _, tc := range []struct {
		name      string
		dirs      []string
		files     []string
		wantRoot  bool
		wantError bool
	}{
		{name: "subfolder", dirs: []string{"MTP"}},
		{name: "root", files: []string{"mtp.onnx", "mtp.onnx.data"}, wantRoot: true},
		{name: "subfolder takes precedence", dirs: []string{"MTP"}, files: []string{"mtp.onnx", "mtp.onnx.data"}},
		{name: "missing model", wantError: true},
		{name: "missing external data", files: []string{"mtp.onnx"}, wantError: true},
		{name: "missing graph", files: []string{"mtp.onnx.data"}, wantError: true},
		{name: "graph is directory", dirs: []string{"mtp.onnx"}, files: []string{"mtp.onnx.data"}, wantError: true},
		{name: "data is directory", dirs: []string{"mtp.onnx.data"}, files: []string{"mtp.onnx"}, wantError: true},
		{name: "subfolder is file", files: []string{"MTP", "mtp.onnx", "mtp.onnx.data"}, wantError: true},
	} {
		t.Run(tc.name, func(t *testing.T) {
			root := t.TempDir()
			for _, dir := range tc.dirs {
				if err := os.Mkdir(filepath.Join(root, dir), 0o755); err != nil {
					t.Fatal(err)
				}
			}
			for _, file := range tc.files {
				if err := os.WriteFile(filepath.Join(root, file), nil, 0o644); err != nil {
					t.Fatal(err)
				}
			}
			got, err := resolveMTPModelPath(root)
			if tc.wantError {
				if err == nil {
					t.Fatalf("expected an error, got path %q", got)
				}
				return
			}
			if err != nil {
				t.Fatal(err)
			}
			want := filepath.Join(root, "MTP")
			if tc.wantRoot {
				want = root
			}
			if got != want {
				t.Fatalf("got %q, want %q", got, want)
			}
		})
	}
}
