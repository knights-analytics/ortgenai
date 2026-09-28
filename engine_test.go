package ortgenai

import (
	"os"
	"runtime"
	"testing"
)

func TestEngineTurnLifecycle(t *testing.T) {
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

	engine, err := CreateEngine(modelPath)
	if err != nil {
		t.Fatalf("failed to create engine: %v", err)
	}
	defer engine.Destroy()
	capabilities, err := engine.Capabilities()
	if err != nil {
		t.Fatalf("failed to read engine capabilities: %v", err)
	}
	if capabilities.ConfiguredMaxBatchSize == 0 {
		t.Fatal("engine reported a zero configured batch size")
	}

	buffer, err := engine.CreateEventBuffer(4)
	if err != nil {
		t.Fatalf("failed to create event buffer: %v", err)
	}
	defer buffer.Destroy()

	request, err := engine.CreateRequest(nil)
	if err != nil {
		t.Fatalf("failed to create request: %v", err)
	}
	defer request.Destroy()
	stats, err := engine.SpeculativeStats()
	if err != nil {
		t.Fatalf("failed to read engine speculative statistics: %v", err)
	}
	defer stats.Destroy()

	options, err := request.CreateTurnOptions()
	if err != nil {
		t.Fatalf("failed to create turn options: %v", err)
	}
	defer options.Destroy()
	if err := options.SetMaxGeneratedTokens(2); err != nil {
		t.Fatalf("failed to set turn token limit: %v", err)
	}
	if err := options.SetDoSample(false); err != nil {
		t.Fatalf("failed to set greedy sampling: %v", err)
	}

	turnID, err := request.BeginTurn([]int32{1, 2, 3}, options)
	if err != nil {
		t.Fatalf("failed to begin turn: %v", err)
	}
	var terminal bool
	for run := 0; run < 1000 && !terminal; run++ {
		events, err := engine.Run(buffer)
		if err != nil {
			t.Fatalf("engine run failed: %v", err)
		}
		for _, event := range events {
			if event.Request != request {
				t.Errorf("event request = %p, want %p", event.Request, request)
			}
			if event.TurnID != turnID {
				t.Errorf("event turn ID = %d, want %d", event.TurnID, turnID)
			}
			if event.Flags&EventFlagTurnFinished != 0 {
				terminal = true
				if event.FinishReason == FinishReasonNone {
					t.Error("terminal event has no finish reason")
				}
			}
		}
	}
	if !terminal {
		t.Fatal("engine did not report a terminal event")
	}

	cancelledTurnID, err := request.BeginTurn([]int32{1, 2, 3}, options)
	if err != nil {
		t.Fatalf("failed to begin cancellable turn: %v", err)
	}
	cancelled, err := request.CancelTurn(cancelledTurnID)
	if err != nil {
		t.Fatalf("failed to cancel turn: %v", err)
	}
	if !cancelled {
		t.Fatal("expected cancellation to accept the active turn")
	}
	var cancellationReported bool
	for run := 0; run < 10 && !cancellationReported; run++ {
		events, err := engine.Run(buffer)
		if err != nil {
			t.Fatalf("engine run after cancellation failed: %v", err)
		}
		for _, event := range events {
			if event.TurnID == cancelledTurnID && event.Flags&EventFlagTurnFinished != 0 {
				cancellationReported = true
				if event.FinishReason != FinishReasonCancelled {
					t.Errorf("cancelled turn finish reason = %d, want %d", event.FinishReason, FinishReasonCancelled)
				}
			}
		}
	}
	if !cancellationReported {
		t.Fatal("engine did not report the cancelled turn")
	}
}
