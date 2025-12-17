package ortgenai

import (
	"context"
	"fmt"
	"strings"
	"testing"
	"time"
)

var testJSONs = []string{
	`{
	"id": "12345",
	"name": "John Doe",
	"email": "john.doe@example.com",
	"age": 30,
	"address": {
		"street": "123 Main St",
		"city": "Anytown",
		"state": "CA",
		"zip": "12345"
	},
	"phone_numbers": [
		{
			"type": "home",
			"number": "555-1234"
		},
		{
			"type": "work",
			"number": "555-5678"
		}
	],
	"preferences": {
		"contact_method": "email",
		"newsletter_subscribed": true
	},
	"tags": ["customer", "premium", "active"],
	"metadata": {
		"last_login": "2024-01-15T10:30:00Z",
		"account_created": "2020-06-20T14:45:00Z"
	}
}`,
	`{
	"id": "67890",
	"name": "Jane Smith",
	"email": "jane.smith@example.com",
	"age": 25,
	"address": {
		"street": "456 Elm St",
		"city": "Othertown",
		"state": "NY",
		"zip": "67890"
	},
	"phone_numbers": [
		{
			"type": "mobile",
			"number": "555-8765"
		}
	],
	"preferences": {
		"contact_method": "phone",
		"newsletter_subscribed": false
	},
	"tags": ["lead", "new"],
	"metadata": {
		"last_login": "2024-02-20T09:15:00Z",
		"account_created": "2023-03-10T11:20:00Z"
	}
}`,
}

func TestGeneration(t *testing.T) {
	SetSharedLibraryPath("/usr/lib/libonnxruntime-genai.so")

	if err := InitializeEnvironment(); err != nil {
		t.Fatalf("failed to initialize environment: %v", err)
	}
	defer func() {
		if err := DestroyEnvironment(); err != nil {
			t.Fatalf("failed to destroy environment: %v", err)
		}
	}()

	modelPath := "./_models/phi3.5"

	session, err := CreateGenerativeSession(modelPath)
	if err != nil {
		t.Fatalf("failed to create session: %v", err)
	}

	inputMessagesFirstGeneration := []Message{
		{Role: "system", Content: "You are a helpful assistant."},
		{Role: "user", Content: fmt.Sprintf(`Hello, I have the following two
		jsons that represent two users:

		first: %s

		second: %s

		Please compare them and tell me the main differences between these users.
		`, testJSONs[0], testJSONs[1])},
	}

	inputMessagesSecondGeneration := []Message{
		{Role: "system", Content: "You are a helpful assistant."},
		{Role: "user", Content: "What is the capital of France?"},
	}

	ctx, cancel := context.WithTimeout(context.Background(), 5*time.Minute)
	defer cancel()
	options := &GenerationOptions{
		MaxLength: 2024,
		BatchSize: 2,
	}
	generateChan, errChan, err := session.Generate(ctx, [][]Message{inputMessagesFirstGeneration, inputMessagesSecondGeneration}, options)
	if err != nil {
		t.Fatalf("failed to start generation: %v", err)
	}
	var firstSequenceOutput []string
	var secondSequenceOutput []string

	for token := range generateChan {
		switch token.Sequence {
		case 0:
			firstSequenceOutput = append(firstSequenceOutput, token.Tokens)
		case 1:
			secondSequenceOutput = append(secondSequenceOutput, token.Tokens)
		}
	}
	for err := range errChan {
		if err != nil {
			t.Fatalf("generation error: %v", err)
		}
	}

	fmt.Printf("First sequence output: %s", strings.Join(firstSequenceOutput, "")+"\n")
	fmt.Printf("Second sequence output: %s", strings.Join(secondSequenceOutput, "")+"\n")

	fmt.Println("statistics:")
	stats := session.GetStatistics()
	fmt.Printf("Cumulative prefill count: %d\n", stats.cumulativePrefillCount)
	fmt.Printf("Cumulative prefill seconds: %.2f\n", stats.cumulativePrefillSum)
	fmt.Printf("Average prefill seconds: %.2f\n", stats.AvgPrefillSeconds)
	fmt.Printf("Cumulative tokens: %d\n", stats.cumulativeTokens)
	fmt.Printf("Cumulative token duration seconds: %.2f\n", stats.cumulativeTokenDurationSeconds)
	fmt.Printf("Tokens per second: %.2f\n", stats.TokensPerSecond)
}
