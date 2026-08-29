//go:build unit

package service

import (
	"bytes"
	"context"
	"io"
	"net/http"
	"net/http/httptest"
	"strings"
	"sync"
	"testing"
	"time"

	"github.com/Wei-Shaw/sub2api/internal/pkg/openai_compat"
	"github.com/gin-gonic/gin"
	"github.com/stretchr/testify/require"
	"github.com/tidwall/gjson"
)

func TestForwardResponses_ForceChatCompletionsRoutesNonStreamingToChatCompletions(t *testing.T) {
	gin.SetMode(gin.TestMode)

	body := []byte(`{"model":"gpt-5.4","input":"hello","stream":false}`)
	rec := httptest.NewRecorder()
	c, _ := gin.CreateTestContext(rec)
	c.Request = httptest.NewRequest(http.MethodPost, "/v1/responses", bytes.NewReader(body))
	c.Request.Header.Set("Content-Type", "application/json")

	upstream := &httpUpstreamRecorder{resp: &http.Response{
		StatusCode: http.StatusOK,
		Header:     http.Header{"Content-Type": []string{"application/json"}, "x-request-id": []string{"rid_resp_chat_json"}},
		Body: io.NopCloser(strings.NewReader(
			`{"id":"chatcmpl_json","object":"chat.completion","model":"gpt-5.4","choices":[{"index":0,"message":{"role":"assistant","content":"ok"},"finish_reason":"stop"}],"usage":{"prompt_tokens":3,"completion_tokens":2,"total_tokens":5,"prompt_tokens_details":{"cached_tokens":1}}}`,
		)),
	}}
	svc := &OpenAIGatewayService{
		cfg:          rawChatCompletionsTestConfig(),
		httpUpstream: upstream,
	}
	SetActualOpenAIUpstreamEndpoint(c, "/v1/responses")

	result, err := svc.Forward(context.Background(), c, forceChatResponsesFallbackAccount(), body)
	require.NoError(t, err)
	require.NotNil(t, result)
	require.Equal(t, "http://upstream.example/v1/chat/completions", upstream.lastReq.URL.String())
	require.Equal(t, "/v1/chat/completions", GetActualOpenAIUpstreamEndpoint(c))
	require.Equal(t, HTTPUpstreamProfileOpenAI, HTTPUpstreamProfileFromContext(upstream.lastReq.Context()))
	require.Equal(t, "hello", gjson.GetBytes(upstream.lastBody, "messages.0.content").String())
	require.False(t, gjson.GetBytes(upstream.lastBody, "input").Exists())
	require.Equal(t, "response", gjson.Get(rec.Body.String(), "object").String())
	require.Equal(t, "ok", gjson.Get(rec.Body.String(), "output.0.content.0.text").String())
	require.Equal(t, 3, result.Usage.InputTokens)
	require.Equal(t, 2, result.Usage.OutputTokens)
	require.Equal(t, 1, result.Usage.CacheReadInputTokens)
	require.False(t, result.Stream)
}

// Scenario: 第三方无推理模型不收到兼容档位。
func TestForwardResponses_ForceChatCompletionsOmitsNoneReasoningEffort(t *testing.T) {
	gin.SetMode(gin.TestMode)

	body := []byte(`{"model":"company-coding-model","input":"hello","reasoning":{"effort":"none"},"stream":false}`)
	rec := httptest.NewRecorder()
	c, _ := gin.CreateTestContext(rec)
	c.Request = httptest.NewRequest(http.MethodPost, "/v1/responses", bytes.NewReader(body))
	c.Request.Header.Set("Content-Type", "application/json")

	upstream := &httpUpstreamRecorder{resp: &http.Response{
		StatusCode: http.StatusOK,
		Header:     http.Header{"Content-Type": []string{"application/json"}},
		Body: io.NopCloser(strings.NewReader(
			`{"id":"chatcmpl_none","object":"chat.completion","model":"company-coding-model","choices":[{"index":0,"message":{"role":"assistant","content":"ok"},"finish_reason":"stop"}],"usage":{"prompt_tokens":1,"completion_tokens":1,"total_tokens":2}}`,
		)),
	}}
	svc := &OpenAIGatewayService{
		cfg:          rawChatCompletionsTestConfig(),
		httpUpstream: upstream,
	}

	result, err := svc.Forward(context.Background(), c, forceChatResponsesFallbackAccount(), body)
	require.NoError(t, err)
	require.NotNil(t, result)
	require.Equal(t, "company-coding-model", gjson.GetBytes(upstream.lastBody, "model").String())
	require.False(t, gjson.GetBytes(upstream.lastBody, "reasoning_effort").Exists())
	require.Nil(t, result.ReasoningEffort)
}

func TestForwardResponses_PassthroughFlagWithUnsupportedResponsesUsesAccountMapping(t *testing.T) {
	gin.SetMode(gin.TestMode)

	for _, path := range []string{"/v1/responses", "/v1/responses/compact"} {
		path := path
		t.Run(path, func(t *testing.T) {
			body := []byte(`{"model":"gpt-5.4-channel","input":"hello","stream":false}`)
			rec := httptest.NewRecorder()
			c, _ := gin.CreateTestContext(rec)
			c.Request = httptest.NewRequest(http.MethodPost, path, bytes.NewReader(body))
			c.Request.Header.Set("Content-Type", "application/json")

			upstream := &httpUpstreamRecorder{resp: &http.Response{
				StatusCode: http.StatusOK,
				Header:     http.Header{"Content-Type": []string{"application/json"}},
				Body: io.NopCloser(strings.NewReader(
					`{"id":"chatcmpl_mapping","object":"chat.completion","model":"gpt-5.4-account","choices":[{"index":0,"message":{"role":"assistant","content":"ok"},"finish_reason":"stop"}],"usage":{"prompt_tokens":1,"completion_tokens":1,"total_tokens":2}}`,
				)),
			}}
			svc := &OpenAIGatewayService{
				cfg:          rawChatCompletionsTestConfig(),
				httpUpstream: upstream,
			}
			account := rawChatCompletionsTestAccount()
			account.Credentials["model_mapping"] = map[string]any{
				"gpt-5.4-channel": "gpt-5.4-account",
			}
			account.Credentials["compact_model_mapping"] = map[string]any{
				"gpt-5.4-account": "gpt-5.4-compact",
			}
			account.Extra = map[string]any{
				"openai_passthrough":                     true,
				openai_compat.ExtraKeyResponsesSupported: false,
			}

			result, err := svc.Forward(context.Background(), c, account, body)
			require.NoError(t, err)
			require.NotNil(t, result)
			require.Equal(t, "http://upstream.example/v1/chat/completions", upstream.lastReq.URL.String())
			require.Equal(t, "gpt-5.4-account", gjson.GetBytes(upstream.lastBody, "model").String())
		})
	}
}

func TestForwardResponses_ForceChatCompletionsRoutesStreamingToChatCompletions(t *testing.T) {
	gin.SetMode(gin.TestMode)

	body := []byte(`{"model":"gpt-5.4","input":"hello","stream":true}`)
	rec := httptest.NewRecorder()
	c, _ := gin.CreateTestContext(rec)
	c.Request = httptest.NewRequest(http.MethodPost, "/v1/responses", bytes.NewReader(body))
	c.Request.Header.Set("Content-Type", "application/json")

	upstreamBody := strings.Join([]string{
		`data: {"id":"chatcmpl_stream","object":"chat.completion.chunk","model":"gpt-5.4","choices":[{"index":0,"delta":{"role":"assistant"},"finish_reason":null}]}`,
		"",
		`data: {"id":"chatcmpl_stream","object":"chat.completion.chunk","model":"gpt-5.4","choices":[{"index":0,"delta":{"content":"he"},"finish_reason":null}]}`,
		"",
		`data: {"id":"chatcmpl_stream","object":"chat.completion.chunk","model":"gpt-5.4","choices":[{"index":0,"delta":{"content":"llo"},"finish_reason":null}]}`,
		"",
		`data: {"id":"chatcmpl_stream","object":"chat.completion.chunk","model":"gpt-5.4","choices":[{"index":0,"delta":{},"finish_reason":"stop"}]}`,
		"",
		`data: {"id":"chatcmpl_stream","object":"chat.completion.chunk","model":"gpt-5.4","choices":[],"usage":{"prompt_tokens":4,"completion_tokens":3,"total_tokens":7}}`,
		"",
		"data: [DONE]",
		"",
	}, "\n")
	upstream := &httpUpstreamRecorder{resp: &http.Response{
		StatusCode: http.StatusOK,
		Header:     http.Header{"Content-Type": []string{"text/event-stream"}, "x-request-id": []string{"rid_resp_chat_stream"}},
		Body:       io.NopCloser(strings.NewReader(upstreamBody)),
	}}
	svc := &OpenAIGatewayService{
		cfg:          rawChatCompletionsTestConfig(),
		httpUpstream: upstream,
	}

	result, err := svc.Forward(context.Background(), c, forceChatResponsesFallbackAccount(), body)
	require.NoError(t, err)
	require.NotNil(t, result)
	require.Equal(t, "http://upstream.example/v1/chat/completions", upstream.lastReq.URL.String())
	require.True(t, gjson.GetBytes(upstream.lastBody, "stream_options.include_usage").Bool())
	require.Contains(t, rec.Body.String(), "event: response.output_text.delta")
	require.Contains(t, rec.Body.String(), `"delta":"he"`)
	require.Contains(t, rec.Body.String(), "event: response.completed")
	require.Contains(t, rec.Body.String(), `"input_tokens":4`)
	require.Contains(t, rec.Body.String(), "data: [DONE]")
	require.Equal(t, 4, result.Usage.InputTokens)
	require.Equal(t, 3, result.Usage.OutputTokens)
	require.True(t, result.Stream)
	require.NotNil(t, result.FirstTokenMs)
}

func TestForwardResponses_DeepSeekAgentHarnessCompactsReasoningAndCachesFullText(t *testing.T) {
	gin.SetMode(gin.TestMode)

	body := []byte(`{
		"model":"deepseek-v4-flash",
		"instructions":"Keep the repository clean.",
		"input":"inspect the workspace",
		"tools":[{"type":"function","name":"exec_command","description":"Run a command","parameters":{"type":"object","properties":{"cmd":{"type":"string"}},"required":["cmd"]}}],
		"reasoning":{"effort":"low"},
		"stream":true
	}`)
	rec := httptest.NewRecorder()
	c, _ := gin.CreateTestContext(rec)
	c.Request = httptest.NewRequest(http.MethodPost, "/v1/responses", bytes.NewReader(body))
	c.Request.Header.Set("Content-Type", "application/json")

	upstreamBody := strings.Join([]string{
		`data: {"id":"chatcmpl_agent","object":"chat.completion.chunk","model":"deepseek-v4-flash","choices":[{"index":0,"delta":{"reasoning_content":"long private reasoning that must stay server side"},"finish_reason":null}]}`,
		"",
		`data: {"id":"chatcmpl_agent","object":"chat.completion.chunk","model":"deepseek-v4-flash","choices":[{"index":0,"delta":{"tool_calls":[{"index":0,"id":"call_agent","type":"function","function":{"name":"exec_command","arguments":"{\"cmd\":\"pwd\"}"}}]},"finish_reason":"tool_calls"}]}`,
		"",
		`data: {"id":"chatcmpl_agent","object":"chat.completion.chunk","model":"deepseek-v4-flash","choices":[],"usage":{"prompt_tokens":20,"completion_tokens":8,"total_tokens":28}}`,
		"",
		"data: [DONE]",
		"",
	}, "\n")
	upstream := &httpUpstreamRecorder{resp: &http.Response{
		StatusCode: http.StatusOK,
		Header:     http.Header{"Content-Type": []string{"text/event-stream"}},
		Body:       io.NopCloser(strings.NewReader(upstreamBody)),
	}}
	cache := &reasoningRecordingCache{}
	svc := &OpenAIGatewayService{
		cfg:          rawChatCompletionsTestConfig(),
		httpUpstream: upstream,
		cache:        cache,
	}

	result, err := svc.Forward(context.Background(), c, forceChatResponsesFallbackAccount(), body)
	require.NoError(t, err)
	require.NotNil(t, result)
	require.True(t, result.Stream)

	systemPrompt := gjson.GetBytes(upstream.lastBody, "messages.0.content").String()
	require.Contains(t, systemPrompt, "Keep the repository clean.")
	require.Contains(t, systemPrompt, "external agent harness")
	require.Contains(t, systemPrompt, "end this turn immediately")
	require.Equal(t, "user", gjson.GetBytes(upstream.lastBody, "messages.1.role").String())
	require.Equal(t, "exec_command", gjson.GetBytes(upstream.lastBody, "tools.0.function.name").String())
	require.Equal(t, "max", gjson.GetBytes(upstream.lastBody, "reasoning_effort").String())
	require.NotNil(t, result.ReasoningEffort)
	require.Equal(t, "max", *result.ReasoningEffort)

	clientBody := rec.Body.String()
	require.Contains(t, clientBody, "Selected the next tool action.")
	require.NotContains(t, clientBody, "long private reasoning")
	require.Contains(t, clientBody, "response.function_call_arguments.done")
	require.Contains(t, clientBody, "data: [DONE]")

	sets := cache.snapshotSets()
	require.Len(t, sets, 1)
	for _, reasoning := range sets {
		require.Equal(t, "long private reasoning that must stay server side", reasoning)
	}
}

func TestForwardResponses_ChatFallbackRejectsInvalidToolArgumentsAtOutputLimit(t *testing.T) {
	gin.SetMode(gin.TestMode)

	body := []byte(`{"model":"deepseek-v4-flash","input":"run the command","stream":true}`)
	rec := httptest.NewRecorder()
	c, _ := gin.CreateTestContext(rec)
	c.Request = httptest.NewRequest(http.MethodPost, "/v1/responses", bytes.NewReader(body))
	c.Request.Header.Set("Content-Type", "application/json")

	upstreamBody := strings.Join([]string{
		`data: {"id":"chatcmpl_length_tool","object":"chat.completion.chunk","model":"deepseek-v4-flash","choices":[{"index":0,"delta":{"tool_calls":[{"index":0,"id":"call_length","type":"function","function":{"name":"exec_command","arguments":"{\"cmd\":\"ssh root@HOST"}}]},"finish_reason":null}]}`,
		"",
		`data: {"id":"chatcmpl_length_tool","object":"chat.completion.chunk","model":"deepseek-v4-flash","choices":[{"index":0,"delta":{},"finish_reason":"length"}],"usage":{"prompt_tokens":4,"completion_tokens":6492,"total_tokens":6496}}`,
		"",
		"data: [DONE]",
		"",
	}, "\n")
	upstream := &httpUpstreamRecorder{resp: &http.Response{
		StatusCode: http.StatusOK,
		Header:     http.Header{"Content-Type": []string{"text/event-stream"}, "x-request-id": []string{"rid_length_tool"}},
		Body:       io.NopCloser(strings.NewReader(upstreamBody)),
	}}
	svc := &OpenAIGatewayService{
		cfg:          rawChatCompletionsTestConfig(),
		httpUpstream: upstream,
	}

	result, err := svc.Forward(context.Background(), c, forceChatResponsesFallbackAccount(), body)
	require.ErrorContains(t, err, "invalid JSON")
	require.NotNil(t, result)
	require.Equal(t, 4, result.Usage.InputTokens)
	require.Equal(t, 6492, result.Usage.OutputTokens)
	require.NotContains(t, rec.Body.String(), "response.function_call_arguments.done")
	require.NotContains(t, rec.Body.String(), "response.output_item.done")
	require.NotContains(t, rec.Body.String(), "data: [DONE]")
}

func TestForwardResponses_DeepSeekReasoningOnlyStreamProducesVisibleText(t *testing.T) {
	gin.SetMode(gin.TestMode)

	body := []byte(`{"model":"deepseek-reasoner","input":"hello","stream":true}`)
	rec := httptest.NewRecorder()
	c, _ := gin.CreateTestContext(rec)
	c.Request = httptest.NewRequest(http.MethodPost, "/v1/responses", bytes.NewReader(body))
	c.Request.Header.Set("Content-Type", "application/json")

	upstreamBody := strings.Join([]string{
		`data: {"id":"chatcmpl_reasoning","object":"chat.completion.chunk","model":"deepseek-reasoner","choices":[{"index":0,"delta":{"role":"assistant","content":null,"reasoning_content":""},"finish_reason":null}]}`,
		"",
		`data: {"id":"chatcmpl_reasoning","object":"chat.completion.chunk","model":"deepseek-reasoner","choices":[{"index":0,"delta":{"reasoning_content":"visible fallback"},"finish_reason":null}]}`,
		"",
		`data: {"id":"chatcmpl_reasoning","object":"chat.completion.chunk","model":"deepseek-reasoner","choices":[{"index":0,"delta":{"content":""},"finish_reason":"length"}],"usage":{"prompt_tokens":4,"completion_tokens":3,"total_tokens":7}}`,
		"",
		"data: [DONE]",
		"",
	}, "\n")
	upstream := &httpUpstreamRecorder{resp: &http.Response{
		StatusCode: http.StatusOK,
		Header:     http.Header{"Content-Type": []string{"text/event-stream"}, "x-request-id": []string{"rid_deepseek_reasoning_responses_stream"}},
		Body:       io.NopCloser(strings.NewReader(upstreamBody)),
	}}
	svc := &OpenAIGatewayService{
		cfg:          rawChatCompletionsTestConfig(),
		httpUpstream: upstream,
	}

	result, err := svc.Forward(context.Background(), c, forceChatResponsesFallbackAccount(), body)
	require.NoError(t, err)
	require.NotNil(t, result)
	require.True(t, result.Stream)
	require.Contains(t, rec.Body.String(), "event: response.output_text.delta")
	require.Contains(t, rec.Body.String(), `"delta":"visible fallback"`)
	require.Contains(t, rec.Body.String(), `"status":"incomplete"`)
	require.Contains(t, rec.Body.String(), "data: [DONE]")
}

func TestForwardResponses_AutoSupportedAccountStillUsesResponsesEndpoint(t *testing.T) {
	gin.SetMode(gin.TestMode)

	body := []byte(`{"model":"gpt-5.4","input":"hello","stream":false}`)
	rec := httptest.NewRecorder()
	c, _ := gin.CreateTestContext(rec)
	c.Request = httptest.NewRequest(http.MethodPost, "/v1/responses", bytes.NewReader(body))
	c.Request.Header.Set("Content-Type", "application/json")

	upstream := &httpUpstreamRecorder{resp: &http.Response{
		StatusCode: http.StatusOK,
		Header:     http.Header{"Content-Type": []string{"application/json"}, "x-request-id": []string{"rid_resp_native"}},
		Body: io.NopCloser(strings.NewReader(
			`{"id":"resp_native","object":"response","model":"gpt-5.4","status":"completed","output":[{"type":"message","role":"assistant","content":[{"type":"output_text","text":"ok"}],"status":"completed"}],"usage":{"input_tokens":5,"output_tokens":2,"total_tokens":7}}`,
		)),
	}}
	svc := &OpenAIGatewayService{
		cfg:          rawChatCompletionsTestConfig(),
		httpUpstream: upstream,
	}
	account := rawChatCompletionsTestAccount()
	account.Extra = map[string]any{
		openai_compat.ExtraKeyResponsesMode:      string(openai_compat.ResponsesSupportModeAuto),
		openai_compat.ExtraKeyResponsesSupported: true,
	}

	result, err := svc.Forward(context.Background(), c, account, body)
	require.NoError(t, err)
	require.NotNil(t, result)
	require.Equal(t, "http://upstream.example/v1/responses", upstream.lastReq.URL.String())
	require.True(t, gjson.GetBytes(upstream.lastBody, "input").Exists())
	require.False(t, gjson.GetBytes(upstream.lastBody, "messages").Exists())
	require.Equal(t, "ok", gjson.Get(rec.Body.String(), "output.0.content.0.text").String())
}

func forceChatResponsesFallbackAccount() *Account {
	account := rawChatCompletionsTestAccount()
	account.Extra = map[string]any{
		openai_compat.ExtraKeyResponsesMode: string(openai_compat.ResponsesSupportModeForceChatCompletions),
	}
	return account
}

// reasoningRecordingCache 记录 reasoning 缓存写入、并按需响应回查。
type reasoningRecordingCache struct {
	stubGatewayCache
	mu             sync.Mutex
	sets           map[string]string
	getResp        map[string]string
	responseStates map[string][]byte
}

func (c *reasoningRecordingCache) SetReasoningContent(_ context.Context, itemID string, content string, _ time.Duration) error {
	c.mu.Lock()
	defer c.mu.Unlock()
	if c.sets == nil {
		c.sets = make(map[string]string)
	}
	c.sets[itemID] = content
	return nil
}

func (c *reasoningRecordingCache) GetReasoningContent(_ context.Context, itemID string) (string, error) {
	if v, ok := c.getResp[itemID]; ok {
		return v, nil
	}
	return "", ErrReasoningContentNotFound
}

func (c *reasoningRecordingCache) snapshotSets() map[string]string {
	c.mu.Lock()
	defer c.mu.Unlock()
	out := make(map[string]string, len(c.sets))
	for k, v := range c.sets {
		out[k] = v
	}
	return out
}

func (c *reasoningRecordingCache) SetResponsesChatFallbackState(_ context.Context, _ int64, responseID string, payload []byte, _ time.Duration) error {
	c.mu.Lock()
	defer c.mu.Unlock()
	if c.responseStates == nil {
		c.responseStates = make(map[string][]byte)
	}
	c.responseStates[responseID] = append([]byte(nil), payload...)
	return nil
}

func (c *reasoningRecordingCache) GetResponsesChatFallbackState(_ context.Context, _ int64, responseID string) ([]byte, error) {
	c.mu.Lock()
	defer c.mu.Unlock()
	return append([]byte(nil), c.responseStates[responseID]...), nil
}

func TestForwardResponses_DeepSeekPreviousResponseRestoresToolTurn(t *testing.T) {
	gin.SetMode(gin.TestMode)

	firstUpstreamBody := strings.Join([]string{
		`data: {"id":"chatcmpl_agent_chain","object":"chat.completion.chunk","model":"deepseek-v4-flash","choices":[{"index":0,"delta":{"reasoning_content":"full private reasoning"},"finish_reason":null}]}`,
		"",
		`data: {"id":"chatcmpl_agent_chain","object":"chat.completion.chunk","model":"deepseek-v4-flash","choices":[{"index":0,"delta":{"content":"I need the intermediate result first."},"finish_reason":null}]}`,
		"",
		`data: {"id":"chatcmpl_agent_chain","object":"chat.completion.chunk","model":"deepseek-v4-flash","choices":[{"index":0,"delta":{"tool_calls":[{"index":0,"id":"call_chain","type":"function","function":{"name":"exec_command","arguments":"{\"cmd\":\"pwd\"}"}}]},"finish_reason":"tool_calls"}]}`,
		"",
		`data: {"id":"chatcmpl_agent_chain","object":"chat.completion.chunk","model":"deepseek-v4-flash","choices":[],"usage":{"prompt_tokens":20,"completion_tokens":8,"total_tokens":28}}`,
		"",
		"data: [DONE]",
		"",
	}, "\n")
	upstream := &httpUpstreamRecorder{responses: []*http.Response{
		{
			StatusCode: http.StatusOK,
			Header:     http.Header{"Content-Type": []string{"text/event-stream"}},
			Body:       io.NopCloser(strings.NewReader(firstUpstreamBody)),
		},
		{
			StatusCode: http.StatusOK,
			Header:     http.Header{"Content-Type": []string{"application/json"}},
			Body: io.NopCloser(strings.NewReader(
				`{"id":"chatcmpl_agent_final","object":"chat.completion","model":"deepseek-v4-flash","choices":[{"index":0,"message":{"role":"assistant","content":"workspace inspected"},"finish_reason":"stop"}],"usage":{"prompt_tokens":30,"completion_tokens":4,"total_tokens":34}}`,
			)),
		},
	}}
	cache := &reasoningRecordingCache{}
	svc := &OpenAIGatewayService{
		cfg:          rawChatCompletionsTestConfig(),
		httpUpstream: upstream,
		cache:        cache,
	}
	account := forceChatResponsesFallbackAccount()

	firstBody := []byte(`{
		"model":"deepseek-v4-flash",
		"instructions":"Keep the repository clean.",
		"input":"inspect the workspace",
		"tools":[{"type":"function","name":"exec_command","description":"Run a command","parameters":{"type":"object","properties":{"cmd":{"type":"string"}},"required":["cmd"]}}],
		"stream":true
	}`)
	firstRecorder := httptest.NewRecorder()
	firstContext, _ := gin.CreateTestContext(firstRecorder)
	firstContext.Request = httptest.NewRequest(http.MethodPost, "/v1/responses", bytes.NewReader(firstBody))
	firstContext.Request.Header.Set("Content-Type", "application/json")
	firstResult, err := svc.Forward(context.Background(), firstContext, account, firstBody)
	require.NoError(t, err)
	require.NotNil(t, firstResult)
	require.NotContains(t, firstRecorder.Body.String(), "intermediate result")
	require.Contains(t, firstRecorder.Body.String(), compactAgentReasoningProgressForTest)

	secondBody := []byte(`{
		"model":"deepseek-v4-flash",
		"instructions":"Keep the repository clean.",
		"previous_response_id":"chatcmpl_agent_chain",
		"input":[{"type":"function_call_output","call_id":"call_chain","output":"/home/kuro"}],
		"tools":[{"type":"function","name":"exec_command","description":"Run a command","parameters":{"type":"object","properties":{"cmd":{"type":"string"}},"required":["cmd"]}}],
		"stream":false
	}`)
	secondRecorder := httptest.NewRecorder()
	secondContext, _ := gin.CreateTestContext(secondRecorder)
	secondContext.Request = httptest.NewRequest(http.MethodPost, "/v1/responses", bytes.NewReader(secondBody))
	secondContext.Request.Header.Set("Content-Type", "application/json")
	secondResult, err := svc.Forward(context.Background(), secondContext, account, secondBody)
	require.NoError(t, err)
	require.NotNil(t, secondResult)
	require.Len(t, upstream.bodies, 2)

	continued := upstream.bodies[1]
	require.Equal(t, "user", gjson.GetBytes(continued, "messages.1.role").String())
	require.Equal(t, "assistant", gjson.GetBytes(continued, "messages.2.role").String())
	require.Equal(t, "full private reasoning", gjson.GetBytes(continued, "messages.2.reasoning_content").String())
	require.Equal(t, "call_chain", gjson.GetBytes(continued, "messages.2.tool_calls.0.id").String())
	require.False(t, gjson.GetBytes(continued, "messages.2.content").Exists())
	require.Equal(t, "tool", gjson.GetBytes(continued, "messages.3.role").String())
	require.Equal(t, "call_chain", gjson.GetBytes(continued, "messages.3.tool_call_id").String())
	require.Equal(t, "/home/kuro", gjson.GetBytes(continued, "messages.3.content").String())
	require.Equal(t, "workspace inspected", gjson.Get(secondRecorder.Body.String(), "output.0.content.0.text").String())
}

const compactAgentReasoningProgressForTest = "Working through the current step."

func TestForwardResponses_MissingPreviousFallbackStateFailsExplicitly(t *testing.T) {
	gin.SetMode(gin.TestMode)
	body := []byte(`{
		"model":"deepseek-v4-flash",
		"previous_response_id":"chatcmpl_missing",
		"input":[{"type":"function_call_output","call_id":"call_missing","output":"ok"}],
		"tools":[{"type":"function","name":"exec_command","parameters":{"type":"object"}}],
		"stream":false
	}`)
	recorder := httptest.NewRecorder()
	c, _ := gin.CreateTestContext(recorder)
	c.Request = httptest.NewRequest(http.MethodPost, "/v1/responses", bytes.NewReader(body))
	c.Request.Header.Set("Content-Type", "application/json")
	upstream := &httpUpstreamRecorder{}
	svc := &OpenAIGatewayService{
		cfg:          rawChatCompletionsTestConfig(),
		httpUpstream: upstream,
		cache:        &reasoningRecordingCache{},
	}

	result, err := svc.Forward(context.Background(), c, forceChatResponsesFallbackAccount(), body)
	require.ErrorIs(t, err, errResponsesChatFallbackStateNotFound)
	require.Nil(t, result)
	require.Equal(t, http.StatusBadRequest, recorder.Code)
	require.Contains(t, recorder.Body.String(), "previous_response_id state is unavailable")
	require.Empty(t, upstream.requests)
}

// 流式响应里的 reasoning_content 应按 reasoning item id 写入缓存，供后续轮次
// 客户端不回传明文 summary 时回注（DeepSeek thinking mode 400 修复的写入侧）。
func TestForwardResponses_ChatFallbackCachesStreamedReasoning(t *testing.T) {
	gin.SetMode(gin.TestMode)

	body := []byte(`{"model":"deepseek-reasoner","input":"hello","stream":true}`)
	rec := httptest.NewRecorder()
	c, _ := gin.CreateTestContext(rec)
	c.Request = httptest.NewRequest(http.MethodPost, "/v1/responses", bytes.NewReader(body))
	c.Request.Header.Set("Content-Type", "application/json")

	upstreamBody := strings.Join([]string{
		`data: {"id":"chatcmpl_rc","object":"chat.completion.chunk","model":"deepseek-reasoner","choices":[{"index":0,"delta":{"role":"assistant"},"finish_reason":null}]}`,
		"",
		`data: {"id":"chatcmpl_rc","object":"chat.completion.chunk","model":"deepseek-reasoner","choices":[{"index":0,"delta":{"reasoning_content":"think "},"finish_reason":null}]}`,
		"",
		`data: {"id":"chatcmpl_rc","object":"chat.completion.chunk","model":"deepseek-reasoner","choices":[{"index":0,"delta":{"reasoning_content":"first"},"finish_reason":null}]}`,
		"",
		`data: {"id":"chatcmpl_rc","object":"chat.completion.chunk","model":"deepseek-reasoner","choices":[{"index":0,"delta":{"content":"answer"},"finish_reason":"stop"}]}`,
		"",
		`data: {"id":"chatcmpl_rc","object":"chat.completion.chunk","model":"deepseek-reasoner","choices":[],"usage":{"prompt_tokens":4,"completion_tokens":3,"total_tokens":7}}`,
		"",
		"data: [DONE]",
		"",
	}, "\n")
	upstream := &httpUpstreamRecorder{resp: &http.Response{
		StatusCode: http.StatusOK,
		Header:     http.Header{"Content-Type": []string{"text/event-stream"}, "x-request-id": []string{"rid_reasoning_cache_stream"}},
		Body:       io.NopCloser(strings.NewReader(upstreamBody)),
	}}
	cache := &reasoningRecordingCache{}
	svc := &OpenAIGatewayService{
		cfg:          rawChatCompletionsTestConfig(),
		httpUpstream: upstream,
		cache:        cache,
	}

	result, err := svc.Forward(context.Background(), c, forceChatResponsesFallbackAccount(), body)
	require.NoError(t, err)
	require.NotNil(t, result)

	sets := cache.snapshotSets()
	require.Len(t, sets, 1, "应恰好缓存一个 reasoning item")
	for itemID, content := range sets {
		require.NotEmpty(t, itemID)
		require.Equal(t, "think first", content)
	}
}

// 请求侧：encrypted-only reasoning item（无明文 summary）经缓存回查补回
// reasoning_content；带明文 summary 的 item 顺手回写缓存（自愈）。
func TestForwardResponses_ChatFallbackRestoresReasoningFromCache(t *testing.T) {
	gin.SetMode(gin.TestMode)

	body := []byte(`{
		"model":"deepseek-reasoner",
		"stream":false,
		"input":[
			{"type":"reasoning","id":"item_plain","summary":[{"type":"summary_text","text":"plain thinking"}]},
			{"type":"function_call","call_id":"call_0","name":"get_value","arguments":"{}"},
			{"type":"function_call_output","call_id":"call_0","output":"ok"},
			{"type":"reasoning","id":"item_enc1","summary":[],"encrypted_content":"opaque"},
			{"type":"function_call","call_id":"call_1","name":"get_value","arguments":"{}"},
			{"type":"function_call_output","call_id":"call_1","output":"ok"},
			{"type":"message","role":"user","content":[{"type":"input_text","text":"go on"}]}
		]
	}`)
	rec := httptest.NewRecorder()
	c, _ := gin.CreateTestContext(rec)
	c.Request = httptest.NewRequest(http.MethodPost, "/v1/responses", bytes.NewReader(body))
	c.Request.Header.Set("Content-Type", "application/json")

	upstream := &httpUpstreamRecorder{resp: &http.Response{
		StatusCode: http.StatusOK,
		Header:     http.Header{"Content-Type": []string{"application/json"}, "x-request-id": []string{"rid_reasoning_cache_restore"}},
		Body: io.NopCloser(strings.NewReader(
			`{"id":"chatcmpl_restore","object":"chat.completion","model":"deepseek-reasoner","choices":[{"index":0,"message":{"role":"assistant","content":"ok"},"finish_reason":"stop"}],"usage":{"prompt_tokens":3,"completion_tokens":2,"total_tokens":5}}`,
		)),
	}}
	cache := &reasoningRecordingCache{
		getResp: map[string]string{"item_enc1": "cached thinking"},
	}
	svc := &OpenAIGatewayService{
		cfg:          rawChatCompletionsTestConfig(),
		httpUpstream: upstream,
		cache:        cache,
	}

	result, err := svc.Forward(context.Background(), c, forceChatResponsesFallbackAccount(), body)
	require.NoError(t, err)
	require.NotNil(t, result)

	// 明文 summary 的 assistant 工具调用消息：reasoning_content 来自 summary 本身。
	require.Equal(t, "plain thinking", gjson.GetBytes(upstream.lastBody, "messages.0.reasoning_content").String())
	require.Equal(t, "call_0", gjson.GetBytes(upstream.lastBody, "messages.0.tool_calls.0.id").String())
	require.Equal(t, "tool", gjson.GetBytes(upstream.lastBody, "messages.1.role").String())
	// encrypted-only 的 assistant 工具调用消息：reasoning_content 来自缓存回查。
	require.Equal(t, "cached thinking", gjson.GetBytes(upstream.lastBody, "messages.2.reasoning_content").String())
	require.Equal(t, "call_1", gjson.GetBytes(upstream.lastBody, "messages.2.tool_calls.0.id").String())
	require.Equal(t, "tool", gjson.GetBytes(upstream.lastBody, "messages.3.role").String())

	// 明文 summary 的 item 被回写进缓存（自愈）。
	require.Equal(t, "plain thinking", cache.snapshotSets()["item_plain"])
}
