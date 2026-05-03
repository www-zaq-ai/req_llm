defmodule ReqLLM.Providers.DeepseekTest do
  @moduledoc """
  Provider-level tests for DeepSeek implementation.

  Tests the provider contract, configuration, and OpenAI-compatible
  request/response handling without making live API calls.
  """

  use ReqLLM.ProviderCase, provider: ReqLLM.Providers.Deepseek

  alias ReqLLM.Providers.Deepseek

  defp deepseek_model(model_id \\ "deepseek-chat", opts \\ []) do
    %LLMDB.Model{
      id: "deepseek:#{model_id}",
      model: model_id,
      name: Keyword.get(opts, :name, "Deepseek Test Model"),
      provider: :deepseek,
      family: Keyword.get(opts, :family, "test"),
      capabilities: Keyword.get(opts, :capabilities, %{chat: true, tools: %{enabled: true}}),
      limits: Keyword.get(opts, :limits, %{context: 64_000, output: 8192})
    }
  end

  describe "provider contract" do
    test "provider identity and configuration" do
      assert Deepseek.provider_id() == :deepseek
      assert is_binary(Deepseek.base_url())
      assert Deepseek.base_url() == "https://api.deepseek.com"
    end

    test "provider uses correct default environment key" do
      assert Deepseek.default_env_key() == "DEEPSEEK_API_KEY"
    end

    test "provider schema is empty (pure OpenAI-compatible)" do
      schema_keys = Deepseek.provider_schema().schema |> Keyword.keys()
      assert schema_keys == []
    end

    test "provider_extended_generation_schema includes all core keys" do
      extended_schema = Deepseek.provider_extended_generation_schema()
      extended_keys = extended_schema.schema |> Keyword.keys()

      core_keys = ReqLLM.Provider.Options.all_generation_keys()
      core_without_meta = Enum.reject(core_keys, &(&1 == :provider_options))

      for core_key <- core_without_meta do
        assert core_key in extended_keys,
               "Extended schema missing core key: #{core_key}"
      end
    end
  end

  describe "request preparation" do
    test "prepare_request for :chat creates /chat/completions request" do
      model = deepseek_model()
      prompt = "Hello world"
      opts = [temperature: 0.7, max_tokens: 100]

      {:ok, request} = Deepseek.prepare_request(:chat, model, prompt, opts)

      assert %Req.Request{} = request
      assert request.url.path == "/chat/completions"
      assert request.method == :post
    end

    test "prepare_request rejects unsupported operations" do
      model = deepseek_model()
      context = context_fixture()

      {:error, error} = Deepseek.prepare_request(:unsupported, model, context, [])
      assert %ReqLLM.Error.Invalid.Parameter{} = error
    end
  end

  describe "authentication wiring" do
    test "attach adds Bearer authorization header" do
      model = deepseek_model()
      request = Req.new()

      attached = Deepseek.attach(request, model, [])

      auth_header = attached.headers["authorization"]
      assert auth_header != nil
      assert String.starts_with?(List.first(auth_header), "Bearer ")
    end

    test "attach adds pipeline steps" do
      model = deepseek_model()
      request = Req.new()

      attached = Deepseek.attach(request, model, [])

      request_steps = Keyword.keys(attached.request_steps)
      response_steps = Keyword.keys(attached.response_steps)

      assert :llm_encode_body in request_steps
      assert :llm_decode_response in response_steps
    end
  end

  describe "base_url configuration" do
    test "uses default base_url when not overridden" do
      model = deepseek_model()
      {:ok, request} = Deepseek.prepare_request(:chat, model, "Hello", [])

      assert request.options[:base_url] == "https://api.deepseek.com"
    end

    test "respects base_url option override" do
      model = deepseek_model()
      custom_url = "https://custom.deepseek.com/v1"
      {:ok, request} = Deepseek.prepare_request(:chat, model, "Hello", base_url: custom_url)

      assert request.options[:base_url] == custom_url
    end
  end

  describe "body encoding" do
    test "encode_body produces valid OpenAI-compatible JSON" do
      model = deepseek_model()
      context = context_fixture()

      mock_request = %Req.Request{
        options: [
          context: context,
          model: model.model,
          stream: false,
          temperature: 0.7
        ]
      }

      updated_request = Deepseek.encode_body(mock_request)

      assert is_binary(updated_request.body)
      decoded = Jason.decode!(updated_request.body)

      assert decoded["model"] == "deepseek-chat"
      assert is_list(decoded["messages"])
      assert decoded["stream"] == false
    end

    test "encode_body handles tools correctly" do
      model = deepseek_model()
      context = context_fixture()

      tool =
        ReqLLM.Tool.new!(
          name: "get_weather",
          description: "Get the weather",
          parameter_schema: [
            location: [type: :string, required: true, doc: "City name"]
          ],
          callback: fn _ -> {:ok, "sunny"} end
        )

      mock_request = %Req.Request{
        options: [
          context: context,
          model: model.model,
          stream: false,
          tools: [tool]
        ]
      }

      updated_request = Deepseek.encode_body(mock_request)
      decoded = Jason.decode!(updated_request.body)

      assert is_list(decoded["tools"])
      assert length(decoded["tools"]) == 1
      assert hd(decoded["tools"])["function"]["name"] == "get_weather"
    end
  end

  describe "response decoding" do
    test "decode_response parses OpenAI-format response" do
      mock_response_body =
        openai_format_json_fixture(
          model: "deepseek-chat",
          content: "Hello from DeepSeek!"
        )

      mock_resp = %Req.Response{
        status: 200,
        body: mock_response_body
      }

      context = context_fixture()

      mock_req = %Req.Request{
        options: [
          context: context,
          model: "deepseek-chat",
          operation: :chat
        ]
      }

      {_req, decoded_resp} = Deepseek.decode_response({mock_req, mock_resp})

      assert %ReqLLM.Response{} = decoded_resp.body
      assert ReqLLM.Response.text(decoded_resp.body) == "Hello from DeepSeek!"
    end

    test "decode_response handles API errors" do
      error_body = %{
        "error" => %{
          "message" => "Invalid API key",
          "type" => "authentication_error"
        }
      }

      mock_resp = %Req.Response{
        status: 401,
        body: error_body
      }

      context = context_fixture()

      mock_req = %Req.Request{
        options: [context: context, id: "deepseek-chat"]
      }

      {_req, error} = Deepseek.decode_response({mock_req, mock_resp})

      assert %ReqLLM.Error.API.Response{} = error
      assert error.status == 401
    end
  end

  describe "streaming support" do
    test "attach_stream builds streaming request" do
      model = deepseek_model()
      context = context_fixture()
      opts = [temperature: 0.7]

      {:ok, finch_request} = Deepseek.attach_stream(model, context, opts, MyApp.Finch)

      assert %Finch.Request{} = finch_request
      assert finch_request.method == "POST"
      assert String.contains?(finch_request.path, "/chat/completions")

      headers_map = Map.new(finch_request.headers)
      assert headers_map["Authorization"] == "Bearer test-key-12345"
    end
  end

  describe "build_body regression" do
    test "adds reasoning_content to assistant messages without it" do
      model = deepseek_model()

      follow_up_context =
        ReqLLM.Context.new([
          ReqLLM.Context.system("You are a helpful assistant."),
          ReqLLM.Context.user("Hello"),
          ReqLLM.Context.assistant("Hi there!"),
          ReqLLM.Context.user("How are you?")
        ])

      mock_request = %Req.Request{
        options: [
          context: follow_up_context,
          model: model.model
        ]
      }

      body = Deepseek.build_body(mock_request)

      assistant_msg =
        Enum.find(body.messages, fn msg ->
          Map.get(msg, :role) == "assistant" or Map.get(msg, "role") == "assistant"
        end)

      refute is_nil(assistant_msg)
      assert Map.get(assistant_msg, :reasoning_content) == ""
    end

    test "preserves existing reasoning_content in assistant messages" do
      model = deepseek_model()

      messages_with_reasoning = [
        %{role: "system", content: "You are helpful."},
        %{role: "user", content: "Hello"},
        %{role: "assistant", content: "Previous response", reasoning_content: "kept reasoning"},
        %{role: "user", content: "Follow up"}
      ]

      mock_request = %Req.Request{
        options: [
          messages: messages_with_reasoning,
          model: model.model
        ]
      }

      body = Deepseek.build_body(mock_request)

      assistant_msg =
        Enum.find(body.messages, fn msg ->
          Map.get(msg, "role") == "assistant" or Map.get(msg, :role) == "assistant"
        end)

      refute is_nil(assistant_msg)
      assert Map.get(assistant_msg, :reasoning_content) == "kept reasoning"
    end

    test "preserves existing string-keyed reasoning_content in assistant messages" do
      model = deepseek_model()

      messages_with_reasoning = [
        %{"role" => "user", "content" => "Hello"},
        %{
          "role" => "assistant",
          "content" => "Previous response",
          "reasoning_content" => "kept reasoning"
        },
        %{"role" => "user", "content" => "Follow up"}
      ]

      mock_request = %Req.Request{
        options: [
          messages: messages_with_reasoning,
          model: model.model
        ]
      }

      body = Deepseek.build_body(mock_request)

      assistant_msg =
        Enum.find(body.messages, fn msg ->
          Map.get(msg, "role") == "assistant" or Map.get(msg, :role) == "assistant"
        end)

      refute is_nil(assistant_msg)
      assert Map.get(assistant_msg, "reasoning_content") == "kept reasoning"
      refute Map.has_key?(assistant_msg, :reasoning_content)
    end
  end
end
