use crate::llm::{CompletionOptions, ContentBlock, LLMError, Message, MessageContent, Model};
use reqwest::header::{HeaderMap, HeaderValue, CONTENT_TYPE};
use serde::{Deserialize, Serialize};
use std::env;

const ANTHROPIC_API_URL: &str = "https://api.anthropic.com/v1/messages";
const DEFAULT_ANTHROPIC_MAX_COMPLETION_TOKENS: i32 = 8192;

#[derive(Deserialize, Debug)]
struct AnthropicResponse {
    content: Vec<AnthropicContent>,
}

#[derive(Deserialize, Debug)]
struct AnthropicContent {
    text: String,
}

#[derive(Serialize, Debug)]
struct AnthropicMessage<'a> {
    role: &'a str,
    content: &'a [ContentBlock],
}

#[derive(Serialize)]
struct AnthropicRequest<'a> {
    model: String,
    messages: Vec<AnthropicMessage<'a>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    system: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    max_tokens: Option<i32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    temperature: Option<f64>,
}

pub async fn completion_anthropic(
    model: Model,
    messages: &[Message],
    options: Option<&CompletionOptions>,
) -> Result<String, LLMError> {
    let (system_content, messages) =
        if !messages.is_empty() && matches!(messages[0].role, crate::llm::Role::System) {
            let content = match &messages[0].content {
                MessageContent::Text(text) => Some(text.clone()),
                MessageContent::MultiContent(_) => None, // System messages should be text only
            };
            (content, &messages[1..])
        } else {
            (None, messages)
        };

    let anthropic_messages: Vec<_> = messages
        .iter()
        .map(|msg| {
            let role = match msg.role {
                crate::llm::Role::User => "user",
                crate::llm::Role::Assistant => "assistant",
                crate::llm::Role::System => "system",
            };

            let content = match &msg.content {
                MessageContent::Text(text) => vec![ContentBlock::Text { text: text.clone() }],
                MessageContent::MultiContent(blocks) => blocks.clone(),
            };

            (role, content)
        })
        .collect();

    let anthropic_messages: Vec<_> = anthropic_messages
        .iter()
        .map(|(role, content)| AnthropicMessage { role, content })
        .collect();
    let req_body = AnthropicRequest {
        model: model.to_string(),
        messages: anthropic_messages,
        system: system_content,
        max_tokens: Some(
            options
                .map(|opt| opt.max_completion_tokens)
                .filter(|&t| t != 0)
                .unwrap_or(DEFAULT_ANTHROPIC_MAX_COMPLETION_TOKENS),
        ),
        temperature: options.and_then(|opt| (opt.temperature != 0.0).then_some(opt.temperature)),
    };

    let api_key = match env::var("ANTHROPIC_API_KEY") {
        Ok(key) => key,
        Err(_) => {
            return Err(LLMError::RequestBuildingError(
                "ANTHROPIC_API_KEY environment variable not set".to_string(),
            ))
        }
    };

    let mut headers = HeaderMap::new();
    let api_header = match HeaderValue::from_str(&api_key) {
        Ok(header) => header,
        Err(e) => return Err(LLMError::RequestBuildingError(e.to_string())),
    };
    headers.insert("x-api-key", api_header);
    headers.insert(CONTENT_TYPE, HeaderValue::from_static("application/json"));
    headers.insert("anthropic-version", HeaderValue::from_static("2023-06-01"));

    let client = reqwest::Client::new();
    let response = match client
        .post(ANTHROPIC_API_URL)
        .headers(headers)
        .json(&req_body)
        .send()
        .await
    {
        Ok(resp) => resp,
        Err(e) => return Err(LLMError::RequestError(e)),
    };

    let status = response.status();
    if !status.is_success() {
        let error_text = response
            .text()
            .await
            .unwrap_or_else(|_| "Unable to read error response".to_string());
        return Err(LLMError::Other(
            format!(
                "Anthropic API request failed with status {}: {}",
                status, error_text
            )
            .into(),
        ));
    }

    let response_body: AnthropicResponse = match response.json().await {
        Ok(body) => body,
        Err(e) => return Err(LLMError::RequestError(e)),
    };

    response_body
        .content
        .first()
        .map(|content| content.text.clone())
        .ok_or(LLMError::EmptyResponse)
}
