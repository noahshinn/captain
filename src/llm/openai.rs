use crate::llm::{CompletionOptions, ContentBlock, LLMError, Message, MessageContent, Model};
use reqwest::header::{HeaderMap, HeaderValue, AUTHORIZATION, CONTENT_TYPE};
use serde::{Deserialize, Serialize};
use std::env;

const OPENAI_API_URL: &str = "https://api.openai.com/v1/chat/completions";

#[derive(Serialize)]
struct RequestBody<'a> {
    model: String,
    messages: Vec<OpenAIMessage<'a>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    temperature: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    max_tokens: Option<i32>,
}

#[derive(Serialize)]
struct OpenAIMessage<'a> {
    role: &'a str,
    content: &'a [OpenAIContentBlock],
}

#[derive(Serialize)]
struct ImageURL {
    url: String,
}

#[derive(Serialize)]
#[serde(untagged)]
enum OpenAIContentBlock {
    Text {
        #[serde(rename = "type")]
        type_: &'static str,
        text: String,
    },
    Image {
        #[serde(rename = "type")]
        type_: &'static str,
        image_url: ImageURL,
    },
}

#[derive(Deserialize)]
struct Response {
    choices: Vec<Choice>,
}

#[derive(Deserialize)]
struct Choice {
    message: ResponseMessage,
}

#[derive(Deserialize)]
struct ResponseMessage {
    content: String,
}

pub(crate) async fn completion_openai(
    model: Model,
    messages: &[Message],
    options: Option<&CompletionOptions>,
) -> Result<String, LLMError> {
    let headers = match build_openai_request_headers() {
        Ok(headers) => headers,
        Err(e) => return Err(e),
    };
    let openai_messages: Vec<_> = messages
        .iter()
        .map(|msg| {
            let role = match msg.role {
                crate::llm::Role::User => "user",
                crate::llm::Role::Assistant => "assistant",
                crate::llm::Role::System => "system",
            };

            let content = match &msg.content {
                MessageContent::Text(text) => vec![OpenAIContentBlock::Text {
                    type_: "text",
                    text: text.clone(),
                }],
                MessageContent::MultiContent(blocks) => blocks
                    .iter()
                    .map(|block| match block {
                        ContentBlock::Text { text } => OpenAIContentBlock::Text {
                            type_: "text",
                            text: text.clone(),
                        },
                        ContentBlock::Image { source } => OpenAIContentBlock::Image {
                            type_: "image_url",
                            image_url: ImageURL {
                                url: format!("data:image/jpeg;base64,{}", source.data),
                            },
                        },
                    })
                    .collect(),
            };
            (role, content)
        })
        .collect();

    let openai_messages: Vec<OpenAIMessage> = openai_messages
        .iter()
        .map(|(role, content)| OpenAIMessage { role, content })
        .collect();
    let req_body = RequestBody {
        model: model.to_string(),
        messages: openai_messages,
        temperature: options.and_then(|opt| (opt.temperature != 0.0).then_some(opt.temperature)),
        max_tokens: options
            .and_then(|opt| (opt.max_completion_tokens != 0).then_some(opt.max_completion_tokens)),
    };
    let client = reqwest::Client::new();
    let response = match client
        .post(OPENAI_API_URL)
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
                "OpenAI API request failed with status {}: {}",
                status, error_text
            )
            .into(),
        ));
    }

    let response_body: Response = match response.json().await {
        Ok(body) => body,
        Err(e) => return Err(LLMError::RequestError(e)),
    };

    response_body
        .choices
        .first()
        .map(|choice| choice.message.content.clone())
        .ok_or(LLMError::EmptyResponse)
}

fn build_openai_request_headers() -> Result<HeaderMap, LLMError> {
    let api_key = match env::var("OPENAI_API_KEY") {
        Ok(key) => key,
        Err(_) => {
            return Err(LLMError::RequestBuildingError(
                "OPENAI_API_KEY environment variable not set".to_string(),
            ))
        }
    };
    let mut headers = HeaderMap::new();
    let auth_header = match HeaderValue::from_str(&format!("Bearer {api_key}")) {
        Ok(header) => header,
        Err(e) => return Err(LLMError::RequestBuildingError(e.to_string())),
    };
    headers.insert(AUTHORIZATION, auth_header);
    headers.insert(CONTENT_TYPE, HeaderValue::from_static("application/json"));
    Ok(headers)
}
