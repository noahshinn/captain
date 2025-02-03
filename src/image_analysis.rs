use crate::llm::{CompletionBuilder, LLMError, Model, Provider};
use crate::llm::{Message, MessageContent, Role};
use crate::prompts::DISCARD_REDUNDANT_SCREENSHOT_SYSTEM_PROMPT;
use crate::screenshot::Screenshot;
use crate::utils::{parse_markdown_code_block, MarkdownCodeBlockMissingError};
use serde::{Deserialize, Serialize};
use thiserror::Error;

#[derive(Error, Debug)]
pub enum DiscardRedundantScreenshotError {
    #[error("Error generating text description of screenshot")]
    LLMError(#[from] LLMError),
    #[error("Markdown code block missing in response")]
    MarkdownCodeBlockMissingError(#[from] MarkdownCodeBlockMissingError),
    #[error("Error parsing JSON response")]
    JSONError(#[from] serde_json::Error),
}

const SIMILARITY_THRESHOLD_NUM_PIXELS: i32 = 1_000_000;

pub async fn is_redundant_screenshot(
    last_screenshot: &Screenshot,
    current_screenshot: &Screenshot,
) -> Result<bool, DiscardRedundantScreenshotError> {
    if last_screenshot.image == current_screenshot.image {
        return Ok(true);
    }
    if detect_temporal_change_in_same_content(last_screenshot, current_screenshot) {
        return should_discard_past_screenshot(last_screenshot, current_screenshot).await;
    }
    Ok(false)
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct PreviousScreenshotContainsImportantInformationNotPresentInCurrentScreenshotResponse {
    previous_screenshot_contains_important_information_not_present_in_current_screenshot: bool,
}

async fn should_discard_past_screenshot(
    last_screenshot: &Screenshot,
    current_screenshot: &Screenshot,
) -> Result<bool, DiscardRedundantScreenshotError> {
    let messages = vec![
        Message {
            role: Role::System,
            content: MessageContent::Text(DISCARD_REDUNDANT_SCREENSHOT_SYSTEM_PROMPT.to_string()),
        },
        last_screenshot.to_llm_message(Some("Previous screenshot".to_string())),
        current_screenshot.to_llm_message(Some("Current screenshot".to_string())),
        Message {
            role: Role::User,
            content: MessageContent::Text(
                "Determine if the previous screenshot should be discarded.".to_string(),
            ),
        },
    ];
    let completion_request = CompletionBuilder::new()
        .model(Model::GPT4oMini)
        .provider(Provider::OpenAI)
        .messages(messages)
        .build();
    let response = match completion_request.do_request().await {
        Ok(completion) => completion,
        Err(e) => return Err(DiscardRedundantScreenshotError::LLMError(e)),
    };
    let json_string = match parse_markdown_code_block(&response) {
        Ok(json_string) => json_string,
        Err(e) => return Err(DiscardRedundantScreenshotError::MarkdownCodeBlockMissingError(e)),
    };
    let json: PreviousScreenshotContainsImportantInformationNotPresentInCurrentScreenshotResponse =
        match serde_json::from_str(&json_string) {
            Ok(json) => json,
            Err(e) => return Err(DiscardRedundantScreenshotError::JSONError(e)),
        };
    Ok(!json.previous_screenshot_contains_important_information_not_present_in_current_screenshot)
}

fn detect_temporal_change_in_same_content(
    last_screenshot: &Screenshot,
    current_screenshot: &Screenshot,
) -> bool {
    if last_screenshot.image == current_screenshot.image {
        return true;
    }
    let mut num_exact_pixels = 0;
    for i in 0..last_screenshot.image.width() {
        for j in 0..last_screenshot.image.height() {
            if last_screenshot.image.get_pixel(i, j) == current_screenshot.image.get_pixel(i, j) {
                num_exact_pixels += 1;
            }
        }
    }
    num_exact_pixels > SIMILARITY_THRESHOLD_NUM_PIXELS
}
