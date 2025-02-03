use crate::llm::{CompletionBuilder, LLMError, Model, Provider};
use crate::llm::{Message, MessageContent, Role};
use crate::prompts::DISCARD_REDUNDANT_SCREENSHOT_SYSTEM_PROMPT;
use crate::screenshot::Screenshot;
use crate::utils::{parse_markdown_code_block, MarkdownCodeBlockMissingError};
use image::{ImageBuffer, Rgba};
use serde::{Deserialize, Serialize};
use std::sync::Mutex;
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

#[derive(Debug, Clone)]
struct ImageMergeResult {
    is_vertical_shift: bool,
    is_horizontal_shift: bool,
    merged_image: Option<ImageBuffer<Rgba<u8>, Vec<u8>>>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct ImageCropConfig {
    // The number of pixels to crop from the left and right of the image
    lpixels: u32,
    rpixels: u32,

    // The number of pixels to crop from the top and bottom of the image
    tpixels: u32,
    bpixels: u32,
}

static CACHED_COMPUTER_OUTLINE: Mutex<Option<ImageCropConfig>> = Mutex::new(None);

pub fn crop_image_from_computer_outline(
    image: &ImageBuffer<Rgba<u8>, Vec<u8>>,
    history: &Vec<ImageBuffer<Rgba<u8>, Vec<u8>>>,
) -> ImageBuffer<Rgba<u8>, Vec<u8>> {
    let computer_outline = CACHED_COMPUTER_OUTLINE.lock().unwrap().clone();
    if let Some(computer_outline) = computer_outline {
        return crop_image(image, computer_outline);
    }
    let computer_outline = find_computer_outline(history);
    CACHED_COMPUTER_OUTLINE
        .lock()
        .unwrap()
        .replace(computer_outline.clone());
    crop_image(image, computer_outline)
}

pub fn find_computer_outline(history: &Vec<ImageBuffer<Rgba<u8>, Vec<u8>>>) -> ImageCropConfig {
    todo!()
}

pub fn crop_image(
    image: &ImageBuffer<Rgba<u8>, Vec<u8>>,
    crop_config: ImageCropConfig,
) -> ImageBuffer<Rgba<u8>, Vec<u8>> {
    let cropped_height = image.height() - crop_config.tpixels - crop_config.bpixels;
    let cropped_width = image.width() - crop_config.lpixels - crop_config.rpixels;
    let mut new_image: ImageBuffer<Rgba<u8>, Vec<u8>> =
        ImageBuffer::new(cropped_width, cropped_height);
    for i in crop_config.lpixels..new_image.width() {
        for j in crop_config.tpixels..new_image.height() {
            new_image.put_pixel(i, j, *image.get_pixel(i, j));
        }
    }
    new_image
}

pub const MAX_VERTICAL_SHIFT_PIXELS: u32 = 256;
pub const MAX_HORIZONTAL_SHIFT_PIXELS: u32 = 512;

pub fn attempt_image_merge(
    last_image: &ImageBuffer<Rgba<u8>, Vec<u8>>,
    current_image: &ImageBuffer<Rgba<u8>, Vec<u8>>,
    history: &Vec<ImageBuffer<Rgba<u8>, Vec<u8>>>,
) -> ImageMergeResult {
    if last_image == current_image {
        return ImageMergeResult {
            is_vertical_shift: false,
            is_horizontal_shift: false,
            merged_image: None,
        };
    }
    let last_image_cropped = crop_image_from_computer_outline(last_image, &history);
    let current_image_cropped = crop_image_from_computer_outline(current_image, &history);
    let vertical_shift_result =
        attempt_image_merge_with_vertical_shift(&last_image_cropped, &current_image_cropped);
    if vertical_shift_result.is_vertical_shift {
        return vertical_shift_result;
    }
    let horizontal_shift_result =
        attempt_image_merge_with_horizontal_shift(&last_image_cropped, &current_image_cropped);
    if horizontal_shift_result.is_horizontal_shift {
        return horizontal_shift_result;
    }
    ImageMergeResult {
        is_vertical_shift: false,
        is_horizontal_shift: false,
        merged_image: None,
    }
}

fn attempt_image_merge_with_vertical_shift(
    last_image_cropped: &ImageBuffer<Rgba<u8>, Vec<u8>>,
    current_image_cropped: &ImageBuffer<Rgba<u8>, Vec<u8>>,
) -> ImageMergeResult {
    todo!()
}

fn attempt_image_merge_with_horizontal_shift(
    last_image_cropped: &ImageBuffer<Rgba<u8>, Vec<u8>>,
    current_image_cropped: &ImageBuffer<Rgba<u8>, Vec<u8>>,
) -> ImageMergeResult {
    todo!()
}
