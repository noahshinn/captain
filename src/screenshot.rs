use crate::llm::{
    CompletionBuilder, ContentBlock, ImageSource, LLMError, Message, MessageContent, Model,
    Provider, Role,
};
use crate::prompts::SCREENSHOT_DESCRIPTION_SYSTEM_PROMPT;
use base64::{engine::general_purpose::STANDARD as BASE64, Engine as _};
use image::{ImageBuffer, Rgba};
use screenshots::Screen;
use std::time::SystemTime;
use thiserror::Error;
use tokio::time::{interval, Duration};

#[derive(Debug, Clone)]
pub struct Screenshot {
    pub timestamp: SystemTime,
    // base64 encoded image data
    pub image_data: String,
    pub image: ImageBuffer<Rgba<u8>, Vec<u8>>,
}

#[derive(Error, Debug)]
pub enum ScreenshotError {
    #[error("Failed to capture screenshot")]
    CaptureError,
    #[error("Failed to encode screenshot")]
    EncodeError,
    #[error("No screens found")]
    NoScreensFound,
}

pub async fn start_capture(on_screenshot: impl Fn(Screenshot) + Send + 'static) {
    println!("Starting screenshot capture");
    let screenshot = take_screenshot().await;
    if let Ok(screenshot) = screenshot {
        on_screenshot(screenshot);
    }
    tokio::spawn(async move {
        let mut interval = interval(Duration::from_secs(5));
        loop {
            interval.tick().await;
            let screenshot = take_screenshot().await;
            if let Ok(screenshot) = screenshot {
                on_screenshot(screenshot);
            }
        }
    });
}

pub async fn take_screenshot() -> Result<Screenshot, ScreenshotError> {
    // for now, just take the first screen
    if let Some(screen) = Screen::all().unwrap().first() {
        if let Ok(image) = screen.capture() {
            let mut buffer = Vec::new();
            let mut encoder = image::codecs::jpeg::JpegEncoder::new(&mut buffer);
            if encoder.encode_image(&image).is_ok() {
                let base64 = BASE64.encode(&buffer);
                Ok(Screenshot {
                    timestamp: SystemTime::now(),
                    image_data: base64,
                    image,
                })
            } else {
                Err(ScreenshotError::EncodeError)
            }
        } else {
            Err(ScreenshotError::CaptureError)
        }
    } else {
        Err(ScreenshotError::NoScreensFound)
    }
}

pub async fn generate_text_description_of_screenshot(
    screenshot: &Screenshot,
    conversation_history: &[Message],
) -> Result<String, LLMError> {
    let mut messages = vec![Message {
        role: Role::System,
        content: MessageContent::Text(SCREENSHOT_DESCRIPTION_SYSTEM_PROMPT.to_string()),
    }];
    for message in conversation_history {
        messages.push(message.clone());
    }
    messages.push(Message {
        role: Role::User,
        content: MessageContent::MultiContent(vec![
            ContentBlock::Image {
                source: ImageSource {
                    source_type: "base64".to_string(),
                    media_type: "image/jpeg".to_string(),
                    data: screenshot.image_data.clone(),
                },
            },
            ContentBlock::Text {
                text: "Write a text description of this screenshot.".to_string(),
            },
        ]),
    });
    let completion_request = CompletionBuilder::new()
        .model(Model::GPT4oMini)
        .provider(Provider::OpenAI)
        .messages(messages)
        .temperature(0.0)
        .build();
    completion_request.do_request().await
}
