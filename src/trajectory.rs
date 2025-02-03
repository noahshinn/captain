use crate::llm::{Message, MessageContent, Role};
use crate::screenshot::Screenshot;
use chrono::{DateTime, Utc};
use std::sync::Arc;
use tokio::sync::Mutex;

// save 5000 tokens for the conversation
// 50 images * 1600 tokens per image = 80000 tokens
// 50 screenshot text descriptions * 500 tokens per description = 25000 tokens
// total = 105000 tokens
const MAX_NUM_IMAGES_PER_LLM_CALL: usize = 50;
const MAX_NUM_SCREENSHOT_TEXT_DESCRIPTIONS_PER_LLM_CALL: usize = 50;

#[derive(Debug, Clone)]
pub struct Trajectory {
    events: Arc<Mutex<Vec<Event>>>,
    discard_redundant_screenshots: bool,
}

#[derive(Debug, Clone)]
pub struct ScreenshotEvent {
    pub text_description: Option<String>,
    pub screenshot: Screenshot,
}

#[derive(Debug, Clone)]
pub enum Event {
    Message(Message),
    Screenshot(ScreenshotEvent),
}

impl Trajectory {
    pub fn new(discard_redundant_screenshots: bool) -> Self {
        Self {
            events: Arc::new(Mutex::new(Vec::new())),
            discard_redundant_screenshots,
        }
    }

    pub async fn add_event(&mut self, event: Event) {
        self.events.lock().await.push(event);
    }

    pub async fn add_message(&mut self, message: Message) {
        self.events.lock().await.push(Event::Message(message));
    }

    pub async fn add_assistant_message(&mut self, text: String) {
        self.events.lock().await.push(Event::Message(Message {
            role: Role::Assistant,
            content: MessageContent::Text(text),
        }));
    }

    pub async fn add_user_message(&mut self, text: String) {
        self.events.lock().await.push(Event::Message(Message {
            role: Role::User,
            content: MessageContent::Text(text),
        }));
    }

    pub async fn add_screenshot(&mut self, screenshot: Screenshot) {
        if self.discard_redundant_screenshots {
            if let Some(last_screenshot_event) =
                self.events
                    .lock()
                    .await
                    .last()
                    .and_then(|event| match event {
                        Event::Screenshot(screenshot) => Some(screenshot),
                        _ => None,
                    })
            {
                if last_screenshot_event.screenshot.image == screenshot.image {
                    return;
                }
            }
        }
        let events = self.events.clone();
        let conversation_history = self
            .build_messages()
            .await
            .into_iter()
            .filter(|message| message.role != Role::System)
            .collect::<Vec<Message>>();
        tokio::spawn(async move {
            let text_description = crate::screenshot::generate_text_description_of_screenshot(
                &screenshot,
                &conversation_history,
            )
            .await;
            match text_description {
                Ok(text_description) => {
                    events.lock().await.push(Event::Screenshot(ScreenshotEvent {
                        text_description: Some(text_description),
                        screenshot: screenshot,
                    }));
                }
                Err(e) => {
                    println!(
                        "[warning] Error generating text description of screenshot: {}",
                        e
                    );
                }
            }
        });
    }

    pub async fn build_messages(&self) -> Vec<Message> {
        let mut messages = Vec::new();
        let mut num_images = 0;
        let mut num_image_text_descriptions = 0;
        for event in self.events.lock().await.clone() {
            match event {
                Event::Message(message) => messages.push(message),
                Event::Screenshot(screenshot_event) => {
                    if num_images < MAX_NUM_IMAGES_PER_LLM_CALL {
                        messages.push(Message {
                            role: Role::User,
                            content: MessageContent::MultiContent(vec![
                                crate::llm::ContentBlock::Image {
                                    source: crate::llm::ImageSource {
                                        source_type: "base64".to_string(),
                                        media_type: "image/jpeg".to_string(),
                                        data: screenshot_event.screenshot.image_data,
                                    },
                                },
                            ]),
                        });
                        num_images += 1;
                    } else if let Some(text_description) = screenshot_event.text_description {
                        if num_image_text_descriptions
                            < MAX_NUM_SCREENSHOT_TEXT_DESCRIPTIONS_PER_LLM_CALL
                        {
                            let datetime: DateTime<Utc> =
                                screenshot_event.screenshot.timestamp.into();
                            let formatted_datetime = datetime.format("%d/%m/%Y %T");
                            messages.push(Message {
                                role: Role::User,
                                content: MessageContent::Text(format!(
                                    "[Text description of screenshot at {}]: {}",
                                    formatted_datetime, text_description
                                )),
                            });
                            num_image_text_descriptions += 1;
                        }
                    }
                }
            }
        }
        messages
    }
}
