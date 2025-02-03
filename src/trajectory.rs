use crate::image_analysis::is_redundant_screenshot;
use crate::llm::{Message, MessageContent, Role};
use crate::screenshot::Screenshot;
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
    pub is_redundant: bool,
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
        events.lock().await.push(Event::Screenshot(ScreenshotEvent {
            text_description: None,
            screenshot: screenshot.clone(),
            is_redundant: false,
        }));
        let new_event_idx = events.lock().await.len() - 1;
        if self.discard_redundant_screenshots {
            let events = events.clone();
            let screenshot = screenshot.clone();
            tokio::spawn(async move {
                let last_screenshot = match events.lock().await.get(new_event_idx - 1) {
                    Some(Event::Screenshot(screenshot_event)) => {
                        screenshot_event.screenshot.clone()
                    }
                    _ => return,
                };
                let should_discard_previous_screenshot =
                    match is_redundant_screenshot(&last_screenshot, &screenshot).await {
                        Ok(should_discard_previous_screenshot) => {
                            should_discard_previous_screenshot
                        }
                        Err(e) => {
                            println!("[warning] Error checking if screenshot is redundant: {}", e);
                            false
                        }
                    };
                if should_discard_previous_screenshot {
                    if let Event::Screenshot(screenshot_event) =
                        &mut events.lock().await[new_event_idx]
                    {
                        screenshot_event.is_redundant = true;
                    }
                }
            });
        }
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
            // mutate the event to add the text description
            match text_description {
                Ok(text_description) => {
                    let mut events = events.lock().await;
                    if let Event::Screenshot(screenshot_event) = &mut events[new_event_idx] {
                        screenshot_event.text_description = Some(text_description);
                    }
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
                    if screenshot_event.is_redundant {
                        continue;
                    }
                    if num_images < MAX_NUM_IMAGES_PER_LLM_CALL {
                        messages.push(screenshot_event.screenshot.to_llm_message(None));
                        num_images += 1;
                    } else if let Some(text_description) = screenshot_event.text_description {
                        if num_image_text_descriptions
                            < MAX_NUM_SCREENSHOT_TEXT_DESCRIPTIONS_PER_LLM_CALL
                        {
                            messages.push(screenshot_event.screenshot.to_llm_message(Some(
                                format!("Text description: {}", text_description),
                            )));
                            num_image_text_descriptions += 1;
                        }
                    }
                }
            }
        }
        messages
    }
}
