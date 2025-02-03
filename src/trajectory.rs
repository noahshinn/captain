use crate::embeddings::embedding;
use crate::image_analysis::is_redundant_screenshot;
use crate::llm::{Message, MessageContent, Role};
use crate::screenshot::{generate_text_description_of_screenshot, Screenshot};
use crate::search::{dense_embedding_search, EmbeddedDocument, SearchError};
use std::sync::Arc;
use thiserror::Error;
use tokio::sync::Mutex;

// save 5000 tokens for the conversation
// 80 images * 1600 tokens per image = 128000 tokens
// total = 5000 + 128000 = 133000 tokens
const MAX_NUM_IMAGES_PER_LLM_CALL: usize = 80;
const MAX_NUM_EXPLICIT_RECENT_IMAGES_PER_LLM_CALL: usize = 40;
const MAX_NUM_RETRIEVED_IMAGES_PER_LLM_CALL: usize =
    MAX_NUM_IMAGES_PER_LLM_CALL - MAX_NUM_EXPLICIT_RECENT_IMAGES_PER_LLM_CALL;

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
    pub text_embedding: Option<Vec<f32>>,
}

#[derive(Debug, Clone)]
pub enum Event {
    Message(Message),
    Screenshot(ScreenshotEvent),
}

#[derive(Error, Debug)]
pub enum BuildMessagesError {
    #[error("Error retrieving images")]
    RetrievalError(#[from] SearchError),
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
            text_embedding: None,
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
            .build_messages(None)
            .await
            .unwrap()
            .into_iter()
            .filter(|message| message.role != Role::System)
            .collect::<Vec<Message>>();
        tokio::spawn(async move {
            let text_description =
                generate_text_description_of_screenshot(&screenshot, &conversation_history).await;
            match text_description {
                Ok(text_description) => {
                    let mut events = events.lock().await;
                    if let Event::Screenshot(screenshot_event) = &mut events[new_event_idx] {
                        screenshot_event.text_description = Some(text_description.clone());
                    }
                    let text_embedding = match embedding(vec![text_description]).await {
                        Ok(text_embedding) => text_embedding,
                        Err(e) => {
                            println!("[warning] Error generating text embedding: {}", e);
                            return;
                        }
                    };
                    if let Event::Screenshot(screenshot_event) = &mut events[new_event_idx] {
                        screenshot_event.text_embedding =
                            Some(text_embedding.first().unwrap().clone());
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

    pub async fn build_messages(
        &self,
        query_for_retrieval: Option<&str>,
    ) -> Result<Vec<Message>, BuildMessagesError> {
        let mut messages_rev = Vec::new();
        let mut num_explicit_recent_images = 0;
        let mut retrieval_corpus_screenshot_idxs: Vec<usize> = Vec::new();
        for (idx, event) in self
            .events
            .lock()
            .await
            .clone()
            .into_iter()
            .rev()
            .enumerate()
        {
            match event {
                Event::Message(message) => messages_rev.push(message),
                Event::Screenshot(screenshot_event) => {
                    if screenshot_event.is_redundant {
                        continue;
                    }
                    if num_explicit_recent_images < MAX_NUM_EXPLICIT_RECENT_IMAGES_PER_LLM_CALL {
                        messages_rev.push(screenshot_event.screenshot.to_llm_message(None));
                        num_explicit_recent_images += 1;
                    } else if query_for_retrieval.is_some() {
                        retrieval_corpus_screenshot_idxs.push(idx);
                    }
                }
            }
        }
        let mut retrieval_corpus: Vec<EmbeddedDocument<Screenshot>> = Vec::new();
        let events = self.events.lock().await.clone();
        for idx in retrieval_corpus_screenshot_idxs {
            if let Event::Screenshot(screenshot_event) = &events[idx] {
                retrieval_corpus.push(EmbeddedDocument {
                    document: screenshot_event.screenshot.clone(),
                    embedding: &screenshot_event.text_embedding.as_ref().unwrap(),
                });
            }
        }
        if let Some(query) = query_for_retrieval {
            if retrieval_corpus.len() > MAX_NUM_RETRIEVED_IMAGES_PER_LLM_CALL {
                let top_k_relevant_images = match dense_embedding_search(
                    query,
                    &retrieval_corpus,
                    MAX_NUM_RETRIEVED_IMAGES_PER_LLM_CALL,
                )
                .await
                {
                    Ok(top_k_relevant_images) => top_k_relevant_images,
                    Err(e) => return Err(BuildMessagesError::RetrievalError(e)),
                };
                top_k_relevant_images.into_iter().for_each(|image| {
                    messages_rev.insert(
                        messages_rev.len() - 1,
                        image.embedded_document.document.to_llm_message(None),
                    );
                });
            } else {
                retrieval_corpus.into_iter().for_each(|image| {
                    messages_rev
                        .insert(messages_rev.len() - 1, image.document.to_llm_message(None));
                });
            }
        }
        messages_rev.reverse();
        Ok(messages_rev)
    }
}
