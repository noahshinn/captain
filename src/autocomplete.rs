use crate::llm::{CompletionBuilder, LLMError, Message, MessageContent, Model, Provider, Role};
use crate::prompts::AUTOCOMPLETE_SYSTEM_PROMPT;
use crate::screenshot::take_screenshot;
use crate::screenshot::ScreenshotError;
use crate::trajectory::Trajectory;
use crate::utils::{parse_markdown_code_block, MarkdownCodeBlockMissingError};
use device_query::{DeviceQuery, DeviceState, Keycode};
use enigo::{Enigo, InputError, Keyboard, Settings};
use serde::{Deserialize, Serialize};
use std::sync::Arc;
use thiserror::Error;
use tokio::sync::Mutex;

#[derive(Error, Debug)]
pub enum AutocompleteError {
    #[error("Error generating autocompletion")]
    GenerateAutocompletionError(#[from] LLMError),
    #[error("Error taking screenshot")]
    ScreenshotError(#[from] ScreenshotError),
    #[error("Error typing text")]
    TypingError(#[from] InputError),
    #[error("Error parsing autocomplete response")]
    ParseAutocompleteResponseError(#[from] ParseAutocompleteResponseError),
}

#[derive(Error, Debug)]
pub enum ParseAutocompleteResponseError {
    #[error("Markdown code block missing")]
    MarkdownCodeBlockMissingError(#[from] MarkdownCodeBlockMissingError),
    #[error("Error parsing JSON")]
    ParseJsonError(#[from] serde_json::Error),
}

#[derive(Debug, Serialize, Deserialize)]
struct AutocompleteResponse {
    autocomplete: String,
}

async fn handle_autocomplete(
    trajectory: Arc<Mutex<Trajectory>>,
) -> Result<String, AutocompleteError> {
    let screenshot = take_screenshot().await?;
    trajectory.lock().await.add_screenshot(screenshot).await;
    let response = match generate_autocompletion(trajectory.clone()).await {
        Ok(response) => response,
        Err(e) => {
            return Err(AutocompleteError::GenerateAutocompletionError(e));
        }
    };
    println!("Autocompletion generated: {}", response);
    let response_clone = response.clone();
    match tokio::task::spawn_blocking(move || {
        let mut enigo = Enigo::new(&Settings::default()).unwrap();
        enigo.text(&response_clone)
    })
    .await
    .unwrap()
    {
        Ok(_) => (),
        Err(e) => {
            return Err(AutocompleteError::TypingError(e));
        }
    }
    let autocomplete_response = match parse_autocomplete_response(&response) {
        Ok(autocomplete_response) => autocomplete_response,
        Err(e) => {
            return Err(AutocompleteError::ParseAutocompleteResponseError(e));
        }
    };
    trajectory
        .lock()
        .await
        .add_assistant_message(autocomplete_response.autocomplete.clone())
        .await;
    Ok(autocomplete_response.autocomplete)
}

async fn generate_autocompletion(trajectory: Arc<Mutex<Trajectory>>) -> Result<String, LLMError> {
    let messages = trajectory.lock().await.build_messages().await;
    let completion_request = CompletionBuilder::new()
        .model(Model::Claude35Sonnet)
        .provider(Provider::Anthropic)
        .messages(messages)
        .temperature(0.0)
        .build();
    completion_request.do_request().await
}

pub async fn run_autocomplete() -> Result<(), Box<dyn std::error::Error>> {
    let trajectory = Arc::new(Mutex::new(Trajectory::new(true)));
    let trajectory_clone = trajectory.clone();
    let screenshot_task_handle = tokio::spawn(async move {
        let mut interval = tokio::time::interval(tokio::time::Duration::from_secs(5));
        loop {
            interval.tick().await;
            match take_screenshot().await {
                Ok(screenshot) => {
                    trajectory_clone
                        .lock()
                        .await
                        .add_screenshot(screenshot)
                        .await;
                }
                Err(e) => eprintln!("Screenshot error: {:?}", e),
            }
        }
    });
    trajectory
        .lock()
        .await
        .add_message(Message {
            role: Role::System,
            content: MessageContent::Text(AUTOCOMPLETE_SYSTEM_PROMPT.to_string()),
        })
        .await;
    let device_state = DeviceState::new();

    println!("Autocomplete is running. Press Cmd to trigger an autocompletion.");
    println!("Press Ctrl to exit.");

    let mut interval = tokio::time::interval(tokio::time::Duration::from_millis(10));
    let mut all_keys = device_state.get_keys();
    loop {
        let new_all_keys = device_state.get_keys();
        let start_idx = all_keys.len();
        if start_idx < new_all_keys.len() {
            let keys_diff = new_all_keys[start_idx..].to_vec();
            if keys_diff.contains(&Keycode::Command) {
                let trajectory = trajectory.clone();
                tokio::spawn(async move {
                    match handle_autocomplete(trajectory).await {
                        Ok(text) => println!("Autocompletion generated: {}", text),
                        Err(e) => println!("Error generating autocompletion: {:?}", e),
                    }
                });
            } else if keys_diff.contains(&Keycode::LControl)
                || keys_diff.contains(&Keycode::RControl)
            {
                println!("Exiting...");
                break;
            }
        }
        all_keys = new_all_keys;
        interval.tick().await;
    }
    screenshot_task_handle.abort();
    Ok(())
}

fn parse_autocomplete_response(
    response: &str,
) -> Result<AutocompleteResponse, ParseAutocompleteResponseError> {
    let json_string = match parse_markdown_code_block(response) {
        Ok(json_string) => json_string,
        Err(e) => return Err(ParseAutocompleteResponseError::MarkdownCodeBlockMissingError(e)),
    };
    match serde_json::from_str(&json_string) {
        Ok(autocomplete_response) => Ok(autocomplete_response),
        Err(e) => Err(ParseAutocompleteResponseError::ParseJsonError(e)),
    }
}
