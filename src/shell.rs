use crate::llm::{CompletionBuilder, Model, Provider};
use crate::screenshot::take_screenshot;
use crate::trajectory::Trajectory;
use std::io::{self, Write};
use std::sync::Arc;
use tokio::sync::Mutex;

pub async fn run_shell() -> Result<(), Box<dyn std::error::Error>> {
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

    println!("Welcome to the chat shell! Type 'exit' to quit.\n");
    let greeting = "How can I help you?";
    trajectory
        .lock()
        .await
        .add_assistant_message(greeting.to_string())
        .await;
    send_message_to_stdout("assistant", greeting);

    loop {
        print!("user: ");
        match io::stdout().flush() {
            Ok(_) => (),
            Err(e) => println!("Error: {}", e),
        }

        let mut input = String::new();
        if let Err(e) = io::stdin().read_line(&mut input) {
            println!("Error: {}", e);
            continue;
        }
        let input = input.trim();

        if input == "exit" {
            println!("\nExiting...");
            break;
        }

        let recent_screenshot = match take_screenshot().await {
            Ok(screenshot) => screenshot,
            Err(e) => {
                println!("Error taking screenshot: {:?}", e);
                continue;
            }
        };
        trajectory
            .lock()
            .await
            .add_screenshot(recent_screenshot)
            .await;
        trajectory
            .lock()
            .await
            .add_user_message(input.to_string())
            .await;

        let messages = trajectory.lock().await.build_messages().await;
        let completion_request = CompletionBuilder::new()
            .model(Model::Claude35Sonnet)
            .provider(Provider::Anthropic)
            .messages(messages)
            .temperature(0.7)
            .build();

        let response = match completion_request.do_request().await {
            Ok(response) => response,
            Err(e) => {
                println!("Error: {}", e);
                continue;
            }
        };
        send_message_to_stdout("assistant", &response);
        trajectory
            .lock()
            .await
            .add_assistant_message(response)
            .await;
    }
    screenshot_task_handle.abort();
    Ok(())
}

fn send_message_to_stdout(author: &str, message: &str) {
    println!("{}: {}", author, message);
}
