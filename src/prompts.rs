use crate::llm::{Message, MessageContent, Role};

#[derive(Debug, Clone)]
pub struct Prompt {
    pub instruction: String,
    pub context: String,
}

impl Prompt {
    pub fn new(instruction: String, context: String) -> Self {
        Self {
            instruction,
            context,
        }
    }

    pub fn build_messages(self) -> Vec<Message> {
        vec![
            Message {
                role: Role::System,
                content: MessageContent::Text(self.instruction),
            },
            Message {
                role: Role::User,
                content: MessageContent::Text(self.context),
            },
        ]
    }
}

pub const SCREENSHOT_DESCRIPTION_SYSTEM_PROMPT: &str = r#"# Task
You will be given a screenshot of the user's screen.
Your job is to write a text description of the screenshot that preserves all of the information in the screenshot.
The context is that this is a call within an AI-powered assistant that watches the user's screen.
A screenshot is taken every 5 seconds.
However, if the tool captures too many screenshots to feed to a multi-modal model, it will use the text description of some of the screenshots.
You are writing this text description for the tool."#;

pub const AUTOCOMPLETE_SYSTEM_PROMPT: &str = r#"# Task
You are an AI assistant that helps users to autocomplete by sending text to the user's machine.
Based on the screenshots of their work history, predict what they're likely trying to do next and provide the exact text.
This is an autocomplete tool.

## Format
Put the text to autocomplete in a markdown code block in the following format:
```
{
    "autocomplete": "<text here>"
}
```"#;
