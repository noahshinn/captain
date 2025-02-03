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

pub const DISCARD_REDUNDANT_SCREENSHOT_SYSTEM_PROMPT: &str = r#"# Task
You will be given two screenshots.
The previous screenshot was captured 5 seconds before the current screenshot.
Determine if the previous screenshot should be discarded (meaning that the current screenshot contains a superset of the information in the previous screenshot).
The context is that these screenshots are part of a continuous monitoring of a user's computer screen as part of an AI tool.
Examples of redundant screenshot scenarios include:
- The user has not made any changes to the screen.
- The user is typing in a text box, so the information in the current screenshot is a superset of the information in the previous screenshot.
- The user is scrolling through a webpage but the missing content from the previous screenshot is not important or was only whitespace, so discarding the previous screenshot would not be detrimental to the history.

## Format
First, write a reasoning trace that analyzes both screenshots and determines if the previous screenshot contains any important information that is not present in the current screenshot.
Then, write a JSON object in a markdown code block with the following format:

```json
{{
    "previous_screenshot_contains_important_information_not_present_in_current_screenshot": boolean
}}

For example:

<your reasoning trace>

```json
{{
    "previous_screenshot_contains_important_information_not_present_in_current_screenshot": <true or false>
}}
```
"#;
