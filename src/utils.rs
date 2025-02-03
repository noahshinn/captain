use regex::Regex;
use thiserror::Error;

#[derive(Error, Debug)]
pub struct MarkdownCodeBlockMissingError {
    pub text: String,
}

impl std::fmt::Display for MarkdownCodeBlockMissingError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "Markdown code block missing in response: {}", self.text)
    }
}

pub fn parse_markdown_code_block(response: &str) -> Result<String, MarkdownCodeBlockMissingError> {
    let re = Regex::new(r"```(\w*)\n([\s\S]*?)\n```").unwrap();
    match re.captures(response) {
        Some(captures) => match captures.get(2) {
            Some(content) => Ok(content.as_str().to_string()),
            None => Err(MarkdownCodeBlockMissingError {
                text: response.to_string(),
            }),
        },
        None => Err(MarkdownCodeBlockMissingError {
            text: response.to_string(),
        }),
    }
}
